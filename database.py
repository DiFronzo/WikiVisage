"""
MariaDB connection module for Wikimedia Toolforge.

Provides a lightweight connection pool with retry logic, health checks,
and proper error handling for the Toolforge environment.

Toolforge ToolsDB enforces max_user_connections=20 shared across ALL
processes for the same DB user account.  Default pool_size=2 keeps the
web process footprint small (2 gunicorn workers × 2 = 4 conns), leaving
room for background workers.
"""

import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from queue import Empty, Full, Queue
from typing import Any, Literal, overload

import pymysql
from pymysql import InterfaceError, OperationalError
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)

# Connection pool configuration
_pool: Queue | None = None
_pool_size: int = int(os.environ.get("WIKIVISAGE_DB_POOL_SIZE", 2))
_db_config: dict[str, Any] = {}

# Connections older than this are proactively replaced to avoid Toolforge
# killing idle connections unexpectedly (ToolsDB drops idle conns ~300s).
MAX_CONNECTION_AGE = 270  # seconds — below the 300s server-side timeout

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds


class DatabaseError(Exception):
    """Base exception for database-related errors."""

    pass


class PoolExhaustedError(DatabaseError):
    """Raised when connection pool is exhausted and timeout is reached."""

    pass


class ConfigurationError(DatabaseError):
    """Raised when database configuration is invalid or missing."""

    pass


def _get_db_config() -> dict[str, Any]:
    """
    Retrieve database configuration from environment variables.

    Returns:
        Dict containing database connection parameters.

    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    user = os.environ.get("TOOL_TOOLSDB_USER")
    password = os.environ.get("TOOL_TOOLSDB_PASSWORD")
    database = os.environ.get("WIKIVISAGE_DB_NAME")

    if not user:
        raise ConfigurationError("Missing required environment variable: TOOL_TOOLSDB_USER")
    if not password:
        raise ConfigurationError("Missing required environment variable: TOOL_TOOLSDB_PASSWORD")
    if not database:
        raise ConfigurationError("Missing required environment variable: WIKIVISAGE_DB_NAME")

    host = os.environ.get("TOOL_TOOLSDB_HOST", "tools.db.svc.wikimedia.cloud")

    return {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "charset": "utf8mb4",
        "connect_timeout": 10,
        "read_timeout": 30,
        "autocommit": False,
        "cursorclass": DictCursor,
    }


class _PooledConnection:
    """Wraps a pymysql connection with creation timestamp for age-based eviction."""

    __slots__ = ("conn", "created_at")

    def __init__(self, conn: pymysql.Connection):
        self.conn = conn
        self.created_at: float = time.monotonic()

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) >= MAX_CONNECTION_AGE


def _create_connection() -> _PooledConnection:
    try:
        conn = pymysql.connect(**_db_config)
        logger.debug("Created new database connection")
        return _PooledConnection(conn)
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        raise DatabaseError(f"Could not connect to database: {e}") from e


def _close_quietly(pc: _PooledConnection) -> None:
    try:
        pc.conn.close()
    except Exception:
        pass


def _is_connection_healthy(pc: _PooledConnection) -> bool:
    if pc.is_expired:
        return False
    try:
        pc.conn.ping(reconnect=False)
        return True
    except Exception:
        return False


def _get_connection_from_pool(timeout: float = 30.0) -> _PooledConnection:
    if _pool is None:
        raise DatabaseError("Connection pool not initialized. Call init_db() first.")

    try:
        pc = _pool.get(timeout=timeout)
    except Empty:
        raise PoolExhaustedError(
            f"Connection pool exhausted after {timeout}s timeout. Consider increasing pool size or reducing query time."
        )

    if _is_connection_healthy(pc):
        return pc

    # Connection is dead or expired — close it and create a fresh one.
    # If creation fails, the pool shrinks by one slot.  This is intentional:
    # re-pooling a dead connection just poisons the next caller.
    _close_quietly(pc)
    logger.warning("Evicted dead/expired connection from pool, creating replacement")
    return _create_connection()


def _try_replenish_pool() -> None:
    """Best-effort: replace a discarded connection with a fresh one.

    Called after closing a bad or expired connection so the pool does not
    permanently shrink under transient failures.  Failures here are logged
    and silently swallowed — the pool will recover on the next successful
    return or on the next caller that creates a fresh connection at GET time.
    """
    if _pool is None:
        return
    try:
        fresh = _create_connection()
        try:
            _pool.put_nowait(fresh)
            logger.debug("Replenished pool with fresh replacement connection")
        except Full:
            _close_quietly(fresh)
    except Exception as e:
        logger.warning(f"Could not replenish pool after discarding connection: {e}")


def _return_connection_to_pool(pc: _PooledConnection) -> None:
    if _pool is None:
        _close_quietly(pc)
        return

    try:
        if pc.conn.open and not pc.is_expired:
            try:
                pc.conn.rollback()
            except Exception:
                _close_quietly(pc)
                _try_replenish_pool()
                return
            try:
                _pool.put_nowait(pc)
            except Full:
                _close_quietly(pc)
        else:
            _close_quietly(pc)
            _try_replenish_pool()
    except Exception:
        _close_quietly(pc)
        _try_replenish_pool()


def _execute_with_retry(func: Callable[..., Any], *args, allow_retry: bool = True, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute.
        *args: Positional arguments to pass to func.
        allow_retry: If False, execute once without retrying (for write operations
            where retrying could cause duplicate inserts).
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func.

    Raises:
        DatabaseError: If all retries fail.
    """
    max_attempts = MAX_RETRIES if allow_retry else 1
    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except (OperationalError, InterfaceError, PoolExhaustedError) as e:
            last_exception = e
            if attempt < max_attempts - 1:
                backoff = INITIAL_BACKOFF * (2**attempt)
                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {backoff}s..."
                )
                time.sleep(backoff)
            else:
                logger.error(f"Database operation failed after {max_attempts} attempts: {e}")

    if last_exception is not None:
        raise DatabaseError(f"Database operation failed: {last_exception}") from last_exception
    raise DatabaseError("Retry logic failed without capturing an exception")


@contextmanager
def get_connection(timeout: float = 30.0):
    """
    Context manager to get a database connection from the pool.

    Automatically returns the connection to the pool after use.
    Handles rollback on exceptions.

    Args:
        timeout: Maximum time to wait for a connection (seconds).

    Yields:
        A database connection.

    Example:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()
    """
    pc: _PooledConnection | None = None
    try:
        result = _execute_with_retry(_get_connection_from_pool, timeout)
        pc = result  # type: _PooledConnection
        yield pc.conn
    except Exception:
        if pc and pc.conn.open:
            try:
                pc.conn.rollback()
            except Exception:
                _close_quietly(pc)
                pc = None
        raise
    finally:
        if pc:
            _return_connection_to_pool(pc)


@overload
def execute_query(
    sql: str, params: tuple | dict | None = None, fetch: Literal[True] = True
) -> list[dict[str, Any]]: ...


@overload
def execute_query(sql: str, params: tuple | dict | None = None, *, fetch: Literal[False]) -> int: ...


def execute_query(
    sql: str, params: tuple | dict | None = None, fetch: bool = True
) -> list[dict[str, Any]] | int | None:
    """
    Execute a SQL query with automatic connection and cursor management.

    Args:
        sql: The SQL query to execute.
        params: Parameters to bind to the query (tuple or dict).
        fetch: If True, fetch and return results. If False, return affected row count.

    Returns:
        If fetch=True: List of result rows as dictionaries.
        If fetch=False: Number of affected rows.

    Raises:
        DatabaseError: If query execution fails.

    Example:
        # SELECT query
        users = execute_query("SELECT * FROM users WHERE id = %s", (user_id,))

        # INSERT/UPDATE query
        affected = execute_query(
            "UPDATE users SET name = %s WHERE id = %s",
            ("Alice", 123),
            fetch=False
        )
    """

    def _execute() -> list[dict[str, Any]] | int | None:
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)

            if fetch:
                results = list(cursor.fetchall())
                logger.debug(f"Query returned {len(results)} rows")
                return results
            else:
                conn.commit()
                affected = cursor.rowcount
                logger.debug(f"Query affected {affected} rows")
                return affected

    try:
        return _execute_with_retry(_execute)
    except Exception as e:
        logger.error(f"Query execution failed: {sql[:100]}... Error: {e}")
        raise DatabaseError(f"Query execution failed: {e}") from e


def execute_insert(sql: str, params: tuple | dict | None = None) -> int:
    """
    Execute an INSERT query and return the auto-generated row ID.

    Uses cursor.lastrowid which is connection-local and race-free,
    unlike SELECT ... ORDER BY id DESC LIMIT 1.

    Args:
        sql: The INSERT SQL query to execute.
        params: Parameters to bind to the query.

    Returns:
        The auto-increment ID of the inserted row.

    Raises:
        DatabaseError: If query execution fails.
    """

    def _execute() -> int:
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    try:
        return _execute_with_retry(_execute, allow_retry=False)
    except Exception as e:
        logger.error(f"Insert execution failed: {sql[:100]}... Error: {e}")
        raise DatabaseError(f"Insert execution failed: {e}") from e


def execute_transaction(
    operations: Callable[[Any, Any], Any],
) -> Any:
    """
    Execute multiple queries in a single database transaction.

    The callable receives (connection, cursor) and should execute all
    queries on that cursor. The transaction is committed on success
    or rolled back on failure.

    Args:
        operations: A callable(conn, cursor) that performs all DB work.

    Returns:
        Whatever the callable returns.

    Raises:
        DatabaseError: If the transaction fails.

    Example:
        def do_work(conn, cursor):
            cursor.execute("INSERT INTO ...", (...,))
            new_id = cursor.lastrowid
            cursor.execute("UPDATE ...", (...,))
            return new_id

        result = execute_transaction(do_work)
    """

    def _execute() -> Any:
        with get_connection() as conn, conn.cursor() as cursor:
            result = operations(conn, cursor)
            conn.commit()
            return result

    try:
        return _execute_with_retry(_execute, allow_retry=False)
    except Exception as e:
        logger.error(f"Transaction execution failed: {e}")
        raise DatabaseError(f"Transaction execution failed: {e}") from e


def init_db(pool_size: int | None = None) -> None:
    """
    Initialize the database connection pool and verify connectivity.

    This must be called before using any database functions.

    Args:
        pool_size: Number of connections to maintain in the pool.
                   Defaults to WIKIVISAGE_DB_POOL_SIZE env var or 2.

    Raises:
        ConfigurationError: If database configuration is invalid.
        DatabaseError: If initial connection test fails.

    Example:
        init_db(pool_size=10)
    """
    global _pool, _pool_size, _db_config

    if pool_size is not None:
        _pool_size = pool_size

    logger.info(f"Initializing database connection pool (size={_pool_size})")

    # Get and validate configuration
    _db_config = _get_db_config()

    # Create the pool
    _pool = Queue(maxsize=_pool_size)

    # Pre-populate with connections
    for i in range(_pool_size):
        try:
            pc = _create_connection()
            _pool.put_nowait(pc)
            logger.debug(f"Created connection {i + 1}/{_pool_size}")
        except Exception as e:
            logger.error(f"Failed to create initial connection {i + 1}/{_pool_size}: {e}")
            # Clean up any connections created so far
            close_pool()
            raise DatabaseError(f"Failed to initialize connection pool: {e}") from e

    # Test connectivity
    try:
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result:
                logger.info(
                    f"Database connection pool initialized successfully. "
                    f"Pool size: {_pool_size}, Database: {_db_config['database']}"
                )
            else:
                raise DatabaseError("Connectivity test failed: No result returned")
    except Exception as e:
        logger.error(f"Database connectivity test failed: {e}")
        close_pool()
        raise DatabaseError(f"Database connectivity test failed: {e}") from e


def close_pool() -> None:
    """
    Close all connections in the pool and clean up resources.

    Should be called during application shutdown.

    Example:
        try:
            # Application code
            pass
        finally:
            close_pool()
    """
    global _pool

    if _pool is None:
        logger.debug("Connection pool already closed or not initialized")
        return

    logger.info("Closing database connection pool")

    closed_count = 0
    while not _pool.empty():
        try:
            pc = _pool.get_nowait()
            _close_quietly(pc)
            closed_count += 1
        except Empty:
            break

    _pool = None
    logger.info(f"Closed {closed_count} database connections")
