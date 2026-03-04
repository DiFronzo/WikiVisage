"""
MariaDB connection module for Wikimedia Toolforge.

Provides a lightweight connection pool with retry logic, health checks,
and proper error handling for the Toolforge environment.
"""

import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from queue import Queue, Empty, Full
from typing import Any, Dict, List, Optional, Tuple, Union

import pymysql
from pymysql import OperationalError, InterfaceError
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)

# Connection pool configuration
_pool: Optional[Queue] = None
_pool_size: int = 5
_db_config: Dict[str, Any] = {}

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


def _get_db_config() -> Dict[str, Any]:
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
        raise ConfigurationError(
            "Missing required environment variable: TOOL_TOOLSDB_USER"
        )
    if not password:
        raise ConfigurationError(
            "Missing required environment variable: TOOL_TOOLSDB_PASSWORD"
        )
    if not database:
        raise ConfigurationError(
            "Missing required environment variable: WIKIVISAGE_DB_NAME"
        )

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


def _create_connection() -> pymysql.Connection:
    """
    Create a new database connection.

    Returns:
        A new pymysql connection object.

    Raises:
        DatabaseError: If connection cannot be established.
    """
    try:
        conn = pymysql.connect(**_db_config)
        logger.debug("Created new database connection")
        return conn
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        raise DatabaseError(f"Could not connect to database: {e}") from e


def _is_connection_alive(conn: pymysql.Connection) -> bool:
    """
    Check if a connection is still alive.

    Args:
        conn: The connection to check.

    Returns:
        True if connection is alive, False otherwise.
    """
    try:
        conn.ping(reconnect=False)
        return True
    except Exception as e:
        logger.debug(f"Connection health check failed: {e}")
        return False


def _get_connection_from_pool(timeout: float = 30.0) -> pymysql.Connection:
    """
    Get a connection from the pool, creating a new one if needed.

    Args:
        timeout: Maximum time to wait for a connection (seconds).

    Returns:
        A healthy database connection.

    Raises:
        PoolExhaustedError: If no connection available within timeout.
        DatabaseError: If connection cannot be created.
    """
    if _pool is None:
        raise DatabaseError("Connection pool not initialized. Call init_db() first.")

    try:
        conn = _pool.get(timeout=timeout)

        # Health check - reconnect if dead
        if not _is_connection_alive(conn):
            logger.warning("Retrieved dead connection from pool, creating new one")
            try:
                conn.close()
            except Exception:
                pass
            conn = _create_connection()

        return conn

    except Empty:
        raise PoolExhaustedError(
            f"Connection pool exhausted after {timeout}s timeout. "
            f"Consider increasing pool size or reducing query time."
        )


def _return_connection_to_pool(conn: pymysql.Connection) -> None:
    """
    Return a connection to the pool.

    Args:
        conn: The connection to return.
    """
    if _pool is None:
        logger.warning("Attempting to return connection but pool is not initialized")
        try:
            conn.close()
        except Exception:
            pass
        return

    try:
        # Rollback any uncommitted transactions
        if conn.open:
            try:
                conn.rollback()
            except Exception as e:
                logger.debug(f"Rollback failed when returning connection: {e}")

        _pool.put_nowait(conn)
    except Full:
        # Pool is full, close the connection
        logger.debug("Pool full, closing connection instead of returning")
        try:
            conn.close()
        except Exception:
            pass


def _execute_with_retry(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The result of func.

    Raises:
        DatabaseError: If all retries fail.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (OperationalError, InterfaceError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2**attempt)
                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)
            else:
                logger.error(
                    f"Database operation failed after {MAX_RETRIES} attempts: {e}"
                )

    if last_exception is not None:
        raise last_exception
    else:
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
    conn = None
    try:
        conn = _execute_with_retry(_get_connection_from_pool, timeout)
        yield conn
    except Exception as e:
        if conn and conn.open:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
        raise
    finally:
        if conn:
            _return_connection_to_pool(conn)


def execute_query(
    sql: str, params: Optional[Union[Tuple, Dict]] = None, fetch: bool = True
) -> Optional[Union[List[Dict[str, Any]], int]]:
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

    def _execute() -> Optional[Union[List[Dict[str, Any]], int]]:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)

                if fetch:
                    results = cursor.fetchall()
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


def init_db(pool_size: int = 5) -> None:
    """
    Initialize the database connection pool and verify connectivity.

    This must be called before using any database functions.

    Args:
        pool_size: Number of connections to maintain in the pool.

    Raises:
        ConfigurationError: If database configuration is invalid.
        DatabaseError: If initial connection test fails.

    Example:
        init_db(pool_size=10)
    """
    global _pool, _pool_size, _db_config

    logger.info("Initializing database connection pool")

    # Get and validate configuration
    _db_config = _get_db_config()
    _pool_size = pool_size

    # Create the pool
    _pool = Queue(maxsize=pool_size)

    # Pre-populate with connections
    for i in range(pool_size):
        try:
            conn = _create_connection()
            _pool.put_nowait(conn)
            logger.debug(f"Created connection {i + 1}/{pool_size}")
        except Exception as e:
            logger.error(
                f"Failed to create initial connection {i + 1}/{pool_size}: {e}"
            )
            # Clean up any connections created so far
            close_pool()
            raise DatabaseError(f"Failed to initialize connection pool: {e}") from e

    # Test connectivity
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result:
                    logger.info(
                        f"Database connection pool initialized successfully. "
                        f"Pool size: {pool_size}, Database: {_db_config['database']}"
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
            conn = _pool.get_nowait()
            conn.close()
            closed_count += 1
        except Empty:
            break
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    _pool = None
    logger.info(f"Closed {closed_count} database connections")
