import time
from queue import Queue

import pytest

import database


def test_error_class_hierarchy_and_messages():
    assert issubclass(database.DatabaseError, Exception)
    assert issubclass(database.PoolExhaustedError, database.DatabaseError)
    assert issubclass(database.ConfigurationError, database.DatabaseError)

    db_error = database.DatabaseError("db message")
    pool_error = database.PoolExhaustedError("pool message")
    config_error = database.ConfigurationError("config message")

    assert str(db_error) == "db message"
    assert str(pool_error) == "pool message"
    assert str(config_error) == "config message"


def test_get_db_config_with_required_env_vars(monkeypatch_env):
    config = database._get_db_config()

    expected_keys = {
        "host",
        "user",
        "password",
        "database",
        "charset",
        "connect_timeout",
        "read_timeout",
        "autocommit",
        "cursorclass",
    }

    assert expected_keys.issubset(config.keys())
    assert config["host"] == "localhost"
    assert config["user"] == "testuser"
    assert config["password"] == "testpass"
    assert config["database"] == "testdb"
    assert config["charset"] == "utf8mb4"
    assert config["autocommit"] is False


def test_get_db_config_missing_user_raises(monkeypatch_env, monkeypatch):
    monkeypatch.delenv("TOOL_TOOLSDB_USER", raising=False)

    with pytest.raises(database.ConfigurationError) as exc_info:
        database._get_db_config()

    assert "TOOL_TOOLSDB_USER" in str(exc_info.value)


def test_get_db_config_missing_password_raises(monkeypatch_env, monkeypatch):
    monkeypatch.delenv("TOOL_TOOLSDB_PASSWORD", raising=False)

    with pytest.raises(database.ConfigurationError) as exc_info:
        database._get_db_config()

    assert "TOOL_TOOLSDB_PASSWORD" in str(exc_info.value)


def test_get_db_config_missing_database_name_raises(monkeypatch_env, monkeypatch):
    monkeypatch.delenv("WIKIVISAGE_DB_NAME", raising=False)

    with pytest.raises(database.ConfigurationError) as exc_info:
        database._get_db_config()

    assert "WIKIVISAGE_DB_NAME" in str(exc_info.value)


def test_get_db_config_default_host_when_not_set(monkeypatch_env, monkeypatch):
    monkeypatch.delenv("TOOL_TOOLSDB_HOST", raising=False)

    config = database._get_db_config()

    assert config["host"] == "tools.db.svc.wikimedia.cloud"
    assert config["autocommit"] is False
    assert config["charset"] == "utf8mb4"


def test_get_db_config_custom_host_when_set(monkeypatch_env, monkeypatch):
    monkeypatch.setenv("TOOL_TOOLSDB_HOST", "custom-host")

    config = database._get_db_config()

    assert config["host"] == "custom-host"


def test_get_connection_from_pool_not_initialized(monkeypatch):
    monkeypatch.setattr(database, "_pool", None)

    with pytest.raises(database.DatabaseError) as exc_info:
        database._get_connection_from_pool()

    assert "not initialized" in str(exc_info.value)


def test_return_connection_to_pool_with_none_pool_closes_connection(monkeypatch):
    class DummyConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    conn = DummyConnection()
    pc = database._PooledConnection(conn)
    monkeypatch.setattr(database, "_pool", None)

    database._return_connection_to_pool(pc)

    assert conn.closed is True


class _FakeConn:
    """Minimal fake pymysql connection for unit tests."""

    def __init__(self, *, is_open=True, rollback_raises=False):
        self.open = is_open
        self._closed = False
        self._rollback_raises = rollback_raises

    def close(self):
        self._closed = True

    def rollback(self):
        if self._rollback_raises:
            raise OSError("rollback failed")


def _make_pc(conn, *, expired=False):
    pc = database._PooledConnection(conn)
    if expired:
        pc.created_at = time.monotonic() - database.MAX_CONNECTION_AGE - 1
    return pc


def test_return_connection_replenishes_after_expired(monkeypatch):
    """Discarding an expired connection triggers best-effort pool replenishment."""
    pool = Queue(maxsize=2)
    monkeypatch.setattr(database, "_pool", pool)

    fresh_conn = _FakeConn()
    fresh_pc = database._PooledConnection(fresh_conn)
    monkeypatch.setattr(database, "_create_connection", lambda: fresh_pc)

    expired_conn = _FakeConn()
    pc = _make_pc(expired_conn, expired=True)

    database._return_connection_to_pool(pc)

    # Original expired connection must be closed.
    assert expired_conn._closed is True
    # Pool should now contain exactly the fresh replacement.
    assert pool.qsize() == 1
    assert pool.get_nowait() is fresh_pc


def test_return_connection_replenishes_after_rollback_failure(monkeypatch):
    """When rollback raises, a replacement connection is put back into the pool."""
    pool = Queue(maxsize=2)
    monkeypatch.setattr(database, "_pool", pool)

    fresh_conn = _FakeConn()
    fresh_pc = database._PooledConnection(fresh_conn)
    monkeypatch.setattr(database, "_create_connection", lambda: fresh_pc)

    bad_conn = _FakeConn(rollback_raises=True)
    pc = _make_pc(bad_conn)

    database._return_connection_to_pool(pc)

    assert bad_conn._closed is True
    assert pool.qsize() == 1
    assert pool.get_nowait() is fresh_pc


def test_return_connection_replenish_failure_is_silent(monkeypatch):
    """If replacement creation fails, the error is swallowed and no exception propagates."""
    pool = Queue(maxsize=2)
    monkeypatch.setattr(database, "_pool", pool)
    monkeypatch.setattr(database, "_create_connection", lambda: (_ for _ in ()).throw(OSError("no DB")))

    expired_conn = _FakeConn()
    pc = _make_pc(expired_conn, expired=True)

    # Must not raise even though replenishment fails.
    database._return_connection_to_pool(pc)

    assert expired_conn._closed is True
    assert pool.qsize() == 0  # pool stays empty — no crash


def test_return_connection_no_replenish_when_pool_full(monkeypatch):
    """When the pool is already full, no replenishment attempt is made."""
    pool = Queue(maxsize=1)
    # Pre-fill so the pool is at capacity
    existing_conn = _FakeConn()
    pool.put_nowait(database._PooledConnection(existing_conn))
    monkeypatch.setattr(database, "_pool", pool)

    create_calls = []

    def _fake_create():
        create_calls.append(1)
        return database._PooledConnection(_FakeConn())

    monkeypatch.setattr(database, "_create_connection", _fake_create)

    # A healthy, non-expired connection that fits in the pool … but pool is full.
    extra_conn = _FakeConn()
    pc = _make_pc(extra_conn)

    database._return_connection_to_pool(pc)

    # The extra connection is closed; no replenishment attempt should happen
    # because the pool-full path only evicts the surplus connection.
    assert extra_conn._closed is True
    assert len(create_calls) == 0


def test_try_replenish_pool_no_op_when_pool_none(monkeypatch):
    """`_try_replenish_pool` exits early when the pool is not initialized."""
    monkeypatch.setattr(database, "_pool", None)
    create_calls = []
    monkeypatch.setattr(database, "_create_connection", lambda: create_calls.append(1) or None)

    database._try_replenish_pool()

    assert len(create_calls) == 0


@pytest.mark.integration
def test_init_db_creates_pool(db_pool, db_conn):
    assert db_pool._pool is not None
    assert db_pool._pool.qsize() > 0


@pytest.mark.integration
def test_execute_query_select(db_pool, db_conn):
    result = db_pool.execute_query("SELECT 1 AS val")
    assert result == [{"val": 1}]


@pytest.mark.integration
def test_execute_query_insert_and_select(db_pool, db_conn):
    affected = db_pool.execute_query(
        "INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (99999, "IntegrationUser1", "token-a", "refresh-a", "2030-01-01 00:00:00"),
        fetch=False,
    )
    assert affected == 1

    rows = db_pool.execute_query("SELECT * FROM users WHERE wiki_user_id = %s", (99999,))
    assert len(rows) == 1
    assert rows[0]["wiki_user_id"] == 99999
    assert rows[0]["wiki_username"] == "IntegrationUser1"


@pytest.mark.integration
def test_execute_insert_returns_lastrowid(db_pool, db_conn):
    user_id = db_pool.execute_insert(
        "INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (100001, "IntegrationUser2", "token-b", "refresh-b", "2030-01-01 00:00:00"),
    )
    assert user_id > 0

    rows = db_pool.execute_query("SELECT * FROM users WHERE id = %s", (user_id,))
    assert len(rows) == 1
    assert rows[0]["wiki_username"] == "IntegrationUser2"


@pytest.mark.integration
def test_execute_transaction_atomicity(db_pool, db_conn):
    def ops(conn, cursor):
        cursor.execute(
            "INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (100002, "IntegrationUser3", "token-c", "refresh-c", "2030-01-01 00:00:00"),
        )
        created_user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, distance_threshold, min_confirmed, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (created_user_id, "Q42", "Integration Category 1", "Douglas Adams", 0.6, 5, "active"),
        )
        return created_user_id

    created_user_id = db_pool.execute_transaction(ops)

    users = db_pool.execute_query("SELECT * FROM users WHERE id = %s", (created_user_id,))
    projects = db_pool.execute_query("SELECT * FROM projects WHERE user_id = %s", (created_user_id,))
    assert len(users) == 1
    assert len(projects) == 1
    assert projects[0]["wikidata_qid"] == "Q42"


@pytest.mark.integration
def test_execute_transaction_rollback_on_error(db_pool, db_conn):
    user_id = db_pool.execute_insert(
        "INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (100003, "IntegrationUser4", "token-d", "refresh-d", "2030-01-01 00:00:00"),
    )

    def ops(conn, cursor):
        cursor.execute(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, distance_threshold, min_confirmed, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, "Q1", "Integration Category 2", "Rollback Label", 0.6, 5, "active"),
        )
        raise RuntimeError("force rollback")

    with pytest.raises(db_pool.DatabaseError):
        db_pool.execute_transaction(ops)

    projects = db_pool.execute_query("SELECT * FROM projects WHERE user_id = %s", (user_id,))
    assert len(projects) == 0


@pytest.mark.integration
def test_get_connection_context_manager(db_pool, db_conn):
    with db_pool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 AS val")
            row = cursor.fetchone()
    assert row["val"] == 1


@pytest.mark.integration
def test_close_pool_and_reinitialize(db_pool, db_conn):
    db_pool.close_pool()
    assert db_pool._pool is None

    db_pool.init_db(pool_size=2)
    result = db_pool.execute_query("SELECT 1 AS val")
    assert result == [{"val": 1}]


@pytest.mark.integration
def test_execute_query_with_parameterized_insert(db_pool, db_conn):
    affected = db_pool.execute_query(
        "INSERT INTO users (wiki_user_id, wiki_username, access_token, refresh_token, token_expires_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (100004, "name-with-'quote'", "token-e", "refresh-e", "2030-01-01 00:00:00"),
        fetch=False,
    )
    assert affected == 1

    rows = db_pool.execute_query("SELECT * FROM users WHERE wiki_user_id = %s", (100004,))
    assert len(rows) == 1
    assert rows[0]["wiki_username"] == "name-with-'quote'"


# ---------------------------------------------------------------------------
# Unit tests for _execute_with_retry
# ---------------------------------------------------------------------------

from pymysql.err import OperationalError


def _fake_op_error():
    """Build a fake pymysql OperationalError."""
    return OperationalError(2006, "Server gone away")


class _FakePCWithCursor:
    """Fake _PooledConnection that also supports cursor context manager."""

    def __init__(self, *, is_open=True, rollback_raises=False):
        self.conn = _FakeConn(is_open=is_open, rollback_raises=rollback_raises)

    @property
    def is_expired(self):
        return False


def test_execute_with_retry_success_on_first_attempt(monkeypatch):
    """`_execute_with_retry` returns the result immediately on success."""

    def _success():
        return 42

    result = database._execute_with_retry(_success)
    assert result == 42


def test_execute_with_retry_retries_on_operational_error(monkeypatch):
    """`_execute_with_retry` retries when OperationalError is raised."""
    call_count = [0]

    def _flaky():
        call_count[0] += 1
        if call_count[0] < 3:
            raise OperationalError(2006, "gone away")
        return "ok"

    monkeypatch.setattr(database, "INITIAL_BACKOFF", 0.001)
    monkeypatch.setattr(database.time, "sleep", lambda *a: None)

    result = database._execute_with_retry(_flaky)
    assert result == "ok"
    assert call_count[0] == 3


def test_execute_with_retry_raises_database_error_after_exhaustion(monkeypatch):
    """`_execute_with_retry` raises DatabaseError after all retries fail."""
    monkeypatch.setattr(database, "INITIAL_BACKOFF", 0.001)
    monkeypatch.setattr(database.time, "sleep", lambda *a: None)

    def _always_fails():
        raise OperationalError(2006, "gone away")

    with pytest.raises(database.DatabaseError):
        database._execute_with_retry(_always_fails)


def test_execute_with_retry_allow_retry_false_no_retries(monkeypatch):
    """`_execute_with_retry` with allow_retry=False executes once and raises immediately."""
    call_count = [0]

    def _flaky():
        call_count[0] += 1
        raise OperationalError(2006, "gone away")

    monkeypatch.setattr(database, "INITIAL_BACKOFF", 0.001)
    monkeypatch.setattr(database.time, "sleep", lambda *a: None)

    with pytest.raises(database.DatabaseError):
        database._execute_with_retry(_flaky, allow_retry=False)

    assert call_count[0] == 1


def test_execute_with_retry_retries_on_pool_exhausted(monkeypatch):
    """`_execute_with_retry` retries when PoolExhaustedError is raised."""
    call_count = [0]

    def _flaky():
        call_count[0] += 1
        if call_count[0] < 2:
            raise database.PoolExhaustedError("exhausted")
        return "done"

    monkeypatch.setattr(database, "INITIAL_BACKOFF", 0.001)
    monkeypatch.setattr(database.time, "sleep", lambda *a: None)

    result = database._execute_with_retry(_flaky)
    assert result == "done"


# ---------------------------------------------------------------------------
# Unit tests for execute_query, execute_insert, execute_transaction
# ---------------------------------------------------------------------------


def test_execute_query_raises_database_error_on_failure(monkeypatch):
    """`execute_query` wraps exceptions as DatabaseError."""
    monkeypatch.setattr(database, "_pool", None)

    with pytest.raises(database.DatabaseError):
        database.execute_query("SELECT 1")


def test_execute_insert_raises_database_error_on_failure(monkeypatch):
    """`execute_insert` wraps exceptions as DatabaseError."""
    monkeypatch.setattr(database, "_pool", None)

    with pytest.raises(database.DatabaseError):
        database.execute_insert("INSERT INTO users VALUES (%s)", (1,))


def test_execute_transaction_raises_database_error_on_failure(monkeypatch):
    """`execute_transaction` wraps exceptions as DatabaseError."""
    monkeypatch.setattr(database, "_pool", None)

    def ops(conn, cursor):
        cursor.execute("SELECT 1")

    with pytest.raises(database.DatabaseError):
        database.execute_transaction(ops)


# ---------------------------------------------------------------------------
# Unit tests for get_connection
# ---------------------------------------------------------------------------


def test_get_connection_rollback_on_exception(monkeypatch):
    """`get_connection` calls rollback when an exception is raised inside the block."""
    rollback_called = [False]

    class FakeConn:
        open = True

        def rollback(self):
            rollback_called[0] = True

        def close(self):
            pass

        def cursor(self):

            class FakeCursor:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return FakeCursor()

    pool = Queue(maxsize=2)
    fake_conn = FakeConn()
    pc = database._PooledConnection(fake_conn)
    # Make connection appear healthy and not expired
    monkeypatch.setattr(database, "_is_connection_healthy", lambda _pc: True)
    pool.put_nowait(pc)
    monkeypatch.setattr(database, "_pool", pool)

    with pytest.raises(RuntimeError):
        with database.get_connection() as _:
            raise RuntimeError("oops")

    assert rollback_called[0]


# ---------------------------------------------------------------------------
# Unit tests for close_pool
# ---------------------------------------------------------------------------


def test_close_pool_no_op_when_pool_is_none(monkeypatch):
    """`close_pool` exits early without error when pool is already None."""
    monkeypatch.setattr(database, "_pool", None)
    database.close_pool()  # Must not raise
    assert database._pool is None


def test_close_pool_drains_all_connections(monkeypatch):
    """`close_pool` closes all connections and sets _pool to None."""
    pool = Queue(maxsize=3)
    closed_flags = []

    for _ in range(3):

        class TrackConn:
            def close(self):
                closed_flags.append(True)

        pool.put_nowait(database._PooledConnection(TrackConn()))

    monkeypatch.setattr(database, "_pool", pool)
    database.close_pool()

    assert database._pool is None
    assert len(closed_flags) == 3


# ---------------------------------------------------------------------------
# Unit tests for _get_connection_from_pool
# ---------------------------------------------------------------------------


def test_get_connection_from_pool_exhausted_raises(monkeypatch):
    """`_get_connection_from_pool` raises PoolExhaustedError when pool is empty."""
    pool = Queue(maxsize=2)
    monkeypatch.setattr(database, "_pool", pool)

    with pytest.raises(database.PoolExhaustedError):
        database._get_connection_from_pool(timeout=0.01)


def test_get_connection_from_pool_evicts_dead_connection(monkeypatch):
    """`_get_connection_from_pool` replaces a dead connection with a fresh one."""
    dead_conn = _FakeConn()

    class DeadPC(database._PooledConnection):
        pass

    dead_pc = database._PooledConnection(dead_conn)

    # Make is_healthy return False to simulate dead connection
    monkeypatch.setattr(database, "_is_connection_healthy", lambda pc: False)

    fresh_conn = _FakeConn()
    fresh_pc = database._PooledConnection(fresh_conn)
    monkeypatch.setattr(database, "_create_connection", lambda: fresh_pc)

    pool = Queue(maxsize=2)
    pool.put_nowait(dead_pc)
    monkeypatch.setattr(database, "_pool", pool)

    result = database._get_connection_from_pool(timeout=1.0)
    assert result is fresh_pc
    assert dead_conn._closed
