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
    monkeypatch.setattr(database, "_pool", None)

    database._return_connection_to_pool(conn)

    assert conn.closed is True


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
