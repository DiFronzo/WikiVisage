import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymysql
import pytest

import database
from database import DatabaseError
from migrate import _ALL_TABLES, _ALTER_MIGRATIONS, _apply_alter_migrations, load_schema, reset_database


def test_load_schema_basic(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "CREATE TABLE users (id INT);\nCREATE TABLE projects (id INT);\n",
        encoding="utf-8",
    )

    statements = load_schema(str(schema_file))

    assert statements == [
        "CREATE TABLE users (id INT)",
        "CREATE TABLE projects (id INT)",
    ]


def test_load_schema_skips_empty(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "  CREATE TABLE users (id INT);\n\nCREATE TABLE faces (id INT);\n; ;\n\n",
        encoding="utf-8",
    )

    statements = load_schema(str(schema_file))

    assert len(statements) == 2
    assert all(stmt.strip() for stmt in statements)


def test_load_schema_skips_comments_only(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "-- comment only\n-- another comment;\nCREATE TABLE images (id INT);",
        encoding="utf-8",
    )

    statements = load_schema(str(schema_file))

    assert statements == ["CREATE TABLE images (id INT)"]


def test_load_schema_preserves_inline_comments(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "CREATE TABLE users (id INT) -- inline comment\n;",
        encoding="utf-8",
    )

    statements = load_schema(str(schema_file))

    assert len(statements) == 1
    assert "-- inline comment" in statements[0]
    assert statements[0].startswith("CREATE TABLE users")


def test_load_schema_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.sql"

    with pytest.raises(FileNotFoundError):
        load_schema(str(missing))


def test_load_schema_empty_file(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text("", encoding="utf-8")

    statements = load_schema(str(schema_file))

    assert statements == []


def test_load_schema_complex(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        """
        CREATE TABLE users (
            id BIGINT
        );

        -- comments between statements
        -- should not become statements

        CREATE TABLE faces (
            id BIGINT,
            user_id BIGINT
        );

        ALTER TABLE faces ADD INDEX idx_faces_user_id (user_id);
        """,
        encoding="utf-8",
    )

    statements = load_schema(str(schema_file))

    assert len(statements) == 3
    assert statements[0].startswith("CREATE TABLE users")
    assert "CREATE TABLE faces" in statements[1]
    assert "-- comments between statements" in statements[1]
    assert statements[2].startswith("ALTER TABLE faces")


def _mock_get_connection_with_error(error: Exception):
    cursor = MagicMock()
    cursor.execute.side_effect = error

    cursor_cm = MagicMock()
    cursor_cm.__enter__.return_value = cursor
    cursor_cm.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = conn
    conn_cm.__exit__.return_value = False

    return conn_cm, conn


def test_alter_migration_skips_duplicate_column(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn_cm, conn = _mock_get_connection_with_error(pymysql.err.OperationalError(1060, "Duplicate column"))

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        caplog.at_level(logging.INFO),
    ):
        _apply_alter_migrations()

    skipped_logs = [r for r in caplog.records if "Skipped migration" in r.message]
    assert len(skipped_logs) == len(_ALTER_MIGRATIONS)
    assert conn.rollback.call_count == len(_ALTER_MIGRATIONS)


def test_alter_migration_skips_duplicate_key(caplog: pytest.LogCaptureFixture) -> None:
    conn_cm, conn = _mock_get_connection_with_error(pymysql.err.OperationalError(1061, "Duplicate key"))

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        caplog.at_level(logging.INFO),
    ):
        _apply_alter_migrations()

    skipped_logs = [r for r in caplog.records if "Skipped migration" in r.message]
    assert len(skipped_logs) == len(_ALTER_MIGRATIONS)
    assert conn.rollback.call_count == len(_ALTER_MIGRATIONS)


def test_alter_migration_skips_duplicate_fk_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn_cm, conn = _mock_get_connection_with_error(pymysql.err.OperationalError(1826, "Duplicate FK constraint"))

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        caplog.at_level(logging.INFO),
    ):
        _apply_alter_migrations()

    skipped_logs = [r for r in caplog.records if "Skipped migration" in r.message]
    assert len(skipped_logs) == len(_ALTER_MIGRATIONS)
    assert conn.rollback.call_count == len(_ALTER_MIGRATIONS)


def test_alter_migration_skips_mariadb_fk_1005(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn_cm, conn = _mock_get_connection_with_error(
        pymysql.err.OperationalError(1005, "Can't create table ... (errno: 121)")
    )

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        caplog.at_level(logging.INFO),
    ):
        _apply_alter_migrations()

    skipped_logs = [r for r in caplog.records if "Skipped migration" in r.message]
    assert len(skipped_logs) == len(_ALTER_MIGRATIONS)
    assert conn.rollback.call_count == len(_ALTER_MIGRATIONS)


def test_alter_migration_raises_on_unknown_error() -> None:
    conn_cm, conn = _mock_get_connection_with_error(pymysql.err.OperationalError(9999, "Some unknown error"))

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        pytest.raises(DatabaseError),
    ):
        _apply_alter_migrations()

    assert conn.rollback.call_count == 1


def test_alter_migration_raises_on_1005_without_errno_121() -> None:
    conn_cm, conn = _mock_get_connection_with_error(
        pymysql.err.OperationalError(1005, "Can't create table ... some other error")
    )

    with (
        patch.object(database, "get_connection", return_value=conn_cm),
        patch("migrate.get_connection", return_value=conn_cm),
        pytest.raises(DatabaseError),
    ):
        _apply_alter_migrations()

    assert conn.rollback.call_count == 1


def _make_mock_connection():
    """Return (conn_cm, conn, cursor) triple with a working cursor mock."""
    cursor = MagicMock()
    cursor_cm = MagicMock()
    cursor_cm.__enter__.return_value = cursor
    cursor_cm.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = conn
    conn_cm.__exit__.return_value = False

    return conn_cm, conn, cursor


def test_reset_database_drops_all_tables(caplog: pytest.LogCaptureFixture) -> None:
    conn_cm, conn, cursor = _make_mock_connection()

    with (
        patch("migrate.init_db") as mock_init_db,
        patch("migrate.close_pool") as mock_close_pool,
        patch("migrate.get_connection", return_value=conn_cm),
        patch("migrate.run_migration") as mock_run_migration,
        caplog.at_level(logging.WARNING),
    ):
        reset_database()

    mock_init_db.assert_called_once_with(pool_size=1)
    mock_close_pool.assert_called_once()
    mock_run_migration.assert_called_once()
    conn.commit.assert_called_once()

    executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
    assert executed_sql[0] == "SET FOREIGN_KEY_CHECKS=0"
    for table in _ALL_TABLES:
        assert f"DROP TABLE IF EXISTS {table}" in executed_sql
    assert "SET FOREIGN_KEY_CHECKS=1" in executed_sql


def test_reset_database_restores_fk_checks_on_error() -> None:
    conn_cm, conn, cursor = _make_mock_connection()

    def _fail_on_drop(sql, *args):
        if sql.startswith("DROP TABLE"):
            raise Exception("Table drop failed")

    cursor.execute.side_effect = _fail_on_drop

    with (
        patch("migrate.init_db"),
        patch("migrate.close_pool"),
        patch("migrate.get_connection", return_value=conn_cm),
        patch("migrate.run_migration"),
        pytest.raises(Exception, match="Table drop failed"),
    ):
        reset_database()

    executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
    assert "SET FOREIGN_KEY_CHECKS=1" in executed_sql


@pytest.mark.integration
def test_all_tables_exist(test_db: str, db_conn) -> None:
    assert test_db == "wikiface_test"

    with db_conn.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        rows = cursor.fetchall()

    existing_tables = {list(row.values())[0] for row in rows}
    expected_tables = {
        "users",
        "sessions",
        "projects",
        "images",
        "faces",
        "worker_heartbeat",
    }

    assert expected_tables.issubset(existing_tables)


@pytest.mark.integration
def test_users_table_columns(test_db: str, db_conn) -> None:
    assert test_db == "wikiface_test"

    with db_conn.cursor() as cursor:
        cursor.execute("DESCRIBE users")
        rows = cursor.fetchall()

    fields = {row["Field"] for row in rows}
    expected_fields = {
        "id",
        "wiki_user_id",
        "wiki_username",
        "access_token",
        "refresh_token",
        "token_expires_at",
        "created_at",
        "updated_at",
    }

    assert expected_fields.issubset(fields)


@pytest.mark.integration
def test_faces_table_columns(test_db: str, db_conn) -> None:
    assert test_db == "wikiface_test"

    with db_conn.cursor() as cursor:
        cursor.execute("DESCRIBE faces")
        rows = cursor.fetchall()

    fields = {row["Field"] for row in rows}
    expected_fields = {
        "id",
        "image_id",
        "encoding",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "bbox_left",
        "is_target",
        "confidence",
        "classified_by",
        "classified_by_user_id",
        "sdc_written",
        "sdc_removal_pending",
        "superseded_by",
        "created_at",
        "updated_at",
    }

    assert expected_fields.issubset(fields)


@pytest.mark.integration
def test_images_table_has_bootstrapped_column(test_db: str, db_conn) -> None:
    assert test_db == "wikiface_test"

    with db_conn.cursor() as cursor:
        cursor.execute("DESCRIBE images")
        rows = cursor.fetchall()

    columns = {row["Field"]: row for row in rows}

    assert "bootstrapped" in columns
    assert columns["bootstrapped"]["Default"] in ("0", 0)


@pytest.mark.integration
def test_projects_table_has_sdc_columns(test_db: str, db_conn) -> None:
    assert test_db == "wikiface_test"

    with db_conn.cursor() as cursor:
        cursor.execute("DESCRIBE projects")
        rows = cursor.fetchall()

    fields = {row["Field"] for row in rows}

    assert "sdc_write_requested" in fields
    assert "sdc_write_error" in fields


@pytest.mark.integration
def test_foreign_keys_exist(test_db: str, db_conn) -> None:
    with db_conn.cursor() as cursor:
        cursor.execute(
            "SELECT CONSTRAINT_NAME "
            "FROM information_schema.TABLE_CONSTRAINTS "
            "WHERE TABLE_SCHEMA = %s AND CONSTRAINT_TYPE = %s",
            (test_db, "FOREIGN KEY"),
        )
        rows = cursor.fetchall()

    constraints = {row["CONSTRAINT_NAME"] for row in rows}
    expected_constraints = {
        "fk_sessions_user",
        "fk_projects_user",
        "fk_images_project",
        "fk_faces_image",
        "fk_faces_classified_by_user",
        "fk_faces_superseded_by",
    }

    assert expected_constraints.issubset(constraints)


@pytest.mark.integration
def test_alter_migrations_idempotent(test_db: str) -> None:
    import os

    from database import close_pool, get_connection, init_db
    from migrate import _apply_alter_migrations

    old_env = {
        "TOOL_TOOLSDB_USER": os.environ.get("TOOL_TOOLSDB_USER"),
        "TOOL_TOOLSDB_PASSWORD": os.environ.get("TOOL_TOOLSDB_PASSWORD"),
        "TOOL_TOOLSDB_HOST": os.environ.get("TOOL_TOOLSDB_HOST"),
        "WIKIVISAGE_DB_NAME": os.environ.get("WIKIVISAGE_DB_NAME"),
    }

    os.environ["TOOL_TOOLSDB_USER"] = "root"
    os.environ["TOOL_TOOLSDB_PASSWORD"] = "devpass"
    os.environ["TOOL_TOOLSDB_HOST"] = "127.0.0.1"
    os.environ["WIKIVISAGE_DB_NAME"] = test_db

    try:
        init_db(pool_size=2)
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE() AS db_name")
            row = cursor.fetchone()
            assert row["db_name"] == test_db

        _apply_alter_migrations()
        _apply_alter_migrations()
    finally:
        close_pool()
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


@pytest.mark.integration
def test_load_schema_real_file(test_db: str) -> None:
    import os

    from migrate import load_schema

    assert test_db == "wikiface_test"

    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "schema.sql",
    )
    statements = load_schema(schema_path)

    assert isinstance(statements, list)
    assert statements
    assert "CREATE TABLE IF NOT EXISTS users" in statements[0]
    assert len(statements) >= 6
