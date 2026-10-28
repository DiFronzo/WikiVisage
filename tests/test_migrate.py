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
        "projects",
        "images",
        "faces",
        "worker_heartbeat",
        "worker_heartbeats",
    }

    assert expected_tables.issubset(existing_tables)
    assert "sessions" not in existing_tables


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
        "embedding_model",
        "det_score",
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
        "fk_projects_user",
        "fk_images_project",
        "fk_faces_image",
        "fk_faces_classified_by_user",
        "fk_faces_superseded_by",
    }

    assert expected_constraints.issubset(constraints)
    assert "fk_sessions_user" not in constraints


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


# ---------------------------------------------------------------------------
# Embedding-model upgrade: one-time wipe + schema changes
# ---------------------------------------------------------------------------


class _FakeDb:
    """Records SQL and simulates just enough of MariaDB for the upgrade gate.

    ``column_present`` is what ``information_schema.COLUMNS`` reports for
    ``faces.embedding_model``; executing the marker ``ADD COLUMN`` flips it, like
    the real thing. ``fail_on`` makes a statement containing the substring raise.
    """

    def __init__(self, column_present: bool, fail_on: str | None = None, fail_with: Exception | None = None):
        self.column_present = column_present
        self.fail_on = fail_on
        self.fail_with = fail_with or RuntimeError("boom")
        self.statements: list[tuple[str, tuple | None]] = []
        self.events: list[str] = []
        self.rowcount = 0
        self._last = ""

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.statements.append((sql, params))
        self._last = sql
        self.events.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise self.fail_with
        if "ADD COLUMN embedding_model" in sql:
            self.column_present = True
        self.rowcount = 3

    def fetchone(self):
        assert "information_schema.COLUMNS" in self._last
        return {"cnt": 1 if self.column_present else 0}

    def commit(self) -> None:
        self.events.append("COMMIT")

    def rollback(self) -> None:
        self.events.append("ROLLBACK")

    def cursor(self):
        cm = MagicMock()
        cm.__enter__.return_value = self
        cm.__exit__.return_value = False
        return cm

    def connection(self):
        cm = MagicMock()
        cm.__enter__.return_value = self
        cm.__exit__.return_value = False
        return cm

    @property
    def sql(self) -> list[str]:
        return [s for s, _ in self.statements]


def _run_upgrade(db: _FakeDb) -> bool:
    import migrate

    with patch("migrate.get_connection", return_value=db.connection()):
        return migrate.apply_embedding_model_upgrade()


def test_upgrade_wipes_and_adds_columns_when_marker_column_is_absent() -> None:
    db = _FakeDb(column_present=False)

    assert _run_upgrade(db) is True

    sql = db.sql
    assert "DELETE FROM images" in sql
    assert "DELETE FROM sdc_claims" in sql
    assert any(s.startswith("UPDATE projects SET images_total = 0") for s in sql)
    assert any("ADD COLUMN embedding_model VARCHAR(32) NOT NULL" in s for s in sql)
    assert any("ADD COLUMN det_score FLOAT NULL" in s for s in sql)
    assert db.column_present is True


def test_upgrade_does_nothing_when_marker_column_exists() -> None:
    """Every later migrate run, and every fresh database, must skip the wipe."""
    db = _FakeDb(column_present=True)

    assert _run_upgrade(db) is False

    assert len(db.statements) == 1
    assert "information_schema.COLUMNS" in db.sql[0]
    assert db.statements[0][1] == ("faces", "embedding_model")


def test_upgrade_wipes_only_once_across_two_runs() -> None:
    db = _FakeDb(column_present=False)

    assert _run_upgrade(db) is True
    first_run = len(db.statements)
    assert _run_upgrade(db) is False

    second_run = db.statements[first_run:]
    assert len(second_run) == 1  # just the information_schema probe
    assert "DELETE" not in second_run[0][0]


def test_upgrade_never_uses_truncate_and_only_touches_derived_tables() -> None:
    db = _FakeDb(column_present=False)
    _run_upgrade(db)

    joined = "\n".join(db.sql).upper()
    assert "TRUNCATE" not in joined
    assert "DROP TABLE" not in joined
    assert "FOREIGN_KEY_CHECKS" not in joined
    for kept in ("USERS", "SESSIONS", "PROJECT_MEMBERS", "USER_STATS", "WORKER_HEARTBEAT"):
        assert f"DELETE FROM {kept}" not in joined
        assert f"UPDATE {kept} " not in joined
    deletes = [s for s in db.sql if s.startswith("DELETE")]
    assert deletes == ["DELETE FROM images", "DELETE FROM sdc_claims"]


def test_upgrade_resets_every_project_counter_and_the_threshold() -> None:
    import migrate

    db = _FakeDb(column_present=False)
    _run_upgrade(db)

    sql, params = next((s, p) for s, p in db.statements if s.startswith("UPDATE projects SET images_total"))
    for column in (
        "images_total = 0",
        "images_processed = 0",
        "faces_confirmed = 0",
        "last_inference_threshold = NULL",
        "last_inference_min_confirmed = NULL",
        "completion_reason = NULL",
        "sdc_write_requested = 0",
        "sdc_write_user_id = NULL",
        "sdc_write_error = NULL",
        "worker_claimed_by = NULL",
        "worker_claimed_at = NULL",
        "distance_threshold = %s",
    ):
        assert column in sql
    assert " WHERE " not in sql  # every project, including paused and deleted
    assert params == (migrate.DEFAULT_DISTANCE_THRESHOLD,)


def test_upgrade_reactivates_completed_projects_only() -> None:
    db = _FakeDb(column_present=False)
    _run_upgrade(db)

    sql = next(s for s in db.sql if "SET status = 'active'" in s)
    assert sql == "UPDATE projects SET status = 'active' WHERE status = 'completed'"


def test_upgrade_adds_the_marker_column_last_and_only_after_the_wipe_committed() -> None:
    db = _FakeDb(column_present=False)
    _run_upgrade(db)

    events = db.events
    marker = next(i for i, e in enumerate(events) if "ADD COLUMN embedding_model" in e)
    delete = next(i for i, e in enumerate(events) if e == "DELETE FROM images")
    other_ddl = [i for i, e in enumerate(events) if e.startswith("ALTER TABLE") and "embedding_model" not in e]

    assert marker == max(i for i, e in enumerate(events) if e.startswith("ALTER TABLE"))
    assert all(i < marker for i in other_ddl)
    assert "COMMIT" in events[delete:marker]  # wipe committed before any DDL


def test_failed_wipe_rolls_back_and_leaves_the_marker_absent() -> None:
    db = _FakeDb(column_present=False, fail_on="DELETE FROM sdc_claims")

    with pytest.raises(RuntimeError, match="boom"):
        _run_upgrade(db)

    assert "ROLLBACK" in db.events
    assert db.column_present is False
    assert not any(s.startswith("ALTER TABLE") for s in db.sql)

    # Next run starts from scratch: it sees no marker and wipes again.
    db.fail_on = None
    assert _run_upgrade(db) is True
    assert db.column_present is True


def test_interrupted_schema_step_leaves_marker_absent_so_the_next_run_repeats_the_wipe() -> None:
    db = _FakeDb(column_present=False, fail_on="ADD COLUMN det_score", fail_with=RuntimeError("lost connection"))

    with pytest.raises(DatabaseError, match="det_score"):
        _run_upgrade(db)

    assert db.column_present is False

    db.fail_on = None
    assert _run_upgrade(db) is True
    assert sum(1 for s in db.sql if s == "DELETE FROM images") == 2


def test_already_applied_ddl_errors_are_tolerated_during_the_upgrade() -> None:
    """e.g. det_score already exists from a half-applied earlier attempt."""
    db = _FakeDb(
        column_present=False,
        fail_on="ADD COLUMN det_score",
        fail_with=pymysql.err.OperationalError(1060, "Duplicate column name 'det_score'"),
    )

    assert _run_upgrade(db) is True
    assert db.column_present is True


def test_upgrade_logs_a_warning_when_it_wipes(caplog: pytest.LogCaptureFixture) -> None:
    db = _FakeDb(column_present=False)

    with caplog.at_level(logging.WARNING):
        _run_upgrade(db)

    assert any("wiping" in r.getMessage() for r in caplog.records)


def test_run_migration_applies_the_upgrade_after_the_alter_migrations() -> None:
    import migrate

    order: list[str] = []
    conn_cm = MagicMock()
    conn_cm.__enter__.return_value.cursor.return_value.__enter__.return_value = MagicMock()
    conn_cm.__exit__.return_value = False

    with (
        patch("migrate.init_db"),
        patch("migrate.close_pool"),
        patch("migrate.get_connection", return_value=conn_cm),
        patch("migrate.load_schema", return_value=["CREATE TABLE IF NOT EXISTS t (id INT)"]),
        patch("migrate._apply_alter_migrations", side_effect=lambda: order.append("alter")),
        patch("migrate.apply_embedding_model_upgrade", side_effect=lambda: order.append("upgrade") or False),
    ):
        migrate.run_migration()

    assert order == ["alter", "upgrade"]


def test_schema_sql_has_the_embedding_columns_so_fresh_databases_skip_the_wipe() -> None:
    schema = (Path(__file__).resolve().parent.parent / "schema.sql").read_text(encoding="utf-8")
    faces = schema[schema.index("CREATE TABLE IF NOT EXISTS faces") :]
    faces = faces[: faces.index(") ENGINE")]

    assert "embedding_model VARCHAR(32)     NOT NULL" in faces
    assert "det_score       FLOAT           NULL" in faces
    assert "float32 embedding, 2048 bytes for 512D" in faces
    assert "Cosine distance" in faces
    assert "128D" not in schema


def test_upgrade_ddl_matches_schema_sql_column_definitions() -> None:
    import migrate

    ddl = "\n".join(sql for _, sql in migrate._EMBEDDING_MODEL_SCHEMA) + migrate._EMBEDDING_MODEL_MARKER_COLUMN[1]
    assert "VARCHAR(32) NOT NULL" in ddl
    assert "float32 embedding, 2048 bytes for 512D" in ddl
    assert f"DEFAULT {migrate.DEFAULT_DISTANCE_THRESHOLD}" in ddl


@pytest.mark.integration
def test_embedding_model_upgrade_wipes_once_on_a_legacy_database(
    test_db: str, db_conn, db_pool, seed_user, seed_project, seed_images, seed_faces
) -> None:
    import migrate

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sdc_claims (commons_page_id, wikidata_qid, project_id, face_id) VALUES (%s, %s, %s, %s)",
            (seed_images[0]["commons_page_id"], "Q42", seed_project["id"], seed_faces["target"][0]),
        )
        cur.execute(
            "UPDATE projects SET status = 'completed', completion_reason = 'no_faces', images_total = 10, "
            "images_processed = 10, faces_confirmed = 5, sdc_write_requested = 1, worker_claimed_by = 'w1', "
            "distance_threshold = 0.6 WHERE id = %s",
            (seed_project["id"],),
        )
        # Make the schema look like the pre-upgrade one.
        cur.execute("ALTER TABLE faces DROP COLUMN embedding_model, DROP COLUMN det_score")

    try:
        assert migrate.apply_embedding_model_upgrade() is True

        with db_conn.cursor() as cur:
            for table in ("images", "faces", "sdc_claims"):
                cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                assert cur.fetchone()["n"] == 0, table
            cur.execute("SELECT COUNT(*) AS n FROM users")
            assert cur.fetchone()["n"] == 1
            cur.execute("SELECT * FROM projects WHERE id = %s", (seed_project["id"],))
            project = cur.fetchone()
            cur.execute("DESCRIBE faces")
            columns = {row["Field"]: row for row in cur.fetchall()}

        assert project["status"] == "active"
        assert project["images_total"] == project["images_processed"] == project["faces_confirmed"] == 0
        assert project["completion_reason"] is None
        assert project["sdc_write_requested"] == 0
        assert project["worker_claimed_by"] is None
        assert project["distance_threshold"] == pytest.approx(migrate.DEFAULT_DISTANCE_THRESHOLD)
        assert "embedding_model" in columns and "det_score" in columns

        # Re-run: data created after the upgrade must survive.
        with db_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO images (project_id, commons_page_id, file_title, status) VALUES (%s, %s, %s, 'processed')",
                (seed_project["id"], 777, "File:New.jpg"),
            )
            image_id = cur.lastrowid
            cur.execute(
                "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
                "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 1, 2, 3, 4)",
                (image_id, b"\x00" * 2048),
            )

        assert migrate.apply_embedding_model_upgrade() is False

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM faces")
            assert cur.fetchone()["n"] == 1
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DESCRIBE faces")
            present = {row["Field"] for row in cur.fetchall()}
            if "embedding_model" not in present:
                migrate.apply_embedding_model_upgrade()


@pytest.mark.integration
def test_embedding_model_upgrade_is_a_noop_on_a_fresh_database(test_db: str, db_pool, seed_user, seed_project) -> None:
    import migrate

    assert migrate.apply_embedding_model_upgrade() is False
    from database import execute_query

    assert len(execute_query("SELECT id FROM projects WHERE id = %s", (seed_project["id"],))) == 1


@pytest.mark.integration
def test_run_migration_upgrades_a_legacy_faces_schema_end_to_end(test_db: str, db_conn, db_pool) -> None:
    """The full entry point must complete against a pre-upgrade faces table.

    ``aligned`` used to be positioned ``AFTER det_score``, but ``det_score`` is
    created later, by ``apply_embedding_model_upgrade()``. On a real legacy
    database that failed with "Unknown column 'det_score'" (error 1054) before
    the upgrade could run. The upgrade's own tests cannot catch this: they start
    from a schema that already has ``aligned``.
    """
    import migrate

    with db_conn.cursor() as cur:
        # Simulate a 0.10.x faces table: none of the columns this release adds
        # exist yet. (aligned and det_score shipped together, so a legacy table
        # has neither.)
        cur.execute("ALTER TABLE faces DROP COLUMN aligned, DROP COLUMN det_score, DROP COLUMN embedding_model")

    try:
        migrate.run_migration()

        with db_conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM faces")
            columns = [row["Field"] for row in cur.fetchall()]

        assert {"aligned", "det_score", "embedding_model"} <= set(columns)
        # Same order as schema.sql: the ALTER list adds aligned after the legacy
        # confidence column, then the upgrade inserts det_score after confidence.
        assert columns.index("confidence") < columns.index("det_score") < columns.index("aligned")
        assert columns.index("encoding") < columns.index("embedding_model")
    finally:
        # Whatever happened, leave a complete faces table for the remaining tests.
        with db_conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM faces LIKE 'aligned'")
            aligned_missing = cur.fetchone() is None
        if aligned_missing:
            database.init_db(pool_size=1)
            try:
                migrate.apply_embedding_model_upgrade()
                migrate._apply_alter_migrations()
            finally:
                database.close_pool()


def test_alter_migrations_include_liftwing_prefilter():
    descriptions = " ".join(desc for desc, _sql in _ALTER_MIGRATIONS)

    assert "prefilter_mode" in descriptions
    assert "prefilter_keep" in descriptions
    assert "prefilter_reason" in descriptions
    assert "prefilter_checked_at" in descriptions
    assert "service_status" in descriptions


def test_alter_migrations_include_faces_aligned():
    matching = [sql for _desc, sql in _ALTER_MIGRATIONS if sql.startswith("ALTER TABLE faces") and "aligned" in sql]

    assert len(matching) == 1
    assert "DEFAULT 1" in matching[0]
    # AFTER confidence, not det_score: det_score is only added later, by
    # apply_embedding_model_upgrade(), so anchoring on it fails on a legacy DB.
    assert "AFTER confidence" in matching[0]


def test_alter_migrations_drop_the_unused_sessions_table():
    matching = [sql for _desc, sql in _ALTER_MIGRATIONS if "sessions" in sql]

    assert matching == ["DROP TABLE IF EXISTS sessions"]


def test_schema_sql_no_longer_defines_a_sessions_table() -> None:
    schema = (Path(__file__).resolve().parent.parent / "schema.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS sessions" not in schema
    # reset still drops the legacy table from older databases.
    assert "sessions" in _ALL_TABLES


def test_alter_migrations_refresh_the_distance_threshold_comment():
    matching = [
        sql
        for _desc, sql in _ALTER_MIGRATIONS
        if sql.startswith("ALTER TABLE projects") and "distance_threshold" in sql and "MODIFY COLUMN" in sql
    ]

    assert len(matching) == 1
    # The comment must not point at a document that no longer exists, and must
    # match schema.sql / the embedding upgrade.
    assert "provisional default)" in matching[0]
    assert "docs/plans" not in matching[0]


def test_alter_migrations_never_anchor_on_columns_the_upgrade_creates():
    """The ALTER list runs before apply_embedding_model_upgrade(), so an AFTER
    on a column only the upgrade creates (det_score, embedding_model) breaks the
    full migration on a pre-upgrade database with error 1054."""
    for description, sql in _ALTER_MIGRATIONS:
        assert "AFTER det_score" not in sql, description
        assert "AFTER embedding_model" not in sql, description


@pytest.mark.integration
def test_alter_migrations_remove_a_legacy_sessions_table(test_db: str, db_conn, db_pool) -> None:
    """Older databases still carry the unused sessions table; the migration drops it."""
    import migrate

    with db_conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE sessions (id VARCHAR(255) PRIMARY KEY, user_id BIGINT UNSIGNED NULL, "
            "data MEDIUMBLOB NOT NULL, expires_at DATETIME NOT NULL, "
            "CONSTRAINT fk_sessions_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)"
        )

    migrate._apply_alter_migrations()

    with db_conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'sessions'")
        assert cur.fetchone() is None


@pytest.mark.integration
def test_alter_migrations_apply_the_current_distance_threshold_comment(test_db: str, db_conn, db_pool) -> None:
    """Every database must end up with the schema.sql comment, not just fresh ones.

    The comment is also written by apply_embedding_model_upgrade(), which runs
    once. A database migrated before the text changed would otherwise keep a
    comment pointing at a deleted document.
    """
    import migrate

    with db_conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE projects MODIFY COLUMN distance_threshold FLOAT NOT NULL DEFAULT 0.5 COMMENT 'stale text'"
        )

    migrate._apply_alter_migrations()

    with db_conn.cursor() as cur:
        cur.execute("SHOW FULL COLUMNS FROM projects LIKE 'distance_threshold'")
        comment = cur.fetchone()["Comment"]

    assert comment == "Cosine distance threshold for autonomous inference (provisional default)"


def test_schema_sql_defines_faces_aligned_column():
    statements = load_schema(str(Path(__file__).resolve().parent.parent / "schema.sql"))
    faces = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS faces" in stmt)

    assert "aligned" in faces
    assert "DEFAULT 1" in faces


def test_all_tables_include_service_status():
    assert "service_status" in _ALL_TABLES


def test_all_tables_include_worker_heartbeats():
    assert "worker_heartbeats" in _ALL_TABLES
    assert "worker_heartbeat" in _ALL_TABLES


def test_schema_sql_defines_session_epoch_and_worker_heartbeats():
    statements = load_schema(str(Path(__file__).resolve().parent.parent / "schema.sql"))
    users = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS users" in stmt)
    heartbeats = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS worker_heartbeats" in stmt)

    assert "session_epoch" in users
    assert "INT UNSIGNED" in users.split("session_epoch", 1)[1].split(",", 1)[0]
    for column in ("worker_id", "state", "last_seen", "expected_back_until", "last_progress", "phase", "started_at"):
        assert column in heartbeats
    assert "'running'" in heartbeats and "'stopping'" in heartbeats


def test_alter_migrations_add_session_epoch_and_worker_heartbeats():
    by_description = dict(_ALTER_MIGRATIONS)
    epoch = by_description["Add session_epoch to users"]
    assert "ADD COLUMN session_epoch" in epoch
    assert "AFTER leaderboard_opt_out" in epoch  # keeps the column order identical to schema.sql
    table = by_description["Create worker_heartbeats table"]
    assert "CREATE TABLE IF NOT EXISTS worker_heartbeats" in table  # idempotent by construction


def test_all_tables_include_worker_wake():
    assert "worker_wake" in _ALL_TABLES  # so --reset drops and recreates it


def test_schema_sql_defines_worker_wake_and_image_detect_attempts():
    statements = load_schema(str(Path(__file__).resolve().parent.parent / "schema.sql"))
    wake = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS worker_wake" in stmt)
    images = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS images" in stmt)

    wake_columns = wake.split("CREATE TABLE", 1)[1]  # past the explanatory comment
    assert "PRIMARY KEY" in wake_columns
    assert "BIGINT UNSIGNED" in wake_columns.split("seq", 1)[1].split("\n", 1)[0]
    columns = images.split("CREATE TABLE", 1)[1]
    attempts = columns.split("detect_attempts", 1)[1].split("\n", 1)[0]
    assert "TINYINT UNSIGNED NOT NULL DEFAULT 0" in attempts
    isolate = " ".join(columns.split("detect_isolate", 1)[1].split("\n", 1)[0].split())  # schema.sql aligns columns
    assert "TINYINT(1) NOT NULL DEFAULT 0" in isolate
    assert columns.index("detect_attempts") < columns.index("detect_isolate") < columns.index("created_at")
    projects = next(stmt for stmt in statements if "CREATE TABLE IF NOT EXISTS projects" in stmt).split(
        "CREATE TABLE", 1
    )[1]
    assert (
        projects.index("last_inference_min_confirmed")
        < projects.index("last_inference_run")
        < projects.index("last_inference_at")
        < projects.index("last_inference_matches")
        < projects.index("sdc_write_requested")
    )  # same order as the ALTERs
    # Strike counting must not need a seed row: wake.request_wake() is an upsert.
    assert not any("INSERT" in stmt and "worker_wake" in stmt for stmt in statements)


def test_alter_migrations_add_worker_wake_and_image_detect_attempts():
    by_description = dict(_ALTER_MIGRATIONS)
    table = by_description["Create worker_wake table"]
    assert "CREATE TABLE IF NOT EXISTS worker_wake" in table  # idempotent by construction
    # One ALTER per table: it applies all its columns or none, and a large table is rebuilt once.
    images = by_description["Add detect_attempts and detect_isolate to images"]
    assert "ADD COLUMN detect_attempts TINYINT UNSIGNED NOT NULL DEFAULT 0" in images
    assert "AFTER error_message" in images  # same column order as schema.sql
    assert "ADD COLUMN detect_isolate TINYINT(1) NOT NULL DEFAULT 0" in images
    assert images.index("detect_attempts") < images.index("AFTER detect_attempts")  # the flag follows the counter
    projects = by_description["Add the last inference run record to projects"]
    assert "ADD COLUMN last_inference_run INT UNSIGNED NOT NULL DEFAULT 0" in projects
    assert "AFTER last_inference_min_confirmed" in projects
    assert "ADD COLUMN last_inference_at DATETIME NULL" in projects and "AFTER last_inference_run" in projects
    assert "ADD COLUMN last_inference_matches INT UNSIGNED NULL" in projects and "AFTER last_inference_at" in projects
    assert sum("ADD COLUMN last_inference_at" in sql for _d, sql in _ALTER_MIGRATIONS) == 1


def test_schema_sql_defines_liftwing_prefilter_objects():
    statements = load_schema(str(Path(__file__).resolve().parent.parent / "schema.sql"))
    joined = "\n".join(statements)

    assert "CREATE TABLE IF NOT EXISTS service_status" in joined
    assert "INSERT IGNORE INTO service_status" in joined
    assert "prefilter_keep" in joined
    assert "prefilter_mode" in joined


def test_schema_semicolons_are_only_statement_terminators():
    """Guards load_schema's naive ';' split.

    A semicolon anywhere but as a lone statement terminator (inside a '--'
    comment or a quoted COMMENT string) chops the surrounding statement in
    half. That mistake is invisible to the unit tests, which run against fake
    cursors and never parse the real file, and only fails on a real database.
    """
    schema_path = Path(__file__).resolve().parent.parent / "schema.sql"
    offenders = [
        (number, line)
        for number, line in enumerate(schema_path.read_text(encoding="utf-8").splitlines(), 1)
        if ";" in line and not (line.rstrip().endswith(";") and line.rstrip().count(";") == 1 and "--" not in line)
    ]

    assert offenders == []
