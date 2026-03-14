"""
Database schema migration script for WikiVisage.

Reads schema.sql and executes it against the configured MariaDB database.
Safe to run multiple times — all CREATE TABLE statements use IF NOT EXISTS.

Usage:
    python migrate.py            # Apply migrations
    python migrate.py --reset    # Drop all tables and recreate from scratch
"""

import argparse
import logging
import os
import sys

from database import DatabaseError, close_pool, get_connection, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# Drop order respects foreign key dependencies (children first).
_ALL_TABLES = ["faces", "images", "projects", "sessions", "worker_heartbeat", "users"]


def load_schema(path: str) -> list[str]:
    """
    Load and parse SQL statements from schema file.

    Args:
        path: Path to the SQL schema file.

    Returns:
        List of individual SQL statements to execute.

    Raises:
        FileNotFoundError: If the schema file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Split on semicolons, strip whitespace, filter empty/comment-only chunks
    raw_statements = content.split(";")
    statements = []

    for raw in raw_statements:
        # Remove leading/trailing whitespace
        stmt = raw.strip()

        # Skip empty strings
        if not stmt:
            continue

        # Skip chunks that are only comments
        lines = [line.strip() for line in stmt.splitlines() if line.strip() and not line.strip().startswith("--")]
        if not lines:
            continue

        statements.append(stmt)

    return statements


def reset_database() -> None:
    """Drop all tables and recreate the schema from scratch."""
    logger.warning("Resetting database — dropping all tables")

    init_db(pool_size=1)

    try:
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0")
            for table in _ALL_TABLES:
                logger.info(f"Dropping table: {table}")
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            conn.commit()
            logger.info("All tables dropped")
    finally:
        close_pool()

    run_migration()


def run_migration() -> None:
    """
    Execute schema migration.

    Loads schema.sql, connects to the database, and executes each
    statement sequentially within a single transaction.
    Then applies any ALTER TABLE migrations for existing databases.

    Raises:
        DatabaseError: If any statement fails.
        FileNotFoundError: If schema.sql is missing.
    """
    logger.info("Starting schema migration")

    statements = load_schema(SCHEMA_FILE)
    logger.info(f"Loaded {len(statements)} SQL statements from {SCHEMA_FILE}")

    if not statements:
        logger.warning("No SQL statements found in schema file")
        return

    init_db(pool_size=1)

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                for i, stmt in enumerate(statements, 1):
                    # Log first line of each statement for context
                    first_line = stmt.splitlines()[0].strip() if stmt.splitlines() else stmt[:80]
                    logger.info(f"Executing statement {i}/{len(statements)}: {first_line}")

                    try:
                        cursor.execute(stmt)
                    except Exception as e:
                        logger.error(f"Failed on statement {i}: {e}")
                        logger.error(f"Statement: {stmt[:200]}")
                        conn.rollback()
                        raise DatabaseError(f"Migration failed on statement {i}: {e}") from e

            conn.commit()
            logger.info("Schema migration completed successfully")

        # Apply incremental ALTER TABLE migrations for existing databases
        _apply_alter_migrations()

    finally:
        close_pool()


# Incremental migrations: each entry is (description, SQL).
# These are idempotent — they check before altering.
_ALTER_MIGRATIONS = [
    (
        "Add classified_by_user_id to faces",
        "ALTER TABLE faces ADD COLUMN classified_by_user_id BIGINT UNSIGNED NULL "
        "COMMENT 'User who classified this face' AFTER classified_by",
    ),
    (
        "Add FK for classified_by_user_id",
        "ALTER TABLE faces ADD CONSTRAINT fk_faces_classified_by_user "
        "FOREIGN KEY (classified_by_user_id) REFERENCES users (id) ON DELETE SET NULL",
    ),
    (
        "Add index for classified_by_user_id",
        "ALTER TABLE faces ADD INDEX idx_faces_classified_by_user (classified_by_user_id)",
    ),
    (
        "Add p18_thumb_url to projects",
        "ALTER TABLE projects ADD COLUMN p18_thumb_url VARCHAR(1024) NULL "
        "COMMENT 'Cached Wikidata P18 image thumbnail URL' AFTER status",
    ),
    (
        "Create worker_heartbeat table",
        "CREATE TABLE IF NOT EXISTS worker_heartbeat ("
        "  id INT NOT NULL DEFAULT 1 PRIMARY KEY,"
        "  last_seen DATETIME NOT NULL,"
        "  CONSTRAINT single_row CHECK (id = 1)"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci",
    ),
    (
        "Add sdc_write_requested to projects",
        "ALTER TABLE projects ADD COLUMN sdc_write_requested TINYINT(1) NOT NULL DEFAULT 0 "
        "COMMENT '1=user requested SDC writes, worker picks up' AFTER faces_confirmed",
    ),
    (
        "Add sdc_write_error to projects",
        "ALTER TABLE projects ADD COLUMN sdc_write_error VARCHAR(1024) NULL "
        "COMMENT 'Error message from last SDC write attempt' AFTER sdc_write_requested",
    ),
    (
        "Add bootstrapped flag to images",
        "ALTER TABLE images ADD COLUMN bootstrapped TINYINT(1) NOT NULL DEFAULT 0 "
        "COMMENT '1=image found via P180 bootstrap' AFTER face_count",
    ),
    (
        "Add detection_width to images",
        "ALTER TABLE images ADD COLUMN detection_width MEDIUMINT UNSIGNED NULL "
        "COMMENT 'Image width in pixels at which face detection was run' AFTER face_count",
    ),
    (
        "Add detection_height to images",
        "ALTER TABLE images ADD COLUMN detection_height MEDIUMINT UNSIGNED NULL "
        "COMMENT 'Image height in pixels at which face detection was run' AFTER detection_width",
    ),
    (
        "Add superseded_by to faces",
        "ALTER TABLE faces ADD COLUMN superseded_by BIGINT UNSIGNED NULL "
        "COMMENT 'FK to replacement face after bbox edit; NULL=active' AFTER sdc_written",
    ),
    (
        "Add FK for superseded_by",
        "ALTER TABLE faces ADD CONSTRAINT fk_faces_superseded_by "
        "FOREIGN KEY (superseded_by) REFERENCES faces (id) ON DELETE SET NULL",
    ),
    (
        "Add index for superseded_by",
        "ALTER TABLE faces ADD INDEX idx_faces_superseded (superseded_by)",
    ),
    (
        "Widen detection_width from SMALLINT to MEDIUMINT",
        "ALTER TABLE images MODIFY COLUMN detection_width MEDIUMINT UNSIGNED NULL "
        "COMMENT 'Image width in pixels at which face detection was run'",
    ),
    (
        "Widen detection_height from SMALLINT to MEDIUMINT",
        "ALTER TABLE images MODIFY COLUMN detection_height MEDIUMINT UNSIGNED NULL "
        "COMMENT 'Image height in pixels at which face detection was run'",
    ),
    (
        "Add index on projects.sdc_write_requested",
        "ALTER TABLE projects ADD INDEX idx_projects_sdc_write_requested (sdc_write_requested)",
    ),
    (
        "Add composite index on images(project_id, bootstrapped)",
        "ALTER TABLE images ADD INDEX idx_images_project_bootstrapped (project_id, bootstrapped)",
    ),
    (
        "Add sdc_removal_pending to faces",
        "ALTER TABLE faces ADD COLUMN sdc_removal_pending TINYINT(1) NOT NULL DEFAULT 0 "
        "COMMENT '1=P180 claim removal queued (rejected bootstrap face)' AFTER sdc_written",
    ),
    (
        "Add index on faces(sdc_removal_pending)",
        "ALTER TABLE faces ADD INDEX idx_faces_sdc_removal (sdc_removal_pending)",
    ),
]


def _apply_alter_migrations() -> None:
    """Apply ALTER TABLE migrations, skipping those already applied."""
    IDEMPOTENT_ERROR_CODES = {
        1060,  # ER_DUP_FIELDNAME — column already exists
        1061,  # ER_DUP_KEYNAME — index/key already exists
        1826,  # ER_DUP_CONSTRAINT_NAME — FK constraint already exists
    }
    logger.info(f"Applying {len(_ALTER_MIGRATIONS)} incremental migrations")
    with get_connection() as conn, conn.cursor() as cursor:
        for desc, sql in _ALTER_MIGRATIONS:
            try:
                cursor.execute(sql)
                conn.commit()
                logger.info(f"Applied migration: {desc}")
            except Exception as e:
                conn.rollback()
                errno = getattr(e, "args", (None,))[0] if hasattr(e, "args") else None
                if isinstance(errno, int) and errno in IDEMPOTENT_ERROR_CODES:
                    logger.info(f"Skipped migration (already applied): {desc}")
                elif isinstance(errno, int) and errno == 1005 and "errno: 121" in str(e):
                    # MariaDB reports duplicate FK constraint names as
                    # error 1005 with inner errno 121, not 1826 like MySQL.
                    logger.info(f"Skipped migration (already applied): {desc}")
                else:
                    raise DatabaseError(f"Migration failed: {desc} — {e}") from e


def verify_tables() -> None:
    """
    Verify that all expected tables exist after migration.
    """
    expected_tables = [
        "users",
        "sessions",
        "projects",
        "images",
        "faces",
        "worker_heartbeat",
    ]

    init_db(pool_size=1)

    try:
        with get_connection() as conn, conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            rows = cursor.fetchall()

            # DictCursor returns rows like {"Tables_in_dbname": "tablename"}
            existing = set()
            for row in rows:
                # Column name varies based on database name, grab first value
                existing.add(list(row.values())[0])

            logger.info(f"Tables in database: {sorted(existing)}")

            missing = [t for t in expected_tables if t not in existing]
            if missing:
                logger.error(f"Missing tables after migration: {missing}")
                sys.exit(1)

            logger.info(f"All {len(expected_tables)} expected tables verified")

    finally:
        close_pool()


def main() -> None:
    """Entry point for the migration script."""
    parser = argparse.ArgumentParser(description="WikiVisage database migration")
    parser.add_argument("--reset", action="store_true", help="Drop all tables and recreate from scratch")
    args = parser.parse_args()

    try:
        if args.reset:
            reset_database()
        else:
            run_migration()
        verify_tables()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except DatabaseError as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
