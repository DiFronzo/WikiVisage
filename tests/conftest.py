"""
Shared test fixtures for WikiVisage.

Two fixture tiers:
  - Unit fixtures: no DB required (monkeypatch_env, flask_app, client)
  - Integration fixtures: require a local MariaDB (Docker)
    Marked with @pytest.mark.integration, skipped in CI.

Integration tests use a dedicated `wikiface_test` database that is
created fresh per session and dropped on teardown.
"""

import importlib
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pymysql
import pytest
from pymysql.cursors import DictCursor

# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the 'integration' marker."""
    config.addinivalue_line(
        "markers",
        "integration: tests requiring a local MariaDB (Docker)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when WIKIVISAGE_TEST_DB env var is not set."""
    skip_integration = pytest.mark.skip(
        reason="Set WIKIVISAGE_TEST_DB=1 to run integration tests (requires Docker MariaDB)"
    )
    for item in items:
        if "integration" in item.keywords:
            if not os.environ.get("WIKIVISAGE_TEST_DB"):
                item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Unit-test fixtures (no DB)
# ---------------------------------------------------------------------------


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Set minimal env vars for database.py unit tests."""
    monkeypatch.setenv("TOOL_TOOLSDB_USER", "testuser")
    monkeypatch.setenv("TOOL_TOOLSDB_PASSWORD", "testpass")
    monkeypatch.setenv("WIKIVISAGE_DB_NAME", "testdb")
    monkeypatch.setenv("TOOL_TOOLSDB_HOST", "localhost")


@pytest.fixture
def flask_app(monkeypatch):
    """Create a Flask app with mocked DB for pure unit tests."""
    monkeypatch.setenv("TOOL_TOOLSDB_USER", "testuser")
    monkeypatch.setenv("TOOL_TOOLSDB_PASSWORD", "testpass")
    monkeypatch.setenv("WIKIVISAGE_DB_NAME", "testdb")
    monkeypatch.setenv("TOOL_TOOLSDB_HOST", "localhost")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "http://localhost/auth/callback")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret")
    monkeypatch.setenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

    import database

    monkeypatch.setattr(database, "init_db", lambda *args, **kwargs: None)

    if "app" in sys.modules:
        app_module = importlib.reload(sys.modules["app"])
    else:
        app_module = importlib.import_module("app")

    create_app = getattr(app_module, "create_app", None)
    if callable(create_app):
        application = create_app()
    else:
        application = app_module.app

    application.config["TESTING"] = True
    application.config["SECRET_KEY"] = "test-secret"
    return application


@pytest.fixture
def client(flask_app):
    """Flask test client backed by mocked DB."""
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Integration-test fixtures (real MariaDB via Docker)
# ---------------------------------------------------------------------------

# Connection params for the local Docker MariaDB.
# Override with env vars if your setup differs.
_DB_HOST = os.environ.get("TOOL_TOOLSDB_HOST", "127.0.0.1")
_DB_PORT = int(os.environ.get("TOOL_TOOLSDB_PORT", "3306"))
_DB_USER = os.environ.get("TOOL_TOOLSDB_USER", "root")
_DB_PASS = os.environ.get("TOOL_TOOLSDB_PASSWORD", "devpass")
_TEST_DB = "wikiface_test"


def _raw_conn(database: str | None = None) -> pymysql.Connection:
    """Open a raw PyMySQL connection (not pooled)."""
    kwargs = {
        "host": _DB_HOST,
        "port": _DB_PORT,
        "user": _DB_USER,
        "password": _DB_PASS,
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": True,
    }
    if database:
        kwargs["database"] = database
    return pymysql.connect(**kwargs)


@pytest.fixture(scope="session")
def test_db():
    """Create the wikiface_test database and run schema + migrations.

    Yields the database name. Drops the DB on teardown.
    """
    conn = _raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{_TEST_DB}`")
            cur.execute(f"CREATE DATABASE `{_TEST_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    finally:
        conn.close()

    # Now run the schema against the test DB via the real migrate module.
    # We need to set env vars so database.py connects to the test DB.
    old_env = {
        "TOOL_TOOLSDB_USER": os.environ.get("TOOL_TOOLSDB_USER"),
        "TOOL_TOOLSDB_PASSWORD": os.environ.get("TOOL_TOOLSDB_PASSWORD"),
        "TOOL_TOOLSDB_HOST": os.environ.get("TOOL_TOOLSDB_HOST"),
        "WIKIVISAGE_DB_NAME": os.environ.get("WIKIVISAGE_DB_NAME"),
    }
    os.environ["TOOL_TOOLSDB_USER"] = _DB_USER
    os.environ["TOOL_TOOLSDB_PASSWORD"] = _DB_PASS
    os.environ["TOOL_TOOLSDB_HOST"] = _DB_HOST
    os.environ["WIKIVISAGE_DB_NAME"] = _TEST_DB

    try:
        # Import migrate and run — uses database.init_db + get_connection internally
        import database
        from migrate import _apply_alter_migrations, load_schema

        database.init_db(pool_size=2)

        # Execute schema.sql
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "schema.sql",
        )
        statements = load_schema(schema_path)
        with database.get_connection() as db_conn, db_conn.cursor() as cursor:
            for stmt in statements:
                cursor.execute(stmt)
            db_conn.commit()

        # Apply ALTER migrations
        _apply_alter_migrations()

        database.close_pool()

        yield _TEST_DB
    finally:
        # Restore env
        for key, val in old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        # Drop test database
        conn = _raw_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{_TEST_DB}`")
        finally:
            conn.close()


@pytest.fixture
def db_conn(test_db):
    """Yield a raw PyMySQL connection to the test DB.

    Truncates all data tables after each test for isolation.
    """
    conn = _raw_conn(database=test_db)
    try:
        yield conn
    finally:
        # Clean up: truncate all data tables (order matters due to FKs)
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            for table in ("faces", "images", "projects", "sessions", "users", "worker_heartbeat"):
                cur.execute(f"TRUNCATE TABLE `{table}`")
            cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.close()


@pytest.fixture
def db_pool(test_db, monkeypatch):
    """Initialize the database.py connection pool pointing at the test DB.

    Closes the pool on teardown. Patches env vars so database.py connects
    to the test DB.
    """
    monkeypatch.setenv("TOOL_TOOLSDB_USER", _DB_USER)
    monkeypatch.setenv("TOOL_TOOLSDB_PASSWORD", _DB_PASS)
    monkeypatch.setenv("TOOL_TOOLSDB_HOST", _DB_HOST)
    monkeypatch.setenv("WIKIVISAGE_DB_NAME", _TEST_DB)

    import database

    # Reset pool state (in case previous test left it)
    database._pool = None
    database._db_config = {}

    database.init_db(pool_size=3)
    yield database
    database.close_pool()


# ---------------------------------------------------------------------------
# Seed-data helpers
# ---------------------------------------------------------------------------


def _make_encoding(seed: int = 0) -> bytes:
    """Generate a deterministic 128D float64 face encoding (1024 bytes)."""
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float64).tobytes()


@pytest.fixture
def seed_user(db_conn):
    """Insert a test user and return its row."""
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (wiki_user_id, wiki_username, access_token, "
            "refresh_token, token_expires_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                99999,
                "TestUser",
                "fake-access-token",
                "fake-refresh-token",
                (datetime.now(UTC) + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        user_id = cur.lastrowid
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()


@pytest.fixture
def seed_project(db_conn, seed_user):
    """Insert a test project (Q42 Douglas Adams) and return its row."""
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
            "distance_threshold, min_confirmed, status, images_total, images_processed) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                seed_user["id"],
                "Q42",
                "Douglas Adams",
                "Douglas Adams",
                0.6,
                5,
                "active",
                10,
                10,
            ),
        )
        project_id = cur.lastrowid
        cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
        return cur.fetchone()


@pytest.fixture
def seed_images(db_conn, seed_project):
    """Insert 5 test images for the project. Returns list of image rows."""
    images = []
    files = [
        ("File:Douglas_Adams_1.jpg", 100001),
        ("File:Douglas_Adams_2.jpg", 100002),
        ("File:Douglas_Adams_3.jpg", 100003),
        ("File:Douglas_Adams_4.jpg", 100004),
        ("File:Douglas_Adams_5.jpg", 100005),
    ]
    with db_conn.cursor() as cur:
        for title, page_id in files:
            cur.execute(
                "INSERT INTO images (project_id, commons_page_id, file_title, status, "
                "face_count, detection_width, detection_height) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (seed_project["id"], page_id, title, "processed", 1, 800, 600),
            )
            img_id = cur.lastrowid
            images.append({"id": img_id, "file_title": title, "commons_page_id": page_id})
    return images


@pytest.fixture
def seed_faces(db_conn, seed_images, seed_user):
    """Insert faces for each image: 5 target (match) + 5 non-target.

    Returns dict with 'target' and 'non_target' lists.
    """
    targets = []
    non_targets = []
    with db_conn.cursor() as cur:
        for i, img in enumerate(seed_images):
            # One target face per image
            cur.execute(
                "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, "
                "bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (img["id"], _make_encoding(i), 50, 200, 200, 50, 1, "human", seed_user["id"]),
            )
            targets.append(cur.lastrowid)

            # One non-target face per image
            cur.execute(
                "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, "
                "bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (img["id"], _make_encoding(100 + i), 300, 450, 450, 300, 0, "human", seed_user["id"]),
            )
            non_targets.append(cur.lastrowid)

    return {"target": targets, "non_target": non_targets}


@pytest.fixture
def seed_unclassified_faces(db_conn, seed_images):
    """Insert unclassified faces (is_target=NULL) for inference tests."""
    face_ids = []
    with db_conn.cursor() as cur:
        for i, img in enumerate(seed_images):
            cur.execute(
                "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, "
                "bbox_bottom, bbox_left) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (img["id"], _make_encoding(200 + i), 100, 300, 300, 100),
            )
            face_ids.append(cur.lastrowid)
    return face_ids


@pytest.fixture
def seed_bootstrap_image(db_conn, seed_project):
    """Insert a bootstrapped image with a bootstrap-classified face."""
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, "
            "face_count, detection_width, detection_height, bootstrapped) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (seed_project["id"], 200001, "File:Bootstrap_Test.jpg", "processed", 1, 800, 600, 1),
        )
        image_id = cur.lastrowid

        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, "
            "bbox_bottom, bbox_left, is_target, classified_by, sdc_written) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (image_id, _make_encoding(500), 50, 200, 200, 50, 1, "bootstrap", 1),
        )
        face_id = cur.lastrowid

    return {"image_id": image_id, "face_id": face_id}


@pytest.fixture
def integration_app(test_db, monkeypatch):
    """Create a Flask app connected to the real test DB.

    Returns (app, app_module) tuple. The app's DB pool is initialized
    and cleaned up automatically.
    """
    monkeypatch.setenv("TOOL_TOOLSDB_USER", _DB_USER)
    monkeypatch.setenv("TOOL_TOOLSDB_PASSWORD", _DB_PASS)
    monkeypatch.setenv("TOOL_TOOLSDB_HOST", _DB_HOST)
    monkeypatch.setenv("WIKIVISAGE_DB_NAME", _TEST_DB)
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "http://localhost/auth/callback")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

    import database

    # Reset pool so it re-initializes with test DB
    database._pool = None
    database._db_config = {}
    database.init_db(pool_size=3)

    if "app" in sys.modules:
        app_module = importlib.reload(sys.modules["app"])
    else:
        app_module = importlib.import_module("app")

    application = app_module.create_app()
    application.config["TESTING"] = True
    application.config["SECRET_KEY"] = "test-secret-key"

    yield application, app_module

    database.close_pool()


@pytest.fixture
def integration_client(integration_app, db_conn, seed_user):
    """Flask test client backed by real test DB with a logged-in user.

    The db_conn fixture handles truncation after each test.
    The seed_user is the authenticated user.
    """
    application, app_module = integration_app

    # Patch whitelist to allow our test user
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"TestUser"})

    test_client = application.test_client()

    # Set the user session to simulate login
    with test_client.session_transaction() as sess:
        sess["user_id"] = seed_user["id"]

    yield test_client, seed_user, app_module

    monkeypatch.undo()
