import os
from datetime import UTC, datetime
from unittest.mock import patch

os.environ.setdefault("TOOL_TOOLSDB_USER", "testuser")
os.environ.setdefault("TOOL_TOOLSDB_PASSWORD", "testpass")
os.environ.setdefault("WIKIVISAGE_DB_NAME", "testdb")
os.environ.setdefault("TOOL_TOOLSDB_HOST", "localhost")
os.environ.setdefault("OAUTH_CLIENT_ID", "test-client-id")
os.environ.setdefault("OAUTH_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

with patch("database.init_db"):
    import app as app_module

    app = app_module.app
    _snap_thumb_width = app_module._snap_thumb_width
    _is_safe_url = app_module._is_safe_url
    commons_thumb_url = app_module.commons_thumb_url
    _validate_csrf = app_module._validate_csrf
    _THUMB_STEPS = app_module._THUMB_STEPS


def test_snap_exact_match():
    assert _snap_thumb_width(330) == 330


def test_snap_rounds_up():
    assert _snap_thumb_width(100) == 120


def test_snap_minimum():
    assert _snap_thumb_width(1) == 20


def test_snap_maximum():
    assert _snap_thumb_width(5000) == 3840


def test_snap_boundary():
    assert _snap_thumb_width(20) == 20
    assert _snap_thumb_width(21) == 40


def test_snap_all_steps():
    for step in _THUMB_STEPS:
        assert _snap_thumb_width(step) == step


def test_snap_between_steps():
    assert _snap_thumb_width(251) == 330
    assert _snap_thumb_width(961) == 1280


def test_safe_relative_path():
    assert _is_safe_url("/dashboard") is True


def test_safe_relative_with_query():
    assert _is_safe_url("/project/1?tab=model") is True


def test_unsafe_absolute_url():
    assert _is_safe_url("https://evil.com/phish") is False


def test_unsafe_protocol_relative():
    assert _is_safe_url("//evil.com/phish") is False


def test_empty_string():
    assert _is_safe_url("") is False


def test_safe_plain_path():
    assert _is_safe_url("dashboard") is True


def test_basic_jpg():
    url = commons_thumb_url("File:Example.jpg", 330)
    assert "/330px-Example.jpg" in url


def test_strips_file_prefix():
    with_prefix = commons_thumb_url("File:Test.jpg", 330)
    without_prefix = commons_thumb_url("Test.jpg", 330)
    assert with_prefix == without_prefix


def test_spaces_to_underscores():
    url = commons_thumb_url("File:My Photo.jpg", 330)
    assert "My_Photo.jpg" in url


def test_svg_becomes_png():
    url = commons_thumb_url("File:Logo.svg", 330)
    assert url.endswith(".png")


def test_tiff_becomes_jpg():
    url = commons_thumb_url("File:Scan.tiff", 330)
    assert url.endswith(".jpg")


def test_webm_double_dash():
    url = commons_thumb_url("File:Video.webm", 330)
    assert "px--Video.webm.jpg" in url


def test_width_snapped():
    url = commons_thumb_url("File:X.jpg", 100)
    assert "/120px-" in url


def test_default_width():
    url = commons_thumb_url("File:X.jpg")
    assert "/330px-" in url


def test_ogv_video():
    url = commons_thumb_url("File:Clip.ogv", 330)
    assert "px--Clip.ogv.jpg" in url


def test_tif_extension():
    url = commons_thumb_url("File:Doc.tif", 330)
    assert url.endswith(".jpg")


def test_csrf_valid():
    with app.test_request_context("/submit", method="POST", data={"csrf_token": "abc123"}):
        from flask import session

        session["csrf_token"] = "abc123"
        assert _validate_csrf() is True


def test_csrf_invalid():
    with app.test_request_context("/submit", method="POST", data={"csrf_token": "xyz"}):
        from flask import session

        session["csrf_token"] = "abc"
        assert _validate_csrf() is False


def test_csrf_missing_session():
    with app.test_request_context("/submit", method="POST", data={"csrf_token": "abc"}):
        assert _validate_csrf() is False


def test_csrf_missing_form():
    with app.test_request_context("/submit", method="POST", data={}):
        from flask import session

        session["csrf_token"] = "abc"
        assert _validate_csrf() is False


def test_commons_thumb_route_requires_login_redirects_to_login():
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_commons_thumb_route_logged_in_redirects_to_generated_thumb(monkeypatch):
    app.config["TESTING"] = True
    client = app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC),
    }

    monkeypatch.setattr(app_module, "execute_query", lambda *args, **kwargs: [fake_user])
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/commons-thumb/File:X.jpg?width=100")

    assert response.status_code == 302
    assert response.headers["Location"] == commons_thumb_url("File:X.jpg", 100)


import pytest

try:
    from conftest import _make_encoding
except ModuleNotFoundError:
    from tests.conftest import _make_encoding


@pytest.mark.integration
def test_health_endpoint(integration_client):
    client, _, _ = integration_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy", "database": "connected"}


@pytest.mark.integration
def test_dashboard_empty(integration_client):
    client, _, _ = integration_client

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"WikiVisage" in response.data


@pytest.mark.integration
def test_project_detail_with_data(integration_client, seed_project, seed_images, seed_faces):
    client, _, _ = integration_client

    response = client.get(f"/project/{seed_project['id']}")

    assert response.status_code == 200
    assert b"Douglas Adams" in response.data or b"Q42" in response.data


@pytest.mark.integration
def test_api_classify_target(integration_client, seed_project, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_encoding(999)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": str(face_id),
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 1
    assert row["classified_by"] == "human"
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_classify_none(integration_client, seed_project, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_encoding(1000)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 0
    assert row["classified_by"] == "human"
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_undo_classify(integration_client, seed_project, seed_images, db_conn):
    client, _, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_encoding(1001)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    classify_response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": str(face_id),
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )
    assert classify_response.status_code == 200

    undo_response = client.post(
        "/api/undo-classify",
        data={"csrf_token": "testtoken"},
    )

    assert undo_response.status_code == 200
    assert undo_response.get_json()["status"] == "ok"

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] is None
    assert row["classified_by"] is None
    assert row["classified_by_user_id"] is None


@pytest.mark.integration
def test_api_reclassify_approve(integration_client, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id) "
            "VALUES (%s, %s, 50, 200, 200, 50, 0, 'model', NULL)",
            (seed_images[0]["id"], _make_encoding(1002)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/reclassify",
        data={
            "csrf_token": "testtoken",
            "face_id": str(face_id),
            "is_target": "1",
        },
    )

    assert response.status_code == 200

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 1
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_reclassify_reject(integration_client, seed_images, db_conn):
    client, _, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id, sdc_written) "
            "VALUES (%s, %s, 50, 200, 200, 50, 1, 'model', NULL, 0)",
            (seed_images[0]["id"], _make_encoding(1003)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/reclassify",
        data={
            "csrf_token": "testtoken",
            "face_id": str(face_id),
            "is_target": "0",
        },
    )

    assert response.status_code == 200

    with db_conn.cursor() as cur:
        cur.execute("SELECT is_target FROM faces WHERE id = %s", (face_id,))
        row = cur.fetchone()

    assert row["is_target"] == 0


@pytest.mark.integration
def test_api_sdc_status(integration_client, seed_project, seed_images, seed_faces):
    client, _, _ = integration_client

    response = client.get(f"/api/sdc-status/{seed_project['id']}")

    assert response.status_code == 200
    payload = response.get_json()
    assert "written" in payload
    assert "pending" in payload


@pytest.mark.integration
def test_api_classify_missing_fields(integration_client):
    client, _, _ = integration_client

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 400


@pytest.mark.integration
def test_api_classify_wrong_project(integration_client, seed_images):
    client, _, _ = integration_client

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": "999999",
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Unit tests for CSRF header validation (MEDIUM #6)
# ---------------------------------------------------------------------------


def test_csrf_via_header():
    """CSRF token sent via X-CSRFToken header should be accepted."""
    with app.test_request_context(
        "/submit",
        method="POST",
        data={},
        headers={"X-CSRFToken": "headertoken"},
    ):
        from flask import session

        session["csrf_token"] = "headertoken"
        assert _validate_csrf() is True


def test_csrf_header_wrong():
    """Wrong X-CSRFToken header should be rejected."""
    with app.test_request_context(
        "/submit",
        method="POST",
        data={},
        headers={"X-CSRFToken": "wrong"},
    ):
        from flask import session

        session["csrf_token"] = "correct"
        assert _validate_csrf() is False


def test_csrf_form_takes_precedence_over_header():
    """When both form token and header token are present, form token is checked first."""
    with app.test_request_context(
        "/submit",
        method="POST",
        data={"csrf_token": "formtoken"},
        headers={"X-CSRFToken": "headertoken"},
    ):
        from flask import session

        session["csrf_token"] = "formtoken"
        assert _validate_csrf() is True


# ---------------------------------------------------------------------------
# Unit tests for session fixation prevention (MEDIUM #9)
# ---------------------------------------------------------------------------


def test_session_cleared_on_login_callback(monkeypatch):
    """OAuth callback should clear session before setting user_id (session fixation prevention)."""
    client = app.test_client()

    fake_token = {"access_token": "tok", "refresh_token": "ref", "expires_at": 9999999999, "expires_in": 14400}
    fake_userinfo = {"sub": 42, "username": "TestUser"}
    fake_user_row = [
        {
            "id": 1,
            "wiki_user_id": 42,
            "wiki_username": "TestUser",
            "access_token": "tok",
            "refresh_token": "ref",
            "token_expires_at": datetime.now(UTC),
        }
    ]

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: fake_user_row)
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"TestUser"})

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_userinfo

    monkeypatch.setattr(app_module.requests, "get", lambda *a, **kw: FakeResp())

    class FakeOAuth:
        def fetch_token(self, *a, **kw):
            return fake_token

    monkeypatch.setattr(app_module, "_make_oauth_session", lambda state=None: FakeOAuth())

    with client.session_transaction() as sess:
        sess["oauth_state"] = "fakestate"
        sess["login_next"] = "/dashboard"
        sess["attacker_data"] = "should_be_removed"

    client.get("/auth/callback?state=fakestate&code=fakecode")

    with client.session_transaction() as sess:
        assert sess.get("user_id") == 1
        assert "attacker_data" not in sess


# ---------------------------------------------------------------------------
# Unit tests for login_required next= parameter (LOW #14)
# ---------------------------------------------------------------------------


def test_login_required_uses_relative_path():
    """login_required should redirect to login with a relative path, not absolute URL."""
    client = app.test_client()

    response = client.get("/dashboard")

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "/login" in location
    if "next=" in location:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(location)
        qs = parse_qs(parsed.query)
        next_val = qs.get("next", [""])[0]
        assert _is_safe_url(next_val), f"next= value '{next_val}' is not safe (would be rejected)"


# ---------------------------------------------------------------------------
# Unit tests for wake file (LOW #15)
# ---------------------------------------------------------------------------


def test_wake_file_path_consistent():
    """Both project creation and SDC write should use the same wake file name."""
    import inspect

    source = inspect.getsource(app_module)
    wake_refs = [line.strip() for line in source.splitlines() if "worker-wake" in line or "worker_wake" in line]
    for ref in wake_refs:
        if ref.lstrip().startswith("#"):
            continue
        assert ".worker-wake-up" in ref, f"Inconsistent wake file name in: {ref}"
