import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

import numpy as np
import pytest
import requests
from flask import Response, abort, g, session

from database import DatabaseError

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

    flask_app = app_module.app
    _snap_thumb_width = app_module._snap_thumb_width
    _is_safe_url = app_module._is_safe_url
    commons_thumb_url = app_module.commons_thumb_url
    _validate_csrf = app_module._validate_csrf
    _THUMB_STEPS = app_module._THUMB_STEPS


try:
    from conftest import _make_encoding
except ModuleNotFoundError:
    from tests.conftest import _make_encoding


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
    with flask_app.test_request_context():
        assert _is_safe_url("/dashboard") is True


def test_safe_relative_with_query():
    with flask_app.test_request_context():
        assert _is_safe_url("/project/1?tab=model") is True


def test_unsafe_absolute_url():
    with flask_app.test_request_context():
        assert _is_safe_url("https://evil.com/phish") is False


def test_unsafe_protocol_relative():
    with flask_app.test_request_context():
        assert _is_safe_url("//evil.com/phish") is False


def test_empty_string():
    with flask_app.test_request_context():
        assert _is_safe_url("") is False


def test_safe_plain_path():
    with flask_app.test_request_context():
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
    with flask_app.test_request_context("/submit", method="POST", data={"csrf_token": "abc123"}):
        from flask import session

        session["csrf_token"] = "abc123"
        assert _validate_csrf() is True


def test_csrf_invalid():
    with flask_app.test_request_context("/submit", method="POST", data={"csrf_token": "xyz"}):
        from flask import session

        session["csrf_token"] = "abc"
        assert _validate_csrf() is False


def test_csrf_missing_session():
    with flask_app.test_request_context("/submit", method="POST", data={"csrf_token": "abc"}):
        assert _validate_csrf() is False


def test_csrf_missing_form():
    with flask_app.test_request_context("/submit", method="POST", data={}):
        from flask import session

        session["csrf_token"] = "abc"
        assert _validate_csrf() is False


def test_commons_thumb_route_requires_login_redirects_to_login():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_commons_thumb_route_logged_in_redirects_to_generated_thumb(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC),
    }

    monkeypatch.setattr(app_module, "execute_query", lambda *args, **kwargs: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/commons-thumb/File:X.jpg?width=100")

    assert response.status_code == 302
    assert response.headers["Location"] == commons_thumb_url("File:X.jpg", 100)


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
    with flask_app.test_request_context(
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
    with flask_app.test_request_context(
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
    with flask_app.test_request_context(
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
    client = flask_app.test_client()

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
    client = flask_app.test_client()

    response = client.get("/dashboard")

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "/login" in location
    if "next=" in location:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(location)
        qs = parse_qs(parsed.query)
        next_val = qs.get("next", [""])[0]
        with flask_app.test_request_context():
            assert _is_safe_url(next_val), f"next= value '{next_val}' is not safe (would be rejected)"


# ---------------------------------------------------------------------------
# Unit tests for wake file (LOW #15)
# ---------------------------------------------------------------------------


def test_wake_file_path_consistent():
    """Both app.py and worker.py should import WAKE_FILE_PATH from config."""
    import inspect

    import config as config_module

    # config.py must define the canonical path using '.worker-wake-up'
    config_source = inspect.getsource(config_module)
    assert ".worker-wake-up" in config_source, "config.py must define WAKE_FILE_PATH with '.worker-wake-up'"

    # app.py must import WAKE_FILE_PATH from config (not define its own)
    app_source = inspect.getsource(app_module)
    assert "from config import WAKE_FILE_PATH" in app_source, "app.py must import WAKE_FILE_PATH from config"


def test_before_request_without_session_user_id_sets_no_user():
    with flask_app.test_request_context("/"):
        app_module.before_request()
        assert g.user is None


def test_before_request_handles_database_error_and_clears_session(monkeypatch):
    def _raise_db(*_args, **_kwargs):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    with flask_app.test_request_context("/"):
        session["user_id"] = 7
        app_module.before_request()
        assert g.user is None
        assert "user_id" not in session


def test_before_request_normalizes_bytes_tokens(monkeypatch):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": b"token-bytes",
        "refresh_token": b"refresh-bytes",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [fake_user.copy()])

    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        app_module.before_request()
        assert g.user["access_token"] == "token-bytes"
        assert g.user["refresh_token"] == "refresh-bytes"


def test_inject_i18n_helpers_ltr_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    data = app_module.inject_i18n_helpers()

    assert data["current_locale"] == "en"
    assert data["languages"] == app_module.LANGUAGES
    assert data["text_direction"] == "ltr"
    assert data["app_version"] == app_module.APP_VERSION


def test_inject_i18n_helpers_rtl_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "ar")
    data = app_module.inject_i18n_helpers()
    assert data["text_direction"] == "rtl"


def test_inject_worker_status_stale_heartbeat(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"is_stale": 1}])
    assert app_module.inject_worker_status() == {"worker_down": True}


def test_inject_worker_status_fresh_heartbeat(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"is_stale": 0}])
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_worker_status_missing_row_table_exists(monkeypatch):
    calls = {"count": 0}

    def _query(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [{"1": 1}]

    monkeypatch.setattr(app_module, "execute_query", _query)
    assert app_module.inject_worker_status() == {"worker_down": True}


def test_inject_worker_status_missing_row_table_missing(monkeypatch):
    calls = {"count": 0}

    def _query(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return []

    monkeypatch.setattr(app_module, "execute_query", _query)
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_worker_status_exception_returns_not_down(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "execute_query", _raise)
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_csrf_token_exposes_callable():
    with flask_app.test_request_context("/"):
        data = app_module.inject_csrf_token()
        assert "csrf_token" in data
        token = data["csrf_token"]()
        assert isinstance(token, str)
        assert len(token) == 64


def test_set_security_headers_sets_all_required_headers():
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)

        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-Frame-Options"] == "DENY"
        assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert result.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
        assert (
            "Strict-Transport-Security" not in result.headers
            or result.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
        )


def test_hsts_header_not_set_in_debug_mode():
    flask_app.debug = True
    try:
        with flask_app.test_request_context():
            response = flask_app.make_response(("ok", 200))
            result = app_module.set_security_headers(response)
            assert "Strict-Transport-Security" not in result.headers
    finally:
        flask_app.debug = False


def test_hsts_header_set_when_not_debug():
    flask_app.debug = False
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)
        assert result.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_csp_header_contains_required_directives():
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)
        csp = result.headers["Content-Security-Policy"]

        assert "default-src 'self'" in csp
        assert "script-src 'self' 'unsafe-inline'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp
        assert "img-src 'self' https://*.wikimedia.org data:" in csp
        assert "connect-src 'self'" in csp
        assert "font-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp


def test_csp_header_does_not_contain_nonce():
    """Nonces are not used in CSP because they make 'unsafe-inline' ignored,
    breaking inline event handlers (onclick, onchange, etc.)."""
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)
        csp = result.headers["Content-Security-Policy"]

        assert "nonce-" not in csp


class _FakeResponse:
    def __init__(self, headers=None, chunks=None, error=None, url="https://upload.wikimedia.org/file.jpg"):
        self.headers = headers or {}
        self._chunks = chunks or []
        self._error = error
        self.closed = False
        self.url = url

    def raise_for_status(self):
        if self._error:
            raise self._error

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True


def test_download_image_success(monkeypatch):
    resp = _FakeResponse(headers={"Content-Length": "6"}, chunks=[b"ab", b"cd", b"ef"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    data = app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert data == b"abcdef"


def test_download_image_content_length_too_large(monkeypatch):
    resp = _FakeResponse(headers={"Content-Length": "11"}, chunks=[b"abc"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    with pytest.raises(ValueError, match="Image too large"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_stream_exceeds_limit(monkeypatch):
    resp = _FakeResponse(headers={}, chunks=[b"12345", b"67890", b"x"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    with pytest.raises(ValueError, match="exceeded"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_raises_http_error(monkeypatch):
    resp = _FakeResponse(error=requests.HTTPError("bad response"))
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    with pytest.raises(requests.HTTPError):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")


def test_download_image_blocks_untrusted_host():
    with pytest.raises(ValueError, match="Blocked download from untrusted host"):
        app_module._download_image("https://evil.example.com/file.jpg")


def test_download_image_rejects_redirect_to_untrusted_host(monkeypatch):
    resp = _FakeResponse(
        headers={"Content-Length": "6"},
        chunks=[b"ab", b"cd", b"ef"],
        url="https://evil.example.com/redirected.jpg",
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    with pytest.raises(ValueError, match="Redirect to untrusted host"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")
    assert resp.closed is True


def test_download_image_allows_redirect_to_trusted_host(monkeypatch):
    resp = _FakeResponse(
        headers={"Content-Length": "3"},
        chunks=[b"abc"],
        url="https://commons.wikimedia.org/redirected.jpg",
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr(app_module, "_reject_private_ip", lambda _h: None)

    data = app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert data == b"abc"


def test_reject_private_ip_blocks_loopback(monkeypatch):
    monkeypatch.setattr(
        app_module.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(ValueError, match="private/reserved IP"):
        app_module._reject_private_ip("upload.wikimedia.org")


def test_reject_private_ip_blocks_rfc1918(monkeypatch):
    monkeypatch.setattr(
        app_module.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("10.0.0.1", 0))],
    )
    with pytest.raises(ValueError, match="private/reserved IP"):
        app_module._reject_private_ip("upload.wikimedia.org")


def test_reject_private_ip_allows_public(monkeypatch):
    monkeypatch.setattr(
        app_module.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("91.198.174.192", 0))],
    )
    app_module._reject_private_ip("upload.wikimedia.org")


def test_login_required_redirects_unauthenticated(monkeypatch):
    with flask_app.test_request_context("/protected?x=1"):
        g.user = None
        monkeypatch.setattr(app_module, "_", lambda x: x)

        @app_module.login_required
        def _protected():
            return "ok"

        resp = _protected()
        assert isinstance(resp, Response)
        assert resp.status_code == 302
        assert "/login?next=" in resp.location
        assert "next=" in resp.location and "protected" in resp.location


def test_login_required_allows_authenticated_user():
    with flask_app.test_request_context("/protected"):
        g.user = {"id": 1}

        @app_module.login_required
        def _protected():
            return "ok"

        assert _protected() == "ok"


def test_make_oauth_session_uses_expected_params(monkeypatch):
    captured = {}

    class _FakeOAuth2Session:
        def __init__(self, client_id, redirect_uri=None, state=None):
            captured["client_id"] = client_id
            captured["redirect_uri"] = redirect_uri
            captured["state"] = state
            self.headers = {}

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth2Session)
    sess = app_module._make_oauth_session(state="abc")

    assert captured["client_id"] == app_module.OAUTH_CLIENT_ID
    assert captured["redirect_uri"] == app_module.OAUTH_REDIRECT_URI
    assert captured["state"] == "abc"
    assert sess.headers["User-Agent"] == "WikiVisage/1.0 (https://github.com/DiFronzo/WikiVisage)"


def test_refresh_access_token_returns_same_user_when_not_expiring(monkeypatch):
    user = {
        "id": 1,
        "wiki_username": "tester",
        "refresh_token": "refresh",
        "access_token": "access",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=1),
    }

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("OAuth2Session should not be constructed for still-valid tokens")

    monkeypatch.setattr(app_module, "OAuth2Session", _forbidden)
    assert app_module._refresh_access_token(user) is user


def test_refresh_access_token_handles_string_naive_datetime(monkeypatch):
    user = {
        "id": 1,
        "wiki_username": "tester",
        "refresh_token": "refresh-old",
        "access_token": "access-old",
        "token_expires_at": "2000-01-01 00:00:00",
    }
    captured = {}

    class _FakeOAuth:
        def __init__(self, client_id):
            captured["client_id"] = client_id

        def refresh_token(self, token_url, refresh_token, client_id, client_secret):
            captured["token_url"] = token_url
            captured["refresh_token"] = refresh_token
            captured["refresh_client_id"] = client_id
            captured["client_secret"] = client_secret
            return {
                "access_token": "access-new",
                "refresh_token": "refresh-new",
                "expires_in": 1234,
            }

    db_calls = []

    def _execute(sql, params, fetch=False):
        db_calls.append((sql, params, fetch))
        return 1

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth)
    monkeypatch.setattr(app_module, "execute_query", _execute)

    updated = app_module._refresh_access_token(user)

    assert updated is user
    assert user["access_token"] == "access-new"
    assert user["refresh_token"] == "refresh-new"
    assert isinstance(user["token_expires_at"], datetime)
    assert db_calls and db_calls[0][2] is False
    assert captured["refresh_token"] == "refresh-old"


def test_refresh_access_token_uses_old_refresh_token_if_not_returned(monkeypatch):
    user = {
        "id": 1,
        "wiki_username": "tester",
        "refresh_token": "refresh-old",
        "access_token": "access-old",
        "token_expires_at": datetime.now(UTC) - timedelta(seconds=1),
    }

    class _FakeOAuth:
        def __init__(self, client_id):
            self.client_id = client_id

        def refresh_token(self, *_args, **_kwargs):
            return {"access_token": "access-new", "expires_in": 1000}

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth)
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_k: 1)

    updated = app_module._refresh_access_token(user)
    assert updated is not None
    assert updated["refresh_token"] == "refresh-old"


def test_refresh_access_token_returns_none_on_failure(monkeypatch):
    user = {
        "id": 1,
        "wiki_username": "tester",
        "refresh_token": "refresh",
        "access_token": "access",
        "token_expires_at": datetime.now(UTC) - timedelta(seconds=1),
    }

    class _FakeOAuth:
        def __init__(self, client_id):
            self.client_id = client_id

        def refresh_token(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth)
    assert app_module._refresh_access_token(user) is None


def test_get_valid_token_returns_none_without_user():
    with flask_app.test_request_context("/"):
        g.user = None
        assert app_module._get_valid_token() is None


def test_get_valid_token_clears_session_when_refresh_fails(monkeypatch):
    with flask_app.test_request_context("/"):
        g.user = {"access_token": "old"}
        session["user_id"] = 1
        monkeypatch.setattr(app_module, "_refresh_access_token", lambda _user: None)

        assert app_module._get_valid_token() is None
        assert "user_id" not in session


def test_get_valid_token_updates_g_user_and_returns_token(monkeypatch):
    with flask_app.test_request_context("/"):
        g.user = {"access_token": "old"}
        refreshed = {"access_token": "new"}
        monkeypatch.setattr(app_module, "_refresh_access_token", lambda _user: refreshed)

        token = app_module._get_valid_token()
        assert token == "new"
        assert g.user is refreshed


def test_csrf_token_generates_new_token():
    with flask_app.test_request_context("/"):
        token = app_module._csrf_token()
        assert isinstance(token, str)
        assert len(token) == 64
        assert session["csrf_token"] == token


def test_csrf_token_returns_existing_token():
    with flask_app.test_request_context("/"):
        session["csrf_token"] = "existing"
        assert app_module._csrf_token() == "existing"


def test_set_language_valid_lang_sets_cookie(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    resp = client.get("/set-language/es", headers={"Referer": "http://localhost/dashboard"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")
    assert "locale=es" in resp.headers.get("Set-Cookie", "")


def test_set_language_invalid_lang_falls_back_to_en(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    resp = client.get("/set-language/zz", headers={"Referer": "http://localhost/dashboard"})
    assert resp.status_code == 302
    assert "locale=en" in resp.headers.get("Set-Cookie", "")


def test_set_language_nocookie_deletes_cookie(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    resp = client.get("/set-language/es?nocookie=1", headers={"Referer": "http://localhost/dashboard"})
    assert resp.status_code == 302
    set_cookie = resp.headers.get("Set-Cookie", "")
    assert "locale=" in set_cookie
    assert "Expires=" in set_cookie


def test_set_language_cross_host_referrer_redirects_to_index(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    resp = client.get("/set-language/es", headers={"Referer": "https://evil.example/path"})
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/")


def test_get_locale_prefers_cookie():
    with flask_app.test_request_context("/", headers={"Cookie": "locale=nb", "Accept-Language": "es"}):
        assert app_module.get_locale() == "nb"


def test_get_locale_falls_back_to_accept_language():
    with flask_app.test_request_context("/", headers={"Accept-Language": "es"}):
        assert app_module.get_locale() == "es"


def test_get_locale_defaults_to_en_when_no_match():
    with flask_app.test_request_context("/", headers={"Accept-Language": "zz-ZZ"}):
        assert app_module.get_locale() == "en"


flask_app.config["TESTING"] = True
flask_app.config["RATELIMIT_ENABLED"] = False


class _MockResponse:
    def __init__(self, json_data=None, should_raise=False):
        self._json_data = {} if json_data is None else json_data
        self._should_raise = should_raise

    def raise_for_status(self):
        if self._should_raise:
            raise RuntimeError("http error")

    def json(self):
        return self._json_data


def _make_authenticated_client(monkeypatch):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    client = flask_app.test_client()
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _set_oauth_state(client, state="state-123", login_next=""):
    with client.session_transaction() as sess:
        sess["oauth_state"] = state
        if login_next:
            sess["login_next"] = login_next


def _get_flashes(client):
    with client.session_transaction() as sess:
        return list(sess.get("_flashes", []))


def _reset_rate_limit():
    app_module.limiter.reset()


def test_propertymock_is_used():
    class _Holder:
        @property
        def value(self):
            return "old"

    obj = _Holder()
    with patch.object(_Holder, "value", new_callable=PropertyMock, return_value="new"):
        assert obj.value == "new"


def test_is_human_entity_true_when_p31_contains_q5(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse(
            {
                "claims": {
                    "P31": [
                        {"mainsnak": {"datavalue": {"value": {"id": "Q95074"}}}},
                        {"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}},
                    ]
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._is_human_entity("Q42") is True


def test_is_human_entity_false_when_no_p31_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {}}))
    assert app_module._is_human_entity("Q1") is False


def test_is_human_entity_false_when_p31_without_q5(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q215627"}}}}]}}
        ),
    )
    assert app_module._is_human_entity("Q2") is False


def test_is_human_entity_false_on_missing_nested_keys(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P31": [{}]}}))
    assert app_module._is_human_entity("Q3") is False


def test_is_human_entity_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._is_human_entity("Q4") is False


def test_is_human_entity_false_on_request_exception(monkeypatch):
    def mock_get(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._is_human_entity("Q5") is False


def test_is_human_entity_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P31": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._is_human_entity("Q123")

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "Q123"
    assert captured["params"]["property"] == "P31"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_category_exists_true_when_page_exists(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"title": "Category:People"}}}}),
    )
    assert app_module._commons_category_exists("People") is True


def test_commons_category_exists_false_when_only_minus_one_page(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"-1": {"missing": ""}}}}),
    )
    assert app_module._commons_category_exists("Missing") is False


def test_commons_category_exists_true_with_empty_pages_dict(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"query": {"pages": {}}}))
    assert app_module._commons_category_exists("EdgeCase") is True


def test_commons_category_exists_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._commons_category_exists("People") is False


def test_commons_category_exists_false_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._commons_category_exists("People") is False


def test_commons_category_exists_calls_commons_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"query": {"pages": {"123": {}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._commons_category_exists("Human faces")

    assert captured["url"] == app_module.COMMONS_API_URL
    assert captured["params"]["action"] == "query"
    assert captured["params"]["titles"] == "Category:Human faces"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_category_has_files_true_when_files_present(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 5, "subcats": 0}}}}}),
    )
    assert app_module._commons_category_has_files("People") is True


def test_commons_category_has_files_true_when_only_subcats(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 0, "subcats": 3}}}}}),
    )
    assert app_module._commons_category_has_files("People") is True


def test_commons_category_has_files_false_when_empty(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 0, "subcats": 0}}}}}),
    )
    assert app_module._commons_category_has_files("EmptyCat") is False


def test_commons_category_has_files_false_when_no_categoryinfo(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {}}}}),
    )
    assert app_module._commons_category_has_files("NoCatInfo") is False


def test_commons_category_has_files_false_on_missing_page(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"-1": {"missing": ""}}}}),
    )
    assert app_module._commons_category_has_files("Missing") is False


def test_commons_category_has_files_false_on_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._commons_category_has_files("People") is False


def test_fetch_p18_thumb_url_returns_none_when_no_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P18": []}}))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_when_filename_missing(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"claims": {"P18": [{"mainsnak": {"datavalue": {"value": None}}}]}}),
    )
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_builds_commons_thumb_from_first_claim(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {
                "claims": {
                    "P18": [
                        {"mainsnak": {"datavalue": {"value": "First.jpg"}}},
                        {"mainsnak": {"datavalue": {"value": "Second.jpg"}}},
                    ]
                }
            }
        ),
    )
    monkeypatch.setattr(app_module, "commons_thumb_url", lambda file_title, width: f"u:{file_title}:{width}")
    assert app_module._fetch_p18_thumb_url("Q42", width=500) == "u:First.jpg:500"


def test_fetch_p18_thumb_url_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P18": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._fetch_p18_thumb_url("Q99", width=120)

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "Q99"
    assert captured["params"]["property"] == "P18"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_fetch_wikidata_label_prefers_current_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "nb")
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {"entities": {"Q42": {"labels": {"nb": {"value": "Douglas Adams NB"}, "en": {"value": "Douglas Adams"}}}}}
        ),
    )
    assert app_module._fetch_wikidata_label("Q42") == "Douglas Adams NB"


def test_fetch_wikidata_label_falls_back_to_english(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "fr")
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"entities": {"Q42": {"labels": {"en": {"value": "Douglas Adams"}}}}}),
    )
    assert app_module._fetch_wikidata_label("Q42") == "Douglas Adams"


def test_fetch_wikidata_label_returns_none_when_no_labels(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "es")
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"entities": {"Q42": {"labels": {}}}})
    )
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_returns_none_when_entity_missing(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"entities": {}}))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_uses_en_when_locale_is_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(app_module, "get_locale", lambda: None)

    def mock_get(url, **kwargs):
        captured.update(kwargs)
        return _MockResponse({"entities": {"Q1": {"labels": {"en": {"value": "Label"}}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    result = app_module._fetch_wikidata_label("Q1")

    assert result == "Label"
    assert captured["params"]["languages"] == "en"


def test_fetch_wikidata_label_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_returns_none_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}
    monkeypatch.setattr(app_module, "get_locale", lambda: "nb")

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"entities": {"Q99": {"labels": {"nb": {"value": "x"}}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._fetch_wikidata_label("Q99")

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetentities"
    assert captured["params"]["ids"] == "Q99"
    assert captured["params"]["props"] == "labels"
    assert captured["params"]["languages"] == "nb|en"
    assert captured["params"]["languagefallback"] == "1"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_thumb_url_regular_jpg():
    url = app_module.commons_thumb_url("File:Example.jpg", 330)
    assert "/330px-Example.jpg" in url


def test_commons_thumb_url_webm_video_uses_double_dash():
    url = app_module.commons_thumb_url("File:Movie.webm", 330)
    assert url.endswith("/330px--Movie.webm.jpg")


def test_commons_thumb_url_ogv_video_uses_double_dash():
    url = app_module.commons_thumb_url("File:Clip.ogv", 330)
    assert url.endswith("/330px--Clip.ogv.jpg")


def test_commons_thumb_url_tif_converts_to_jpg():
    url = app_module.commons_thumb_url("File:Scan.tif", 330)
    assert url.endswith("/330px-Scan.tif.jpg")


def test_commons_thumb_url_tiff_converts_to_jpg():
    url = app_module.commons_thumb_url("File:Scan.tiff", 330)
    assert url.endswith("/330px-Scan.tiff.jpg")


def test_commons_thumb_url_svg_converts_to_png():
    url = app_module.commons_thumb_url("File:Logo.svg", 330)
    assert url.endswith("/330px-Logo.svg.png")


def test_commons_thumb_url_handles_uppercase_extension():
    url = app_module.commons_thumb_url("File:Logo.SVG", 330)
    assert url.endswith("/330px-Logo.SVG.png")


def test_commons_thumb_url_snaps_width_up_to_next_step():
    url = app_module.commons_thumb_url("File:X.jpg", 100)
    assert "/120px-" in url


def test_commons_thumb_url_caps_width_at_max_step():
    url = app_module.commons_thumb_url("File:X.jpg", 9999)
    assert "/3840px-" in url


def test_commons_thumb_url_normalizes_file_prefix_and_spaces():
    url = app_module.commons_thumb_url("File:My Photo.jpg", 330)
    assert "My_Photo.jpg" in url


def test_login_redirects_to_dashboard_when_already_authenticated(monkeypatch):
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)

    response = client.get("/login")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_login_starts_oauth_flow_and_stores_state_and_next(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    oauth = MagicMock()
    oauth.authorization_url.return_value = ("https://oauth.example/authorize", "state-abc")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    response = client.get("/login?next=/project/12")

    assert response.status_code == 302
    assert response.headers["Location"] == "https://oauth.example/authorize"
    with client.session_transaction() as sess:
        assert sess["oauth_state"] == "state-abc"
        assert sess["login_next"] == "/project/12"


def test_login_sets_empty_next_when_missing(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    oauth = MagicMock()
    oauth.authorization_url.return_value = ("https://oauth.example/authorize", "state-xyz")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    response = client.get("/login")

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert sess["login_next"] == ""


def test_oauth_callback_no_stored_state_redirects_to_index():
    _reset_rate_limit()
    client = flask_app.test_client()

    response = client.get("/auth/callback")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Invalid OAuth state" in msg for _cat, msg in flashes)


def test_oauth_callback_token_fetch_exception_redirects_to_index(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.side_effect = RuntimeError("token fail")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Authentication failed" in msg for _cat, msg in flashes)


def test_oauth_callback_profile_fetch_exception_redirects_to_index(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Could not retrieve your profile" in msg for _cat, msg in flashes)


def test_oauth_callback_no_wiki_user_id_redirects_to_index(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"username": "tester"}))

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Invalid profile data received" in msg for _cat, msg in flashes)


def test_oauth_callback_existing_user_updates_and_sets_session(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)
    with client.session_transaction() as sess:
        sess["csrf_token"] = "keep-me"

    oauth = MagicMock()
    oauth.fetch_token.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
    }
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 123, "username": "tester"})
    )

    calls = []

    def fake_execute_query(sql, params=None, fetch=True):
        calls.append((sql, params, fetch))
        if sql.startswith("SELECT id FROM users WHERE wiki_user_id"):
            return [{"id": 77}]
        return 1

    monkeypatch.setattr(app_module, "execute_query", fake_execute_query)

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert any(sql.startswith("UPDATE users SET") for sql, _params, _fetch in calls)
    with client.session_transaction() as sess:
        assert sess["user_id"] == 77
        assert sess["csrf_token"] == "keep-me"


def test_oauth_callback_new_user_inserts_and_reads_back_id(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 555, "username": "tester"})
    )

    calls = []

    def fake_execute_query(sql, params=None, fetch=True):
        calls.append((sql, params, fetch))
        if sql.startswith("SELECT id FROM users WHERE wiki_user_id") and len(calls) == 1:
            return []
        if sql.startswith("SELECT id FROM users WHERE wiki_user_id") and len(calls) == 3:
            return [{"id": 88}]
        return 1

    monkeypatch.setattr(app_module, "execute_query", fake_execute_query)

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert any(sql.startswith("INSERT INTO users") for sql, _params, _fetch in calls)
    with client.session_transaction() as sess:
        assert sess["user_id"] == 88


def test_oauth_callback_db_error_during_upsert_redirects_to_index(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 777, "username": "tester"})
    )

    def _raise_db(*_a, **_kw):
        raise app_module.DatabaseError("db fail")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Database error" in msg for _cat, msg in flashes)


def test_oauth_callback_uses_safe_login_next_redirect(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client, login_next="/project/42")

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 42, "username": "tester"}))

    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"id": 42}])

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/42")


def test_oauth_callback_unsafe_login_next_falls_back_to_dashboard(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client, login_next="https://evil.example/phish")

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 5, "username": "tester"}))

    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"id": 5}])

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_oauth_callback_profile_call_uses_bearer_token_and_timeout(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "abc-token", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"sub": 1, "username": "tester"})

    monkeypatch.setattr(app_module.requests, "get", mock_get)

    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"id": 1}])

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert captured["url"] == app_module.OAUTH_PROFILE_URL
    assert captured["headers"]["Authorization"] == "Bearer abc-token"
    assert captured["timeout"] == 10


def test_logout_invalid_csrf_returns_403(monkeypatch):
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)
    with client.session_transaction() as sess:
        sess["csrf_token"] = "expected"

    response = client.post("/logout", data={"csrf_token": "wrong"})

    assert response.status_code == 403


def test_logout_valid_csrf_clears_session_and_redirects(monkeypatch):
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)
    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"
        sess["some_key"] = "value"

    response = client.post("/logout", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("logged out" in msg for _cat, msg in flashes)
    with client.session_transaction() as sess:
        assert "user_id" not in sess
        assert "some_key" not in sess


def _fake_user_chunk3(username="tester"):
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": username,
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _make_authenticated_client_chunk3(monkeypatch, execute_query_fn=None):
    fake_user = _fake_user_chunk3()
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    monkeypatch.setattr(app_module.limiter, "enabled", False)
    client = flask_app.test_client()

    if execute_query_fn is None:

        def _default_execute_query(*_a, **_kw):
            return [fake_user]

        execute_query_fn = _default_execute_query

    monkeypatch.setattr(app_module, "execute_query", execute_query_fn)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _user_aware_execute(fake_user, route_handler):
    def _execute_query(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "SHOW TABLES LIKE 'worker_heartbeat'" in sql:
            return [{"1": 1}]
        return route_handler(sql, params, fetch)

    return _execute_query


def _capture_render(monkeypatch):
    captured = {}

    def _render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return f"rendered:{template}"

    monkeypatch.setattr(app_module, "render_template", _render)
    return captured


def _set_csrf_chunk3(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes_chunk3(client):
    with client.session_transaction() as sess:
        return sess.get("_flashes", [])


class _FakeResponse_chunk3:
    def __init__(self, payload=None, raise_exc=None):
        self.payload = {} if payload is None else payload
        self.raise_exc = raise_exc

    def raise_for_status(self):
        if self.raise_exc:
            raise self.raise_exc

    def json(self):
        return self.payload


def test_index_not_logged_in_renders_index_template(monkeypatch):
    flask_app.config["TESTING"] = True
    captured = _capture_render(monkeypatch)
    client = flask_app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "rendered:index.html"
    assert captured["template"] == "index.html"


def test_index_logged_in_redirects_to_dashboard(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_dashboard_defaults_page_and_totals(monkeypatch):
    captured = _capture_render(monkeypatch)
    calls = []
    fake_user = _fake_user_chunk3()

    def _handler(sql, params, fetch):
        calls.append((sql, params, fetch))
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["template"] == "dashboard.html"
    assert captured["context"]["page"] == 1
    assert captured["context"]["total_pages"] == 1
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["projects"] == []
    assert any("COUNT(DISTINCT p.id) AS cnt" in sql for sql, _params, _fetch in calls)
    assert any("SELECT DISTINCT p.*" in sql for sql, _params, _fetch in calls)


def test_dashboard_invalid_page_falls_back_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=abc")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_negative_page_clamped_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=-5")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_page_clamped_to_total_pages(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    select_params = {}

    def _handler(sql, params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 30}]
        if "SELECT DISTINCT p.*" in sql:
            select_params["params"] = params
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=99")

    assert response.status_code == 200
    assert captured["context"]["page"] == 2
    assert captured["context"]["total_pages"] == 2
    assert select_params["params"][2] == 25


def test_dashboard_count_database_error_sets_total_zero(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            raise app_module.DatabaseError("count failed")
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=3")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["page"] == 1


def test_dashboard_projects_database_error_flashes_and_uses_empty(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 10}]
        if "SELECT DISTINCT p.*" in sql:
            raise app_module.DatabaseError("projects failed")
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["projects"] == []
    assert any("Failed to load projects." in msg for _cat, msg in _flashes_chunk3(client))


def test_dashboard_count_row_empty_tuple(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return ()
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0


def test_dashboard_lazily_populates_missing_p18_and_updates_db(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    updates = []

    projects = [
        {"id": 101, "wikidata_qid": "Q42", "p18_thumb_url": None},
        {"id": 102, "wikidata_qid": "Q1", "p18_thumb_url": "existing"},
        {"id": 103, "wikidata_qid": None, "p18_thumb_url": None},
    ]

    def _handler(sql, params, fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 3}]
        if "SELECT DISTINCT p.*" in sql:
            return projects
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((params, fetch))
            return 1
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"thumb-{qid}")
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert projects[0]["p18_thumb_url"] == "thumb-Q42"
    assert len(updates) == 1
    assert updates[0][0] == ("thumb-Q42", 101)
    assert updates[0][1] is False


def test_dashboard_missing_thumb_but_fetch_returns_none_no_update(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    update_called = {"called": False}

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            update_called["called"] = True
            return 1
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert update_called["called"] is False
    assert captured["template"] == "dashboard.html"


def test_dashboard_p18_update_database_error_is_non_critical(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("update failed")
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200


def test_dashboard_projects_tuple_skips_lazy_population(monkeypatch):
    fake_user = _fake_user_chunk3()
    fetch_calls = {"count": 0}

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return ()
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: fetch_calls.update(count=1))
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert fetch_calls["count"] == 0


def test_api_category_info_missing_category_returns_400(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/api/category-info")

    assert response.status_code == 400
    assert response.get_json() == {"error": "missing category"}


@pytest.mark.parametrize(
    "category",
    [
        "Bad|Name",
        "Bad\nName",
        "x" * 201,
    ],
)
def test_api_category_info_invalid_category_name_returns_400(monkeypatch, category):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/api/category-info", query_string={"category": category})

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid category name"}


def test_api_category_info_root_category_not_found_returns_404(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse_chunk3({"query": {"pages": {"-1": {"missing": True}}}}),
    )

    response = client.get("/api/category-info?category=People")

    assert response.status_code == 404
    assert response.get_json() == {"error": "not_found"}


def test_api_category_info_root_no_subcats_returns_direct_counts(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(*_args, **_kwargs):
        return _FakeResponse_chunk3(
            {
                "query": {
                    "pages": {
                        "123": {
                            "title": "Category:People",
                            "categoryinfo": {"files": 7, "subcats": 0},
                        }
                    }
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=People")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 7
    assert body["subcats"] == 0
    assert body["categories_visited"] == 1
    assert not body["approximate"]


def test_api_category_info_bfs_traversal_sums_subcategories(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 3, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 4, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:B"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:B":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "3": {
                                "title": "Category:B",
                                "categoryinfo": {"files": 6, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 13
    assert body["subcats"] == 2
    assert body["categories_visited"] == 3
    assert not body["approximate"]


def test_api_category_info_sets_approximate_on_continue_from_root_subcats(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 2, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers":
            return _FakeResponse_chunk3(
                {
                    "query": {"categorymembers": [{"title": "Category:A"}]},
                    "continue": {"cmcontinue": "next"},
                }
            )
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 3, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_api_category_info_hits_max_categories_marks_approximate(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    many = [{"title": f"Category:C{i}"} for i in range(60)]

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 1, "subcats": len(many)},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": many}})
        if params.get("prop") == "categoryinfo" and "Category:C" in params.get("titles", ""):
            pages = {}
            for i, title in enumerate(params["titles"].split("|"), start=100):
                pages[str(i)] = {"title": title, "categoryinfo": {"files": 1, "subcats": 0}}
            return _FakeResponse_chunk3({"query": {"pages": pages}})
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["approximate"] is True
    assert body["categories_visited"] == 50


def test_api_category_info_timeout_marks_approximate(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 5, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        raise AssertionError(params)

    calls = {"n": 0}

    def _monotonic():
        calls["n"] += 1
        if calls["n"] == 1:
            return 0
        return 9

    monkeypatch.setattr(app_module.requests, "get", _get)
    monkeypatch.setattr(app_module.time, "monotonic", _monotonic)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_api_category_info_request_exception_returns_502(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *a, **kw: (_ for _ in ()).throw(requests.RequestException("boom"))
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_api_category_info_http_error_from_raise_for_status_returns_502(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse_chunk3(raise_exc=requests.HTTPError("bad status")),
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_project_new_get_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/project/new")

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"


def test_project_new_post_invalid_csrf_returns_400(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client, "session-token")

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "bad-token",
            "wikidata_qid": "Q42",
            "commons_category": "People",
        },
    )

    assert response.status_code == 400


def test_project_new_post_missing_qid_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "commons_category": "People"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_qid_without_q_prefix_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_missing_category_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Commons category is required." in msg for _cat, msg in _flashes_chunk3(client))


@pytest.mark.parametrize("value", ["abc", "0.09", "1.1"])
def test_project_new_post_invalid_distance_threshold(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "distance_threshold": value,
        },
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Distance threshold" in msg for msg in messages)


@pytest.mark.parametrize("value", ["abc", "0", "-5"])
def test_project_new_post_invalid_min_confirmed(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "min_confirmed": value,
        },
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Minimum confirmed" in msg for msg in messages)


def test_project_new_post_multiple_validation_errors_are_all_flashed(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "", "commons_category": "", "distance_threshold": "oops"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Wikidata Q-ID must start" in msg for msg in messages)
    assert any("Commons category is required." in msg for msg in messages)
    assert any("Distance threshold must be a number." in msg for msg in messages)


def test_project_new_validation_short_circuits_remote_checks_on_basic_errors(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    is_human_called = {"called": False}
    category_called = {"called": False}
    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: is_human_called.update(called=True))
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _c: category_called.update(called=True))

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "bad", "commons_category": ""},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert is_human_called["called"] is False
    assert category_called["called"] is False


def test_project_new_post_non_human_qid_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: False)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q123", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("is not an instance of human" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_missing_commons_category_on_wikimedia_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: False)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "NotReal"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("does not exist" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_empty_commons_category_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: False)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "EmptyCat"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("exists but contains no files" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_duplicate_project_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return [{"id": 99}]
        if "INSERT INTO projects" in sql:
            raise AssertionError("Should not insert duplicates")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People", "label": "My Label"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("already exists" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_duplicate_check_db_error_continues_to_create(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            raise app_module.DatabaseError("duplicate check failed")
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb-q42")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Douglas Adams")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert len(insert_calls) == 1


def test_project_new_join_while_banned_creates_own_project(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql and "user_id = %s" in sql:
            return ()
        if "u.wiki_username" in sql:
            return [{"id": 42, "label": "Existing Project", "wiki_username": "OtherUser"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "banned"}]
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT" in sql:
            insert_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert len(insert_calls) == 1


def test_project_new_successful_creation_with_label_fetch_and_p18(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return ()
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "https://thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Douglas Adams"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": " q42 ", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert label_calls["n"] == 1
    assert insert_calls[0][0][1] == "Q42"
    assert insert_calls[0][0][3] == "Douglas Adams"
    assert insert_calls[0][0][6] == "https://thumb"
    assert insert_calls[0][1] is False
    assert any("Project created successfully!" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_successful_creation_with_user_label_skips_wikidata_label_fetch(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Should Not Be Used"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "label": "Custom Label",
        },
    )

    assert response.status_code == 302
    assert label_calls["n"] == 0
    assert insert_calls[0][0][3] == "Custom Label"


def test_project_new_successful_creation_with_missing_p18_thumb(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert insert_calls[0][0][6] is None


def test_project_new_wake_file_oserror_is_non_critical(monkeypatch):
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    def _raise_open(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise_open)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_project_new_db_error_on_insert_renders_form_with_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise app_module.DatabaseError("insert failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project. Please try again." in msg for _cat, msg in _flashes_chunk3(client))


def _auth_client_chunk4(monkeypatch, execute_query_impl=None):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    flask_app.config["TESTING"] = True
    monkeypatch.setattr(app_module.limiter, "_check_request_limit", lambda *a, **k: None)
    client = flask_app.test_client()

    if execute_query_impl is None:

        def _default_execute_query_impl(sql, params=None, fetch=True):
            del params, fetch
            if "FROM users WHERE id = %s" in sql:
                return [fake_user]
            return ()

        execute_query_impl = _default_execute_query_impl

    monkeypatch.setattr(app_module, "execute_query", execute_query_impl)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _set_csrf_chunk4(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _capture_render_template_chunk4(monkeypatch):
    captured = {}

    def fake_render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return app_module.jsonify({"template": template})

    monkeypatch.setattr(app_module, "render_template", fake_render)
    return captured


def test_project_detail_db_error_loading_project(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        raise app_module.DatabaseError("boom")

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 500


def test_project_detail_project_not_found(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 404


def test_project_detail_full_success_with_lazy_p18_update(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    updates = []

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "active"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((sql, params, fetch))
            return 1
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 10, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 5}]
        if "status = 'pending'" in sql:
            return [{"cnt": 7}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 4}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"https://thumb/{qid}.jpg")
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb/Q42.jpg"
    assert captured["context"]["stats"]["total_faces"] == 10
    assert captured["context"]["gallery_total"] == 5
    assert captured["context"]["pending_images"] == 7
    assert captured["context"]["inference_eligible"] == 4
    assert len(updates) == 1


def test_project_detail_lazy_p18_not_updated_when_fetch_empty(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    seen_update = {"called": False}

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "completed"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            seen_update["called"] = True
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] is None
    assert seen_update["called"] is False


def test_project_detail_lazy_p18_update_db_error_ignored(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "completed"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("nope")
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "https://thumb.jpg")
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb.jpg"


def test_project_detail_face_stats_db_error_returns_empty_stats(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            raise app_module.DatabaseError("stats fail")
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["stats"] == {}


def test_project_detail_gallery_count_db_error_returns_zero(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 3}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            raise app_module.DatabaseError("gallery fail")
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["gallery_total"] == 0


def test_project_detail_pending_images_only_for_active(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    pending_calls = {"count": 0}

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "status = 'pending'" in sql:
            pending_calls["count"] += 1
            return [{"cnt": 99}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert pending_calls["count"] == 0
    assert captured["context"]["pending_images"] == 0


def test_project_detail_pending_images_db_error_defaults_zero(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "status = 'pending'" in sql:
            raise app_module.DatabaseError("pending fail")
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 3}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["pending_images"] == 0


def test_project_detail_inference_eligible_db_error_defaults_zero(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "status = 'pending'" in sql:
            return [{"cnt": 2}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            raise app_module.DatabaseError("eligible fail")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["inference_eligible"] == 0


def test_classify_project_not_found(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 404


def test_classify_project_load_db_error(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        raise app_module.DatabaseError("oops")

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 500


def test_classify_skip_image_id_normal_mode_adds_session(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 15,
                    "file_title": "File:A.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 1, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=77")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_1"] == [77]
    assert captured["context"]["skipped_count"] == 1


def test_classify_skip_image_id_review_mode_adds_review_session(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 22,
                    "file_title": "File:B.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 9, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": 0.2}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=88&skip_reviewing=1")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_review_1"] == [88]
    assert captured["context"]["skipped_review_count"] == 1


def test_classify_skip_image_id_invalid_ignored(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=not-an-int")
    assert response.status_code == 200
    assert captured["context"]["skipped_count"] == 0


def test_classify_forced_image_review_mode_when_no_unclassified(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 44,
                    "file_title": "File:C.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": 0}]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 100, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=44")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"][0]["face_id"] == 100


def test_classify_forced_image_normal_mode_with_unclassified(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 45,
                    "file_title": "File:D.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": 2}]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 101, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=45")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is False
    assert captured["context"]["faces"][0]["face_id"] == 101


def test_classify_skipped_ids_path_uses_not_in(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[-1])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "i.id NOT IN" in sql and "f.is_target IS NULL" in sql:
            assert params == (1, 99)
            return [
                {
                    "image_id": 2,
                    "file_title": "File:E.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 3,
                    "file_title": "File:F.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 5, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_1"] = [99]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 3


def test_classify_normal_no_skips_uses_random_choice(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[1])

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 20,
                    "file_title": "File:G.jpg",
                    "commons_page_id": 20,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 21,
                    "file_title": "File:H.jpg",
                    "commons_page_id": 21,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 7, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 21


def test_classify_fallback_to_model_review_without_skipped_review(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 55,
                    "file_title": "File:I.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 8, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["image"]["image_id"] == 55


def test_classify_fallback_with_skipped_review_ids(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[-1])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "i.id NOT IN" in sql and "f.classified_by = 'model'" in sql:
            assert params == (1, 5, 6)
            return [
                {
                    "image_id": 70,
                    "file_title": "File:J.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 71,
                    "file_title": "File:K.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 81, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_review_1"] = [5, 6]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 71


def test_classify_image_loading_db_error_results_no_image(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            raise app_module.DatabaseError("load fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"] is None
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_normal(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 1,
                    "file_title": "File:X.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            raise app_module.DatabaseError("faces fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_review(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 1,
                    "file_title": "File:X.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            raise app_module.DatabaseError("faces fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"] == []


def test_classify_remaining_count_db_error(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            raise app_module.DatabaseError("count fail")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["remaining"] == 0
    assert captured["context"]["model_review_count"] == 0


def test_clear_skips_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/project/1/classify/clear-skips", data={"csrf_token": "wrong"})
    assert response.status_code == 403


def test_clear_skips_nonmember_returns_404(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post("/project/999/classify/clear-skips", data={"csrf_token": "testtoken"})
    assert response.status_code == 404


def test_clear_skips_clears_normal_and_review_keys(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, execute_query_impl=eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_1"] = [1, 2]
        sess["skipped_images_review_1"] = [3]

    response = client.post("/project/1/classify/clear-skips", data={"csrf_token": "testtoken"})
    assert response.status_code == 302
    assert response.location.endswith("/project/1/classify")
    with client.session_transaction() as sess:
        assert "skipped_images_1" not in sess
        assert "skipped_images_review_1" not in sess


def test_api_classify_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/api/classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


class TestChunk4ApiClassify:
    def test_api_classify_missing_fields(self, monkeypatch):
        client, _ = _auth_client_chunk4(monkeypatch)
        _set_csrf_chunk4(client)
        response = client.post(
            "/api/classify",
            data={"csrf_token": "testtoken", "project_id": "1", "image_id": "2"},
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == "Missing required fields"


def test_api_classify_invalid_field_values(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "bad", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid field values"


def test_api_classify_ownership_check_fail(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 404
    assert response.get_json()["error"] == "Image not found or access denied"


def test_api_classify_ownership_check_db_error(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            raise app_module.DatabaseError("db")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_classify_invalid_selected_face_id(monkeypatch):
    queries = []

    def eq(sql, params=None, fetch=True):
        queries.append((sql, params, fetch))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "abc", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid face ID"
    assert any("SELECT i.id, i.file_title, i.status FROM images i" in q[0] for q in queries)


def test_api_classify_none_normal_mode_sets_last_classify(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["manual_faces_22"] = [400]

    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    with client.session_transaction() as sess:
        assert sess["last_classify"]["action"] == "none"
        assert sess["last_classify"]["face_ids"] == [10, 11]
        assert sess["last_classify"]["manual_face_ids"] == [400]
        assert sess["last_classify"]["was_review"] is False
        assert "manual_faces_22" not in sess
    assert any("UPDATE faces SET is_target = 0" in sql for sql, _ in executed)


def test_api_classify_none_review_mode_updates_human_flags(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET classified_by = 'human'" in sql for sql, _ in executed)


def test_api_classify_none_bootstrapped_without_sibling_queues_removal(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET sdc_removal_pending = 1" in sql for sql, _ in executed)


def test_api_classify_none_bootstrapped_with_sibling_does_not_queue_removal(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 1}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert not any("UPDATE faces SET sdc_removal_pending = 1" in sql for sql, _ in executed)


def test_api_classify_target_normal_mode_updates_and_counter(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET is_target = 1" in sql for sql, _ in executed)
    assert any("id != %s AND is_target IS NULL" in sql for sql, _ in executed)
    assert any("UPDATE projects SET faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["action"] == "target"
        assert sess["last_classify"]["selected_face_id"] == 10
        assert sess["last_classify"]["was_review"] is False


def test_api_classify_target_review_mode_updates_other_faces_human(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "AND is_target = 0 AND classified_by = 'model'" in sql and "SET is_target = 1" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "10",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    assert any(
        "AND is_target = 0 AND classified_by = 'model'" in sql and "SET is_target = 1" in sql for sql, _ in executed
    )
    assert any("AND is_target = 0 AND classified_by = 'model'" in sql and "id != %s" in sql for sql, _ in executed)
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["was_review"] is True


def test_api_classify_target_review_mode_does_not_use_is_target_null(monkeypatch):
    """Regression: review-mode target UPDATE must NOT use is_target IS NULL (would match 0 rows)."""
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "SET is_target = 1" in sql:
                if "is_target IS NULL" in sql:
                    cursor.rowcount = 0
                elif "is_target = 0" in sql:
                    cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "10",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    target_sqls = [sql for sql, _ in executed if "SET is_target = 1" in sql]
    assert len(target_sqls) == 1
    assert "is_target IS NULL" not in target_sqls[0]
    assert "is_target = 0" in target_sqls[0]
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)


def test_api_classify_db_error_during_transaction(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("tx fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to save classification"


def test_api_undo_classify_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/api/undo-classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_undo_classify_no_last_classify(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nothing to undo"


def test_api_undo_classify_empty_face_ids_clears_session(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nothing to undo"
    with client.session_transaction() as sess:
        assert "last_classify" not in sess


def test_api_undo_classify_ownership_check_fail(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [10], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Image not found or access denied"


def test_api_undo_classify_ownership_check_db_error(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            raise app_module.DatabaseError("db")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [10], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_undo_classify_target_normal_decrements_counter(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "target",
            "face_ids": [10, 11],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("SET is_target = NULL" in sql for sql, _ in executed)
    assert any("faces_confirmed = GREATEST" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert "last_classify" not in sess


def test_api_undo_classify_none_review_restores_model(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "none",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": True,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("SET is_target = 0, classified_by = 'model'" in sql for sql, _ in executed)
    assert not any("faces_confirmed = GREATEST" in sql for sql, _ in executed)


def test_api_undo_classify_manual_face_deletion_and_exclusion(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "none",
            "face_ids": [10, 11, 12],
            "manual_face_ids": [11, 12],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    delete_sql = [sql for sql, _ in executed if sql.startswith("DELETE FROM faces")]
    assert delete_sql
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (10, 2)


def test_api_undo_classify_manual_face_review_action_decrements(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "manual_face",
            "face_ids": [20],
            "manual_face_ids": [20],
            "was_review": True,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("faces_confirmed = GREATEST" in sql for sql, _ in executed)


def test_api_undo_classify_db_error_during_transaction(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("undo fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "target",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to undo classification"
    with client.session_transaction() as sess:
        assert "last_classify" in sess


def test_api_undo_classify_success_response_contains_ids(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.return_value = None
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 9,
            "image_id": 77,
            "action": "none",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 9, "image_id": 77}


# --- Undo-after-skip tests ---


def test_classify_skip_sets_last_classify_with_skip_action(monkeypatch):
    """Skip sets session['last_classify'] with action='skip' so undo button appears."""
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 15,
                    "file_title": "File:A.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 1, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=77")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        last = sess.get("last_classify")
        assert last is not None
        assert last["action"] == "skip"
        assert last["project_id"] == 1
        assert last["image_id"] == 77
        assert last["was_review"] is False
    assert captured["context"]["has_undo"] is True


def test_classify_skip_review_sets_last_classify_with_was_review(monkeypatch):
    """Skip in review mode sets was_review=True in last_classify."""
    _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 22,
                    "file_title": "File:B.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 9, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": 0.2}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=88&skip_reviewing=1")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        last = sess.get("last_classify")
        assert last is not None
        assert last["action"] == "skip"
        assert last["was_review"] is True
        assert last["image_id"] == 88


def test_api_undo_skip_removes_from_skip_list(monkeypatch):
    """Undo of skip removes image from skip list and returns image_id for redirect."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 5, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_5"] = [10, 20, 30]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 5,
            "image_id": 20,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 5, "image_id": 20}
    with client.session_transaction() as sess:
        assert 20 not in sess["skipped_images_5"]
        assert sess["skipped_images_5"] == [10, 30]
        assert "last_classify" not in sess


def test_api_undo_skip_review_removes_from_review_skip_list(monkeypatch):
    """Undo of review-mode skip removes from the review skip list."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 3, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_review_3"] = [50, 60]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 3,
            "image_id": 60,
            "was_review": True,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 3, "image_id": 60}
    with client.session_transaction() as sess:
        assert sess["skipped_images_review_3"] == [50]
        assert "last_classify" not in sess


def test_api_undo_skip_idempotent_when_not_in_list(monkeypatch):
    """Undo of skip succeeds even if image_id is not in skip list (idempotent)."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 7, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_7"] = [100]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 7,
            "image_id": 999,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 7, "image_id": 999}
    with client.session_transaction() as sess:
        assert sess["skipped_images_7"] == [100]
        assert "last_classify" not in sess


def test_api_undo_skip_no_db_transaction_needed(monkeypatch):
    """Undo of skip does NOT call execute_transaction (session-only operation)."""
    tx_called = []

    def tx(_fn):
        tx_called.append(True)
        raise AssertionError("execute_transaction should not be called for skip undo")

    monkeypatch.setattr(app_module, "execute_transaction", tx)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 2, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_2"] = [42]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 2,
            "image_id": 42,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert len(tx_called) == 0


def test_api_undo_skip_returns_404_when_not_owner_or_member(monkeypatch):
    """Undo of skip returns 404 if user is not the project owner or a member."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_99"] = [10]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 99,
            "image_id": 10,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 404
    with client.session_transaction() as sess:
        assert sess["skipped_images_99"] == [10]
        assert sess.get("last_classify") is not None


def test_api_undo_skip_works_for_project_member(monkeypatch):
    """Undo of skip succeeds when user is a project member (not owner)."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 2,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 5, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["user_id"] = 2
        sess["skipped_images_5"] = [10, 20]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 5,
            "image_id": 20,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 5, "image_id": 20}
    with client.session_transaction() as sess:
        assert sess["skipped_images_5"] == [10]
        assert "last_classify" not in sess


def _fake_user_chunk5():
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _authed_client(monkeypatch, route_execute_query=None, csrf_token=None):
    if csrf_token is None:
        csrf_token = "testtoken"
    fake_user = _fake_user_chunk5()
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)
    client = flask_app.test_client()

    def _query(sql, *args, **kwargs):
        del args, kwargs
        if "FROM users WHERE id" in sql:
            return [fake_user]
        if route_execute_query is not None:
            return route_execute_query(sql)
        return [fake_user]

    monkeypatch.setattr(app_module, "execute_query", _query)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        if csrf_token is not None:
            sess["csrf_token"] = csrf_token

    return client, fake_user


def _manual_face_form(**overrides):
    payload = {
        "csrf_token": "testtoken",
        "project_id": "1",
        "image_id": "10",
        "bbox_top": "10",
        "bbox_right": "30",
        "bbox_bottom": "40",
        "bbox_left": "10",
    }
    payload.update(overrides)
    return payload


def _update_bbox_form(**overrides):
    payload = {
        "csrf_token": "testtoken",
        "face_id": "1",
        "bbox_top": "10",
        "bbox_right": "30",
        "bbox_bottom": "40",
        "bbox_left": "10",
    }
    payload.update(overrides)
    return payload


def _build_face_recognition_mock(encodings=None, face_encodings_side_effect=None):
    mock_fr = MagicMock()
    mock_fr.load_image_file.return_value = MagicMock()
    if face_encodings_side_effect is not None:
        mock_fr.face_encodings.side_effect = face_encodings_side_effect
    else:
        mock_fr.face_encodings.return_value = [] if encodings is None else encodings
    return mock_fr


def _default_reclassify_face_row(**overrides):
    row = {
        "id": 1,
        "image_id": 10,
        "old_is_target": 0,
        "sdc_written": 0,
        "classified_by": "model",
        "commons_page_id": 555,
        "bootstrapped": 0,
        "project_id": 1,
        "wikidata_qid": "Q42",
    }
    row.update(overrides)
    return row


def _reclassify_query_router(
    face_row=None, ownership_exists=True, sibling_match=0, ownership_error=False, sibling_error=False
):
    local_face_row = _default_reclassify_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            if ownership_error:
                raise app_module.DatabaseError("ownership failure")
            return [local_face_row] if ownership_exists else []
        if "has_sibling" in sql:
            if sibling_error:
                raise app_module.DatabaseError("sibling failure")
            return [{"has_sibling": sibling_match}]
        return []

    return _route_query


def _default_bbox_face_row(**overrides):
    row = {
        "id": 1,
        "image_id": 10,
        "is_target": 1,
        "classified_by": "model",
        "confidence": 0.23,
        "classified_by_user_id": None,
        "sdc_written": 0,
        "file_title": "File:Face.jpg",
        "commons_page_id": 12345,
        "project_id": 1,
        "wikidata_qid": "Q42",
        "image_status": "processed",
    }
    row.update(overrides)
    return row


def _bbox_query_router(face_row=None, exists=True, ownership_error=False):
    local_row = _default_bbox_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "file_title" in sql and "project_members" in sql:
            if ownership_error:
                raise app_module.DatabaseError("ownership failure")
            return [local_row] if exists else []
        return []

    return _route_query


def test_api_manual_face_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_manual_face_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data={"csrf_token": "testtoken"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_manual_face_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(project_id="abc"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_manual_face_invalid_bbox_dimensions(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(bbox_top="50", bbox_bottom="40"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid bounding box dimensions"


def test_api_manual_face_bbox_out_of_range_negative(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(bbox_left="-1"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_bbox_out_of_range_too_large(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(bbox_right="10001"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_bbox_out_of_range_too_small_area(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(bbox_top="10", bbox_left="10", bbox_bottom="19", bbox_right="20"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_dismiss_face_ids_too_many(monkeypatch):
    import json as _json

    client, _ = _authed_client(monkeypatch)
    oversized = _json.dumps(list(range(101)))
    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(dismiss_face_ids=oversized),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Too many face IDs in dismiss list"


def test_api_manual_face_ownership_check_fail(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return []
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Image not found or access denied"


def test_api_manual_face_no_encoding_result(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(encodings=[])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_manual_face_success_normal_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    with client.session_transaction() as sess:
        assert sess["manual_faces_10"] == [99]
        assert sess["last_classify"]["action"] == "manual_face"
        assert sess["last_classify"]["was_review"] is False
        assert sess["last_classify"]["face_ids"] == []
        assert sess["last_classify"]["manual_face_ids"] == [99]


def test_api_manual_face_dismiss_faces_on_normal_insert(monkeypatch):
    """When dismiss_face_ids is sent, the API marks those faces as non-target
    in the same transaction, preventing the image from reappearing."""
    executed_sqls = []

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_cursor.rowcount = 1

        original_execute = mock_cursor.execute

        def tracking_execute(sql, params=None):
            executed_sqls.append((sql, params))
            return original_execute(sql, params)

        mock_cursor.execute = tracking_execute
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    import json

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post(
            "/api/manual-face",
            data=_manual_face_form(dismiss_face_ids=json.dumps([5, 7])),
        )

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    dismiss_sqls = [(sql, params) for sql, params in executed_sqls if "UPDATE faces SET is_target = 0" in sql]
    assert len(dismiss_sqls) == 1
    sql, params = dismiss_sqls[0]
    assert "id IN (%s,%s)" in sql
    # params: (user_id, face_id_1, face_id_2, image_id)
    assert params[1] == 5
    assert params[2] == 7

    with client.session_transaction() as sess:
        assert sess["last_classify"]["face_ids"] == [5, 7]
        assert sess["last_classify"]["was_review"] is False
        assert sess["last_classify"]["manual_face_ids"] == [99]

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 11}, {"id": 12}]
        mock_cursor.lastrowid = 100
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form(reviewing_model="1"))

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    with client.session_transaction() as sess:
        assert sess["manual_faces_10"] == [100]
        assert sess["last_classify"]["was_review"] is True
        assert sess["last_classify"]["face_ids"] == [11, 12]
        assert sess["last_classify"]["manual_face_ids"] == [100]


def test_api_manual_face_download_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("network")

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", _raise)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 502
    assert resp.get_json()["error"] == "Failed to download image from Commons"


def test_api_manual_face_db_error_on_ownership(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            raise app_module.DatabaseError("db")
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_manual_face_db_error_on_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_manual_face_unexpected_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    mock_fr = _build_face_recognition_mock(face_encodings_side_effect=RuntimeError("boom"))
    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


def test_api_manual_face_image_not_processed(monkeypatch):
    """Return 409 when the image has not finished processing (status != 'processed')."""

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "pending"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Image has not finished processing yet"


def test_remove_sdc_claim_success_claim_found_and_removed(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"success": 1}

    get_mock = MagicMock(side_effect=[claim_resp, token_resp])
    post_mock = MagicMock(return_value=remove_resp)
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", post_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 2
    assert post_mock.call_count == 1
    post_data = post_mock.call_args.kwargs["data"]
    assert post_data["action"] == "wbremoveclaims"
    assert post_data["claim"] == "M555$ABC"


def test_remove_sdc_claim_no_matching_claim_returns_true(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$NOTME", "mainsnak": {"datavalue": {"value": {"id": "Q1"}}}},
            ]
        }
    }

    get_mock = MagicMock(return_value=claim_resp)
    post_mock = MagicMock()
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", post_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 1
    assert post_mock.call_count == 0


def test_remove_sdc_claim_fetches_csrf_token(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"success": 1}

    get_mock = MagicMock(side_effect=[claim_resp, token_resp])
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", MagicMock(return_value=remove_resp))

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    second_get_params = get_mock.call_args_list[1].kwargs["params"]
    assert second_get_params["action"] == "query"
    assert second_get_params["meta"] == "tokens"
    assert second_get_params["type"] == "csrf"


def test_remove_sdc_claim_removal_api_returns_error(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"error": {"code": "badtoken"}}

    monkeypatch.setattr(app_module.requests, "get", MagicMock(side_effect=[claim_resp, token_resp]))
    monkeypatch.setattr(app_module.requests, "post", MagicMock(return_value=remove_resp))

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False


def test_remove_sdc_claim_exception_during_api_calls(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        MagicMock(side_effect=requests.RequestException("network")),
    )

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False


def test_api_reclassify_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_reclassify_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_reclassify_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "abc", "is_target": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_reclassify_invalid_is_target_outside_allowed(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "2"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_reclassify_face_not_found(monkeypatch):
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(ownership_exists=False),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Face not found or access denied"


def test_api_reclassify_approve_from_model_face(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["is_target"] == 1
    assert payload["classified_by"] == "model"
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    assert payload["updated_faces"] == []


def test_api_reclassify_reject_bootstrap_no_sibling_sdc_removal_queued(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True


def test_api_reclassify_reject_bootstrap_with_sibling_no_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=1),
    )

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False


def test_api_reclassify_reject_nonbootstrap_sdc_written_no_sibling_immediate_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is True
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_called_once_with(555, "Q42", "valid-token")


def test_api_reclassify_reject_nonbootstrap_sdc_written_with_sibling_no_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=1),
    )

    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_not_called()


def test_api_reclassify_reject_nonbootstrap_sdc_removal_fails(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", lambda *_a, **_k: False)
    tx_mock = MagicMock()
    monkeypatch.setattr(app_module, "execute_transaction", tx_mock)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 502
    assert "Failed to remove SDC claim" in resp.get_json()["error"]
    tx_mock.assert_not_called()


def test_api_reclassify_reject_nonbootstrap_token_expired(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    monkeypatch.setattr(app_module, "_get_valid_token", lambda: None)
    tx_mock = MagicMock()
    monkeypatch.setattr(app_module, "execute_transaction", tx_mock)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 401
    assert "OAuth token expired" in resp.get_json()["error"]
    tx_mock.assert_not_called()


def test_api_reclassify_already_reviewed_by_another_user_returns_409(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {"classified_by_user_id": 999}
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "This face has already been reviewed by another user"


def test_api_reclassify_same_user_reclick_noop_success(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {"classified_by_user_id": 1}
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_reclassify_counter_increment_on_approve(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0)
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql in executed_sql)
    assert any("UPDATE faces SET sdc_removal_pending = 0" in sql for sql in executed_sql)


def test_api_reclassify_counter_decrement_on_reject(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=1, sdc_written=0)
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any("CAST(faces_confirmed AS SIGNED) - 1" in sql for sql in executed_sql)


def test_api_reclassify_db_error_on_ownership(monkeypatch):
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(ownership_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_sibling_lookup_bootstrap(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_sibling_lookup_nonbootstrap(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_execute_transaction(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save reclassification"


def test_api_update_face_bbox_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_update_face_bbox_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data={"csrf_token": "testtoken"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_update_face_bbox_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(face_id="abc"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_update_face_bbox_invalid_bbox_dimensions(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/update-face-bbox",
        data=_update_bbox_form(bbox_top="40", bbox_bottom="40"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid bounding box dimensions"


def test_api_update_face_bbox_bbox_out_of_range_negative(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(bbox_top="-1"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_bbox_out_of_range_too_large(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(bbox_right="20000"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_bbox_out_of_range_too_small_area(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/update-face-bbox",
        data=_update_bbox_form(bbox_top="10", bbox_left="10", bbox_bottom="19", bbox_right="20"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_face_not_found(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(exists=False))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Face not found or access denied"


def test_api_update_face_bbox_no_encoding_result(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(encodings=[])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_update_face_bbox_success_new_face_inserted_original_superseded(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 321
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload == {"status": "ok", "new_face_id": 321, "original_face_id": 1}

    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any(sql.startswith("INSERT INTO faces") for sql in executed_sql)
    assert any("UPDATE faces SET superseded_by" in sql for sql in executed_sql)


def test_api_update_face_bbox_download_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(
        app_module,
        "_download_image",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.RequestException("network")),
    )

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 502
    assert resp.get_json()["error"] == "Failed to download image from Commons"


def test_api_update_face_bbox_db_error_on_ownership(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(ownership_error=True))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_update_face_bbox_db_error_on_insert(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )
    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_update_face_bbox_unexpected_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(face_encodings_side_effect=RuntimeError("boom"))

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


def test_api_update_face_bbox_image_not_processed(monkeypatch):
    face_row = _default_bbox_face_row(image_status="pending")
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))
    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Image has not finished processing yet"


@pytest.fixture
def fake_user():
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _make_authed_client(monkeypatch, fake_user, route_execute=None):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if route_execute is not None:
            return route_execute(sql, params, fetch)
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client


def _set_csrf_chunk6(client, token=None):
    token = token or "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes_chunk6(client):
    with client.session_transaction() as sess:
        return list(sess.get("_flashes", []))


def test_api_write_sdc_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_write_sdc_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_write_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_pending_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_already_requested(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 5, "removal_cnt": 2}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["message"] == "already_requested"
    assert payload["pending"] == 5
    assert payload["removal_pending"] == 2


def test_api_write_sdc_no_pending_writes(monkeypatch, fake_user):
    updates = []

    def route_execute(sql, _params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 0, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            updates.append((sql, fetch))
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 0, "removal_pending": 0}
    assert updates == []


def test_api_write_sdc_successful_flag_set(monkeypatch, fake_user):
    update_calls = []

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 3, "removal_cnt": 1}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            update_calls.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    fake_file = MagicMock()
    with patch("builtins.open", return_value=fake_file):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 3, "removal_pending": 1}
    assert len(update_calls) == 1


def test_api_write_sdc_update_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 7, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_wakeup_file_oserror_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 1, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    with patch("builtins.open", side_effect=OSError("nope")):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_api_sdc_status_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_sdc_status_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1, "sdc_write_error": None}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            return [{"written": 9, "pending": 2, "removal_pending": 1}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {
        "status": "ok",
        "written": 9,
        "pending": 2,
        "removal_pending": 1,
        "in_progress": True,
        "error": None,
    }


def test_api_sdc_status_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_sdc_status_counts_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0, "sdc_write_error": "oops"}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


# ── /api/stop-sdc/<id> ──────────────────────────────────────────────


def test_api_stop_sdc_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_stop_sdc_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_stop_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_stop_sdc_not_in_progress(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["message"] == "not_in_progress"


def test_api_stop_sdc_success(monkeypatch, fake_user):
    updates = []

    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            updates.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "message" not in payload
    assert len(updates) == 1
    assert "sdc_write_error = NULL" in updates[0]


def test_api_stop_sdc_update_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_progress_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 404


def test_api_progress_active_with_pending_and_stats(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 4, "images_total": 10, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 6}]
        if "COUNT(*) AS total_faces" in sql:
            return [
                {
                    "total_faces": 12,
                    "confirmed_matches": 5,
                    "confirmed_non_matches": 3,
                    "unclassified": 4,
                    "sdc_written": 2,
                    "by_human": 7,
                    "by_model": 4,
                    "by_bootstrap": 1,
                }
            ]
        if "AND f.is_target IS NULL" in sql and "inference_eligible" not in sql:
            return [{"cnt": 4}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["images_processed"] == 4
    assert payload["images_total"] == 10
    assert payload["pending_images"] == 6
    assert payload["complete"] is False
    assert payload["face_stats"]["total_faces"] == 12
    assert payload["inference_eligible"] == 4


def test_api_progress_completed_project(monkeypatch, fake_user):
    seen_pending_query = {"called": False}

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 10, "images_total": 10, "status": "completed"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            seen_pending_query["called"] = True
            return [{"cnt": 1}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["complete"] is True
    assert payload["pending_images"] == 0
    assert seen_pending_query["called"] is False


def test_api_progress_face_stats_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 1, "images_total": 3, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 2}]
        if "COUNT(*) AS total_faces" in sql:
            raise app_module.DatabaseError("boom")
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 9}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["face_stats"] == {}
    assert payload["inference_eligible"] == 9


def test_api_progress_inference_eligible_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 2, "images_total": 3, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 1}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5}]
        if "AND f.is_target IS NULL" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["inference_eligible"] == 0


def test_api_progress_pending_images_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 0, "images_total": 8, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            raise app_module.DatabaseError("boom")
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 200
    assert response.get_json()["pending_images"] == 0


def test_api_progress_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def _project_settings_base_row():
    return {
        "id": 1,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Douglas_Adams",
        "label": "Douglas Adams",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }


def test_project_settings_get_renders_form(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Settings" in response.data


def test_project_settings_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 404


def test_project_settings_csrf_fail_on_post(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client, "expected")

    response = client.post("/project/1/settings", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_settings_invalid_distance_threshold_string(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "abc",
            "min_confirmed": "5",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Distance threshold must be a number." in response.data


def test_project_settings_invalid_distance_threshold_out_of_range(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "1.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Distance threshold must be between 0.1 and 1.0." in response.data


def test_project_settings_invalid_min_confirmed_string(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "abc",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Minimum confirmed must be a whole number." in response.data


def test_project_settings_invalid_min_confirmed_below_one(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "0",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Minimum confirmed must be at least 1." in response.data


def test_project_settings_invalid_status(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "5",
            "status": "invalid",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Invalid status." in response.data


def test_project_settings_successful_update_redirects(monkeypatch, fake_user):
    project = _project_settings_base_row()
    updates = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            updates.append((params, fetch))
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.55",
            "min_confirmed": "6",
            "status": "paused",
            "label": "New Label",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert len(updates) == 1


def test_project_settings_success_with_warning_for_min_confirmed(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 1, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "8",
            "status": "active",
            "label": "New Label",
        },
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert ("success", "Settings updated.") in flashes
    assert any(cat == "warning" and "Autonomous inference will not run" in msg for cat, msg in flashes)


def test_project_settings_db_error_on_update(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "8",
            "status": "active",
            "label": "New Label",
        },
    )

    assert response.status_code == 200
    assert b"Failed to update settings." in response.data


def test_project_settings_completed_status_preserves_completion_reason(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["status"] = "completed"
    sqls_seen = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            sqls_seen.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "completed",
            "label": "My Project",
        },
    )

    assert response.status_code == 302
    assert len(sqls_seen) == 1
    # Must NOT include completion_reason = NULL when keeping completed status
    assert "completion_reason" not in sqls_seen[0]


def test_project_settings_reactivation_clears_completion_reason(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["status"] = "completed"
    sqls_seen = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            sqls_seen.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "My Project",
        },
    )

    assert response.status_code == 302
    assert len(sqls_seen) == 1
    # MUST include completion_reason = NULL when reactivating
    assert "completion_reason" in sqls_seen[0]


def test_project_rerun_inference_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_rerun_inference_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 404


def test_project_rerun_inference_settings_unchanged_noop(monkeypatch, fake_user):
    calls = {"reset_sql_seen": False}

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "UPDATE faces f" in sql:
            calls["reset_sql_seen"] = True
            return 4
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    flashes = _flashes_chunk6(client)
    assert (
        "info",
        "Settings have not changed since the last inference run. No re-run needed.",
    ) in flashes
    assert calls["reset_sql_seen"] is False


def test_project_rerun_inference_successful_reset_with_affected_faces(monkeypatch, fake_user):
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 3}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 4
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    assert any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)
    flashes = _flashes_chunk6(client)
    assert any(cat == "success" and "Reset 4 model-classified faces." in msg for cat, msg in flashes)
    assert any(cat == "info" and "10/7 human-confirmed" in msg for cat, msg in flashes)


def test_project_rerun_inference_null_last_inference_values_still_resets(monkeypatch, fake_user):
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": None,
                    "last_inference_min_confirmed": None,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 3, "by_bootstrap": 2}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 2
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    assert any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)
    flashes = _flashes_chunk6(client)
    # Below threshold (3/5) + has bootstrap faces → warning + bootstrap tip
    assert any(cat == "warning" and "3/5 human-confirmed" in msg for cat, msg in flashes)
    assert any(cat == "info" and "2 bootstrapped" in msg for cat, msg in flashes)


def test_project_rerun_inference_no_faces_to_reset(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 8, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    executed_sql = []

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("info", "No model-classified faces to reset.") in _flashes_chunk6(client)
    # Project update must not run when affected=0
    assert not any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)


def test_project_rerun_inference_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 0, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(_fn):
        raise app_module.DatabaseError("boom")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to re-run inference.") in _flashes_chunk6(client)


def test_project_delete_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/delete", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_delete_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert ("warning", "Project deleted.") in _flashes_chunk6(client)


def test_project_delete_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Project not found.") in _flashes_chunk6(client)


def test_project_delete_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to delete project.") in _flashes_chunk6(client)


def test_leaderboard_success_with_rows(monkeypatch):
    rows = [
        {"wiki_username": "alice", "classifications": 7, "sdc_tags": 3},
        {"wiki_username": "bob", "classifications": 2, "sdc_tags": 1},
    ]

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"Leaderboard" in response.data
    assert b"alice" in response.data
    assert b">9<" in response.data
    assert b">4<" in response.data


def test_leaderboard_db_error(monkeypatch):
    def boom(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        raise app_module.DatabaseError("boom")

    monkeypatch.setattr(app_module, "execute_query", boom)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"Failed to load leaderboard." in response.data


def test_leaderboard_empty_results(monkeypatch):
    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"No contributions yet" in response.data


def test_leaderboard_period_month(monkeypatch):
    """Month filter uses time-filtered SQL without user_stats."""
    captured = {}
    rows = [{"wiki_username": "alice", "classifications": 3, "sdc_tags": 1}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=month")

    assert response.status_code == 200
    assert b"alice" in response.data
    assert "classified_at" in captured["sql"]
    assert "user_stats" not in captured["sql"]
    assert captured["params"] is not None
    assert len(captured["params"]) == 2


def test_leaderboard_period_daily(monkeypatch):
    """Daily filter uses time-filtered SQL without user_stats."""
    captured = {}
    rows = [{"wiki_username": "bob", "classifications": 1, "sdc_tags": 0}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=daily")

    assert response.status_code == 200
    assert b"bob" in response.data
    assert "classified_at" in captured["sql"]
    assert captured["params"] is not None
    assert len(captured["params"]) == 2


def test_leaderboard_period_all_includes_user_stats(monkeypatch):
    """All-time filter includes user_stats for archived project totals."""
    captured = {}
    rows = [{"wiki_username": "carol", "classifications": 10, "sdc_tags": 5}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=all")

    assert response.status_code == 200
    assert b"carol" in response.data
    assert "user_stats" in captured["sql"]
    assert captured["params"] is None


def test_leaderboard_invalid_period_defaults_to_all(monkeypatch):
    """Invalid period value falls back to all-time query."""
    captured = {}
    rows = [{"wiki_username": "dave", "classifications": 2, "sdc_tags": 0}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=bogus")

    assert response.status_code == 200
    assert b"dave" in response.data
    assert "user_stats" in captured["sql"]
    assert captured["params"] is None


def test_leaderboard_filter_bar_active_state(monkeypatch):
    """Filter bar renders with correct active class for each period."""

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    # Default (all)
    resp = client.get("/leaderboard")
    assert b'class="active"' in resp.data
    assert b"All Time" in resp.data

    # Month
    resp = client.get("/leaderboard?period=month")
    html = resp.data.decode()
    assert "period=month" in html

    # Daily
    resp = client.get("/leaderboard?period=daily")
    html = resp.data.decode()
    assert "period=daily" in html


def test_health_healthy_db(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 1}])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy", "database": "connected"}


def test_health_unhealthy_db_exception(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(app_module, "execute_query", boom)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy", "error": "database unavailable"}


def test_health_unhealthy_empty_rows(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy"}


def test_health_unhealthy_wrong_value(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 0}])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy"}


def test_error_handler_400(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: abort(400))

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/")

    assert response.status_code == 400
    assert b"400" in response.data


def test_error_handler_404():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404
    assert b"Page not found" in response.data


def test_error_handler_429(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: abort(429))

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/")

    assert response.status_code == 429
    assert b"Rate limit exceeded" in response.data


def test_error_handler_500(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    previous_testing = flask_app.config.get("TESTING", False)
    previous_propagate = flask_app.config.get("PROPAGATE_EXCEPTIONS")
    flask_app.config["TESTING"] = False
    flask_app.config["PROPAGATE_EXCEPTIONS"] = False
    client = flask_app.test_client()

    response = client.get("/")

    flask_app.config["TESTING"] = previous_testing
    flask_app.config["PROPAGATE_EXCEPTIONS"] = previous_propagate
    assert response.status_code == 500
    assert b"Internal server error" in response.data


def test_commons_thumb_route_default_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 330)


def test_commons_thumb_route_custom_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg?width=500")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 500)


def test_create_app_returns_app():
    assert app_module.create_app() is flask_app


def test_account_settings_get(monkeypatch, fake_user):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "SELECT leaderboard_opt_out" in sql:
            return [{"leaderboard_opt_out": 0}]
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert b"Account Settings" in response.data
    assert b"leaderboard_opt_out" in response.data


def test_account_settings_get_opted_out(monkeypatch, fake_user):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "SELECT leaderboard_opt_out" in sql:
            return [{"leaderboard_opt_out": 1}]
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert b"checked" in response.data


def test_account_settings_post_opt_out(monkeypatch, fake_user):
    updated = {}

    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            updated["opt_out"] = params[0]
            return 1
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 302
    assert updated["opt_out"] == 1
    assert ("success", "Settings saved.") in _flashes_chunk6(client)


def test_account_settings_post_opt_in(monkeypatch, fake_user):
    updated = {}

    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            updated["opt_out"] = params[0]
            return 1
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 302
    assert updated["opt_out"] == 0


def test_account_settings_post_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "wrong", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 400


def test_account_settings_post_db_error(monkeypatch, fake_user):
    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            raise app_module.DatabaseError("boom")
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 302
    assert ("error", "Failed to save settings.") in _flashes_chunk6(client)


def test_account_settings_requires_login(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    response = client.get("/account/settings")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_leaderboard_excludes_opted_out_users(monkeypatch):
    captured_sql = {}

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured_sql["leaderboard"] = sql
            return [{"wiki_username": "visible", "classifications": 5, "sdc_tags": 2}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"visible" in response.data
    assert "leaderboard_opt_out = 0" in captured_sql["leaderboard"]


def test_leaderboard_sdc_tags_attributed_to_classifier_not_project_owner(monkeypatch):
    """SDC tag count must be grouped by faces.classified_by_user_id, matching
    the archival logic in worker.py, not by projects.user_id (project owner)."""
    captured_sql = {}

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured_sql["leaderboard"] = sql
            return [{"wiki_username": "alice", "classifications": 3, "sdc_tags": 1}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    sql = captured_sql["leaderboard"]
    # Live query must credit the face classifier, not the project owner
    assert "classified_by_user_id" in sql
    assert "JOIN projects p ON sc.project_id = p.id" not in sql


def _flashes_tail(client):
    with client.session_transaction() as sess:
        return sess.get("_flashes", [])


def test_set_language_without_referrer_falls_back_to_index():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/set-language/en")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_api_category_info_breaks_when_batched_queue_only_contains_visited(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    repeated_subcats = [{"title": "Category:A"} for _ in range(51)]

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 4, "subcats": 51},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": repeated_subcats}})
        if params.get("prop") == "categoryinfo" and "Category:A" in params.get("titles", ""):
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 3, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 7
    assert body["categories_visited"] == 2


def test_api_category_info_breaks_subcategory_fetch_when_deadline_exceeded(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 1, "subcats": 2},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3(
                {"query": {"categorymembers": [{"title": "Category:A"}, {"title": "Category:B"}]}}
            )
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A|Category:B":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "10": {"title": "Category:A", "categoryinfo": {"files": 2, "subcats": 1}},
                            "11": {"title": "Category:B", "categoryinfo": {"files": 3, "subcats": 1}},
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3({"query": {"categorymembers": []}})
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:B":
            raise AssertionError("Should break before requesting Category:B")
        raise AssertionError(params)

    monotonic_values = iter([0, 1, 1, 9, 9, 9])

    def _monotonic():
        return next(monotonic_values, 9)

    monkeypatch.setattr(app_module.requests, "get", _get)
    monkeypatch.setattr(app_module.time, "monotonic", _monotonic)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["files"] == 6


def test_api_category_info_marks_approximate_when_continue_in_subcategory_fetch(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 2, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 5, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {"categorymembers": []},
                    "continue": {"cmcontinue": "next-page"},
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_project_new_duplicate_entry_detected_via_exception_cause(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    cause = Exception(1062, "Duplicate entry")
    db_exc = DatabaseError("duplicate")
    db_exc.__cause__ = cause

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("still being cleaned up" in msg for _cat, msg in _flashes_tail(client))


def test_project_new_duplicate_error_with_non_numeric_cause_args_falls_back(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    cause = Exception("not-a-number")
    db_exc = DatabaseError("insert failed")
    db_exc.__cause__ = cause

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project. Please try again." in msg for _cat, msg in _flashes_tail(client))


def _make_project_new_base_execute(insert_side_effect):
    """Return an execute_query stub for /project/new that calls insert_side_effect on INSERT."""

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return insert_side_effect()
        raise AssertionError(sql)

    return _execute_query


def test_project_new_invite_code_collision_retries_and_succeeds(monkeypatch):
    """First INSERT raises invite_code collision; second attempt succeeds."""
    call_count = {"n": 0}

    def _insert():
        call_count["n"] += 1
        if call_count["n"] == 1:
            cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
            exc = DatabaseError("dup invite_code")
            exc.__cause__ = cause
            raise exc
        return 1

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _make_project_new_base_execute(_insert))
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert call_count["n"] == 2
    assert any("successfully" in msg.lower() for _cat, msg in _flashes_tail(client))


def test_project_new_invite_code_collision_fails_after_max_retries(monkeypatch):
    """All retry attempts raise invite_code collisions; shows 'Failed to create project'."""
    captured = _capture_render_template_chunk4(monkeypatch)

    def _insert():
        cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
        exc = DatabaseError("dup invite_code")
        exc.__cause__ = cause
        raise exc

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _make_project_new_base_execute(_insert))
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project" in msg for _cat, msg in _flashes_tail(client))

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM faces f" in sql and "WHERE f.id = %s" in sql and "project_members" in sql:
            return [
                {
                    "id": 1,
                    "image_id": 10,
                    "old_is_target": 0,
                    "sdc_written": 0,
                    "classified_by": "model",
                    "commons_page_id": 555,
                    "bootstrapped": 0,
                    "project_id": 1,
                    "wikidata_qid": "Q42",
                }
            ]
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda _fn: (_ for _ in ()).throw(ValueError("unexpected")),
    )

    with pytest.raises(ValueError, match="unexpected"):
        client.post(
            "/api/reclassify",
            data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"},
        )


def test_project_settings_get_db_error_returns_500(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            raise DatabaseError("boom")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)

    response = client.get("/project/1/settings")

    assert response.status_code == 500


def test_project_settings_post_shows_bootstrap_tip(monkeypatch):
    project = {
        "id": 1,
        "user_id": 1,
        "distance_threshold": 0.6,
        "min_confirmed": 5,
        "status": "active",
        "label": "Existing",
    }

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 1, "by_bootstrap": 2}]
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "Updated",
        },
    )

    assert response.status_code == 302
    flashes = _flashes_tail(client)
    assert any(cat == "info" and "bootstrapped matches" in msg for cat, msg in flashes)


def test_project_settings_post_ignores_stats_query_db_error(monkeypatch):
    project = {
        "id": 1,
        "user_id": 1,
        "distance_threshold": 0.6,
        "min_confirmed": 5,
        "status": "active",
        "label": "Existing",
    }

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            raise DatabaseError("stats failed")
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "Updated",
        },
    )

    assert response.status_code == 302
    assert ("success", "Settings updated.") in _flashes_tail(client)


def test_project_rerun_inference_db_error_loading_project_returns_500(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            raise DatabaseError("load failed")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 500


def test_project_rerun_inference_ignores_wake_file_oserror(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 8, "by_bootstrap": 0}]
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)
    monkeypatch.setattr(app_module, "execute_transaction", lambda _fn: 3)
    monkeypatch.setattr("builtins.open", MagicMock(side_effect=OSError("disk error")))

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any(cat == "success" and "Reset 3 model-classified faces." in msg for cat, msg in _flashes_tail(client))


def test_project_rerun_inference_ignores_stats_query_db_error(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            raise DatabaseError("stats query failed")
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)
    monkeypatch.setattr(app_module, "execute_transaction", lambda _fn: 1)
    monkeypatch.setattr("builtins.open", mock_open())

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any(cat == "success" and "Reset 1 model-classified faces." in msg for cat, msg in _flashes_tail(client))


def test_account_settings_get_db_error_falls_back_to_opt_out_zero(monkeypatch):
    captured = {}

    def _fake_render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return app_module.jsonify(context)

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "SELECT leaderboard_opt_out" in sql:
            raise DatabaseError("cannot load")
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        return ()

    monkeypatch.setattr(app_module, "render_template", _fake_render)
    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert captured["template"] == "account_settings.html"
    assert captured["context"]["leaderboard_opt_out"] == 0


def test_service_worker_route_sets_required_headers():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/sw.js")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/javascript"
    assert response.headers["Service-Worker-Allowed"] == "/"
    assert response.headers["Cache-Control"] == "no-cache"


def test_chrome_devtools_json_returns_empty_json():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/.well-known/appspecific/com.chrome.devtools.json")

    assert response.status_code == 200
    assert response.get_json() == {}


def test_main_guard_runs_app(monkeypatch):
    import runpy

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

    with patch.dict(os.environ, {"PORT": "8765", "FLASK_DEBUG": "1"}, clear=False):
        with patch("database.init_db"):
            with patch("flask.app.Flask.run") as run_mock:
                runpy.run_path(app_path, run_name="__main__")

    run_mock.assert_called_once()
    kwargs = run_mock.call_args.kwargs
    assert kwargs["debug"] is True
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8765


def test_robots_txt_returns_plain_text():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.content_type == "text/plain; charset=utf-8"
    body = response.data.decode()
    assert "User-agent: *" in body
    assert "Allow: /$" in body
    assert "Allow: /leaderboard" in body
    assert "Disallow: /" in body
    assert "Sitemap:" in body
    assert "sitemap.xml" in body


def test_project_settings_get_shows_members_list(monkeypatch, fake_user):
    project = _project_settings_base_row()
    members = [
        {
            "user_id": 5,
            "role": "member",
            "joined_at": datetime(2025, 1, 10, 12, 0, 0),
            "wiki_username": "Alice",
            "status": "active",
        },
        {
            "user_id": 8,
            "role": "member",
            "joined_at": datetime(2025, 2, 15, 9, 30, 0),
            "wiki_username": "Bob",
            "status": "active",
        },
    ]

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql and "JOIN users" in sql:
            return members
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Alice" in response.data
    assert b"Bob" in response.data
    assert b"Project Members" in response.data


def test_project_settings_get_empty_members_list(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Members" in response.data


def test_project_settings_get_members_db_error_returns_empty(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            raise app_module.DatabaseError("members failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Settings" in response.data


def test_remove_member_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    banned = {"called": False}

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            banned["called"] = True
            assert params == (1, 5)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    assert "/project/1/settings" in response.headers["Location"]
    assert banned["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("banned" in msg.lower() for _cat, msg in flashes)


def test_remove_member_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "wrong", "member_user_id": "5"},
    )

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_remove_member_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 404


def test_remove_member_self_removal_blocked(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "1"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("cannot remove yourself" in msg.lower() for _cat, msg in flashes)


def test_remove_member_no_member_specified(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("no member" in msg.lower() for _cat, msg in flashes)


def test_remove_member_invalid_member_id(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "abc"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_remove_member_not_found_in_db(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "99"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("not found" in msg.lower() for _cat, msg in flashes)


def test_remove_member_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE project_members SET status = 'banned'" in sql:
            raise app_module.DatabaseError("ban failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/remove-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def test_unban_member_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    unbanned = {"called": False}

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            unbanned["called"] = True
            assert params == (1, 5)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    assert "/project/1/settings" in response.headers["Location"]
    assert unbanned["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("unbanned" in msg.lower() for _cat, msg in flashes)


def test_unban_member_not_found(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "99"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("not found" in msg.lower() for _cat, msg in flashes)


def test_unban_member_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "wrong", "member_user_id": "5"},
    )

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_unban_member_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 404


def test_unban_member_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql and "status = 'banned'" in sql:
            raise app_module.DatabaseError("unban failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings/unban-member",
        data={"csrf_token": "testtoken", "member_user_id": "5"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate(monkeypatch, fake_user):
    project = _project_settings_base_row()
    update_calls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            update_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert len(update_calls) == 1
    assert update_calls[0][0] is not None
    assert len(update_calls[0][0]) == 8
    flashes = _flashes_chunk6(client)
    assert any("generated" in msg.lower() for _cat, msg in flashes)


def test_invite_code_revoke(monkeypatch, fake_user):
    project = _project_settings_base_row()
    update_calls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code = NULL" in sql:
            update_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "revoke"},
    )

    assert response.status_code == 302
    assert len(update_calls) == 1
    flashes = _flashes_chunk6(client)
    assert any("revoked" in msg.lower() for _cat, msg in flashes)


def test_invite_code_invalid_action(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "bogus"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_invite_code_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 404


def test_invite_code_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/invite-code", data={"csrf_token": "wrong"})

    assert response.status_code == 400


def test_invite_code_generate_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)


def _make_invite_code_collision_error():
    """Build a DatabaseError that looks like a MySQL 1062 on the invite_code index."""
    cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
    exc = app_module.DatabaseError("duplicate invite_code")
    exc.__cause__ = cause
    return exc


def test_is_invite_code_collision_true():
    exc = _make_invite_code_collision_error()
    assert app_module._is_invite_code_collision(exc)


def test_is_invite_code_collision_false_wrong_key():
    cause = Exception(1062, "Duplicate entry '...' for key 'idx_projects_user_qid_cat'")
    exc = app_module.DatabaseError("duplicate project")
    exc.__cause__ = cause
    assert not app_module._is_invite_code_collision(exc)


def test_is_invite_code_collision_false_not_1062():
    cause = Exception(1064, "Syntax error")
    exc = app_module.DatabaseError("syntax error")
    exc.__cause__ = cause
    assert not app_module._is_invite_code_collision(exc)


def test_invite_code_generate_retries_on_collision(monkeypatch, fake_user):
    """First UPDATE raises an invite_code collision; second attempt succeeds."""
    project = _project_settings_base_row()
    call_count = {"n": 0}

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _make_invite_code_collision_error()
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    assert call_count["n"] == 2
    flashes = _flashes_chunk6(client)
    assert any("generated" in msg.lower() for _cat, msg in flashes)


def test_invite_code_generate_fails_after_max_retries(monkeypatch, fake_user):
    """All retry attempts raise invite_code collisions; route shows failure flash."""
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            raise _make_invite_code_collision_error()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert response.status_code == 302
    flashes = _flashes_chunk6(client)
    assert any("failed" in msg.lower() for _cat, msg in flashes)

    insert_calls = []

    def route_execute(sql, params, fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return ()
        if "INSERT INTO project_members" in sql:
            insert_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    assert len(insert_calls) == 1
    flashes = _flashes_chunk6(client)
    assert any("joined" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_invalid_code(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return ()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "BADCODE1"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_empty_code(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": ""},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("enter" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_banned_user(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "banned"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("banned" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_already_member(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "Cool Project", "user_id": 99, "wiki_username": "Alice"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "active"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("already" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_owner(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            return [{"id": 42, "label": "My Project", "user_id": 1, "wiki_username": "tester"}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert "/project/42" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("owner" in msg.lower() for _cat, msg in flashes)


def test_join_by_code_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/join", data={"csrf_token": "wrong"})

    assert response.status_code == 400


def test_join_by_code_db_error_on_lookup(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            raise app_module.DatabaseError("lookup failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("wrong" in msg.lower() for _cat, msg in flashes)


def test_project_settings_shows_banned_members(monkeypatch, fake_user):
    project = _project_settings_base_row()
    members = [
        {
            "user_id": 5,
            "role": "member",
            "joined_at": datetime(2025, 1, 10, 12, 0, 0),
            "wiki_username": "Alice",
            "status": "active",
        },
        {
            "user_id": 8,
            "role": "member",
            "joined_at": datetime(2025, 2, 15, 9, 30, 0),
            "wiki_username": "Eve",
            "status": "banned",
        },
    ]

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql and "JOIN users" in sql:
            return members
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Alice" in response.data
    assert b"Eve" in response.data
    assert b"Banned" in response.data
    assert b"Unban" in response.data


def test_dashboard_member_counts_passed_to_template(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, params, fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 2}]
        if "SELECT DISTINCT p.*" in sql:
            return [
                {"id": 10, "wikidata_qid": "Q42", "p18_thumb_url": "t1"},
                {"id": 20, "wikidata_qid": "Q1", "p18_thumb_url": "t2"},
            ]
        if "project_members" in sql and "COUNT" in sql:
            return [{"project_id": 10, "cnt": 3}, {"project_id": 20, "cnt": 1}]
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {10: 3, 20: 1}


def test_dashboard_member_counts_db_error_is_non_critical(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 10, "wikidata_qid": "Q42", "p18_thumb_url": "t1"}]
        if "project_members" in sql and "COUNT" in sql:
            raise app_module.DatabaseError("members count failed")
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {}


def test_dashboard_member_counts_empty_when_no_projects(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {}


def test_project_detail_member_count_passed_to_template(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": "t.jpg", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS cnt" in sql and "project_members" in sql:
            return [{"cnt": 4}]
        if "status = 'pending'" in sql:
            return [{"cnt": 1}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["context"]["member_count"] == 4


def test_project_detail_member_count_db_error_defaults_zero(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": "t.jpg", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS cnt" in sql and "project_members" in sql:
            raise app_module.DatabaseError("member count failed")
        if "status = 'pending'" in sql:
            return [{"cnt": 1}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["context"]["member_count"] == 0


def test_sitemap_xml_returns_valid_xml():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/sitemap.xml")

    assert response.status_code == 200
    assert response.content_type == "application/xml; charset=utf-8"
    body = response.data.decode()

    # Parse XML to ensure it is well-formed and has the expected structure.
    root = ET.fromstring(body)
    assert root.tag.endswith("urlset")

    loc_texts = [(elem.text or "") for elem in root.iter() if elem.tag.endswith("loc")]
    assert any("/leaderboard" in text for text in loc_texts)


# --- _check_p180_exists tests ---


def test_check_p180_exists_true_when_claim_matches(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse({"claims": {"P180": [{"mainsnak": {"datavalue": {"value": {"id": "Q42"}}}}]}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is True


def test_check_p180_exists_false_when_different_qid(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse({"claims": {"P180": [{"mainsnak": {"datavalue": {"value": {"id": "Q99"}}}}]}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_when_no_p180_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {}}))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_true_among_multiple_claims(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse(
            {
                "claims": {
                    "P180": [
                        {"mainsnak": {"datavalue": {"value": {"id": "Q99"}}}},
                        {"mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
                    ]
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is True


def test_check_p180_exists_false_on_missing_nested_keys(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P180": [{}]}}))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_calls_commons_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P180": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._check_p180_exists(777, "Q76")

    assert captured["url"] == app_module.COMMONS_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "M777"
    assert captured["params"]["property"] == "P180"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


# --- P180 check at classify time tests ---


def test_api_classify_target_marks_sdc_written_when_p180_exists(monkeypatch):
    executed_queries = []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            return [{"commons_page_id": 555, "wikidata_qid": "Q42"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed_queries.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: True)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)
    assert any("UPDATE images SET bootstrapped = 1" in sql for sql, _ in executed_queries)


def test_api_classify_target_no_sdc_written_when_p180_missing(monkeypatch):
    executed_queries = []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            return [{"commons_page_id": 555, "wikidata_qid": "Q42"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed_queries.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: False)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert not any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)


def test_api_classify_target_p180_check_db_error_non_fatal(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            raise app_module.DatabaseError("meta lookup failed")
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


# --- P180 check at reclassify time tests ---


def test_api_reclassify_approve_marks_sdc_written_when_p180_exists(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    executed_queries = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        if "FROM images i JOIN projects p" in sql and "commons_page_id" in sql:
            return [{"commons_page_id": face_row["commons_page_id"], "wikidata_qid": face_row["wikidata_qid"]}]
        return []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: True)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)
    assert any("UPDATE images SET bootstrapped = 1" in sql for sql, _ in executed_queries)


def test_api_reclassify_approve_no_sdc_written_when_p180_missing(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    executed_queries = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        return []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: False)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert not any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)


def test_api_reclassify_approve_skips_p180_check_when_already_sdc_written(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model", sdc_written=1)
    p180_called = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        return []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    def mock_check(cpid, qid):
        p180_called.append((cpid, qid))
        return True

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", mock_check)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert len(p180_called) == 0


def test_project_detail_shows_no_faces_banner(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [
                {
                    "id": 1,
                    "user_id": 1,
                    "wikidata_qid": "Q42",
                    "p18_thumb_url": "https://thumb.jpg",
                    "status": "completed",
                    "completion_reason": "no_faces",
                    "images_total": 50,
                    "images_processed": 50,
                }
            ]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0, "confirmed_matches": 0}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 0}]
        if "status = 'pending'" in sql:
            return [{"cnt": 0}]
        if "f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["completion_reason"] == "no_faces"


def test_project_detail_shows_insufficient_faces_banner(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [
                {
                    "id": 1,
                    "user_id": 1,
                    "wikidata_qid": "Q42",
                    "p18_thumb_url": "https://thumb.jpg",
                    "status": "completed",
                    "completion_reason": "insufficient_faces",
                    "images_total": 50,
                    "images_processed": 50,
                }
            ]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 3, "confirmed_matches": 0}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 0}]
        if "status = 'pending'" in sql:
            return [{"cnt": 0}]
        if "f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["completion_reason"] == "insufficient_faces"


def test_project_settings_clears_completion_reason_on_update(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["completion_reason"] = "no_faces"
    project["status"] = "completed"
    update_sqls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            update_sqls.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.55",
            "min_confirmed": "3",
            "status": "active",
            "label": "Reactivated",
        },
    )

    assert response.status_code == 302
    assert len(update_sqls) == 1
    assert "completion_reason = NULL" in update_sqls[0]


def test_leave_project_success(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99
    deleted = {"called": False}

    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            deleted["called"] = True
            assert params == (1, 1)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]
    assert deleted["called"] is True
    flashes = _flashes_chunk6(client)
    assert any("left the project" in msg.lower() for _cat, msg in flashes)


def test_leave_project_owner_blocked(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/project/1" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("cannot leave" in msg.lower() for _cat, msg in flashes)


def test_leave_project_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)
    _set_csrf_chunk6(client, "expected")

    response = client.post("/project/1/leave", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_leave_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 404


def test_leave_project_not_a_member(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99

    def route_execute(sql, _params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/dashboard" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("not a member" in msg.lower() for _cat, msg in flashes)


def test_leave_project_db_error(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["user_id"] = 99

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project.copy()]
        if "DELETE FROM project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/leave", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "/project/1" in response.headers["Location"]
    flashes = _flashes_chunk6(client)
    assert any("failed to leave" in msg.lower() for _cat, msg in flashes)


# --- Invite code TTL tests ---


def test_invite_code_generate_sql_includes_created_at(monkeypatch, fake_user):
    project = _project_settings_base_row()
    captured_sql = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code" in sql:
            captured_sql.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "generate"},
    )

    assert len(captured_sql) == 1
    assert "invite_code_created_at = NOW()" in captured_sql[0]


def test_invite_code_revoke_sql_clears_created_at(monkeypatch, fake_user):
    project = _project_settings_base_row()
    captured_sql = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET invite_code = NULL" in sql:
            captured_sql.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    client.post(
        "/project/1/invite-code",
        data={"csrf_token": "testtoken", "action": "revoke"},
    )

    assert len(captured_sql) == 1
    assert "invite_code_created_at = NULL" in captured_sql[0]


def test_join_by_code_expired_shows_invalid(monkeypatch, fake_user):
    def route_execute(sql, params, _fetch):
        if "WHERE p.invite_code = %s" in sql:
            assert "invite_code_created_at" in sql
            assert "INTERVAL %s DAY" in sql
            assert params == ("ABC12345", app_module._INVITE_CODE_TTL_DAYS)
            return ()
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/join",
        data={"csrf_token": "testtoken", "invite_code": "ABC12345"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    flashes = _flashes_chunk6(client)
    assert any("invalid" in msg.lower() for _cat, msg in flashes)


def test_invite_code_ttl_days_is_seven():
    assert app_module._INVITE_CODE_TTL_DAYS == 7


# --- XML escaping in sitemap/robots tests ---


def test_sitemap_xml_escapes_url_root():
    flask_app.config["TESTING"] = True
    flask_app.config["SERVER_NAME"] = "evil.com/<script>"
    try:
        client = flask_app.test_client()
        response = client.get("/sitemap.xml", base_url="http://evil.com/<script>/")
        body = response.data.decode()

        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        root = ET.fromstring(body)
        assert root.tag.endswith("urlset")
    finally:
        flask_app.config.pop("SERVER_NAME", None)


def test_robots_txt_escapes_url_root():
    flask_app.config["TESTING"] = True
    flask_app.config["SERVER_NAME"] = "evil.com/<script>"
    try:
        client = flask_app.test_client()
        response = client.get("/robots.txt", base_url="http://evil.com/<script>/")
        body = response.data.decode()

        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert "Sitemap:" in body
    finally:
        flask_app.config.pop("SERVER_NAME", None)


# --- Gallery API filtered SDC pending tests ---


def test_api_gallery_filtered_sdc_pending_with_source_filter(monkeypatch, fake_user):
    call_log = []

    def route_execute(sql, params, fetch):
        call_log.append(sql)
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM faces" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS total" in sql and "sdc_pending" not in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS total" in sql and "sdc_pending" in sql:
            return [
                {
                    "total": 10,
                    "matches": 6,
                    "non_matches": 3,
                    "rejected": 1,
                    "source_model": 4,
                    "source_bootstrap": 2,
                    "source_human": 4,
                    "sdc_pending": 5,
                }
            ]
        if "sdc_pending" in sql and "COUNT(*) AS total" not in sql:
            return [{"sdc_pending": 2}]
        if "SELECT f.id" in sql:
            return []
        return [] if fetch else 0

    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client = _make_authed_client(monkeypatch, fake_user, route_execute)
        response = client.get("/api/project/1/gallery?page=1&source=model")
        payload = response.get_json()
    finally:
        flask_app.config.pop("SERVER_NAME", None)

    assert response.status_code == 200
    assert "counts" in payload
    assert payload["counts"]["sdc_pending"] == 5
    assert payload["counts"]["filtered_sdc_pending"] == 2


def test_api_gallery_filtered_sdc_pending_no_filter_equals_global(monkeypatch, fake_user):
    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM faces" in sql:
            return [{"cnt": 5}]
        if "COUNT(*) AS total" in sql:
            return [
                {
                    "total": 10,
                    "matches": 6,
                    "non_matches": 3,
                    "rejected": 1,
                    "source_model": 4,
                    "source_bootstrap": 2,
                    "source_human": 4,
                    "sdc_pending": 5,
                }
            ]
        if "SELECT f.id" in sql:
            return []
        return [] if fetch else 0

    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client = _make_authed_client(monkeypatch, fake_user, route_execute)
        response = client.get("/api/project/1/gallery?page=1")
        payload = response.get_json()
    finally:
        flask_app.config.pop("SERVER_NAME", None)

    assert response.status_code == 200
    assert "counts" in payload
    assert payload["counts"]["filtered_sdc_pending"] == payload["counts"]["sdc_pending"]


def test_api_reclassify_approve_rejects_sibling_target_faces(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))

        executed_sql = []

        def _transaction(fn):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 1
            mock_cursor.fetchall.return_value = [{"id": 100, "sdc_written": 0}, {"id": 101, "sdc_written": 1}]

            def capture_execute(sql, params=None):
                executed_sql.append(sql)
                mock_cursor.rowcount = 1

            mock_cursor.execute = capture_execute
            return fn(mock_conn, mock_cursor)

        monkeypatch.setattr(app_module, "execute_transaction", _transaction)
        resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})
    finally:
        flask_app.config.pop("SERVER_NAME", None)

    assert resp.status_code == 200

    payload = resp.get_json()
    assert len(payload["updated_faces"]) == 2
    assert {uf["face_id"] for uf in payload["updated_faces"]} == {100, 101}

    # Verify sdc_removal_queued matches pre-demotion sdc_written state per sibling
    sdc_by_face = {uf["face_id"]: uf["sdc_removal_queued"] for uf in payload["updated_faces"]}
    assert sdc_by_face[100] is False, "Face 100 had sdc_written=0, should not queue removal"
    assert sdc_by_face[101] is True, "Face 101 had sdc_written=1, should queue removal"

    sibling_reject_sqls = [s for s in executed_sql if "is_target = 0" in s and "image_id" in s and "id !=" in s]
    assert len(sibling_reject_sqls) > 0, "Expected UPDATE to reject sibling target faces"

    # Verify sdc_written=0 and sdc_removal_pending CASE are in the demotion UPDATE
    demotion_sql = sibling_reject_sqls[0]
    assert "sdc_written = 0" in demotion_sql, "Expected sdc_written reset on demoted siblings"
    assert "sdc_removal_pending" in demotion_sql, "Expected sdc_removal_pending handling on demoted siblings"

    counter_decrement_sqls = [s for s in executed_sql if "GREATEST" in s]
    assert len(counter_decrement_sqls) > 0, "Expected faces_confirmed decrement for rejected siblings"
