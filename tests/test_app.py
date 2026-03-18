import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

import numpy as np
import pytest
import requests
from flask import Response, abort, g, session
import xml.etree.ElementTree as ET

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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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


def _reset_whitelist_cache(monkeypatch, cache=None, cache_time=0.0):
    monkeypatch.setattr(app_module, "_whitelist_cache", set() if cache is None else cache.copy())
    monkeypatch.setattr(app_module, "_whitelist_cache_time", cache_time)


def test_parse_whitelist_empty():
    assert app_module._parse_whitelist("") == set()


def test_parse_whitelist_ignores_comments_and_blank_lines():
    text = "\n# comment\n\nuser1\n"
    assert app_module._parse_whitelist(text) == {"user1"}


def test_parse_whitelist_strips_whitespace():
    text = "  user1  \n\tuser2\t\n"
    assert app_module._parse_whitelist(text) == {"user1", "user2"}


def test_parse_whitelist_keeps_hash_if_not_line_start():
    text = "user#name\n"
    assert app_module._parse_whitelist(text) == {"user#name"}


def test_load_whitelist_returns_cache_within_ttl(monkeypatch):
    _reset_whitelist_cache(monkeypatch, cache={"cached"}, cache_time=100.0)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 150.0)

    called = {"value": False}

    def _should_not_call(*_args, **_kwargs):
        called["value"] = True
        raise AssertionError("requests.get should not be called when cache is fresh")

    monkeypatch.setattr(app_module.requests, "get", _should_not_call)
    result = app_module._load_whitelist()

    assert result == {"cached"}
    assert called["value"] is False


def test_load_whitelist_refreshes_from_github_success(monkeypatch):
    _reset_whitelist_cache(monkeypatch)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 1000.0)

    resp = MagicMock()
    resp.text = "alice\nbob\n"
    resp.raise_for_status.return_value = None
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    result = app_module._load_whitelist()

    assert result == {"alice", "bob"}
    assert app_module._whitelist_cache == {"alice", "bob"}
    assert app_module._whitelist_cache_time == 1000.0


def test_load_whitelist_github_empty_falls_back_to_local(monkeypatch):
    _reset_whitelist_cache(monkeypatch)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 101.0)

    resp = MagicMock()
    resp.text = "\n# only comments\n"
    resp.raise_for_status.return_value = None
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)
    monkeypatch.setattr("builtins.open", mock_open(read_data="local1\nlocal2\n"))

    result = app_module._load_whitelist()

    assert result == {"local1", "local2"}


def test_load_whitelist_github_fail_local_success(monkeypatch):
    _reset_whitelist_cache(monkeypatch)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 500.0)

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("network")

    monkeypatch.setattr(app_module.requests, "get", _raise)
    monkeypatch.setattr("builtins.open", mock_open(read_data="localuser\n"))

    result = app_module._load_whitelist()
    assert result == {"localuser"}


def test_load_whitelist_both_fail_returns_last_known_good_cache(monkeypatch):
    _reset_whitelist_cache(monkeypatch, cache={"known"}, cache_time=0.0)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 1000.0)

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("network")

    monkeypatch.setattr(app_module.requests, "get", _raise)

    def _raise_file(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("builtins.open", _raise_file)
    result = app_module._load_whitelist()

    assert result == {"known"}


def test_load_whitelist_both_fail_no_cache_returns_empty(monkeypatch):
    _reset_whitelist_cache(monkeypatch)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: 2000.0)

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("network")

    monkeypatch.setattr(app_module.requests, "get", _raise)

    def _raise_file(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("builtins.open", _raise_file)
    assert app_module._load_whitelist() == set()


def test_load_whitelist_cache_expired_refetches(monkeypatch):
    _reset_whitelist_cache(monkeypatch, cache={"old"}, cache_time=0.0)
    monkeypatch.setattr(app_module.time, "monotonic", lambda: app_module._WHITELIST_CACHE_TTL + 1.0)

    resp = MagicMock()
    resp.text = "newuser\n"
    resp.raise_for_status.return_value = None
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    assert app_module._load_whitelist() == {"newuser"}


def test_before_request_without_session_user_id_sets_no_user():
    with flask_app.test_request_context("/"):
        app_module.before_request()
        assert g.user is None


def test_before_request_sets_user_when_valid_and_whitelisted(monkeypatch):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [fake_user.copy()])
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        app_module.before_request()
        assert g.user is not None
        assert g.user["wiki_username"] == "tester"
        assert session.get("user_id") == 1


def test_before_request_revokes_session_when_not_whitelisted(monkeypatch):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [fake_user.copy()])
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"someone_else"})

    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        app_module.before_request()
        assert g.user is None
        assert "user_id" not in session


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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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
    response = flask_app.make_response(("ok", 200))
    result = app_module.set_security_headers(response)

    assert result.headers["X-Content-Type-Options"] == "nosniff"
    assert result.headers["X-Frame-Options"] == "DENY"
    assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert result.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


class _FakeResponse:
    def __init__(self, headers=None, chunks=None, error=None):
        self.headers = headers or {}
        self._chunks = chunks or []
        self._error = error
        self.closed = False

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

    data = app_module._download_image("https://example.org/file.jpg", max_bytes=10)
    assert data == b"abcdef"


def test_download_image_content_length_too_large(monkeypatch):
    resp = _FakeResponse(headers={"Content-Length": "11"}, chunks=[b"abc"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="Image too large"):
        app_module._download_image("https://example.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_stream_exceeds_limit(monkeypatch):
    resp = _FakeResponse(headers={}, chunks=[b"12345", b"67890", b"x"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="exceeded"):
        app_module._download_image("https://example.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_raises_http_error(monkeypatch):
    resp = _FakeResponse(error=requests.HTTPError("bad response"))
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(requests.HTTPError):
        app_module._download_image("https://example.org/file.jpg")


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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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


def test_oauth_callback_whitelist_denied_redirects_to_index(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 999, "username": "blocked"})
    )
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    flashes = _get_flashes(client)
    assert any("Access is currently restricted" in msg for _cat, msg in flashes)


def test_oauth_callback_whitelist_empty_denies_login(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)

    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 999, "username": "tester"})
    )
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: set())

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT * FROM projects" in sql:
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
    assert any("COUNT(*) AS cnt" in sql for sql, _params, _fetch in calls)
    assert any("SELECT * FROM projects" in sql for sql, _params, _fetch in calls)


def test_dashboard_invalid_page_falls_back_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
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
        if "COUNT(*) AS cnt" in sql:
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 30}]
        if "SELECT * FROM projects" in sql:
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
        if "COUNT(*) AS cnt" in sql:
            raise app_module.DatabaseError("count failed")
        if "SELECT * FROM projects" in sql:
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 10}]
        if "SELECT * FROM projects" in sql:
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
        if "COUNT(*) AS cnt" in sql:
            return ()
        if "SELECT * FROM projects" in sql:
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 3}]
        if "SELECT * FROM projects" in sql:
            return projects
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((params, fetch))
            return 1
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT * FROM projects" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            update_called["called"] = True
            return 1
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
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT * FROM projects" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("update failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200


def test_dashboard_projects_tuple_skips_lazy_population(monkeypatch):
    fake_user = _fake_user_chunk3()
    fetch_calls = {"count": 0}

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT * FROM projects" in sql:
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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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


def test_project_new_successful_creation_with_label_fetch_and_p18(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise app_module.DatabaseError("insert failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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


def test_clear_skips_clears_normal_and_review_keys(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "abc", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid face ID"
    assert any("SELECT i.id, i.file_title FROM images i" in q[0] for q in queries)


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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
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
    assert any("AND is_target = 0 AND classified_by = 'model'" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["was_review"] is True


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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
        if "SELECT i.id, i.file_title FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg"}]
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

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
        if "FROM faces f " in sql and "old_is_target" in sql:
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
        "file_title": "File:Face.jpg",
        "project_id": 1,
    }
    row.update(overrides)
    return row


def _bbox_query_router(face_row=None, exists=True, ownership_error=False):
    local_row = _default_bbox_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "file_title" in sql:
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
            return [{"id": 10, "file_title": "File:Face.jpg"}]
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
            return [{"id": 10, "file_title": "File:Face.jpg"}]
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


def test_api_manual_face_success_review_mode_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
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
            return [{"id": 10, "file_title": "File:Face.jpg"}]
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
            return [{"id": 10, "file_title": "File:Face.jpg"}]
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
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    mock_fr = _build_face_recognition_mock(face_encodings_side_effect=RuntimeError("boom"))
    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
        if "SELECT * FROM projects WHERE id" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_write_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_pending_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_sdc_status_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_sdc_status_counts_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0, "sdc_write_error": "oops"}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_progress_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 404


def test_api_progress_active_with_pending_and_stats(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
        if "SELECT * FROM projects WHERE id" in sql:
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
    assert response.headers["Location"].endswith("/project/1")
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
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
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: set())

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


def test_project_new_duplicate_entry_detected_via_orig_error_code(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    class _Orig:
        args = (1062,)

    db_exc = DatabaseError("duplicate")
    db_exc.orig = _Orig()  # type: ignore[attr-defined]

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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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


def test_project_new_duplicate_error_with_non_numeric_orig_args_falls_back(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    class _Orig:
        args = ("not-a-number",)

    db_exc = DatabaseError("insert failed")
    db_exc.orig = _Orig()  # type: ignore[attr-defined]

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
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
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


def test_api_reclassify_reraises_unexpected_value_error(monkeypatch):
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
        if "FROM faces f" in sql and "WHERE f.id = %s" in sql:
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

    with patch.dict(os.environ, {"PORT": "8765"}, clear=False):
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
