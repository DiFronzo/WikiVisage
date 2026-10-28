"""Tests for the web app: auth."""

import re
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from flask import Response, g, session

from tests._app_preamble import (
    _is_safe_url,
    _validate_csrf,
    app_module,
    flask_app,
)
from tests._app_support import (
    _capture_render,
    _get_flashes,
    _make_authenticated_client_chunk3,
    _MockResponse,
)


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


def _reset_rate_limit():
    app_module.limiter.reset()


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


def test_before_request_without_session_user_id_sets_no_user():
    with flask_app.test_request_context("/"):
        app_module.before_request()
        assert g.user is None


def test_before_request_database_error_keeps_session(monkeypatch):
    """A ToolsDB hiccup must not log the user out: the session survives, g.user is just unavailable."""

    def _raise_db(*_args, **_kwargs):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    with flask_app.test_request_context("/"):
        session["user_id"] = 7
        app_module.before_request()
        assert g.user is None
        assert g.user_load_failed is True
        assert session["user_id"] == 7


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


def test_login_required_redirects_unauthenticated(monkeypatch):
    with flask_app.test_request_context("/protected?x=1"):
        g.user = None
        monkeypatch.setattr(app_module, "_", lambda x: x)

        @app_module.login_required
        def _protected():
            return "ok"

        response = _protected()
        assert isinstance(response, Response)
        assert response.status_code == 302
        assert response.location is not None
        assert "/login?next=" in response.location
        assert "next=" in response.location and "protected" in response.location


def test_login_required_allows_authenticated_user():
    with flask_app.test_request_context("/protected"):
        g.user = {"id": 1}

        @app_module.login_required
        def _protected():
            return "ok"

        assert _protected() == "ok"


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

        def refresh_token(self, token_url, refresh_token, client_id, client_secret, **kwargs):
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
        if fetch:
            return [
                {
                    "access_token": "access-old",
                    "refresh_token": "refresh-old",
                    "token_expires_at": "2000-01-01 00:00:00",
                }
            ]
        return 1

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth)
    monkeypatch.setattr(app_module, "execute_query", _execute)

    updated = app_module._refresh_access_token(user)

    assert updated is user
    assert user["access_token"] == "access-new"
    assert user["refresh_token"] == "refresh-new"
    assert isinstance(user["token_expires_at"], datetime)
    assert db_calls[-1][0].startswith("UPDATE users SET access_token")
    assert db_calls[-1][2] is False
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

    def _execute(sql, params, fetch=False):
        if fetch:
            return [
                {
                    "access_token": "access-old",
                    "refresh_token": "refresh-old",
                    "token_expires_at": "2000-01-01 00:00:00",
                }
            ]
        return 1

    monkeypatch.setattr(app_module, "execute_query", _execute)

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

    def _execute(sql, params, fetch=False):
        return [
            {
                "access_token": "access",
                "refresh_token": "refresh",
                "token_expires_at": "2000-01-01 00:00:00",
            }
        ]

    monkeypatch.setattr(app_module, "execute_query", _execute)
    assert app_module._refresh_access_token(user) is None


def test_get_valid_token_returns_none_without_user():
    with flask_app.test_request_context("/"):
        g.user = None
        assert app_module._get_valid_token() is None


def test_get_valid_token_keeps_session_when_refresh_fails(monkeypatch):
    with flask_app.test_request_context("/"):
        g.user = {"access_token": "old"}
        session["user_id"] = 1
        monkeypatch.setattr(app_module, "_refresh_access_token", lambda _user: None)

        assert app_module._get_valid_token() is None
        assert session["user_id"] == 1


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
    locale_cookie = next((c for c in resp.headers.getlist("Set-Cookie") if c.startswith("locale=")), "")
    assert "locale=es" in locale_cookie
    assert "Max-Age=" in locale_cookie


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


def test_set_language_nocookie_uses_session_scoped_cookie(monkeypatch):
    """Without consent the choice still applies for the visit, but is not persisted."""
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
    locale_cookie = next((c for c in resp.headers.getlist("Set-Cookie") if c.startswith("locale=")), "")
    assert "locale=es" in locale_cookie
    # Session-scoped: no persistence attributes, so the choice lasts until the browser closes.
    assert "Expires=" not in locale_cookie
    assert "Max-Age=" not in locale_cookie


def test_set_language_without_consent_changes_language():
    """A visitor who has not accepted cookies still gets the language they picked."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    resp = client.get("/set-language/es?nocookie=1", headers={"Referer": "http://localhost/"})
    assert resp.status_code == 302

    page = client.get("/")
    assert page.status_code == 200
    assert b'<html lang="es"' in page.data


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


def test_login_redirects_to_dashboard_when_already_authenticated(monkeypatch):
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)

    response = client.get("/login")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_login_with_reauth_runs_the_oauth_flow_even_when_logged_in(monkeypatch):
    """One-click re-authorize: the browser login is fine, Wikimedia no longer accepts the token."""
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)
    oauth = MagicMock()
    oauth.authorization_url.return_value = ("https://oauth.example/authorize", "state-reauth")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    response = client.get("/login?reauth=1&next=/project/12")

    assert response.status_code == 302
    assert response.headers["Location"] == "https://oauth.example/authorize"  # not the dashboard
    with client.session_transaction() as sess:
        assert sess["oauth_state"] == "state-reauth"
        assert sess["login_next"] == "/project/12"
        assert sess["user_id"] == 1  # still logged in until the callback replaces the session


def test_login_with_reauth_from_another_site_goes_to_the_dashboard(monkeypatch):
    """The state-changing GET must not be triggerable by a link on a third-party page."""
    _reset_rate_limit()
    client, _ = _make_authenticated_client(monkeypatch)
    oauth = MagicMock()
    oauth.authorization_url.return_value = ("https://oauth.example/authorize", "state-x")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    response = client.get("/login?reauth=1", headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    with client.session_transaction() as sess:
        assert "oauth_state" not in sess


def test_login_with_reauth_from_the_same_site_or_a_typed_url_runs_oauth(monkeypatch):
    oauth = MagicMock()
    oauth.authorization_url.return_value = ("https://oauth.example/authorize", "state-y")
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)

    for site in ("same-origin", "none"):
        _reset_rate_limit()
        client, _ = _make_authenticated_client(monkeypatch)

        response = client.get("/login?reauth=1", headers={"Sec-Fetch-Site": site})

        assert response.headers["Location"] == "https://oauth.example/authorize", site


def test_login_with_reauth_is_still_blocked_while_the_database_is_down(monkeypatch):
    _reset_rate_limit()
    monkeypatch.setattr(
        app_module, "execute_query", lambda *a, **k: (_ for _ in ()).throw(app_module.DatabaseError("x"))
    )
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1

    assert client.get("/login?reauth=1").status_code == 503


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
        if sql.startswith("SELECT id, session_epoch FROM users WHERE wiki_user_id"):
            return [{"id": 77}]
        return 1

    monkeypatch.setattr(app_module, "execute_query", fake_execute_query)

    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert any(sql.startswith("UPDATE users SET") for sql, _params, _fetch in calls)
    with client.session_transaction() as sess:
        assert sess["user_id"] == 77
        # Security: the pre-login CSRF token must NOT be preserved across login.
        # Preserving it would let an attacker who fixated a session before login
        # retain a valid CSRF token afterwards. See _is_safe_url tests + login handler.
        assert sess["csrf_token"] != "keep-me"
        assert isinstance(sess["csrf_token"], str) and len(sess["csrf_token"]) >= 32


def _run_callback_as(monkeypatch, client, resolved_user_id):
    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "a", "refresh_token": "r", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 123, "username": "tester"})
    )

    def fake_execute_query(sql, params=None, fetch=True):
        if sql.startswith("SELECT id, session_epoch FROM users WHERE wiki_user_id"):
            return [{"id": resolved_user_id}]
        if "FROM users WHERE id = %s" in sql:  # the signed-in user the old session belongs to
            return [
                {
                    "id": params[0],
                    "wiki_username": "tester",
                    "access_token": "t",
                    "refresh_token": "r",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        return 1

    monkeypatch.setattr(app_module, "execute_query", fake_execute_query)
    return client.get("/auth/callback?code=x")


def _seed_work_state(client, user_id):
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["auth_at"] = int(time.time()) - 1000
        sess["csrf_token"] = "keep"
        sess["last_classify"] = {"action": "yes", "project_id": 5, "image_id": 9, "manual_face_ids": [3]}
        sess["skipped_images_5"] = [1]
        sess["skipped_images_review_5"] = [2]
        sess["manual_faces_9"] = [3]
        sess["something_else"] = "dropped"
    _set_oauth_state(client)


def test_oauth_callback_reauth_same_user_keeps_undo_skips_and_csrf(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _seed_work_state(client, 77)

    response = _run_callback_as(monkeypatch, client, 77)

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert sess["user_id"] == 77
        assert sess["csrf_token"] == "keep"
        assert sess["last_classify"]["image_id"] == 9
        assert sess["skipped_images_5"] == [1]
        assert sess["skipped_images_review_5"] == [2]
        assert sess["manual_faces_9"] == [3]
        assert sess["auth_at"] > int(time.time()) - 500  # refreshed
        assert "something_else" not in sess  # only the work state is carried over


def test_oauth_callback_as_another_account_drops_state_and_rotates_csrf(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _seed_work_state(client, 77)

    response = _run_callback_as(monkeypatch, client, 88)

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert sess["user_id"] == 88
        assert sess["csrf_token"] != "keep"
        assert "last_classify" not in sess
        assert "skipped_images_5" not in sess
        assert "manual_faces_9" not in sess


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
        if sql.startswith("SELECT id, session_epoch FROM users WHERE wiki_user_id") and len(calls) == 1:
            return []
        if sql.startswith("SELECT id, session_epoch FROM users WHERE wiki_user_id") and len(calls) == 3:
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


def _database_down(monkeypatch):
    def boom(*_a, **_kw):
        raise app_module.DatabaseError("db down")

    monkeypatch.setattr(app_module, "execute_query", boom)


def test_logout_still_works_while_the_database_is_down(monkeypatch):
    _reset_rate_limit()
    _database_down(monkeypatch)
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "tok"

    response = client.post("/logout", data={"csrf_token": "tok"})

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_logout_without_a_session_just_redirects(monkeypatch):
    _reset_rate_limit()
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])

    response = flask_app.test_client().post("/logout", data={"csrf_token": "x"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_landing_page_answers_503_while_the_signed_in_user_cannot_be_loaded(monkeypatch):
    _reset_rate_limit()
    _database_down(monkeypatch)
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/")

    assert response.status_code == 503


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


def test_set_language_without_referrer_falls_back_to_index():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/set-language/en")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_set_language_external_netloc_referrer_redirects_to_index(monkeypatch):
    """set_language strips external referrers and falls back to index."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [])

    # External host in referrer — should be redirected to '/' (index), not to evil.com
    response = client.get(
        "/set-language/en",
        headers={"Referer": "https://evil.example.com/phish"},
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    # Must NOT redirect to external domain
    assert "evil.example.com" not in location


def test_set_language_scheme_in_referrer_redirects_to_index(monkeypatch):
    """set_language rejects referrers that contain scheme (e.g. http://...) after stripping netloc."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [])

    response = client.get(
        "/set-language/nb",
        headers={"Referer": "javascript:alert(1)"},
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "javascript" not in location


def test_set_language_double_slash_referrer_redirects_to_index(monkeypatch):
    """set_language rejects referrers starting with '//' (protocol-relative URLs)."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [])

    response = client.get(
        "/set-language/en",
        headers={"Referer": "//evil.example.com/path"},
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "evil.example.com" not in location


def test_before_request_token_decryption_error_clears_session(monkeypatch):
    """When token decryption fails, session is cleared and g.user is set to None."""
    from token_crypto import TokenDecryptionError

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user_row = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "encrypted-garbage",
        "refresh_token": "encrypted-garbage",
        "token_expires_at": None,
    }

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [fake_user_row])
    monkeypatch.setattr(app_module, "decrypt_token", lambda t: (_ for _ in ()).throw(TokenDecryptionError("bad")))

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/")

    assert response.status_code == 200
    # Session should now be cleared (user_id removed)
    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_is_safe_url_relative_path():
    with flask_app.test_request_context("/"):
        assert _is_safe_url("/dashboard") is True


def test_is_safe_url_external_scheme():
    with flask_app.test_request_context("/"):
        assert _is_safe_url("http://evil.com") is False


def test_is_safe_url_javascript_scheme():
    with flask_app.test_request_context("/"):
        assert _is_safe_url("javascript:alert(1)") is False


def test_is_safe_url_empty_string():
    with flask_app.test_request_context("/"):
        assert _is_safe_url("") is False


def test_is_safe_url_double_slash():
    with flask_app.test_request_context("/"):
        assert _is_safe_url("//evil.com/path") is False


# ---------------------------------------------------------------------------
# Session persistence, revocation and transient failures
# ---------------------------------------------------------------------------


def _session_cookies(response):
    return [h for h in response.headers.getlist("Set-Cookie") if h.startswith("session=")]


def _days_until_expiry(set_cookie: str) -> float:
    from email.utils import parsedate_to_datetime

    match = re.search(r"Expires=([^;]+)", set_cookie)
    assert match, f"no Expires in {set_cookie!r}"
    return (parsedate_to_datetime(match.group(1)) - datetime.now(UTC)).total_seconds() / 86400


def _assert_persistent(response):
    cookies = _session_cookies(response)
    assert cookies, "response sent no session cookie"
    for cookie in cookies:
        assert "Expires=" in cookie or "Max-Age=" in cookie, cookie


class _FakeUsers:
    """A one-user table: serves the before_request SELECT and counts epoch UPDATEs."""

    def __init__(self, epoch=0):
        self.epoch = epoch
        self.statements = []
        self.fail_updates = False

    def __call__(self, sql, params=None, fetch=True):
        self.statements.append(sql)
        if sql.startswith("SELECT id, wiki_user_id"):
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                    "session_epoch": self.epoch,
                }
            ]
        if sql.startswith("UPDATE users SET session_epoch"):
            if self.fail_updates:
                raise app_module.DatabaseError("db down")
            self.epoch += 1
            return 1
        return []

    def epoch_updates(self):
        return [s for s in self.statements if "session_epoch" in s and s.startswith("UPDATE")]


def _logged_in_client(epoch=0, csrf="tok", auth_at=None):
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess.permanent = True
        sess["user_id"] = 1
        sess["csrf_token"] = csrf
        sess["auth_at"] = int(time.time()) if auth_at is None else auth_at
        sess["epoch"] = epoch
    return client


def _is_logged_in(client) -> bool:
    """/login bounces a logged-in user to the dashboard, everyone else to the OAuth server."""
    response = client.get("/login")
    return response.status_code == 302 and response.headers["Location"].endswith("/dashboard")


def test_oauth_callback_sets_permanent_session_with_auth_at_and_epoch(monkeypatch):
    _reset_rate_limit()
    client = flask_app.test_client()
    _set_oauth_state(client)
    oauth = MagicMock()
    oauth.fetch_token.return_value = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}
    monkeypatch.setattr(app_module, "_make_oauth_session", lambda *_a, **_kw: oauth)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"sub": 123, "username": "tester"})
    )

    def fake_execute_query(sql, params=None, fetch=True):
        if sql.startswith("SELECT id, session_epoch FROM users WHERE wiki_user_id"):
            return [{"id": 77, "session_epoch": 4}]
        return 1

    monkeypatch.setattr(app_module, "execute_query", fake_execute_query)

    before = int(time.time())
    response = client.get("/auth/callback?code=x")

    assert response.status_code == 302
    _assert_persistent(response)
    assert 29 < _days_until_expiry(_session_cookies(response)[-1]) < 31  # the idle lifetime, 30 days
    with client.session_transaction() as sess:
        assert sess.permanent is True
        assert before <= sess["auth_at"] <= int(time.time())
        assert sess["epoch"] == 4


def test_login_sends_a_persistent_pre_login_cookie():
    """iOS home-screen apps reset session-scoped cookies, which would drop oauth_state mid-login."""
    _reset_rate_limit()
    client = flask_app.test_client()

    response = client.get("/login")

    assert response.status_code == 302
    assert "oauth2/authorize" in response.headers["Location"]
    _assert_persistent(response)
    with client.session_transaction() as sess:
        assert sess["oauth_state"]


def test_anonymous_csrf_cookie_is_persistent(monkeypatch):
    """The banner only asks for a token once a language cookie exists; that cookie must be persistent."""
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])
    client = flask_app.test_client()
    client.set_cookie("locale", "nb")

    response = client.get("/")

    assert response.status_code == 200
    _assert_persistent(response)
    with client.session_transaction() as sess:
        assert sess["csrf_token"]


def test_flash_only_session_cookie_is_persistent():
    """A login_required bounce flashes into an anonymous session: that cookie must be persistent too."""
    client = flask_app.test_client()

    response = client.get("/dashboard")

    assert response.status_code == 302
    _assert_persistent(response)


def test_logout_keeps_its_flash_cookie_persistent(monkeypatch):
    _reset_rate_limit()
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    client = _logged_in_client()

    response = client.post("/logout", data={"csrf_token": "tok"})

    assert response.status_code == 302
    _assert_persistent(response)


def test_session_survives_more_than_the_old_eight_hours(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    client = _logged_in_client(auth_at=int(time.time()) - 20 * 86400)

    assert _is_logged_in(client)


def test_before_request_epoch_mismatch_clears_session(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers(epoch=2))
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        session["auth_at"] = int(time.time())
        session["epoch"] = 1
        app_module.before_request()
        assert g.user is None
        assert "user_id" not in session


def test_before_request_missing_epoch_matches_row_epoch_zero(monkeypatch):
    """Sessions created before the epoch existed carry no key and must survive."""
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers(epoch=0))
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        app_module.before_request()
        assert g.user is not None
        assert session["user_id"] == 1


def test_before_request_user_row_without_epoch_key_counts_as_zero(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "execute_query",
        lambda *_a, **_kw: [
            {
                "id": 1,
                "wiki_user_id": 1,
                "wiki_username": "t",
                "access_token": "a",
                "refresh_token": "r",
                "token_expires_at": None,
            }
        ],
    )
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        session["epoch"] = 0
        app_module.before_request()
        assert g.user is not None


def test_before_request_backfills_missing_auth_at(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        before = int(time.time())
        app_module.before_request()
        assert g.user is not None
        assert before <= session["auth_at"] <= int(time.time())


def test_before_request_session_older_than_max_age_is_cleared_with_a_flash(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    too_old = int(time.time()) - app_module._SESSION_MAX_AGE_SECONDS - 60
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        session["auth_at"] = too_old
        app_module.before_request()
        assert g.user is None
        assert "user_id" not in session
        assert ("warning", "Your session has expired. Please log in again.") in session["_flashes"]


def test_before_request_session_just_inside_max_age_is_kept(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    with flask_app.test_request_context("/"):
        session["user_id"] = 1
        session["auth_at"] = int(time.time()) - app_module._SESSION_MAX_AGE_SECONDS + 3600
        app_module.before_request()
        assert g.user is not None


def test_database_error_answers_503_page_and_keeps_the_session(monkeypatch):
    def _raise_db(*_a, **_kw):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    client = _logged_in_client()

    response = client.get("/dashboard")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.get_data(as_text=True)
    assert response.headers["Retry-After"]
    with client.session_transaction() as sess:
        assert sess["user_id"] == 1


def test_database_error_answers_503_json_on_api_routes(monkeypatch):
    def _raise_db(*_a, **_kw):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    client = _logged_in_client()

    response = client.get("/api/progress/1")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.get_json()["error"]
    with client.session_transaction() as sess:
        assert sess["user_id"] == 1


def test_database_error_does_not_start_a_login_round_trip(monkeypatch):
    def _raise_db(*_a, **_kw):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    client = _logged_in_client()

    assert client.get("/login").status_code == 503


def test_plain_logout_is_this_device_only(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers()
    monkeypatch.setattr(app_module, "execute_query", db)
    first = _logged_in_client()
    second = _logged_in_client()

    response = first.post("/logout", data={"csrf_token": "tok"})

    assert response.status_code == 302
    assert db.epoch_updates() == []
    assert not _is_logged_in(first)
    assert _is_logged_in(second)


def test_logout_everywhere_requires_login(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers()
    monkeypatch.setattr(app_module, "execute_query", db)
    client = flask_app.test_client()

    response = client.post("/account/logout-everywhere", data={"csrf_token": "x"})

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    assert db.epoch_updates() == []


def test_logout_everywhere_requires_csrf(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers()
    monkeypatch.setattr(app_module, "execute_query", db)
    client = _logged_in_client()

    assert client.post("/account/logout-everywhere", data={"csrf_token": "wrong"}).status_code == 400
    assert client.post("/account/logout-everywhere").status_code == 400
    assert db.epoch_updates() == []
    assert _is_logged_in(client)


def test_logout_everywhere_ends_every_session(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers()
    monkeypatch.setattr(app_module, "execute_query", db)
    here = _logged_in_client()
    elsewhere = _logged_in_client()

    response = here.post("/account/logout-everywhere", data={"csrf_token": "tok"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    assert len(db.epoch_updates()) == 1
    assert db.epoch == 1
    _assert_persistent(response)
    with here.session_transaction() as sess:
        assert "user_id" not in sess
        assert any("all devices" in msg for _cat, msg in sess["_flashes"])
    assert not _is_logged_in(here)
    assert not _is_logged_in(elsewhere)  # its cookie still says epoch 0


def test_session_created_after_logout_everywhere_carries_the_new_epoch(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers(epoch=1)
    monkeypatch.setattr(app_module, "execute_query", db)

    assert _is_logged_in(_logged_in_client(epoch=1))


def test_logout_everywhere_database_failure_keeps_the_session(monkeypatch):
    _reset_rate_limit()
    db = _FakeUsers()
    db.fail_updates = True
    monkeypatch.setattr(app_module, "execute_query", db)
    client = _logged_in_client()

    response = client.post("/account/logout-everywhere", data={"csrf_token": "tok"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/account/settings")
    with client.session_transaction() as sess:
        assert sess["user_id"] == 1
        assert any(cat == "error" for cat, _msg in sess["_flashes"])
    assert _is_logged_in(client)


def test_logout_everywhere_is_rate_limited(monkeypatch):
    db = _FakeUsers()
    monkeypatch.setattr(app_module, "execute_query", db)
    limiters = [lim for lim in flask_app.extensions["limiter"]]
    for lim in limiters:
        lim.enabled = True
        lim.reset()
    try:
        client = _logged_in_client()
        statuses = [client.post("/account/logout-everywhere", data={"csrf_token": "bad"}).status_code for _ in range(6)]
    finally:
        for lim in limiters:
            lim.reset()
            lim.enabled = False

    assert statuses == [400] * 5 + [429]


def test_account_settings_page_offers_log_out_everywhere(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    client = _logged_in_client()

    response = client.get("/account/settings")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'action="/account/logout-everywhere"' in html
    assert "data-confirm=" in html


def test_token_refresh_failure_keeps_session_and_records_whether_to_retry(monkeypatch):
    from oauthlib.oauth2.rfc6749.errors import InvalidGrantError

    user = {
        "id": 1,
        "wiki_username": "tester",
        "access_token": "old",
        "refresh_token": "ref",
        "token_expires_at": datetime.now(UTC) - timedelta(hours=1),
    }
    monkeypatch.setattr(app_module, "redis_client", None)
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [dict(user)])

    def run(exc):
        def _exchange(_refresh):
            raise exc

        monkeypatch.setattr(app_module, "_oauth_exchange", _exchange)
        with flask_app.test_request_context("/"):
            session["user_id"] = 1
            g.user = dict(user)
            assert app_module._get_valid_token() is None
            assert session["user_id"] == 1
            return g.token_refresh_transient

    assert run(InvalidGrantError()) is False  # revoked: log in again
    assert run(TimeoutError("slow")) is True  # transient: just retry


def test_token_unavailable_error_distinguishes_permanent_from_transient():
    with flask_app.test_request_context("/"):
        g.token_refresh_transient = False
        permanent = app_module._token_unavailable_error()
        g.token_refresh_transient = True
        transient = app_module._token_unavailable_error()
    assert permanent.status == 401
    assert transient.status == 503
    assert permanent.message != transient.message
    assert permanent.extra == {"reauth_url": "/login?reauth=1"}
    assert transient.extra == {}


def _write_sdc_with_refresh_failure(monkeypatch, *, transient):
    _reset_rate_limit()

    def execute(sql, params=None, fetch=True):
        if sql.startswith("SELECT id, wiki_user_id"):
            return _FakeUsers()(sql, params, fetch)
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql:
            return [{"write_cnt": 3, "removal_cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def failing_token():
        g.token_refresh_transient = transient
        return None

    monkeypatch.setattr(app_module, "execute_query", execute)
    monkeypatch.setattr(app_module, "_get_valid_token", failing_token)
    client = _logged_in_client()
    response = client.post("/api/write-sdc/1", data={"csrf_token": "tok"})
    with client.session_transaction() as sess:
        assert sess["user_id"] == 1  # neither outcome logs the user out
    return response


def test_write_sdc_asks_for_a_new_login_only_when_the_grant_is_gone(monkeypatch):
    response = _write_sdc_with_refresh_failure(monkeypatch, transient=False)

    assert response.status_code == 401
    body = response.get_json()
    assert "re-authorize" in body["error"]
    assert body["reauth_url"] == "/login?reauth=1"  # the page appends &next=


def test_write_sdc_answers_503_for_a_transient_refresh_failure(monkeypatch):
    response = _write_sdc_with_refresh_failure(monkeypatch, transient=True)

    assert response.status_code == 503
    body = response.get_json()
    assert "try again" in body["error"]
    assert "reauth_url" not in body  # re-authorizing would not help while Wikimedia is unreachable


# ---------------------------------------------------------------------------
# Session-loss log line
# ---------------------------------------------------------------------------


def _session_loss_records(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("session loss:")]


def test_session_loss_logged_when_no_cookie_was_sent(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", True)
    _reset_rate_limit()
    client = flask_app.test_client()

    with caplog.at_level("INFO"):
        client.get("/login", headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)"})

    [line] = _session_loss_records(caplog)
    assert "reason=no cookie sent" in line
    assert "had_cookie=False" in line
    assert "client=ios" in line


def test_session_loss_logged_when_the_cookie_holds_no_user(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", True)
    _reset_rate_limit()
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["csrf_token"] = "abc"

    with caplog.at_level("INFO"):
        client.get("/login")

    [line] = _session_loss_records(caplog)
    assert "reason=valid but empty" in line
    assert "had_cookie=True" in line


def test_session_loss_reasons_for_expired_and_forged_cookies(monkeypatch):
    serializer = app_module.app.session_interface.get_signing_serializer(app_module.app)
    cookie = serializer.dumps({"csrf_token": "abc"})

    with flask_app.test_request_context("/", headers={"Cookie": "session=not-a-valid-cookie"}):
        assert app_module._session_loss_reason() == "bad signature"

    monkeypatch.setitem(flask_app.config, "PERMANENT_SESSION_LIFETIME", timedelta(seconds=-5))
    with flask_app.test_request_context("/", headers={"Cookie": f"session={cookie}"}):
        assert app_module._session_loss_reason() == "expired"


def test_session_loss_not_logged_for_static_health_and_service_worker(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", True)

    with caplog.at_level("INFO"):
        for path in ("/static/app.js", "/health", "/sw.js", "/robots.txt", "/commons-thumb/x"):
            with flask_app.test_request_context(path):
                app_module._log_session_loss()

    assert _session_loss_records(caplog) == []


def test_session_loss_can_be_switched_off_and_is_sampled(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 1.0)
    with caplog.at_level("INFO"):
        monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", False)
        with flask_app.test_request_context("/login"):
            app_module._log_session_loss()
        monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", True)
        monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 0.0)
        with flask_app.test_request_context("/login"):
            app_module._log_session_loss()

    assert _session_loss_records(caplog) == []


def test_session_loss_not_logged_for_a_logged_in_request(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "_SESSION_LOSS_SAMPLE_RATE", 1.0)
    monkeypatch.setattr(app_module, "_LOG_SESSION_LOSS", True)
    monkeypatch.setattr(app_module, "execute_query", _FakeUsers())
    client = _logged_in_client()

    with caplog.at_level("INFO"):
        client.get("/login")

    assert _session_loss_records(caplog) == []


def test_client_bucket():
    assert app_module._client_bucket("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5)") == "ios"
    assert app_module._client_bucket("Mozilla/5.0 (Linux; Android 14; Pixel 8)") == "android"
    assert app_module._client_bucket("Mozilla/5.0 (Windows NT 10.0; Win64; x64)") == "desktop"
    assert app_module._client_bucket("Mozilla/5.0 (X11; Linux x86_64)") == "desktop"
    assert app_module._client_bucket("Googlebot/2.1 (+http://www.google.com/bot.html)") == "other"
    assert app_module._client_bucket("") == "other"


def test_requests_that_never_touch_the_session_get_no_cookie():
    """The persistence safety net must not hand a cookie to every crawler."""
    response = flask_app.test_client().get("/robots.txt")

    assert response.status_code == 200
    assert _session_cookies(response) == []


def test_anonymous_landing_page_and_leaderboard_get_no_cookie(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])

    for path in ("/", "/leaderboard"):
        response = flask_app.test_client().get(path)

        assert response.status_code == 200, path
        assert _session_cookies(response) == [], path


def test_after_request_makes_any_non_empty_session_persistent():
    """The safety net behind the explicit session.permanent calls: no writer can leave a session cookie."""
    with flask_app.test_request_context("/"):
        session["written_by_a_path_that_forgot"] = 1
        assert session.permanent is False

        app_module._keep_session_cookie_persistent(Response())

        assert session.permanent is True


def test_after_request_leaves_an_empty_session_alone():
    with flask_app.test_request_context("/"):
        app_module._keep_session_cookie_persistent(Response())

        assert not session
