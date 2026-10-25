"""Tests for the web app: auth."""

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
        # Security: the pre-login CSRF token must NOT be preserved across login.
        # Preserving it would let an attacker who fixated a session before login
        # retain a valid CSRF token afterwards. See _is_safe_url tests + login handler.
        assert sess["csrf_token"] != "keep-me"
        assert isinstance(sess["csrf_token"], str) and len(sess["csrf_token"]) >= 32


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
