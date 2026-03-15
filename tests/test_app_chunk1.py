import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests
from flask import Response, g, session

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
    with app.test_request_context("/"):
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

    with app.test_request_context("/"):
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

    with app.test_request_context("/"):
        session["user_id"] = 1
        app_module.before_request()
        assert g.user is None
        assert "user_id" not in session


def test_before_request_handles_database_error_and_clears_session(monkeypatch):
    def _raise_db(*_args, **_kwargs):
        raise app_module.DatabaseError("db failure")

    monkeypatch.setattr(app_module, "execute_query", _raise_db)
    with app.test_request_context("/"):
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

    with app.test_request_context("/"):
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
    with app.test_request_context("/"):
        data = app_module.inject_csrf_token()
        assert "csrf_token" in data
        token = data["csrf_token"]()
        assert isinstance(token, str)
        assert len(token) == 64


def test_set_security_headers_sets_all_required_headers():
    response = app.make_response(("ok", 200))
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
    with app.test_request_context("/protected?x=1"):
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
    with app.test_request_context("/protected"):
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
    with app.test_request_context("/"):
        g.user = None
        assert app_module._get_valid_token() is None


def test_get_valid_token_clears_session_when_refresh_fails(monkeypatch):
    with app.test_request_context("/"):
        g.user = {"access_token": "old"}
        session["user_id"] = 1
        monkeypatch.setattr(app_module, "_refresh_access_token", lambda _user: None)

        assert app_module._get_valid_token() is None
        assert "user_id" not in session


def test_get_valid_token_updates_g_user_and_returns_token(monkeypatch):
    with app.test_request_context("/"):
        g.user = {"access_token": "old"}
        refreshed = {"access_token": "new"}
        monkeypatch.setattr(app_module, "_refresh_access_token", lambda _user: refreshed)

        token = app_module._get_valid_token()
        assert token == "new"
        assert g.user is refreshed


def test_csrf_token_generates_new_token():
    with app.test_request_context("/"):
        token = app_module._csrf_token()
        assert isinstance(token, str)
        assert len(token) == 64
        assert session["csrf_token"] == token


def test_csrf_token_returns_existing_token():
    with app.test_request_context("/"):
        session["csrf_token"] = "existing"
        assert app_module._csrf_token() == "existing"


def test_set_language_valid_lang_sets_cookie(monkeypatch):
    app.config["TESTING"] = True
    client = app.test_client()

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
    app.config["TESTING"] = True
    client = app.test_client()

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
    app.config["TESTING"] = True
    client = app.test_client()

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
    app.config["TESTING"] = True
    client = app.test_client()

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
    with app.test_request_context("/", headers={"Cookie": "locale=nb", "Accept-Language": "es"}):
        assert app_module.get_locale() == "nb"


def test_get_locale_falls_back_to_accept_language():
    with app.test_request_context("/", headers={"Accept-Language": "es"}):
        assert app_module.get_locale() == "es"


def test_get_locale_defaults_to_en_when_no_match():
    with app.test_request_context("/", headers={"Accept-Language": "zz-ZZ"}):
        assert app_module.get_locale() == "en"
