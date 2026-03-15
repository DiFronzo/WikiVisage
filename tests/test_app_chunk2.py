import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

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
