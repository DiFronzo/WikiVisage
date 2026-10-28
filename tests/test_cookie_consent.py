"""Cookie consent: the /cookie-consent endpoint, the locale cookie flags and the banner markup."""

import pytest

from tests._app_preamble import app_module, flask_app


@pytest.fixture(autouse=True)
def production_mode(monkeypatch):
    """The Secure flag follows app.debug, which a developer's .env may switch on."""
    monkeypatch.setattr(app_module.app, "debug", False)


def _client_with_csrf(locale=None):
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess.permanent = True
        sess["csrf_token"] = "tok"
    if locale:
        client.set_cookie("locale", locale)
    return client


def _locale_cookies(response):
    return [h for h in response.headers.getlist("Set-Cookie") if h.startswith("locale=")]


def _post(client, **form):
    app_module.limiter.reset()
    return client.post("/cookie-consent", data={"csrf_token": "tok", **form})


def test_cookie_consent_requires_a_csrf_token():
    client = _client_with_csrf("nb")

    assert client.post("/cookie-consent", data={"choice": "accepted"}).status_code == 400
    assert client.post("/cookie-consent", data={"choice": "accepted", "csrf_token": "nope"}).status_code == 400


def test_cookie_consent_accepts_the_csrf_token_as_a_header():
    client = _client_with_csrf("nb")

    response = client.post("/cookie-consent", data={"choice": "accepted"}, headers={"X-CSRFToken": "tok"})

    assert response.status_code == 204


def test_cookie_consent_is_rejected_without_a_session_at_all():
    client = flask_app.test_client()

    assert client.post("/cookie-consent", data={"choice": "accepted", "csrf_token": ""}).status_code == 400


def test_accepting_makes_the_language_cookie_last_a_year():
    client = _client_with_csrf("nb")

    response = _post(client, choice="accepted")

    assert response.status_code == 204
    [cookie] = _locale_cookies(response)
    assert cookie.startswith("locale=nb;")
    assert "Max-Age=31536000" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    assert "Secure" in cookie


def test_declining_turns_the_language_cookie_session_scoped():
    client = _client_with_csrf("fr")

    response = _post(client, choice="declined")

    assert response.status_code == 204
    [cookie] = _locale_cookies(response)
    assert cookie.startswith("locale=fr;")
    assert "Max-Age" not in cookie
    assert "Expires" not in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie


def test_without_a_language_cookie_there_is_nothing_to_change():
    client = _client_with_csrf()

    for choice in ("accepted", "declined"):
        response = _post(client, choice=choice)
        assert response.status_code == 204
        assert _locale_cookies(response) == []


def test_an_unknown_language_cookie_is_never_echoed_back():
    client = _client_with_csrf("<script>")

    response = _post(client, choice="accepted")

    assert response.status_code == 204
    assert _locale_cookies(response) == []


def test_an_invalid_choice_is_rejected():
    client = _client_with_csrf("nb")

    assert _post(client, choice="maybe").status_code == 400
    assert _post(client).status_code == 400
    assert _post(client, choice="").status_code == 400


def test_cookie_consent_is_rate_limited():
    limiters = list(flask_app.extensions["limiter"])
    for lim in limiters:
        lim.enabled = True
        lim.reset()
    try:
        client = _client_with_csrf("nb")
        statuses = [
            client.post("/cookie-consent", data={"csrf_token": "tok", "choice": "bad"}).status_code for _ in range(21)
        ]
    finally:
        for lim in limiters:
            lim.reset()
            lim.enabled = False

    assert statuses == [400] * 20 + [429]


def test_set_language_cookie_flags():
    client = flask_app.test_client()

    remembered = client.get("/set-language/es")
    session_only = client.get("/set-language/es?nocookie=1")

    [kept] = _locale_cookies(remembered)
    assert "Max-Age=31536000" in kept and "Secure" in kept and "HttpOnly" in kept
    [dropped] = _locale_cookies(session_only)
    assert "Max-Age" not in dropped and "Expires" not in dropped
    assert "Secure" in dropped and "HttpOnly" in dropped and "SameSite=Lax" in dropped


def _rendered_landing_page(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])
    response = flask_app.test_client().get("/")
    assert response.status_code == 200
    return response.get_data(as_text=True)


def test_banner_describes_the_cookies_and_posts_the_choice(monkeypatch):
    html = _rendered_landing_page(monkeypatch)

    assert 'id="cookie-consent"' in html
    assert "session cookie is required" in html
    assert "language cookie" in html
    assert "local storage" in html
    assert '"/cookie-consent"' in html  # the URL the script posts the choice to
    assert "X-CSRFToken" in html


def test_footer_offers_cookie_settings_and_the_pet_never_uses_a_cookie(monkeypatch):
    html = _rendered_landing_page(monkeypatch)

    assert 'id="cookieSettings"' in html
    assert "cookieSettings" in html.split("removeItem('cookie_consent')")[0]
    assert 'document.cookie = "jimmy_pet=" +' not in html
    assert "jimmy_pet=; path=/; max-age=0" in html  # a leftover cookie is deleted
    assert 'getItem("jimmy_pet")' in html
    # An existing choice in the old cookie is carried over to local storage before it is deleted.
    assert "document.cookie.match(" in html
    assert "localStorage.setItem('jimmy_pet', decodeURIComponent(" in html
    assert html.index("localStorage.setItem('jimmy_pet'") < html.index("jimmy_pet=; path=/; max-age=0")


def test_banner_is_a_labelled_region_and_settings_focus_the_accept_button(monkeypatch):
    html = _rendered_landing_page(monkeypatch)

    assert 'id="cookie-consent" class="cookie-banner" role="region" aria-label="Cookie consent"' in html
    assert "document.getElementById('cookieAccept').focus()" in html


def test_locale_cookie_is_not_secure_in_local_debug_mode(monkeypatch):
    """OAUTHLIB_INSECURE_TRANSPORT dev runs are plain http, where a Secure cookie would be dropped."""
    monkeypatch.setattr(app_module.app, "debug", True)

    [cookie] = _locale_cookies(flask_app.test_client().get("/set-language/es"))

    assert "Secure" not in cookie


@pytest.mark.parametrize(
    ("debug", "lan_mode", "expected"),
    [
        (False, False, True),  # production
        (True, False, False),  # local debug over http
        (False, True, False),  # local-network dev mode over http
        (True, True, False),
    ],
)
def test_cookie_secure_requires_debug_off_and_no_lan_mode(monkeypatch, debug, lan_mode, expected):
    monkeypatch.setattr(app_module.app, "debug", debug)
    monkeypatch.setattr(app_module, "INSECURE_HTTP", lan_mode)

    assert app_module._cookie_secure() is expected


def test_lan_dev_mode_drops_secure_from_the_session_and_locale_cookies(monkeypatch):
    """A phone reaches the dev server over plain http, where browsers discard Secure cookies."""
    monkeypatch.setattr(app_module, "INSECURE_HTTP", True)
    client = _client_with_csrf("nb")

    response = _post(client, choice="accepted")

    session_cookie = next(h for h in response.headers.getlist("Set-Cookie") if h.startswith("session="))
    [locale_cookie] = _locale_cookies(response)
    assert "Secure" not in session_cookie
    assert "Secure" not in locale_cookie


@pytest.mark.parametrize(
    ("flag", "toolforge", "expected"),
    [
        ("1", None, True),
        ("true", None, True),
        ("", None, False),
        ("0", None, False),
        ("1", "https://api.svc.toolforge.org/", False),  # a stray flag in production is ignored
    ],
)
def test_lan_dev_mode_is_never_honoured_on_toolforge(monkeypatch, flag, toolforge, expected):
    monkeypatch.setenv("WIKIVISAGE_INSECURE_HTTP", flag)
    if toolforge:
        monkeypatch.setenv("TOOL_TOOLFORGE_API_URL", toolforge)
    else:
        monkeypatch.delenv("TOOL_TOOLFORGE_API_URL", raising=False)

    assert app_module._insecure_http_enabled() is expected


def _set_cookies(response, name="session"):
    return [h for h in response.headers.getlist("Set-Cookie") if h.startswith(f"{name}=")]


def test_anonymous_pages_do_not_set_a_session_cookie(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])

    for path in ("/", "/leaderboard"):
        response = flask_app.test_client().get(path)

        assert response.status_code == 200, path
        assert _set_cookies(response) == [], path
        assert 'var csrfToken = "";' in response.get_data(as_text=True), path


def test_banner_gets_a_csrf_token_once_there_is_a_language_cookie(monkeypatch):
    import re

    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [])
    client = flask_app.test_client()
    client.set_cookie("locale", "nb")

    response = client.get("/")

    assert response.status_code == 200
    match = re.search(r'var csrfToken = "([0-9a-f]+)";', response.get_data(as_text=True))
    assert match
    assert _set_cookies(response)  # the token lives in the session the GET just created
    app_module.limiter.reset()
    posted = client.post("/cookie-consent", data={"choice": "accepted", "csrf_token": match.group(1)})
    assert posted.status_code == 204


def test_banner_skips_the_post_without_a_token():
    from pathlib import Path

    source = Path(app_module.app.root_path, "templates", "base.html").read_text(encoding="utf-8")

    assert "if (!csrfToken) return;" in source
