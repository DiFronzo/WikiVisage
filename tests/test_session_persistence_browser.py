"""Session persistence in a real browser over real HTTP (the iOS failure mode).

The other browser tests route a mocked transport through the Flask test client. This one starts
the app on a real socket and loads its pages in a persistent browser profile, so the browser
itself (not Playwright's HTTP client) accepts the ``Set-Cookie`` header, stores the cookie and
brings it back after a full restart: exactly what a session-scoped cookie does not survive.
The OAuth callback then reads ``oauth_state`` back from the restored cookie, which is the part
of the login round trip that breaks when the cookie is dropped.
"""

import os
import sys
import threading
import time
from urllib.parse import urlparse

import pytest

from config import SESSION_IDLE_DAYS

if os.environ.get("WIKIVISAGE_BROWSER_TESTS") == "1":
    # An explicitly enabled browser run must fail rather than silently skip.
    from playwright import sync_api as playwright
else:
    playwright = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server

pytestmark = pytest.mark.browser


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    monkeypatch.setenv("WIKIVISAGE_REDIS_URL", "memory://")


@pytest.fixture(scope="module", params=["chromium", "webkit"])
def browser_type(request):
    """A Playwright browser type; the test launches its own persistent context with it."""
    with playwright.sync_playwright() as runtime:
        yield getattr(runtime, request.param), request.param


@pytest.fixture
def live_server(flask_app, monkeypatch):
    """Run the real app on a loopback socket; no database or outbound network is available."""
    app_module = sys.modules["app"]
    # Plain http on loopback, so the browser accepts the cookie without a TLS certificate.
    monkeypatch.setattr(app_module, "INSECURE_HTTP", True)
    # Safety net: the callback fails before any token request (the URL carries no code), but
    # should that change it must still never reach Wikimedia.
    monkeypatch.setattr(app_module, "OAUTH_TOKEN_URL", "http://127.0.0.1:1/access_token")
    # Anonymous pages read worker/LiftWing status; answer every query with "no rows".
    monkeypatch.setattr(app_module, "execute_query", lambda *args, **kwargs: ())

    server = make_server("127.0.0.1", 0, flask_app, threaded=True)
    base = f"http://127.0.0.1:{server.server_port}"
    # /login redirects to the OAuth provider; keep that hop on loopback (Playwright does not
    # route the targets of redirects, so a stub for the real host would never be hit).
    monkeypatch.setattr(app_module, "OAUTH_AUTHORIZE_URL", f"{base}/robots.txt")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _session_cookie(context, url):
    return {cookie["name"]: cookie for cookie in context.cookies(url)}["session"]


def test_session_cookie_survives_a_browser_restart_and_the_oauth_state_round_trips(browser_type, live_server, tmp_path):
    browser, browser_name = browser_type
    profile = tmp_path / browser_name

    context = browser.launch_persistent_context(str(profile), headless=True, service_workers="block")
    try:
        # Nothing but the loopback server may be reached.
        context.route(lambda url: urlparse(url).hostname != "127.0.0.1", lambda route: route.abort())
        page = context.new_page()
        page.goto(f"{live_server}/login")
        assert urlparse(page.url).path == "/robots.txt"  # followed the redirect to the "provider"

        session_cookie = _session_cookie(context, live_server)
        # A session-scoped cookie has expires == -1 and would not come back after the restart.
        assert session_cookie["expires"] > time.time() + (SESSION_IDLE_DAYS - 1) * 86400
        assert session_cookie["httpOnly"] is True
        assert session_cookie["sameSite"] == "Lax"
        value = session_cookie["value"]
    finally:
        context.close()

    context = browser.launch_persistent_context(str(profile), headless=True, service_workers="block")
    try:
        assert _session_cookie(context, live_server)["value"] == value

        # The callback pops oauth_state from the restored session and flashes "Authentication
        # failed" because the request carries no authorization code. "Invalid OAuth state" would
        # mean the cookie (and with it the login round trip) was lost.
        page = context.new_page()
        page.goto(f"{live_server}/auth/callback")
        body = page.content()
        assert "Authentication failed" in body
        assert "Invalid OAuth state" not in body
    finally:
        context.close()
