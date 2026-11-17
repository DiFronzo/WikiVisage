"""Exercise the nav language picker in real browsers.

Picking a language applies it immediately (no Apply button); a keyboard user who
arrows through the option list must not be sent to a new page on every step, so
the change waits for Enter or for the select to lose focus.
"""

import importlib
import os
import re
from urllib.parse import urlparse

import pytest

if os.environ.get("WIKIVISAGE_BROWSER_TESTS") == "1":
    # An explicitly enabled browser run must fail rather than silently skip.
    from playwright import sync_api as playwright
else:
    playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_LANDING = "<html><body><h1>WikiVisage</h1></body></html>"


@pytest.fixture(scope="module", params=["chromium", "webkit"])
def browser(request):
    with playwright.sync_playwright() as runtime:
        instance = getattr(runtime, request.param).launch()
        yield instance
        instance.close()


def _open_landing(browser, flask_app, monkeypatch):
    """Load the real landing page (which renders the nav picker) over a mocked transport."""
    module = importlib.import_module("app")
    # The landing page asks for the worker heartbeat and the LiftWing status; no database here.
    monkeypatch.setattr(module, "execute_query", lambda *args, **kwargs: ())
    client = flask_app.test_client()
    visited = []

    def serve(route):
        parsed = urlparse(route.request.url)
        if parsed.hostname != "wikivisage.test":
            route.abort()
            return
        if parsed.path.startswith("/set-language/"):
            # The real route redirects back; record the visit and answer with a marker page.
            visited.append(parsed.path + ("?" + parsed.query if parsed.query else ""))
            route.fulfill(status=200, content_type="text/html", body=_LANDING)
            return
        response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
        route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)

    context = browser.new_context(service_workers="block")
    context.route("**/*", serve)
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.goto("https://wikivisage.test/")
    return page, visited, context


def test_picking_a_language_switches_it_without_an_apply_button(browser, flask_app, monkeypatch):
    page, visited, context = _open_landing(browser, flask_app, monkeypatch)
    try:
        assert page.locator("#langApply").count() == 0, "the Apply button is gone"
        assert page.locator("#langSelect").is_visible()

        page.select_option("#langSelect", "nb")
        playwright.expect(page).to_have_url(re.compile(r"/set-language/nb"))
        assert visited[0].startswith("/set-language/nb")
        assert not page.errors
    finally:
        context.close()


def test_arrowing_through_the_list_waits_for_a_commit(browser, flask_app, monkeypatch):
    page, visited, context = _open_landing(browser, flask_app, monkeypatch)
    try:
        page.locator("#langSelect").focus()

        page.dispatch_event("#langSelect", "keydown", {"key": "ArrowDown"})
        page.select_option("#langSelect", "es")  # what the arrow key does to the value
        page.wait_for_timeout(300)
        assert visited == [], "arrowing must not navigate"
        assert "/set-language/" not in page.url

        # Enter commits the choice the keyboard user landed on.
        page.dispatch_event("#langSelect", "keydown", {"key": "Enter"})
        playwright.expect(page).to_have_url(re.compile(r"/set-language/es"))
        assert not page.errors
    finally:
        context.close()


def test_leaving_the_select_commits_a_keyboard_choice(browser, flask_app, monkeypatch):
    page, visited, context = _open_landing(browser, flask_app, monkeypatch)
    try:
        page.locator("#langSelect").focus()

        page.dispatch_event("#langSelect", "keydown", {"key": "ArrowDown"})
        page.select_option("#langSelect", "fr")
        page.wait_for_timeout(300)
        assert visited == []

        page.locator("#langSelect").blur()  # tabbing away applies it, so the page matches the picker
        playwright.expect(page).to_have_url(re.compile(r"/set-language/fr"))
        assert not page.errors
    finally:
        context.close()
