"""The project page stat boxes must not clip or break their labels in any locale.

Two regressions live here. Long uppercase translations ("CORRESPONDANCES CIBLES",
"UKLASSIFISERT") used to run past a six-column box, where `overflow: hidden` cut
them; telling the browser to break the word instead produced "UNCLASSIFIE D" and
"AN-SIKTER". The row is three across now, which leaves the longest translation
whole, and the label forbids breaks inside a word, so a string that no longer fits
fails these tests instead of hyphenating. Five-digit values are checked too.
"""

import importlib
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import pytest

if os.environ.get("WIKIVISAGE_BROWSER_TESTS") == "1":
    # An explicitly enabled browser run must fail rather than silently skip.
    from playwright import sync_api as playwright
else:
    playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

_PROJECT = {
    "id": 1,
    "user_id": 1,
    "label": "Portrett",
    "wikidata_qid": "Q42",
    "commons_category": "Portraits",
    "p18_thumb_url": None,
    "status": "completed",
    "min_confirmed": 5,
    "distance_threshold": 0.6,
    "images_total": 1200,
    "images_processed": 1188,
    "sdc_write_requested": 0,
    "sdc_write_error": None,
    "completion_reason": None,
    "prefilter_mode": "off",
}
# Five-digit counts: the three-across row leaves about 230px of content per box,
# enough for the longest word at 12px, and only just for a five-digit number.
_STATS = {
    "total_faces": 12345,
    "confirmed_matches": 1188,
    "confirmed_non_matches": 2048,
    "unclassified": 9109,
    "human_confirmed": 1188,
    "model_matches": 77,
    "sdc_written": 1111,
    "sdc_pending": 77,
    "sdc_removal_pending_faces": 3,
    "sdc_removal_pending": 0,
    "by_human": 1188,
    "by_model": 2048,
    "by_bootstrap": 33,
}
# What the "Unclassified" tile reads in each locale; a harness that loses the
# locale renders English and would pass an overflow check for the wrong reason.
_UNCLASSIFIED = {"en": "UNCLASSIFIED", "nb": "UKLASSIFISERT", "es": "SIN CLASIFICAR", "fr": "NON CLASSÉS"}

_OVERFLOW_JS = """() => {
    const out = [];
    for (const box of document.querySelectorAll('.stats-grid .stat-box')) {
        const b = box.getBoundingClientRect();
        for (const el of box.querySelectorAll('.stat-value, .stat-label')) {
            const r = el.getBoundingClientRect();
            if (el.scrollWidth > el.clientWidth + 1) out.push([el.className, el.textContent.trim(), 'scroll']);
            if (r.right > b.right + 1 || r.left < b.left - 1 || r.bottom > b.bottom + 1) {
                out.push([el.className, el.textContent.trim(), 'outside']);
            }
        }
    }
    return out;
}"""

# The rules that keep a label whole: three columns (so the longest translation
# fits), and no automatic hyphenation or mid-word wrapping.
_LAYOUT_JS = """() => {
    const grid = document.querySelector('.stats-grid');
    const label = grid.querySelector('.stat-label');
    const style = getComputedStyle(label);
    // Cluster the tops: WebKit lays rows out on sub-pixel boundaries, so rounding
    // each top on its own splits one row into two.
    const tops = [];
    for (const box of grid.querySelectorAll('.stat-box')) {
        const top = box.getBoundingClientRect().top;
        if (!tops.some((seen) => Math.abs(seen - top) < 4)) tops.push(top);
    }
    return {
        columns: getComputedStyle(grid).gridTemplateColumns.split(' ').length,
        rows: tops.length,
        tops: [...grid.querySelectorAll('.stat-box')].map((b) => +b.getBoundingClientRect().top.toFixed(1)),
        boxes: grid.querySelectorAll('.stat-box').length,
        hyphens: style.hyphens,
        overflowWrap: style.overflowWrap,
        wordBreak: style.wordBreak,
    };
}"""


@pytest.fixture(scope="module", params=["chromium", "webkit"])
def browser(request):
    with playwright.sync_playwright() as runtime:
        instance = getattr(runtime, request.param).launch()
        yield instance
        instance.close()


def _open_project_page(browser, flask_app, monkeypatch, locale, width):
    module = importlib.import_module("app")

    def query(sql, params=None, fetch=True):
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
        return ()

    monkeypatch.setattr(module, "execute_query", query)
    monkeypatch.setattr(module, "get_project_for_actor", lambda *args: _PROJECT.copy())
    monkeypatch.setattr(module, "_get_face_stats", lambda *args: _STATS.copy())
    client = flask_app.test_client()
    client.set_cookie("locale", locale)  # the app reads this cookie first, in every engine
    with client.session_transaction() as session:
        session["user_id"] = 1

    def serve(route):
        parsed = urlparse(route.request.url)
        if parsed.hostname != "wikivisage.test":
            route.abort()
            return
        if parsed.path.startswith("/commons-thumb/"):
            route.fulfill(content_type="image/svg+xml", body='<svg xmlns="http://www.w3.org/2000/svg"/>')
            return
        if parsed.path.startswith("/api/"):
            route.fulfill(json={"faces": [], "page": 1, "total_pages": 1, "total": 0})
            return
        response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
        route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)

    context = browser.new_context(service_workers="block", viewport={"width": width, "height": 1000})
    context.route("**/*", serve)
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.goto("https://wikivisage.test/project/1")
    page.wait_for_selector(".stats-grid .stat-box")
    # Each tile rises with a staggered `stat-entrance` animation (up to 0.3s of
    # delay plus 0.6s), so the first row is briefly offset by up to 20px and a
    # measurement then finds more "rows" than the grid has. Finish them first;
    # the decorative looping ones (scanline, readout) cannot be finished.
    page.evaluate("() => document.getAnimations().forEach((a) => { try { a.finish(); } catch (e) {} })")
    return page, context


@pytest.mark.parametrize("locale", ["en", "nb", "es", "fr"])
# 360 is the phone layout: two columns, where the label follows its box (cqi) so
# "CORRESPONDANCES" still fits at 108px of content on a 320px screen.
@pytest.mark.parametrize("width", [1280, 900, 360])
def test_stat_boxes_do_not_clip_their_content(browser, flask_app, monkeypatch, locale, width):
    page, context = _open_project_page(browser, flask_app, monkeypatch, locale, width)
    try:
        unclassified = page.locator("#statUnclassified + .stat-label")
        assert unclassified.inner_text().strip().upper() == _UNCLASSIFIED[locale], "the locale did not apply"
        assert page.evaluate(_OVERFLOW_JS) == []
        assert not page.errors
    finally:
        context.close()


@pytest.mark.parametrize("locale", ["en", "nb", "es", "fr"])
@pytest.mark.parametrize("width", [1280, 900])
def test_labels_are_never_broken_inside_a_word(browser, flask_app, monkeypatch, locale, width):
    """Three tiles per row, and no hyphenation: "AN-SIKTER" is what this prevents."""
    page, context = _open_project_page(browser, flask_app, monkeypatch, locale, width)
    try:
        layout = page.evaluate(_LAYOUT_JS)
        assert layout["boxes"] == 6, "the primary stats row lost a tile"
        assert layout["columns"] == 3, layout
        assert layout["rows"] == 2, layout  # no ragged last row: 6 tiles fill 3x2
        assert layout["hyphens"] in ("manual", "none"), layout
        assert layout["overflowWrap"] == "normal", layout
        assert layout["wordBreak"] == "normal", layout
        assert not page.errors
    finally:
        context.close()
