"""Exercise the Classification Results gallery thumbnails (zoom cap, face marker) in real browsers."""

import importlib
import os
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, quote, unquote, urlparse

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
    "label": "Portrait",
    "wikidata_qid": "Q42",
    "commons_category": "Portraits",
    "p18_thumb_url": "/commons-thumb/Portrait.jpg?width=120",
    "status": "completed",
    "min_confirmed": 5,
    "distance_threshold": 0.6,
    "images_total": 1,
    "images_processed": 1,
    "sdc_write_requested": 0,
    "sdc_write_error": None,
    "completion_reason": None,
}
_STATS = {
    "total_faces": 6,
    "confirmed_matches": 6,
    "confirmed_non_matches": 0,
    "unclassified": 0,
    "human_confirmed": 5,
    "model_matches": 1,
    "sdc_written": 0,
    "sdc_pending": 1,
    "sdc_removal_pending_faces": 0,
    "sdc_removal_pending": 0,
    "by_human": 5,
    "by_model": 1,
    "by_bootstrap": 0,
}

# In-page: how far the marker is from where the face box should be, in CSS px.
_MARKER_ERROR_JS = """(index) => {
    const thumb = document.querySelectorAll('.result-thumb')[index];
    const img = thumb.querySelector('img');
    const marker = thumb.querySelector('.face-marker');
    if (!marker || !img.naturalWidth) return Infinity;
    const t = thumb.getBoundingClientRect();
    const i = img.getBoundingClientRect();
    const d = thumb.dataset;
    // Rendered image box: object-fit: contain letterboxes, a cropped image fills its own box.
    let left = i.left, top = i.top, width = i.width, height = i.height;
    if (!img.classList.contains('face-cropped')) {
        const s = Math.min(i.width / img.naturalWidth, i.height / img.naturalHeight);
        width = img.naturalWidth * s; height = img.naturalHeight * s;
        left = i.left + (i.width - width) / 2; top = i.top + (i.height - height) / 2;
    }
    const sx = width / +d.detectionWidth, sy = height / +d.detectionHeight;
    const m = marker.getBoundingClientRect();
    const want = {
        left: left + d.bboxLeft * sx, top: top + d.bboxTop * sy,
        width: (d.bboxRight - d.bboxLeft) * sx, height: (d.bboxBottom - d.bboxTop) * sy,
    };
    // Very small faces get a marker of at least MIN px around the same centre.
    const MIN = 10;
    const wantW = Math.max(want.width, MIN), wantH = Math.max(want.height, MIN);
    const cx = want.left + want.width / 2, cy = want.top + want.height / 2;
    return Math.max(Math.abs(m.left + m.width / 2 - cx), Math.abs(m.top + m.height / 2 - cy),
                    Math.abs(m.width - wantW), Math.abs(m.height - wantH));
}"""

_ALL_LOADED_JS = """() => {
    const imgs = [...document.querySelectorAll('.result-thumb img')];
    return imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0);
}"""


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    monkeypatch.setenv("WIKIVISAGE_REDIS_URL", "memory://")


@pytest.fixture(scope="module", params=["chromium", "webkit"])
def browser(request):
    with playwright.sync_playwright() as runtime:
        instance = getattr(runtime, request.param).launch()
        yield instance
        instance.close()


def direct_thumb_url(file_title, width):
    """A direct Commons thumbnail URL, shaped like the server's commons_thumb_url()."""
    name = file_title.replace("File:", "").replace(" ", "_")
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/{quote(name, safe='')}/{width}px-{quote(name, safe='')}"


def make_face(face_id, file_title="File:Photo.jpg", bbox=(100, 200, 160, 260), det=(960, 1440), **extra):
    """A gallery JSON row. ``bbox`` is (left, top, right, bottom) in detection space.

    ``thumb_urls`` follows the server: direct URLs for the standard steps above 330 px that
    are at most min(960, detection width).
    """
    left, top, right, bottom = bbox
    steps = [s for s in (500, 960) if det[0] > 0 and s <= min(960, det[0])]
    face = {
        "id": face_id,
        "image_id": 100 + face_id,
        "file_title": file_title,
        "thumb_url": "/commons-thumb/" + file_title.replace("File:", "") + "?width=330",
        "thumb_urls": {str(s): direct_thumb_url(file_title, s) for s in steps},
        "bbox_top": top,
        "bbox_right": right,
        "bbox_bottom": bottom,
        "bbox_left": left,
        "detection_width": det[0],
        "detection_height": det[1],
        "result": "match",
        "is_target": 1,
        "classified_by": "model",
        "confidence": 0.2,
        "sdc_written": 0,
        "sdc_pending": 0,
        "sdc_removal": 0,
        "bootstrapped": 0,
    }
    face.update(extra)
    return face


@pytest.fixture
def open_gallery(browser, flask_app, monkeypatch):
    """Factory: open the real project page with mocked gallery rows and generated thumbnails.

    ``originals`` maps a file title to the (width, height) of the original Commons file; a
    thumbnail request answers with an SVG of that width (never wider than the original, like
    Commons) and the matching height. Returns ``(page, requests)`` where ``requests`` lists
    ``(file name, width)`` for every thumbnail fetched.
    """
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
        if "f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1 if "FROM faces f" in sql else 0}]
        return ()

    monkeypatch.setattr(module, "execute_query", query)
    monkeypatch.setattr(module, "get_project_for_actor", lambda *args: _PROJECT.copy())
    monkeypatch.setattr(module, "_get_face_stats", lambda *args: _STATS.copy())
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1

    contexts = []

    def open_page(faces, originals, viewport=None, dpr=1):
        requests = []
        direct = []  # thumbnails fetched straight from upload.wikimedia.org
        proxied = []  # thumbnails fetched through the app's /commons-thumb proxy

        def serve(route):
            parsed = urlparse(route.request.url)
            thumb = None
            if parsed.hostname == "upload.wikimedia.org":
                # Direct Commons thumbnail URL: .../thumb/x/xy/<name>/<width>px-<name>
                parts = parsed.path.split("/")
                match = re.match(r"(\d+)px-", parts[-1])
                thumb = (unquote(parts[-2]), int(match.group(1))) if match else None
                if thumb is None:
                    route.abort()
                    return
            elif parsed.hostname != "wikivisage.test":
                route.abort()
                return
            elif parsed.path.startswith("/commons-thumb/"):
                thumb = (unquote(parsed.path[len("/commons-thumb/") :]), int(parse_qs(parsed.query)["width"][0]))
            if thumb is not None:
                name, width = thumb
                if "File:" + name not in originals:  # e.g. the project's P18 portrait
                    route.fulfill(
                        content_type="image/svg+xml",
                        body='<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>',
                    )
                    return
                orig_w, orig_h = originals["File:" + name]
                requests.append((name, width))
                (direct if parsed.hostname == "upload.wikimedia.org" else proxied).append((name, width))
                if width > orig_w:
                    route.fulfill(status=429, body="thumbnail wider than the original")
                    return
                height = round(orig_h * width / orig_w)
                svg = (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                    f'<rect width="{width}" height="{height}" fill="#667"/></svg>'
                )
                route.fulfill(content_type="image/svg+xml", body=svg)
            elif parsed.path == "/api/project/1/gallery":
                route.fulfill(json={"faces": faces, "page": 1, "total_pages": 1, "total": len(faces)})
            else:
                response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
                route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)

        context = browser.new_context(
            viewport=viewport or {"width": 1280, "height": 1000}, device_scale_factor=dpr, service_workers="block"
        )
        contexts.append(context)
        context.route("**/*", serve)
        page = context.new_page()
        page.errors = []
        page.direct_thumbs = direct
        page.proxied_thumbs = proxied
        page.on("pageerror", lambda error: page.errors.append(str(error)))
        page.goto("https://wikivisage.test/project/1")
        page.wait_for_selector(".result-thumb img")
        page.wait_for_function(_ALL_LOADED_JS)
        page.wait_for_load_state("networkidle")
        page.wait_for_function(_ALL_LOADED_JS)
        return page, requests

    yield open_page
    for context in contexts:
        context.close()


def device_upscale(page, index=0):
    """Device pixels drawn per source pixel for the card's image (object-fit: contain letterboxes)."""
    return page.evaluate(
        """(index) => {
            const img = document.querySelectorAll('.result-thumb img')[index];
            const r = img.getBoundingClientRect();
            const scale = img.classList.contains('face-cropped')
                ? r.width / img.naturalWidth
                : Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
            return scale * window.devicePixelRatio;
        }""",
        index,
    )


def max_upscale(page):
    return page.evaluate("MAX_DEVICE_UPSCALE")


def is_cropped(page, index=0):
    return page.evaluate(
        "(i) => document.querySelectorAll('.result-thumb img')[i].classList.contains('face-cropped')", index
    )


def widths(requests, name):
    return [w for n, w in requests if n == name]


@pytest.mark.parametrize("dpr", [1, 2])
def test_tiny_face_is_capped_and_asks_for_one_larger_source(open_gallery, dpr):
    face = make_face(1, bbox=(470, 700, 490, 720))
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)}, dpr=dpr)
    assert widths(requests, "Photo.jpg") == [330, 960]
    assert page.evaluate("document.querySelector('.result-thumb img').naturalWidth") == 960
    assert is_cropped(page)
    assert device_upscale(page) <= max_upscale(page) + 0.01
    assert page.evaluate(_MARKER_ERROR_JS, 0) < 1.5
    assert not page.errors


def test_small_original_never_requests_more_than_it_has(open_gallery):
    face = make_face(1, bbox=(140, 190, 155, 205), det=(300, 400), thumb_url="/commons-thumb/Photo.jpg?width=250")
    page, requests = open_gallery([face], {"File:Photo.jpg": (300, 400)}, dpr=2)
    assert widths(requests, "Photo.jpg") == [250]
    assert device_upscale(page) <= max_upscale(page) + 0.01
    assert page.evaluate(_MARKER_ERROR_JS, 0) < 1.5
    assert not page.errors


def test_two_faces_each_row_marks_its_own_face(open_gallery):
    original = {"File:Group.jpg": (2000, 1333)}
    left_face = make_face(1, "File:Group.jpg", bbox=(100, 250, 180, 330), det=(960, 640))
    right_face = make_face(2, "File:Group.jpg", bbox=(700, 250, 780, 330), det=(960, 640))
    page, _ = open_gallery([left_face, right_face], original)
    for index in (0, 1):
        assert page.evaluate(_MARKER_ERROR_JS, index) < 1.5
    # The marker is the only thing telling the two cards apart: it moves with the row.
    centres = page.evaluate(
        """() => [...document.querySelectorAll('.result-thumb')].map(t => {
            const img = t.querySelector('img').getBoundingClientRect();
            const m = t.querySelector('.face-marker').getBoundingClientRect();
            return (m.left + m.width / 2 - img.left) / img.width;
        })"""
    )
    assert abs(centres[0] - centres[1]) > 0.05
    for index in (0, 1):
        inside = page.evaluate(
            """(i) => {
                const t = document.querySelectorAll('.result-thumb')[i];
                const a = t.getBoundingClientRect(), m = t.querySelector('.face-marker').getBoundingClientRect();
                return m.left >= a.left - 1 && m.top >= a.top - 1 && m.right <= a.right + 1 && m.bottom <= a.bottom + 1;
            }""",
            index,
        )
        assert inside
    assert not page.errors


def test_aspect_mismatch_shows_the_plain_image_without_marker(open_gallery):
    # Detection saw 600x800 but the thumbnail is landscape (an EXIF-rotated original).
    face = make_face(1, bbox=(200, 300, 300, 400), det=(600, 800))
    page, _ = open_gallery([face], {"File:Photo.jpg": (800, 600)})
    assert not is_cropped(page)
    assert page.locator(".face-marker").count() == 0
    assert page.evaluate("document.querySelector('.result-thumb img').getAttribute('style')") in (None, "")
    assert not page.errors


def test_legacy_row_without_detection_size_shows_the_plain_image(open_gallery):
    face = make_face(1, det=(0, 0))
    page, _ = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    assert not is_cropped(page)
    assert page.locator(".face-marker").count() == 0


def test_resize_relayouts_and_the_marker_stays_on_the_face(open_gallery):
    face = make_face(1, bbox=(470, 700, 530, 760))
    page, _ = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    before = page.evaluate("document.querySelector('.result-thumb img').style.width")
    page.set_viewport_size({"width": 520, "height": 900})
    page.wait_for_function("(before) => document.querySelector('.result-thumb img').style.width !== before", arg=before)
    page.wait_for_function(f"({_MARKER_ERROR_JS})(0) < 1.5")
    assert is_cropped(page)
    assert device_upscale(page) <= max_upscale(page) + 0.01
    assert not page.errors


def test_bootstrap_row_is_not_zoomed_but_is_marked(open_gallery):
    face = make_face(1, bbox=(470, 700, 490, 720), classified_by="bootstrap")
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    assert not is_cropped(page)
    assert widths(requests, "Photo.jpg") == [330]
    assert page.locator(".face-marker").count() == 1
    assert page.evaluate(_MARKER_ERROR_JS, 0) < 1.5
    assert not page.errors


def test_large_single_portrait_hides_the_marker(open_gallery):
    face = make_face(1, bbox=(100, 40, 860, 600), det=(960, 640))
    page, _ = open_gallery([face], {"File:Photo.jpg": (2000, 1333)})
    assert page.locator(".face-marker").count() == 0


def test_marker_is_decorative_and_does_not_block_the_card_buttons(open_gallery):
    face = make_face(1, bbox=(470, 700, 490, 720))
    page, _ = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    marker = page.locator(".face-marker")
    assert marker.get_attribute("aria-hidden") == "true"
    assert marker.evaluate("m => getComputedStyle(m).pointerEvents") == "none"
    assert marker.evaluate("m => m.textContent") == ""
    page.locator(".result-card").hover()
    page.locator(".result-actions button").nth(2).click()
    page.wait_for_function("document.getElementById('bboxModal').classList.contains('active')")


def test_failed_larger_source_falls_back_to_the_first_thumbnail(open_gallery):
    # The original claims to be 2000 wide, but Commons answers 429 for width=960.
    face = make_face(1, bbox=(470, 700, 490, 720))
    page, requests = open_gallery([face], {"File:Photo.jpg": (500, 750)})
    assert widths(requests, "Photo.jpg")[:2] == [330, 960]
    page.wait_for_function("document.querySelector('.result-thumb img').getAttribute('src').endsWith('width=330')")
    page.wait_for_function(_ALL_LOADED_JS)
    assert page.evaluate("document.querySelector('.result-thumb img').naturalWidth") == 330


def test_compute_face_view_applies_the_cap_at_dpr_1_and_2(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (2000, 3000)})
    result = page.evaluate(
        """() => [1, 2].map(dpr => {
            const v = computeFaceView({left: 470, top: 700, right: 490, bottom: 720},
                                      960, 1440, 960, 1440, 260, 180, dpr);
            return {mode: v.mode, scale: v.imgWidth / 960, cap: MAX_DEVICE_UPSCALE / dpr,
                    need: v.needNatW};
        })"""
    )
    for entry in result:
        assert entry["mode"] == "crop"
        assert entry["scale"] == pytest.approx(entry["cap"])
    assert result[0]["cap"] == pytest.approx(2 * result[1]["cap"])
    assert result[0]["need"] > 960


def test_compute_face_view_clamps_so_the_image_covers_the_card(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (2000, 3000)})
    corners = page.evaluate(
        """() => {
            const view = b => computeFaceView(b, 960, 1440, 960, 1440, 260, 180, 1);
            return {
                topLeft: view({left: 0, top: 0, right: 20, bottom: 20}),
                bottomRight: view({left: 940, top: 1420, right: 960, bottom: 1440}),
            };
        }"""
    )
    top_left, bottom_right = corners["topLeft"], corners["bottomRight"]
    assert top_left["imgLeft"] == 0 and top_left["imgTop"] == 0
    assert bottom_right["imgLeft"] == pytest.approx(260 - bottom_right["imgWidth"])
    assert bottom_right["imgTop"] == pytest.approx(180 - bottom_right["imgHeight"])
    for view in (top_left, bottom_right):
        box = view["box"]
        assert box["left"] >= -0.01 and box["top"] >= -0.01
        assert box["left"] + box["width"] <= 260.01 and box["top"] + box["height"] <= 180.01


def test_compute_face_view_falls_back_to_contain_or_null(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (2000, 3000)})
    out = page.evaluate(
        """() => {
            const bbox = {left: 40, top: 40, right: 60, bottom: 60};
            const box100 = computeFaceView(bbox, 100, 100, 100, 100, 260, 180, 2);   // cap below full cover
            const bigFace = computeFaceView({left: 10, top: 10, right: 950, bottom: 1400},
                                            960, 1440, 330, 495, 260, 180, 1);          // zoom < 1.1
            return {
                capBelowCover: box100.mode,
                smallZoom: bigFace.mode,
                zeroDetection: computeFaceView(bbox, 0, 0, 100, 100, 260, 180, 1),
                degenerate: computeFaceView({left: 5, top: 5, right: 5, bottom: 9}, 100, 100, 100, 100, 260, 180, 1),
                mismatch: computeFaceView(bbox, 600, 800, 330, 248, 260, 180, 1),
                zoomOff: computeFaceView(bbox, 100, 100, 100, 100, 260, 180, 1, {zoom: false}).mode,
                srcSmall: faceViewSourceWidth(5000, 330, 300),
                srcLarge: faceViewSourceWidth(5000, 330, 960),
                srcEnough: faceViewSourceWidth(300, 330, 960),
                srcAlreadyAtLimit: faceViewSourceWidth(5000, 250, 300),
            };
        }"""
    )
    assert out["capBelowCover"] == "contain"
    assert out["smallZoom"] == "contain"
    assert out["zeroDetection"] is None and out["degenerate"] is None and out["mismatch"] is None
    assert out["zoomOff"] == "contain"
    assert out["srcSmall"] == 0  # largest step <= 300 is 250, which is below the 330 already loaded
    assert out["srcLarge"] == 960
    assert out["srcEnough"] == 0
    assert out["srcAlreadyAtLimit"] == 0


def test_compute_face_view_marker_matches_both_modes(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (2000, 3000)})
    out = page.evaluate(
        """() => {
            const bbox = {left: 300, top: 400, right: 400, bottom: 500};
            return {
                contain: computeFaceView(bbox, 960, 1440, 330, 495, 260, 180, 1, {zoom: false}),
                crop: computeFaceView(bbox, 960, 1440, 960, 1440, 260, 180, 1),
            };
        }"""
    )
    contain = out["contain"]
    scale = contain["imgWidth"] / 960
    assert contain["box"]["left"] == pytest.approx(contain["imgLeft"] + 300 * scale)
    assert contain["box"]["width"] == pytest.approx(100 * scale)
    crop = out["crop"]
    assert crop["box"]["left"] == pytest.approx(crop["imgLeft"] + 300 * crop["imgWidth"] / 960)
    assert contain["mode"] == "contain"
    assert crop["mode"] == "crop"


def test_compute_face_view_asks_for_no_larger_source_when_the_card_stays_uncropped(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (2000, 3000)})
    out = page.evaluate(
        """() => {
            const big = {left: 10, top: 10, right: 950, bottom: 1400};   // ~98 % of the image
            const small = {left: 470, top: 700, right: 490, bottom: 720};
            const at = (b, natW, natH, dpr) => computeFaceView(b, 960, 1440, natW, natH, 260, 180, dpr);
            return {
                bigDpr1: at(big, 330, 495, 1),
                bigDpr3: at(big, 330, 495, 3),
                bigFull: at(big, 960, 1440, 2),
                smallDpr3: at(small, 330, 495, 3),
                bigSourceWidth: faceViewSourceWidth(at(big, 330, 495, 3).needNatW, 330, 960),
            };
        }"""
    )
    for key in ("bigDpr1", "bigDpr3", "bigFull"):
        assert out[key]["mode"] == "contain"
        assert out[key]["needNatW"] == 0
    assert out["bigSourceWidth"] == 0
    # A tiny face still wants a larger source: the guard only applies to crops that cannot happen.
    assert out["smallDpr3"]["needNatW"] > 330


def test_hires_swap_uses_the_server_supplied_url(open_gallery):
    face = make_face(1, bbox=(470, 700, 490, 720))
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    src = page.evaluate("document.querySelector('.result-thumb img').getAttribute('src')")
    assert src == face["thumb_urls"]["960"]
    assert src.startswith("https://upload.wikimedia.org/")
    assert widths(requests, "Photo.jpg") == [330, 960]
    # The swap never goes through the login-gated, rate-limited proxy.
    assert page.proxied_thumbs == [("Photo.jpg", 330)]
    assert page.direct_thumbs == [("Photo.jpg", 960)]
    assert not page.errors


def test_no_swap_without_a_server_url(open_gallery):
    face = make_face(1, bbox=(470, 700, 490, 720), thumb_urls={})
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    assert widths(requests, "Photo.jpg") == [330]
    assert page.evaluate("document.querySelector('.result-thumb img').getAttribute('src')") == face["thumb_url"]
    assert page.proxied_thumbs == [("Photo.jpg", 330)] and page.direct_thumbs == []
    assert not page.errors


def test_no_swap_when_the_server_offers_a_different_width(open_gallery):
    # The browser would want 960, the server only sent 500: keep the 330 px source.
    face = make_face(1, bbox=(470, 700, 490, 720))
    face["thumb_urls"] = {"500": face["thumb_urls"]["500"]}
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    assert widths(requests, "Photo.jpg") == [330]
    assert not page.errors


def test_card_that_cannot_crop_does_not_request_a_larger_thumbnail(open_gallery):
    face = make_face(1, bbox=(10, 10, 950, 1400))
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)}, dpr=3)
    assert widths(requests, "Photo.jpg") == [330]
    assert not is_cropped(page)
    assert page.direct_thumbs == []


def test_resize_after_the_swap_does_not_request_another_larger_thumbnail(open_gallery):
    face = make_face(1, bbox=(470, 700, 490, 720))
    page, requests = open_gallery([face], {"File:Photo.jpg": (2000, 3000)})
    assert widths(requests, "Photo.jpg") == [330, 960]
    seen = list(requests)

    for size in ({"width": 520, "height": 900}, {"width": 1400, "height": 1000}):
        page.set_viewport_size(size)
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        page.wait_for_load_state("networkidle")
    page.wait_for_function(f"({_MARKER_ERROR_JS})(0) < 1.5")

    assert requests == seen
    assert page.evaluate("document.querySelector('.result-thumb img').naturalWidth") == 960
    assert not page.errors


# One-click re-authorize: the page offers a button when Wikimedia no longer accepts the stored token.
_REAUTH_ERROR = {
    "error": "Your Wikimedia authorization has expired or was revoked. Please re-authorize to continue.",
    "reauth_url": "/login?reauth=1",
}


def _reject_first_card(page, status, body):
    page.route("**/api/reclassify", lambda route: route.fulfill(status=status, json=body))
    page.locator(".result-card button[data-action='reject']").first.click(force=True)
    page.wait_for_selector("#inline-alert:not([hidden])")


def test_reauthorize_button_comes_back_to_the_page_after_an_authorization_failure(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (960, 1440)})
    _reject_first_card(page, 401, _REAUTH_ERROR)

    action = page.locator("#inline-alert-action")
    assert action.is_visible()
    assert action.inner_text().lower().replace("-", "") == "reauthorize"
    # The server rendered the link for this page, so the user lands back on it after the OAuth round trip.
    assert action.get_attribute("href") == "/login?reauth=1&next=/project/1"
    assert "expired or was revoked" in page.locator("#inline-alert-text").inner_text()
    assert not page.errors


def test_other_errors_show_no_reauthorize_button(open_gallery):
    # .btn sets its own display, which once beat the hidden attribute and showed an empty button.
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (960, 1440)})
    _reject_first_card(page, 502, {"error": "Failed to remove SDC claim from Commons."})

    assert page.locator("#inline-alert-action").is_hidden()
    assert "Failed to remove" in page.locator("#inline-alert-text").inner_text()


def test_reauthorize_button_does_not_linger_on_the_next_error(open_gallery):
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (960, 1440)})
    _reject_first_card(page, 401, _REAUTH_ERROR)
    assert page.locator("#inline-alert-action").is_visible()

    page.locator("#inline-alert-close").click()
    page.unroute("**/api/reclassify")
    _reject_first_card(page, 502, {"error": "Failed to remove SDC claim from Commons."})

    assert page.locator("#inline-alert-action").is_hidden()


def test_reauthorize_button_ignores_the_url_in_the_api_answer(open_gallery):
    """The answer only says "offer the button"; its reauth_url text never reaches the link."""
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (960, 1440)})
    _reject_first_card(page, 401, {**_REAUTH_ERROR, "reauth_url": "javascript:alert(1)"})

    action = page.locator("#inline-alert-action")
    playwright.expect(action).to_be_visible()
    assert action.get_attribute("href") == "/login?reauth=1&next=/project/1"
    assert not page.errors


def test_inline_alert_action_only_accepts_same_origin_paths(open_gallery):
    """A caller passing any other URL still cannot make the link leave the site."""
    page, _ = open_gallery([make_face(1)], {"File:Photo.jpg": (960, 1440)})
    action = page.locator("#inline-alert-action")

    hostile = [
        "javascript:alert(1)",
        "data:text/html,x",
        "https://evil.example/login?reauth=1",
        "//evil.example/x",
        "\\\\evil.example/x",
        # These normalize to "//evil.example/x", which a browser reads as another site.
        "/.//evil.example/x",
        "/%2e//evil.example/x",
        "/a/..//evil.example/x",
        "https://wikivisage.test//evil.example/x",
    ]
    for url in hostile:
        page.evaluate("(url) => showInlineAlert('boom', url)", url)
        playwright.expect(action).to_be_hidden()

    page.evaluate("showInlineAlert('boom', '/login?reauth=1')")
    playwright.expect(action).to_be_visible()
    assert action.get_attribute("href") == "/login?reauth=1"
    assert not page.errors


# The waiting card: a new project shows "waiting" until a worker takes it, then "reading the category".


def _open_project_page(browser, flask_app, monkeypatch, project_overrides):
    module = importlib.import_module("app")
    project = {**_PROJECT, "status": "active", "images_total": 0, "images_processed": 0, **project_overrides}

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
        if "worker_claimed_by IS NOT NULL" in sql:  # is a worker holding the project right now?
            return [{"id": 1}] if project.get("worker_claimed_by") else ()
        if "status = 'pending'" in sql:
            return [{"cnt": counts["pending"]}]
        if "f.is_target IS NULL" in sql:  # faces waiting for the model
            return [{"cnt": counts["eligible"]}]
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    counts = {"pending": 0, "eligible": 0}
    monkeypatch.setattr(module, "execute_query", query)
    monkeypatch.setattr(module, "get_project_for_actor", lambda *args: project.copy())
    monkeypatch.setattr(module, "_get_face_stats", lambda *args: _STATS.copy())
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1

    def serve(route):
        parsed = urlparse(route.request.url)
        if parsed.hostname != "wikivisage.test":
            route.abort()
            return
        if parsed.path.startswith("/commons-thumb/"):  # the app answers with a redirect, which cannot be fulfilled
            route.fulfill(content_type="image/svg+xml", body='<svg xmlns="http://www.w3.org/2000/svg"/>')
            return
        if parsed.path == "/api/project/1/gallery":
            route.fulfill(json={"faces": [], "page": 1, "total_pages": 1, "total": 0})
            return
        response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
        route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)

    context = browser.new_context(service_workers="block")
    context.route("**/*", serve)
    page = context.new_page()
    page.counts = counts
    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.clock.install()
    page.goto("https://wikivisage.test/project/1")
    return page, project, context


def test_the_waiting_card_turns_into_reading_the_category_when_a_worker_takes_the_project(
    browser, flask_app, monkeypatch
):
    page, project, context = _open_project_page(browser, flask_app, monkeypatch, {"worker_claimed_by": None})
    try:
        message, label = page.locator("#progressMsg"), page.locator("#progressPct")
        playwright.expect(label).to_have_text("Waiting for worker\u2026")
        assert "waiting for a background worker" in message.inner_text()

        project["worker_claimed_by"] = "ml-worker-1"  # a worker claims it: the next poll shows it
        page.clock.run_for(5500)
        playwright.expect(label).to_have_text("Reading category\u2026")
        assert "listing the images in this category" in message.inner_text()
        assert "waiting for a background worker" not in message.inner_text()

        project["images_total"] = 12  # the first rows land: the card becomes a real progress bar
        page.clock.run_for(5500)
        playwright.expect(message).to_be_hidden()
        playwright.expect(label).to_have_text("0.0%")
        assert not page.errors
    finally:
        context.close()


def test_the_waiting_card_is_already_reading_when_the_page_loads_after_a_worker_claimed_it(
    browser, flask_app, monkeypatch
):
    page, _project, context = _open_project_page(browser, flask_app, monkeypatch, {"worker_claimed_by": "ml-worker-2"})
    try:
        playwright.expect(page.locator("#progressPct")).to_have_text("Reading category\u2026")
        assert "listing the images in this category" in page.locator("#progressMsg").inner_text()
        assert not page.errors
    finally:
        context.close()


# The "inference complete" card stays until the user closes it: across reloads, and for a later run too.

_FINISHED = {
    "images_total": 12,
    "images_processed": 12,
    "status": "completed",
    "last_inference_run": 1,
    "last_inference_matches": 12,
}


def test_the_complete_card_survives_reloads_and_stays_closed_until_a_newer_run_finishes(
    browser, flask_app, monkeypatch
):
    first_run = datetime(2026, 10, 28, 12, 0)
    page, project, context = _open_project_page(
        browser, flask_app, monkeypatch, {**_FINISHED, "last_inference_at": first_run}
    )
    try:
        card = page.locator("#inferenceBanner")
        playwright.expect(card).to_be_visible()
        assert "Model inference complete — 12 new matches found." in card.inner_text()

        for _ in range(2):  # a reload, and another: nothing but the user closing it makes it go
            page.reload()
            playwright.expect(card).to_be_visible()

        page.locator("#inferenceBannerClose").click()
        playwright.expect(card).to_have_count(0)
        page.reload()
        playwright.expect(card).to_have_count(0)  # closed means closed, also after a reload

        project["last_inference_at"] = datetime(2026, 10, 28, 13, 30)  # a newer run finished
        project["last_inference_run"] = 2
        project["last_inference_matches"] = 3
        page.reload()
        playwright.expect(card).to_be_visible()
        assert "Model inference complete — 3 new matches found." in card.inner_text()
        assert not page.errors
    finally:
        context.close()


def test_a_card_that_was_live_when_the_run_ended_remembers_that_run_when_closed(browser, flask_app, monkeypatch):
    """Closing the card the poller just showed must not bring it back on the next load."""
    page, project, context = _open_project_page(browser, flask_app, monkeypatch, {**_FINISHED, "status": "active"})
    try:
        page.counts["eligible"] = 7  # the model is working on 7 faces
        page.reload()
        card = page.locator("#inferenceBanner")
        playwright.expect(card).to_contain_text("Model inference in progress")

        page.counts["eligible"] = 0  # it finishes while the user watches
        project["last_inference_at"] = datetime(2026, 10, 28, 14, 0)
        project["last_inference_run"] = 2
        project["last_inference_matches"] = 5
        page.clock.run_for(5500)
        playwright.expect(card).to_contain_text("Model inference complete — 5 new matches found.")

        page.clock.run_for(60000)  # and it does not go away by itself
        playwright.expect(card).to_be_visible()

        page.locator("#inferenceBannerClose").click()
        page.reload()
        playwright.expect(card).to_have_count(0)
        assert not page.errors
    finally:
        context.close()


def test_the_card_waits_for_the_images_then_runs_the_model_then_says_it_is_done(browser, flask_app, monkeypatch):
    """Waiting for the remaining images, the model working, the result: one card, live, in that order."""
    page, project, context = _open_project_page(
        browser, flask_app, monkeypatch, {"images_total": 12, "images_processed": 8}
    )
    try:
        page.counts.update(pending=4, eligible=30)  # 4 images to go, 30 faces already waiting for the model
        page.reload()
        card = page.locator("#inferenceBanner")
        playwright.expect(card).to_contain_text("Waiting for 4 more images to be processed before running the model")
        assert "in progress" not in card.inner_text()

        page.counts["pending"] = 1
        page.clock.run_for(5500)
        playwright.expect(card).to_contain_text("Waiting for the last image to be processed before running the model")

        page.counts["pending"] = 0  # the last image is done: now the model runs
        project["images_processed"] = 12
        page.clock.run_for(5500)
        playwright.expect(card).to_contain_text("Model inference in progress — 30 faces remaining")

        page.counts["eligible"] = 0  # and it is over
        project.update(last_inference_at=datetime(2026, 10, 28, 15, 0), last_inference_run=1, last_inference_matches=21)
        page.clock.run_for(5500)
        playwright.expect(card).to_contain_text("Model inference complete — 21 new matches found.")
        assert "in progress" not in card.inner_text()
        assert not page.errors
    finally:
        context.close()


def test_a_closed_waiting_card_does_not_hide_the_later_completion(browser, flask_app, monkeypatch):
    """Closing a card that has no finished run must not remember a run: it used to store "1" for ever."""
    page, project, context = _open_project_page(
        browser, flask_app, monkeypatch, {"images_total": 12, "images_processed": 8}
    )
    try:
        page.counts.update(pending=4, eligible=30)
        page.reload()
        playwright.expect(page.locator("#inferenceBanner")).to_contain_text("Waiting for 4 more images")
        page.locator("#inferenceBannerClose").click()

        page.counts.update(pending=0, eligible=0)
        project.update(
            images_processed=12,
            last_inference_at=datetime(2026, 10, 28, 15, 0),
            last_inference_run=1,
            last_inference_matches=21,
        )
        page.reload()
        playwright.expect(page.locator("#inferenceBanner")).to_contain_text("Model inference complete — 21 new matches")
        assert not page.errors
    finally:
        context.close()


def test_a_complete_card_gives_way_to_the_waiting_card_on_the_next_visit_after_new_images_arrived(
    browser, flask_app, monkeypatch
):
    page, project, context = _open_project_page(
        browser,
        flask_app,
        monkeypatch,
        {**_FINISHED, "status": "active", "last_inference_at": datetime(2026, 10, 28, 12)},
    )
    try:
        card = page.locator("#inferenceBanner")
        playwright.expect(card).to_contain_text("Model inference complete")

        project.update(images_total=16)  # four more images turn up in the category
        page.counts.update(pending=4, eligible=9)
        page.reload()
        playwright.expect(card).to_contain_text("Waiting for 4 more images")
        assert not page.errors
    finally:
        context.close()
