"""Exercise the rendered bbox editor in real browsers without DB or network services."""

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

_PORTRAIT = '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="1440"><rect width="960" height="1440" fill="grey"/></svg>'


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    monkeypatch.setenv("WIKIVISAGE_REDIS_URL", "memory://")


@pytest.fixture(scope="module", params=["chromium", "webkit"])
def browser(request):
    with playwright.sync_playwright() as runtime:
        instance = getattr(runtime, request.param).launch()
        yield instance
        instance.close()


@pytest.fixture
def editor(browser, flask_app, monkeypatch):
    module = importlib.import_module("app")
    project = {
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
    stats = {
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
    monkeypatch.setattr(module, "get_project_for_actor", lambda *args: project.copy())
    monkeypatch.setattr(module, "_get_face_stats", lambda *args: stats.copy())
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1

    face = {
        "id": 10,
        "image_id": 100,
        "file_title": "File:Portrait.jpg",
        "thumb_url": "/commons-thumb/Portrait.jpg?width=250",
        "bbox_top": 200,
        "bbox_right": 400,
        "bbox_bottom": 600,
        "bbox_left": 100,
        "detection_width": 960,
        "detection_height": 1440,
        "result": "match",
        "is_target": 1,
        "classified_by": "model",
        "confidence": 0.2,
        "sdc_written": 0,
        "sdc_pending": 1,
        "sdc_removal": 0,
        "bootstrapped": 0,
    }

    def serve(route):
        parsed = urlparse(route.request.url)
        if parsed.hostname != "wikivisage.test":
            route.abort()
        elif parsed.path.startswith("/commons-thumb/"):
            route.fulfill(content_type="image/svg+xml", body=_PORTRAIT)
        elif parsed.path == "/api/project/1/gallery":
            route.fulfill(json={"faces": [face], "page": 1, "total_pages": 1, "total": 1})
        else:
            response = client.get(parsed.path + ("?" + parsed.query if parsed.query else ""))
            route.fulfill(status=response.status_code, content_type=response.content_type, body=response.data)

    with browser.new_context(
        viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True, service_workers="block"
    ) as context:
        context.route("**/*", serve)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("https://wikivisage.test/project/1")
        page.locator(".result-actions button").nth(2).click()
        page.wait_for_function("document.getElementById('modalImg').naturalWidth === 960 && currentOldBbox !== null")
        yield page
        assert not errors


def touch(page, event_type, touches, changed=None):
    """Dispatch touch identities at detection-space coordinates through the real listeners."""
    page.evaluate(
        """({eventType, touches, changed}) => {
            const area = document.getElementById('drawingArea');
            const rect = document.getElementById('modalImg').getBoundingClientRect();
            const makeTouch = ([identifier, x, y]) => ({
                identifier, target: area,
                clientX: rect.left + x * rect.width / 960,
                clientY: rect.top + y * rect.height / 1440
            });
            // WebKit does not expose a constructible Touch. Replay the same
            // touch data through the DOM listeners using a portable Event.
            const event = new Event(eventType, {bubbles: true, cancelable: true});
            Object.defineProperties(event, {
                touches: {value: touches.map(makeTouch)},
                changedTouches: {value: changed.map(makeTouch)}
            });
            (eventType === 'touchstart' ? area : document).dispatchEvent(event);
        }""",
        {"eventType": event_type, "touches": touches, "changed": changed if changed is not None else touches},
    )


def draw_box(page):
    touch(page, "touchstart", [(7, 100, 200)])
    touch(page, "touchmove", [(7, 300, 500)])
    touch(page, "touchend", [], [(7, 320, 520)])


def assert_aborted(page):
    assert page.evaluate("currentBbox") is None
    assert page.evaluate("isDrawing") is False
    assert page.evaluate("drawTouchId") is None
    assert page.locator("#saveBboxBtn").is_disabled()
    assert not page.locator("#drawingBox").is_visible()
    assert page.locator("#oldBbox").is_visible()


def test_touch_drag_records_detection_coordinates_and_final_position(editor):
    draw_box(editor)
    assert editor.evaluate("currentBbox") == {"left": 100, "top": 200, "right": 320, "bottom": 520}
    assert editor.locator("#saveBboxBtn").is_enabled()


@pytest.mark.parametrize("second_touch_event", ["touchstart", "touchmove"])
def test_multitouch_aborts_without_switching_fingers(editor, second_touch_event):
    touch(editor, "touchstart", [(7, 100, 200)])
    touch(editor, "touchmove", [(7, 300, 500)])
    touch(editor, second_touch_event, [(7, 300, 500), (8, 600, 900)])
    touch(editor, "touchend", [(8, 600, 900)], [(7, 300, 500)])
    touch(editor, "touchmove", [(8, 700, 1100)])
    touch(editor, "touchend", [], [(8, 700, 1100)])
    assert_aborted(editor)


def test_unrelated_touch_end_cannot_finish_drawing(editor):
    touch(editor, "touchstart", [(7, 100, 200)])
    touch(editor, "touchmove", [(7, 300, 500)])
    touch(editor, "touchend", [(7, 300, 500)], [(8, 600, 900)])
    assert editor.evaluate("isDrawing") is True
    assert editor.locator("#saveBboxBtn").is_disabled()
    touch(editor, "touchend", [], [(7, 320, 520)])
    assert editor.evaluate("currentBbox") == {"left": 100, "top": 200, "right": 320, "bottom": 520}


def test_touchcancel_restores_previous_box(editor):
    touch(editor, "touchstart", [(7, 100, 200)])
    touch(editor, "touchmove", [(7, 300, 500)])
    touch(editor, "touchcancel", [], [(7, 300, 500)])
    assert_aborted(editor)


def test_rotation_cancels_an_in_progress_drag(editor):
    touch(editor, "touchstart", [(7, 100, 200)])
    touch(editor, "touchmove", [(7, 300, 500)])
    editor.set_viewport_size({"width": 956, "height": 440})
    editor.wait_for_function("!isDrawing")
    assert_aborted(editor)


def test_finished_box_tracks_resize_and_new_drag_disables_save(editor):
    draw_box(editor)
    expected = editor.evaluate("currentBbox")
    editor.set_viewport_size({"width": 956, "height": 440})
    editor.wait_for_function("window.innerWidth === 956")
    # Dispatch after layout as well, so the test does not depend on resize-event timing.
    editor.evaluate("window.dispatchEvent(new Event('resize'))")
    assert editor.evaluate("currentBbox") == expected
    image = editor.locator("#modalImg").bounding_box()
    box = editor.locator("#drawingBox").bounding_box()
    assert box["x"] == pytest.approx(image["x"] + expected["left"] * image["width"] / 960, abs=1)
    assert box["y"] == pytest.approx(image["y"] + expected["top"] * image["height"] / 1440, abs=1)
    assert box["width"] == pytest.approx((expected["right"] - expected["left"]) * image["width"] / 960, abs=1)
    assert box["height"] == pytest.approx((expected["bottom"] - expected["top"]) * image["height"] / 1440, abs=1)
    touch(editor, "touchstart", [(9, 200, 300)])
    assert editor.evaluate("currentBbox") is None
    assert editor.locator("#saveBboxBtn").is_disabled()


def test_landscape_portrait_image_is_scrollable_instead_of_clipped(editor):
    editor.set_viewport_size({"width": 956, "height": 440})
    editor.wait_for_function("window.innerWidth === 956")
    geometry = editor.evaluate("""() => {
        const area = document.getElementById('drawingArea').getBoundingClientRect();
        const image = document.getElementById('modalImg').getBoundingClientRect();
        const body = document.querySelector('.modal-body');
        return {areaHeight: area.height, imageHeight: image.height,
                scrollHeight: body.scrollHeight, clientHeight: body.clientHeight};
    }""")
    assert geometry["areaHeight"] == pytest.approx(geometry["imageHeight"], abs=1)
    assert geometry["scrollHeight"] > geometry["clientHeight"]
    editor.locator("#saveBboxBtn").scroll_into_view_if_needed()
    save = editor.locator("#saveBboxBtn").bounding_box()
    assert save["y"] >= 0 and save["y"] + save["height"] <= 440


def test_losing_image_geometry_clears_partial_box(editor):
    touch(editor, "touchstart", [(7, 100, 200)])
    touch(editor, "touchmove", [(7, 300, 500)])
    editor.locator("#modalImg").evaluate("image => image.style.display = 'none'")
    touch(editor, "touchend", [], [(7, 300, 500)])
    assert editor.evaluate("currentBbox") is None
    assert editor.locator("#saveBboxBtn").is_disabled()
    assert not editor.locator("#drawingBox").is_visible()


def test_mouse_drawing_still_records_detection_coordinates(editor):
    image = editor.locator("#modalImg").bounding_box()
    start_x = round(image["x"] + 100 * image["width"] / 960)
    start_y = round(image["y"] + 200 * image["height"] / 1440)
    end_x = round(image["x"] + 320 * image["width"] / 960)
    end_y = round(image["y"] + 520 * image["height"] / 1440)
    editor.mouse.move(start_x, start_y)
    editor.mouse.down()
    editor.mouse.move(end_x, end_y)
    editor.mouse.up()
    assert editor.evaluate("currentBbox") == {
        "left": round((start_x - image["x"]) * 960 / image["width"]),
        "top": round((start_y - image["y"]) * 1440 / image["height"]),
        "right": round((end_x - image["x"]) * 960 / image["width"]),
        "bottom": round((end_y - image["y"]) * 1440 / image["height"]),
    }
    assert editor.locator("#saveBboxBtn").is_enabled()
