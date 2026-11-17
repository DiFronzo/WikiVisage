"""Gallery hi-res thumbnail URLs: api_gallery hands the browser direct Commons URLs, not the proxy."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._app_preamble import app_module, flask_app
from tests._app_support import _make_authed_client
from tests.conftest import require_node_env

_ROOT = Path(__file__).resolve().parent.parent
_NODE = shutil.which("node")


def test_node_is_installed_when_the_suite_requires_it():
    """The service-worker test skips without node, so CI asserts it is there (WIKIVISAGE_REQUIRE_NODE=1)."""
    if require_node_env():
        assert _NODE is not None, "WIKIVISAGE_REQUIRE_NODE=1 but node is not installed"


def _face_row(detection_width, file_title="File:Example photo.jpg"):
    return {
        "id": 7,
        "image_id": 70,
        "is_target": 1,
        "confidence": 0.2,
        "classified_by": "model",
        "classified_by_user_id": None,
        "bbox_top": 10,
        "bbox_right": 90,
        "bbox_bottom": 90,
        "bbox_left": 10,
        "sdc_written": 0,
        "sdc_removal_pending": 0,
        "has_confirmed_target_sibling": 0,
        "file_title": file_title,
        "commons_page_id": 1234,
        "detection_width": detection_width,
        "detection_height": 640,
        "bootstrapped": 0,
    }


def _gallery_row(monkeypatch, fake_user, detection_width, file_title="File:Example photo.jpg"):
    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM faces" in sql:
            return [{"cnt": 1}]
        if "SELECT f.id" in sql:
            return [_face_row(detection_width, file_title)]
        return [] if fetch else 0

    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client = _make_authed_client(monkeypatch, fake_user, route_execute)
        response = client.get("/api/project/1/gallery?page=2")
    finally:
        flask_app.config["SERVER_NAME"] = None
    assert response.status_code == 200
    (row,) = response.get_json()["faces"]
    return row


@pytest.mark.parametrize(
    ("detection_width", "expected"),
    [
        (960, [500, 960]),
        (1280, [500, 960]),
        (3000, [500, 960]),
        (700, [500]),
        (500, [500]),
        (499, []),
        (400, []),
        (330, []),
        (0, []),
        (-5, []),
    ],
)
def test_gallery_hires_steps(detection_width, expected):
    assert app_module._gallery_hires_steps(detection_width) == expected


def test_gallery_hires_limit_matches_the_browser_constant():
    source = (_ROOT / "static" / "wikivisage-common.js").read_text()
    assert f"var FACE_VIEW_MAX_SOURCE_WIDTH = {app_module._GALLERY_HIRES_MAX};" in source


def test_gallery_row_carries_direct_urls_for_a_960_detection(monkeypatch, fake_user):
    row = _gallery_row(monkeypatch, fake_user, 960)

    assert set(row["thumb_urls"]) == {"500", "960"}
    assert row["thumb_urls"]["960"] == app_module.commons_thumb_url("File:Example photo.jpg", 960)
    assert row["thumb_urls"]["500"] == app_module.commons_thumb_url("File:Example photo.jpg", 500)
    for url in row["thumb_urls"].values():
        assert url.startswith("https://upload.wikimedia.org/")
        assert "/commons-thumb/" not in url
    assert row["thumb_url"] == app_module.commons_thumb_url("File:Example photo.jpg", 330)


def test_gallery_row_offers_only_what_a_small_original_has(monkeypatch, fake_user):
    row = _gallery_row(monkeypatch, fake_user, 700)

    assert set(row["thumb_urls"]) == {"500"}
    assert row["thumb_urls"]["500"].startswith("https://upload.wikimedia.org/")


def test_gallery_row_without_detection_width_offers_no_hires(monkeypatch, fake_user):
    row = _gallery_row(monkeypatch, fake_user, None)

    assert row["thumb_urls"] == {}
    assert row["detection_width"] == 0


def test_gallery_row_urls_follow_the_file_type(monkeypatch, fake_user):
    row = _gallery_row(monkeypatch, fake_user, 960, file_title="File:Scan.tif")

    assert row["thumb_urls"]["960"].endswith("/960px-Scan.tif.jpg")


def test_gallery_row_carries_no_per_image_face_count(monkeypatch, fake_user):
    """Bootstrap cards are never outlined, so the row needs no face count: the count that
    would matter is the number of *people* in the photo, which detection cannot supply."""
    row = _gallery_row(monkeypatch, fake_user, 960)

    assert "faces_on_image" not in row


@pytest.mark.integration
def test_gallery_lists_a_bootstrapped_face_without_a_face_count(
    integration_client, db_conn, seed_project, seed_images, seed_faces
):
    """Against MariaDB: a bootstrapped face is listed for the card, with no count to gate on."""
    client, _, _ = integration_client
    url = f"/api/project/{seed_project['id']}/gallery?per_page=100"

    with db_conn.cursor() as cur:
        cur.execute("UPDATE faces SET classified_by = 'bootstrap' WHERE id = %s", (seed_faces["target"][0],))

    faces = client.get(url).get_json()["faces"]
    boot = [row for row in faces if row["classified_by"] == "bootstrap"]
    assert boot, "the bootstrapped face is still listed"
    assert all("faces_on_image" not in row for row in faces)


# sw.js has no module system; run it in node with a fake service-worker scope.
_SW_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const cases = JSON.parse(process.argv[3]);
const source = fs.readFileSync(process.argv[2], 'utf8');

async function run(scenario) {
  const listeners = {};
  const store = new Map();
  if (scenario.cached) store.set(scenario.url, { cachedCopy: true, url: scenario.url });
  if (scenario.stale) store.set(scenario.stale, { cachedCopy: true, url: scenario.stale });
  const keyOf = (req) => (typeof req === 'string' ? req : req.url);
  const cache = {
    put: async (req, res) => { store.set(keyOf(req), res); },
    addAll: async () => {},
    match: async (req) => store.get(keyOf(req)),
    keys: async () => [...store.keys()].map((url) => ({ url })),
    delete: async (req) => store.delete(keyOf(req)),
  };
  const caches = {
    open: async () => cache,
    match: async (req) => store.get(keyOf(req)),
    keys: async () => [],
    delete: async () => true,
  };
  const context = {
    self: { addEventListener: (name, fn) => { listeners[name] = fn; }, location: { origin: 'https://x.test' },
            skipWaiting() {}, clients: { claim() {} } },
    caches, URL, Response,
    fetch: async () => {
      if (scenario.network === 'throw') throw new TypeError('offline');
      return { ok: scenario.network < 400, status: scenario.network, clone() { return this; }, fresh: true };
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  // event.waitUntil() work: cache writes must finish before we report.
  const dispatch = (url) => {
    const entry = { handled: false, background: [] };
    listeners.fetch({
      request: { url, method: 'GET' },
      respondWith: (promise) => { entry.handled = true; entry.answered = promise; },
      waitUntil: (promise) => { entry.background.push(promise); },
    });
    return entry;
  };
  // scenario.urls dispatches several requests before awaiting any of them, which is how
  // two tabs (or two tabs' reloads) hit the same script concurrently.
  const dispatched = (scenario.urls || [scenario.url]).map(dispatch);
  const responses = await Promise.all(dispatched.map((entry) => entry.answered));
  await Promise.allSettled(dispatched.flatMap((entry) => entry.background));
  const cachedUrls = [...store.keys()].sort();
  if (scenario.urls) {
    return { handled: dispatched.every((entry) => entry.handled), cachedUrls };
  }
  const response = responses[0];
  if (!dispatched[0].handled) return { handled: false };
  return { handled: true, cachedCopy: !!(response && response.cachedCopy), fresh: !!(response && response.fresh),
           status: response && response.status, cachedUrls };
}

(async () => {
  const out = [];
  for (const c of cases) out.push(await run(c));
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_service_worker_scripts_fall_back_to_the_cache_on_5xx_and_errors(tmp_path):
    js = "https://x.test/static/wikivisage-common.js"
    png = "https://x.test/static/icon.png"
    cases = [
        {"url": js, "network": 200, "cached": True},  # fresh copy wins
        {"url": js, "network": 502, "cached": True},  # 5xx: cached copy
        {"url": js, "network": 502, "cached": False},  # 5xx, nothing cached: the 5xx passes through
        {"url": js, "network": "throw", "cached": True},  # offline: cached copy
        {"url": js, "network": 404, "cached": True},  # a 4xx is a real answer, not a fallback case
        {"url": png, "network": 200, "cached": False},  # other static files keep their own strategy
        {"url": "https://x.test/project/1", "network": 200, "cached": False},  # routes: not handled
        # A page asks for ?v=<release>: the copy of the previous release is dropped, so the
        # cache cannot grow by one script per release for ever.
        {"url": js + "?v=2", "network": 200, "cached": False, "stale": js + "?v=1"},
        # Two tabs ask for two releases at once. Pruning must not delete the copy the
        # other request just stored, or the cache is empty and a later 5xx has no
        # fallback at all.
        {"urls": [js + "?v=3", js + "?v=4"], "network": 200},
    ]
    harness = tmp_path / "sw_harness.js"
    harness.write_text(_SW_HARNESS)
    done = subprocess.run(  # noqa: S603
        [_NODE, str(harness), str(_ROOT / "static" / "sw.js"), json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    fresh, five_cached, five_bare, offline, four, other_static, route, reversion, concurrent = json.loads(done.stdout)

    assert fresh["fresh"] and not fresh["cachedCopy"]
    assert five_cached["cachedCopy"] and not five_cached["fresh"]
    assert five_bare["fresh"] and five_bare["status"] == 502
    assert offline["cachedCopy"]
    assert four["fresh"] and four["status"] == 404
    assert other_static["handled"] and other_static["fresh"]
    assert route["handled"] is False
    assert reversion["cachedUrls"] == [js + "?v=2"]
    # One of the two concurrently fetched releases survives; an empty cache means the two
    # pruning jobs deleted each other's entry.
    assert concurrent["handled"]
    assert len(concurrent["cachedUrls"]) == 1, concurrent["cachedUrls"]
    assert concurrent["cachedUrls"][0] in (js + "?v=3", js + "?v=4")
