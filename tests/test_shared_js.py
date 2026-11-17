"""`static/wikivisage-common.js` behaviour that the server cannot see.

The file is a plain browser script with no module system, so the pieces that
matter for correctness (postForm's answer classification) are run in node
against a scripted fetch. Every page that writes a classification goes through
postForm, and it is the only place that decides whether an HTTP answer means
"the write happened".
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import require_node_env

_ROOT = Path(__file__).resolve().parent.parent

_NODE = shutil.which("node")


def test_node_is_installed_when_the_suite_requires_it():
    """These JavaScript tests skip without node, so CI asserts it is there (WIKIVISAGE_REQUIRE_NODE=1)."""
    if require_node_env():
        assert _NODE is not None, "WIKIVISAGE_REQUIRE_NODE=1 but node is not installed"


# The script is loaded into a bare context, then postForm is driven with answers
# that a real deployment can produce: the JSON body it expects, an HTML page from
# a proxy, an unparseable half-body, a bodiless 204, and a redirect to /login.
_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const cases = JSON.parse(process.argv[3]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const context = { console, URL, FormData, Promise, Object, Number, isFinite };
vm.createContext(context);
vm.runInContext(source, context);
const postForm = context.postForm;

async function run(c) {
  context.fetch = async () => ({
    ok: c.status >= 200 && c.status < 300,
    status: c.status,
    redirected: !!c.redirected,
    url: c.url || 'https://x.test/api/classify',
    headers: { get: () => c.contentType || '' },
    text: async () => c.body,
  });
  try {
    const data = await postForm('/api/classify', 'token', {});
    return { resolved: true, data };
  } catch (e) {
    return {
      resolved: false,
      message: e.message,
      notJson: !!e.notJson,
      redirected: !!e.redirected,
      status: e.status,
      data: e.data || null,
    };
  }
}

(async () => {
  const out = [];
  for (const c of cases) out.push(await run(c));
  console.log(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_post_form_only_accepts_a_json_answer_as_a_saved_write(tmp_path):
    cases = [
        # The normal path: a JSON body resolves with it.
        {"status": 200, "contentType": "application/json", "body": '{"status": "ok", "face_id": 4}'},
        # A 204 has no body by definition and is still a success.
        {"status": 204, "contentType": "", "body": ""},
        # A proxy or login interstitial: this used to resolve as {} and the caller
        # reported the classification as saved.
        {"status": 200, "contentType": "text/html", "body": "<html>maintenance</html>"},
        # A JSON content type with a truncated body is just as unusable.
        {"status": 200, "contentType": "application/json", "body": '{"status": "o'},
        # A normal API error still carries the server's message.
        {"status": 409, "contentType": "application/json", "body": '{"error": "already classified"}'},
        # An expired session redirects /api/* to the login page.
        {"status": 200, "contentType": "text/html", "body": "<html>login</html>", "redirected": True},
    ]
    harness = tmp_path / "post_form_harness.js"
    harness.write_text(_HARNESS)
    done = subprocess.run(  # noqa: S603
        [_NODE, str(harness), str(_ROOT / "static" / "wikivisage-common.js"), json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    ok, no_content, html_page, truncated, conflict, logged_out = json.loads(done.stdout)

    assert ok["resolved"] and ok["data"]["status"] == "ok"
    assert no_content["resolved"] and no_content["data"] == {}

    assert not html_page["resolved"]
    assert html_page["notJson"] and html_page["status"] == 200
    assert "Server error (HTTP 200)" in html_page["message"]

    # An unparseable body is rejected as well, though `notJson` stays false: the
    # content type was right, the JSON was not. Callers fall back to the
    # translated generic message either way.
    assert not truncated["resolved"] and truncated["data"] == {}

    assert not conflict["resolved"]
    assert conflict["data"]["error"] == "already classified"

    assert not logged_out["resolved"] and logged_out["redirected"]
