"""Unit tests for model-server/guard.py, the face service's public-facing guard.

The face service is published on the internet, so these pin down exactly which
requests reach the application. guard.py is stdlib-only and is loaded by path,
so this runs in the normal suite without KServe or dlib installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

_GUARD_PATH = Path(__file__).resolve().parent.parent / "model-server" / "guard.py"
_spec = importlib.util.spec_from_file_location("face_service_guard", _GUARD_PATH)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

TOKEN = "t" * 40
HEALTH_PATH = "/v1/models/wikivisage"
PREDICT_PATH = "/v1/models/wikivisage:predict"
CAP = 1000


class _App:
    def __init__(self) -> None:
        self.calls = 0
        self.body = b""

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if scope["type"] == "http":
            while True:
                message = await receive()
                self.body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


def _run(app, *, method="POST", path=PREDICT_PATH, headers=None, chunks=(b"",), scope_type="http"):
    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1} for i, chunk in enumerate(chunks)
    ]
    sent: list[dict] = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    asyncio.run(app(scope, receive, send))

    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    if start is None:
        return None, {}, body
    return start["status"], dict(start["headers"]), body


def _guarded(token=TOKEN, cap=CAP):
    app = _App()
    return app, guard.RequestGuard(app, token=token, max_body_bytes=cap, open_paths=[HEALTH_PATH])


def _auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_predict_without_token_is_rejected_before_reaching_the_app():
    app, g = _guarded()
    status, headers, body = _run(g, chunks=(b"{}",), headers={"Content-Length": "2"})
    assert status == 401
    assert headers[b"www-authenticate"] == b"Bearer"
    assert json.loads(body)["error"]
    assert app.calls == 0


def test_predict_with_wrong_token_is_rejected():
    app, g = _guarded()
    status, _, _ = _run(g, headers={**_auth("x" * 40), "Content-Length": "0"})
    assert status == 401
    assert app.calls == 0


def test_predict_with_correct_token_reaches_the_app():
    app, g = _guarded()
    status, _, _ = _run(g, chunks=(b"{}",), headers={**_auth(), "Content-Length": "2"})
    assert status == 200
    assert app.body == b"{}"


def test_scheme_is_case_insensitive():
    app, g = _guarded()
    status, _, _ = _run(g, headers={"Authorization": f"bearer {TOKEN}", "Content-Length": "0"})
    assert status == 200


@pytest.mark.parametrize("header", [TOKEN, f"Basic {TOKEN}", "Bearer", ""])
def test_malformed_authorization_is_rejected(header):
    app, g = _guarded()
    status, _, _ = _run(g, headers={"Authorization": header, "Content-Length": "0"})
    assert status == 401
    assert app.calls == 0


def test_health_path_is_open_to_unauthenticated_get():
    """Toolforge's HTTP health check cannot send headers."""
    app, g = _guarded()
    status, _, _ = _run(g, method="GET", path=HEALTH_PATH)
    assert status == 200


def test_health_path_still_checks_credentials_that_are_presented():
    app, g = _guarded()
    status, _, _ = _run(g, method="GET", path=HEALTH_PATH, headers=_auth("x" * 40))
    assert status == 401


def test_health_path_is_not_open_to_post():
    app, g = _guarded()
    status, _, _ = _run(g, method="POST", path=HEALTH_PATH, headers={"Content-Length": "0"})
    assert status == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v2/repository/models/wikivisage/unload"),
        ("POST", "/v2/models/wikivisage/infer"),
        ("GET", "/v1/models"),
        ("GET", "/metrics"),
        ("GET", "/docs"),
        ("GET", "/"),
        ("GET", HEALTH_PATH + "/"),
    ],
)
def test_everything_but_the_health_path_needs_the_token(method, path):
    """Includes KServe's model-unload endpoint, which would be a remote off switch."""
    app, g = _guarded()
    status, _, _ = _run(g, method=method, path=path, headers={"Content-Length": "0"})
    assert status == 401
    assert app.calls == 0


def test_websocket_is_refused_when_a_token_is_configured():
    app, g = _guarded()
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    asyncio.run(g({"type": "websocket", "path": "/", "headers": []}, receive, send))
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert app.calls == 0


def test_lifespan_passes_through():
    app, g = _guarded()
    asyncio.run(g({"type": "lifespan"}, None, None))
    assert app.calls == 1


def test_without_a_token_every_path_is_open():
    app, g = _guarded(token="")
    status, _, _ = _run(g, chunks=(b"{}",), headers={"Content-Length": "2"})
    assert status == 200


def test_short_tokens_are_refused_at_construction():
    with pytest.raises(ValueError, match="at least"):
        guard.RequestGuard(_App(), token="short", max_body_bytes=CAP, open_paths=[])


def test_declared_oversized_body_is_rejected_without_reading_it():
    app, g = _guarded()
    status, _, _ = _run(g, headers={**_auth(), "Content-Length": str(CAP + 1)})
    assert status == 413
    assert app.calls == 0


def test_body_exactly_at_the_cap_is_accepted():
    app, g = _guarded()
    payload = b"x" * CAP
    status, _, _ = _run(g, chunks=(payload,), headers={**_auth(), "Content-Length": str(CAP)})
    assert status == 200
    assert app.body == payload


@pytest.mark.parametrize("value", ["abc", "-1"])
def test_invalid_content_length_is_rejected(value):
    app, g = _guarded()
    status, _, _ = _run(g, headers={**_auth(), "Content-Length": value})
    assert status == 400
    assert app.calls == 0


def test_chunked_body_over_the_cap_is_rejected_before_the_app_sees_it():
    app, g = _guarded()
    status, _, _ = _run(g, chunks=(b"x" * 600, b"x" * 600), headers=_auth())
    assert status == 413
    assert app.calls == 0


def test_chunked_body_under_the_cap_is_replayed_intact():
    app, g = _guarded()
    status, _, _ = _run(g, chunks=(b"ab", b"cd", b"ef"), headers=_auth())
    assert status == 200
    assert app.body == b"abcdef"


def test_body_cap_applies_even_without_a_token():
    app, g = _guarded(token="")
    status, _, _ = _run(g, headers={"Content-Length": str(CAP + 1)})
    assert status == 413


def test_client_disconnect_during_chunked_upload_sends_nothing():
    app, g = _guarded()
    sent = []
    messages = [{"type": "http.request", "body": b"x", "more_body": True}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": PREDICT_PATH,
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
    }
    asyncio.run(g(scope, receive, send))
    assert sent == []
    assert app.calls == 0
