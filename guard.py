"""Bearer-token and request-size guard for the face service.

The service is published on the public internet (``https://<tool>.toolforge.org``),
so this middleware is what stands between it and anyone who finds the URL.
It is plain ASGI with no third-party imports so it can be unit-tested without
installing KServe or dlib.

With a token configured:

* Every request needs ``Authorization: Bearer <token>`` — except a plain
  ``GET``/``HEAD`` to one of ``open_paths``, so the platform health check (which
  cannot send headers) still works. This matters beyond ``predict``: KServe also
  serves ``POST /v2/repository/models/<name>/unload``, which would let anyone
  take the model offline.
* A request that *presents* credentials is always checked, even on an open
  path, so a caller holding a wrong token never gets a misleading success.

With no token configured the guard only enforces the body cap, which keeps
local development and CI unchanged.

The body cap is enforced before the application sees a single byte: a declared
``Content-Length`` is checked up front (the HTTP parser guarantees the body
cannot exceed it), and a chunked body is read into memory up to the cap and
then replayed to the application.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

MIN_TOKEN_LENGTH = 32

_OPEN_METHODS = frozenset({"GET", "HEAD"})


class RequestGuard:
    def __init__(self, app: ASGIApp, *, token: str, max_body_bytes: int, open_paths: Iterable[str]) -> None:
        if token and len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"face service token must be at least {MIN_TOKEN_LENGTH} characters")
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self._token = token.encode("utf-8") if token else b""
        self._max_body_bytes = max_body_bytes
        self._open_paths = frozenset(open_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket" and self._token:
            # The service has no websocket routes; refuse rather than pass an
            # unauthenticated upgrade through to the application.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
        else:
            await self.app(scope, receive, send)

    async def _http(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = _headers(scope)

        if self._token and not self._authorized(scope, headers.get(b"authorization")):
            await _respond(send, 401, "missing or invalid bearer token", {b"www-authenticate": b"Bearer"})
            return

        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _respond(send, 400, "invalid Content-Length header")
                return
            if declared < 0:
                await _respond(send, 400, "invalid Content-Length header")
                return
            if declared > self._max_body_bytes:
                await _respond(send, 413, f"request body exceeds {self._max_body_bytes} bytes")
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length: the body is chunked (or empty). Read it here, up to
        # the cap, so an oversized upload is refused before the app buffers it.
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self._max_body_bytes:
                await _respond(send, 413, f"request body exceeds {self._max_body_bytes} bytes")
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        await self.app(scope, _replay(b"".join(chunks), receive), send)

    def _authorized(self, scope: Scope, authorization: bytes | None) -> bool:
        if authorization is not None:
            scheme, _, credentials = authorization.partition(b" ")
            return scheme.lower() == b"bearer" and hmac.compare_digest(credentials.strip(), self._token)
        return scope.get("method") in _OPEN_METHODS and scope.get("path") in self._open_paths


def _headers(scope: Scope) -> dict[bytes, bytes]:
    """Lower-cased header map. Repeated headers keep the first value."""
    result: dict[bytes, bytes] = {}
    for name, value in scope.get("headers", []):
        result.setdefault(name.lower(), value)
    return result


def _replay(body: bytes, receive: Receive) -> Receive:
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


async def _respond(send: Send, status: int, error: str, extra_headers: dict[bytes, bytes] | None = None) -> None:
    payload = json.dumps({"error": error}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode("ascii")),
    ]
    headers.extend((extra_headers or {}).items())
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
