"""Best-effort Redis-backed single-flight lock.

Used to coordinate exclusive operations across the gunicorn web workers and the
two Toolforge background workers. Currently the only call site is OAuth token
refresh, where two concurrent requests for the same user must NOT both call the
provider's `/token` endpoint with the same refresh_token (Wikimedia rotates the
refresh token on every successful exchange — the second call would invalidate
the first call's freshly issued tokens, silently logging the user out).

Design constraints:
- Best-effort: if Redis is unreachable, the lock is a no-op (acquire returns
  True). The DB-level optimistic concurrency check (CAS on `token_expires_at`)
  remains as a second line of defense, so a Redis outage degrades us back to
  the pre-fix behavior rather than breaking auth entirely.
- Stateless: connection sharing only. A single `SET key value NX EX ttl`.
- Auto-expiring: TTL ensures a crashed holder cannot wedge the lock forever.
- Atomic release: a Lua script performs the compare-and-delete atomically so
  an expired-then-reacquired lock is never accidentally released by the old
  holder.
- Connection sharing: callers pass an existing redis client (already created
  for the rate limiter in app.py / could be created lazily in worker.py).

Usage:
    with single_flight(redis_client, "oauth-refresh:42", ttl_seconds=10) as got:
        if got:
            ... do the exclusive work ...
        else:
            ... another process is doing it; wait + re-read shared state ...
"""

from __future__ import annotations

import contextlib
import logging
import secrets
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

_LOCK_KEY_PREFIX = "wikivisage:lock:"

# Lua script for atomic compare-and-delete.
# Returns 1 if the key was deleted (we owned it), 0 otherwise.
# This closes the TOCTOU race between GET+compare and DEL: if the lock expired
# and was re-acquired by another process between our GET and our DEL, the
# script detects the mismatch and leaves the new owner's lock intact.
_LUA_RELEASE = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


@contextlib.contextmanager
def single_flight(
    redis_client: Any | None,
    name: str,
    ttl_seconds: int = 10,
) -> Iterator[bool]:
    """Try to acquire a Redis single-flight lock; release on exit.

    Yields True if the lock was acquired (caller is the single flight).
    Yields False if another process holds the lock OR Redis is unavailable
    in a way that distinguishes contention from outage.

    A None redis_client (Redis disabled at startup) yields True — caller
    proceeds unprotected. This preserves liveness in dev / Redis outage.

    The token written under the key is a random 128-bit value used to
    guarantee we only DEL keys we ourselves set (otherwise an expired-then-
    re-acquired lock could be released by the wrong holder).
    """
    if redis_client is None:
        # No Redis configured — degrade to no-op single flight.
        yield True
        return

    key = f"{_LOCK_KEY_PREFIX}{name}"
    token = secrets.token_hex(16)
    acquired = False

    try:
        # SET key value NX EX ttl — atomic acquire.
        # `nx=True` means "only set if not exists"; `ex=ttl` sets expiry.
        result = redis_client.set(key, token, nx=True, ex=ttl_seconds)
        acquired = bool(result)
    except Exception:
        # Redis went down between startup ping and now. Fail-open: yield True
        # so callers proceed (DB CAS still protects correctness). Log so ops
        # can correlate with subsequent token-rotation issues.
        logger.warning("Redis unreachable while acquiring lock %r — proceeding without single-flight", key)
        yield True
        return

    try:
        yield acquired
    finally:
        if acquired:
            # Atomic compare-and-delete via Lua to prevent releasing a lock that
            # expired and was re-acquired by another process between our GET and
            # our DEL.
            try:
                redis_client.eval(_LUA_RELEASE, 1, key, token)
            except Exception:
                logger.warning("Redis unreachable while releasing lock %r — letting TTL expire it", key)
