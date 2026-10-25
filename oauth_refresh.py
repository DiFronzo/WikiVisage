"""The OAuth refresh-token protocol, shared by the web app and the background worker.

Wikimedia rotates the refresh token on every exchange, so two processes refreshing
the same user at once silently invalidate each other's tokens (the user is logged
out and an SDC write batch dies halfway). The web app and the worker both refresh,
and used to carry their own copy of the protection against that. This module is the
single copy. It has three layers, all best effort in that order:

1. A Redis single-flight lock per user, so only one process in the cluster calls the
   token endpoint with a given refresh token. Without Redis it is a no-op.
2. An optimistic compare-and-swap on ``token_expires_at`` when storing the result:
   zero rows updated means another process won the race.
3. A re-read of the winner's freshly stored tokens whenever a guard trips.

What differs between the two callers is passed in: how to talk to the token endpoint
(``exchange``), which database function to use (``query``, looked up by the caller at
call time so tests can patch it where they always did), and where Redis comes from.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from database import DatabaseError
from redis_lock import single_flight
from token_crypto import TokenDecryptionError, decrypt_token, encrypt_token

logger = logging.getLogger(__name__)

# Refresh when the access token has this many seconds or less left.
TOKEN_REFRESH_BUFFER = 300
# Must comfortably exceed one token-exchange round trip, and bounds how long a
# crashed lock holder can delay other refreshes.
LOCK_TTL_SECONDS = 15
# How long to wait for a peer that holds the lock before re-reading its result.
PEER_WAIT_SECONDS = 1.0
# Used when the token endpoint omits ``expires_in``.
DEFAULT_LIFETIME_SECONDS = 14400

_DB_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

Query = Callable[..., Any]


@dataclass
class TokenSet:
    """A user's decrypted credentials. ``source`` says where this instance came from."""

    access_token: str
    refresh_token: str
    expires_at: datetime | None
    source: str = "current"  # "current" (as passed in), "reread" (from the database) or "refreshed"


def as_utc(value: datetime | str | None) -> datetime | None:
    """Normalise a database value (naive, aware or ISO string) to an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def is_fresh(expires_at: datetime | None) -> bool:
    """True while the access token has more than TOKEN_REFRESH_BUFFER seconds left."""
    return expires_at is not None and (expires_at - datetime.now(UTC)).total_seconds() > TOKEN_REFRESH_BUFFER


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def read_user_tokens(query: Query, user_id: int, *, label: str) -> TokenSet | None:
    """Read and decrypt a user's tokens. None when the row is gone or cannot be decrypted."""
    rows = query(
        "SELECT access_token, refresh_token, token_expires_at FROM users WHERE id = %s",
        (user_id,),
        fetch=True,
    )
    if not rows:
        return None
    row = rows[0]
    try:
        access_token = decrypt_token(_text(row["access_token"]))
        refresh_token = decrypt_token(_text(row.get("refresh_token", "")))
    except TokenDecryptionError:
        logger.error(f"Cannot decrypt the stored tokens for {label} — possible key rotation")
        return None
    return TokenSet(access_token, refresh_token, as_utc(row.get("token_expires_at")), source="reread")


def store_tokens(query: Query, user_id: int, new: TokenSet, old_expires_at: datetime | None) -> int:
    """Save refreshed tokens if nobody else did first. Returns the number of rows updated.

    With *old_expires_at* the UPDATE is the compare-and-swap (0 rows: a peer won).
    Without one there is nothing to compare, so it is unconditional.
    """
    assert new.expires_at is not None
    values = (
        encrypt_token(new.access_token),
        encrypt_token(new.refresh_token),
        new.expires_at.strftime(_DB_TIME_FORMAT),
    )
    if old_expires_at is None:
        return query(
            "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s WHERE id = %s",
            (*values, user_id),
            fetch=False,
        )
    return query(
        "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s "
        "WHERE id = %s AND token_expires_at = %s",
        (*values, user_id, old_expires_at.strftime(_DB_TIME_FORMAT)),
        fetch=False,
    )


def refresh_tokens(
    user_id: int,
    *,
    label: str,
    query: Query,
    get_redis: Callable[[], Any],
    exchange: Callable[[str], Mapping[str, Any]],
    current: TokenSet | None = None,
    keep_on_store_failure: bool = False,
) -> TokenSet | None:
    """Return valid credentials for the user, refreshing them first when they are about to expire.

    *current* are credentials the caller already holds; without them they are read from the
    database. *exchange* trades a refresh token for the token endpoint's reply and raises on
    failure. *get_redis* is only called when a refresh is actually needed, so a worker whose
    tokens are valid never opens a Redis connection. With *keep_on_store_failure* a refresh
    that succeeded at the provider but could not be saved is still returned (the caller can
    use the new access token for the work in hand); otherwise it counts as a failure.

    Returns None when the user cannot be refreshed (no row, no refresh token, endpoint error).
    """
    credentials = current if current is not None else read_user_tokens(query, user_id, label=label)
    if credentials is None:
        return None
    if is_fresh(credentials.expires_at):
        return credentials
    if not credentials.refresh_token:
        logger.error(f"No refresh token for {label}, cannot refresh")
        return None

    with single_flight(get_redis(), f"oauth-refresh:{user_id}", ttl_seconds=LOCK_TTL_SECONDS) as got_lock:
        if got_lock:
            credentials = read_user_tokens(query, user_id, label=label)
            if credentials is None:
                return None
            if is_fresh(credentials.expires_at):
                return credentials
            if not credentials.refresh_token:
                logger.error(f"No refresh token for {label}, cannot refresh")
                return None
            return _exchange_and_store(
                user_id,
                credentials,
                label=label,
                query=query,
                exchange=exchange,
                keep_on_store_failure=keep_on_store_failure,
            )
        # Another process is mid-refresh. Give it a moment to commit, then use its result.
        logger.info(f"Token refresh in flight for {label} — waiting on peer")
        time.sleep(PEER_WAIT_SECONDS)
        peer = read_user_tokens(query, user_id, label=label)

    if peer is None:
        return None
    if is_fresh(peer.expires_at):
        return peer
    # The peer did not finish (its UPDATE may have failed): try ourselves, outside the lock.
    logger.warning(f"Peer refresh did not complete for {label}; retrying")
    return refresh_tokens(
        user_id,
        label=label,
        query=query,
        get_redis=get_redis,
        exchange=exchange,
        current=peer,
        keep_on_store_failure=keep_on_store_failure,
    )


def _exchange_and_store(
    user_id: int,
    credentials: TokenSet,
    *,
    label: str,
    query: Query,
    exchange: Callable[[str], Mapping[str, Any]],
    keep_on_store_failure: bool,
) -> TokenSet | None:
    logger.info(f"Refreshing access token for {label}")
    try:
        reply = exchange(credentials.refresh_token)
        access_token = reply.get("access_token")
        if not access_token:
            logger.error(f"Token refresh response missing access_token for {label}")
            return None
        refreshed = TokenSet(
            access_token,
            reply.get("refresh_token", credentials.refresh_token),
            datetime.now(UTC) + timedelta(seconds=reply.get("expires_in", DEFAULT_LIFETIME_SECONDS)),
            source="refreshed",
        )
        try:
            updated = store_tokens(query, user_id, refreshed, credentials.expires_at)
        except DatabaseError:
            if not keep_on_store_failure:
                raise
            logger.exception(f"Failed to persist refreshed token for {label}")
            return refreshed
        if updated == 0 and credentials.expires_at is not None:
            logger.info(f"Token already refreshed by another process for {label}")
            return read_user_tokens(query, user_id, label=label)
        return refreshed
    except Exception:
        logger.exception(f"Failed to refresh access token for {label}")
        return None
