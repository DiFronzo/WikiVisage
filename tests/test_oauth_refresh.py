"""Tests for the OAuth refresh protocol shared by the web app and the worker."""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import oauth_refresh
from database import DatabaseError
from oauth_refresh import TokenSet, refresh_tokens
from token_crypto import TokenDecryptionError

SOON = datetime.now(UTC) + timedelta(seconds=30)  # inside the refresh buffer
LATER = datetime.now(UTC) + timedelta(hours=3)


def row(expires: datetime | str | None, access: Any = "old-access", refresh: Any = "old-refresh") -> dict[str, Any]:
    return {"access_token": access, "refresh_token": refresh, "token_expires_at": expires}


class FakeDb:
    """Scripted ``execute_query``: SELECT results in order, UPDATE rowcounts in order."""

    def __init__(self, selects: list[Any] | None = None, updates: list[Any] | None = None) -> None:
        self.selects = list(selects or [])
        self.updates = list(updates or [])
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, sql: str, params: Any = None, fetch: bool = True) -> Any:
        self.calls.append((" ".join(sql.split()), params))
        queue = self.selects if fetch else self.updates
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def exchange_ok(reply: dict[str, Any] | None = None) -> MagicMock:
    default = {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 7200}
    return MagicMock(return_value=reply or default)


def run(db: FakeDb, exchange: MagicMock, **kwargs: Any) -> TokenSet | None:
    kwargs.setdefault("get_redis", lambda: None)
    return refresh_tokens(1, label="user 1", query=db, exchange=exchange, **kwargs)


@contextlib.contextmanager
def lock_held_by_peer():
    @contextlib.contextmanager
    def single_flight(*_args: Any, **_kwargs: Any):
        yield False

    with (
        patch.object(oauth_refresh, "single_flight", single_flight),
        patch.object(oauth_refresh.time, "sleep") as sleep,
    ):
        yield sleep


# -- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [datetime(2026, 1, 1, 12, 0), datetime(2026, 1, 1, 12, 0, tzinfo=UTC), "2026-01-01T12:00:00"],
)
def test_expiry_values_are_normalised_to_aware_utc(value: Any) -> None:
    assert oauth_refresh.as_utc(value) == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_a_missing_expiry_stays_missing_and_is_never_fresh() -> None:
    assert oauth_refresh.as_utc(None) is None
    assert oauth_refresh.is_fresh(None) is False


def test_freshness_respects_the_refresh_buffer() -> None:
    assert oauth_refresh.is_fresh(datetime.now(UTC) + timedelta(seconds=oauth_refresh.TOKEN_REFRESH_BUFFER + 60))
    assert not oauth_refresh.is_fresh(datetime.now(UTC) + timedelta(seconds=oauth_refresh.TOKEN_REFRESH_BUFFER - 60))


def test_stored_bytes_are_decoded_before_decrypting() -> None:
    db = FakeDb(selects=[[row(LATER, access=b"access-bytes", refresh=b"refresh-bytes")]])

    tokens = oauth_refresh.read_user_tokens(db, 1, label="user 1")

    assert tokens == TokenSet("access-bytes", "refresh-bytes", LATER, source="reread")


def test_a_vanished_user_reads_as_none() -> None:
    assert oauth_refresh.read_user_tokens(FakeDb(selects=[[]]), 1, label="user 1") is None


def test_tokens_that_cannot_be_decrypted_read_as_none(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb(selects=[[row(LATER)]])

    with (
        patch.object(oauth_refresh, "decrypt_token", side_effect=TokenDecryptionError("rotated key")),
        caplog.at_level(logging.ERROR, logger="oauth_refresh"),
    ):
        assert oauth_refresh.read_user_tokens(db, 1, label="user 1") is None

    assert "possible key rotation" in caplog.text


def test_storing_compares_the_old_expiry() -> None:
    db = FakeDb(updates=[1])
    new = TokenSet("a", "r", datetime(2026, 5, 1, 8, 30, tzinfo=UTC))

    oauth_refresh.store_tokens(db, 7, new, datetime(2026, 5, 1, 4, 0, tzinfo=UTC))

    sql, params = db.calls[0]
    assert sql.endswith("WHERE id = %s AND token_expires_at = %s")
    assert params[2:] == ("2026-05-01 08:30:00", 7, "2026-05-01 04:00:00")


def test_storing_without_an_old_expiry_is_unconditional() -> None:
    db = FakeDb(updates=[1])

    oauth_refresh.store_tokens(db, 7, TokenSet("a", "r", datetime(2026, 5, 1, tzinfo=UTC)), None)

    sql, params = db.calls[0]
    assert sql.endswith("WHERE id = %s")
    assert len(params) == 4  # access, refresh, expiry, user id: nothing to compare against


# -- the protocol ----------------------------------------------------------


def test_valid_credentials_are_returned_without_touching_redis_or_the_endpoint() -> None:
    get_redis = MagicMock()
    exchange = exchange_ok()
    current = TokenSet("a", "r", LATER)

    result = run(FakeDb(), exchange, current=current, get_redis=get_redis)

    assert result is current
    get_redis.assert_not_called()
    exchange.assert_not_called()


def test_credentials_are_read_from_the_database_when_none_are_passed() -> None:
    db = FakeDb(selects=[[row(LATER)]])

    result = run(db, exchange_ok())

    assert result is not None and result.access_token == "old-access"
    assert result.source == "reread"


def test_a_user_without_a_row_cannot_be_refreshed() -> None:
    exchange = exchange_ok()

    assert run(FakeDb(selects=[[]]), exchange) is None
    exchange.assert_not_called()


def test_without_a_refresh_token_there_is_nothing_to_exchange(caplog: pytest.LogCaptureFixture) -> None:
    exchange = exchange_ok()

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(FakeDb(selects=[[row(SOON, refresh="")]]), exchange) is None

    exchange.assert_not_called()
    assert "No refresh token for user 1" in caplog.text


def test_expiring_credentials_are_exchanged_and_stored_with_a_compare_and_swap() -> None:
    db = FakeDb(updates=[1])
    exchange = exchange_ok()

    result = run(db, exchange, current=TokenSet("old-access", "old-refresh", SOON))

    exchange.assert_called_once_with("old-refresh")
    assert result is not None
    assert (result.access_token, result.refresh_token, result.source) == ("new-access", "new-refresh", "refreshed")
    assert result.expires_at is not None and result.expires_at > datetime.now(UTC) + timedelta(hours=1)
    sql, params = db.calls[0]
    assert "WHERE id = %s AND token_expires_at = %s" in sql
    assert params[3] == 1 and params[4] == SOON.strftime("%Y-%m-%d %H:%M:%S")


def test_the_old_refresh_token_is_kept_when_the_endpoint_does_not_rotate_it() -> None:
    result = run(
        FakeDb(updates=[1]),
        exchange_ok({"access_token": "new-access"}),
        current=TokenSet("old-access", "old-refresh", SOON),
    )

    assert result is not None
    assert result.refresh_token == "old-refresh"
    assert result.expires_at is not None
    assert result.expires_at > datetime.now(UTC) + timedelta(seconds=oauth_refresh.DEFAULT_LIFETIME_SECONDS - 60)


def test_a_reply_without_an_access_token_is_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb()

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(db, exchange_ok({"error": "invalid_grant"}), current=TokenSet("a", "r", SOON)) is None

    assert db.calls == []
    assert "missing access_token" in caplog.text


def test_an_endpoint_error_is_a_failure_and_nothing_is_stored(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb()
    exchange = MagicMock(side_effect=RuntimeError("HTTP 500"))

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(db, exchange, current=TokenSet("a", "r", SOON)) is None

    assert db.calls == []
    assert "Failed to refresh access token for user 1" in caplog.text


def test_losing_the_compare_and_swap_returns_the_winners_tokens() -> None:
    db = FakeDb(selects=[[row(LATER, access="winner-access", refresh="winner-refresh")]], updates=[0])

    result = run(db, exchange_ok(), current=TokenSet("old", "old", SOON))

    assert result is not None
    assert (result.access_token, result.source) == ("winner-access", "reread")


def test_an_unconditional_update_never_triggers_the_lost_race_path() -> None:
    """Without an old expiry there is no compare, so 0 rows is not a lost race."""
    db = FakeDb(updates=[0])

    result = run(db, exchange_ok(), current=TokenSet("old", "old", None))

    assert result is not None and result.source == "refreshed"


def test_a_store_failure_fails_the_refresh_by_default() -> None:
    db = FakeDb(updates=[DatabaseError("deadlock")])

    assert run(db, exchange_ok(), current=TokenSet("old", "old", SOON)) is None


def test_a_store_failure_can_keep_the_new_token_for_the_work_in_hand(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb(updates=[DatabaseError("deadlock")])

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        result = run(db, exchange_ok(), current=TokenSet("old", "old", SOON), keep_on_store_failure=True)

    assert result is not None and result.access_token == "new-access"
    assert "Failed to persist refreshed token for user 1" in caplog.text


# -- a peer holds the lock -------------------------------------------------


def test_when_a_peer_refreshes_its_result_is_used_without_calling_the_endpoint() -> None:
    db = FakeDb(selects=[[row(LATER, access="peer-access")]])
    exchange = exchange_ok()

    with lock_held_by_peer() as sleep:
        result = run(db, exchange, current=TokenSet("old", "old", SOON))

    sleep.assert_called_once_with(oauth_refresh.PEER_WAIT_SECONDS)
    exchange.assert_not_called()
    assert result is not None and result.access_token == "peer-access"


def test_a_peer_whose_row_disappeared_ends_the_refresh() -> None:
    with lock_held_by_peer():
        assert run(FakeDb(selects=[[]]), exchange_ok(), current=TokenSet("old", "old", SOON)) is None


def test_a_peer_that_did_not_finish_is_retried_once_the_lock_is_free(caplog: pytest.LogCaptureFixture) -> None:
    """First pass: lock held, peer's UPDATE never landed. Second pass: the lock is ours."""
    db = FakeDb(selects=[[row(SOON, access="still-old", refresh="still-old-refresh")]], updates=[1])
    exchange = exchange_ok()
    passes = iter([False, True])

    @contextlib.contextmanager
    def single_flight(*_args: Any, **_kwargs: Any):
        yield next(passes)

    with (
        patch.object(oauth_refresh, "single_flight", single_flight),
        patch.object(oauth_refresh.time, "sleep"),
        caplog.at_level(logging.WARNING, logger="oauth_refresh"),
    ):
        result = run(db, exchange, current=TokenSet("old", "old", SOON))

    exchange.assert_called_once_with("still-old-refresh")  # the retry used what the peer left in the database
    assert result is not None and result.access_token == "new-access"
    assert "Peer refresh did not complete for user 1" in caplog.text


def test_the_lock_is_named_per_user() -> None:
    seen: list[str] = []

    @contextlib.contextmanager
    def single_flight(client: Any, name: str, **_kwargs: Any):
        seen.append(name)
        yield True

    with patch.object(oauth_refresh, "single_flight", single_flight):
        refresh_tokens(
            42,
            label="user 42",
            query=FakeDb(updates=[1]),
            get_redis=lambda: None,
            exchange=exchange_ok(),
            current=TokenSet("a", "r", SOON),
        )

    assert seen == ["oauth-refresh:42"]
