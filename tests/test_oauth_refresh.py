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
    db = FakeDb(selects=[[row(SOON)]], updates=[1])
    exchange = exchange_ok()

    result = run(db, exchange, current=TokenSet("old-access", "old-refresh", SOON))

    exchange.assert_called_once_with("old-refresh")
    assert result is not None
    assert (result.access_token, result.refresh_token, result.source) == ("new-access", "new-refresh", "refreshed")
    assert result.expires_at is not None and result.expires_at > datetime.now(UTC) + timedelta(hours=1)
    sql, params = db.calls[-1]
    assert "WHERE id = %s AND token_expires_at = %s" in sql
    assert params[3] == 1 and params[4] == SOON.strftime("%Y-%m-%d %H:%M:%S")


def test_credentials_the_caller_holds_are_re_read_once_the_lock_is_held() -> None:
    """A peer may have rotated the refresh token since the caller read its copy."""
    db = FakeDb(selects=[[row(SOON, refresh="rotated-refresh")]], updates=[1])
    exchange = exchange_ok()

    run(db, exchange, current=TokenSet("old-access", "stale-refresh", SOON))

    exchange.assert_called_once_with("rotated-refresh")
    assert db.calls[0][0].startswith("SELECT")


def test_a_peer_refresh_that_landed_before_the_lock_is_used_without_an_exchange() -> None:
    db = FakeDb(selects=[[row(LATER, access="peer-access")]])
    exchange = exchange_ok()

    result = run(db, exchange, current=TokenSet("old-access", "stale-refresh", SOON))

    exchange.assert_not_called()
    assert result is not None and (result.access_token, result.source) == ("peer-access", "reread")


def test_a_caller_token_without_a_refresh_token_never_reaches_the_database(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb()

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(db, exchange_ok(), current=TokenSet("a", "", SOON)) is None

    assert db.calls == []
    assert "No refresh token for user 1" in caplog.text


def test_the_old_refresh_token_is_kept_when_the_endpoint_does_not_rotate_it() -> None:
    result = run(
        FakeDb(selects=[[row(SOON)]], updates=[1]),
        exchange_ok({"access_token": "new-access"}),
        current=TokenSet("old-access", "old-refresh", SOON),
    )

    assert result is not None
    assert result.refresh_token == "old-refresh"
    assert result.expires_at is not None
    assert result.expires_at > datetime.now(UTC) + timedelta(seconds=oauth_refresh.DEFAULT_LIFETIME_SECONDS - 60)


def test_a_reply_without_an_access_token_is_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb(selects=[[row(SOON)]])

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(db, exchange_ok({"error": "invalid_grant"}), current=TokenSet("a", "r", SOON)) is None

    assert not any(sql.startswith("UPDATE") for sql, _ in db.calls)
    assert "missing access_token" in caplog.text


def test_an_endpoint_error_is_a_failure_and_nothing_is_stored(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb(selects=[[row(SOON)]])
    exchange = MagicMock(side_effect=RuntimeError("HTTP 500"))

    with caplog.at_level(logging.ERROR, logger="oauth_refresh"):
        assert run(db, exchange, current=TokenSet("a", "r", SOON)) is None

    assert not any(sql.startswith("UPDATE") for sql, _ in db.calls)
    assert "Failed to refresh access token for user 1" in caplog.text


def test_losing_the_compare_and_swap_returns_the_winners_tokens() -> None:
    db = FakeDb(
        selects=[[row(SOON)], [row(LATER, access="winner-access", refresh="winner-refresh")]],
        updates=[0],
    )

    result = run(db, exchange_ok(), current=TokenSet("old", "old", SOON))

    assert result is not None
    assert (result.access_token, result.source) == ("winner-access", "reread")


def test_an_unconditional_update_never_triggers_the_lost_race_path() -> None:
    """Without an old expiry there is no compare, so 0 rows is not a lost race."""
    db = FakeDb(selects=[[row(None)]], updates=[0])

    result = run(db, exchange_ok(), current=TokenSet("old", "old", None))

    assert result is not None and result.source == "refreshed"


def test_a_store_failure_fails_the_refresh_by_default() -> None:
    db = FakeDb(selects=[[row(SOON)]], updates=[DatabaseError("deadlock")])

    assert run(db, exchange_ok(), current=TokenSet("old", "old", SOON)) is None


def test_a_store_failure_can_keep_the_new_token_for_the_work_in_hand(caplog: pytest.LogCaptureFixture) -> None:
    db = FakeDb(selects=[[row(SOON)]], updates=[DatabaseError("deadlock")])

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
    db = FakeDb(
        selects=[
            [row(SOON, access="still-old", refresh="still-old-refresh")],
            [row(SOON, access="still-old", refresh="still-old-refresh")],
        ],
        updates=[1],
    )
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
            query=FakeDb(selects=[[row(SOON)]], updates=[1]),
            get_redis=lambda: None,
            exchange=exchange_ok(),
            current=TokenSet("a", "r", SOON),
        )

    assert seen == ["oauth-refresh:42"]


# -- permanent versus transient failures -----------------------------------


class _Http400(Exception):
    """What ``requests`` raises for a refused refresh: an HTTP error carrying the response."""

    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"HTTP {status}")
        self.response = MagicMock(status_code=status)
        if isinstance(body, Exception):
            self.response.json.side_effect = body
        else:
            self.response.json.return_value = body


def _failure_of(db: FakeDb, exchange: MagicMock, **kwargs: Any) -> list[bool]:
    reports: list[bool] = []
    assert run(db, exchange, on_failure=reports.append, **kwargs) is None
    return reports


def test_a_revoked_refresh_token_is_a_permanent_failure(caplog: pytest.LogCaptureFixture) -> None:
    from oauthlib.oauth2.rfc6749.errors import InvalidGrantError

    exchange = MagicMock(side_effect=InvalidGrantError())

    with caplog.at_level(logging.WARNING, logger="oauth_refresh"):
        reports = _failure_of(FakeDb(selects=[[row(SOON)]]), exchange, current=TokenSet("a", "r", SOON))

    assert reports == [True]
    assert "refresh token of user 1 was refused" in caplog.text


@pytest.mark.parametrize("status", [400, 401])
def test_a_plain_http_invalid_grant_is_permanent_too(status: int) -> None:
    """The worker talks to the endpoint with plain requests, which has no InvalidGrantError."""
    exchange = MagicMock(side_effect=_Http400(status, {"error": "invalid_grant"}))

    assert _failure_of(FakeDb(selects=[[row(SOON)]]), exchange, current=TokenSet("a", "r", SOON)) == [True]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("HTTP 500"),
        TimeoutError("slow"),
        _Http400(500, {"error": "invalid_grant"}),
        _Http400(400, {"error": "invalid_request"}),
        _Http400(400, ValueError("not json")),
        _Http400(429, ["rate limited"]),
    ],
)
def test_other_endpoint_failures_are_transient(error: Exception) -> None:
    exchange = MagicMock(side_effect=error)

    assert _failure_of(FakeDb(selects=[[row(SOON)]]), exchange, current=TokenSet("a", "r", SOON)) == [False]


def test_a_reply_without_an_access_token_is_transient() -> None:
    reports = _failure_of(FakeDb(selects=[[row(SOON)]]), exchange_ok({"oops": 1}), current=TokenSet("a", "r", SOON))

    assert reports == [False]


def test_a_store_failure_is_transient() -> None:
    db = FakeDb(selects=[[row(SOON)]], updates=[DatabaseError("deadlock")])

    assert _failure_of(db, exchange_ok(), current=TokenSet("a", "r", SOON)) == [False]


def test_a_missing_user_or_refresh_token_is_permanent() -> None:
    assert _failure_of(FakeDb(selects=[[]]), exchange_ok()) == [True]
    assert _failure_of(FakeDb(), exchange_ok(), current=TokenSet("a", "", SOON)) == [True]


def test_a_vanished_user_after_the_lock_or_after_a_peer_is_permanent() -> None:
    assert _failure_of(FakeDb(selects=[[]]), exchange_ok(), current=TokenSet("a", "r", SOON)) == [True]
    with lock_held_by_peer():
        assert _failure_of(FakeDb(selects=[[]]), exchange_ok(), current=TokenSet("a", "r", SOON)) == [True]


def test_the_failure_is_reported_once_even_when_a_peer_retry_fails() -> None:
    db = FakeDb(selects=[[row(SOON)], [row(SOON)]])
    exchange = MagicMock(side_effect=TimeoutError("slow"))
    passes = iter([False, True])

    @contextlib.contextmanager
    def single_flight(*_args: Any, **_kwargs: Any):
        yield next(passes)

    reports: list[bool] = []
    with patch.object(oauth_refresh, "single_flight", single_flight), patch.object(oauth_refresh.time, "sleep"):
        assert run(db, exchange, current=TokenSet("a", "r", SOON), on_failure=reports.append) is None

    assert reports == [False]


def test_success_never_reports_a_failure() -> None:
    reports: list[bool] = []
    db = FakeDb(selects=[[row(SOON)]], updates=[1])

    assert run(db, exchange_ok(), current=TokenSet("a", "r", SOON), on_failure=reports.append) is not None
    assert reports == []
