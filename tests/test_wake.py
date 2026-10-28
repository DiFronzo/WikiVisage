"""The database wake-up: the web app bumps a counter, every worker notices it.

A file could not do this job. The web service and the workers are different Toolforge containers
(the web service mounts nothing), so a file written by one never reached the other and a new
project waited for the worker's next poll.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import wake
from database import DatabaseError


class Counter:
    """A stand-in for the one ``worker_wake`` row, answering the two statements wake.py runs."""

    def __init__(self, seq: int | None = 0) -> None:
        self.seq = seq  # None = no row yet
        self.reads = 0
        self.down = False

    def __call__(self, sql: str, params: Any = None, fetch: bool = True) -> Any:
        if self.down:
            raise DatabaseError("database is down")
        if sql.startswith("INSERT INTO worker_wake"):
            self.seq = 1 if self.seq is None else self.seq + 1
            return 1
        assert sql == "SELECT seq FROM worker_wake WHERE id = 1"
        self.reads += 1
        return [] if self.seq is None else [{"seq": self.seq}]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def counter():
    row = Counter()
    with patch.object(wake, "execute_query", row):
        yield row


def watcher(interval: float = 0, clock: Any = None) -> wake.WakeWatcher:
    return wake.WakeWatcher(interval=interval, clock=clock or Clock())


# --- request_wake ------------------------------------------------------------------------


def test_request_wake_bumps_the_counter_and_creates_the_row_on_first_use(counter: Counter) -> None:
    counter.seq = None

    assert wake.request_wake() is True
    assert counter.seq == 1
    assert wake.request_wake() is True
    assert counter.seq == 2


def test_request_wake_is_one_upsert_so_no_seed_row_is_needed() -> None:
    execute = MagicMock()
    with patch.object(wake, "execute_query", execute):
        wake.request_wake()

    sql = execute.call_args.args[0]
    assert "INSERT INTO worker_wake" in sql and "ON DUPLICATE KEY UPDATE seq = seq + 1" in sql
    assert execute.call_args.kwargs == {"fetch": False}


def test_request_wake_never_raises_when_the_database_is_down(counter: Counter, caplog) -> None:
    counter.down = True

    assert wake.request_wake() is False
    assert "Could not wake the worker" in caplog.text


# --- WakeWatcher -------------------------------------------------------------------------


def test_a_primed_watcher_wakes_once_per_request(counter: Counter) -> None:
    w = watcher()
    w.prime()

    assert w.consume() is False
    wake.request_wake()
    assert w.consume() is True
    assert w.consume() is False  # handled


def test_requests_made_while_busy_are_not_lost(counter: Counter) -> None:
    w = watcher()
    w.prime()
    wake.request_wake()  # the worker is mid-cycle and does not look
    wake.request_wake()

    assert w.consume() is True
    assert w.consume() is False  # two requests, one cycle: that cycle looks at everything


def test_requests_from_before_priming_do_not_wake_a_new_worker(counter: Counter) -> None:
    counter.seq = 41
    w = watcher()
    w.prime()

    assert w.consume() is False


def test_every_worker_has_its_own_baseline_so_all_of_them_wake(counter: Counter) -> None:
    first, second = watcher(), watcher()
    first.prime()
    second.prime()
    wake.request_wake()

    assert first.consume() is True
    assert second.consume() is True  # the file this replaces went to whichever worker got there first


def test_a_missing_row_reads_as_zero(counter: Counter) -> None:
    counter.seq = None
    w = watcher()
    w.prime()
    assert w.consume() is False

    wake.request_wake()  # creates the row
    assert w.consume() is True


def test_the_counter_is_only_read_once_per_interval(counter: Counter) -> None:
    clock = Clock()
    w = watcher(interval=2, clock=clock)
    w.prime()
    reads_after_prime = counter.reads

    assert w.consume() is False  # first look
    wake.request_wake()
    clock.now += 1.9
    assert w.consume() is False  # too soon: no read
    assert counter.reads == reads_after_prime + 1
    clock.now += 0.2
    assert w.consume() is True
    assert counter.reads == reads_after_prime + 2


def test_a_database_error_is_not_a_wake_and_loses_nothing(counter: Counter) -> None:
    w = watcher()
    w.prime()
    wake.request_wake()
    counter.down = True

    assert w.consume() is False  # cannot look

    counter.down = False
    assert w.consume() is True  # the request that arrived meanwhile is still there


def test_a_watcher_primed_while_the_database_was_down_adopts_the_counter_on_first_sight(counter: Counter) -> None:
    counter.seq = 7
    counter.down = True
    w = watcher()
    w.prime()  # fails quietly
    counter.down = False

    assert w.consume() is False  # first value seen: adopted, not a wake
    wake.request_wake()
    assert w.consume() is True


def test_a_truncated_table_counts_as_a_change_not_as_silence(counter: Counter) -> None:
    counter.seq = 5
    w = watcher()
    w.prime()
    counter.seq = None  # table emptied (tests, a reset)

    assert w.consume() is True


def test_the_default_interval_comes_from_the_environment_constant() -> None:
    assert wake.WakeWatcher()._interval == wake.WAKE_CHECK_INTERVAL == 2.0


# --- against the real table --------------------------------------------------------------


@pytest.mark.integration
def test_wake_round_trip_against_the_real_table(db_pool: Any) -> None:
    w, other = wake.WakeWatcher(interval=0), wake.WakeWatcher(interval=0)
    w.prime()
    other.prime()
    assert w.consume() is False

    assert wake.request_wake() is True
    assert wake.request_wake() is True

    assert w.consume() is True
    assert other.consume() is True
    assert w.consume() is False
