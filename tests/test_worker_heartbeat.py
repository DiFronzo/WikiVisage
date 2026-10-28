"""Tests for the worker's heartbeat thread and progress watchdog.

The thread writes the per-worker ``worker_heartbeats`` row and touches the Toolforge health-check
file while the main loop reports progress, stops doing both when it does not, and announces a
planned stop on shutdown. Everything here runs against a fake connection and a fake clock; the
integration test at the end proves the SQL against the real table.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

import worker
from database import DatabaseError


class FakeConn:
    """Stands in for ``database.DedicatedConnection``: records statements, can be told to fail."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.errors: list[Exception] = []
        self.closed = False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        if self.errors:
            raise self.errors.pop(0)
        self.statements.append((sql, params))
        return 1

    def close(self) -> None:
        self.closed = True

    def running(self) -> list[tuple[str, tuple[Any, ...]]]:
        return [s for s in self.statements if "VALUES (%s, 'running'" in s[0]]

    def stopping(self) -> list[tuple[str, tuple[Any, ...]]]:
        return [s for s in self.statements if "VALUES (%s, 'stopping'" in s[0]]


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(worker, "shutdown_requested", False)
    monkeypatch.setattr(worker, "_worker_id", "ml-worker-1")
    monkeypatch.setattr(worker, "_current_phase", "process_images")
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic())
    # Most tests are about a worker that finished startup; the startup-gate tests clear the event.
    worker._startup_done.set()
    yield
    worker._startup_done.clear()


@pytest.fixture
def touch():
    with patch.object(worker, "_touch_heartbeat_file") as mock:
        yield mock


def make_beat() -> tuple[worker._Heartbeat, FakeConn]:
    conn = FakeConn()
    return worker._Heartbeat(conn), conn  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# _mark_progress
# --------------------------------------------------------------------------


def test_mark_progress_records_phase_and_a_fresh_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_last_progress", 0.0)

    worker._mark_progress("bootstrap")

    assert worker._current_phase == "bootstrap"
    assert worker.time.monotonic() - worker._last_progress < 5


def test_mark_progress_does_no_io() -> None:
    with patch.object(worker, "execute_query") as query, patch.object(worker, "_touch_heartbeat_file") as touch:
        worker._mark_progress("x")

    query.assert_not_called()
    touch.assert_not_called()


# --------------------------------------------------------------------------
# One tick
# --------------------------------------------------------------------------


def test_a_tick_with_fresh_progress_writes_the_row_and_touches_the_file(touch: Any) -> None:
    beat, conn = make_beat()

    beat.tick()

    touch.assert_called_once_with()
    ((sql, params),) = conn.running()
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "expected_back_until = NULL" in sql  # a restarted worker cancels its own grace period
    worker_id, idle, phase, version, first = params
    assert (worker_id, phase, version, first) == ("ml-worker-1", "process_images", worker.PROJECT_VERSION, 1)
    assert 0 <= idle < 5


def test_only_the_first_successful_write_sets_started_at(touch: Any) -> None:
    beat, conn = make_beat()

    beat.tick()
    beat.tick()

    assert [params[-1] for _, params in conn.running()] == [1, 0]


def test_started_at_is_retried_until_a_write_succeeds(touch: Any) -> None:
    beat, conn = make_beat()
    conn.errors = [DatabaseError("down")]

    beat.tick()
    beat.tick()

    assert [params[-1] for _, params in conn.running()] == [1]


def test_the_phase_and_version_fit_their_columns(touch: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_current_phase", "p" * 200)
    monkeypatch.setattr(worker, "PROJECT_VERSION", "v" * 100)
    beat, conn = make_beat()

    beat.tick()

    _, _, phase, version, _ = conn.running()[0][1]
    assert len(phase) == 64
    assert len(version) == 32


# --------------------------------------------------------------------------
# Stall watchdog
# --------------------------------------------------------------------------


def test_a_stalled_worker_stops_touching_and_writing_and_logs_once_with_phase_and_stack(
    touch: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic() - worker.STALL_LIMIT - 1)
    monkeypatch.setattr(worker, "_current_phase", "sdc_write")
    beat, conn = make_beat()

    with caplog.at_level(logging.ERROR, logger="worker"):
        beat.tick()
        beat.tick()

    touch.assert_not_called()
    assert conn.statements == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "phase=sdc_write" in errors[0].getMessage()
    # pytest runs this test on the main thread, so its own frame is in the reported stack
    assert "test_a_stalled_worker_stops_touching" in errors[0].getMessage()


def test_progress_after_a_stall_resumes_the_heartbeat_and_a_new_stall_logs_again(
    touch: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    beat, conn = make_beat()
    stalled = worker.time.monotonic() - worker.STALL_LIMIT - 1

    with caplog.at_level(logging.INFO, logger="worker"):
        monkeypatch.setattr(worker, "_last_progress", stalled)
        beat.tick()
        worker._mark_progress("cycle")
        beat.tick()
        monkeypatch.setattr(worker, "_last_progress", stalled)
        beat.tick()

    assert touch.call_count == 1
    assert len(conn.running()) == 1
    assert "progress resumed" in caplog.text
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 2


def test_the_stall_report_survives_a_missing_main_thread_frame(
    touch: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic() - worker.STALL_LIMIT - 1)
    beat, _ = make_beat()

    with patch.object(worker.sys, "_current_frames", return_value={}), caplog.at_level(logging.ERROR, logger="worker"):
        beat.tick()

    assert "stack unavailable" in caplog.text


def test_the_stall_limit_is_well_above_one_slow_call() -> None:
    assert worker.STALL_LIMIT >= 600


# --------------------------------------------------------------------------
# Database trouble never costs the file touch
# --------------------------------------------------------------------------


def test_a_database_failure_still_touches_the_file_and_warns(touch: Any, caplog: pytest.LogCaptureFixture) -> None:
    beat, conn = make_beat()
    conn.errors = [DatabaseError("pool gone")]

    with caplog.at_level(logging.WARNING, logger="worker"):
        beat.tick()

    touch.assert_called_once_with()
    assert "Heartbeat write failed" in caplog.text
    assert "pool gone" in caplog.text


def test_the_next_tick_retries_and_reports_the_recovery(touch: Any, caplog: pytest.LogCaptureFixture) -> None:
    beat, conn = make_beat()
    conn.errors = [DatabaseError("blip")]

    with caplog.at_level(logging.INFO, logger="worker"):
        beat.tick()
        beat.tick()

    assert touch.call_count == 2
    assert len(conn.running()) == 1
    assert "Heartbeat writes recovered" in caplog.text


def test_failure_warnings_are_limited_to_one_per_minute(
    touch: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    beat, conn = make_beat()
    clock = {"now": 1000.0}
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(worker, "_last_progress", 1000.0)
    conn.errors = [DatabaseError("1"), DatabaseError("2"), DatabaseError("3")]

    with caplog.at_level(logging.WARNING, logger="worker"):
        beat.tick()
        clock["now"] += worker.HEARTBEAT_INTERVAL
        monkeypatch.setattr(worker, "_last_progress", clock["now"])
        beat.tick()
        assert caplog.text.count("Heartbeat write failed") == 1
        clock["now"] += worker.HEARTBEAT_WARN_INTERVAL
        monkeypatch.setattr(worker, "_last_progress", clock["now"])
        beat.tick()

    assert caplog.text.count("Heartbeat write failed") == 2
    assert touch.call_count == 3


# --------------------------------------------------------------------------
# The thread body
# --------------------------------------------------------------------------


def run_loop(conn: FakeConn, *, stop_after_sleeps: int | None) -> list[float]:
    """Run ``_heartbeat_loop`` in this thread; flip shutdown after N one-second sleeps."""
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if stop_after_sleeps is not None and len(sleeps) >= stop_after_sleeps:
            worker.shutdown_requested = True

    with (
        patch.object(worker, "open_dedicated_connection", return_value=conn),
        patch.object(worker.time, "sleep", side_effect=sleep),
    ):
        worker._heartbeat_loop()
    return sleeps


def test_the_loop_ticks_every_interval_in_one_second_slices(touch: Any) -> None:
    conn = FakeConn()

    sleeps = run_loop(conn, stop_after_sleeps=worker.HEARTBEAT_INTERVAL * 2 + 1)

    assert set(sleeps) == {1}
    assert len(conn.running()) == 3  # at once, after 30 s, after 60 s
    assert touch.call_count == 3


def test_shutdown_writes_exactly_one_stopping_row_with_a_future_deadline_and_exits(touch: Any) -> None:
    conn = FakeConn()

    run_loop(conn, stop_after_sleeps=3)

    ((sql, params),) = conn.stopping()
    assert "expected_back_until" in sql
    assert "NOW() + INTERVAL %s SECOND" in sql
    assert params[:2] == ("ml-worker-1", worker.WORKER_RESTART_GRACE_SECONDS)
    assert params[1] > 0
    assert conn.statements[-1] == conn.stopping()[0]  # nothing runs after the announcement
    assert len(conn.running()) == 1
    assert conn.closed is True


def test_shutdown_is_noticed_within_a_second(touch: Any) -> None:
    conn = FakeConn()

    sleeps = run_loop(conn, stop_after_sleeps=1)

    assert len(sleeps) == 1  # not a whole interval
    assert len(conn.stopping()) == 1


def test_a_shutdown_before_the_first_tick_still_announces_the_stop(touch: Any) -> None:
    conn = FakeConn()
    worker.shutdown_requested = True

    sleeps = run_loop(conn, stop_after_sleeps=None)

    assert sleeps == []
    assert conn.running() == []
    assert len(conn.stopping()) == 1
    touch.assert_not_called()


def test_a_running_tick_after_the_stop_marker_clears_the_deadline(touch: Any) -> None:
    """A restarted worker's first tick upserts state=running with a NULL deadline over the old marker."""
    conn = FakeConn()
    run_loop(conn, stop_after_sleeps=1)

    worker.shutdown_requested = False
    beat = worker._Heartbeat(conn)  # type: ignore[arg-type]
    beat.tick()

    sql, _ = conn.running()[-1]
    assert "state = 'running'" in sql
    assert "expected_back_until = NULL" in sql
    assert conn.statements[-2] == conn.stopping()[0]
    assert conn.statements[-1] == conn.running()[-1]


def test_a_failing_stop_announcement_is_logged_not_raised(touch: Any, caplog: pytest.LogCaptureFixture) -> None:
    conn = FakeConn()
    worker.shutdown_requested = True
    conn.errors = [DatabaseError("gone")]

    with caplog.at_level(logging.WARNING, logger="worker"):
        run_loop(conn, stop_after_sleeps=None)

    assert "Could not announce the planned stop" in caplog.text
    assert conn.closed is True


def test_a_failing_tick_is_logged_and_the_next_tick_runs(touch: Any, caplog: pytest.LogCaptureFixture) -> None:
    conn = FakeConn()
    touch.side_effect = [RuntimeError("disk exploded"), None]

    with caplog.at_level(logging.ERROR, logger="worker"):
        run_loop(conn, stop_after_sleeps=worker.HEARTBEAT_INTERVAL + 1)

    assert touch.call_count == 2  # the thread survived the first tick
    assert len(conn.running()) == 1  # and the second one wrote its row
    assert caplog.text.count("Heartbeat tick failed") == 1
    assert len(conn.stopping()) == 1
    assert conn.closed is True


def test_a_stalled_worker_that_is_terminated_does_not_announce_a_planned_stop(
    touch: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic() - worker.STALL_LIMIT - 1)
    conn = FakeConn()

    with caplog.at_level(logging.INFO, logger="worker"):
        run_loop(conn, stop_after_sleeps=2)

    assert conn.statements == []  # no running row and, above all, no stopping row
    assert conn.stopping() == []
    assert "Not announcing a planned stop: the worker was stalled" in caplog.text
    assert conn.closed is True


def test_a_worker_that_resumed_after_a_stall_still_announces_its_stop(
    touch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    beat, conn = make_beat()
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic() - worker.STALL_LIMIT - 1)
    beat.tick()
    worker._mark_progress("cycle")
    beat.tick()

    assert beat.stall_reported is False
    beat.announce_planned_stop()

    assert len(conn.stopping()) == 1


# --------------------------------------------------------------------------
# Startup gate: no database row until the startup checks passed
# --------------------------------------------------------------------------


def test_a_tick_before_startup_finished_touches_the_file_but_writes_no_row(touch: Any) -> None:
    worker._startup_done.clear()
    beat, conn = make_beat()

    beat.tick()

    touch.assert_called_once_with()  # Toolforge's liveness must keep working during a slow startup
    assert conn.statements == []
    assert beat.first_write is True


def test_the_first_tick_after_startup_writes_the_row(touch: Any) -> None:
    worker._startup_done.clear()
    beat, conn = make_beat()
    beat.tick()

    worker._startup_done.set()
    beat.tick()

    ((_, params),) = conn.running()
    assert params[-1] == 1  # started_at is still set by the first real write


def test_a_planned_stop_before_startup_finished_writes_nothing(touch: Any) -> None:
    worker._startup_done.clear()
    conn = FakeConn()
    worker.shutdown_requested = True

    run_loop(conn, stop_after_sleeps=None)

    assert conn.statements == []
    assert conn.closed is True


def test_main_never_writes_a_running_row_when_the_startup_check_exits(
    touch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash-looping worker (face service model mismatch) must not look alive to the banner."""
    import threading

    worker._startup_done.clear()
    conn = FakeConn()
    monkeypatch.setattr(worker, "open_dedicated_connection", lambda: conn)
    monkeypatch.setattr(worker, "init_db", lambda **_: None)
    monkeypatch.setattr(worker, "_db_pool_size", lambda: 5)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--worker-id", "ml-worker-1"])

    def exit_after_the_first_heartbeat_tick() -> None:
        deadline = worker.time.monotonic() + 5
        while not touch.called and worker.time.monotonic() < deadline:
            worker.time.sleep(0.01)
        assert touch.called
        raise SystemExit(1)

    monkeypatch.setattr(worker, "_check_face_service_at_startup", exit_after_the_first_heartbeat_tick)

    with pytest.raises(SystemExit):
        worker.main()
    heartbeat = next(t for t in threading.enumerate() if t.name == "worker-heartbeat")
    worker.shutdown_requested = True  # what SIGTERM does
    heartbeat.join(timeout=10)

    assert not heartbeat.is_alive()
    assert conn.running() == []
    assert conn.stopping() == []


def test_main_marks_startup_done_before_the_loop_and_cleans_up_old_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    worker._startup_done.clear()
    seen: dict[str, Any] = {}
    for name in ("init_db", "_check_face_service_at_startup", "_log_project_diagnostics", "_release_resources"):
        monkeypatch.setattr(worker, name, lambda *a, **k: None)
    monkeypatch.setattr(worker, "_heartbeat_loop", lambda: None)
    monkeypatch.setattr(worker, "_shadow_prefilter_loop", lambda: None)
    monkeypatch.setattr(worker, "_db_pool_size", lambda: 5)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--worker-id", "ml-worker-1"])
    monkeypatch.setattr(
        worker, "_delete_retired_worker_rows", lambda: seen.setdefault("startup", worker._startup_done.is_set())
    )
    monkeypatch.setattr(worker, "_run_cycle", lambda: setattr(worker, "shutdown_requested", True))
    monkeypatch.setattr(worker, "_sleep_until_next_cycle", lambda: None)

    worker.main()

    assert worker._startup_done.is_set()
    assert seen == {"startup": True}


# --------------------------------------------------------------------------
# Retired worker rows
# --------------------------------------------------------------------------


def test_retired_rows_are_deleted_through_the_pool_except_our_own() -> None:
    with patch.object(worker, "execute_query", return_value=2) as query:
        worker._delete_retired_worker_rows()

    ((sql, params), kwargs) = query.call_args
    assert sql.startswith("DELETE FROM worker_heartbeats WHERE last_seen < NOW() - INTERVAL %s DAY")
    assert "worker_id <> %s" in sql
    assert params == (worker.RETIRED_WORKER_ROW_DAYS, "ml-worker-1")
    assert kwargs == {"fetch": False}


def test_a_database_error_while_removing_retired_rows_is_only_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.object(worker, "execute_query", side_effect=DatabaseError("down")),
        caplog.at_level(logging.WARNING, logger="worker"),
    ):
        worker._delete_retired_worker_rows()

    assert "Could not remove retired worker heartbeat rows" in caplog.text


# --------------------------------------------------------------------------
# The shadow-sweep thread does not count as progress
# --------------------------------------------------------------------------


def _run_in_thread(target: Any) -> None:
    import threading

    errors: list[BaseException] = []

    def body() -> None:
        try:
            target()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=body)
    thread.start()
    thread.join(timeout=10)
    assert not errors, errors


def test_an_exempt_thread_does_not_report_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_last_progress", 123.0)

    def body() -> None:
        worker._progress_exempt.on = True
        worker._mark_progress("x")

    _run_in_thread(body)

    assert worker._last_progress == 123.0
    assert worker._current_phase == "process_images"


def test_other_threads_still_report_progress_after_an_exempt_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_last_progress", 123.0)

    def exempt() -> None:
        worker._progress_exempt.on = True

    _run_in_thread(exempt)
    _run_in_thread(lambda: worker._mark_progress("project_thread"))

    assert worker._last_progress != 123.0
    assert worker._current_phase == "project_thread"


def test_the_sweep_does_not_hide_a_stalled_main_loop(touch: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_last_progress", worker.time.monotonic() - worker.STALL_LIMIT - 1)
    project = {"id": 7, "status": "completed", "prefilter_mode": "shadow"}

    def marking_prefilter(_project: dict[str, Any]) -> int:
        worker._mark_progress("prefilter")
        return 3

    def sweep() -> None:
        worker._progress_exempt.on = True
        with (
            patch.object(worker, "_liftwing_service_down", return_value=False),
            patch.object(worker, "_claim_shadow_project", return_value=project),
            patch.object(worker, "prefilter_images", side_effect=marking_prefilter),
            patch.object(worker, "_release_project"),
        ):
            assert worker._sweep_shadow_prefilter() == 3

    stale = worker._last_progress
    _run_in_thread(sweep)
    beat, conn = make_beat()
    beat.tick()

    assert worker._last_progress == stale
    assert beat.stall_reported is True
    assert conn.statements == []
    touch.assert_not_called()


def test_the_sweep_loop_exempts_its_own_thread_and_restores_it(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def sweep() -> int:
        seen.append(worker._progress_exempt.on)
        worker.shutdown_requested = True
        return 0

    monkeypatch.setattr(worker, "_sweep_shadow_prefilter", sweep)

    worker._shadow_prefilter_loop()

    assert seen == [True]
    assert worker._progress_exempt.on is False  # a thread that runs the loop is not exempt afterwards


def test_the_signal_handler_does_no_database_work() -> None:
    with patch.object(worker, "execute_query") as query, patch.object(worker, "open_dedicated_connection") as open_conn:
        worker.signal_handler(15, None)

    assert worker.shutdown_requested is True
    query.assert_not_called()
    open_conn.assert_not_called()


# --------------------------------------------------------------------------
# Integration: the SQL against the real worker_heartbeats table
# --------------------------------------------------------------------------


def _row(db_conn: Any) -> dict[str, Any]:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT state, last_seen, expected_back_until, last_progress, phase, started_at, version, "
            "NOW() AS db_now FROM worker_heartbeats WHERE worker_id = %s",
            ("ml-worker-1",),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.integration
def test_retired_worker_rows_are_deleted_and_current_ones_kept_against_mariadb(
    db_pool: Any, db_conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with db_conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO worker_heartbeats (worker_id, state, last_seen) VALUES (%s, 'running', NOW() - INTERVAL %s DAY)",
            [("renamed-old", 8), ("ml-worker-2", 0), ("ml-worker-1", 30)],
        )

    worker._delete_retired_worker_rows()

    with db_conn.cursor() as cur:
        cur.execute("SELECT worker_id FROM worker_heartbeats ORDER BY worker_id")
        remaining = [row["worker_id"] for row in cur.fetchall()]
    # the eight-day-old row of a renamed worker goes; a fresh row and this worker's own row stay
    assert remaining == ["ml-worker-1", "ml-worker-2"]


@pytest.mark.integration
def test_upsert_insert_update_stopping_then_running_against_mariadb(
    db_pool: Any, db_conn: Any, touch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db_pool.open_dedicated_connection()
    try:
        beat = worker._Heartbeat(conn)

        # insert
        monkeypatch.setattr(worker, "_current_phase", "bootstrap")
        beat.tick()
        row = _row(db_conn)
        assert row["state"] == "running"
        assert row["expected_back_until"] is None
        assert row["phase"] == "bootstrap"
        assert row["version"] == worker.PROJECT_VERSION
        assert row["started_at"] is not None
        assert row["last_progress"] is not None

        # update: a later tick refreshes the row but keeps started_at
        with db_conn.cursor() as cur:
            cur.execute(
                "UPDATE worker_heartbeats SET started_at = '2020-01-01 00:00:00', "
                "last_seen = '2020-01-01 00:00:00' WHERE worker_id = 'ml-worker-1'"
            )
        monkeypatch.setattr(worker, "_current_phase", "inference")
        beat.tick()
        row = _row(db_conn)
        assert row["phase"] == "inference"
        assert row["last_seen"] > row["started_at"]
        assert str(row["started_at"]) == "2020-01-01 00:00:00"

        # planned stop: state flips and the deadline is in the future
        beat.announce_planned_stop()
        row = _row(db_conn)
        assert row["state"] == "stopping"
        assert row["expected_back_until"] is not None
        seconds_left = (row["expected_back_until"] - row["db_now"]).total_seconds()
        assert worker.WORKER_RESTART_GRACE_SECONDS - 5 <= seconds_left <= worker.WORKER_RESTART_GRACE_SECONDS
        assert row["phase"] == "inference"

        # a restarted worker: its first tick is running again, deadline cleared, started_at reset
        restarted = worker._Heartbeat(conn)
        restarted.tick()
        row = _row(db_conn)
        assert row["state"] == "running"
        assert row["expected_back_until"] is None
        assert str(row["started_at"]) != "2020-01-01 00:00:00"

        # a stop announced before the worker ever wrote a running row inserts a stopping row
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM worker_heartbeats")
        worker._Heartbeat(conn).announce_planned_stop()
        row = _row(db_conn)
        assert row["state"] == "stopping"
        assert row["expected_back_until"] > row["db_now"]
    finally:
        conn.close()
