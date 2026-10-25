"""Tests for the worker's main loop, split into the functions ``main`` calls.

``main`` used to be one 356-line function that nothing could test without running the
whole process. Each piece (startup checks, the supervisor that watches project threads,
the inference drain, the sleep between cycles, shutdown) is exercised here directly.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

import worker
from database import DatabaseError


@pytest.fixture(autouse=True)
def _running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "shutdown_requested", False)
    monkeypatch.setattr(worker, "_worker_id", "test-worker")


def finished(value: Any = None, error: Exception | None = None) -> Future[Any]:
    future: Future[Any] = Future()
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)
    return future


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------


def test_worker_id_comes_from_the_command_line() -> None:
    assert worker._parse_worker_id(["worker.py", "--worker-id", "ml-worker-2"]) == "ml-worker-2"


def test_worker_id_defaults_to_hostname_and_pid() -> None:
    with (
        patch.object(worker.socket, "gethostname", return_value="pod-1"),
        patch.object(worker.os, "getpid", return_value=7),
    ):
        assert worker._parse_worker_id(["worker.py"]) == "pod-1-7"
        assert worker._parse_worker_id(["worker.py", "--worker-id"]) == "pod-1-7"


@pytest.mark.parametrize("bad", ["../etc", "a/b", "id with space", "x;rm"])
def test_worker_id_that_could_traverse_the_heartbeat_path_is_refused(bad: str) -> None:
    with pytest.raises(SystemExit):
        worker._parse_worker_id(["worker.py", "--worker-id", bad])


def test_pool_size_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKIVISAGE_DB_POOL_SIZE", "4")

    assert worker._db_pool_size() == 4


def test_pool_size_is_derived_and_capped_at_six(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKIVISAGE_DB_POOL_SIZE", raising=False)

    assert worker._db_pool_size() == min(worker.MAX_CONCURRENT_PROJECTS * worker.IMAGE_THREADS + 3, 6)


def test_a_pool_above_the_toolsdb_budget_is_allowed_but_warned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("WIKIVISAGE_DB_POOL_SIZE", "9")

    with caplog.at_level(logging.WARNING, logger="worker"):
        assert worker._db_pool_size() == 9

    assert "max_user_connections" in caplog.text


def test_a_ready_face_service_is_checked_for_the_expected_model() -> None:
    with (
        patch.object(worker.face_client, "is_healthy", return_value=True),
        patch.object(worker, "_check_face_service_model") as check,
    ):
        worker._check_face_service_at_startup()

    check.assert_called_once_with()


def test_an_unready_face_service_only_warns(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.object(worker.face_client, "is_healthy", return_value=False),
        patch.object(worker, "_check_face_service_model") as check,
        caplog.at_level(logging.WARNING, logger="worker"),
    ):
        worker._check_face_service_at_startup()

    check.assert_not_called()
    assert "Face service not ready" in caplog.text


def test_project_diagnostics_log_every_project(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {
            "id": 3,
            "wikidata_qid": "Q42",
            "status": "active",
            "min_confirmed": 5,
            "faces_confirmed": 2,
            "actual_target_faces": 4,
            "actual_non_target": 1,
            "unclassified_faces": 9,
        }
    ]
    with patch.object(worker, "execute_query", return_value=rows), caplog.at_level(logging.INFO, logger="worker"):
        worker._log_project_diagnostics()

    assert "Project 3 (Q42)" in caplog.text
    assert "unclassified=9" in caplog.text


def test_a_failing_diagnostic_query_never_stops_startup(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.object(worker, "execute_query", side_effect=RuntimeError("boom")),
        caplog.at_level(logging.WARNING, logger="worker"),
    ):
        worker._log_project_diagnostics()

    assert "Diagnostic query failed" in caplog.text


# --------------------------------------------------------------------------
# Purging deleted projects
# --------------------------------------------------------------------------


def test_purge_archives_counts_before_deleting() -> None:
    cursor = MagicMock()
    cursor.rowcount = 2

    assert worker._archive_and_purge(MagicMock(), cursor) == 2

    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert statements[0].startswith("INSERT INTO user_stats")
    assert statements[1] == "DELETE FROM projects WHERE status = 'deleted'"


def test_purge_reports_what_it_removed(caplog: pytest.LogCaptureFixture) -> None:
    with patch.object(worker, "execute_transaction", return_value=3), caplog.at_level(logging.INFO, logger="worker"):
        worker._purge_deleted_projects()

    assert "Purged 3 soft-deleted project(s)" in caplog.text


def test_a_failed_purge_is_logged_and_the_cycle_continues(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch.object(worker, "execute_transaction", side_effect=DatabaseError("locked")),
        caplog.at_level(logging.WARNING, logger="worker"),
    ):
        worker._purge_deleted_projects()

    assert "Failed to purge soft-deleted projects" in caplog.text


# --------------------------------------------------------------------------
# Wake-up signal
# --------------------------------------------------------------------------


def test_a_wake_up_file_is_consumed_exactly_once(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    wake = tmp_path / ".wake"
    wake.write_text("")
    monkeypatch.setattr(worker, "WAKE_FILE_PATH", str(wake))

    assert worker._consume_wake_signal() is True
    assert not wake.exists()
    assert worker._consume_wake_signal() is False


def test_a_wake_up_file_that_cannot_be_removed_still_counts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    wake = tmp_path / ".wake"
    wake.write_text("")
    monkeypatch.setattr(worker, "WAKE_FILE_PATH", str(wake))

    with patch.object(worker.os, "remove", side_effect=OSError("read-only")):
        assert worker._consume_wake_signal() is True


# --------------------------------------------------------------------------
# The supervisor
# --------------------------------------------------------------------------


@pytest.fixture
def quiet_cycle(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace everything the supervisor calls so only its own decisions remain."""
    mocks = {
        name: MagicMock(return_value=False if name == "_consume_wake_signal" else None)
        for name in (
            "_consume_wake_signal",
            "_keep_alive_during_cycle",
            "_process_sdc_writes",
        )
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(worker, name, mock)
    monkeypatch.setattr(worker.time, "sleep", MagicMock())
    return mocks


def test_finished_futures_are_handed_over_and_forgotten(quiet_cycle: dict[str, MagicMock]) -> None:
    futures = {finished("a"): 1, finished(error=RuntimeError("x")): 2}
    seen: list[tuple[Any, int]] = []

    worker._supervise(
        futures,
        on_wake=lambda _f: pytest.fail("no wake-up was requested"),
        on_done=lambda future, project_id: seen.append((future.exception() or future.result(), project_id)),
    )

    assert futures == {}
    assert sorted(project_id for _, project_id in seen) == [1, 2]


def test_a_wake_up_runs_the_hook_with_the_live_futures(quiet_cycle: dict[str, MagicMock]) -> None:
    quiet_cycle["_consume_wake_signal"].side_effect = [True, False, False]
    futures = {finished(): 1}
    hook = MagicMock()

    worker._supervise(futures, on_wake=hook, on_done=lambda *_: None)

    hook.assert_called_once()
    assert hook.call_args.args[0] is futures


def test_a_hook_may_add_work_while_waiting(quiet_cycle: dict[str, MagicMock]) -> None:
    quiet_cycle["_consume_wake_signal"].side_effect = [True, False, False, False]
    handled: list[int] = []

    def add_project(current: dict[Future[Any], int]) -> None:
        current[finished()] = 99

    worker._supervise({finished(): 1}, on_wake=add_project, on_done=lambda _f, project_id: handled.append(project_id))

    assert sorted(handled) == [1, 99]


def test_nothing_is_awaited_after_shutdown(quiet_cycle: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "shutdown_requested", True)
    pending: Future[Any] = Future()
    futures = {pending: 1}

    worker._supervise(futures, on_wake=lambda _f: None, on_done=lambda *_: pytest.fail("never finished"))

    assert futures == {pending: 1}


def test_a_running_project_waits_one_second_at_a_time(quiet_cycle: dict[str, MagicMock]) -> None:
    pending: Future[Any] = Future()
    futures = {pending: 1}
    sleeps = {"n": 0}

    def sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] == 3:
            pending.set_result(None)

    with patch.object(worker.time, "sleep", side_effect=sleep) as sleeper:
        worker._supervise(futures, on_wake=lambda _f: None, on_done=lambda *_: None)

    assert sleeper.call_args_list == [call(1)] * 3


def _clock(*steps: float):
    """A time.time() that returns the given values in order, then keeps the last."""
    values = list(steps)
    return lambda: values.pop(0) if len(values) > 1 else values[0]


def test_heartbeat_and_claims_are_refreshed_every_minute(quiet_cycle: dict[str, MagicMock]) -> None:
    pending: Future[Any] = Future()
    sleeps = {"n": 0}

    def sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] == 2:
            pending.set_result(None)

    # start, first iteration (+30 s: too early), second iteration (+61 s), timestamp after the refresh
    with (
        patch.object(worker.time, "time", side_effect=_clock(0, 30, 30, 61, 61, 61, 61, 61)),
        patch.object(worker.time, "sleep", side_effect=sleep),
    ):
        worker._supervise({pending: 1}, on_wake=lambda _f: None, on_done=lambda *_: None)

    quiet_cycle["_keep_alive_during_cycle"].assert_called_once_with()


def test_sdc_requests_are_polled_every_ten_seconds(quiet_cycle: dict[str, MagicMock]) -> None:
    """The wake-up file does not cross pods, so the poll is what a user's click depends on."""
    pending: Future[Any] = Future()
    sleeps = {"n": 0}

    def sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] == 2:
            pending.set_result(None)

    with (
        patch.object(worker.time, "time", side_effect=_clock(0, 5, 5, 11, 11, 11, 11)),
        patch.object(worker.time, "sleep", side_effect=sleep),
    ):
        worker._supervise({pending: 1}, on_wake=lambda _f: None, on_done=lambda *_: None)

    quiet_cycle["_process_sdc_writes"].assert_called_once_with()


def test_keep_alive_survives_a_database_hiccup() -> None:
    with (
        patch.object(worker, "execute_query", side_effect=DatabaseError("flapped")),
        patch.object(worker, "_touch_heartbeat_file") as touch,
        patch.object(worker, "_refresh_claims") as refresh,
        patch.object(worker, "_probe_liftwing_if_due") as probe,
        patch.object(worker, "_run_new_inference_projects") as infer,
    ):
        worker._keep_alive_during_cycle()

    touch.assert_not_called()
    refresh.assert_called_once_with()
    probe.assert_called_once_with()
    infer.assert_called_once_with(1)


# --------------------------------------------------------------------------
# Inference drain
# --------------------------------------------------------------------------


def test_only_new_inference_projects_are_run_and_the_rest_are_released() -> None:
    projects = [{"id": 1, "status": "completed"}, {"id": 2}, {"id": 3}]
    seen: set[int] = {3}
    claims = iter([projects, [], []])

    with (
        patch.object(worker, "_claim_inference_projects", side_effect=lambda _n: next(claims)),
        patch.object(worker, "_infer_and_release", return_value=4) as infer,
        patch.object(worker, "_release_project") as release,
        patch.object(worker, "_supervise", wraps=worker._supervise),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker.time, "sleep"),
    ):
        worker._drain_inference_queue(active_ids={2}, seen_ids=seen)

    assert sorted(c.args[0]["id"] for c in infer.call_args_list) == [1]
    release.assert_called_once_with(3)  # id 2 belongs to the active pipeline, which releases it
    assert seen == {1, 3}


def test_the_drain_keeps_claiming_until_nothing_new_is_left() -> None:
    batches = iter([[{"id": 1}], [{"id": 2}], [{"id": 2}]])  # 2 stays eligible; it must not loop forever
    ran: list[int] = []

    with (
        patch.object(worker, "_claim_inference_projects", side_effect=lambda _n: next(batches)),
        patch.object(worker, "_infer_and_release", side_effect=lambda p: ran.append(p["id"]) or 1),
        patch.object(worker, "_release_project") as release,
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker.time, "sleep"),
    ):
        worker._drain_inference_queue(active_ids=set(), seen_ids=set())

    assert ran == [1, 2]
    release.assert_called_once_with(2)


def test_a_failed_inference_is_logged_and_the_drain_goes_on(caplog: pytest.LogCaptureFixture) -> None:
    batches = iter([[{"id": 5}], []])

    with (
        patch.object(worker, "_claim_inference_projects", side_effect=lambda _n: next(batches)),
        patch.object(worker, "_infer_and_release", side_effect=RuntimeError("numpy blew up")),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker.time, "sleep"),
        caplog.at_level(logging.ERROR, logger="worker"),
    ):
        worker._drain_inference_queue(active_ids=set(), seen_ids=set())

    assert "Inference failed for project 5: numpy blew up" in caplog.text


def test_a_wake_up_during_inference_services_sdc_and_new_inference_projects() -> None:
    batches = iter([[{"id": 5}], []])
    wake = iter([True, False, False])

    with (
        patch.object(worker, "_claim_inference_projects", side_effect=lambda _n: next(batches)),
        patch.object(worker, "_infer_and_release", return_value=1),
        patch.object(worker, "_consume_wake_signal", side_effect=lambda: next(wake, False)),
        patch.object(worker, "_process_sdc_writes") as sdc,
        patch.object(worker, "_run_new_inference_projects") as new_inference,
        patch.object(worker.time, "sleep"),
    ):
        worker._drain_inference_queue(active_ids=set(), seen_ids=set())

    sdc.assert_called_once_with()
    new_inference.assert_called_once_with(1)


# --------------------------------------------------------------------------
# Active projects
# --------------------------------------------------------------------------


def test_every_active_project_is_processed_and_marked_seen() -> None:
    processed: list[int] = []
    seen: set[int] = set()

    with (
        patch.object(worker, "_process_and_release", side_effect=lambda p: processed.append(p["id"])),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker.time, "sleep"),
    ):
        worker._process_active_projects([{"id": 1}, {"id": 2}], seen)

    assert sorted(processed) == [1, 2]
    assert seen == {1, 2}


def test_no_active_projects_means_no_threads() -> None:
    with patch.object(worker, "ThreadPoolExecutor") as pool:
        worker._process_active_projects([], set())

    pool.assert_not_called()


def test_a_failing_project_is_logged_without_stopping_the_others(caplog: pytest.LogCaptureFixture) -> None:
    def process(project: dict[str, Any]) -> None:
        if project["id"] == 1:
            raise RuntimeError("category gone")

    with (
        patch.object(worker, "_process_and_release", side_effect=process),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker.time, "sleep"),
        caplog.at_level(logging.ERROR, logger="worker"),
    ):
        worker._process_active_projects([{"id": 1}, {"id": 2}], set())

    assert "Project 1 processing failed: category gone" in caplog.text


def test_a_wake_up_mid_cycle_adds_new_projects_but_not_ones_already_running() -> None:
    processed: list[int] = []
    wake = iter([True, False, False])
    claim = MagicMock(return_value=[{"id": 1}, {"id": 7}])  # 1 is already running

    with (
        patch.object(worker, "_process_and_release", side_effect=lambda p: processed.append(p["id"])),
        patch.object(worker, "_consume_wake_signal", side_effect=lambda: next(wake, False)),
        patch.object(worker, "_claim_active_projects", claim),
        patch.object(worker, "_process_sdc_writes") as sdc,
        patch.object(worker, "_run_new_inference_projects") as new_inference,
        patch.object(worker.time, "sleep"),
    ):
        worker._process_active_projects([{"id": 1}], set())

    assert sorted(processed) == [1, 7]
    assert 1 <= claim.call_args.args[0] <= worker.MAX_CONCURRENT_PROJECTS
    sdc.assert_called_once_with()
    new_inference.assert_called_once_with(1)


def test_no_projects_are_claimed_mid_cycle_when_every_slot_is_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "MAX_CONCURRENT_PROJECTS", 1)
    release = threading.Event()
    wake = iter([True, False])
    claim = MagicMock(return_value=[])

    def process(_project: dict[str, Any]) -> None:
        release.wait(timeout=5)

    def consume() -> bool:
        value = next(wake, False)
        if not value:
            release.set()
        return value

    with (
        patch.object(worker, "_process_and_release", side_effect=process),
        patch.object(worker, "_consume_wake_signal", side_effect=consume),
        patch.object(worker, "_claim_active_projects", claim),
        patch.object(worker, "_process_sdc_writes"),
        patch.object(worker, "_run_new_inference_projects"),
        patch.object(worker.time, "sleep"),
    ):
        worker._process_active_projects([{"id": 1}], set())

    claim.assert_not_called()


# --------------------------------------------------------------------------
# One cycle, the sleep between cycles, shutdown, main
# --------------------------------------------------------------------------


def test_a_cycle_runs_its_stages_in_order() -> None:
    order: list[str] = []

    def stage(name: str, result: Any = None):
        return MagicMock(side_effect=lambda *a, **k: order.append(name) or result)

    with (
        patch.object(worker, "execute_query", stage("heartbeat")),
        patch.object(worker, "_touch_heartbeat_file", stage("touch")),
        patch.object(worker, "_probe_liftwing_if_due", stage("probe")),
        patch.object(worker, "_purge_deleted_projects", stage("purge")),
        patch.object(worker, "_process_sdc_writes", stage("sdc")),
        patch.object(worker, "_claim_active_projects", stage("claim", [{"id": 4}])),
        patch.object(worker, "_drain_inference_queue", stage("inference")) as drain,
        patch.object(worker, "_process_active_projects", stage("active")) as process,
    ):
        worker._run_cycle()

    assert order == ["heartbeat", "touch", "probe", "purge", "sdc", "claim", "inference", "active"]
    active_ids, seen_ids = drain.call_args.args
    assert active_ids == {4}
    assert process.call_args.args == ([{"id": 4}], seen_ids)  # the same set, so inference and active share it


def test_the_cycle_lets_a_database_error_reach_the_main_loop() -> None:
    with patch.object(worker, "execute_query", side_effect=DatabaseError("down")), pytest.raises(DatabaseError):
        worker._run_cycle()


def test_sleep_ends_early_for_a_wake_up_signal() -> None:
    with (
        patch.object(worker, "POLL_INTERVAL", 60),
        patch.object(worker.random, "randint", return_value=0),
        patch.object(worker, "_consume_wake_signal", side_effect=[False, False, True]),
        patch.object(worker.time, "sleep") as sleeper,
    ):
        worker._sleep_until_next_cycle()

    assert sleeper.call_count == 2


def test_sleep_ends_early_when_an_sdc_write_finished() -> None:
    with (
        patch.object(worker, "POLL_INTERVAL", 60),
        patch.object(worker.random, "randint", return_value=0),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker, "_process_sdc_writes", return_value=3) as sdc,
        patch.object(worker.time, "time", side_effect=_clock(0, 11)),
        patch.object(worker.time, "sleep") as sleeper,
    ):
        worker._sleep_until_next_cycle()

    sdc.assert_called_once_with()
    assert sleeper.call_count == 0


def test_sleep_lasts_one_poll_interval_plus_jitter_when_nothing_happens() -> None:
    with (
        patch.object(worker, "POLL_INTERVAL", 5),
        patch.object(worker.random, "randint", return_value=3),
        patch.object(worker, "_consume_wake_signal", return_value=False),
        patch.object(worker, "_process_sdc_writes", return_value=0),
        patch.object(worker.time, "time", return_value=0.0),
        patch.object(worker.time, "sleep") as sleeper,
    ):
        worker._sleep_until_next_cycle()

    assert sleeper.call_count == 8


def test_sleep_returns_at_once_after_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "shutdown_requested", True)

    with patch.object(worker.time, "sleep") as sleeper:
        worker._sleep_until_next_cycle()

    sleeper.assert_not_called()


def test_shutdown_stops_the_sweep_before_releasing_claims_and_pools() -> None:
    order: list[str] = []
    sweep = MagicMock()
    sweep.join.side_effect = lambda timeout: order.append("join")
    sweep.is_alive.return_value = False

    with (
        patch.object(worker, "_release_all_claims", side_effect=lambda: order.append("claims")),
        patch.object(worker.face_client, "reset_session", side_effect=lambda: order.append("face")),
        patch.object(worker.liftwing_client, "reset_client", side_effect=lambda: order.append("liftwing")),
        patch.object(worker, "close_pool", side_effect=lambda: order.append("pool")),
    ):
        worker._release_resources(sweep)

    assert order == ["join", "claims", "face", "liftwing", "pool"]
    assert worker.shutdown_requested is True
    sweep.join.assert_called_once_with(timeout=worker.SWEEP_JOIN_TIMEOUT)


def test_a_sweep_that_outlives_its_join_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    sweep = MagicMock()
    sweep.is_alive.return_value = True

    with (
        patch.object(worker, "_release_all_claims"),
        patch.object(worker.face_client, "reset_session"),
        patch.object(worker.liftwing_client, "reset_client"),
        patch.object(worker, "close_pool"),
        caplog.at_level(logging.WARNING, logger="worker"),
    ):
        worker._release_resources(sweep)

    assert "still running" in caplog.text


@pytest.fixture
def main_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks = {
        "init_db": MagicMock(),
        "_check_face_service_at_startup": MagicMock(),
        "_log_project_diagnostics": MagicMock(),
        "_sleep_until_next_cycle": MagicMock(),
        "_release_resources": MagicMock(),
        "_shadow_prefilter_loop": MagicMock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(worker, name, mock)
    monkeypatch.setattr(worker.signal, "signal", MagicMock())
    monkeypatch.setattr(worker.time, "sleep", MagicMock())
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--worker-id", "ml-worker-1"])
    monkeypatch.setattr(worker, "_db_pool_size", lambda: 6)
    return mocks


def test_main_runs_cycles_until_shutdown_then_releases_everything(
    main_env: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    cycles = {"n": 0}

    def cycle() -> None:
        cycles["n"] += 1
        if cycles["n"] == 2:
            monkeypatch.setattr(worker, "shutdown_requested", True)

    monkeypatch.setattr(worker, "_run_cycle", cycle)

    worker.main()

    assert worker._worker_id == "ml-worker-1"
    main_env["init_db"].assert_called_once_with(pool_size=6)
    main_env["_check_face_service_at_startup"].assert_called_once_with()
    assert cycles["n"] == 2
    assert main_env["_sleep_until_next_cycle"].call_count == 2
    main_env["_release_resources"].assert_called_once()


def test_main_survives_a_database_error_and_an_unexpected_one(
    main_env: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = iter([DatabaseError("down"), RuntimeError("odd"), None])

    def cycle() -> None:
        outcome = next(outcomes)
        if outcome is None:
            monkeypatch.setattr(worker, "shutdown_requested", True)
        else:
            raise outcome

    monkeypatch.setattr(worker, "_run_cycle", cycle)

    worker.main()

    assert worker.time.sleep.call_args_list == [call(10), call(10)]  # type: ignore[attr-defined]
    main_env["_release_resources"].assert_called_once()


def test_main_exits_when_the_database_cannot_be_initialised(main_env: dict[str, MagicMock]) -> None:
    main_env["init_db"].side_effect = RuntimeError("no credentials")

    with pytest.raises(SystemExit) as exit_info:
        worker.main()

    assert exit_info.value.code == 1
    main_env["_release_resources"].assert_not_called()


def test_main_releases_resources_even_if_a_cycle_is_interrupted(
    main_env: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "_run_cycle", MagicMock(side_effect=KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        worker.main()

    main_env["_release_resources"].assert_called_once()
