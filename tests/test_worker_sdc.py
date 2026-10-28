"""Behavioural tests for ``worker.write_sdc_claims``.

The function is the only code that edits Commons on a user's behalf, and its
error paths (token refresh, aborts, cancellation, the removal phase) were the
least covered code in the repository. These tests drive the whole run through a
scripted fake database and a scripted Commons API, so they describe observable
behaviour (what is written, what the project row says afterwards) and survive
restructuring of the function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import worker
from config import SDC_AUTH_EXPIRED_MESSAGE, SDC_AUTH_UNAVAILABLE_MESSAGE, SDC_REAUTH_MESSAGES
from database import DatabaseError

QID = "Q42"
PAGE = 9999
ABORT_SQL = "UPDATE projects SET sdc_write_requested = 0"
GUID = "M9999$abc-123"
CLAIM_PRESENT = {"claims": {"P180": [{"id": GUID, "mainsnak": {"datavalue": {"value": {"id": QID}}}}]}}
AUTH_ERROR = {"error": {"code": "badtoken", "info": "Invalid CSRF token"}}


def _face(face_id: int = 100, page: int = PAGE) -> dict[str, int]:
    return {"face_id": face_id, "image_id": face_id + 100, "commons_page_id": page}


class World:
    """One scripted run: fake database, fake Commons API, fake token helpers."""

    def __init__(self) -> None:
        self.project: dict[str, Any] = {"id": 5, "user_id": 1, "wikidata_qid": QID}
        self.calls: list[tuple[str, Any]] = []
        self.api_calls: list[tuple[str, dict[str, Any], dict[str, str] | None]] = []
        self.write_batches: list[list[dict[str, int]]] | Callable[[], list[dict[str, int]]] = []
        self.removal_batches: list[list[dict[str, int]]] = []
        self.requested: Callable[[], int] = lambda: 1
        self.live: Callable[[int], dict[str, Any]] = lambda face_id: {"is_target": 1, "superseded_by": None}
        self.claim_insert = 1
        self.reclaim = 1
        self.sleeps: list[float] = []
        self.existing_claim: list[dict[str, Any]] = []
        self.still_pending: list[dict[str, int]] = [{"one": 1}]
        self.pending_after_shutdown = 0
        self.api_script: dict[str, list[Any]] = {"wbgetclaims": [], "wbeditentity": [], "wbremoveclaims": []}
        self.token_script: list[str | None] = []
        self.csrf_script: list[Any] = []
        self.refresh_calls = 0
        self.csrf_calls = 0
        self.then_shutdown_after_edit = False

    # -- fake database ----------------------------------------------------
    def db(self, sql: str, params: Any = None, fetch: bool = True) -> Any:
        text = " ".join(sql.split())
        self.calls.append((text, params))
        if text.startswith("SELECT sdc_write_requested FROM projects"):
            return [{"sdc_write_requested": self.requested()}]
        if text.startswith("SELECT f.id as face_id"):
            if callable(self.write_batches):
                return self.write_batches()
            return self.write_batches.pop(0) if self.write_batches else []
        if text.startswith("SELECT is_target, superseded_by FROM faces"):
            return [self.live(params[0])]
        if text.startswith("INSERT IGNORE INTO sdc_claims"):
            return self.claim_insert
        if text.startswith("SELECT project_id, written_at, claimed_at FROM sdc_claims"):
            return self.existing_claim
        if text.startswith("UPDATE sdc_claims SET project_id"):
            return self.reclaim
        if text.startswith("SELECT DISTINCT i.commons_page_id"):
            return self.removal_batches.pop(0) if self.removal_batches else []
        if text.startswith("SELECT 1 FROM faces f JOIN images i"):
            return self.still_pending
        if text.startswith("SELECT EXISTS"):
            return [{"pending": self.pending_after_shutdown}]
        return [] if fetch else 1

    # -- fake Commons API and token helpers -------------------------------
    def api(self, url: str, params=None, headers=None, method="get", data=None, timeout=30) -> MagicMock:
        body = data if data is not None else params
        action = body["action"]
        self.api_calls.append((action, body, headers))
        queue = self.api_script[action]
        payload = queue.pop(0) if queue else ({"claims": {}} if action == "wbgetclaims" else {"success": 1})
        if isinstance(payload, Exception):
            raise payload
        response = MagicMock()
        response.json.return_value = payload
        if action == "wbeditentity" and self.then_shutdown_after_edit:
            worker.shutdown_requested = True
        return response

    def refresh(self, user_id: int) -> str | None:
        self.refresh_calls += 1
        return self.token_script.pop(0) if self.token_script else f"token-{self.refresh_calls}"

    def csrf(self, access_token: str) -> str:
        self.csrf_calls += 1
        item = self.csrf_script.pop(0) if self.csrf_script else f"csrf-{self.csrf_calls}"
        if isinstance(item, Exception):
            raise item
        return item

    # -- running and inspecting -------------------------------------------
    def run(self) -> int:
        with (
            patch.object(worker, "execute_query", side_effect=self.db),
            patch.object(worker, "_api_request", side_effect=self.api),
            patch.object(worker, "_refresh_worker_token", side_effect=self.refresh),
            patch.object(worker, "_get_csrf_token", side_effect=self.csrf),
            patch.object(worker.time, "sleep", side_effect=self.sleeps.append),
            patch.object(worker, "shutdown_requested", False),
        ):
            return worker.write_sdc_claims(self.project)

    def actions(self) -> list[str]:
        return [action for action, _, _ in self.api_calls]

    def sql_like(self, fragment: str) -> list[tuple[str, Any]]:
        return [(sql, params) for sql, params in self.calls if fragment in sql]

    def project_updates(self) -> list[str | None]:
        """Error message of every statement that ends the run (abort or final cleanup)."""
        messages: list[str | None] = []
        for sql, params in self.calls:
            if not sql.startswith(ABORT_SQL):
                continue
            if "sdc_write_error = %s" in sql:
                messages.append(params[0])
            else:
                messages.append(sql.split("sdc_write_error = '")[1].split("'")[0])
        return messages


@pytest.fixture
def world() -> World:
    return World()


def two_passes(face: dict[str, int]) -> Callable[[], list[dict[str, int]]]:
    """The same face is selected on the first two passes (it stayed unwritten), then none."""
    batches = [[face], [face]]
    return lambda: batches.pop(0) if batches else []


# --------------------------------------------------------------------------
# Start-up failures
# --------------------------------------------------------------------------


def test_no_valid_token_aborts_before_any_api_call(world: World) -> None:
    world.token_script = [None]

    assert world.run() == 0

    assert world.api_calls == []
    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]


def _run_with_the_real_token_refresh(world: World, user_rows: list[dict[str, Any]], exchange: Any) -> int:
    """write_sdc_claims with ``_refresh_worker_token`` itself in play; only the token endpoint is scripted."""

    def db(sql: str, params: Any = None, fetch: bool = True) -> Any:
        if "FROM users WHERE id" in sql:
            return user_rows
        return world.db(sql, params, fetch)

    with (
        patch.object(worker, "execute_query", side_effect=db),
        patch.object(worker, "_api_request", side_effect=world.api),
        patch.object(worker, "_get_csrf_token", side_effect=world.csrf),
        patch.object(worker, "_get_redis_client", return_value=None),
        patch.object(worker, "_oauth_exchange", side_effect=exchange),
        patch("oauth_refresh.decrypt_token", side_effect=lambda token: token),
        patch.object(worker, "shutdown_requested", False),
    ):
        return worker.write_sdc_claims(world.project)


def _expired_user() -> dict[str, Any]:
    from datetime import UTC, datetime, timedelta

    return {
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "token_expires_at": datetime.now(UTC) - timedelta(hours=1),
    }


def test_a_refused_refresh_token_stores_the_reauthorize_message(world: World) -> None:
    from oauthlib.oauth2.rfc6749.errors import InvalidGrantError

    assert _run_with_the_real_token_refresh(world, [_expired_user()], InvalidGrantError()) == 0

    assert world.api_calls == []
    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]


def test_a_login_service_timeout_stores_the_try_again_message_without_a_reauthorize_button(world: World) -> None:
    import requests

    assert _run_with_the_real_token_refresh(world, [_expired_user()], requests.Timeout("slow")) == 0

    assert world.api_calls == []
    assert world.project_updates() == [SDC_AUTH_UNAVAILABLE_MESSAGE]
    assert SDC_AUTH_UNAVAILABLE_MESSAGE not in SDC_REAUTH_MESSAGES


def test_a_login_service_5xx_is_transient_too(world: World) -> None:
    import requests

    response = requests.Response()
    response.status_code = 503

    exc = requests.HTTPError("503", response=response)
    _run_with_the_real_token_refresh(world, [_expired_user()], exc)

    assert world.project_updates() == [SDC_AUTH_UNAVAILABLE_MESSAGE]


def test_an_invalid_grant_http_reply_is_permanent(world: World) -> None:
    import requests

    response = requests.Response()
    response.status_code = 400
    response._content = b'{"error": "invalid_grant"}'

    _run_with_the_real_token_refresh(world, [_expired_user()], requests.HTTPError("400", response=response))

    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]


def test_a_missing_user_row_or_refresh_token_counts_as_permanent(world: World) -> None:
    _run_with_the_real_token_refresh(world, [], RuntimeError("never called"))
    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]

    world.calls.clear()
    no_refresh_token = {**_expired_user(), "refresh_token": ""}
    _run_with_the_real_token_refresh(world, [no_refresh_token], RuntimeError("never called"))
    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]


def test_an_earlier_transient_failure_does_not_leak_into_the_next_run(world: World) -> None:
    import requests

    _run_with_the_real_token_refresh(world, [_expired_user()], requests.Timeout("slow"))
    world.calls.clear()

    world.token_script = [None]  # a patched refresh that fails without saying why
    world.run()

    assert world.project_updates() == [SDC_AUTH_EXPIRED_MESSAGE]


def test_csrf_failure_at_start_aborts(world: World) -> None:
    world.csrf_script = [RuntimeError("no csrf")]

    assert world.run() == 0

    assert world.project_updates() == ["Failed to get CSRF token"]


def test_invalid_qid_aborts(world: World) -> None:
    world.project["wikidata_qid"] = "QX"

    assert world.run() == 0

    assert world.project_updates() == ["Invalid QID format"]


def test_the_requesting_member_is_the_editor(world: World) -> None:
    world.project["sdc_write_user_id"] = 77
    seen: list[int] = []
    world.refresh = lambda user_id: seen.append(user_id) or "token"  # type: ignore[method-assign]

    world.run()

    assert seen == [77]


# --------------------------------------------------------------------------
# Write phase
# --------------------------------------------------------------------------


def test_one_face_is_written_and_recorded(world: World) -> None:
    world.write_batches = [[_face()]]

    assert world.run() == 1

    assert world.actions() == ["wbgetclaims", "wbeditentity"]
    edit = world.api_calls[1][1]
    assert edit["id"] == f"M{PAGE}"
    assert edit["token"] == "csrf-1"
    assert '"numeric-id": 42' in edit["data"]
    assert world.api_calls[1][2] == {"Authorization": "Bearer token-1"}
    written = world.sql_like("UPDATE faces SET sdc_written = 1 WHERE id = %s AND is_target = 1")
    assert [params for _, params in written] == [(100,)]
    assert world.project_updates() == [None]


def test_cancel_before_a_batch_stops_without_touching_commons(world: World) -> None:
    world.requested = lambda: 0
    world.write_batches = [[_face()]]

    assert world.run() == 0

    assert world.api_calls == []
    assert world.project_updates() == []


def test_a_failed_cancel_check_does_not_stop_the_write(world: World) -> None:
    answers: list[Any] = [DatabaseError("flapped"), 1, 1, 1]

    def requested() -> int:
        item = answers.pop(0) if answers else 1
        if isinstance(item, Exception):
            raise item
        return item

    world.requested = requested
    world.write_batches = [[_face()]]

    assert world.run() == 1


def test_stop_button_is_noticed_within_five_faces(world: World) -> None:
    """The batch holds up to 50 faces; a stop must not let all of them through."""
    world.write_batches = [[_face(100 + n, PAGE + n) for n in range(7)]]
    checks = iter([1, 0])  # outer check, then the check made when the fifth face starts
    world.requested = lambda: next(checks, 0)

    assert world.run() == 4

    assert world.actions().count("wbeditentity") == 4
    assert len(world.sql_like("worker_claimed_at = NOW()")) == 1
    assert world.project_updates() == []


def test_a_failed_mid_batch_cancel_check_does_not_stop_the_write(world: World) -> None:
    world.write_batches = [[_face(100 + n, PAGE + n) for n in range(7)]]
    answers: list[Any] = [1, DatabaseError("flapped")]

    def requested() -> int:
        item = answers.pop(0) if answers else 1
        if isinstance(item, Exception):
            raise item
        return item

    world.requested = requested

    assert world.run() == 7


def test_shutdown_mid_batch_reports_edits_still_pending(world: World) -> None:
    world.write_batches = [[_face(100 + n, PAGE + n) for n in range(3)]]
    world.then_shutdown_after_edit = True
    world.pending_after_shutdown = 1

    assert world.run() == 1

    assert world.actions().count("wbeditentity") == 1
    assert world.project_updates() == [
        "Worker shutdown left Commons edits pending. Edits already sent were kept. Send the remaining edits again."
    ]


def test_shutdown_after_the_last_edit_is_not_reported_as_an_error(world: World) -> None:
    world.write_batches = [[_face()]]
    world.then_shutdown_after_edit = True
    world.pending_after_shutdown = 0

    assert world.run() == 1

    assert world.project_updates() == [None]


def test_a_missing_dedup_row_skips_the_face_for_this_pass(world: World) -> None:
    world.write_batches = [[_face()]]
    world.claim_insert = 0
    world.existing_claim = []

    assert world.run() == 0

    assert world.api_calls == []


def test_claim_already_written_elsewhere_marks_the_face_written(world: World) -> None:
    from datetime import datetime

    world.write_batches = [[_face()]]
    world.claim_insert = 0
    world.existing_claim = [{"project_id": 9, "written_at": datetime(2026, 1, 1), "claimed_at": datetime(2026, 1, 1)}]

    assert world.run() == 1

    assert world.api_calls == []
    assert world.sql_like("UPDATE faces SET sdc_written = 1 WHERE id = %s") == [
        ("UPDATE faces SET sdc_written = 1 WHERE id = %s", (100,))
    ]


def test_face_rejected_while_waiting_is_not_written(world: World) -> None:
    world.write_batches = [[_face()]]
    world.live = lambda face_id: {"is_target": 0, "superseded_by": None}

    assert world.run() == 0

    assert world.api_calls == []


def test_unverifiable_face_before_the_edit_is_skipped(world: World) -> None:
    """The first re-check passes, the one right before wbeditentity cannot be read."""
    world.write_batches = [[_face()]]
    reads = iter([{"is_target": 1, "superseded_by": None}])

    def live(face_id: int) -> dict[str, Any]:
        try:
            return next(reads)
        except StopIteration:
            raise DatabaseError("flapped") from None

    world.live = live

    assert world.run() == 0

    assert world.actions() == ["wbgetclaims"]


def test_face_unconfirmed_during_the_lookup_releases_its_reservation(world: World) -> None:
    world.write_batches = [[_face()]]
    states = iter([{"is_target": 1, "superseded_by": None}, {"is_target": None, "superseded_by": None}])
    world.live = lambda face_id: next(states)

    assert world.run() == 0

    assert world.actions() == ["wbgetclaims"]
    assert world.sql_like("DELETE FROM sdc_claims WHERE face_id = %s AND written_at IS NULL") == [
        ("DELETE FROM sdc_claims WHERE face_id = %s AND written_at IS NULL", (100,))
    ]


def test_existing_claim_for_a_rejected_face_is_not_marked_written(world: World) -> None:
    world.write_batches = [[_face()]]
    world.api_script["wbgetclaims"] = [CLAIM_PRESENT]
    marked = iter([0])
    original_db = world.db

    def db(sql: str, params: Any = None, fetch: bool = True) -> Any:
        if " ".join(sql.split()).startswith("UPDATE faces SET sdc_written = 1 WHERE id = %s AND is_target = 1"):
            world.calls.append((" ".join(sql.split()), params))
            return next(marked)
        return original_db(sql, params, fetch)

    world.db = db  # type: ignore[method-assign]

    assert world.run() == 1

    assert "wbeditentity" not in world.actions()


def test_edit_without_success_key_is_still_counted_and_warned(world: World, caplog: pytest.LogCaptureFixture) -> None:
    world.write_batches = [[_face()]]
    world.api_script["wbeditentity"] = [{"unexpected": True}]

    with caplog.at_level(logging.WARNING, logger="worker"):
        assert world.run() == 1

    assert "no 'success' key" in caplog.text


def test_no_such_entity_during_the_edit_marks_the_face_terminal(world: World) -> None:
    world.write_batches = [[_face()]]
    world.api_script["wbeditentity"] = [{"error": {"code": "no-such-entity", "info": "gone"}}]

    assert world.run() == 0

    assert world.sql_like("UPDATE faces SET sdc_written = 1 WHERE id = %s") != []
    assert world.project_updates() == [None]


def test_unexpected_api_exception_aborts_the_run_with_its_message(world: World) -> None:
    world.write_batches = [[_face()]]
    world.api_script["wbgetclaims"] = [RuntimeError("connection reset")]

    assert world.run() == 0

    assert world.project_updates() == ["connection reset"]


def test_unknown_error_code_from_the_idempotency_lookup_aborts(world: World) -> None:
    world.write_batches = [[_face(), _face(101, PAGE + 1)]]
    world.api_script["wbgetclaims"] = [{"error": {"code": "weird", "info": "odd"}}]

    assert world.run() == 0

    assert world.project_updates() == ["weird: odd"]
    assert world.actions() == ["wbgetclaims"]


def test_unknown_error_code_from_the_edit_aborts_after_counting_earlier_writes(world: World) -> None:
    world.write_batches = [[_face(), _face(101, PAGE + 1)]]
    world.api_script["wbeditentity"] = [{"success": 1}, {"error": {"code": "readonly", "info": "x"}}]

    assert world.run() == 1

    assert world.project_updates() == [worker._SDC_ERROR_MESSAGES["readonly"]]


def test_skippable_error_in_the_idempotency_lookup_marks_the_face_and_releases_its_claim(world: World) -> None:
    world.write_batches = [[_face()]]
    world.api_script["wbgetclaims"] = [{"error": {"code": "no-such-entity", "info": "gone"}}]

    assert world.run() == 0

    assert world.sql_like("DELETE FROM sdc_claims WHERE face_id = %s") != []
    assert world.project_updates() == [None]


def test_every_face_unverifiable_ends_the_run_with_an_explanation(world: World) -> None:
    world.write_batches = [[_face(), _face(101, PAGE + 1)]]

    def live(face_id: int) -> dict[str, Any]:
        raise DatabaseError("flapped")

    world.live = live

    assert world.run() == 0

    assert world.project_updates() == ["Could not verify the faces before writing. Please try again."]


# --------------------------------------------------------------------------
# Token errors: the same handling at all four Commons call sites
# --------------------------------------------------------------------------

SITES = ["write-lookup", "write-edit", "removal-lookup", "removal-edit"]


def arrange_site(world: World, site: str, error: dict[str, Any], repeats: int) -> None:
    if site.startswith("write"):
        world.write_batches = two_passes(_face()) if repeats == 1 else (lambda: [_face()])
    else:
        batches = [[{"commons_page_id": PAGE}] for _ in range(4)]
        world.removal_batches = batches
    if site == "write-lookup":
        world.api_script["wbgetclaims"] = [error] * repeats
    elif site == "write-edit":
        world.api_script["wbeditentity"] = [error] * repeats
    elif site == "removal-lookup":
        world.api_script["wbgetclaims"] = [error] * repeats
    else:
        world.api_script["wbgetclaims"] = [CLAIM_PRESENT] * (repeats + 1)
        world.api_script["wbremoveclaims"] = [error] * repeats


@pytest.mark.parametrize("site", SITES)
def test_a_token_error_refreshes_the_token_and_the_next_pass_succeeds(world: World, site: str) -> None:
    arrange_site(world, site, AUTH_ERROR, repeats=1)
    world.token_script = ["first", "second"]
    world.csrf_script = ["csrf-a", "csrf-b"]

    world.run()

    assert world.refresh_calls == 2  # initial token + one refresh
    assert world.csrf_calls == 2
    last_call_headers = world.api_calls[-1][2]
    assert last_call_headers == {"Authorization": "Bearer second"}
    assert world.project_updates() == [None]


@pytest.mark.parametrize("site", SITES)
def test_repeated_token_errors_give_up_after_the_retry_limit(world: World, site: str) -> None:
    arrange_site(world, site, AUTH_ERROR, repeats=worker.MAX_TOKEN_RETRIES + 1)

    result = world.run()

    assert result == 0
    assert world.refresh_calls == 1 + worker.MAX_TOKEN_RETRIES
    assert world.project_updates() == [worker._SDC_ERROR_MESSAGES["badtoken"]]


@pytest.mark.parametrize("site", SITES)
def test_a_failing_csrf_refresh_aborts_with_the_original_error(world: World, site: str) -> None:
    arrange_site(world, site, AUTH_ERROR, repeats=1)
    world.csrf_script = ["csrf-a", RuntimeError("no csrf")]

    world.run()

    assert world.project_updates() == [worker._SDC_ERROR_MESSAGES["badtoken"]]


@pytest.mark.parametrize("site", SITES)
def test_an_invalid_authorization_error_is_treated_like_a_token_error(world: World, site: str) -> None:
    """mwoauth-invalid-authorization means the token is bad: refresh, do not abort."""
    error = {"error": {"code": "mwoauth-invalid-authorization", "info": "Invalid authorization"}}
    arrange_site(world, site, error, repeats=1)

    world.run()

    assert world.refresh_calls == 2
    assert world.project_updates() == [None]


# --------------------------------------------------------------------------
# Removal phase
# --------------------------------------------------------------------------


def clears_removal_flags(world: World) -> list[tuple[str, Any]]:
    return world.sql_like("SET f.sdc_removal_pending = 0, f.sdc_written = 0")


def test_a_pending_removal_deletes_the_claim_and_clears_the_flags(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    world.api_script["wbgetclaims"] = [CLAIM_PRESENT]

    assert world.run() == 0

    assert world.actions() == ["wbgetclaims", "wbremoveclaims"]
    assert world.api_calls[1][1]["claim"] == GUID
    assert [params for _, params in clears_removal_flags(world)] == [(PAGE, 5)]
    assert world.sql_like("DELETE FROM sdc_claims WHERE commons_page_id = %s AND wikidata_qid = %s") == [
        ("DELETE FROM sdc_claims WHERE commons_page_id = %s AND wikidata_qid = %s", (PAGE, QID))
    ]
    assert world.project_updates() == [None]


def test_removal_when_the_claim_is_already_gone_only_clears_state(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]

    world.run()

    assert world.actions() == ["wbgetclaims"]
    assert len(clears_removal_flags(world)) == 1
    assert world.sql_like("DELETE FROM sdc_claims WHERE commons_page_id") != []


def test_removal_is_skipped_when_a_sibling_was_approved_meanwhile(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    world.still_pending = []

    world.run()

    assert world.api_calls == []
    assert clears_removal_flags(world) == []


@pytest.mark.parametrize("step", ["lookup", "removal"])
def test_no_such_entity_during_removal_counts_as_removed(world: World, step: str) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    error = {"error": {"code": "no-such-entity", "info": "gone"}}
    if step == "lookup":
        world.api_script["wbgetclaims"] = [error]
    else:
        world.api_script["wbgetclaims"] = [CLAIM_PRESENT]
        world.api_script["wbremoveclaims"] = [error]

    world.run()

    assert len(clears_removal_flags(world)) == 1
    assert world.project_updates() == [None]


@pytest.mark.parametrize("step", ["lookup", "removal"])
def test_an_unknown_error_during_removal_aborts_the_run(world: World, step: str) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    error = {"error": {"code": "readonly", "info": "x"}}
    if step == "lookup":
        world.api_script["wbgetclaims"] = [error]
    else:
        world.api_script["wbgetclaims"] = [CLAIM_PRESENT]
        world.api_script["wbremoveclaims"] = [error]

    assert world.run() == 0

    assert world.project_updates() == [worker._SDC_ERROR_MESSAGES["readonly"]]
    assert clears_removal_flags(world) == []


def test_an_exception_during_removal_aborts_with_its_message(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    world.api_script["wbgetclaims"] = [RuntimeError("socket closed")]

    assert world.run() == 0

    assert world.project_updates() == ["socket closed"]


def test_stop_during_the_removal_phase_ends_quietly(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    checks = iter([1, 0])  # write phase check, removal phase check
    world.requested = lambda: next(checks, 0)

    world.run()

    assert world.api_calls == []
    assert world.project_updates() == [None]


def test_a_failed_cancel_check_in_the_removal_phase_does_not_stop_it(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}]]
    answers: list[Any] = [1, DatabaseError("flapped")]

    def requested() -> int:
        item = answers.pop(0) if answers else 1
        if isinstance(item, Exception):
            raise item
        return item

    world.requested = requested
    world.api_script["wbgetclaims"] = [CLAIM_PRESENT]

    world.run()

    assert world.actions() == ["wbgetclaims", "wbremoveclaims"]


def test_removals_refresh_liveness_for_every_page(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}, {"commons_page_id": PAGE + 1}]]

    world.run()

    assert len(world.sql_like("worker_claimed_at = NOW()")) == 2


def test_shutdown_during_removals_stops_the_loop(world: World) -> None:
    world.removal_batches = [[{"commons_page_id": PAGE}, {"commons_page_id": PAGE + 1}]]
    world.api_script["wbgetclaims"] = [CLAIM_PRESENT, CLAIM_PRESENT]
    original = world.api

    def api(*args: Any, **kwargs: Any) -> MagicMock:
        response = original(*args, **kwargs)
        if (kwargs.get("data") or {}).get("action") == "wbremoveclaims":
            worker.shutdown_requested = True
        return response

    world.api = api  # type: ignore[method-assign]
    world.pending_after_shutdown = 1

    world.run()

    assert world.actions().count("wbremoveclaims") == 1
    assert world.project_updates() == [
        "Worker shutdown left Commons edits pending. Edits already sent were kept. Send the remaining edits again."
    ]


# --------------------------------------------------------------------------
# A page held by another project must not turn the loop into a busy wait
# --------------------------------------------------------------------------


def held_by_another_project(world: World, *, max_selections: int = 200) -> None:
    """Every pass selects the same face, and its claim belongs to a project that is still working."""
    from datetime import datetime

    selections = {"n": 0}

    def batch() -> list[dict[str, int]]:
        selections["n"] += 1
        if selections["n"] > max_selections:
            raise AssertionError("the write loop is spinning on a face it cannot write")
        return [_face()]

    world.write_batches = batch
    world.claim_insert = 0
    world.reclaim = 0
    world.existing_claim = [{"project_id": 9, "written_at": None, "claimed_at": datetime(2026, 1, 1)}]


def test_faces_held_by_another_project_back_off_and_the_run_ends_with_an_explanation(world: World) -> None:
    held_by_another_project(world)

    assert world.run() == 0

    assert world.api_calls == []
    assert len(world.sql_like("INSERT IGNORE INTO sdc_claims")) == worker._SDC_BLOCKED_MAX_PASSES
    waited = sum(world.sleeps)
    assert waited >= (worker._SDC_BLOCKED_MAX_PASSES - 1) * worker._SDC_BLOCKED_BACKOFF_SECONDS
    assert world.project_updates() == [
        "Some files are being edited by another project and were skipped. Send the edits again later."
    ]


def test_blocked_faces_still_let_the_removal_phase_run(world: World) -> None:
    held_by_another_project(world)
    world.removal_batches = [[{"commons_page_id": PAGE + 5}]]
    world.api_script["wbgetclaims"] = [CLAIM_PRESENT]

    world.run()

    assert world.actions() == ["wbgetclaims", "wbremoveclaims"]


def test_waiting_keeps_the_worker_alive_and_stops_on_shutdown(world: World) -> None:
    held_by_another_project(world)

    def sleep_then_shutdown(seconds: float) -> None:
        world.sleeps.append(seconds)
        worker.shutdown_requested = True

    with patch.object(worker.time, "sleep", side_effect=sleep_then_shutdown):
        with (
            patch.object(worker, "execute_query", side_effect=world.db),
            patch.object(worker, "_api_request", side_effect=world.api),
            patch.object(worker, "_refresh_worker_token", side_effect=world.refresh),
            patch.object(worker, "_get_csrf_token", side_effect=world.csrf),
            patch.object(worker, "shutdown_requested", False),
        ):
            assert worker.write_sdc_claims(world.project) == 0

    assert len(world.sleeps) == 1  # it did not keep sleeping after the shutdown signal


@pytest.mark.parametrize("code", ["notloggedin", "badtoken", "mwoauth-invalid-authorization"])
def test_authorization_failures_store_the_message_the_web_app_turns_into_a_reauthorize_button(code: str) -> None:
    message = worker._sdc_error_message(code, "ignored")

    assert message == SDC_AUTH_EXPIRED_MESSAGE
    assert message in SDC_REAUTH_MESSAGES  # app.api_sdc_status keys the button on this set


def test_a_missing_edit_permission_is_not_offered_as_a_reauthorize() -> None:
    # That one is fixed by the tool's OAuth consumer registration, not by the user signing in again.
    assert worker._sdc_error_message("permissiondenied", "") not in SDC_REAUTH_MESSAGES
