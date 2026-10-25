"""Tests for the helpers extracted from the oversized classification routes."""

from __future__ import annotations

import importlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

import face_client
from api_errors import ApiError

app_module = importlib.import_module("app")
flask_app = app_module.app


@pytest.fixture(autouse=True)
def _request_context():
    flask_app.config["TESTING"] = True
    with flask_app.test_request_context("/", method="POST"):
        yield


def form(**fields: Any):
    return flask_app.test_request_context("/", method="POST", data={k: str(v) for k, v in fields.items()})


# -- _read_box_request ----------------------------------------------------


def test_a_complete_box_request_is_parsed_into_ids_and_a_box() -> None:
    with form(project_id=3, image_id=9, bbox_top=10, bbox_right=200, bbox_bottom=300, bbox_left=20):
        ids, box = app_module._read_box_request("project_id", "image_id")

    assert ids == (3, 9)
    assert box == (10, 200, 300, 20)


def test_a_face_id_request_has_one_id() -> None:
    with form(face_id=5, bbox_top=0, bbox_right=100, bbox_bottom=100, bbox_left=0):
        ids, box = app_module._read_box_request("face_id")

    assert ids == (5,)
    assert box == (0, 100, 100, 0)


@pytest.mark.parametrize("missing", ["project_id", "image_id", "bbox_top", "bbox_right", "bbox_bottom", "bbox_left"])
def test_any_missing_field_is_a_400(missing: str) -> None:
    fields = {"project_id": 1, "image_id": 2, "bbox_top": 0, "bbox_right": 90, "bbox_bottom": 90, "bbox_left": 0}
    del fields[missing]

    with form(**fields), pytest.raises(ApiError) as error:
        app_module._read_box_request("project_id", "image_id")

    assert error.value.status == 400
    assert error.value.message == "Missing required fields"


def test_non_integer_values_are_a_400() -> None:
    with (
        form(face_id="abc", bbox_top=0, bbox_right=90, bbox_bottom=90, bbox_left=0),
        pytest.raises(ApiError) as error,
    ):
        app_module._read_box_request("face_id")

    assert (error.value.status, error.value.message) == (400, "Invalid field values")


@pytest.mark.parametrize(
    "box",
    [
        (50, 10, 40, 90),  # top below bottom
        (-5, 100, 100, 0),  # negative
        (0, 100000, 100, 0),  # beyond MAX_BBOX_PX
        (0, 3, 3, 0),  # smaller than MIN_BBOX_AREA
    ],
)
def test_an_invalid_box_is_a_400_with_the_validators_message(box: tuple[int, int, int, int]) -> None:
    top, right, bottom, left = box
    with (
        form(face_id=1, bbox_top=top, bbox_right=right, bbox_bottom=bottom, bbox_left=left),
        pytest.raises(ApiError) as error,
    ):
        app_module._read_box_request("face_id")

    assert error.value.status == 400
    assert error.value.message == app_module._validate_bbox(*box)


# -- _read_dismiss_ids ----------------------------------------------------


def test_no_dismiss_field_means_no_ids() -> None:
    assert app_module._read_dismiss_ids() == []


def test_a_json_list_of_ids_is_read() -> None:
    with form(dismiss_face_ids=json.dumps([4, "5"])):
        assert app_module._read_dismiss_ids() == [4, 5]


@pytest.mark.parametrize("raw", ["not json", '{"a": 1}', '["x"]', "7"])
def test_malformed_dismiss_ids_are_ignored(raw: str) -> None:
    with form(dismiss_face_ids=raw):
        assert app_module._read_dismiss_ids() == []


def test_too_many_dismiss_ids_are_refused() -> None:
    with (
        form(dismiss_face_ids=json.dumps(list(range(app_module.MAX_DISMISS_FACE_IDS + 1)))),
        pytest.raises(ApiError) as error,
    ):
        app_module._read_dismiss_ids()

    assert error.value.status == 400


# -- the JSON error contract ---------------------------------------------


def test_an_api_error_becomes_the_json_body_the_front_end_reads() -> None:
    response, status = app_module._handle_api_error(ApiError("nope", 409, reason="already_reviewed"))

    assert status == 409
    assert response.get_json() == {"error": "nope", "reason": "already_reviewed"}


def test_the_error_handler_is_registered_so_routes_can_raise() -> None:
    current_app = importlib.import_module("app").app
    assert current_app.error_handler_spec[None][None][ApiError] is importlib.import_module("app")._handle_api_error


def test_already_reviewed_names_the_other_contributor_when_known() -> None:
    error = app_module._already_reviewed_error("Alice")

    assert error.status == 409
    assert error.extra == {"reason": "already_reviewed"}
    assert "Alice" in error.message and "Only they can change" in error.message


def test_already_reviewed_has_a_generic_message_without_a_name() -> None:
    error = app_module._already_reviewed_error("")

    assert error.status == 409
    assert error.message == "This face has already been classified by another contributor."


# -- Commons URL and the face-service step -------------------------------


def test_the_file_prefix_is_dropped_from_the_commons_url() -> None:
    expected = app_module.FILE_PATH_URL.format(file_title="Ada.jpg")

    assert app_module._commons_file_url("File:Ada.jpg") == expected
    assert app_module._commons_file_url("Ada.jpg") == expected


IMAGE = {"file_title": "File:Ada.jpg", "detection_width": 960, "detection_height": 1280}
BOX = (10, 200, 300, 20)


def encode(**patches: Any):
    return patch.object(face_client, "encode_known_face", **patches)


def test_a_drawn_box_is_embedded_in_the_stored_detection_space() -> None:
    encoded = face_client.KnownFaceEncoding(b"x" * 2048, "model", True)

    with (
        patch.object(app_module, "_download_image", return_value=b"jpeg") as download,
        encode(return_value=encoded) as call,
    ):
        result = app_module._encode_drawn_box(IMAGE, BOX, purpose="testing")

    assert result is encoded
    download.assert_called_once_with(app_module._commons_file_url("File:Ada.jpg"))
    call.assert_called_once_with(b"jpeg", BOX, box_space=(960, 1280))


def test_an_image_the_service_cannot_read_is_a_422() -> None:
    with (
        patch.object(app_module, "_download_image", return_value=b"x"),
        encode(side_effect=face_client.FaceServiceRejected("bad image", code="invalid_image")),
        pytest.raises(ApiError) as error,
    ):
        app_module._encode_drawn_box(IMAGE, BOX, purpose="testing")

    assert error.value.status == 422
    assert "could not be read" in error.value.message


def test_an_unavailable_service_is_a_503() -> None:
    with (
        patch.object(app_module, "_download_image", return_value=b"x"),
        encode(side_effect=face_client.FaceServiceUnavailable("down")),
        pytest.raises(ApiError) as error,
    ):
        app_module._encode_drawn_box(IMAGE, BOX, purpose="testing")

    assert error.value.status == 503


def test_a_box_without_a_usable_embedding_asks_for_a_larger_box() -> None:
    with (
        patch.object(app_module, "_download_image", return_value=b"x"),
        encode(return_value=None),
        pytest.raises(ApiError) as error,
    ):
        app_module._encode_drawn_box(IMAGE, BOX, purpose="testing")

    assert error.value.status == 422
    assert "slightly larger box" in error.value.message


def test_a_failed_download_is_left_for_the_route_to_report_as_502() -> None:
    with (
        patch.object(app_module, "_download_image", side_effect=requests.ConnectionError("reset")),
        pytest.raises(requests.RequestException),
    ):
        app_module._encode_drawn_box(IMAGE, BOX, purpose="testing")


def test_a_drawn_face_is_inserted_as_a_human_confirmed_target() -> None:
    cursor = MagicMock(lastrowid=77)
    encoded = face_client.KnownFaceEncoding(b"enc", "model-x", False)

    new_id = app_module._insert_human_target_face(cursor, 9, encoded, BOX, 4)

    assert new_id == 77
    sql, params = cursor.execute.call_args.args
    assert "is_target, classified_by" in sql and "1, 'human'" in sql
    assert params == (9, b"enc", "model-x", 0, 10, 200, 300, 20, 4)


# -- undo decisions -------------------------------------------------------


def state(*, written: int = 0, pending: int = 0, target: int | None = 1, bootstrapped: int = 0) -> dict[str, Any]:
    return {"is_target": target, "sdc_written": written, "sdc_removal_pending": pending, "bootstrapped": bootstrapped}


def flags(face_state: dict[str, Any] | None, **overrides: Any) -> tuple[int, int]:
    options = {"undoing_rejection": False, "sibling_keeps_claim": False, "sdc_removed": False, "write_in_flight": False}
    options.update(overrides)
    return app_module._undo_claim_flags(face_state, **options)


@pytest.mark.parametrize(
    ("description", "face_state", "overrides", "expected"),
    [
        ("unknown face just reverts", None, {}, (0, 0)),
        ("a sibling keeps the claim: just revert", state(written=1), {"sibling_keeps_claim": True}, (0, 0)),
        ("claim removed now: nothing left to clean", state(written=1), {"sdc_removed": True}, (0, 0)),
        ("claim on record, removal failed: queue it and keep the marker", state(written=1), {}, (1, 1)),
        ("queued removal, none written: keep it queued", state(pending=1, written=0), {}, (1, 0)),
        ("target, nothing on record, no write running", state(), {}, (0, 0)),
        ("target, nothing on record, write running: queue to be safe", state(), {"write_in_flight": True}, (1, 0)),
        ("non-target, write running: nothing to clean up", state(target=None), {"write_in_flight": True}, (0, 0)),
        (
            "undo of a rejection on a bootstrapped image cancels the queued removal",
            state(pending=1, written=1, target=0, bootstrapped=1),
            {"undoing_rejection": True},
            (0, 1),
        ),
        (
            "undo of a rejection elsewhere keeps an older queue and the marker",
            state(pending=1, written=1, target=0),
            {"undoing_rejection": True},
            (1, 1),
        ),
        (
            "undo of a rejection never queues a removal",
            state(pending=0, written=1, target=0),
            {"undoing_rejection": True},
            (0, 1),
        ),
        (
            "undo of a rejection ignores a write run",
            state(target=0),
            {"undoing_rejection": True, "write_in_flight": True},
            (0, 0),
        ),
        (
            "a sibling wins over a rejection too",
            state(pending=1, target=0),
            {"undoing_rejection": True, "sibling_keeps_claim": True},
            (0, 0),
        ),
    ],
)
def test_the_flags_a_reverted_face_ends_up_with(
    description: str, face_state: dict[str, Any] | None, overrides: dict[str, Any], expected: tuple[int, int]
) -> None:
    assert flags(face_state, **overrides) == expected, description


def claim_context(face_states: dict[int, dict[str, Any]], *, sibling: bool = False):
    return app_module._UndoClaimContext(face_states=face_states, sibling_keeps_claim=sibling, write_in_flight=False)


def claimed(**extra: Any) -> dict[str, Any]:
    return {**state(written=1), "commons_page_id": 4242, "wikidata_qid": "Q42", **extra}


def test_no_claim_on_record_means_nothing_to_remove() -> None:
    with patch.object(app_module, "_get_valid_token") as token:
        result = app_module._remove_claims_before_undo(
            claim_context({1: state(written=0)}), [], undoing_rejection=False
        )

    assert result == (False, False)
    token.assert_not_called()


def test_undoing_a_rejection_never_removes_a_claim() -> None:
    with patch.object(app_module, "_remove_sdc_claim") as remove:
        result = app_module._remove_claims_before_undo(claim_context({1: claimed()}), [], undoing_rejection=True)

    assert result == (False, False)
    remove.assert_not_called()


def test_a_sibling_target_keeps_the_claim() -> None:
    with patch.object(app_module, "_remove_sdc_claim") as remove:
        result = app_module._remove_claims_before_undo(
            claim_context({1: claimed()}, sibling=True), [], undoing_rejection=False
        )

    assert result == (False, False)
    remove.assert_not_called()


def test_a_recorded_claim_is_removed_from_commons() -> None:
    with (
        patch.object(app_module, "_get_valid_token", return_value="token"),
        patch.object(app_module, "_remove_sdc_claim", return_value=True) as remove,
    ):
        result = app_module._remove_claims_before_undo(claim_context({1: claimed()}), [], undoing_rejection=False)

    assert result == (True, False)
    remove.assert_called_once_with(4242, "Q42", "token")


def test_a_failed_removal_is_queued_for_the_worker() -> None:
    with (
        patch.object(app_module, "_get_valid_token", return_value="token"),
        patch.object(app_module, "_remove_sdc_claim", return_value=False),
    ):
        result = app_module._remove_claims_before_undo(claim_context({1: claimed()}), [], undoing_rejection=False)

    assert result == (False, True)


def test_a_missing_token_queues_the_removal_instead_of_failing() -> None:
    with patch.object(app_module, "_get_valid_token", return_value=None):
        result = app_module._remove_claims_before_undo(claim_context({1: claimed()}), [], undoing_rejection=False)

    assert result == (False, True)


def test_a_manual_face_is_never_deleted_while_its_claim_survives() -> None:
    with (
        patch.object(app_module, "_get_valid_token", return_value="token"),
        patch.object(app_module, "_remove_sdc_claim", return_value=False),
        pytest.raises(ApiError) as error,
    ):
        app_module._remove_claims_before_undo(claim_context({1: claimed()}), [1], undoing_rejection=False)

    assert error.value.status == 502


def test_an_expired_login_blocks_the_undo_of_a_manual_face() -> None:
    with (
        patch.object(app_module, "_get_valid_token", return_value=None),
        pytest.raises(ApiError) as error,
    ):
        app_module._remove_claims_before_undo(claim_context({1: claimed()}), [1], undoing_rejection=False)

    assert error.value.status == 401


# -- reclassify -----------------------------------------------------------


def face_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": 5,
        "image_id": 50,
        "project_id": 500,
        "old_is_target": None,
        "sdc_written": 0,
        "bootstrapped": 0,
        "commons_page_id": 4242,
        "wikidata_qid": "Q42",
        "classified_by": "model",
    }
    return {**row, **overrides}


def test_reclassify_form_is_read() -> None:
    with form(face_id=5, is_target=0):
        assert app_module._read_reclassify_request() == (5, 0)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"face_id": 5}, "Missing required fields"),
        ({"is_target": 1}, "Missing required fields"),
        ({"face_id": "x", "is_target": 1}, "Invalid field values"),
        ({"face_id": 5, "is_target": 2}, "Invalid field values"),
    ],
)
def test_a_bad_reclassify_form_is_a_400(fields: dict[str, Any], message: str) -> None:
    with form(**fields), pytest.raises(ApiError) as error:
        app_module._read_reclassify_request()

    assert (error.value.status, error.value.message) == (400, message)


def test_a_face_the_user_may_not_act_on_is_a_404() -> None:
    app_module.g.user = {"id": 4}
    with patch.object(app_module, "execute_query", return_value=[]), pytest.raises(ApiError) as error:
        app_module._load_face_for_reclassify(5)

    assert error.value.status == 404


def test_a_database_failure_while_loading_the_face_is_a_500() -> None:
    app_module.g.user = {"id": 4}
    with (
        patch.object(app_module, "execute_query", side_effect=app_module.DatabaseError("down")),
        pytest.raises(ApiError) as error,
    ):
        app_module._load_face_for_reclassify(5)

    assert error.value.status == 500


def settle(row: dict[str, Any], *, sibling: bool = False, **patches: Any):
    with (
        patch.object(app_module, "has_sibling_match", return_value=sibling),
        patch.object(app_module, "_get_valid_token", return_value=patches.get("token", "tok")),
        patch.object(app_module, "_remove_sdc_claim", return_value=patches.get("removed", True)) as remove,
    ):
        return app_module._settle_claim_for_rejection(row, 5), remove


def test_a_sibling_target_means_the_claim_stays() -> None:
    result, remove = settle(face_row(sdc_written=1), sibling=True)

    assert result == (False, False)
    remove.assert_not_called()


def test_a_bootstrap_face_is_queued_for_the_worker() -> None:
    result, remove = settle(face_row(sdc_written=1, bootstrapped=1))

    assert result == (False, True)
    remove.assert_not_called()


def test_an_unwritten_face_is_queued_in_case_a_write_is_in_flight() -> None:
    assert settle(face_row(sdc_written=0))[0] == (False, True)


def test_a_claim_on_record_is_removed_immediately() -> None:
    result, remove = settle(face_row(sdc_written=1))

    assert result == (True, False)
    remove.assert_called_once_with(4242, "Q42", "tok")


def test_rejection_is_refused_when_the_claim_cannot_be_removed() -> None:
    with pytest.raises(ApiError) as error:
        settle(face_row(sdc_written=1), removed=False)

    assert error.value.status == 502


def test_rejection_is_refused_without_a_valid_login() -> None:
    with pytest.raises(ApiError) as error:
        settle(face_row(sdc_written=1), token=None)

    assert error.value.status == 401


def test_a_database_failure_in_the_sibling_check_is_a_500() -> None:
    with (
        patch.object(app_module, "has_sibling_match", side_effect=app_module.DatabaseError("down")),
        pytest.raises(ApiError) as error,
    ):
        app_module._settle_claim_for_rejection(face_row(), 5)

    assert error.value.status == 500


def cursor_returning(row: dict[str, Any] | None) -> MagicMock:
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    return cursor


def test_the_same_user_repeating_an_action_is_a_no_op() -> None:
    app_module._require_same_reviewer(cursor_returning({"classified_by_user_id": 4, "wiki_username": "Me"}), 5, 4)


def test_another_reviewer_blocks_the_change_and_is_named() -> None:
    with pytest.raises(ValueError, match="already_reviewed:Alice"):
        app_module._require_same_reviewer(
            cursor_returning({"classified_by_user_id": 9, "wiki_username": "Alice"}), 5, 4
        )


def test_a_vanished_face_blocks_the_change_without_a_name() -> None:
    with pytest.raises(ValueError, match="already_reviewed:$"):
        app_module._require_same_reviewer(cursor_returning(None), 5, 4)


def statements(cursor: MagicMock) -> list[str]:
    return [" ".join(call.args[0].split()) for call in cursor.execute.call_args_list]


def apply(is_target: int, row: dict[str, Any], cursor: MagicMock | None = None, **kwargs: Any):
    cursor = cursor or MagicMock(rowcount=1, fetchall=MagicMock(return_value=[]))
    result = app_module._apply_reclassification(
        cursor,
        row,
        5,
        is_target,
        sdc_written=kwargs.get("sdc_written", 0),
        removal_queued=kwargs.get("removal_queued", False),
        user_id=4,
    )
    return result, cursor


def test_approving_a_new_target_counts_it_and_clears_its_siblings_queue() -> None:
    (demoted, sdc), cursor = apply(1, face_row(old_is_target=None))

    assert (demoted, sdc) == ([], {})
    sql = statements(cursor)
    assert sql[0].startswith("UPDATE faces SET is_target = %s, classified_by = 'human'")
    assert sql[1] == "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s"
    assert sql[2].startswith("UPDATE faces SET sdc_removal_pending = 0 WHERE image_id")
    assert "FOR UPDATE" in sql[3]


def test_approving_replaces_the_sibling_targets_and_reports_them() -> None:
    cursor = MagicMock(rowcount=2)
    cursor.fetchall.return_value = [{"id": 7, "sdc_written": 1}, {"id": 8, "sdc_written": 0}]

    (demoted, sdc), cursor = apply(1, face_row(old_is_target=0), cursor)

    assert demoted == [7, 8]
    assert sdc == {7: 1, 8: 0}
    assert any(
        s.startswith("UPDATE projects SET faces_confirmed = GREATEST(0, CAST(faces_confirmed AS SIGNED) - %s)")
        for s in statements(cursor)
    )


def test_rejecting_a_target_uncounts_it_and_demotes_nothing() -> None:
    result, cursor = apply(0, face_row(old_is_target=1))

    assert result == ([], {})
    assert statements(cursor)[-1].endswith("- 1) WHERE id = %s")


def test_approving_an_already_approved_face_changes_no_counters() -> None:
    result, cursor = apply(1, face_row(old_is_target=1))

    assert result == ([], {})
    assert len(statements(cursor)) == 1


def test_an_update_that_touched_nothing_is_checked_for_a_different_reviewer() -> None:
    cursor = MagicMock(rowcount=0)
    cursor.fetchone.return_value = {"classified_by_user_id": 9, "wiki_username": "Alice"}

    with pytest.raises(ValueError, match="already_reviewed:Alice"):
        apply(1, face_row(), cursor)


# -- classify -------------------------------------------------------------

IMAGE_ROW = {
    "image_id": 11,
    "file_title": "File:A.jpg",
    "commons_page_id": 1,
    "detection_width": 960,
    "detection_height": 1280,
}


class Script:
    """execute_query that answers by substring of the SQL, recording what was asked."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, tuple]] = []

    def __call__(self, sql: str, params: tuple = ()) -> Any:
        self.calls.append((" ".join(sql.split()), params))
        for fragment, answer in self.answers.items():
            if fragment in sql:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return []


def pick(
    script: Script, forced: int | None = None, skipped: list[int] | None = None, skipped_review: list[int] | None = None
):
    with patch.object(app_module, "execute_query", script):
        return app_module._pick_image_to_classify(7, forced, skipped or [], skipped_review or [])


def test_an_image_with_unclassified_faces_comes_first() -> None:
    script = Script({"f.is_target IS NULL AND f.superseded_by IS NULL": [IMAGE_ROW]})

    assert pick(script) == (IMAGE_ROW, False)
    assert len(script.calls) == 1


def test_skipped_images_are_excluded_from_the_choice() -> None:
    script = Script({"NOT IN": [IMAGE_ROW]})

    pick(script, skipped=[3, 4])

    sql, params = script.calls[0]
    assert "i.id NOT IN (%s,%s)" in sql and sql.endswith("LIMIT 200")
    assert params == (7, 3, 4)


def test_with_nothing_unclassified_the_models_rejections_are_offered_for_review() -> None:
    script = Script({"f.is_target = 0 AND f.classified_by = 'model'": [IMAGE_ROW]})

    assert pick(script, skipped_review=[9]) == (IMAGE_ROW, True)
    assert script.calls[-1][1] == (7, 9)


def test_an_empty_project_has_no_image_and_is_in_review_mode() -> None:
    assert pick(Script({})) == (None, True)


def test_a_forced_image_with_unclassified_faces_is_shown_normally() -> None:
    script = Script({"AND i.id = %s": [IMAGE_ROW], "SELECT COUNT(*) AS cnt FROM faces": [{"cnt": 2}]})

    assert pick(script, forced=11) == (IMAGE_ROW, False)


def test_a_forced_image_with_only_model_rejections_opens_in_review_mode() -> None:
    script = Script({"AND i.id = %s": [IMAGE_ROW], "SELECT COUNT(*) AS cnt FROM faces": [{"cnt": 0}]})

    assert pick(script, forced=11) == (IMAGE_ROW, True)


def test_a_forced_image_that_no_longer_qualifies_goes_straight_to_the_review_fallback() -> None:
    """Not to the unclassified queue: that is how the route has always behaved after an undo."""
    script = Script({"f.is_target = 0 AND f.classified_by = 'model'": [IMAGE_ROW]})

    assert pick(script, forced=99) == (IMAGE_ROW, True)
    assert not any("f.is_target IS NULL AND f.superseded_by IS NULL AND" in sql for sql, _ in script.calls)


def test_a_database_failure_while_choosing_shows_no_image() -> None:
    assert pick(Script({"SELECT DISTINCT": app_module.DatabaseError("down")}))[0] is None


def test_the_faces_of_a_review_image_are_the_models_rejections() -> None:
    script = Script({"FROM faces f": [{"face_id": 1}]})

    with patch.object(app_module, "execute_query", script):
        assert app_module._faces_to_classify(11, True) == [{"face_id": 1}]

    assert "f.is_target = 0 AND f.classified_by = 'model'" in script.calls[0][0]
    assert script.calls[0][0].endswith("ORDER BY f.bbox_left ASC")


def test_the_faces_of_a_normal_image_are_the_unclassified_ones() -> None:
    script = Script({"FROM faces f": []})

    with patch.object(app_module, "execute_query", script):
        app_module._faces_to_classify(11, False)

    assert "f.is_target IS NULL AND f.superseded_by IS NULL" in script.calls[0][0]


def test_a_failed_face_lookup_shows_no_faces() -> None:
    with patch.object(app_module, "execute_query", Script({"FROM faces f": app_module.DatabaseError("down")})):
        assert app_module._faces_to_classify(11, False) == []


def test_image_counts_default_to_zero() -> None:
    with patch.object(app_module, "execute_query", Script({"COUNT(DISTINCT": [{"cnt": 5}]})):
        assert app_module._count_images(app_module._UNCLASSIFIED_FACE_CONDITIONS, 7) == 5
    with patch.object(app_module, "execute_query", Script({})):
        assert app_module._count_images(app_module._UNCLASSIFIED_FACE_CONDITIONS, 7) == 0
    with patch.object(app_module, "execute_query", Script({"COUNT(DISTINCT": app_module.DatabaseError("down")})):
        assert app_module._count_images(app_module._UNCLASSIFIED_FACE_CONDITIONS, 7) == 0


def skip_request(**args: Any):
    return flask_app.test_request_context("/?" + "&".join(f"{k}={v}" for k, v in args.items()))


def test_a_skip_request_is_remembered_and_can_be_undone() -> None:
    with skip_request(skip_image_id=11):
        app_module._record_skip_request(7)

        assert app_module.session["skipped_images_7"] == [11]
        assert app_module.session["last_classify"]["action"] == "skip"
        assert app_module.session["last_classify"]["was_review"] is False


def test_a_review_skip_goes_into_its_own_list() -> None:
    with skip_request(skip_image_id=11, skip_reviewing=1):
        app_module._record_skip_request(7)

        assert app_module.session["skipped_images_review_7"] == [11]
        assert "skipped_images_7" not in app_module.session


def test_skipping_the_same_image_twice_lists_it_once() -> None:
    with skip_request(skip_image_id=11):
        app_module.session["skipped_images_7"] = [11]
        app_module._record_skip_request(7)

        assert app_module.session["skipped_images_7"] == [11]


def test_a_malformed_skip_id_is_ignored() -> None:
    with skip_request(skip_image_id="abc"):
        app_module._record_skip_request(7)

        assert "last_classify" not in app_module.session


# -- project creation -----------------------------------------------------


def new_form(**overrides: Any):
    fields = {
        "wikidata_qid": "q42",
        "commons_category": "Ada",
        "label": "",
        "distance_threshold": "0.5",
        "min_confirmed": "5",
    }
    fields.update(overrides)
    return form(**fields)


@pytest.fixture
def valid_entities():
    with (
        patch.object(app_module, "_is_human_entity", return_value=True) as human,
        patch.object(app_module, "_commons_category_exists", return_value=True),
        patch.object(app_module, "_commons_category_has_files", return_value=True),
    ):
        yield human


def test_a_valid_form_has_no_errors_and_a_normalised_qid(valid_entities: MagicMock) -> None:
    with new_form():
        result = app_module._read_new_project_form()

    assert result.errors == []
    assert (result.wikidata_qid, result.distance_threshold, result.min_confirmed) == ("Q42", 0.5, 5)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"wikidata_qid": "X1"}, "must start with 'Q'"),
        ({"commons_category": ""}, "Commons category is required"),
        ({"distance_threshold": "2"}, "between 0.1 and 1.0"),
        ({"distance_threshold": "abc"}, "must be a number"),
        ({"min_confirmed": "0"}, "at least 1"),
        ({"min_confirmed": "x"}, "whole number"),
    ],
)
def test_cheap_validation_errors_are_reported_without_any_network_lookup(
    valid_entities: MagicMock, overrides: dict[str, str], fragment: str
) -> None:
    with new_form(**overrides):
        result = app_module._read_new_project_form()

    assert any(fragment in error for error in result.errors)
    valid_entities.assert_not_called()


def test_an_invalid_number_falls_back_to_a_usable_default_for_redisplay(valid_entities: MagicMock) -> None:
    with new_form(distance_threshold="abc", min_confirmed="x"):
        result = app_module._read_new_project_form()

    assert (result.distance_threshold, result.min_confirmed) == (app_module.DEFAULT_DISTANCE_THRESHOLD, 5)


def test_a_non_human_entity_is_rejected() -> None:
    with patch.object(app_module, "_is_human_entity", return_value=False), new_form():
        result = app_module._read_new_project_form()

    assert any("not an instance of human" in error for error in result.errors)


@pytest.mark.parametrize(
    ("exists", "has_files", "fragment"), [(False, True, "does not exist"), (True, False, "contains no files")]
)
def test_a_missing_or_empty_category_is_rejected(exists: bool, has_files: bool, fragment: str) -> None:
    with (
        patch.object(app_module, "_is_human_entity", return_value=True),
        patch.object(app_module, "_commons_category_exists", return_value=exists),
        patch.object(app_module, "_commons_category_has_files", return_value=has_files),
        new_form(),
    ):
        result = app_module._read_new_project_form()

    assert any(fragment in error for error in result.errors)


def a_form() -> Any:
    return app_module._NewProjectForm("Q42", "Ada", "Ada Lovelace", 0.5, 5, [])


def test_an_existing_project_is_detected_and_a_failed_lookup_is_not_fatal() -> None:
    app_module.g.user = {"id": 4}
    with patch.object(app_module, "execute_query", return_value=[{"id": 1}]):
        assert app_module._user_already_has_project(a_form()) is True
    with patch.object(app_module, "execute_query", return_value=[]):
        assert app_module._user_already_has_project(a_form()) is False
    with patch.object(app_module, "execute_query", side_effect=app_module.DatabaseError("down")):
        assert app_module._user_already_has_project(a_form()) is False


def test_the_project_row_is_inserted_with_the_form_values() -> None:
    app_module.g.user = {"id": 4}
    with (
        patch.object(app_module, "execute_query") as insert,
        patch.object(app_module, "_generate_invite_code", return_value="ABCD1234"),
    ):
        app_module._insert_project(a_form(), "https://thumb")

    params = insert.call_args.args[1]
    assert params == (4, "Q42", "Ada", "Ada Lovelace", 0.5, 5, "https://thumb", "ABCD1234")


def test_an_invite_code_collision_is_retried_with_a_new_code() -> None:
    app_module.g.user = {"id": 4}
    collision = app_module.DatabaseError("Duplicate entry for key 'idx_projects_invite_code'")
    codes = iter(["AAAA", "BBBB"])

    with (
        patch.object(app_module, "execute_query", side_effect=[collision, 1]) as insert,
        patch.object(app_module, "_generate_invite_code", side_effect=lambda: next(codes)),
        patch.object(app_module, "_is_invite_code_collision", return_value=True),
    ):
        app_module._insert_project(a_form(), None)

    assert [call.args[1][-1] for call in insert.call_args_list] == ["AAAA", "BBBB"]


def test_other_database_errors_are_not_retried() -> None:
    app_module.g.user = {"id": 4}
    with (
        patch.object(app_module, "execute_query", side_effect=app_module.DatabaseError("boom")) as insert,
        patch.object(app_module, "_is_invite_code_collision", return_value=False),
        pytest.raises(app_module.DatabaseError),
    ):
        app_module._insert_project(a_form(), None)

    assert insert.call_count == 1


def test_collisions_that_never_clear_are_raised_after_the_last_attempt() -> None:
    app_module.g.user = {"id": 4}
    with (
        patch.object(app_module, "execute_query", side_effect=app_module.DatabaseError("dup")) as insert,
        patch.object(app_module, "_is_invite_code_collision", return_value=True),
        pytest.raises(app_module.DatabaseError),
    ):
        app_module._insert_project(a_form(), None)

    assert insert.call_count == app_module._MAX_INVITE_CODE_RETRIES
