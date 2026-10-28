import os
import time

os.environ.setdefault("TOOL_TOOLSDB_USER", "testuser")
os.environ.setdefault("TOOL_TOOLSDB_PASSWORD", "testpass")
os.environ.setdefault("WIKIVISAGE_DB_NAME", "testdb")
os.environ.setdefault("TOOL_TOOLSDB_HOST", "localhost")

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

import face_client
import liftwing_client
from config import SDC_AUTH_EXPIRED_MESSAGE
from database import DatabaseError

with patch("database.init_db"):
    import worker as _worker_module
    from worker import (
        _claim_active_projects,
        _claim_inference_projects,
        _claim_sdc_projects,
        _claim_shadow_project,
        _fetch_metadata,
        _infer_and_release,
        _plain_text,
        _probe_liftwing_if_due,
        _process_and_release,
        _process_image_wave,
        _record_liftwing_failure,
        _refresh_claims,
        _release_all_claims,
        _release_project,
        _run_new_inference_projects,
        _shadow_prefilter_loop,
        _sweep_shadow_prefilter,
        _touch_heartbeat_file,
        bootstrap_from_sparql,
        prefilter_images,
        process_images,
        process_project,
        run_autonomous_inference,
        write_sdc_claims,
    )

_handle_detection_failure = _worker_module._handle_detection_failure


def _make_embedding(seed: int, dim: int = 512) -> bytes:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype("<f4")
    vector /= np.linalg.norm(vector)
    return vector.tobytes()


MODEL_ID = face_client.EXPECTED_MODEL_ID
DIM = face_client.EMBEDDING_DIM


def _detection_result(width: int, height: int, faces: list) -> face_client.DetectionResult:
    return face_client.DetectionResult(width=width, height=height, faces=faces, model_id=MODEL_ID, dim=DIM)


def _extract_update_maps(
    params: tuple,
) -> tuple[dict[int, int], dict[int, float], list[int]]:
    n = len(params) // 5
    is_target_map = {params[i * 2]: params[i * 2 + 1] for i in range(n)}

    conf_start = 2 * n
    confidence_map = {params[conf_start + i * 2]: float(params[conf_start + i * 2 + 1]) for i in range(n)}

    in_ids = list(params[4 * n : 5 * n])
    return is_target_map, confidence_map, in_ids


def _fake_inference_transaction(fn):
    """Reuse query mocks for the inference algorithm tests' batch transaction."""
    cursor = MagicMock()

    def execute(sql, params=None):
        if sql.startswith("SELECT id FROM projects"):
            rows = [{"id": params[0]}]
        else:
            result = _worker_module.execute_query(sql, params, fetch=sql.startswith("SELECT"))
            rows = result if isinstance(result, (list, tuple)) else []
            cursor.rowcount = result if isinstance(result, int) else 0
        cursor.fetchone.return_value = rows[0] if rows else None
        cursor.fetchall.return_value = [dict(row, image_has_target=row.get("image_has_target", False)) for row in rows]

    cursor.execute.side_effect = execute
    return fn(None, cursor)


@pytest.fixture
def inference_transaction(monkeypatch):
    monkeypatch.setattr(_worker_module, "execute_transaction", _fake_inference_transaction)


def test_persist_inference_batch_rechecks_sibling_targets_under_project_lock():
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 16}
    cursor.fetchall.return_value = [
        {"id": 601, "image_id": 1, "image_has_target": 1},
        {"id": 602, "image_id": 2, "image_has_target": 0},
    ]
    cursor.rowcount = 2

    with patch("worker.execute_transaction", side_effect=lambda fn: fn(None, cursor)):
        result = _worker_module._persist_inference_batch(16, [(601, 1, 0.1), (602, 1, 0.4), (603, 1, 0.2)])

    assert result == (2, 1)
    calls = cursor.execute.call_args_list
    assert "FROM projects" in calls[0].args[0] and "FOR UPDATE" in calls[0].args[0]
    live_sql = calls[1].args[0]
    assert "t.is_target = 1" in live_sql and "FOR UPDATE" in live_sql
    assert "f.is_target IS NULL" in live_sql
    assert "f.classified_by_user_id IS NULL" in live_sql
    assert "f.sdc_removal_pending = 0" in live_sql
    assert "f.embedding_model = %s" in live_sql
    targets, _, ids = _extract_update_maps(calls[2].args[1])
    assert targets == {601: 0, 602: 1}
    assert ids == [601, 602]  # 603 is no longer eligible.


@pytest.mark.parametrize("project_exists,live_rows", [(False, []), (True, [])])
def test_persist_inference_batch_does_not_write_without_live_candidates(project_exists, live_rows):
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 1} if project_exists else None
    cursor.fetchall.return_value = live_rows

    with patch("worker.execute_transaction", side_effect=lambda fn: fn(None, cursor)):
        assert _worker_module._persist_inference_batch(1, [(101, 1, 0.1)]) == (0, 0)

    assert not any(call.args[0].startswith("UPDATE") for call in cursor.execute.call_args_list)


def test_persist_inference_batch_empty_does_not_open_transaction():
    with patch("worker.execute_transaction") as transaction:
        assert _worker_module._persist_inference_batch(1, []) == (0, 0)
    transaction.assert_not_called()


def test_inference_skips_below_min_confirmed():
    project = {"id": 10, "min_confirmed": 5, "distance_threshold": 0.6}

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 3}]
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query) as mock_db,
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 0
    assert mock_db.call_count == 1
    assert all("UPDATE faces" not in c.args[0] for c in mock_db.call_args_list)


def test_inference_single_face_per_image(inference_transaction):
    project = {"id": 11, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 101, "image_id": 1, "encoding": _make_embedding(101)},
        {"id": 102, "image_id": 2, "encoding": _make_embedding(102)},
        {"id": 103, "image_id": 3, "encoding": _make_embedding(103)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            assert params is not None
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    distances = [0.3, 0.8, 0.4]
    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._face_distance", side_effect=distances),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 3
    assert len(update_params) == 1
    is_target_map, confidence_map, in_ids = _extract_update_maps(update_params[0])

    assert set(in_ids) == {101, 102, 103}
    assert is_target_map[101] == 1
    assert is_target_map[102] == 0
    assert is_target_map[103] == 1
    assert confidence_map[101] == 0.3
    assert confidence_map[102] == 0.8
    assert confidence_map[103] == 0.4


def test_inference_multi_face_dedup(inference_transaction):
    project = {"id": 12, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 201, "image_id": 1, "encoding": _make_embedding(201)},
        {"id": 202, "image_id": 1, "encoding": _make_embedding(202)},
        {"id": 203, "image_id": 2, "encoding": _make_embedding(203)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            assert params is not None
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._face_distance", side_effect=[0.5, 0.3, 0.4]),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 3
    assert len(update_params) == 1
    is_target_map, _, in_ids = _extract_update_maps(update_params[0])

    assert set(in_ids) == {201, 202, 203}
    assert is_target_map[202] == 1
    assert is_target_map[201] == 0
    assert is_target_map[203] == 1


def test_inference_never_matches_second_face_on_image_with_existing_target(inference_transaction):
    """A face on an image with a confirmed target is always a non-match.

    A re-run resets model faces while preserving a human-approved target on the
    same image. The reset face must not become a second match just because its
    distance is below the threshold.
    """
    project = {"id": 16, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 601, "image_id": 1, "encoding": _make_embedding(601), "image_has_target": True},
        {"id": 602, "image_id": 2, "encoding": _make_embedding(602), "image_has_target": False},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            assert params is not None
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        # 601 is the closer face, but its image already has a confirmed target.
        patch("worker._face_distance", side_effect=[0.1, 0.4]),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 2
    is_target_map, _, _ = _extract_update_maps(update_params[0])
    assert is_target_map[601] == 0
    assert is_target_map[602] == 1


def test_inference_ignores_snapshot_sibling_when_it_was_undone():
    """A sibling target removed after candidate selection must not force a non-match.

    The precomputed per-image flag is only a snapshot. The closest candidate is
    computed from distance regardless of it, and the locked write decides the
    final target value from live sibling state.
    """
    project = {"id": 17, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 701, "image_id": 1, "encoding": _make_embedding(701), "image_has_target": True},
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding" in sql:
            return unclassified_rows
        if "UPDATE projects SET last_inference" in sql:
            return 1
        raise AssertionError(f"Unexpected query executed: {sql}")

    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 17}
    # The sibling target was undone after the candidate snapshot.
    cursor.fetchall.return_value = [{"id": 701, "image_id": 1, "image_has_target": 0}]
    cursor.rowcount = 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.execute_transaction", side_effect=lambda fn: fn(None, cursor)),
        patch("worker._face_distance", return_value=0.1),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 1
    update_sql, update_params = cursor.execute.call_args_list[-1].args
    assert update_sql.startswith("UPDATE faces SET")
    is_target_map, _, _ = _extract_update_maps(update_params)
    assert is_target_map == {701: 1}


def test_inference_multi_face_all_above_threshold(inference_transaction):
    project = {"id": 13, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 301, "image_id": 1, "encoding": _make_embedding(301)},
        {"id": 302, "image_id": 1, "encoding": _make_embedding(302)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            assert params is not None
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._face_distance", side_effect=[0.7, 0.8]),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 2
    assert len(update_params) == 1
    is_target_map, _, _ = _extract_update_maps(update_params[0])
    assert is_target_map[301] == 0
    assert is_target_map[302] == 0


def test_inference_returns_classified_count(inference_transaction):
    project = {"id": 14, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 401, "image_id": 1, "encoding": _make_embedding(401)},
        {"id": 402, "image_id": 1, "encoding": _make_embedding(402)},
        {"id": 403, "image_id": 2, "encoding": _make_embedding(403)},
        {"id": 404, "image_id": 3, "encoding": _make_embedding(404)},
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._face_distance", side_effect=[0.3, 0.5, 0.2, 0.9]),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 4


def test_inference_candidates_exclude_faces_with_a_queued_removal():
    """A face whose P180 removal is queued must not be model-classified.

    A model re-target would let the removal phase strip the claim of a fresh
    match; the face becomes eligible again once the queue is resolved.
    """
    project = {"id": 15, "min_confirmed": 5, "distance_threshold": 0.6}
    candidate_sql: list[str] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return [{"encoding": _make_embedding(i)} for i in range(5)]
        if "SELECT f.id, f.image_id, f.encoding" in sql:
            candidate_sql.append(sql)
            return []
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 0
    assert candidate_sql and "f.sdc_removal_pending = 0" in candidate_sql[0]


def test_process_image_wave_bootstrapped_single_face_auto_classifies():
    """Single-face bootstrapped image should auto-classify as target, set sdc_written=1, and increment faces_confirmed."""
    fake_location = (10, 110, 110, 10)
    fake_encoding = _make_embedding(1)
    detection_result = _detection_result(
        width=800,
        height=600,
        faces=[face_client.DetectedFace(*fake_location, encoding=fake_encoding)],
    )

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        if "UPDATE faces SET is_target" in sql:
            return 1  # one face auto-classified
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"fake"),
        patch("worker.face_client.detect_faces_batch", return_value={"42": detection_result}),
    ):
        processed, _timings = _process_image_wave([{"id": 42, "file_title": "File:Test.jpg", "bootstrapped": 1}], 5)

    assert processed == 1

    auto_classify_calls = [c for c in query_calls if "UPDATE faces SET is_target" in c[0]]
    assert len(auto_classify_calls) == 1
    assert auto_classify_calls[0][1] == (42,)

    project_update_calls = [c for c in query_calls if "UPDATE projects SET faces_confirmed" in c[0]]
    assert len(project_update_calls) == 1
    assert project_update_calls[0][1] == (1, 5)

    # sdc_written=1 must be set — the image already has P180 on Commons
    sdc_written_calls = [c for c in query_calls if "UPDATE faces SET sdc_written = 1" in c[0]]
    assert len(sdc_written_calls) == 1
    assert sdc_written_calls[0][1] == (42,)


def test_process_image_wave_bootstrapped_multi_face_no_auto_classify():
    """Multi-face bootstrapped image should NOT trigger auto-classify or faces_confirmed update."""
    fake_locations = [(10, 110, 110, 10), (200, 310, 310, 200)]
    fake_encodings = [_make_embedding(2), _make_embedding(3)]
    detection_result = _detection_result(
        width=800,
        height=600,
        faces=[
            face_client.DetectedFace(*fake_locations[0], encoding=fake_encodings[0]),
            face_client.DetectedFace(*fake_locations[1], encoding=fake_encodings[1]),
        ],
    )

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"fake"),
        patch("worker.face_client.detect_faces_batch", return_value={"43": detection_result}),
    ):
        processed, _timings = _process_image_wave([{"id": 43, "file_title": "File:Multi.jpg", "bootstrapped": 1}], 5)

    assert processed == 1
    assert not any("UPDATE faces SET is_target" in c[0] for c in query_calls)
    assert not any("UPDATE projects SET faces_confirmed" in c[0] for c in query_calls)


def test_process_image_wave_non_bootstrapped_no_auto_classify():
    """Non-bootstrapped single-face image should NOT trigger auto-classify or faces_confirmed update."""
    fake_location = (10, 110, 110, 10)
    fake_encoding = _make_embedding(4)
    detection_result = _detection_result(
        width=800,
        height=600,
        faces=[face_client.DetectedFace(*fake_location, encoding=fake_encoding)],
    )

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"fake"),
        patch("worker.face_client.detect_faces_batch", return_value={"44": detection_result}),
    ):
        processed, _timings = _process_image_wave([{"id": 44, "file_title": "File:NonBoot.jpg", "bootstrapped": 0}], 5)

    assert processed == 1
    assert not any("UPDATE faces SET is_target" in c[0] for c in query_calls)
    assert not any("UPDATE projects SET faces_confirmed" in c[0] for c in query_calls)


def test_process_images_prioritises_non_bootstrap():
    project = {"id": 99}

    non_bs_rows = [
        {"id": 1, "file_title": "File:A.jpg", "bootstrapped": 0},
        {"id": 2, "file_title": "File:B.jpg", "bootstrapped": 0},
    ]
    bs_rows = [
        {"id": 3, "file_title": "File:C.jpg", "bootstrapped": 1},
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SUM(CASE WHEN bootstrapped = 1 AND status != 'pending'" in sql:
            return [{"bs_done": 0, "total": 100}]
        if "COUNT(*) AS cnt FROM images" in sql and "bootstrapped = 1" in sql:
            return [{"cnt": 10}]
        if "bootstrapped = 0" in sql and "LIMIT" in sql:
            return non_bs_rows
        if "bootstrapped = 1" in sql and "LIMIT" in sql:
            return bs_rows
        if "UPDATE projects SET images_processed" in sql:
            return 1
        return ()

    persisted = []

    def fake_persist(img_id, result, *, bootstrapped=False, project_id=None):
        persisted.append((img_id, bootstrapped, len(result.faces), project_id))
        return True

    detect_result = _detection_result(
        width=100,
        height=80,
        faces=[face_client.DetectedFace(1, 2, 3, 4, encoding=_make_embedding(10))],
    )

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"fake-bytes"),
        patch(
            "worker.face_client.detect_faces_batch",
            return_value={"3": detect_result, "1": detect_result, "2": detect_result},
        ),
        patch("worker._persist_detection", side_effect=fake_persist),
        patch("worker.shutdown_requested", False),
    ):
        count = process_images(project)

    assert count == 3
    assert (1, False, 1, 99) in persisted
    assert (2, False, 1, 99) in persisted
    assert (3, True, 1, 99) in persisted


def test_process_images_caps_bootstrap_when_already_processed():
    # bs_ratio = 400/1000 = 0.4 <= 0.5  →  bootstrap_cap = base_cap = 900
    # bs_already_processed = 900 >= bootstrap_cap = 900  →  bootstrap_remaining = 0
    # The bootstrapped=1 pending query must NOT be executed.
    project = {"id": 99}
    bootstrap_pending_queried = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SUM(CASE WHEN bootstrapped = 1 AND status != 'pending'" in sql:
            return [{"bs_done": 900, "total": 1000}]
        if "COUNT(*) AS cnt FROM images" in sql and "bootstrapped = 1" in sql:
            # bs_ratio = 400/1000 = 0.4  →  bootstrap_cap stays at base_cap (900)
            return [{"cnt": 400}]
        if "bootstrapped = 0" in sql and "LIMIT" in sql:
            return []
        if "bootstrapped = 1" in sql and "LIMIT" in sql:
            bootstrap_pending_queried.append(sql)
            return []
        if "UPDATE projects SET images_processed" in sql:
            return 1
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection"),
        patch("worker.face_client.detect_faces_batch"),
        patch("worker.shutdown_requested", False),
    ):
        count = process_images(project)

    assert count == 0
    assert not bootstrap_pending_queried, "Bootstrap pending query should not be executed when cap is already met"


try:
    from conftest import _make_embedding
except ModuleNotFoundError:
    from tests.conftest import _make_embedding


@pytest.mark.integration
def test_inference_with_real_db_skips_below_min_confirmed(db_pool, db_conn, seed_project, seed_user):
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (seed_project["id"], 300001, "File:Below_Min_1.jpg", "processed", 1, 800, 600),
        )
        image_id = cur.lastrowid

        for idx in range(3):
            cur.execute(
                "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s, %s, %s, %s)",
                (image_id, _make_embedding(idx), 50, 200, 200, 50, 1, "human", seed_user["id"]),
            )

    project = {"id": seed_project["id"], "min_confirmed": 5, "distance_threshold": 0.6}

    with patch("worker.shutdown_requested", False):
        classified = run_autonomous_inference(project)

    assert classified == 0

    # A run that did nothing is nothing to announce.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT last_inference_at, last_inference_matches FROM projects WHERE id = %s", (seed_project["id"],)
        )
        recorded = cur.fetchone()
    assert recorded["last_inference_at"] is None and recorded["last_inference_matches"] is None


@pytest.mark.integration
def test_inference_with_real_db_classifies_faces(
    db_pool,
    db_conn,
    seed_project,
    seed_images,
    seed_faces,
    seed_unclassified_faces,
):
    project = {
        "id": seed_project["id"],
        "min_confirmed": seed_project["min_confirmed"],
        "distance_threshold": float(seed_project["distance_threshold"]),
    }

    with patch("worker.shutdown_requested", False):
        classified = run_autonomous_inference(project)

    assert classified == 5

    with db_conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(seed_unclassified_faces))
        cur.execute(
            "SELECT id, is_target, classified_by, confidence FROM faces WHERE id IN (" + placeholders + ") ORDER BY id",
            tuple(seed_unclassified_faces),
        )
        rows = cur.fetchall()

    assert len(rows) == 5
    for row in rows:
        assert row["is_target"] in (0, 1)
        assert row["classified_by"] == "model"
        assert row["confidence"] is not None

    # The run is on record, with what it found: the project page keeps its "complete" card from it.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT last_inference_at, last_inference_matches FROM projects WHERE id = %s", (seed_project["id"],)
        )
        recorded = cur.fetchone()
    assert recorded["last_inference_at"] is not None
    assert recorded["last_inference_matches"] == sum(1 for row in rows if row["is_target"] == 1)
    with db_conn.cursor() as cur:
        cur.execute("SELECT last_inference_run FROM projects WHERE id = %s", (seed_project["id"],))
        assert cur.fetchone()["last_inference_run"] == 1


@pytest.mark.integration
def test_inference_with_real_db_skips_faces_with_a_queued_removal(
    db_pool,
    db_conn,
    seed_project,
    seed_images,
    seed_faces,
    seed_unclassified_faces,
):
    """A face with sdc_removal_pending=1 is not model-classified (claim rescue)."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE faces SET sdc_removal_pending = 1 WHERE id = %s", (seed_unclassified_faces[0],))

    project = {
        "id": seed_project["id"],
        "min_confirmed": seed_project["min_confirmed"],
        "distance_threshold": float(seed_project["distance_threshold"]),
    }

    with patch("worker.shutdown_requested", False):
        classified = run_autonomous_inference(project)

    assert classified == 4

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, sdc_removal_pending FROM faces WHERE id = %s",
            (seed_unclassified_faces[0],),
        )
        row = cur.fetchone()

    assert row["is_target"] is None
    assert row["classified_by"] is None
    assert row["sdc_removal_pending"] == 1


@pytest.mark.integration
def test_inference_rechecks_removal_queued_after_candidate_selection(
    db_pool, db_conn, seed_project, seed_images, seed_faces, seed_unclassified_faces, monkeypatch
):
    real_query = _worker_module.execute_query

    def queue_after_snapshot(sql, params=None, fetch=True):
        rows = real_query(sql, params, fetch)
        if "SELECT f.id, f.image_id, f.encoding" in sql:
            with db_conn.cursor() as cur:
                cur.execute("UPDATE faces SET sdc_removal_pending = 1 WHERE id = %s", (seed_unclassified_faces[0],))
        return rows

    monkeypatch.setattr(_worker_module, "execute_query", queue_after_snapshot)
    monkeypatch.setattr(_worker_module, "shutdown_requested", False)
    assert run_autonomous_inference(seed_project) == len(seed_unclassified_faces) - 1

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, sdc_removal_pending FROM faces WHERE id = %s",
            (seed_unclassified_faces[0],),
        )
        face = cur.fetchone()
    assert face == {"is_target": None, "classified_by": None, "sdc_removal_pending": 1}


@pytest.mark.integration
def test_inference_with_real_db_respects_threshold(db_pool, db_conn, seed_user, seed_project, seed_images):
    with db_conn.cursor() as cur:
        for img in seed_images:
            cur.execute(
                "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s, %s, %s, %s)",
                (img["id"], _make_embedding(0), 50, 200, 200, 50, 1, "human", seed_user["id"]),
            )

        # Two images without a confirmed target: a candidate on an image that
        # already has one is always a non-match, which would mask the threshold
        # outcome this test pins.
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, face_count, "
            "detection_width, detection_height) VALUES (%s, %s, %s, 'processed', 1, 800, 600)",
            (seed_project["id"], 100006, "File:Douglas_Adams_6.jpg"),
        )
        close_image_id = cur.lastrowid
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, face_count, "
            "detection_width, detection_height) VALUES (%s, %s, %s, 'processed', 1, 800, 600)",
            (seed_project["id"], 100007, "File:Douglas_Adams_7.jpg"),
        )
        far_image_id = cur.lastrowid

        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s)",
            (close_image_id, _make_embedding(0), 100, 300, 300, 100),
        )
        close_face_id = cur.lastrowid

        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s)",
            (far_image_id, _make_embedding(9999), 120, 320, 320, 120),
        )
        far_face_id = cur.lastrowid

    project = {"id": seed_project["id"], "min_confirmed": 5, "distance_threshold": 0.6}

    with patch("worker.shutdown_requested", False):
        classified = run_autonomous_inference(project)

    assert classified == 2

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT id, is_target FROM faces WHERE id IN (%s, %s) ORDER BY id",
            (close_face_id, far_face_id),
        )
        rows = cur.fetchall()

    assert rows[0]["id"] == close_face_id
    assert rows[0]["is_target"] == 1
    assert rows[1]["id"] == far_face_id
    assert rows[1]["is_target"] == 0


@pytest.mark.integration
def test_inference_classifies_unclassified_faces_on_bootstrap_images(
    db_pool,
    db_conn,
    seed_project,
    seed_images,
    seed_faces,
    seed_bootstrap_image,
):
    """Bootstrapped images are inference candidates, by design.

    Bootstrap auto-classifies the single best face (is_target = 1), so in the
    normal flow that face is never a candidate — candidates require
    is_target IS NULL. Multi-face bootstrap images still get their remaining
    faces classified. The original version of this test asserted the opposite;
    that expectation predates the worker change that deliberately included
    bootstrapped images so multi-face images are handled.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE faces SET is_target = NULL, classified_by = NULL, confidence = NULL WHERE id = %s",
            (seed_bootstrap_image["face_id"],),
        )

    project = {"id": seed_project["id"], "min_confirmed": 5, "distance_threshold": 0.6}

    with patch("worker.shutdown_requested", False):
        _ = run_autonomous_inference(project)

    with db_conn.cursor() as cur:
        cur.execute("SELECT is_target, classified_by FROM faces WHERE id = %s", (seed_bootstrap_image["face_id"],))
        bootstrap_face = cur.fetchone()

    assert bootstrap_face["is_target"] is not None
    assert bootstrap_face["classified_by"] == "model"


@pytest.mark.integration
def test_soft_delete_recreate_no_fk_errors(db_pool, db_conn, seed_user):
    """Soft-delete → hard-delete (CASCADE) → re-create with same QID/category causes no FK errors."""
    cur = db_conn.cursor()

    # --- Phase 1: create project and simulate worker output (images + faces) ---
    cur.execute(
        "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
        "distance_threshold, min_confirmed, status, images_total, images_processed) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (seed_user["id"], "Q42", "Douglas Adams", "Douglas Adams", 0.6, 5, "active", 3, 3),
    )
    project_id_v1 = cur.lastrowid

    img_ids = []
    for title, page_id in [("File:DA_1.jpg", 900001), ("File:DA_2.jpg", 900002), ("File:DA_3.jpg", 900003)]:
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (project_id_v1, page_id, title, "processed", 1, 800, 600),
        )
        img_ids.append(cur.lastrowid)

    face_ids = []
    for i, img_id in enumerate(img_ids):
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s, %s, %s, %s)",
            (img_id, _make_embedding(700 + i), 50, 200, 200, 50, 1, "human", seed_user["id"]),
        )
        face_ids.append(cur.lastrowid)

    # Sanity: data exists
    cur.execute("SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s", (project_id_v1,))
    assert cur.fetchone()["cnt"] == 3
    cur.execute("SELECT COUNT(*) AS cnt FROM faces WHERE image_id IN (%s,%s,%s)", tuple(img_ids))
    assert cur.fetchone()["cnt"] == 3

    # --- Phase 2: soft-delete then hard-delete (mirrors worker main loop) ---
    cur.execute("UPDATE projects SET status = 'deleted' WHERE id = %s", (project_id_v1,))
    cur.execute("DELETE FROM projects WHERE id = %s AND status = 'deleted'", (project_id_v1,))

    # CASCADE must have removed child rows
    cur.execute("SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s", (project_id_v1,))
    assert cur.fetchone()["cnt"] == 0
    cur.execute("SELECT COUNT(*) AS cnt FROM faces WHERE image_id IN (%s,%s,%s)", tuple(img_ids))
    assert cur.fetchone()["cnt"] == 0

    # --- Phase 3: re-create project with same QID + category ---
    cur.execute(
        "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
        "distance_threshold, min_confirmed, status, images_total, images_processed) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (seed_user["id"], "Q42", "Douglas Adams", "Douglas Adams", 0.6, 5, "active", 0, 0),
    )
    project_id_v2 = cur.lastrowid
    assert project_id_v2 != project_id_v1  # new row, different auto-increment id

    # Insert fresh images + faces — must not hit FK errors
    cur.execute(
        "INSERT INTO images (project_id, commons_page_id, file_title, status, "
        "face_count, detection_width, detection_height) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (project_id_v2, 900010, "File:DA_New.jpg", "processed", 1, 800, 600),
    )
    new_img_id = cur.lastrowid

    cur.execute(
        "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
        "VALUES (%s, %s, 'scrfd500m+arcface_mbf', %s, %s, %s, %s)",
        (new_img_id, _make_embedding(800), 50, 200, 200, 50),
    )
    new_face_id = cur.lastrowid

    # Verify new project data is independently correct
    cur.execute("SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s", (project_id_v2,))
    assert cur.fetchone()["cnt"] == 1
    cur.execute("SELECT id, image_id FROM faces WHERE id = %s", (new_face_id,))
    row = cur.fetchone()
    assert row is not None
    assert row["image_id"] == new_img_id

    cur.close()


@pytest.mark.integration
def test_inference_does_not_touch_human_classified(db_pool, db_conn, seed_project, seed_images, seed_faces):
    with db_conn.cursor() as cur:
        face_ids = seed_faces["target"] + seed_faces["non_target"]
        placeholders = ",".join(["%s"] * len(face_ids))
        cur.execute(
            "SELECT id, is_target FROM faces WHERE id IN (" + placeholders + ") ORDER BY id",
            tuple(face_ids),
        )
        before_rows = cur.fetchall()

    project = {"id": seed_project["id"], "min_confirmed": 5, "distance_threshold": 0.6}

    with patch("worker.shutdown_requested", False):
        _ = run_autonomous_inference(project)

    with db_conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(face_ids))
        cur.execute(
            "SELECT id, is_target FROM faces WHERE id IN (" + placeholders + ") ORDER BY id",
            tuple(face_ids),
        )
        after_rows = cur.fetchall()

    assert before_rows == after_rows


@pytest.mark.integration
def test_claim_shadow_project_real_db_picks_finished_project(db_pool, db_conn, seed_project):
    """A completed project with unchecked crawled images is sweep-claimable."""
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status = 'completed', prefilter_mode = 'shadow' WHERE id = %s",
            (seed_project["id"],),
        )
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900001, "File:Finished_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == seed_project["id"]

    with db_conn.cursor() as cur:
        cur.execute("SELECT worker_claimed_by FROM projects WHERE id = %s", (seed_project["id"],))
        assert cur.fetchone()["worker_claimed_by"] == "test-worker-1"


@pytest.mark.integration
def test_claim_shadow_project_real_db_picks_quiescent_active_project(db_pool, db_conn, seed_project):
    """Active but fully processed and classified: the pipelines leave it alone."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900003, "File:Quiescent_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == seed_project["id"]


@pytest.mark.integration
def test_claim_shadow_project_real_db_skips_project_with_pending_images(db_pool, db_conn, seed_project):
    """An active project with pending images belongs to the active pipeline."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'pending', 0, 0, 800, 600)",
            (seed_project["id"], 900002, "File:Pending_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_shadow_project() is None


@pytest.mark.integration
def test_claim_shadow_project_real_db_skips_inference_eligible_completed_project(
    db_pool, db_conn, seed_project, seed_images, seed_faces, seed_unclassified_faces
):
    """The inference path owns a completed project that crossed min_confirmed."""
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status = 'completed', prefilter_mode = 'shadow' WHERE id = %s",
            (seed_project["id"],),
        )

    with patch("worker._worker_id", "test-worker-1"):
        # The inference claim does take it …
        assert [p["id"] for p in _claim_inference_projects(1)] == [seed_project["id"]]
        # … and the sweep must leave it alone even after that claim is released.
        _release_project(seed_project["id"])
        assert _claim_shadow_project() is None


@pytest.mark.integration
def test_inference_claim_waits_for_the_images_of_an_active_project_but_not_of_a_completed_one(
    db_pool, db_conn, seed_project, seed_images, seed_faces, seed_unclassified_faces
):
    """The model starts once every image is processed: an idle worker must not pick the project up before."""
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status) VALUES (%s, %s, %s, 'pending')",
            (seed_project["id"], 900100, "File:Still_pending.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_inference_projects(1) == []  # active, faces waiting, but an image is still pending

        with db_conn.cursor() as cur:
            cur.execute("UPDATE images SET status = 'processed' WHERE status = 'pending'")
        assert [p["id"] for p in _claim_inference_projects(1)] == [seed_project["id"]]  # all processed: now it may
        _release_project(seed_project["id"])

        # A completed project is no longer processed, so a leftover pending image does not hold it back.
        with db_conn.cursor() as cur:
            cur.execute("UPDATE projects SET status = 'completed' WHERE id = %s", (seed_project["id"],))
            cur.execute(
                "INSERT INTO images (project_id, commons_page_id, file_title, status) VALUES (%s, %s, %s, 'pending')",
                (seed_project["id"], 900101, "File:Left_over.jpg"),
            )
        assert [p["id"] for p in _claim_inference_projects(1)] == [seed_project["id"]]


@pytest.mark.integration
def test_claim_shadow_project_real_db_takes_project_whose_only_unclassified_faces_are_queued_removals(
    db_pool, db_conn, seed_project, seed_images, seed_faces, seed_unclassified_faces
):
    """Queued removals make a face non-inferable, so the sweep may cover the project.

    ``_claim_inference_projects`` excludes faces with ``sdc_removal_pending = 1``
    because ``run_autonomous_inference`` skips them; with no inferable face left,
    the completed project belongs to the shadow sweep instead of bouncing through
    the inference queue every cycle.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET status = 'completed', prefilter_mode = 'shadow' WHERE id = %s",
            (seed_project["id"],),
        )
        placeholders = ",".join(["%s"] * len(seed_unclassified_faces))
        cur.execute(
            f"UPDATE faces SET sdc_removal_pending = 1 WHERE id IN ({placeholders})",
            seed_unclassified_faces,
        )
        # An unchecked non-bootstrapped image so the sweep has pre-filter work.
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900004, "File:Queued_Removal_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        # No inferable face left: the inference claim must not take the project …
        assert _claim_inference_projects(1) == []
        # … and the sweep claims it as otherwise unclaimed work.
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == seed_project["id"]


@pytest.mark.integration
def test_claim_shadow_project_real_db_respects_other_worker_claims(db_pool, db_conn, seed_project):
    """A fresh claim by the other worker defers the sweep; an expired one does not."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900010, "File:Claimed_1.jpg"),
        )
        cur.execute(
            "UPDATE projects SET worker_claimed_by = 'other-worker', worker_claimed_at = NOW() WHERE id = %s",
            (seed_project["id"],),
        )

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_shadow_project() is None

    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET worker_claimed_at = NOW() - INTERVAL 20 MINUTE WHERE id = %s",
            (seed_project["id"],),
        )

    with patch("worker._worker_id", "test-worker-1"):
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == seed_project["id"]


@pytest.mark.integration
def test_claim_shadow_project_real_db_skips_sdc_requested_project(db_pool, db_conn, seed_project):
    """A queued Send Edits request belongs to the SDC path, not the sweep."""
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE projects SET prefilter_mode = 'shadow', sdc_write_requested = 1 WHERE id = %s",
            (seed_project["id"],),
        )
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900011, "File:Sdc_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_shadow_project() is None

    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET sdc_write_requested = 0 WHERE id = %s", (seed_project["id"],))

    with patch("worker._worker_id", "test-worker-1"):
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == seed_project["id"]


@pytest.mark.integration
def test_claim_shadow_project_real_db_skips_mode_off_and_bootstrapped_backlog(db_pool, db_conn, seed_project):
    """Mode off or an only-bootstrapped backlog are never sweep candidates."""
    with db_conn.cursor() as cur:
        # prefilter_mode defaults to 'off'
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900012, "File:Off_1.jpg"),
        )

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_shadow_project() is None

    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        cur.execute("UPDATE images SET bootstrapped = 1 WHERE commons_page_id = %s", (900012,))

    with patch("worker._worker_id", "test-worker-1"):
        assert _claim_shadow_project() is None


@pytest.mark.integration
def test_claim_shadow_project_real_db_rotates_to_least_recently_claimed(db_pool, db_conn, seed_project):
    """A never-claimed candidate is picked before a freshly claimed one."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (seed_project["id"], 900013, "File:Rotation_1.jpg"),
        )
        cur.execute(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
            "distance_threshold, min_confirmed, status, images_total, images_processed, prefilter_mode) "
            "VALUES (%s, 'Q43', 'Category_B', 'B', 0.6, 5, 'active', 1, 1, 'shadow')",
            (seed_project["user_id"],),
        )
        other_id = cur.lastrowid
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, bootstrapped, "
            "face_count, detection_width, detection_height) "
            "VALUES (%s, %s, %s, 'processed', 0, 0, 800, 600)",
            (other_id, 900014, "File:B_1.jpg"),
        )
        # The seed project was just swept by the other worker; the second one never was.
        cur.execute(
            "UPDATE projects SET worker_claimed_by = 'other-worker', worker_claimed_at = NOW() WHERE id = %s",
            (seed_project["id"],),
        )

    with patch("worker._worker_id", "test-worker-1"):
        project = _claim_shadow_project()

    assert project is not None
    assert project["id"] == other_id


# ---------------------------------------------------------------------------
# Unit tests: distributed-claim helpers
# ---------------------------------------------------------------------------


def test_claim_active_projects_success():
    """_claim_active_projects returns whatever the transaction closure produces."""
    fake_projects = [{"id": 1, "status": "active"}, {"id": 2, "status": "active"}]

    with patch("worker.execute_transaction", return_value=fake_projects) as mock_txn:
        result = _claim_active_projects(max_count=3)

    assert result == fake_projects
    mock_txn.assert_called_once()


def test_claim_active_projects_returns_empty_on_db_error():
    """_claim_active_projects swallows DatabaseError and returns []."""
    from database import DatabaseError

    with patch("worker.execute_transaction", side_effect=DatabaseError("boom")):
        result = _claim_active_projects(max_count=3)

    assert result == []


def test_claim_active_projects_returns_empty_when_no_rows():
    """_claim_active_projects returns [] when the transaction finds nothing to claim."""
    with patch("worker.execute_transaction", return_value=[]):
        result = _claim_active_projects(max_count=3)

    assert result == []


def test_claim_inference_projects_success():
    """_claim_inference_projects returns whatever the transaction closure produces."""
    fake_projects = [{"id": 5, "status": "completed"}]

    with patch("worker.execute_transaction", return_value=fake_projects) as mock_txn:
        result = _claim_inference_projects(max_count=2)

    assert result == fake_projects
    mock_txn.assert_called_once()


def test_claim_inference_projects_returns_empty_on_db_error():
    """_claim_inference_projects swallows DatabaseError and returns []."""
    from database import DatabaseError

    with patch("worker.execute_transaction", side_effect=DatabaseError("boom")):
        result = _claim_inference_projects(max_count=2)

    assert result == []


def test_claim_sdc_projects_success():
    """_claim_sdc_projects returns (projects, pre_claimed_ids) from the transaction."""
    fake_projects = [{"id": 7, "sdc_write_requested": 1}]
    fake_pre_claimed = set()

    with patch("worker.execute_transaction", return_value=(fake_projects, fake_pre_claimed)) as mock_txn:
        projects, pre_claimed_ids = _claim_sdc_projects()

    assert projects == fake_projects
    assert pre_claimed_ids == fake_pre_claimed
    mock_txn.assert_called_once()


def test_claim_sdc_projects_returns_empty_on_db_error():
    """_claim_sdc_projects swallows DatabaseError and returns ([], set())."""
    from database import DatabaseError

    with patch("worker.execute_transaction", side_effect=DatabaseError("boom")):
        projects, pre_claimed_ids = _claim_sdc_projects()

    assert projects == []
    assert pre_claimed_ids == set()


def test_release_project_issues_correct_query():
    """_release_project updates the correct project row using the worker id."""
    _worker_module._worker_id = "test-worker-1"

    with patch("worker.execute_query") as mock_q:
        _release_project(42)

    mock_q.assert_called_once()
    call_args = mock_q.call_args
    sql, params = call_args[0][0], call_args[0][1]
    assert "worker_claimed_by = NULL" in sql
    assert params == (42, "test-worker-1")


def test_release_project_swallows_db_error():
    """_release_project does not raise when execute_query throws DatabaseError."""
    from database import DatabaseError

    _worker_module._worker_id = "test-worker-1"

    with patch("worker.execute_query", side_effect=DatabaseError("gone")):
        _release_project(99)  # must not raise


def test_release_all_claims_issues_correct_query():
    """_release_all_claims releases all rows owned by this worker."""
    _worker_module._worker_id = "test-worker-2"

    with patch("worker.execute_query", return_value=3) as mock_q:
        _release_all_claims()

    mock_q.assert_called_once()
    call_args = mock_q.call_args
    sql, params = call_args[0][0], call_args[0][1]
    assert "worker_claimed_by = NULL" in sql
    assert params == ("test-worker-2",)


def test_release_all_claims_swallows_db_error():
    """_release_all_claims does not raise when execute_query throws DatabaseError."""
    from database import DatabaseError

    _worker_module._worker_id = "test-worker-2"

    with patch("worker.execute_query", side_effect=DatabaseError("gone")):
        _release_all_claims()  # must not raise


def test_refresh_claims_issues_correct_query():
    """_refresh_claims updates worker_claimed_at for this worker's rows."""
    _worker_module._worker_id = "test-worker-3"

    with patch("worker.execute_query", return_value=2) as mock_q:
        _refresh_claims()

    mock_q.assert_called_once()
    call_args = mock_q.call_args
    sql, params = call_args[0][0], call_args[0][1]
    assert "worker_claimed_at = NOW()" in sql
    assert params == ("test-worker-3",)


def test_refresh_claims_swallows_db_error():
    """_refresh_claims does not raise when execute_query throws DatabaseError."""
    from database import DatabaseError

    _worker_module._worker_id = "test-worker-3"

    with patch("worker.execute_query", side_effect=DatabaseError("gone")):
        _refresh_claims()  # must not raise


def test_process_and_release_releases_on_success():
    """_process_and_release releases the project claim after process_project succeeds."""
    project = {"id": 10}
    _worker_module._worker_id = "test-worker-4"

    with (
        patch("worker.process_project") as mock_process,
        patch("worker._release_project") as mock_release,
    ):
        _process_and_release(project)

    mock_process.assert_called_once_with(project, skip_discovery=False)
    mock_release.assert_called_once_with(10)


def test_process_and_release_releases_even_on_exception():
    """_process_and_release releases the project claim even when process_project raises."""
    project = {"id": 11}
    _worker_module._worker_id = "test-worker-4"

    with (
        patch("worker.process_project", side_effect=RuntimeError("crash")),
        patch("worker._release_project") as mock_release,
    ):
        try:
            _process_and_release(project)
        except RuntimeError:
            pass

    mock_release.assert_called_once_with(11)


def test_infer_and_release_releases_on_success():
    """_infer_and_release releases the claim and returns the inference count."""
    project = {"id": 20}
    _worker_module._worker_id = "test-worker-5"

    with (
        patch("worker.run_autonomous_inference", return_value=7) as mock_infer,
        patch("worker._release_project") as mock_release,
    ):
        result = _infer_and_release(project)

    assert result == 7
    mock_infer.assert_called_once_with(project)
    mock_release.assert_called_once_with(20)


def test_infer_and_release_releases_even_on_exception():
    """_infer_and_release releases the project claim even when run_autonomous_inference raises."""
    project = {"id": 21}
    _worker_module._worker_id = "test-worker-5"

    with (
        patch("worker.run_autonomous_inference", side_effect=RuntimeError("crash")),
        patch("worker._release_project") as mock_release,
    ):
        try:
            _infer_and_release(project)
        except RuntimeError:
            pass

    mock_release.assert_called_once_with(21)


def test_run_new_inference_projects_claims_and_runs_each_project():
    """_run_new_inference_projects claims once and runs every claimed project inline."""
    projects = [{"id": 30}, {"id": 31}]

    with (
        patch("worker._claim_inference_projects", return_value=projects) as mock_claim,
        patch("worker._infer_and_release", side_effect=[5, 0]) as mock_infer,
    ):
        result = _run_new_inference_projects(2)

    assert result == 2
    mock_claim.assert_called_once_with(2)
    assert [call.args[0]["id"] for call in mock_infer.call_args_list] == [30, 31]


def test_run_new_inference_projects_handles_empty_and_failed_projects():
    """No claims means no inference; one failing project does not stop the pass."""
    with (
        patch("worker._claim_inference_projects", return_value=[]) as mock_claim,
        patch("worker._infer_and_release") as mock_infer,
    ):
        assert _run_new_inference_projects() == 0

    mock_claim.assert_called_once_with(1)
    mock_infer.assert_not_called()

    with (
        patch("worker._claim_inference_projects", return_value=[{"id": 40}, {"id": 41}]),
        patch("worker._infer_and_release", side_effect=[RuntimeError("boom"), 3]),
    ):
        assert _run_new_inference_projects(2) == 1


def test_write_sdc_marks_image_bootstrapped_on_idempotency_hit():
    """When P180 already exists on Commons, write_sdc_claims sets bootstrapped=1 on the image."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    existing_claim_response = {"claims": {"P180": [{"mainsnak": {"datavalue": {"value": {"id": "Q42"}}}}]}}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not any("UPDATE faces SET sdc_written" in c[0] for c in db_calls):
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE images SET bootstrapped" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return []

    mock_api_resp = MagicMock()
    mock_api_resp.json.return_value = existing_claim_response

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", return_value=mock_api_resp),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    # Verify sdc_written=1 was set on the face, only while it is still a target.
    # The claim predates this run, so a reset face must not be marked written
    # and must not have a removal queued (that could delete someone else's claim).
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 1
    assert face_updates[0][1] == (100,)
    assert "is_target = 1" in face_updates[0][0]
    assert not any("UPDATE faces" in sql and "sdc_removal_pending = 1" in sql for sql, _ in db_calls)

    # Verify bootstrapped=1 was set on the image
    img_updates = [(sql, p) for sql, p in db_calls if "UPDATE images SET bootstrapped" in sql]
    assert len(img_updates) == 1
    assert img_updates[0][1] == (200,)


def test_write_sdc_insert_ignore_zero_written_at_not_null():
    """When INSERT IGNORE returns 0 and existing claim has written_at set, mark face as written."""
    from datetime import datetime

    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not any("UPDATE faces SET sdc_written" in c[0] for c in db_calls):
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 0  # duplicate row
        if "SELECT project_id, written_at, claimed_at FROM sdc_claims" in sql:
            return [{"project_id": 99, "written_at": datetime(2025, 1, 1), "claimed_at": datetime(2025, 1, 1)}]
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request"),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    # Verify sdc_written=1 was set on the face
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 1
    assert face_updates[0][1] == (100,)

    # No API call should have been made
    api_calls = [c for c in db_calls if "wbgetclaims" in str(c)]
    assert len(api_calls) == 0


def test_write_sdc_insert_ignore_zero_same_project_retries_write():
    """When INSERT IGNORE returns 0 and same project owns the claim, fall through to write."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not any("UPDATE faces SET sdc_written" in c[0] for c in db_calls):
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 0  # duplicate row
        if "SELECT project_id, written_at, claimed_at FROM sdc_claims" in sql:
            # Same project (id=5) owns the claim, not yet written
            return [{"project_id": 5, "written_at": None, "claimed_at": None}]
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    no_claims_response = {"claims": {}}
    write_response = {"success": 1}

    api_responses = [no_claims_response, write_response]
    call_count = [0]

    def mock_api_side_effect(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = api_responses[call_count[0] % len(api_responses)]
        call_count[0] += 1
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api_side_effect),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    # Verify sdc_written=1 was set on the face (write actually happened)
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 1

    # Verify written_at was updated in sdc_claims
    sdc_updates = [(sql, p) for sql, p in db_calls if "UPDATE sdc_claims SET written_at" in sql]
    assert len(sdc_updates) == 1


def test_write_sdc_insert_ignore_zero_other_project_fresh_skips():
    """When INSERT IGNORE returns 0 and another project holds a fresh claim, skip the face."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            # Return the face once, then empty to end the loop
            face_query_calls = [c for c in db_calls if "SELECT f.id as face_id" in c[0]]
            if len(face_query_calls) == 1:
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 0  # duplicate row
        if "SELECT project_id, written_at, claimed_at FROM sdc_claims" in sql:
            # Another project (id=99) holds a fresh claim
            return [{"project_id": 99, "written_at": None, "claimed_at": None}]
        if "UPDATE sdc_claims SET project_id" in sql:
            # Reclaim attempt fails (fresh claim held by other project)
            return 0
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request"),
        patch("worker.time.sleep"),  # the blocked-page back-off must not really wait
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    # Face was skipped — no write happened
    assert result == 0

    # Verify face was NOT marked as written
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 0

    # Verify reclaim was attempted
    reclaim_attempts = [(sql, p) for sql, p in db_calls if "UPDATE sdc_claims SET project_id" in sql]
    assert len(reclaim_attempts) == 1


def test_write_sdc_insert_ignore_zero_other_project_stale_reclaims():
    """When INSERT IGNORE returns 0 and another project holds a stale claim, reclaim and write."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not any("UPDATE faces SET sdc_written" in c[0] for c in db_calls):
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 0  # duplicate row
        if "SELECT project_id, written_at, claimed_at FROM sdc_claims" in sql:
            # Another project (id=99) holds the claim, not yet written
            return [{"project_id": 99, "written_at": None, "claimed_at": None}]
        if "UPDATE sdc_claims SET project_id" in sql:
            # Reclaim succeeds (stale claim)
            return 1
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    no_claims_response = {"claims": {}}
    write_response = {"success": 1}
    api_responses = [no_claims_response, write_response]
    call_count = [0]

    def mock_api_side_effect(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = api_responses[call_count[0] % len(api_responses)]
        call_count[0] += 1
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api_side_effect),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    # Verify reclaim was attempted and succeeded
    reclaim_attempts = [(sql, p) for sql, p in db_calls if "UPDATE sdc_claims SET project_id" in sql]
    assert len(reclaim_attempts) == 1

    # Verify face was marked as written after reclaim + API write
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 1


def test_write_sdc_uses_sdc_write_user_id_when_set():
    """write_sdc_claims should use sdc_write_user_id for token lookup when set, not the project owner."""
    project = {"id": 5, "user_id": 1, "sdc_write_user_id": 99, "wikidata_qid": "Q42"}

    token_user_ids = []

    def mock_refresh(uid):
        token_user_ids.append(uid)
        return "fake-token"

    def mock_execute_query(sql, params=None, fetch=True):
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            return []
        if "sdc_removal_pending" in sql and "SELECT" in sql:
            return []
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", side_effect=mock_refresh),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker.shutdown_requested", False),
    ):
        write_sdc_claims(project)

    assert token_user_ids[0] == 99, f"Expected user 99 (sdc_write_user_id), got {token_user_ids[0]}"


def test_write_sdc_falls_back_to_owner_when_sdc_write_user_id_missing():
    """write_sdc_claims should fall back to project owner when sdc_write_user_id is not set."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    token_user_ids = []

    def mock_refresh(uid):
        token_user_ids.append(uid)
        return "fake-token"

    def mock_execute_query(sql, params=None, fetch=True):
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            return []
        if "sdc_removal_pending" in sql and "SELECT" in sql:
            return []
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", side_effect=mock_refresh),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker.shutdown_requested", False),
    ):
        write_sdc_claims(project)

    assert token_user_ids[0] == 1, f"Expected user 1 (owner), got {token_user_ids[0]}"


def test_write_sdc_no_such_entity_on_wbgetclaims_skips_face():
    """no-such-entity on wbgetclaims idempotency check skips the face, writes remaining."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []
    face_batch_call = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            face_batch_call[0] += 1
            if face_batch_call[0] == 1:
                return [
                    {"face_id": 100, "image_id": 200, "commons_page_id": 8888},
                    {"face_id": 101, "image_id": 201, "commons_page_id": 9999},
                ]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 1
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    api_call_count = [0]

    def mock_api_side_effect(*args, **kwargs):
        api_call_count[0] += 1
        resp = MagicMock()
        if api_call_count[0] == 1:
            resp.json.return_value = {"error": {"code": "no-such-entity", "info": "no-such-entity"}}
        elif api_call_count[0] == 2:
            resp.json.return_value = {"claims": {}}
        else:
            resp.json.return_value = {"success": 1}
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api_side_effect),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 2
    assert face_updates[0][1] == (100,)  # skipped face marked terminal
    assert face_updates[1][1] == (101,)  # successfully written face

    error_aborts = [(sql, params) for sql, params in db_calls if "sdc_write_error" in sql and "UPDATE projects" in sql]
    assert all(params is None or params[0] is None for _, params in error_aborts)


def test_write_sdc_no_such_entity_on_wbeditentity_skips_face():
    """no-such-entity on wbeditentity skips the face, writes remaining."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []
    face_batch_call = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            face_batch_call[0] += 1
            if face_batch_call[0] == 1:
                return [
                    {"face_id": 100, "image_id": 200, "commons_page_id": 8888},
                    {"face_id": 101, "image_id": 201, "commons_page_id": 9999},
                ]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 1
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    api_call_count = [0]

    def mock_api_side_effect(*args, **kwargs):
        api_call_count[0] += 1
        resp = MagicMock()
        if api_call_count[0] == 1:
            resp.json.return_value = {"claims": {}}
        elif api_call_count[0] == 2:
            resp.json.return_value = {"error": {"code": "no-such-entity", "info": "no-such-entity"}}
        elif api_call_count[0] == 3:
            resp.json.return_value = {"claims": {}}
        else:
            resp.json.return_value = {"success": 1}
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api_side_effect),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 2
    assert face_updates[0][1] == (100,)  # skipped face marked terminal
    assert face_updates[1][1] == (101,)  # successfully written face


def test_write_sdc_no_such_entity_on_removal_skips_and_clears_flag():
    """no-such-entity on wbgetclaims during removal clears sdc_removal_pending and continues."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            return []
        if "SELECT DISTINCT i.commons_page_id" in sql and "sdc_removal_pending" in sql:
            if not any("sdc_removal_pending = 0" in c[0] for c in db_calls):
                return [{"commons_page_id": 7777}]
            return []
        if "WHERE i.commons_page_id" in sql and "sdc_removal_pending = 1" in sql and "NOT EXISTS" in sql:
            return [{"face_id": 300}]
        if "sdc_removal_pending = 0" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    def mock_api_side_effect(*args, **kwargs):
        resp = MagicMock()
        resp.json.return_value = {"error": {"code": "no-such-entity", "info": "no-such-entity"}}
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api_side_effect),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0

    removal_clears = [(sql, p) for sql, p in db_calls if "sdc_removal_pending = 0" in sql]
    assert len(removal_clears) == 1
    assert removal_clears[0][1] == (7777, 5)

    # The stale page/QID deduplication row is dropped: keeping a written_at
    # record would make the next send mark faces written without restoring
    # the claim.
    dedup_deletes = [(sql, p) for sql, p in db_calls if "DELETE FROM sdc_claims" in sql]
    assert len(dedup_deletes) == 1
    assert dedup_deletes[0][1] == (7777, "Q42")
    # The clearing statement also drops the stale sdc_written marker.
    assert "f.sdc_written = 0" in removal_clears[0][0]

    # The removal loop keeps liveness and the project claim fresh.
    claim_refreshes = [
        (sql, p) for sql, p in db_calls if "worker_claimed_at = NOW()" in sql and "worker_claimed_by" in sql
    ]
    assert claim_refreshes and claim_refreshes[0][1][0] == 5

    error_aborts = [
        (sql, p)
        for sql, p in db_calls
        if "sdc_write_error" in sql and "UPDATE projects" in sql and p and p[0] is not None
    ]
    assert len(error_aborts) == 0


def test_bootstrap_flags_existing_images_at_cap():
    """When project is at MAX_IMAGES_PER_PROJECT, bootstrap still flags existing images."""
    project = {"id": 7, "user_id": 1, "wikidata_qid": "Q22686", "commons_category": "Donald Trump"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        sql_s = sql.strip()

        # COUNT(*) — project at cap
        if "SELECT COUNT(*) AS cnt FROM images" in sql_s:
            return [{"cnt": _worker_module.MAX_IMAGES_PER_PROJECT}]

        # SELECT existing image by page_id
        if "SELECT id, status FROM images WHERE project_id" in sql_s:
            assert params is not None
            page_id = params[1]
            if page_id == 5001:
                return [{"id": 101, "status": "processed"}]
            if page_id == 5002:
                return [{"id": 102, "status": "processed"}]
            return []

        if "UPDATE images SET bootstrapped = 1 WHERE id" in sql_s:
            return 1
        if "UPDATE faces SET sdc_written = 1" in sql_s:
            return 1
        if "UPDATE faces" in sql_s and "classified_by = 'bootstrap'" in sql_s:
            return 1
        if "UPDATE projects SET faces_confirmed" in sql_s:
            return 1
        if "UPDATE projects SET images_total" in sql_s:
            return 1
        return []

    api_response = MagicMock()
    api_response.json.return_value = {
        "query": {
            "search": [
                {"pageid": 5001, "title": "File:Trump photo.jpg"},
                {"pageid": 5002, "title": "File:Trump rally.jpg"},
                {"pageid": 5003, "title": "File:New trump image.jpg"},
            ]
        },
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_response),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 2

    bootstrap_updates = [(sql, p) for sql, p in db_calls if "UPDATE images SET bootstrapped = 1 WHERE id" in sql]
    assert len(bootstrap_updates) == 2
    flagged_ids = {p[0] for _, p in bootstrap_updates}
    assert flagged_ids == {101, 102}

    sdc_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written = 1" in sql]
    assert len(sdc_updates) == 2

    # Auto-classified bootstrap targets drop any queued P180 removal — the
    # pre-existing claim must stay.
    auto_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces" in sql and "classified_by = 'bootstrap'" in sql]
    assert auto_updates
    assert all("sdc_removal_pending = 0" in sql for sql, _ in auto_updates)

    inserts = [sql for sql, _ in db_calls if "INSERT IGNORE INTO images" in sql]
    assert len(inserts) == 0


def test_bootstrap_at_cap_does_not_insert_new_images():
    """When at cap, bootstrap skips new images entirely (no INSERT)."""
    project = {"id": 8, "user_id": 1, "wikidata_qid": "Q42", "commons_category": "Test"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        sql_s = sql.strip()

        if "SELECT COUNT(*) AS cnt FROM images" in sql_s:
            return [{"cnt": _worker_module.MAX_IMAGES_PER_PROJECT}]
        if "SELECT id, status FROM images WHERE project_id" in sql_s:
            return []
        return []

    api_response = MagicMock()
    api_response.json.return_value = {
        "query": {
            "search": [
                {"pageid": 7001, "title": "File:New1.jpg"},
                {"pageid": 7002, "title": "File:New2.jpg"},
            ]
        },
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_response),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 0

    inserts = [sql for sql, _ in db_calls if "INSERT IGNORE INTO images" in sql]
    assert len(inserts) == 0


def test_process_project_auto_completes_no_faces():
    """process_project sets status=completed, completion_reason=no_faces when 0 faces detected."""
    project = {
        "id": 99,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Empty_Category",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }
    update_calls = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        # _is_still_active check
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        # Face count check — 0 faces
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 0}]
        # Auto-complete UPDATE
        if "UPDATE projects SET status = 'completed'" in sql_s:
            update_calls.append((sql_s, params))
            return 1
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", return_value=0),
        patch("worker.run_autonomous_inference") as mock_inference,
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    # Should have called UPDATE with completion_reason = 'no_faces'
    assert len(update_calls) == 1
    assert "no_faces" in update_calls[0][0]
    # Should NOT have called inference
    mock_inference.assert_not_called()


def test_process_project_auto_completes_insufficient_faces():
    """process_project sets status=completed, completion_reason=insufficient_faces when ≤5 faces < min_confirmed."""
    project = {
        "id": 100,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Few_Faces",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }
    update_calls = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        # 3 faces detected — less than min_confirmed=5 and ≤5
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 3}]
        if "UPDATE projects SET status = 'completed'" in sql_s:
            update_calls.append((sql_s, params))
            return 1
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", return_value=0),
        patch("worker.run_autonomous_inference") as mock_inference,
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    assert len(update_calls) == 1
    assert "insufficient_faces" in update_calls[0][0]
    mock_inference.assert_not_called()


def test_process_project_never_autocompletes_while_images_are_pending():
    """A transient download outage must not be recorded as a permanent verdict.

    Regression for the thumb.wikimedia.org incident: every download failed, so
    process_images returned 0 and the batch loop exited after one pass. Step 3b
    then saw 0 faces and wrote completion_reason='no_faces' — even though 173
    images were still pending. Because _claim_active_projects only picks up
    status='active', those images were never retried.
    """
    project = {
        "id": 20,
        "user_id": 1,
        "wikidata_qid": "Q232101",
        "commons_category": "Rachael Leigh Cook by year",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 10,
    }
    completion_updates = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        # 173 images never got processed because every download failed.
        if "FROM images WHERE project_id" in sql_s and "status = 'pending'" in sql_s:
            return [{"cnt": 173}]
        # Zero faces — but only because detection never ran.
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 0}]
        if "SELECT min_confirmed FROM projects" in sql_s:
            return [{"min_confirmed": 10}]
        if "UPDATE projects SET status = 'completed'" in sql_s:
            completion_updates.append((sql_s, params))
            return 1
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", return_value=0),
        patch("worker.run_autonomous_inference", return_value=0),
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    assert completion_updates == [], f"Project was auto-completed despite 173 pending images: {completion_updates}"


def test_worker_download_image_follows_redirect_to_thumb_host():
    """Commons serves ?width=N thumbnails from thumb.wikimedia.org.

    Real chain: Special:FilePath -> commons/index.php -> thumb.wikimedia.org.
    Rejecting that host fails every thumbnail download.
    """
    hop1 = MagicMock()
    hop1.is_redirect = True
    hop1.headers = {"Location": "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/X.jpg"}
    hop1.close = MagicMock()

    hop2 = MagicMock()
    hop2.is_redirect = True
    hop2.headers = {"Location": "https://thumb.wikimedia.org/wikipedia/commons/thumb/6/61/X.jpg/500px-X.jpg"}
    hop2.close = MagicMock()

    content_resp = MagicMock()
    content_resp.is_redirect = False
    content_resp.headers = {"Content-Length": "4"}
    content_resp.raise_for_status = MagicMock()
    content_resp.iter_content.return_value = iter([b"data"])
    content_resp.close = MagicMock()

    mock_session = MagicMock()
    mock_session.get.side_effect = [hop1, hop2, content_resp]
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        result = _worker_module._download_image("https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg?width=500")
    assert result == b"data"


def test_worker_allowlist_still_rejects_non_wikimedia_hosts():
    """Adding thumb.wikimedia.org must not widen the allowlist to lookalikes."""
    expected = frozenset({"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"})
    assert expected == _worker_module._ALLOWED_DOWNLOAD_HOSTS


def test_process_project_no_auto_complete_when_enough_faces():
    """process_project does NOT auto-complete when faces > 5."""
    project = {
        "id": 101,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Many_Faces",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }
    auto_complete_calls = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        # 10 faces detected — above threshold
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 10}]
        if "UPDATE projects SET status = 'completed'" in sql_s:
            auto_complete_calls.append((sql_s, params))
            return 1
        # Fresh project fetch for inference
        if "SELECT * FROM projects WHERE id" in sql_s:
            return [project.copy()]
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", return_value=0),
        patch("worker.run_autonomous_inference") as mock_inference,
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    # Should NOT auto-complete
    assert len(auto_complete_calls) == 0
    # Should have called inference
    mock_inference.assert_called_once()


def test_process_project_no_auto_complete_faces_at_min_confirmed():
    """process_project does NOT auto-complete when faces ≤5 but >= min_confirmed."""
    project = {
        "id": 102,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Exact_Min",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 3,
    }
    auto_complete_calls = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        # 3 faces = min_confirmed, so should NOT auto-complete even though ≤5
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 3}]
        if "UPDATE projects SET status = 'completed'" in sql_s:
            auto_complete_calls.append((sql_s, params))
            return 1
        if "SELECT * FROM projects WHERE id" in sql_s:
            return [project.copy()]
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", return_value=0),
        patch("worker.run_autonomous_inference") as mock_inference,
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    # faces >= min_confirmed → no auto-complete
    assert len(auto_complete_calls) == 0
    mock_inference.assert_called_once()


def test_process_project_runs_inference_before_discovery():
    """Faces reset by a re-run are classified before crawl/bootstrap.

    A re-run on an active project otherwise waits for the full category
    traversal and SPARQL bootstrap before the inference card can resolve.
    """
    project = {
        "id": 103,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Rerun_Category",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }
    events: list[str] = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        if "SELECT 1 AS found FROM faces f" in sql_s:
            return [{"found": 1}]
        if "SELECT * FROM projects WHERE id" in sql_s:
            return [project.copy()]
        if "COUNT(*) AS cnt FROM images WHERE project_id" in sql_s:
            return [{"cnt": 0}]
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 10}]
        if "SELECT min_confirmed FROM projects" in sql_s:
            return [{"min_confirmed": 5}]
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", side_effect=lambda _p: events.append("traverse")),
        patch("worker.bootstrap_from_sparql", side_effect=lambda _p: events.append("bootstrap")),
        patch("worker.process_images", return_value=0),
        patch("worker.prefilter_images", return_value=0),
        patch("worker.run_autonomous_inference", side_effect=lambda _p: events.append("inference") or 1),
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    assert events[0] == "inference"
    assert events.count("inference") == 2  # early pass + final catch-up pass


# ---------------------------------------------------------------------------
# _touch_heartbeat_file
# ---------------------------------------------------------------------------


def test_touch_heartbeat_file_creates_file(tmp_path):
    """_touch_heartbeat_file creates the heartbeat file with a timestamp."""
    with patch("worker.HEARTBEAT_FILE_DIR", str(tmp_path)), patch("worker._worker_id", "test-worker"):
        _touch_heartbeat_file()
    hb_file = tmp_path / ".wikivisage-worker-alive-test-worker"
    assert hb_file.exists()
    float(hb_file.read_text())


def test_touch_heartbeat_file_updates_mtime(tmp_path):
    """Calling _touch_heartbeat_file twice overwrites the file content."""
    with patch("worker.HEARTBEAT_FILE_DIR", str(tmp_path)), patch("worker._worker_id", "test-worker"):
        _touch_heartbeat_file()
        first_content = (tmp_path / ".wikivisage-worker-alive-test-worker").read_text()
        _touch_heartbeat_file()
        second_content = (tmp_path / ".wikivisage-worker-alive-test-worker").read_text()
    assert float(second_content) >= float(first_content)


def test_touch_heartbeat_file_swallows_oserror():
    """_touch_heartbeat_file silently handles OSError (e.g. read-only path)."""
    with patch("worker.HEARTBEAT_FILE_DIR", "/nonexistent/dir"), patch("worker._worker_id", "test-worker"):
        _touch_heartbeat_file()


def test_touch_heartbeat_file_uses_worker_id(tmp_path):
    """Each worker writes to a unique file based on _worker_id."""
    with patch("worker.HEARTBEAT_FILE_DIR", str(tmp_path)), patch("worker._worker_id", "ml-worker-1"):
        _touch_heartbeat_file()
    with patch("worker.HEARTBEAT_FILE_DIR", str(tmp_path)), patch("worker._worker_id", "ml-worker-2"):
        _touch_heartbeat_file()
    assert (tmp_path / ".wikivisage-worker-alive-ml-worker-1").exists()
    assert (tmp_path / ".wikivisage-worker-alive-ml-worker-2").exists()


# ---------------------------------------------------------------------------
# Tests for _api_request
# ---------------------------------------------------------------------------


def test_api_request_maxlag_503_retries():
    """_api_request should retry on 503 with Retry-After header (maxlag)."""
    from worker import _api_request

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.headers = {}
    ok_resp.raise_for_status = MagicMock()

    lag_resp = MagicMock()
    lag_resp.status_code = 503
    lag_resp.headers = {"Retry-After": "0.01"}
    lag_resp.close = MagicMock()

    call_count = [0]

    def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return lag_resp
        return ok_resp

    with (
        patch("worker._get_session") as mock_session,
        patch("worker.time.sleep"),
        patch("worker.shutdown_requested", False),
    ):
        mock_session.return_value.get = mock_get
        result = _api_request("https://commons.wikimedia.org/w/api.php")

    assert result is ok_resp
    assert call_count[0] == 2


def test_api_request_maxlag_200_json_retries():
    """_api_request should retry when 200 response contains maxlag error in JSON."""
    from worker import _api_request

    lag_resp = MagicMock()
    lag_resp.status_code = 200
    lag_resp.headers = {"Retry-After": "0.01"}
    lag_resp.json.return_value = {"error": {"code": "maxlag", "info": "lag"}}
    lag_resp.raise_for_status = MagicMock()
    lag_resp.close = MagicMock()

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.headers = {}
    ok_resp.raise_for_status = MagicMock()

    call_count = [0]

    def mock_get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return lag_resp
        return ok_resp

    with (
        patch("worker._get_session") as mock_session,
        patch("worker.time.sleep"),
        patch("worker.shutdown_requested", False),
    ):
        mock_session.return_value.get = mock_get
        result = _api_request("https://commons.wikimedia.org/w/api.php")

    assert result is ok_resp


def test_api_request_raises_after_exhausted_retries():
    """_api_request should raise Exception after 3 failed attempts."""
    import requests as req

    from worker import _api_request

    with (
        patch("worker._get_session") as mock_session,
        patch("worker.time.sleep"),
        patch("worker.shutdown_requested", False),
    ):
        mock_session.return_value.get.side_effect = req.exceptions.RequestException("timeout")
        with pytest.raises(Exception, match="Failed to execute API request after 3 attempts"):
            _api_request("https://commons.wikimedia.org/w/api.php")


def test_api_request_shutdown_raises_interrupted():
    """_api_request should raise InterruptedError when shutdown_requested is True."""
    from worker import _api_request

    with (
        patch("worker.shutdown_requested", True),
    ):
        with pytest.raises(InterruptedError, match="shutting down"):
            _api_request("https://commons.wikimedia.org/w/api.php")


def test_api_request_post_method():
    """_api_request should use session.post when method='post'."""
    from worker import _api_request

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.headers = {}
    ok_resp.raise_for_status = MagicMock()

    with (
        patch("worker._get_session") as mock_session,
        patch("worker.shutdown_requested", False),
    ):
        mock_session.return_value.post.return_value = ok_resp
        result = _api_request("https://commons.wikimedia.org/w/api.php", method="post", data={"key": "val"})

    assert result is ok_resp
    mock_session.return_value.post.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for _download_image
# ---------------------------------------------------------------------------


def test_download_image_untrusted_host_raises():
    from worker import _download_image

    with pytest.raises(ValueError, match="untrusted host"):
        _download_image("https://evil.example.com/image.jpg")


def test_download_image_non_https_raises():
    from worker import _download_image

    with pytest.raises(ValueError, match="untrusted host"):
        _download_image("ftp://upload.wikimedia.org/image.jpg")


def test_download_image_http_scheme_raises():
    from worker import _download_image

    with pytest.raises(ValueError, match="untrusted host"):
        _download_image("http://upload.wikimedia.org/image.jpg")


def test_download_image_content_length_too_large_raises():
    from worker import _download_image

    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100 MB
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()

    with patch("worker._get_session") as mock_session:
        mock_session.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="too large"):
            _download_image("https://upload.wikimedia.org/image.jpg", max_bytes=50 * 1024 * 1024)


def test_download_image_streaming_size_exceeded_raises():
    from worker import _download_image

    chunks = [b"x" * 1024 * 1024] * 60  # 60 MB in 1 MB chunks

    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()
    mock_resp.iter_content.return_value = iter(chunks)

    with patch("worker._get_session") as mock_session:
        mock_session.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="exceeded"):
            _download_image("https://upload.wikimedia.org/image.jpg", max_bytes=50 * 1024 * 1024)


def test_download_image_success():
    from worker import _download_image

    image_data = b"FAKE_IMAGE_DATA"
    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()
    mock_resp.iter_content.return_value = iter([image_data])

    with patch("worker._get_session") as mock_session:
        mock_session.return_value.get.return_value = mock_resp
        result = _download_image("https://upload.wikimedia.org/image.jpg")

    assert result == image_data


# ---------------------------------------------------------------------------
# Tests for _validate_image_dimensions
# ---------------------------------------------------------------------------


def test_validate_image_dimensions_too_large_raises():
    import io

    from PIL import Image

    from worker import _validate_image_dimensions

    img = Image.new("RGB", (15000, 8000))  # 120 megapixels > 100MP limit
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    with pytest.raises(ValueError, match="too large"):
        _validate_image_dimensions(buf.getvalue())


def test_validate_image_dimensions_valid_passes():
    import io

    from PIL import Image

    from worker import _validate_image_dimensions

    img = Image.new("RGB", (800, 600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    _validate_image_dimensions(buf.getvalue())  # Should not raise


# ---------------------------------------------------------------------------
# Tests for _get_csrf_token
# ---------------------------------------------------------------------------


def test_get_csrf_token_success():
    from worker import _get_csrf_token

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf+\\"}}}

    with patch("worker._api_request", return_value=mock_resp):
        token = _get_csrf_token("fake-access-token")

    assert token == "csrf+\\"


def test_get_csrf_token_propagates_exception():
    from worker import _get_csrf_token

    with patch("worker._api_request", side_effect=RuntimeError("network error")):
        with pytest.raises(RuntimeError):
            _get_csrf_token("fake-token")


# ---------------------------------------------------------------------------
# Tests for _refresh_worker_token
# ---------------------------------------------------------------------------


def test_refresh_worker_token_no_user():
    from worker import _refresh_worker_token

    with patch("worker.execute_query", return_value=[]):
        result = _refresh_worker_token(99)

    assert result is None


def test_refresh_worker_token_still_valid():
    """Token not yet expired — return it without refresh."""
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    future = datetime.now(UTC) + timedelta(hours=2)
    user = {
        "access_token": "plaintext-token",
        "refresh_token": "plaintext-refresh",
        "token_expires_at": future,
    }

    with (
        patch("worker.execute_query", return_value=[user]),
        patch("oauth_refresh.decrypt_token", side_effect=lambda t: t),
    ):
        result = _refresh_worker_token(1)

    assert result == "plaintext-token"


def test_refresh_worker_token_no_refresh_token():
    """Expired token but no refresh token — return None."""
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    past = datetime.now(UTC) - timedelta(hours=1)
    user = {
        "access_token": "plaintext-token",
        "refresh_token": "",
        "token_expires_at": past,
    }

    with (
        patch("worker.execute_query", return_value=[user]),
        patch("oauth_refresh.decrypt_token", side_effect=lambda t: t),
    ):
        result = _refresh_worker_token(1)

    assert result is None


def test_refresh_worker_token_api_failure():
    """Network error during refresh — return None."""
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    past = datetime.now(UTC) - timedelta(hours=1)
    user = {
        "access_token": "plaintext-token",
        "refresh_token": "plaintext-refresh",
        "token_expires_at": past,
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("API error")
    mock_resp.close = MagicMock()

    with (
        patch("worker.execute_query", return_value=[user]),
        patch("oauth_refresh.decrypt_token", side_effect=lambda t: t),
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.post.return_value = mock_resp
        result = _refresh_worker_token(1)

    assert result is None


def test_refresh_worker_token_missing_access_token_in_response():
    """Token refresh response missing access_token field — return None."""
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    past = datetime.now(UTC) - timedelta(hours=1)
    user = {
        "access_token": "plaintext-token",
        "refresh_token": "plaintext-refresh",
        "token_expires_at": past,
    }

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {}  # No access_token field
    mock_resp.close = MagicMock()

    with (
        patch("worker.execute_query", return_value=[user]),
        patch("oauth_refresh.decrypt_token", side_effect=lambda t: t),
        patch("oauth_refresh.encrypt_token", side_effect=lambda t: t),
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.post.return_value = mock_resp
        result = _refresh_worker_token(1)

    assert result is None


def test_refresh_worker_token_rowcount_zero_reads_fresh():
    """When rowcount==0 (another process refreshed), re-read token from DB.

    Round 2 (H2 single-flight refactor): the re-read path now goes through
    ``_read_decrypted_user_tokens``, which SELECTs all three credential
    columns. The mock returns a fresh row with all three so both the initial
    read AND the post-CAS re-read succeed.
    """
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    past = datetime.now(UTC) - timedelta(hours=1)
    future = datetime.now(UTC) + timedelta(hours=4)
    initial_row = {
        "access_token": "old-token",
        "refresh_token": "old-refresh",
        "token_expires_at": past,
    }
    fresh_row = {
        "access_token": "fresh-token",
        "refresh_token": "fresh-refresh",
        "token_expires_at": future,
    }

    select_calls = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT access_token, refresh_token, token_expires_at" in sql:
            select_calls[0] += 1
            # First SELECT = initial credential read; second = post-CAS re-read.
            return [initial_row] if select_calls[0] == 1 else [fresh_row]
        if "UPDATE users SET access_token" in sql:
            return 0  # rowcount == 0: another process refreshed
        return []

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"access_token": "new-token", "refresh_token": "new-refresh", "expires_in": 14400}
    mock_resp.close = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("oauth_refresh.decrypt_token", side_effect=lambda t: t),
        patch("oauth_refresh.encrypt_token", side_effect=lambda t: t),
        patch("worker._get_session") as mock_session,
        # Round 2 H2: refresh now acquires a single-flight lock. Force the
        # acquire path so we exercise the full UPDATE → rowcount==0 → re-read.
        patch("worker._get_redis_client", return_value=None),
    ):
        mock_session.return_value.post.return_value = mock_resp
        result = _refresh_worker_token(1)

    assert result == "fresh-token"


# ---------------------------------------------------------------------------
# Tests for traverse_category
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_strikes_and_fresh_first_ordering_against_the_real_database(db_pool, db_conn, seed_project):
    """Three strikes write an image off, and images that keep failing are tried last."""
    project_id = seed_project["id"]
    with db_conn.cursor() as cur:
        for page_id, attempts in ((1, 2), (2, 0), (3, 1), (4, 0)):
            cur.execute(
                "INSERT INTO images (project_id, commons_page_id, file_title, status, detect_attempts) "
                "VALUES (%s, %s, %s, 'pending', %s)",
                (project_id, page_id, f"File:{page_id}.jpg", attempts),
            )
        cur.execute("SELECT id, commons_page_id FROM images WHERE project_id = %s", (project_id,))
        ids = {row["commons_page_id"]: row["id"] for row in cur.fetchall()}

    # Fresh images first, the repeat offenders last (the old query had no ORDER BY at all).
    waves: list[list[str]] = []

    def record_wave(wave, _project_id):
        waves.append([row["file_title"] for row in wave])
        return 0, (0.0, 0.0, 0.0)

    with patch("worker._process_image_wave", side_effect=record_wave), patch("worker.shutdown_requested", False):
        process_images({"id": project_id})
    assert [title for wave in waves for title in wave] == ["File:2.jpg", "File:4.jpg", "File:3.jpg", "File:1.jpg"]

    # Image 1 already has two strikes: the third writes it off, so it counts as processed.
    assert _worker_module._record_detect_failure(ids[1], "File:1.jpg", "timeout") == 3
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, error_message, detect_attempts FROM images WHERE id = %s", (ids[1],))
        row = cur.fetchone()
    assert row["status"] == "error" and row["detect_attempts"] == 3
    assert "after 3 attempts" in row["error_message"] and "timeout" in row["error_message"]

    # Image 3 has one: it just gets its second strike and stays pending.
    assert _worker_module._record_detect_failure(ids[3], "File:3.jpg", "timeout") == 2
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM images WHERE id = %s", (ids[3],))
        assert cur.fetchone()["status"] == "pending"

    # An image that is no longer pending is never struck again, nor written off a second time.
    assert _worker_module._record_detect_failure(ids[1], "File:1.jpg", "timeout") == 0
    with db_conn.cursor() as cur:
        cur.execute("SELECT detect_attempts FROM images WHERE id = %s", (ids[1],))
        assert cur.fetchone()["detect_attempts"] == 3

    # Images whose request failed while the service could not be tested are flagged to go alone next
    # time. Only pending rows: image 1 is an error by now, and a flag is not a strike.
    _worker_module._flag_for_solo_retry([ids[2], ids[1]])
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, detect_isolate, detect_attempts FROM images WHERE project_id = %s", (project_id,))
        rows = {row["id"]: row for row in cur.fetchall()}
    assert rows[ids[2]]["detect_isolate"] == 1 and rows[ids[2]]["detect_attempts"] == 0
    assert rows[ids[1]]["detect_isolate"] == 0
    assert rows[ids[4]]["detect_isolate"] == 0

    # ...and the flag reaches the wave, which is what sends them alone.
    waves.clear()
    seen_flags: dict[str, int] = {}

    def record_flags(wave, _project_id):
        seen_flags.update({row["file_title"]: row["detect_isolate"] for row in wave})
        return 0, (0.0, 0.0, 0.0)

    with patch("worker._process_image_wave", side_effect=record_flags), patch("worker.shutdown_requested", False):
        process_images({"id": project_id})
    assert seen_flags["File:2.jpg"] == 1 and seen_flags["File:4.jpg"] == 0


def _two_page_category(insert_results, total_update=None):
    """Run traverse_category over a category that answers in two pages, logging every step."""
    events: list[str] = []
    pages = iter(
        [
            {
                "query": {"categorymembers": [{"ns": 6, "title": "File:A.jpg", "pageid": 1}]},
                "continue": {"cmcontinue": "next"},
            },
            {"query": {"categorymembers": [{"ns": 6, "title": "File:B.jpg", "pageid": 2}]}},
        ]
    )
    inserts = iter(insert_results)

    def execute(sql, params=None, fetch=True):
        if "COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "INSERT IGNORE INTO images" in sql:
            events.append("insert")
            return next(inserts)
        if "UPDATE projects SET images_total" in sql:
            events.append("total")
            if total_update is not None:
                return total_update()
            return 1
        return []

    def api(*_args, **_kwargs):
        events.append("api")
        response = MagicMock()
        response.json.return_value = next(pages)
        return response

    with (
        patch("worker.execute_query", side_effect=execute),
        patch("worker._api_request", side_effect=api),
        patch("worker.time.sleep"),
        patch("worker.shutdown_requested", False),
    ):
        added = _worker_module.traverse_category({"id": 1, "commons_category": "Cat"})
    return added, events


def test_traverse_category_publishes_the_image_total_after_every_page():
    """The page shows "waiting" until images_total is above 0: not after the whole tree has been read."""
    added, events = _two_page_category([1, 1])

    assert added == 2
    # the total is written right after page 1's rows, before page 2 is even requested
    assert events == ["api", "insert", "total", "api", "insert", "total", "total"]


def test_traverse_category_does_not_touch_the_total_for_a_page_with_nothing_new():
    added, events = _two_page_category([0, 0])

    assert added == 0
    assert events == ["api", "insert", "api", "insert", "total"]  # only the final, exact figure


def test_traverse_category_carries_on_when_publishing_the_total_fails():
    from database import DatabaseError

    def fail_once(calls=[]):  # noqa: B006 - deliberate per-test state
        calls.append(1)
        if len(calls) == 1:
            raise DatabaseError("lock wait timeout")
        return 1

    added, events = _two_page_category([1, 1], total_update=fail_once)

    assert added == 2  # page 2 was still read and stored
    assert events.count("insert") == 2


def test_traverse_category_already_at_limit():
    """When project already has MAX_IMAGES_PER_PROJECT images, skip traversal."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "TestCat"}

    with patch("worker.execute_query", return_value=[{"cnt": _worker_module.MAX_IMAGES_PER_PROJECT}]):
        result = traverse_category(project)

    assert result == 0


def test_traverse_category_basic_files():
    """Traverse a category with image files — inserts them and returns count."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "TestCat"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append(sql.strip())
        if "COUNT(*)" in sql:
            return [{"cnt": 0}]
        if "INSERT IGNORE INTO images" in sql:
            return 2  # 2 rows inserted
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "categorymembers": [
                {"ns": 6, "title": "File:Photo1.jpg", "pageid": 1001},
                {"ns": 6, "title": "File:Photo2.jpg", "pageid": 1002},
            ]
        }
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        result = traverse_category(project)

    assert result == 2


def test_traverse_category_skips_non_image_extensions():
    """Traverse a category — skip video/audio files."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "TestCat"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append(sql.strip())
        if "COUNT(*)" in sql:
            return [{"cnt": 0}]
        if "INSERT IGNORE INTO images" in sql:
            return 1
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "categorymembers": [
                {"ns": 6, "title": "File:Video.webm", "pageid": 2001},
                {"ns": 6, "title": "File:Audio.ogg", "pageid": 2002},
                {"ns": 6, "title": "File:Photo.jpg", "pageid": 2003},
            ]
        }
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        traverse_category(project)

    # Only one INSERT should happen (for the jpg), with one image
    inserts = [sql for sql in db_calls if "INSERT IGNORE INTO images" in sql]
    assert len(inserts) == 1


def test_traverse_category_handles_subcategories():
    """Traverse discovers subcategories and queues them."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "ParentCat"}

    call_count = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*)" in sql:
            return [{"cnt": 0}]
        if "INSERT IGNORE INTO images" in sql:
            return 1
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    def mock_api_request(url, params=None, **kwargs):
        resp = MagicMock()
        call_count[0] += 1
        assert params is not None
        cmtitle = params.get("cmtitle", "")
        if "ParentCat" in cmtitle:
            resp.json.return_value = {
                "query": {
                    "categorymembers": [
                        {"ns": 14, "title": "Category:SubCat"},
                    ]
                }
            }
        elif "SubCat" in cmtitle:
            resp.json.return_value = {
                "query": {
                    "categorymembers": [
                        {"ns": 6, "title": "File:Image.jpg", "pageid": 3001},
                    ]
                }
            }
        else:
            resp.json.return_value = {"query": {"categorymembers": []}}
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", side_effect=mock_api_request),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = traverse_category(project)

    assert result == 1
    assert call_count[0] >= 2  # At least parent + subcat


def test_traverse_category_api_exception_breaks_inner_loop():
    """When API call fails, traversal logs error and continues to next category."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "TestCat"}

    with (
        patch("worker.execute_query", return_value=[{"cnt": 0}]),
        patch("worker._api_request", side_effect=RuntimeError("API down")),
        patch("worker.shutdown_requested", False),
    ):
        result = traverse_category(project)

    assert result == 0


def test_traverse_category_respects_image_limit():
    """Traverse stops adding images once remaining capacity is reached."""
    from worker import traverse_category

    project = {"id": 1, "commons_category": "BigCat"}
    # Project already has MAX-2 images
    existing = _worker_module.MAX_IMAGES_PER_PROJECT - 2

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*)" in sql:
            return [{"cnt": existing}]
        if "INSERT IGNORE INTO images" in sql:
            # Pretend all inserted
            flat = params
            assert flat is not None
            return len([x for x in range(0, len(flat), 3)])
        if "UPDATE projects SET images_total" in sql:
            return 1
        db_calls.append(sql)
        return []

    # API returns more files than remaining capacity
    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {"categorymembers": [{"ns": 6, "title": f"File:Img{i}.jpg", "pageid": 5000 + i} for i in range(10)]}
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        traverse_category(project)
    # No assertion needed — just verify no exception and it terminates


# ---------------------------------------------------------------------------
# Tests for bootstrap_from_sparql
# ---------------------------------------------------------------------------


def test_bootstrap_new_images_inserted():
    """bootstrap_from_sparql inserts new images and returns flagged count."""

    project = {"id": 1, "wikidata_qid": "Q42", "commons_category": "Douglas Adams"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "SELECT id, status FROM images" in sql:
            return []  # Image does not exist
        if "INSERT IGNORE INTO images" in sql:
            return 1  # One row inserted
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "search": [
                {"pageid": 100, "title": "File:Adams.jpg"},
            ]
        }
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 1


def test_bootstrap_existing_image_gets_flagged():
    """bootstrap_from_sparql flags existing images and marks processed ones as sdc_written."""

    project = {"id": 1, "wikidata_qid": "Q42", "commons_category": "Douglas Adams"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 5}]
        if "SELECT id, status FROM images" in sql:
            return [{"id": 99, "status": "processed"}]
        if "UPDATE images SET bootstrapped = 1" in sql:
            return 1
        if "UPDATE faces SET sdc_written = 1" in sql:
            return 0  # No target faces to mark
        if "UPDATE faces" in sql and "classified_by = 'bootstrap'" in sql:
            return 0  # No single-face images
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "search": [
                {"pageid": 200, "title": "File:Existing.jpg"},
            ]
        }
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 1
    flag_calls = [(s, p) for s, p in db_calls if "UPDATE images SET bootstrapped = 1" in s]
    assert len(flag_calls) == 1


def test_bootstrap_skips_video_extensions():
    """bootstrap_from_sparql skips video/audio files."""

    project = {"id": 1, "wikidata_qid": "Q42", "commons_category": "TestCat"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append(sql.strip())
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        return []

    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "search": [
                {"pageid": 301, "title": "File:Video.webm"},
                {"pageid": 302, "title": "File:Audio.ogg"},
            ]
        }
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", return_value=api_resp),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 0


def test_bootstrap_pagination():
    """bootstrap_from_sparql follows pagination (sroffset) to fetch all pages."""

    project = {"id": 1, "wikidata_qid": "Q42", "commons_category": "TestCat"}

    api_call_count = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "SELECT id, status FROM images" in sql:
            return []
        if "INSERT IGNORE INTO images" in sql:
            return 1
        if "UPDATE projects SET images_total" in sql:
            return 1
        return []

    def mock_api_request(url, params=None, **kwargs):
        resp = MagicMock()
        api_call_count[0] += 1
        if api_call_count[0] == 1:
            resp.json.return_value = {
                "query": {"search": [{"pageid": 401, "title": "File:Page1.jpg"}]},
                "continue": {"sroffset": 1},
            }
        else:
            resp.json.return_value = {
                "query": {"search": [{"pageid": 402, "title": "File:Page2.jpg"}]},
            }
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._api_request", side_effect=mock_api_request),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 2
    assert api_call_count[0] == 2


def test_bootstrap_api_exception_returns_zero():
    """bootstrap_from_sparql returns 0 when API throws an exception."""

    project = {"id": 1, "wikidata_qid": "Q42", "commons_category": "TestCat"}

    with (
        patch("worker.execute_query", return_value=[{"cnt": 0}]),
        patch("worker._api_request", side_effect=RuntimeError("network error")),
        patch("worker.shutdown_requested", False),
    ):
        result = bootstrap_from_sparql(project)

    assert result == 0


# ---------------------------------------------------------------------------
# Tests for _process_image_wave error paths
# ---------------------------------------------------------------------------


def test_process_image_wave_download_failure_is_not_detected_or_persisted():
    """A failed download drops out of the wave: never sent for detection, never persisted.

    _download_for_detection marks the row 'error' itself, so the wave must not
    issue any further DB writes for it.
    """
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=None),
        patch("worker.face_client.detect_faces_batch") as mock_batch,
    ):
        processed, _timings = _process_image_wave([{"id": 55, "file_title": "File:Broken.jpg", "bootstrapped": 0}], 7)

    assert processed == 0
    assert db_calls == []
    mock_batch.assert_not_called()


def test_process_image_wave_no_faces_no_auto_classify():
    """When no faces are detected, no is_target update is made."""
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return None

    detection_result = _detection_result(width=800, height=600, faces=[])

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"fake"),
        patch("worker.face_client.detect_faces_batch", return_value={"56": detection_result}),
    ):
        processed, _timings = _process_image_wave([{"id": 56, "file_title": "File:Empty.jpg", "bootstrapped": 1}], 10)

    assert processed == 1
    auto_class = [s for s, _ in db_calls if "UPDATE faces SET is_target" in s]
    assert len(auto_class) == 0


def test_process_image_wave_mixed_success_and_failure():
    """One unusable image in a wave must not stop its siblings being persisted."""
    good = _detection_result(
        width=800,
        height=600,
        faces=[face_client.DetectedFace(10, 110, 110, 10, encoding=_make_embedding(9))],
    )
    batch = {
        "1": good,
        "2": face_client.FaceServiceRejected("corrupt image"),
        "3": good,
    }

    error_updates: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "status = 'error'" in sql:
            assert params is not None
            error_updates.append(params)
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", side_effect=lambda i, t: b"bytes"),
        patch("worker.face_client.detect_faces_batch", return_value=batch),
    ):
        processed, _timings = _process_image_wave(
            [
                {"id": 1, "file_title": "File:A.jpg", "bootstrapped": 0},
                {"id": 2, "file_title": "File:B.jpg", "bootstrapped": 0},
                {"id": 3, "file_title": "File:C.jpg", "bootstrapped": 0},
            ],
            5,
        )

    assert processed == 2
    # The rejected image is terminal, so exactly one row is marked 'error'.
    assert len(error_updates) == 1
    assert error_updates[0][1] == 2


def test_process_image_wave_unavailable_leaves_row_pending():
    """A service outage must NOT write the image off as a permanent error."""
    batch = {"1": face_client.FaceServiceUnavailable("connection refused")}

    db_calls: list[str] = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append(sql)
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection", return_value=b"bytes"),
        patch("worker.face_client.detect_faces_batch", return_value=batch),
        # an outage: the test detection cannot work either (the autouse default), so nobody is to blame
    ):
        processed, _timings = _process_image_wave([{"id": 1, "file_title": "File:A.jpg", "bootstrapped": 0}], 5)

    assert processed == 0
    assert not any("status = 'error'" in sql for sql in db_calls)
    assert not any("detect_attempts" in sql for sql in db_calls)  # and no strike is counted


# ---------------------------------------------------------------------------
# One image that breaks the face service must not hold its neighbours (and the project) hostage,
# and a face service that is merely broken must never get the images blamed
# ---------------------------------------------------------------------------

_REAL_SERVICE_DETECTS = _worker_module._service_detects  # the autouse fixture below replaces it


@pytest.fixture(autouse=True)
def _no_real_face_service(monkeypatch):
    """No test here may reach a real face service: by default it cannot detect, and nothing waits."""
    monkeypatch.setattr(_worker_module, "_service_detects", lambda: False)
    monkeypatch.setattr(_worker_module, "DETECT_RECOVERY_WAIT", 0)
    monkeypatch.setattr(_worker_module, "_service_down_until", 0.0)


class _FakeTime:
    """``worker.time`` on a clock that only moves when the code sleeps (or the test advances it)."""

    def __init__(self, sleeps=None):
        self.now = 1000.0
        self.sleeps = sleeps if sleeps is not None else []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __getattr__(self, name):
        return getattr(time, name)


def _wave(*flagged):
    return [
        {"id": i, "file_title": f"File:{i}.jpg", "bootstrapped": 0, "detect_isolate": int(i in flagged)}
        for i in (1, 2, 3)
    ]


def _good():
    return _detection_result(
        width=800, height=600, faces=[face_client.DetectedFace(10, 110, 110, 10, encoding=_make_embedding(9))]
    )


class FakeFaceService:
    """Answers detect requests; any request that contains a *poison* image fails as a whole.

    ``crash_polls`` makes the test detection fail that many times after a poisoned request,
    like a service that crashed on the image and is restarting.
    """

    def __init__(self, poison=(), crash_polls=0, detects=True):
        self.poison = {str(key) for key in poison}
        self.crash_polls = crash_polls
        self.down_polls = 0
        self.detects = detects  # False: readiness would answer, but nothing detects
        self.requests: list[tuple[list[str], bool]] = []
        self.canaries = 0

    def detect(self, items, retry=True):
        keys = [key for key, _data in items]
        self.requests.append((sorted(keys), retry))
        if self.poison & set(keys):
            self.down_polls = self.crash_polls
            return {key: face_client.FaceServiceUnavailable("timeout") for key in keys}
        return {key: _good() for key in keys}

    def canary(self):
        self.canaries += 1
        if not self.detects:
            return False
        if self.down_polls:
            self.down_polls -= 1
            return False
        return True


def _run_wave(service, rows, *, strikes=1, wait=0, strike_updates=1):
    """Run one wave against *service*, returning (processed, db statements, sleeps)."""
    db: list[tuple[str, Any]] = []
    sleeps: list[float] = []

    def execute(sql, params=None, fetch=True):
        db.append((sql, params))
        if "SET detect_attempts" in sql:
            return strike_updates
        if "SELECT detect_attempts" in sql:
            return [{"detect_attempts": strikes}]
        return None

    with (
        patch("worker.execute_query", side_effect=execute),
        patch("worker._download_for_detection", return_value=b"bytes"),
        patch("worker.face_client.detect_faces_batch", side_effect=service.detect),
        patch("worker._service_detects", side_effect=service.canary),
        patch("worker.DETECT_RECOVERY_WAIT", wait),
        patch("worker.time", _FakeTime(sleeps)),
    ):
        processed, _timings = _process_image_wave(rows, 5)
    return processed, db, sleeps


def _struck(db):
    return sorted(params[0] for sql, params in db if "SET detect_attempts = detect_attempts + 1" in sql)


def _flagged(db):
    return sorted(p for sql, params in db if "SET detect_isolate = 1" in sql for p in params)


def test_a_failed_request_is_retried_image_by_image_and_only_the_culprit_is_struck():
    service = FakeFaceService(poison=["2"])

    processed, db, _sleeps = _run_wave(service, _wave())

    assert processed == 2  # the two good neighbours were saved after all
    # (keys come back in download-completion order, so compare sorted)
    assert service.requests[0] == (["1", "2", "3"], True)  # the batch, with the client's normal retries
    # then alone, no retries; the one that failed is asked a second time before it is blamed
    assert sorted(service.requests[1:]) == [(["1"], False), (["2"], False), (["2"], False), (["3"], False)]
    assert _struck(db) == [2]
    # One strike is not the end: the image stays pending for the next cycle.
    assert not any("status = 'error'" in sql for sql, _params in db)
    assert _flagged(db) == []


def test_the_third_strike_writes_the_image_off_so_the_project_can_finish():
    processed, db, _sleeps = _run_wave(
        FakeFaceService(poison=["2"]), _wave(), strikes=_worker_module.MAX_DETECT_ATTEMPTS
    )

    assert processed == 2
    errors = [params for sql, params in db if "UPDATE images SET status = 'error'" in sql]
    assert len(errors) == 1
    message, image_id = errors[0]
    assert image_id == 2 and "after 3 attempts" in message and "timeout" in message


def test_nothing_is_struck_when_detection_is_broken_though_the_service_says_it_is_up():
    """Readiness answers, every detection fails: blaming images would write off every pending image."""
    service = FakeFaceService(poison=["1", "2", "3"], detects=False)

    processed, db, _sleeps = _run_wave(service, _wave())

    assert processed == 0
    assert len(service.requests) == 1  # the batch only: no image was sent alone to a service that cannot detect
    assert _struck(db) == []
    assert not any("status = 'error'" in sql for sql, _params in db)
    assert _flagged(db) == [1, 2, 3]  # they are sent alone once it works again, without a strike


def test_a_service_that_crashed_on_an_image_gets_time_to_restart_and_the_culprit_is_found():
    service = FakeFaceService(poison=["1"], crash_polls=2)

    processed, db, sleeps = _run_wave(service, _wave(), wait=60)

    assert processed == 2  # the innocent images were processed once the service was back
    assert _struck(db) == [1]  # and only the culprit got the strike
    assert len(sleeps) >= 2 and set(sleeps) == {3}  # it polled while the service restarted
    assert _flagged(db) == []


def test_images_are_flagged_not_struck_when_the_service_does_not_recover_in_time():
    service = FakeFaceService(poison=["1"], crash_polls=10**6)

    processed, db, sleeps = _run_wave(service, _wave(), wait=9)

    assert processed == 0
    assert len(service.requests) == 1  # nothing was sent alone: it never worked again
    assert sleeps == [3, 3, 3]  # polled at 0, 3, 6 and 9 s: the wait is a time limit
    assert _struck(db) == []
    assert _flagged(db) == [1, 2, 3]


def test_after_a_timeout_the_next_wave_does_not_wait_again():
    service = FakeFaceService(poison=["1", "2", "3"], detects=False)

    _run_wave(service, _wave(), wait=9)
    canaries_after_first_wave = service.canaries
    _processed, _db, sleeps = _run_wave(service, _wave(), wait=9)

    assert service.canaries == canaries_after_first_wave  # not even tested: it is known to be down
    assert sleeps == []


def test_flagged_images_are_sent_alone_from_the_start():
    service = FakeFaceService(poison=["2"])

    processed, db, _sleeps = _run_wave(service, _wave(2))

    assert processed == 2
    assert (["1", "3"], True) in service.requests  # the others still travel as one batch
    assert (["2"], False) in service.requests  # the flagged one never joins a batch
    assert all("2" not in keys for keys, _retry in service.requests if len(keys) > 1)
    assert _struck(db) == [2]


def test_a_flagged_image_that_works_alone_is_simply_processed():
    processed, db, _sleeps = _run_wave(FakeFaceService(), _wave(2))

    assert processed == 3
    assert _struck(db) == [] and _flagged(db) == []


def test_a_test_detection_must_work_before_every_single_image():
    """If the service dies on one image, nothing is attributed and nothing else is tried until it works."""
    service = FakeFaceService(poison=["1", "2", "3"], crash_polls=10**6)
    working = iter([True])  # the first test detection works, then the service is down for good

    def canary_once():
        service.canaries += 1
        return next(working, False)

    db: list[tuple[str, Any]] = []

    def execute(sql, params=None, fetch=True):
        db.append((sql, params))
        return (
            1
            if "SET detect_attempts" in sql
            else ([{"detect_attempts": 1}] if "SELECT detect_attempts" in sql else None)
        )

    with (
        patch("worker.execute_query", side_effect=execute),
        patch("worker._download_for_detection", return_value=b"bytes"),
        patch("worker.face_client.detect_faces_batch", side_effect=service.detect),
        patch("worker._service_detects", side_effect=canary_once),
    ):
        _process_image_wave(_wave(1, 2, 3), 5)

    assert len(service.requests) == 1  # only the first image went out
    # It failed right after a working test, but the service did not work again to confirm it:
    # something else may have hurt the service meanwhile, so nobody is struck.
    assert _struck(db) == []
    assert _flagged(db) == [1, 2, 3]  # all three wait, unstruck, for the service


def test_a_model_mismatch_is_the_services_problem_not_the_images():
    service = FakeFaceService()
    service.detect = lambda items, retry=True: {  # type: ignore[method-assign]
        key: face_client.FaceServiceModelMismatch("wrong model") for key, _data in items
    }

    processed, db, _sleeps = _run_wave(service, _wave())

    assert processed == 0
    assert _struck(db) == [] and _flagged(db) == []  # not even retried: no single image is to blame


def test_recording_a_strike_survives_a_database_error():
    with patch("worker.execute_query", side_effect=DatabaseError("down")):
        assert _worker_module._record_detect_failure(1, "File:A.jpg", "timeout") == 0


def test_an_image_that_is_no_longer_pending_is_not_struck_or_written_off():
    executed: list[str] = []

    def execute(sql, params=None, fetch=True):
        executed.append(sql)
        return 0  # the UPDATE matched nothing: processed or written off in the meantime

    with patch("worker.execute_query", side_effect=execute):
        assert _worker_module._record_detect_failure(1, "File:A.jpg", "timeout") == 0

    assert len(executed) == 1  # no read-back and, above all, no "error" for an image that was just processed


def test_flagging_images_survives_a_database_error_and_skips_nothing_to_do():
    with patch("worker.execute_query") as query:
        _worker_module._flag_for_solo_retry([])
    query.assert_not_called()
    with patch("worker.execute_query", side_effect=DatabaseError("down")):
        _worker_module._flag_for_solo_retry([1, 2])  # must not raise


# --- the test detection and the wait for a restarting service -------------------------------


def test_the_test_detection_is_a_real_detection_without_retries():
    with patch("worker.face_client.check_service_model", return_value=("model", 512)) as check:
        assert _REAL_SERVICE_DETECTS() is True
    check.assert_called_once_with(retry=False, read_timeout=_worker_module.SERVICE_PROBE_TIMEOUT)


@pytest.mark.parametrize(
    "error",
    [
        face_client.FaceServiceUnavailable("HTTP 500"),
        face_client.FaceServiceModelMismatch("wrong model"),
        face_client.FaceServiceRejected("bad probe"),
        RuntimeError("anything else"),
    ],
)
def test_the_test_detection_is_false_for_every_way_it_can_fail_and_never_raises(error):
    with patch("worker.face_client.check_service_model", side_effect=error):
        assert _REAL_SERVICE_DETECTS() is False


def test_waiting_for_the_service_returns_as_soon_as_it_detects_again():
    answers = iter([False, False, True])
    sleeps: list[float] = []
    with (
        patch("worker._service_detects", side_effect=lambda: next(answers)),
        patch("worker.DETECT_RECOVERY_WAIT", 60),
        patch("worker.time", _FakeTime(sleeps)),
    ):
        assert _worker_module._wait_for_service() is True

    assert sleeps == [3, 3]


def test_waiting_for_the_service_gives_up_after_the_wait_and_then_does_not_wait_again():
    polls: list[int] = []
    sleeps: list[float] = []
    with (
        patch("worker._service_detects", side_effect=lambda: polls.append(1) or False),
        patch("worker.DETECT_RECOVERY_WAIT", 9),
        patch("worker.time", _FakeTime(sleeps)),
    ):
        assert _worker_module._wait_for_service() is False
        assert (len(polls), len(sleeps)) == (4, 3)
        assert _worker_module._wait_for_service() is False  # known to be down: answered at once
        assert (len(polls), len(sleeps)) == (4, 3)


def test_the_wait_is_wall_clock_time_even_when_every_test_detection_is_slow():
    """A hung service makes each test detection take its whole timeout: the wait must still end on time."""
    clock = _FakeTime()

    def slow_probe():
        clock.now += 10  # a probe that runs into its timeout
        return False

    with (
        patch("worker._service_detects", side_effect=slow_probe),
        patch("worker.DETECT_RECOVERY_WAIT", 60),
        patch("worker.time", clock),
    ):
        assert _worker_module._wait_for_service() is False

    assert clock.now - 1000.0 < 60 + 10  # the limit plus at most one probe (it was ~16 minutes)


def test_without_a_wait_the_service_is_tested_once_and_never_slept_on():
    sleeps: list[float] = []
    with (
        patch("worker._service_detects", return_value=False) as check,
        patch("worker.time.sleep", side_effect=sleeps.append),
    ):
        assert _worker_module._wait_for_service() is False

    check.assert_called_once_with()
    assert sleeps == []


def test_waiting_for_the_service_stops_at_shutdown():
    with (
        patch("worker._service_detects", return_value=False) as check,
        patch("worker.DETECT_RECOVERY_WAIT", 60),
        patch("worker.time.sleep"),
        patch.object(_worker_module, "shutdown_requested", True),
    ):
        assert _worker_module._wait_for_service() is False

    check.assert_called_once_with()


# --- the loop keeps going while images leave 'pending', whatever the reason ---------------


def _loop(pending_counts, batch_results):
    counts = iter(pending_counts)
    results = iter(batch_results)
    calls = []

    def process(project):
        calls.append("batch")
        return next(results)

    with (
        patch("worker._pending_image_count", side_effect=lambda _id: next(counts)),
        patch("worker.process_images", side_effect=process),
        patch("worker._project_still_active", return_value=True),
    ):
        _seconds, total = _worker_module._process_pending_images({"id": 5})
    return len(calls), total


def test_the_image_loop_continues_when_a_batch_only_wrote_images_off_as_errors():
    # 40 pending -> a batch of failures written off (0 successes, 6 fewer pending) -> carry on
    batches, total = _loop([40, 34, 34, 34], [0, 0])

    assert batches == 2
    assert total == 0


def test_the_image_loop_stops_when_nothing_left_pending_and_nothing_succeeded():
    batches, _total = _loop([40, 40], [0])

    assert batches == 1


def test_the_image_loop_continues_after_successes_even_if_the_count_is_unknown():
    batches, total = _loop([None, None, None, None], [5, 0])

    assert batches == 2
    assert total == 5


def test_waiting_faces_are_classified_with_the_projects_current_settings():
    fresh = {"id": 5, "distance_threshold": 0.4}
    with (
        patch("worker._pending_image_count", return_value=0),
        patch("worker._has_inference_candidates", return_value=True),
        patch("worker.execute_query", return_value=[fresh]),
        patch("worker.run_autonomous_inference", return_value=7) as infer,
    ):
        assert _worker_module._classify_waiting_faces({"id": 5, "distance_threshold": 0.9}) == 7

    infer.assert_called_once_with(fresh)


def test_a_finished_inference_run_is_recorded_with_what_it_found():
    with patch("worker.execute_query") as query:
        _worker_module._record_inference_completion(5, classified=9, matches=4)

    sql, params = query.call_args.args[0], query.call_args.args[1]
    assert "last_inference_run = last_inference_run + 1" in sql  # a counter: runs cannot share a number
    assert "last_inference_at = NOW()" in sql and "last_inference_matches = %s" in sql
    assert params == (4, 5)


def test_a_run_that_classified_nothing_is_not_announced():
    with patch("worker.execute_query") as query:
        _worker_module._record_inference_completion(5, classified=0, matches=0)

    query.assert_not_called()


def test_recording_the_completion_never_fails_the_run(caplog):
    with patch("worker.execute_query", side_effect=DatabaseError("down")):
        _worker_module._record_inference_completion(5, classified=3, matches=1)  # must not raise

    assert "failed to record the inference completion" in caplog.text


def test_nothing_is_classified_without_candidates_or_during_shutdown():
    with (
        patch("worker._pending_image_count", return_value=0),
        patch("worker._has_inference_candidates", return_value=False),
        patch("worker.run_autonomous_inference") as infer,
    ):
        assert _worker_module._classify_waiting_faces({"id": 5}) is None
    with (
        patch.object(_worker_module, "shutdown_requested", True),
        patch("worker._pending_image_count", return_value=0),
        patch("worker._has_inference_candidates", return_value=True),
        patch("worker.run_autonomous_inference") as infer,
    ):
        assert _worker_module._classify_waiting_faces({"id": 5}) is None
    infer.assert_not_called()


# --- the model starts once every image has been processed, on every path ------------------


@pytest.mark.parametrize("pending", [40, 1, None])  # None: the database could not say, so do not guess
def test_inference_waits_while_images_are_pending(pending):
    with (
        patch("worker._pending_image_count", return_value=pending),
        patch("worker._has_inference_candidates", return_value=True) as candidates,
        patch("worker.run_autonomous_inference") as infer,
    ):
        assert _worker_module._classify_waiting_faces({"id": 5}) is None

    infer.assert_not_called()
    candidates.assert_not_called()  # not even asked: the images decide first


def test_early_inference_runs_for_a_rerun_on_a_fully_processed_project(caplog):
    """Its purpose: after "re-run" the user should not wait through the crawl. No pending images: it runs."""
    caplog.set_level("INFO", logger="worker")
    with (
        patch("worker._pending_image_count", return_value=0),
        patch("worker._has_inference_candidates", return_value=True),
        patch("worker.execute_query", return_value=[{"id": 5}]),
        patch("worker.run_autonomous_inference", return_value=3) as infer,
    ):
        _worker_module._run_early_inference({"id": 5})

    infer.assert_called_once()
    assert "early inference classified 3 face(s) before discovery" in caplog.text


def test_early_inference_skips_a_project_that_still_has_images_to_process(caplog):
    caplog.set_level("INFO", logger="worker")
    with patch("worker._pending_image_count", return_value=40), patch("worker.run_autonomous_inference") as infer:
        _worker_module._run_early_inference({"id": 5})

    infer.assert_not_called()
    assert "early inference classified" not in caplog.text


def test_final_inference_waits_when_the_image_loop_ended_with_images_left(caplog):
    caplog.set_level("INFO", logger="worker")
    with patch("worker._pending_image_count", return_value=3), patch("worker.run_autonomous_inference") as infer:
        _worker_module._run_final_inference({"id": 5})

    infer.assert_not_called()
    assert "images are still pending, inference waits for them" in caplog.text


def test_final_inference_runs_once_no_image_is_pending():
    fresh = {"id": 5, "distance_threshold": 0.4}
    with (
        patch("worker._pending_image_count", return_value=0),
        patch("worker._project_still_active", return_value=True),
        patch("worker.execute_query", return_value=[fresh]),
        patch("worker.run_autonomous_inference") as infer,
    ):
        _worker_module._run_final_inference({"id": 5})

    infer.assert_called_once_with(fresh)


def test_the_image_loop_never_runs_inference_between_batches():
    counts = iter([100, 50, 50, 0, 0, 0])
    results = iter([50, 50, 0])
    with (
        patch("worker._pending_image_count", side_effect=lambda _id: next(counts)),
        patch("worker.process_images", side_effect=lambda _p: next(results)),
        patch("worker._project_still_active", return_value=True),
        patch("worker.run_autonomous_inference") as infer,
        patch("worker._classify_waiting_faces") as classify,
    ):
        _worker_module._process_pending_images({"id": 5})

    infer.assert_not_called()
    classify.assert_not_called()


def test_the_inference_only_claim_leaves_a_project_that_is_still_being_processed_alone():
    """An idle worker must not pick up an active project with pending images just because faces are waiting."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    with patch("worker.execute_transaction", side_effect=lambda fn: fn(MagicMock(), cursor)):
        _claim_inference_projects(2)

    sql = " ".join(cursor.execute.call_args_list[0].args[0].split())
    assert (
        "AND (p.status = 'completed' OR NOT EXISTS ( SELECT 1 FROM images i WHERE i.project_id = p.id AND i.status = 'pending')) "
        in sql
    )


def test_pending_image_count_reads_the_projects_pending_images_and_tolerates_a_database_error():
    with patch("worker.execute_query", return_value=[{"cnt": 40}]) as query:
        assert _worker_module._pending_image_count(5) == 40
    assert "status = 'pending'" in query.call_args.args[0]
    with patch("worker.execute_query", side_effect=DatabaseError("down")):
        assert _worker_module._pending_image_count(5) is None


def test_process_images_takes_fresh_images_before_the_ones_that_keep_failing():
    selects: list[str] = []

    def execute(sql, params=None, fetch=True):
        if "SUM(CASE WHEN bootstrapped" in sql:
            return [{"bs_done": 0, "total": 3}]
        if "COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "status = 'pending'" in sql and "LIMIT" in sql:
            selects.append(sql)
        return []

    with patch("worker.execute_query", side_effect=execute):
        process_images({"id": 99})

    assert len(selects) == 2  # bootstrap images, then the rest
    assert all("ORDER BY detect_attempts ASC, id ASC" in sql for sql in selects)


def test_process_images_splits_batch_into_memory_bounded_waves():
    """process_images must not hold a whole BATCH_SIZE of image bytes at once.

    DETECTION_WAVE_SIZE is what bounds worker peak memory, so a pending set
    larger than one wave has to be processed across several waves.
    """
    wave_size = _worker_module.DETECTION_WAVE_SIZE
    pending = [{"id": i, "file_title": f"File:{i}.jpg", "bootstrapped": 0} for i in range(wave_size * 2 + 1)]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SUM(CASE WHEN bootstrapped" in sql:
            return [{"bs_done": 0, "total": len(pending)}]
        if "COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "status = 'pending' AND bootstrapped = 1" in sql:
            return []
        if "status = 'pending' AND bootstrapped = 0" in sql:
            return pending
        return None

    wave_sizes: list[int] = []

    def fake_wave(wave, project_id):
        wave_sizes.append(len(wave))
        return len(wave), (0.0, 0.0, 0.0)

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._process_image_wave", side_effect=fake_wave),
    ):
        processed = process_images({"id": 99})

    assert processed == len(pending)
    assert len(wave_sizes) == 3
    assert wave_sizes == [wave_size, wave_size, 1]
    assert max(wave_sizes) <= wave_size


# ---------------------------------------------------------------------------
# Tests for process_images skip logic
# ---------------------------------------------------------------------------


def test_process_images_no_pending_bootstrap_at_cap_skips_leftover():
    """When bootstrap_remaining==0 and no non-bootstrap pending, leftover bootstrap images are skipped."""
    project = {"id": 99}

    skipped_bootstrap_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        # bs_ratio = 400/1000 = 0.4 <= 0.5, so bootstrap_cap = base_cap = 900
        # bs_already_processed = 900 >= 900, so bootstrap_remaining = 0
        if "SUM(CASE WHEN bootstrapped = 1 AND status != 'pending'" in sql:
            return [{"bs_done": 900, "total": 1000}]
        if "COUNT(*) AS cnt FROM images" in sql and "bootstrapped = 1" in sql:
            return [{"cnt": 400}]  # bs_ratio = 0.4 → base_cap = 900
        if "bootstrapped = 0" in sql and "LIMIT" in sql:
            return []
        if "bootstrapped = 1" in sql and "LIMIT" in sql:
            return []
        if "UPDATE images SET status = 'skipped'" in sql:
            skipped_bootstrap_calls.append(sql)
            return 5  # 5 leftover bootstrap images skipped
        if "UPDATE projects SET images_processed" in sql:
            return 1
        return ()

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_for_detection"),
        patch("worker.face_client.detect_faces_batch"),
        patch("worker.shutdown_requested", False),
    ):
        count = process_images(project)

    assert count == 0
    assert len(skipped_bootstrap_calls) == 1


# ---------------------------------------------------------------------------
# Tests for write_sdc_claims early exits
# ---------------------------------------------------------------------------


def test_write_sdc_no_access_token_sets_error():
    """write_sdc_claims aborts and sets error when access token cannot be obtained."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value=None),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    error_updates = [(s, p) for s, p in db_calls if "sdc_write_error" in s and "UPDATE projects" in s]
    assert len(error_updates) == 1
    assert error_updates[0][1][0] == SDC_AUTH_EXPIRED_MESSAGE


def test_write_sdc_csrf_failure_sets_error():
    """write_sdc_claims aborts and sets error when CSRF token fetch fails."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", side_effect=RuntimeError("csrf error")),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    error_updates = [(s, p) for s, p in db_calls if "sdc_write_error" in s and "UPDATE projects" in s]
    assert len(error_updates) == 1
    assert "CSRF" in error_updates[0][1][0]


def test_write_sdc_invalid_qid_sets_error():
    """write_sdc_claims aborts with error when project QID is malformed."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "INVALID"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    error_updates = [(s, p) for s, p in db_calls if "UPDATE projects" in s and "Invalid QID" in str(p)]
    assert len(error_updates) == 1


def test_write_sdc_user_cancelled_returns_early():
    """write_sdc_claims returns when sdc_write_requested is set to 0 (user cancelled)."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 0}]  # User cancelled
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0


def test_write_sdc_successful_write_increments_counter():
    """write_sdc_claims writes P180 claim and returns total_written=1."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}

    db_calls = []
    face_batch_returned = [False]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not face_batch_returned[0]:
                face_batch_returned[0] = True
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "sdc_removal_pending" in sql and "DISTINCT" in sql:
            return []
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 1  # Successfully claimed
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    api_call_count = [0]

    def mock_api(url, params=None, data=None, headers=None, method="get", **kwargs):
        resp = MagicMock()
        api_call_count[0] += 1
        if api_call_count[0] == 1:
            resp.json.return_value = {"claims": {}}  # No existing claim
        else:
            resp.json.return_value = {"success": 1}  # Write succeeds
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1

    face_sdc_updates = [(s, p) for s, p in db_calls if "UPDATE faces SET sdc_written" in s]
    assert len(face_sdc_updates) == 1


@pytest.mark.parametrize("phase", ["write", "removal"])
@pytest.mark.parametrize(
    "remaining,expected_error",
    [
        (False, None),
        (True, "Worker shutdown left Commons edits pending."),
        ("db_error", "Worker shut down before completion could be verified."),
        ("missing_result", "Worker shut down before completion could be verified."),
    ],
)
def test_sdc_shutdown_after_success_checks_remaining_work(phase, remaining, expected_error, monkeypatch):
    """SIGTERM after the last API success is not itself evidence of an incomplete run."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    calls = []
    batch_returned = False
    actions = []

    def query(sql, params=None, fetch=True):
        nonlocal batch_returned
        calls.append((sql.strip(), params))
        if ") AS pending" in sql:
            if remaining == "db_error":
                raise _worker_module.DatabaseError("DB unavailable")
            if remaining == "missing_result":
                return ()
            return [{"pending": int(remaining)}]
        if "SELECT sdc_write_requested" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if phase == "write" and not batch_returned:
                batch_returned = True
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT DISTINCT i.commons_page_id" in sql:
            if phase == "removal" and not batch_returned:
                batch_returned = True
                return [{"commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "SELECT 1 FROM faces f" in sql:
            return [{"pending": 1}]
        return 1

    def api(url, params=None, data=None, **kwargs):
        action = (params or data)["action"]
        actions.append(action)
        response = MagicMock()
        if action == "wbgetclaims":
            claims = []
            if phase == "removal":
                claims = [{"id": "M9999$test", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}}]
            response.json.return_value = {"claims": {"P180": claims}}
        else:
            response.json.return_value = {"success": 1}
        return response

    def stop_after_edit(_seconds):
        _worker_module.signal_handler(_worker_module.signal.SIGTERM, None)

    monkeypatch.setattr(_worker_module, "execute_query", query)
    monkeypatch.setattr(_worker_module, "_refresh_worker_token", lambda _user_id: "fake-token")
    monkeypatch.setattr(_worker_module, "_get_csrf_token", lambda _token: "fake-csrf")
    monkeypatch.setattr(_worker_module, "_api_request", api)
    monkeypatch.setattr(_worker_module, "_keepalive_sdc_write", lambda _project_id: None)
    monkeypatch.setattr(_worker_module, "shutdown_requested", False)
    monkeypatch.setattr(_worker_module.time, "sleep", stop_after_edit)

    assert write_sdc_claims(project) == (1 if phase == "write" else 0)
    assert actions == ["wbgetclaims", "wbeditentity" if phase == "write" else "wbremoveclaims"]
    assert _worker_module.shutdown_requested
    finish = [(sql, params) for sql, params in calls if sql.startswith("UPDATE projects SET sdc_write_requested = 0")]
    assert len(finish) == 1
    error, project_id = finish[0][1]
    assert project_id == 5
    if expected_error is None:
        assert error is None
    else:
        assert error.startswith(expected_error)
    # Local bookkeeping was committed before the shutdown/completion check.
    if phase == "write":
        assert any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in calls)
        assert any("UPDATE sdc_claims SET written_at" in sql for sql, _ in calls)
    else:
        assert any("SET f.sdc_removal_pending = 0, f.sdc_written = 0" in sql for sql, _ in calls)
        assert any("DELETE FROM sdc_claims" in sql for sql, _ in calls)


@pytest.mark.integration
@pytest.mark.parametrize(
    "state,expected",
    [
        ("settled", False),
        ("write", True),
        ("bootstrap_image", False),
        ("bootstrap_face", False),
        ("unclassified", False),
        ("superseded", False),
        ("removal", True),
        ("guarded_removal", False),
        ("superseded_sibling", True),
    ],
)
def test_pending_sdc_work_matches_actionable_queue(
    db_pool, db_conn, seed_project, seed_images, seed_faces, state, expected
):
    target = seed_faces["target"][0]
    sibling = seed_faces["non_target"][0]
    image_id = seed_images[0]["id"]
    with db_conn.cursor() as cur:
        cur.execute("UPDATE faces SET sdc_written = 1")
        if state in {"write", "bootstrap_image", "bootstrap_face", "superseded", "unclassified"}:
            cur.execute("UPDATE faces SET sdc_written = 0 WHERE id = %s", (target,))
        if state == "bootstrap_image":
            cur.execute("UPDATE images SET bootstrapped = 1 WHERE id = %s", (image_id,))
        elif state == "bootstrap_face":
            cur.execute("UPDATE faces SET classified_by = 'bootstrap' WHERE id = %s", (target,))
        elif state == "unclassified":
            cur.execute("UPDATE faces SET is_target = NULL, classified_by = NULL WHERE id = %s", (target,))
        elif state == "superseded":
            cur.execute("UPDATE faces SET superseded_by = %s WHERE id = %s", (sibling, target))
        elif state in {"removal", "guarded_removal", "superseded_sibling"}:
            cur.execute("UPDATE faces SET is_target = 0, sdc_removal_pending = 1 WHERE id = %s", (target,))
            if state != "removal":
                cur.execute("UPDATE faces SET is_target = 1 WHERE id = %s", (sibling,))
            if state == "superseded_sibling":
                cur.execute("UPDATE faces SET superseded_by = %s WHERE id = %s", (target, sibling))

    assert _worker_module._has_pending_sdc_work(seed_project["id"]) is expected
    assert _worker_module._has_pending_sdc_work(seed_project["id"] + 1000) is False


@pytest.mark.parametrize("reset_is_target", [None, 0])
def test_write_sdc_queues_removal_when_face_changes_during_write(reset_is_target):
    """A face reset or undone between the live-state re-check and the write must
    not strand the P180 claim the worker just added.

    Classify undo restores ``is_target=NULL``; review undo restores 0. Both must
    hit the recovery UPDATE, including the stop-then-undo interleaving where the
    undo sees no in-flight write request and therefore queues nothing itself.
    """
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []
    face_batch_returned = [False]
    recovered = [False]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not face_batch_returned[0]:
                face_batch_returned[0] = True
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            return [{"is_target": 1, "superseded_by": None}]
        if "sdc_removal_pending" in sql and "DISTINCT" in sql:
            return []
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 1
        if "UPDATE faces SET sdc_written = 1 WHERE id = %s AND is_target = 1" in sql:
            return 0  # changed after the live-state re-check
        if "UPDATE faces SET sdc_written = 1, sdc_removal_pending = 1" in sql:
            if reset_is_target is None:
                recovered[0] = "is_target IS NULL" in sql
            else:
                recovered[0] = "is_target = 0" in sql
            return int(recovered[0])
        if "UPDATE faces SET sdc_written" in sql:
            return 1
        if "UPDATE sdc_claims SET written_at" in sql:
            return 1
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            return 1
        return 0

    api_call_count = [0]

    def mock_api(url, params=None, data=None, headers=None, method="get", **kwargs):
        resp = MagicMock()
        api_call_count[0] += 1
        if api_call_count[0] == 1:
            resp.json.return_value = {"claims": {}}  # no existing claim
        else:
            resp.json.return_value = {"success": 1}
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 1
    assert recovered[0]
    queued = [sql for sql, _ in db_calls if "UPDATE faces" in sql and "sdc_removal_pending = 1" in sql]
    assert queued
    assert "is_target IS NULL" in queued[0]
    assert "is_target = 0" in queued[0]


@pytest.mark.integration
def test_write_sdc_recovers_claim_when_stop_and_review_undo_race_edit(
    db_pool, db_conn, seed_project, seed_images, seed_faces, monkeypatch
):
    """Stopping first makes undo skip its own queue; the worker must recover.

    Reproduces the real interleaving: the user stops the run (so the undo route
    sees no in-flight write and clears both SDC flags), then undoes a reviewed
    match while wbeditentity is still in flight. The post-write recovery must
    recognize is_target=0, not only NULL, or the new P180 claim is stranded.
    """
    target = seed_faces["target"][0]
    image_id = seed_images[0]["id"]
    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET sdc_write_requested = 1 WHERE id = %s", (seed_project["id"],))
        cur.execute("UPDATE faces SET sdc_written = 1")
        cur.execute(
            "UPDATE faces SET sdc_written = 0, sdc_removal_pending = 0 WHERE id = %s",
            (target,),
        )
        cur.execute("UPDATE images SET bootstrapped = 0 WHERE id = %s", (image_id,))

    actions = []

    def api(_url, params=None, data=None, **_kwargs):
        action = (params or data)["action"]
        actions.append(action)
        response = MagicMock()
        if action == "wbgetclaims":
            response.json.return_value = {"claims": {}}
            return response
        assert action == "wbeditentity"
        with db_conn.cursor() as cur:
            cur.execute("UPDATE projects SET sdc_write_requested = 0 WHERE id = %s", (seed_project["id"],))
            cur.execute(
                "UPDATE faces SET is_target = 0, classified_by = 'model', "
                "classified_by_user_id = NULL, sdc_written = 0, "
                "sdc_removal_pending = 0, classified_at = NULL WHERE id = %s",
                (target,),
            )
        response.json.return_value = {"success": 1}
        return response

    monkeypatch.setattr(_worker_module, "_refresh_worker_token", lambda _user_id: "fake-token")
    monkeypatch.setattr(_worker_module, "_get_csrf_token", lambda _token: "fake-csrf")
    monkeypatch.setattr(_worker_module, "_api_request", api)
    monkeypatch.setattr(_worker_module, "_keepalive_sdc_write", lambda _project_id: None)
    monkeypatch.setattr(_worker_module, "shutdown_requested", False)
    monkeypatch.setattr(_worker_module.time, "sleep", lambda _seconds: None)

    assert write_sdc_claims(seed_project) == 1
    assert actions == ["wbgetclaims", "wbeditentity"]
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, sdc_written, sdc_removal_pending FROM faces WHERE id = %s",
            (target,),
        )
        face = cur.fetchone()
    assert face == {"is_target": 0, "sdc_written": 1, "sdc_removal_pending": 1}


def _mock_write_batch_query(face_rows, state_rows):
    """Mock execute_query for one write batch: *face_rows* once, then empty.

    ``state_rows`` is either a list returned for the per-face re-check or an
    exception instance raised by it.
    """
    db_calls = []
    batch_returned = [False]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not batch_returned[0]:
                batch_returned[0] = True
                return list(face_rows)
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            if isinstance(state_rows, Exception):
                raise state_rows
            return list(state_rows)
        if "SELECT DISTINCT i.commons_page_id" in sql:
            return []
        return 0

    return mock_execute_query, db_calls


def test_write_sdc_skips_face_rejected_before_write():
    """A face rejected while its batch was queued is not written to Commons."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    mock_eq, db_calls = _mock_write_batch_query(
        [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}],
        [{"is_target": 0, "superseded_by": None}],
    )
    api_mock = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", api_mock),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    api_mock.assert_not_called()
    assert not any("INSERT IGNORE INTO sdc_claims" in sql for sql, _ in db_calls)
    assert not any("UPDATE faces SET sdc_written" in sql for sql, _ in db_calls)


def test_write_sdc_skips_face_superseded_before_write():
    """A face replaced by a bbox edit is not written to Commons."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    mock_eq, db_calls = _mock_write_batch_query(
        [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}],
        [{"is_target": 1, "superseded_by": 101}],
    )
    api_mock = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", api_mock),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    api_mock.assert_not_called()
    assert not any("INSERT IGNORE INTO sdc_claims" in sql for sql, _ in db_calls)


def test_write_sdc_skips_face_that_vanished_before_write():
    """A face row deleted (manual draw undone) between batch and write is skipped."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    mock_eq, db_calls = _mock_write_batch_query(
        [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}],
        [],
    )
    api_mock = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", api_mock),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    api_mock.assert_not_called()
    assert not any("INSERT IGNORE INTO sdc_claims" in sql for sql, _ in db_calls)


def test_write_sdc_rechecks_live_state_after_idempotency_check():
    """A face undone while the idempotency request is in flight is not written.

    The first live-state check runs before wbgetclaims; the second runs right
    before wbeditentity, so an undo that deletes a manual face (cascading its
    dedup row away) during that network call cannot strand an orphaned P180 claim.
    """
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    db_calls = []
    batch_returned = [False]
    state_calls = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        if "sdc_write_requested" in sql and "SELECT" in sql:
            return [{"sdc_write_requested": 1}]
        if "SELECT f.id as face_id" in sql:
            if not batch_returned[0]:
                batch_returned[0] = True
                return [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}]
            return []
        if "SELECT is_target, superseded_by FROM faces" in sql:
            state_calls[0] += 1
            if state_calls[0] == 1:
                return [{"is_target": 1, "superseded_by": None}]
            return []  # the face row vanished while wbgetclaims was in flight
        if "INSERT IGNORE INTO sdc_claims" in sql:
            return 1
        if "DELETE FROM sdc_claims" in sql:
            return 1
        if "SELECT DISTINCT i.commons_page_id" in sql:
            return []
        return 0

    api_calls = []

    def mock_api(url, params=None, data=None, headers=None, method="get", **kwargs):
        api_calls.append((params or {}).get("action") or (data or {}).get("action"))
        resp = MagicMock()
        resp.json.return_value = {"claims": {}}  # no P180 claim on Commons yet
        return resp

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", side_effect=mock_api),
        patch("worker.shutdown_requested", False),
        patch("worker.time.sleep"),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    # Only the idempotency request went out — wbeditentity was never reached.
    assert api_calls == ["wbgetclaims"]
    # The unwritten reservation is released so a later send for a re-confirmed
    # face starts clean.
    releases = [(sql, p) for sql, p in db_calls if "DELETE FROM sdc_claims" in sql]
    assert len(releases) == 1
    assert releases[0][1] == (100,)
    assert not any("UPDATE faces SET sdc_written" in sql for sql, _ in db_calls)


def test_write_sdc_skips_face_when_state_recheck_fails():
    """An unreadable per-face re-check skips the face and ends the run through the
    error path, recording why the faces stayed unwritten."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    mock_eq, db_calls = _mock_write_batch_query(
        [{"face_id": 100, "image_id": 200, "commons_page_id": 9999}],
        _worker_module.DatabaseError("db down"),
    )
    api_mock = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._api_request", api_mock),
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    api_mock.assert_not_called()
    assert not any("INSERT IGNORE INTO sdc_claims" in sql for sql, _ in db_calls)

    # The run must not fall through to the silent cleanup: the request flag is
    # cleared together with an error message so the UI can explain the failure
    # and the user can retry.
    error_updates = [
        (sql, p)
        for sql, p in db_calls
        if "sdc_write_error = %s" in sql and "UPDATE projects" in sql and p and p[0] is not None
    ]
    assert len(error_updates) == 1
    assert "Could not verify the faces" in error_updates[0][1][0]
    # … and the error survives: the normal cleanup (which nulls the message)
    # never runs because the guard returns directly.
    flag_updates = [(sql, p) for sql, p in db_calls if "UPDATE projects SET sdc_write_requested = 0" in sql]
    assert flag_updates[-1][1][0] is not None


def test_keepalive_sdc_write_reports_progress_and_refreshes_the_claim():
    """The SDC-run keepalive tells the heartbeat thread it is alive and keeps the project claim."""
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._touch_heartbeat_file") as touch,
        patch("worker._mark_progress") as mark,
    ):
        _worker_module._keepalive_sdc_write(7)

    mark.assert_called_once_with("sdc_keepalive")
    touch.assert_not_called()  # the file is the heartbeat thread's job now
    assert not any("worker_heartbeat" in sql for sql, _ in db_calls)
    claim_updates = [
        (sql, p) for sql, p in db_calls if "worker_claimed_at = NOW()" in sql and "worker_claimed_by" in sql
    ]
    assert len(claim_updates) == 1
    assert claim_updates[0][1][0] == 7


def test_write_sdc_keepalive_runs_every_five_faces():
    """A long write run reports progress and refreshes its claim every 5 faces."""
    project = {"id": 5, "user_id": 1, "wikidata_qid": "Q42"}
    faces = [{"face_id": 100 + i, "image_id": 200, "commons_page_id": 9999} for i in range(5)]
    mock_eq, _ = _mock_write_batch_query(faces, [{"is_target": 0, "superseded_by": None}])

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker._refresh_worker_token", return_value="fake-token"),
        patch("worker._get_csrf_token", return_value="fake-csrf"),
        patch("worker._keepalive_sdc_write") as keepalive,
        patch("worker.shutdown_requested", False),
    ):
        result = write_sdc_claims(project)

    assert result == 0
    keepalive.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# Tests for run_autonomous_inference edge cases
# ---------------------------------------------------------------------------


def test_inference_skips_invalid_confirmed_encodings():
    """Confirmed faces with invalid encoding lengths are skipped."""
    project = {"id": 20, "min_confirmed": 2, "distance_threshold": 0.6}

    confirmed_rows = [
        {"encoding": b"short"},  # Invalid: not 2048 bytes
        {"encoding": _make_embedding(1)},  # Valid
        {"encoding": _make_embedding(2)},  # Valid
    ]
    unclassified_rows = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 2}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding" in sql:
            return unclassified_rows
        if "UPDATE projects SET last_inference" in sql:
            return 1
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.shutdown_requested", False),
    ):
        result = run_autonomous_inference(project)

    # Invalid encoding is skipped — but 2 valid ones remain, so inference runs
    assert result == 0  # No candidates to classify


def test_inference_all_confirmed_encodings_invalid_returns_zero():
    """When all confirmed encodings are invalid, inference returns 0."""
    project = {"id": 21, "min_confirmed": 1, "distance_threshold": 0.6}

    confirmed_rows = [
        {"encoding": b"bad"},
        {"encoding": None},
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 2}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.shutdown_requested", False),
    ):
        result = run_autonomous_inference(project)

    assert result == 0


def test_inference_skips_invalid_candidate_encodings(inference_transaction):
    """Candidate faces with invalid encoding lengths are skipped without crashing."""
    project = {"id": 22, "min_confirmed": 2, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_embedding(i)} for i in range(3)]
    unclassified_rows = [
        {"id": 500, "image_id": 1, "encoding": b"bad"},  # Invalid — should be skipped
        {"id": 501, "image_id": 2, "encoding": _make_embedding(500)},  # Valid
    ]
    update_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 3}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id," in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            update_calls.append(params)
            return 1
        if "UPDATE projects SET last_inference" in sql:
            return 1
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._face_distance", return_value=0.3),
        patch("worker.shutdown_requested", False),
    ):
        result = run_autonomous_inference(project)

    # Only the valid candidate (id=501) should be classified
    assert result == 1


def test_handle_detection_failure_rejected_marks_image_error():
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql, params, fetch))
        return 1

    with patch("worker.execute_query", side_effect=mock_execute_query):
        _handle_detection_failure(77, "File:Rejected.jpg", face_client.FaceServiceRejected("bad image"))

    assert len(db_calls) == 1
    sql, params, fetch = db_calls[0]
    assert "UPDATE images SET status = 'error'" in sql
    assert params == ("face service rejected image: bad image", 77)
    assert fetch is False


def test_handle_detection_failure_unavailable_leaves_pending():
    with patch("worker.execute_query") as mock_execute_query:
        _handle_detection_failure(78, "File:Retry.jpg", face_client.FaceServiceUnavailable("timeout"))

    mock_execute_query.assert_not_called()


# --- Security: _worker_id sanitization ---


def test_worker_id_rejects_path_traversal():
    with patch("worker.sys") as mock_sys, patch("worker.init_db"), patch("worker.signal.signal"):
        mock_sys.argv = ["worker.py", "--worker-id", "../../etc/passwd"]
        with pytest.raises(SystemExit, match="Invalid worker ID"):
            _worker_module.main()


def test_worker_id_rejects_slashes():
    with patch("worker.sys") as mock_sys, patch("worker.init_db"), patch("worker.signal.signal"):
        mock_sys.argv = ["worker.py", "--worker-id", "foo/bar"]
        with pytest.raises(SystemExit, match="Invalid worker ID"):
            _worker_module.main()


def test_worker_id_accepts_valid_ids():
    import re

    for wid in ("ml-worker-1", "ml-worker-2", "test_worker.3", "worker123"):
        assert re.fullmatch(r"[a-zA-Z0-9._-]+", wid), f"Expected valid: {wid}"


# --- Security: _api_request allow_redirects=False ---


def test_api_request_disables_redirects():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.json.return_value = {}
    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        _worker_module._api_request("https://commons.wikimedia.org/w/api.php", params={"action": "query"})
    _, kwargs = mock_session.get.call_args
    assert kwargs.get("allow_redirects") is False


def test_api_request_post_disables_redirects():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.json.return_value = {}
    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        _worker_module._api_request("https://commons.wikimedia.org/w/api.php", method="post", data={"action": "edit"})
    _, kwargs = mock_session.post.call_args
    assert kwargs.get("allow_redirects") is False


# --- Security: _download_image redirect rejection ---


def test_worker_download_image_rejects_redirect_to_untrusted():
    mock_resp = MagicMock()
    mock_resp.is_redirect = True
    mock_resp.headers = {"Location": "https://evil.example.com/redirected.jpg"}
    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        with pytest.raises(ValueError, match="Redirect to untrusted host"):
            _worker_module._download_image("https://upload.wikimedia.org/file.jpg")


def test_worker_download_image_rejects_redirect_to_http_scheme():
    mock_resp = MagicMock()
    mock_resp.is_redirect = True
    mock_resp.headers = {"Location": "http://upload.wikimedia.org/redirected.jpg"}
    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        with pytest.raises(ValueError, match="Redirect to untrusted host"):
            _worker_module._download_image("https://upload.wikimedia.org/file.jpg")


def test_worker_download_image_multi_hop_redirect_succeeds():
    hop1 = MagicMock()
    hop1.is_redirect = True
    hop1.headers = {"Location": "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/X.jpg"}
    hop1.close = MagicMock()

    hop2 = MagicMock()
    hop2.is_redirect = True
    hop2.headers = {"Location": "https://upload.wikimedia.org/wikipedia/commons/a/X.jpg"}
    hop2.close = MagicMock()

    content_resp = MagicMock()
    content_resp.is_redirect = False
    content_resp.headers = {"Content-Length": "4"}
    content_resp.raise_for_status = MagicMock()
    content_resp.iter_content.return_value = iter([b"data"])
    content_resp.close = MagicMock()

    mock_session = MagicMock()
    mock_session.get.side_effect = [hop1, hop2, content_resp]
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        result = _worker_module._download_image("https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg")
    assert result == b"data"
    hop1.close.assert_called_once()
    hop2.close.assert_called_once()


def test_worker_download_image_too_many_redirects_raises():
    mock_resp = MagicMock()
    mock_resp.is_redirect = True
    mock_resp.headers = {"Location": "https://upload.wikimedia.org/loop"}
    mock_resp.close = MagicMock()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    with patch.object(_worker_module, "_get_session", return_value=mock_session):
        with pytest.raises(ValueError, match="Too many redirects"):
            _worker_module._download_image("https://upload.wikimedia.org/file.jpg")


# ---------------------------------------------------------------------------
# Security: face embedding validation (anti-poisoning)
# ---------------------------------------------------------------------------


def _unit(seed: int, dim: int = DIM) -> np.ndarray:
    return np.frombuffer(_make_embedding(seed, dim), dtype="<f4").astype(np.float32)


def test_load_embedding_accepts_unit_512d_float32():
    vec = _unit(1)
    result = _worker_module._load_embedding(vec.astype("<f4").tobytes(), DIM, face_id=1)
    assert result is not None
    assert result.shape == (DIM,)
    assert result.dtype == np.float32
    assert np.allclose(result, vec)
    # Must be writable (np.frombuffer returns read-only views; we copy)
    assert result.flags.writeable


def test_load_embedding_is_little_endian_on_the_wire():
    vec = _unit(2)
    assert np.allclose(_worker_module._load_embedding(vec.astype("<f4").tobytes(), DIM, face_id=1), vec)
    # The same numbers written big-endian must not be silently accepted as this vector.
    swapped = _worker_module._load_embedding(vec.astype(">f4").tobytes(), DIM, face_id=1)
    assert swapped is None or not np.allclose(swapped, vec)


def test_load_embedding_rejects_none():
    assert _worker_module._load_embedding(None, DIM, face_id="x") is None


def test_load_embedding_rejects_wrong_length_including_legacy_dlib_encodings():
    assert _worker_module._load_embedding(b"\x00" * 1024, DIM, face_id=2) is None  # old 128 x float64
    assert _worker_module._load_embedding(b"\x00" * 2047, DIM, face_id=3) is None
    assert _worker_module._load_embedding(b"\x00" * 4096, DIM, face_id=4) is None
    assert _worker_module._load_embedding(b"", DIM, face_id=5) is None


def test_load_embedding_rejects_nan_and_inf():
    vec = _unit(3)
    vec[42] = np.nan
    assert _worker_module._load_embedding(vec.tobytes(), DIM, face_id=6) is None
    vec[42] = np.inf
    assert _worker_module._load_embedding(vec.tobytes(), DIM, face_id=7) is None
    vec[42] = -np.inf
    assert _worker_module._load_embedding(vec.tobytes(), DIM, face_id=8) is None


@pytest.mark.parametrize("factor", [0.0, 0.5, 0.89, 1.11, 3.0, 100.0])
def test_load_embedding_rejects_norm_outside_band(factor):
    vec = _unit(4) * np.float32(factor)
    assert _worker_module._load_embedding(vec.astype("<f4").tobytes(), DIM, face_id=9) is None


@pytest.mark.parametrize("factor", [0.91, 0.95, 1.0, 1.05, 1.09])
def test_load_embedding_renormalises_values_inside_the_band(factor):
    vec = _unit(5) * np.float32(factor)
    result = _worker_module._load_embedding(vec.astype("<f4").tobytes(), DIM, face_id=10)
    assert result is not None
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)


def test_load_embedding_honours_a_different_dim():
    vec = _unit(6, 64)
    assert _worker_module._load_embedding(vec.tobytes(), 64, face_id=11) is not None
    assert _worker_module._load_embedding(vec.tobytes(), DIM, face_id=12) is None


# ---------------------------------------------------------------------------
# Cosine distance, centroid and threshold gating (real math, no patched distance)
# ---------------------------------------------------------------------------


def test_face_distance_is_cosine_distance_of_unit_vectors():
    a, b = _unit(10), _unit(11)
    assert _worker_module._face_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert _worker_module._face_distance(a, b) == pytest.approx(1.0 - float(a @ b), abs=1e-6)
    assert _worker_module._face_distance(a, -a) == pytest.approx(2.0, abs=1e-6)


def test_face_distance_never_negative():
    a = _unit(12)
    assert _worker_module._face_distance(a, a * np.float32(1.0000001)) >= 0.0


def test_face_distance_orders_close_before_far():
    base = _unit(20)
    near = base + 0.05 * _unit(21)
    near /= np.linalg.norm(near)
    far = _unit(22)
    assert _worker_module._face_distance(base, near) < _worker_module._face_distance(base, far)


def _run_inference_with(project, confirmed, candidates):
    """Run inference with real distance math; return {face_id: (is_target, confidence)}."""
    updates: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": len(confirmed)}]
        if "SELECT f.id, f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return [{"id": i, "encoding": blob} for i, blob in enumerate(confirmed)]
        if "SELECT f.id, f.image_id," in sql:
            return [{"id": fid, "image_id": img, "encoding": blob} for fid, img, blob in candidates]
        if "UPDATE faces SET" in sql:
            updates.append(params)
            return len(candidates)
        return 1

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.execute_transaction", side_effect=_fake_inference_transaction),
        patch("worker.shutdown_requested", False),
    ):
        run_autonomous_inference(project)

    result = {}
    for params in updates:
        is_target_map, confidence_map, _ = _extract_update_maps(params)
        for fid in is_target_map:
            result[fid] = (is_target_map[fid], confidence_map[fid])
    return result


def test_inference_uses_renormalised_mean_centroid_and_cosine_threshold():
    """Same-person vectors (small perturbations of one direction) match; unrelated vectors do not."""
    person = _unit(30)
    confirmed = []
    for i in range(5):
        v = person + 0.15 * _unit(40 + i)
        confirmed.append((v / np.linalg.norm(v)).astype("<f4").tobytes())
    same = person + 0.15 * _unit(60)
    same /= np.linalg.norm(same)
    stranger = _unit(70)

    project = {"id": 31, "min_confirmed": 5, "distance_threshold": 0.5}
    result = _run_inference_with(
        project,
        confirmed,
        [(901, 1, same.astype("<f4").tobytes()), (902, 2, stranger.astype("<f4").tobytes())],
    )

    assert result[901][0] == 1
    assert result[901][1] < 0.5
    assert result[902][0] == 0
    assert result[902][1] > 0.5


def test_inference_threshold_gates_on_strictly_less_than():
    """A candidate exactly at the threshold is not a match; one just inside is."""
    person = _unit(80)
    confirmed = [person.astype("<f4").tobytes()] * 5

    def at_distance(d: float) -> np.ndarray:
        # unit vector at cosine distance d from `person`
        other = _unit(81)
        other -= person * float(other @ person)
        other /= np.linalg.norm(other)
        cos = 1.0 - d
        return (cos * person + np.sqrt(1 - cos**2) * other).astype(np.float32)

    inside, outside = at_distance(0.45), at_distance(0.55)
    result = _run_inference_with(
        {"id": 32, "min_confirmed": 5, "distance_threshold": 0.5},
        confirmed,
        [(911, 1, inside.tobytes()), (912, 2, outside.tobytes())],
    )

    assert result[911][0] == 1
    assert result[911][1] == pytest.approx(0.45, abs=1e-4)
    assert result[912][0] == 0
    assert result[912][1] == pytest.approx(0.55, abs=1e-4)


def test_inference_keeps_only_the_closest_face_per_image():
    person = _unit(90)
    confirmed = [person.astype("<f4").tobytes()] * 5
    close = (person + 0.05 * _unit(91)) / np.linalg.norm(person + 0.05 * _unit(91))
    closer = (person + 0.02 * _unit(92)) / np.linalg.norm(person + 0.02 * _unit(92))

    result = _run_inference_with(
        {"id": 33, "min_confirmed": 5, "distance_threshold": 0.5},
        confirmed,
        [(921, 1, close.astype("<f4").tobytes()), (922, 1, closer.astype("<f4").tobytes())],
    )

    assert result[922][0] == 1
    assert result[921][0] == 0


def test_inference_skips_when_confirmed_embeddings_cancel_out():
    person = _unit(100)
    confirmed = [person.astype("<f4").tobytes(), (-person).astype("<f4").tobytes()] * 3

    result = _run_inference_with(
        {"id": 34, "min_confirmed": 5, "distance_threshold": 0.5},
        confirmed,
        [(931, 1, person.astype("<f4").tobytes())],
    )

    assert result == {}


def test_inference_only_reads_embeddings_of_the_expected_model():
    seen = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "embedding_model = %s" in sql:
            seen.append(params)
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.shutdown_requested", False),
    ):
        run_autonomous_inference({"id": 35, "min_confirmed": 5, "distance_threshold": 0.5})

    assert seen, "confirmed-faces query must filter on embedding_model"
    assert all(p[1] == face_client.EXPECTED_MODEL_ID for p in seen)


def test_inference_default_threshold_is_the_cosine_default():
    assert _worker_module.DEFAULT_DISTANCE_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Mixed detection spaces
# ---------------------------------------------------------------------------


def test_convert_box_is_identity_for_same_space():
    assert _worker_module._convert_box_to_stored_space((10, 110, 90, 20), (500, 400), (500, 400)) == (10, 110, 90, 20)


def test_convert_box_scales_a_960_box_into_a_stored_500_space():
    # width 960 -> 500, height 720 -> 375: factors 0.5208 / 0.5208
    top, right, bottom, left = _worker_module._convert_box_to_stored_space((96, 480, 288, 192), (960, 720), (500, 375))
    assert (top, right, bottom, left) == (50, 250, 150, 100)


def test_convert_box_scales_up_and_handles_non_uniform_scale():
    assert _worker_module._convert_box_to_stored_space((10, 20, 30, 5), (100, 50), (200, 150)) == (30, 40, 90, 10)


def test_convert_box_clips_to_stored_space_and_keeps_positive_extent():
    top, right, bottom, left = _worker_module._convert_box_to_stored_space((0, 10_000, 10_000, 0), (100, 100), (50, 50))
    assert (top, right, bottom, left) == (0, 50, 50, 0)
    top, right, bottom, left = _worker_module._convert_box_to_stored_space((10, 10, 10, 10), (100, 100), (50, 50))
    assert right > left and bottom > top


def test_convert_box_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        _worker_module._convert_box_to_stored_space((1, 2, 3, 4), (0, 100), (50, 50))
    with pytest.raises(ValueError):
        _worker_module._convert_box_to_stored_space((1, 2, 3, 4), (100, 100), (50, 0))


def test_persist_detection_converts_boxes_into_a_stored_space():
    calls = []
    result = _detection_result(
        960, 720, [face_client.DetectedFace(96, 480, 288, 192, encoding=_make_embedding(1), score=0.9)]
    )

    with patch(
        "worker.execute_query", side_effect=lambda sql, params=None, fetch=True: calls.append((sql, params)) or 1
    ):
        assert _worker_module._persist_detection(7, result, stored_size=(500, 375)) is True

    insert = next(c for c in calls if c[0].startswith("INSERT INTO faces"))
    assert insert[1][4:] == (50, 250, 150, 100)
    update = next(c for c in calls if c[0].startswith("UPDATE images SET status = 'processed'"))
    # The stored detection space is kept, not overwritten by the new image size.
    assert update[1] == (1, 500, 375, 7)


def test_persist_detection_without_stored_space_keeps_boxes_and_records_size():
    calls = []
    result = _detection_result(
        960, 720, [face_client.DetectedFace(96, 480, 288, 192, encoding=_make_embedding(1), score=0.9)]
    )

    with patch(
        "worker.execute_query", side_effect=lambda sql, params=None, fetch=True: calls.append((sql, params)) or 1
    ):
        assert _worker_module._persist_detection(7, result) is True

    insert = next(c for c in calls if c[0].startswith("INSERT INTO faces"))
    assert insert[1][4:] == (96, 480, 288, 192)
    update = next(c for c in calls if c[0].startswith("UPDATE images SET status = 'processed'"))
    assert update[1] == (1, 960, 720, 7)


def test_persist_detection_stores_embedding_model_and_detector_score():
    calls = []
    faces = [
        face_client.DetectedFace(1, 50, 50, 1, encoding=_make_embedding(1), score=0.93),
        face_client.DetectedFace(60, 90, 90, 60, encoding=_make_embedding(2), score=None),
    ]

    with patch(
        "worker.execute_query", side_effect=lambda sql, params=None, fetch=True: calls.append((sql, params)) or 1
    ):
        _worker_module._persist_detection(8, _detection_result(100, 100, faces))

    sql, params = next(c for c in calls if c[0].startswith("INSERT INTO faces"))
    assert "embedding_model" in sql and "det_score" in sql
    assert sql.count("(%s, %s, %s, %s, %s, %s, %s, %s)") == 2
    # image_id, encoding, embedding_model, det_score, then the box
    assert params[:4] == (8, _make_embedding(1), MODEL_ID, 0.93)
    assert params[8:12] == (8, _make_embedding(2), MODEL_ID, None)
    assert len(params) == 16


def test_persist_detection_bootstrap_auto_classify_clears_queued_removal():
    """Auto-classifying the single face of a bootstrapped image drops any queued
    P180 removal — the pre-existing claim on Commons must stay."""
    calls = []
    faces = [face_client.DetectedFace(1, 50, 50, 1, encoding=_make_embedding(3), score=0.9)]

    with patch(
        "worker.execute_query", side_effect=lambda sql, params=None, fetch=True: calls.append((sql, params)) or 1
    ):
        assert _worker_module._persist_detection(8, _detection_result(100, 100, faces), bootstrapped=True, project_id=1)

    auto = next(sql for sql, _ in calls if "classified_by = 'bootstrap'" in sql)
    assert "sdc_removal_pending = 0" in auto


def test_detection_url_uses_the_960_thumbnail():
    assert _worker_module.FILE_PATH_URL.endswith("?width=960")
    assert _worker_module.FILE_PATH_URL.format(file_title="X.jpg").endswith("/X.jpg?width=960")


def test_detection_width_must_be_a_standard_commons_step(monkeypatch):
    import importlib

    import config

    monkeypatch.setenv("WIKIVISAGE_DETECTION_WIDTH", "1024")
    try:
        with pytest.raises(ValueError, match="thumbnail step"):
            importlib.reload(config)
        monkeypatch.setenv("WIKIVISAGE_DETECTION_WIDTH", "1280")
        assert importlib.reload(config).DETECTION_WIDTH == 1280
    finally:
        monkeypatch.delenv("WIKIVISAGE_DETECTION_WIDTH", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("reset"),
        requests.Timeout("slow"),
        requests.HTTPError(response=MagicMock(status_code=503)),
        requests.HTTPError(response=MagicMock(status_code=429)),
    ],
)
def test_download_for_detection_leaves_transient_failures_pending(exc):
    """'error' counts as processed, so marking transient failures would allow false auto-completion."""
    with (
        patch.object(_worker_module, "_download_image", side_effect=exc),
        patch.object(_worker_module, "_mark_image_error") as mark,
    ):
        with pytest.raises(_worker_module.TransientDownloadError) as caught:
            _worker_module._download_for_detection(1, "File:X.jpg")
    assert caught.value.rate_limited is (getattr(exc.response, "status_code", None) == 429)
    mark.assert_not_called()


def test_download_for_detection_marks_permanent_http_errors():
    exc = requests.HTTPError(response=MagicMock(status_code=404))
    with (
        patch.object(_worker_module, "_download_image", side_effect=exc),
        patch.object(_worker_module, "_mark_image_error") as mark,
    ):
        assert _worker_module._download_for_detection(1, "File:X.jpg") is None
    mark.assert_called_once()


# ---------------------------------------------------------------------------
# Startup check of the face service's embedding model
# ---------------------------------------------------------------------------


def test_startup_model_check_accepts_the_expected_model(caplog):
    with (
        patch("worker.face_client.check_service_model", return_value=(MODEL_ID, DIM)),
        caplog.at_level("INFO", logger="worker"),
    ):
        _worker_module._check_face_service_model()

    assert any(MODEL_ID in rec.getMessage() for rec in caplog.records)


def test_startup_model_check_exits_on_a_different_model():
    err = face_client.FaceServiceModelMismatch("face service model_id is 'other', expected 'scrfd500m+arcface_mbf'")
    with patch("worker.face_client.check_service_model", side_effect=err), pytest.raises(SystemExit) as exc:
        _worker_module._check_face_service_model()

    assert exc.value.code == 1


@pytest.mark.parametrize(
    "err",
    [face_client.FaceServiceUnavailable("connection refused"), face_client.FaceServiceRejected("probe image refused")],
)
def test_startup_model_check_only_warns_when_the_service_is_not_answering(err, caplog):
    with (
        patch("worker.face_client.check_service_model", side_effect=err),
        caplog.at_level("WARNING", logger="worker"),
    ):
        _worker_module._check_face_service_model()  # must not raise or exit

    assert any("Could not verify" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Experimental LiftWing metadata pre-filter (Phase 2, shadow mode)
# ---------------------------------------------------------------------------


def test_prefilter_disabled_mode_makes_no_calls():
    with (
        patch("worker.execute_query") as mock_db,
        patch("worker.liftwing_client.get_client") as get_client,
    ):
        assert prefilter_images({"id": 1, "prefilter_mode": "off"}) == 0

    mock_db.assert_not_called()
    get_client.assert_not_called()


def test_prefilter_defaults_to_off_when_column_missing():
    with (
        patch("worker.execute_query") as mock_db,
        patch("worker.liftwing_client.get_client") as get_client,
    ):
        assert prefilter_images({"id": 1}) == 0

    mock_db.assert_not_called()
    get_client.assert_not_called()


def test_prefilter_skips_when_service_marked_down():
    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "down"}]
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client") as get_client,
    ):
        assert prefilter_images({"id": 1, "prefilter_mode": "shadow"}) == 0

    get_client.assert_not_called()


def test_prefilter_records_verdicts_and_keeps_missing_pages():
    project = {"id": 7, "prefilter_mode": "shadow"}
    rows = [
        {"id": 101, "commons_page_id": 501, "file_title": "File:A.jpg"},
        {"id": 102, "commons_page_id": 502, "file_title": "File:B.svg"},
        {"id": 103, "commons_page_id": 503, "file_title": "File:C.ogg"},
    ]
    stored: list[tuple] = []
    select_sql: list[str] = []
    up_updates: list[tuple] = []
    claim_refreshes: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "up"}]
        if "SELECT id, commons_page_id, file_title FROM images" in sql:
            select_sql.append(sql)
            assert "bootstrapped = 0" in sql
            assert "prefilter_checked_at IS NULL" in sql
            # Shadow verdicts are metadata-only: already-processed images are
            # eligible too, so a backlog is worked off over later cycles (H1).
            assert "status = 'pending'" not in sql
            return rows
        if "SET status = 'up'" in sql:
            up_updates.append((sql, params))
            return 1
        if "UPDATE projects SET worker_claimed_at = NOW()" in sql:
            claim_refreshes.append(params)
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()
    fake_client.filter_metadata.return_value = {
        501: liftwing_client.Verdict(page_id=501, keep=True, reason="portrait"),
        502: liftwing_client.Verdict(page_id=502, keep=False, reason="diagram"),
        # 503 is missing: the model omitted it, so it must be kept with a NULL reason
    }
    metadata = {
        "File:A.jpg": {"description": "portrait", "categories": ("Category:X",), "mime": "image/jpeg"},
        "File:B.svg": {"description": "", "categories": (), "mime": "image/svg+xml"},
        "File:C.ogg": {"description": "", "categories": (), "mime": "audio/ogg"},
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._fetch_metadata", return_value=metadata),
        patch("worker._store_prefilter_verdicts", side_effect=lambda updates: stored.extend(updates)),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker-1"),
    ):
        checked = prefilter_images(project)

    assert checked == 3
    assert set(stored) == {(1, "portrait", 101), (0, "diagram", 102), (1, None, 103)}
    assert len(select_sql) == 1
    # A stored batch is a real success; service_status is updated accordingly (H3).
    assert len(up_updates) == 1
    # The claim is refreshed after every stored batch, so a slow pass cannot
    # expire while the sweep thread runs and the main loop is idle.
    assert claim_refreshes == [(7, "test-worker-1")]

    items = fake_client.filter_metadata.call_args.args[0]
    assert [item.page_id for item in items] == [501, 502, 503]
    assert items[0].description == "portrait"
    assert items[0].categories == ("Category:X",)
    assert items[2].mime == "audio/ogg"


def test_prefilter_stops_when_claim_refresh_fails():
    """A failed keepalive stops the pass — no processing on without a claim."""
    from database import DatabaseError

    rows = [
        {"id": 301, "commons_page_id": 701, "file_title": "File:A.jpg"},
        {"id": 302, "commons_page_id": 702, "file_title": "File:B.jpg"},
    ]
    stored: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "up"}]
        if "SELECT id, commons_page_id, file_title FROM images" in sql:
            return rows
        if "UPDATE projects SET worker_claimed_at = NOW()" in sql:
            raise DatabaseError("claim refresh failed")
        if "SET status = 'up'" in sql:
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()
    fake_client.filter_metadata.return_value = {
        701: liftwing_client.Verdict(page_id=701, keep=True, reason="portrait"),
    }

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker._fetch_metadata",
            return_value={
                "File:A.jpg": {"description": "", "categories": (), "mime": "image/jpeg"},
                "File:B.jpg": {"description": "", "categories": (), "mime": "image/jpeg"},
            },
        ),
        patch("worker._store_prefilter_verdicts", side_effect=lambda updates: stored.extend(updates)),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker-1"),
        patch("worker.LIFTWING_BATCH_SIZE", 1),
    ):
        assert prefilter_images({"id": 11, "prefilter_mode": "shadow"}) == 1

    # The first batch landed, then the refresh failed and the pass stopped:
    # File:B.jpg is left for a later pass instead of being processed without
    # a live claim.
    assert stored == [(1, "portrait", 301)]
    assert fake_client.filter_metadata.call_count == 1


def test_prefilter_fail_open_when_client_returns_none():
    rows = [{"id": 201, "commons_page_id": 601, "file_title": "File:A.jpg"}]
    stored: list[tuple] = []
    failure_updates: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "up"}]
        if "SELECT id, commons_page_id, file_title FROM images" in sql:
            return rows
        if sql.startswith("SELECT status, consecutive_failures"):
            return [{"status": "up", "consecutive_failures": 0}]
        if sql.startswith("UPDATE service_status SET status = %s"):
            failure_updates.append(params)
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()
    fake_client.filter_metadata.return_value = None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker._fetch_metadata",
            return_value={"File:A.jpg": {"description": "", "categories": (), "mime": ""}},
        ),
        patch("worker._store_prefilter_verdicts", side_effect=lambda updates: stored.extend(updates)),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
        patch("worker.shutdown_requested", False),
    ):
        checked = prefilter_images({"id": 8, "prefilter_mode": "shadow"})

    assert checked == 0
    assert stored == []
    # H3: a real failed batch is a recorded failure, not just a log line, so the
    # banner cannot claim "up" while completions fail.
    assert failure_updates == [("up", 1, "pre-filter request failed (see worker log)")]


def test_prefilter_leaves_rows_unchecked_when_metadata_lookup_fails():
    rows = [{"id": 202, "commons_page_id": 602, "file_title": "File:A.jpg"}]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "up"}]
        if "SELECT id, commons_page_id, file_title FROM images" in sql:
            return rows
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._fetch_metadata", return_value={}),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
        patch("worker.shutdown_requested", False),
    ):
        checked = prefilter_images({"id": 9, "prefilter_mode": "shadow"})

    assert checked == 0
    fake_client.filter_metadata.assert_not_called()


def test_prefilter_no_unchecked_rows_returns_zero():
    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT status FROM service_status" in sql:
            return [{"status": "up"}]
        if "SELECT id, commons_page_id, file_title FROM images" in sql:
            return []
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client") as get_client,
        patch("worker.shutdown_requested", False),
    ):
        assert prefilter_images({"id": 10, "prefilter_mode": "shadow"}) == 0

    get_client.assert_not_called()


def test_process_project_runs_prefilter_after_image_processing():
    """H1: the shadow pre-filter is metadata-only, so detection comes first.

    It used to run before the image loop with ``status='pending'`` in its
    selection, which both delayed downloads and left every processed image
    permanently unchecked (so a backlog could never be evaluated).
    """
    project = {
        "id": 98,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Some_Category",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
        "prefilter_mode": "shadow",
    }
    order: list[str] = []

    def mock_eq(sql, params=None, fetch=True):
        sql_s = sql if isinstance(sql, str) else str(sql)
        if "SELECT status, worker_claimed_by FROM projects" in sql_s:
            return [{"status": "active", "worker_claimed_by": "test-worker"}]
        if "UPDATE projects SET worker_claimed_at" in sql_s:
            return 1
        if "COUNT(*) AS cnt FROM faces" in sql_s:
            return [{"cnt": 0}]
        if "UPDATE projects SET status = 'completed'" in sql_s:
            return 1
        return ()

    def fake_process_images(_project):
        order.append("images")
        return 0  # a single empty batch ends the loop

    def fake_prefilter(_project):
        order.append("prefilter")
        return 0

    with (
        patch("worker.execute_query", side_effect=mock_eq),
        patch("worker.traverse_category", return_value=0),
        patch("worker.bootstrap_from_sparql", return_value=0),
        patch("worker.process_images", side_effect=fake_process_images),
        patch("worker.prefilter_images", side_effect=fake_prefilter),
        patch("worker.run_autonomous_inference"),
        patch("worker.shutdown_requested", False),
        patch("worker._worker_id", "test-worker"),
    ):
        process_project(project)

    assert order == ["images", "prefilter"]


def test_claim_shadow_project_claims_and_refetches():
    """The sweep claim targets non-off projects with unchecked crawled images."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        {"id": 42},
        {"id": 42, "status": "completed", "prefilter_mode": "shadow"},
    ]

    def run_txn(operations):
        return operations(MagicMock(), cursor)

    with (
        patch("worker.execute_transaction", side_effect=run_txn),
        patch("worker._worker_id", "test-worker-1"),
    ):
        project = _claim_shadow_project()

    assert project["id"] == 42
    select_sql = cursor.execute.call_args_list[0].args[0]
    assert "prefilter_mode != 'off'" in select_sql
    assert "p.sdc_write_requested = 0" in select_sql
    assert "i.bootstrapped = 0 AND i.prefilter_checked_at IS NULL" in select_sql
    assert "FOR UPDATE SKIP LOCKED" in select_sql
    # The sweep must not compete with the pipelines: active projects with work
    # and inference-eligible completed projects are left to their own claims.
    assert "i.status = 'pending'" in select_sql
    assert "f.is_target IS NULL" in select_sql
    assert "p.min_confirmed" in select_sql
    update_args = cursor.execute.call_args_list[1].args
    assert "SET worker_claimed_by = %s" in update_args[0]
    assert update_args[1] == ("test-worker-1", 42)


def test_claim_shadow_project_returns_none_when_nothing_is_eligible():
    cursor = MagicMock()
    cursor.fetchone.return_value = None

    with patch("worker.execute_transaction", side_effect=lambda operations: operations(MagicMock(), cursor)):
        assert _claim_shadow_project() is None


def test_claim_shadow_project_returns_none_on_db_error():
    from database import DatabaseError

    with patch("worker.execute_transaction", side_effect=DatabaseError("boom")):
        assert _claim_shadow_project() is None


def test_sweep_skips_when_service_marked_down():
    with (
        patch("worker._liftwing_service_down", return_value=True),
        patch("worker._claim_shadow_project") as claim,
        patch("worker.prefilter_images") as prefilter,
    ):
        assert _sweep_shadow_prefilter() == 0

    claim.assert_not_called()
    prefilter.assert_not_called()


def test_sweep_records_verdicts_for_a_finished_project():
    """Regression (prod report 2026-10-17): completed projects get swept."""
    project = {"id": 7, "status": "completed", "prefilter_mode": "shadow"}

    with (
        patch("worker._liftwing_service_down", return_value=False),
        patch("worker._claim_shadow_project", return_value=project),
        patch("worker.prefilter_images", return_value=250) as prefilter,
        patch("worker._release_project") as release,
    ):
        assert _sweep_shadow_prefilter() == 250

    prefilter.assert_called_once_with(project)
    release.assert_called_once_with(7)


def test_sweep_releases_claim_when_prefilter_fails():
    project = {"id": 7, "status": "completed", "prefilter_mode": "shadow"}

    with (
        patch("worker._liftwing_service_down", return_value=False),
        patch("worker._claim_shadow_project", return_value=project),
        patch("worker.prefilter_images", side_effect=RuntimeError("boom")),
        patch("worker._release_project") as release,
    ):
        assert _sweep_shadow_prefilter() == 0

    release.assert_called_once_with(7)


def test_sweep_returns_zero_when_no_project_needs_coverage():
    with (
        patch("worker._liftwing_service_down", return_value=False),
        patch("worker._claim_shadow_project", return_value=None),
        patch("worker.prefilter_images") as prefilter,
    ):
        assert _sweep_shadow_prefilter() == 0

    prefilter.assert_not_called()


def test_shadow_loop_sweeps_until_shutdown():
    calls = []

    def sweep():
        calls.append(1)
        _worker_module.shutdown_requested = True
        return 0

    with (
        patch("worker.shutdown_requested", False),
        patch("worker._sweep_shadow_prefilter", side_effect=sweep),
    ):
        _shadow_prefilter_loop()

    assert calls == [1]


def test_shadow_loop_disabled_with_zero_interval():
    with (
        patch.object(_worker_module, "LIFTWING_SWEEP_INTERVAL", 0),
        patch("worker._sweep_shadow_prefilter") as sweep,
    ):
        _shadow_prefilter_loop()

    sweep.assert_not_called()


def test_shadow_loop_survives_sweep_errors():
    calls = []

    def sweep():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        _worker_module.shutdown_requested = True
        return 0

    with (
        patch("worker.shutdown_requested", False),
        patch("worker._sweep_shadow_prefilter", side_effect=sweep),
        patch("worker.time.sleep"),
    ):
        _shadow_prefilter_loop()

    assert len(calls) == 2


def test_probe_disabled_interval_does_nothing():
    with (
        patch("worker.LIFTWING_PROBE_INTERVAL", 0),
        patch("worker.execute_query") as mock_db,
    ):
        _probe_liftwing_if_due()

    mock_db.assert_not_called()


def test_probe_skipped_when_no_project_uses_the_filter():
    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*) AS cnt FROM projects" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.LIFTWING_PROBE_INTERVAL", 300),
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client") as get_client,
    ):
        _probe_liftwing_if_due()

    get_client.assert_not_called()


def test_probe_skipped_when_claim_lost_to_another_worker():
    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*) AS cnt FROM projects" in sql:
            return [{"cnt": 1}]
        if sql.startswith("UPDATE service_status SET last_checked = NOW()"):
            return 0  # another worker probed first
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.LIFTWING_PROBE_INTERVAL", 300),
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client") as get_client,
    ):
        _probe_liftwing_if_due()

    get_client.assert_not_called()


def test_probe_claims_with_the_interval_and_marks_up_on_success():
    claim_params: list = []
    up_updates: list = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*) AS cnt FROM projects" in sql:
            return [{"cnt": 2}]
        if sql.startswith("UPDATE service_status SET last_checked = NOW()"):
            claim_params.append(params)
            return 1
        if "SET status = 'up'" in sql:
            up_updates.append(sql)
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()
    fake_client.probe.return_value = (True, None)

    with (
        patch("worker.LIFTWING_PROBE_INTERVAL", 300),
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
    ):
        _probe_liftwing_if_due()

    assert claim_params == [(300,)]
    assert len(up_updates) == 1
    fake_client.probe.assert_called_once()


def test_probe_records_failure_without_marking_down_below_threshold():
    captured: list = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "COUNT(*) AS cnt FROM projects" in sql:
            return [{"cnt": 1}]
        if sql.startswith("UPDATE service_status SET last_checked = NOW()"):
            return 1
        if sql.startswith("SELECT status, consecutive_failures"):
            return [{"status": "unknown", "consecutive_failures": 0}]
        if sql.startswith("UPDATE service_status SET status = %s"):
            captured.append((sql, params))
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    fake_client = MagicMock()
    fake_client.probe.return_value = (False, "HTTP 429")

    with (
        patch("worker.LIFTWING_PROBE_INTERVAL", 300),
        patch("worker.LIFTWING_FAIL_THRESHOLD", 3),
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.liftwing_client.get_client", return_value=fake_client),
    ):
        _probe_liftwing_if_due()

    assert len(captured) == 1
    sql, params = captured[0]
    assert "status = %s" in sql
    assert params == ("unknown", 1, "HTTP 429")


def test_record_liftwing_failure_below_threshold_keeps_status():
    captured: list = []

    def mock_execute_query(sql, params=None, fetch=True):
        if sql.startswith("SELECT status, consecutive_failures"):
            return [{"status": "up", "consecutive_failures": 1}]
        if sql.startswith("UPDATE service_status SET status = %s"):
            captured.append(params)
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.LIFTWING_FAIL_THRESHOLD", 3),
    ):
        _record_liftwing_failure("timeout")

    assert captured == [("up", 2, "timeout")]


def test_record_liftwing_failure_at_threshold_marks_down():
    captured: list = []

    def mock_execute_query(sql, params=None, fetch=True):
        if sql.startswith("SELECT status, consecutive_failures"):
            return [{"status": "up", "consecutive_failures": 2}]
        if sql.startswith("UPDATE service_status SET status = %s"):
            captured.append(params)
            return 1
        raise AssertionError(f"Unexpected query: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.LIFTWING_FAIL_THRESHOLD", 3),
    ):
        _record_liftwing_failure("HTTP 500")

    assert captured == [("down", 3, "HTTP 500")]


def test_fetch_metadata_parses_extmetadata_and_categories():
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 501,
                    "title": "File:A.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "extmetadata": {"ImageDescription": {"value": "<b>A</b> &amp; B <i>photo</i>"}},
                        }
                    ],
                    "categories": [{"title": "Category:People"}, {"title": "Category:X"}],
                }
            ]
        }
    }
    response = MagicMock()
    response.json.return_value = payload

    with patch("worker._api_request", return_value=response) as api:
        result = _fetch_metadata(["File:A.jpg"])

    assert result == {
        "File:A.jpg": {
            "description": "A & B photo",
            "categories": ("Category:People", "Category:X"),
            "mime": "image/jpeg",
        }
    }
    assert api.call_args.kwargs["method"] == "post"


def test_fetch_metadata_falls_back_to_object_name():
    payload = {
        "query": {
            "pages": [
                {
                    "title": "File:B.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "extmetadata": {"ObjectName": {"value": "Photo of a person"}},
                        }
                    ],
                    "categories": [],
                }
            ]
        }
    }
    response = MagicMock()
    response.json.return_value = payload

    with patch("worker._api_request", return_value=response):
        result = _fetch_metadata(["File:B.jpg"])

    assert result["File:B.jpg"]["description"] == "Photo of a person"


def test_plain_text_strips_wikitext_and_truncates():
    text = _plain_text("{{en|noise}} Photo of a person " + "x" * 400)

    assert text.startswith("Photo of a person")
    assert "{{" not in text
    assert len(text) == 250


def test_fetch_metadata_maps_non_canonical_titles_through_normalized():
    """The API answers under the canonical title and lists the rewrite in
    ``query.normalized``; results stay keyed by the requested title (L8)."""
    payload = {
        "query": {
            "normalized": [{"from": "File:photo_of_a_person.jpg", "to": "File:Photo of a person.jpg"}],
            "pages": [
                {
                    "title": "File:Photo of a person.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "extmetadata": {"ImageDescription": {"value": "A portrait"}},
                        }
                    ],
                    "categories": [{"title": "Category:People"}],
                }
            ],
        }
    }
    response = MagicMock()
    response.json.return_value = payload

    with patch("worker._api_request", return_value=response):
        result = _fetch_metadata(["File:photo_of_a_person.jpg"])

    assert result == {
        "File:photo_of_a_person.jpg": {
            "description": "A portrait",
            "categories": ("Category:People",),
            "mime": "image/jpeg",
        }
    }


def test_fetch_metadata_tolerates_a_page_without_imageinfo():
    payload = {"query": {"pages": [{"title": "File:C.jpg", "categories": []}]}}
    response = MagicMock()
    response.json.return_value = payload

    with patch("worker._api_request", return_value=response):
        result = _fetch_metadata(["File:C.jpg"])

    assert result == {"File:C.jpg": {"description": "", "categories": (), "mime": ""}}


def test_fetch_metadata_returns_empty_on_failure():
    with patch("worker._api_request", side_effect=RuntimeError("boom")):
        assert _fetch_metadata(["File:A.jpg"]) == {}


def test_fetch_metadata_empty_titles_makes_no_request():
    with patch("worker._api_request") as api:
        assert _fetch_metadata([]) == {}

    api.assert_not_called()


# ---------------------------------------------------------------------------
# A download that keeps failing must not keep a project from finishing (and so from inference)
# ---------------------------------------------------------------------------


def _run_download_wave(failing, *, rate_limited=(), canary=True, strikes=1):
    """One wave of images 1..3 where the downloads of *failing* raise a transient error."""
    db: list[tuple[str, Any]] = []
    canary_calls: list[str] = []

    def execute(sql, params=None, fetch=True):
        db.append((sql, params))
        if "SET detect_attempts" in sql:
            return 1
        if "SELECT detect_attempts" in sql:
            return [{"detect_attempts": strikes}]
        return None

    def download(img_id, title):
        if img_id in failing:
            raise _worker_module.TransientDownloadError("HTTP 500", rate_limited=img_id in rate_limited)
        return b"bytes"

    def canary_download(url):
        canary_calls.append(url)
        if not canary:
            raise requests.ConnectionError("down")
        return b"bytes"

    service = FakeFaceService()
    with (
        patch("worker.execute_query", side_effect=execute),
        patch("worker._download_for_detection", side_effect=download),
        patch("worker._download_image", side_effect=canary_download),
        patch("worker.face_client.detect_faces_batch", side_effect=service.detect),
        patch("worker._service_detects", side_effect=service.canary),
    ):
        processed, _timings = _process_image_wave(_wave(), 5)
    return processed, db, canary_calls


@pytest.fixture(autouse=True)
def _no_canary_memory(monkeypatch):
    monkeypatch.setattr(_worker_module, "_canary_ok_until", 0.0)


def test_a_download_that_fails_while_the_others_work_is_struck_without_asking_commons_again():
    processed, db, canary_calls = _run_download_wave({2})

    assert processed == 2
    assert _struck(db) == [2]
    assert canary_calls == []  # the other downloads of the wave already proved Commons works


def test_a_download_that_fails_alone_is_struck_when_the_canary_thumbnail_downloads():
    """The tail of a project: one image left, nothing in its wave to compare with."""
    processed, db, canary_calls = _run_download_wave({1, 2, 3})

    assert processed == 0
    assert _struck(db) == [1, 2, 3]
    assert canary_calls == [_worker_module.COMMONS_CANARY_URL]


def test_nobody_is_struck_for_failed_downloads_while_commons_serves_nothing():
    _processed, db, canary_calls = _run_download_wave({1, 2, 3}, canary=False)

    assert _struck(db) == []  # an outage is nobody's fault, the images stay pending
    assert len(canary_calls) == 1


def test_a_rate_limited_download_is_never_struck():
    _processed, db, canary_calls = _run_download_wave({1, 2}, rate_limited={1, 2})

    assert _struck(db) == []
    assert canary_calls == []  # nothing to blame, so nothing to prove


def test_the_third_download_strike_writes_the_image_off():
    _processed, db, _canary = _run_download_wave({2}, strikes=_worker_module.MAX_DETECT_ATTEMPTS)

    errors = [params for sql, params in db if "UPDATE images SET status = 'error'" in sql]
    assert len(errors) == 1 and errors[0][1] == 2 and "download failed" in errors[0][0]


def test_the_canary_answer_is_remembered_for_a_while():
    with patch("worker._download_image", return_value=b"x") as download:
        assert _worker_module._commons_serves_thumbnails() is True
        assert _worker_module._commons_serves_thumbnails() is True
    assert download.call_count == 1


def test_the_remaining_waves_are_left_alone_once_the_service_is_known_down(monkeypatch):
    rows = [
        {"id": i, "file_title": f"File:{i}.jpg", "bootstrapped": 0, "detect_isolate": 0}
        for i in range(1, _worker_module.DETECTION_WAVE_SIZE * 3 + 1)
    ]

    def execute(sql, params=None, fetch=True):
        if "SELECT id, file_title" in sql and "bootstrapped = 0" in sql:
            return rows[: _worker_module.BATCH_SIZE]
        if "COUNT(*)" in sql or "SUM(" in sql:
            return [{"bs_done": 0, "total": len(rows), "cnt": 0}]
        return []

    waves = []

    def wave(batch, project_id):
        waves.append(len(batch))
        _worker_module._mark_service_down()  # the first wave found the service down
        return 0, (0.0, 0.0, 0.0)

    monkeypatch.setattr(_worker_module, "DETECT_RECOVERY_WAIT", 60)
    monkeypatch.setattr(_worker_module, "BATCH_SIZE", len(rows))
    with patch("worker.execute_query", side_effect=execute), patch("worker._process_image_wave", side_effect=wave):
        _worker_module.process_images({"id": 5})

    assert waves == [_worker_module.DETECTION_WAVE_SIZE]  # the other two waves were not even tried


def test_a_failure_that_does_not_repeat_is_not_blamed_on_the_image():
    """Another thread or worker can hurt the shared service while one image is in flight."""
    service = FakeFaceService()
    attempts: list[str] = []

    def detect(items, retry=True):
        keys = [key for key, _data in items]
        service.requests.append((sorted(keys), retry))
        if keys == ["1"] and not attempts:  # the service is hurt once, by someone else, during image 1
            attempts.append("hurt")
            return {"1": face_client.FaceServiceUnavailable("connection reset")}
        if len(keys) > 1:
            return {key: face_client.FaceServiceUnavailable("batch failed") for key in keys}
        return {key: _good() for key in keys}

    service.detect = detect  # type: ignore[method-assign]

    processed, db, _sleeps = _run_wave(service, _wave(1, 2, 3))  # all flagged: they go alone from the start

    assert processed == 3  # the second attempt at image 1 worked
    assert _struck(db) == []
    assert service.requests.count((["1"], False)) == 2


def test_an_image_that_fails_twice_alone_is_blamed():
    service = FakeFaceService(poison=["2"])

    _processed, db, _sleeps = _run_wave(service, _wave(1, 2, 3))

    assert _struck(db) == [2]
    assert service.requests.count((["2"], False)) == 2


def test_a_download_that_is_still_rate_limited_after_the_retries_stays_recognisable_as_a_429():
    """urllib3 raised RetryError (no response) when its 429 retries ran out, so the download looked like
    any other failure and was struck. The session must hand back the last response instead."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Limited(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(429)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Limited)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/thumb.jpg"

    real_retry = _worker_module.Retry

    def quick_retry(**kwargs):
        return real_retry(**{**kwargs, "backoff_factor": 0})

    def download(_url, max_bytes=0):
        with patch("worker.Retry", quick_retry):
            session = _worker_module._create_session()
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        return resp.content

    try:
        with (
            patch.object(_worker_module, "_download_image", side_effect=download),
            patch.object(_worker_module, "_mark_image_error") as mark,
            pytest.raises(_worker_module.TransientDownloadError) as caught,
        ):
            _worker_module._download_for_detection(1, "File:X.jpg")
    finally:
        server.shutdown()

    assert caught.value.rate_limited is True
    mark.assert_not_called()


def test_the_session_hands_back_the_last_response_when_its_retries_run_out():
    session = _worker_module._create_session()
    policy = session.get_adapter("https://commons.wikimedia.org/").max_retries
    assert policy.raise_on_status is False
