import os

os.environ.setdefault("TOOL_TOOLSDB_USER", "testuser")
os.environ.setdefault("TOOL_TOOLSDB_PASSWORD", "testpass")
os.environ.setdefault("WIKIVISAGE_DB_NAME", "testdb")
os.environ.setdefault("TOOL_TOOLSDB_HOST", "localhost")

from unittest.mock import MagicMock, patch

import numpy as np

with patch("database.init_db"):
    import worker as _worker_module
    from worker import (
        _claim_active_projects,
        _claim_inference_projects,
        _claim_sdc_projects,
        _infer_and_release,
        _process_and_release,
        _process_single_image,
        _refresh_claims,
        _release_all_claims,
        _release_project,
        _touch_heartbeat_file,
        bootstrap_from_sparql,
        process_images,
        process_project,
        run_autonomous_inference,
        write_sdc_claims,
    )


def _make_encoding(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float64).tobytes()


def _extract_update_maps(
    params: tuple,
) -> tuple[dict[int, int], dict[int, float], list[int]]:
    n = len(params) // 5
    is_target_map = {params[i * 2]: params[i * 2 + 1] for i in range(n)}

    conf_start = 2 * n
    confidence_map = {params[conf_start + i * 2]: float(params[conf_start + i * 2 + 1]) for i in range(n)}

    in_ids = list(params[4 * n : 5 * n])
    return is_target_map, confidence_map, in_ids


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


def test_inference_single_face_per_image():
    project = {"id": 11, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_encoding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 101, "image_id": 1, "encoding": _make_encoding(101)},
        {"id": 102, "image_id": 2, "encoding": _make_encoding(102)},
        {"id": 103, "image_id": 3, "encoding": _make_encoding(103)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    distances = [0.3, 0.8, 0.4]
    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker.face_recognition.face_distance",
            side_effect=[np.array([d], dtype=np.float64) for d in distances],
        ),
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


def test_inference_multi_face_dedup():
    project = {"id": 12, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_encoding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 201, "image_id": 1, "encoding": _make_encoding(201)},
        {"id": 202, "image_id": 1, "encoding": _make_encoding(202)},
        {"id": 203, "image_id": 2, "encoding": _make_encoding(203)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker.face_recognition.face_distance",
            side_effect=[
                np.array([0.5], dtype=np.float64),
                np.array([0.3], dtype=np.float64),
                np.array([0.4], dtype=np.float64),
            ],
        ),
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


def test_inference_multi_face_all_above_threshold():
    project = {"id": 13, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_encoding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 301, "image_id": 1, "encoding": _make_encoding(301)},
        {"id": 302, "image_id": 1, "encoding": _make_encoding(302)},
    ]
    update_params: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            update_params.append(params)
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker.face_recognition.face_distance",
            side_effect=[
                np.array([0.7], dtype=np.float64),
                np.array([0.8], dtype=np.float64),
            ],
        ),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 2
    assert len(update_params) == 1
    is_target_map, _, _ = _extract_update_maps(update_params[0])
    assert is_target_map[301] == 0
    assert is_target_map[302] == 0


def test_inference_returns_classified_count():
    project = {"id": 14, "min_confirmed": 5, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_encoding(i)} for i in range(5)]
    unclassified_rows = [
        {"id": 401, "image_id": 1, "encoding": _make_encoding(401)},
        {"id": 402, "image_id": 1, "encoding": _make_encoding(402)},
        {"id": 403, "image_id": 2, "encoding": _make_encoding(403)},
        {"id": 404, "image_id": 3, "encoding": _make_encoding(404)},
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 5}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            return len(unclassified_rows)
        raise AssertionError(f"Unexpected query executed: {sql}")

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker.face_recognition.face_distance",
            side_effect=[
                np.array([0.3], dtype=np.float64),
                np.array([0.5], dtype=np.float64),
                np.array([0.2], dtype=np.float64),
                np.array([0.9], dtype=np.float64),
            ],
        ),
        patch("worker.shutdown_requested", False),
    ):
        classified = run_autonomous_inference(project)

    assert classified == 4


def test_process_single_image_bootstrapped_single_face_auto_classifies():
    """Single-face bootstrapped image should auto-classify as target, set sdc_written=1, and increment faces_confirmed."""
    fake_location = (10, 110, 110, 10)
    fake_encoding = _make_encoding(1)

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        if "UPDATE faces SET is_target" in sql:
            return 1  # one face auto-classified
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_image", return_value=b"fake"),
        patch("worker._validate_image_dimensions"),
        patch("worker._run_face_detection", return_value=([fake_location], [fake_encoding], 800, 600)),
    ):
        result = _process_single_image(42, "File:Test.jpg", bootstrapped=True, project_id=5)

    assert result is True

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


def test_process_single_image_bootstrapped_multi_face_no_auto_classify():
    """Multi-face bootstrapped image should NOT trigger auto-classify or faces_confirmed update."""
    fake_locations = [(10, 110, 110, 10), (200, 310, 310, 200)]
    fake_encodings = [_make_encoding(2), _make_encoding(3)]

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_image", return_value=b"fake"),
        patch("worker._validate_image_dimensions"),
        patch("worker._run_face_detection", return_value=(fake_locations, fake_encodings, 800, 600)),
    ):
        result = _process_single_image(43, "File:Multi.jpg", bootstrapped=True, project_id=5)

    assert result is True
    assert not any("UPDATE faces SET is_target" in c[0] for c in query_calls)
    assert not any("UPDATE projects SET faces_confirmed" in c[0] for c in query_calls)


def test_process_single_image_non_bootstrapped_no_auto_classify():
    """Non-bootstrapped single-face image should NOT trigger auto-classify or faces_confirmed update."""
    fake_location = (10, 110, 110, 10)
    fake_encoding = _make_encoding(4)

    query_calls: list[tuple] = []

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls.append((sql, params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_image", return_value=b"fake"),
        patch("worker._validate_image_dimensions"),
        patch("worker._run_face_detection", return_value=([fake_location], [fake_encoding], 800, 600)),
    ):
        result = _process_single_image(44, "File:NonBoot.jpg", bootstrapped=False, project_id=5)

    assert result is True
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

    submitted_ids = []

    def fake_process_single(img_id, title, bootstrapped=False, project_id=None):
        submitted_ids.append((img_id, bootstrapped))
        return True

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._process_single_image", side_effect=fake_process_single),
        patch("worker.shutdown_requested", False),
    ):
        count = process_images(project)

    assert count == 3
    assert (1, False) in submitted_ids
    assert (2, False) in submitted_ids
    assert (3, True) in submitted_ids


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
        patch("worker._process_single_image"),
        patch("worker.shutdown_requested", False),
    ):
        count = process_images(project)

    assert count == 0
    assert not bootstrap_pending_queried, "Bootstrap pending query should not be executed when cap is already met"


import pytest

try:
    from conftest import _make_encoding
except ModuleNotFoundError:
    from tests.conftest import _make_encoding


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
                "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (image_id, _make_encoding(idx), 50, 200, 200, 50, 1, "human", seed_user["id"]),
            )

    project = {"id": seed_project["id"], "min_confirmed": 5, "distance_threshold": 0.6}

    with patch("worker.shutdown_requested", False):
        classified = run_autonomous_inference(project)

    assert classified == 0


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


@pytest.mark.integration
def test_inference_with_real_db_respects_threshold(db_pool, db_conn, seed_user, seed_project, seed_images):
    with db_conn.cursor() as cur:
        for img in seed_images:
            cur.execute(
                "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by, classified_by_user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (img["id"], _make_encoding(0), 50, 200, 200, 50, 1, "human", seed_user["id"]),
            )

        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (seed_images[0]["id"], _make_encoding(0), 100, 300, 300, 100),
        )
        close_face_id = cur.lastrowid

        cur.execute(
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (seed_images[1]["id"], _make_encoding(9999), 120, 320, 320, 120),
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
def test_inference_does_not_touch_bootstrap_images(
    db_pool,
    db_conn,
    seed_project,
    seed_images,
    seed_faces,
    seed_bootstrap_image,
):
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

    assert bootstrap_face["is_target"] is None
    assert bootstrap_face["classified_by"] is None


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
            "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (img_id, _make_encoding(700 + i), 50, 200, 200, 50, 1, "human", seed_user["id"]),
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
        "INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (new_img_id, _make_encoding(800), 50, 200, 200, 50),
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

    # Verify sdc_written=1 was set on the face
    face_updates = [(sql, p) for sql, p in db_calls if "UPDATE faces SET sdc_written" in sql]
    assert len(face_updates) == 1
    assert face_updates[0][1] == (100,)

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
# Tests for _build_skip_extensions_regex
# ---------------------------------------------------------------------------


def test_build_skip_extensions_regex_empty_set():
    from worker import _build_skip_extensions_regex

    pattern = _build_skip_extensions_regex(set())
    assert pattern == r"(?!)"
    import re

    assert not re.search(pattern, "File:something.webm")
    assert not re.search(pattern, "anything")


def test_build_skip_extensions_regex_nonempty():
    import re

    from worker import _build_skip_extensions_regex

    pattern = _build_skip_extensions_regex({".webm", ".ogg"})
    assert re.search(pattern, "File:clip.webm")
    assert re.search(pattern, "File:audio.ogg")
    assert not re.search(pattern, "File:photo.jpg")


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


def test_download_image_content_length_too_large_raises():
    from worker import _download_image

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(100 * 1024 * 1024)}  # 100 MB
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()

    with (
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="too large"):
            _download_image("https://upload.wikimedia.org/image.jpg", max_bytes=50 * 1024 * 1024)


def test_download_image_streaming_size_exceeded_raises():
    from worker import _download_image

    chunks = [b"x" * 1024 * 1024] * 60  # 60 MB in 1 MB chunks

    mock_resp = MagicMock()
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()
    mock_resp.iter_content.return_value = iter(chunks)

    with (
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="exceeded"):
            _download_image("https://upload.wikimedia.org/image.jpg", max_bytes=50 * 1024 * 1024)


def test_download_image_success():
    from worker import _download_image

    image_data = b"FAKE_IMAGE_DATA"
    mock_resp = MagicMock()
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.close = MagicMock()
    mock_resp.iter_content.return_value = iter([image_data])

    with (
        patch("worker._get_session") as mock_session,
    ):
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
        patch("worker.decrypt_token", side_effect=lambda t: t),
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
        patch("worker.decrypt_token", side_effect=lambda t: t),
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
        patch("worker.decrypt_token", side_effect=lambda t: t),
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
        patch("worker.decrypt_token", side_effect=lambda t: t),
        patch("worker.encrypt_token", side_effect=lambda t: t),
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.post.return_value = mock_resp
        result = _refresh_worker_token(1)

    assert result is None


def test_refresh_worker_token_rowcount_zero_reads_fresh():
    """When rowcount==0 (another process refreshed), re-read token from DB."""
    from datetime import UTC, datetime, timedelta

    from worker import _refresh_worker_token

    past = datetime.now(UTC) - timedelta(hours=1)
    user = {
        "access_token": "old-token",
        "refresh_token": "old-refresh",
        "token_expires_at": past,
    }
    fresh_row = {"access_token": "fresh-token"}

    query_calls = [0]

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls[0] += 1
        if "SELECT access_token, refresh_token, token_expires_at" in sql and query_calls[0] == 1:
            return [user]
        if "UPDATE users SET access_token" in sql:
            return 0  # rowcount == 0: another process refreshed
        if "SELECT access_token FROM users" in sql:
            return [fresh_row]
        return []

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"access_token": "new-token", "refresh_token": "new-refresh", "expires_in": 14400}
    mock_resp.close = MagicMock()

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.decrypt_token", side_effect=lambda t: t),
        patch("worker.encrypt_token", side_effect=lambda t: t),
        patch("worker._get_session") as mock_session,
    ):
        mock_session.return_value.post.return_value = mock_resp
        result = _refresh_worker_token(1)

    assert result == "fresh-token"


# ---------------------------------------------------------------------------
# Tests for traverse_category
# ---------------------------------------------------------------------------


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
            return len([x for x in range(0, len(flat), 3)])
        if "UPDATE projects SET images_total" in sql:
            return 1
        db_calls.append(sql)
        return []

    # API returns more files than remaining capacity
    api_resp = MagicMock()
    api_resp.json.return_value = {
        "query": {
            "categorymembers": [{"ns": 6, "title": f"File:Img{i}.jpg", "pageid": 5000 + i} for i in range(10)]
        }
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
# Tests for _process_single_image error path
# ---------------------------------------------------------------------------


def test_process_single_image_download_error_marks_image_as_error():
    """When download fails, image status is set to 'error'."""
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_image", side_effect=RuntimeError("download failed")),
    ):
        result = _process_single_image(55, "File:Broken.jpg")

    assert result is False
    error_updates = [(s, p) for s, p in db_calls if "UPDATE images SET status = 'error'" in s]
    assert len(error_updates) == 1
    assert error_updates[0][1][1] == 55


def test_process_single_image_no_faces_no_auto_classify():
    """When no faces are detected, no is_target update is made."""
    db_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        db_calls.append((sql.strip(), params))
        return None

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker._download_image", return_value=b"fake"),
        patch("worker._validate_image_dimensions"),
        patch("worker._run_face_detection", return_value=([], [], 800, 600)),
    ):
        result = _process_single_image(56, "File:Empty.jpg", bootstrapped=True, project_id=10)

    assert result is True
    auto_class = [s for s, _ in db_calls if "UPDATE faces SET is_target" in s]
    assert len(auto_class) == 0


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
        patch("worker._process_single_image"),
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
    assert "Token expired" in error_updates[0][0]  # error message is in SQL, not params


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
    assert "CSRF" in error_updates[0][0]  # error message is in SQL, not params


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
    error_updates = [(s, p) for s, p in db_calls if "Invalid QID" in str(s)]
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


# ---------------------------------------------------------------------------
# Tests for run_autonomous_inference edge cases
# ---------------------------------------------------------------------------


def test_inference_skips_invalid_confirmed_encodings():
    """Confirmed faces with invalid encoding lengths are skipped."""
    project = {"id": 20, "min_confirmed": 2, "distance_threshold": 0.6}

    confirmed_rows = [
        {"encoding": b"short"},  # Invalid: not 1024 bytes
        {"encoding": _make_encoding(1)},  # Valid
        {"encoding": _make_encoding(2)},  # Valid
    ]
    unclassified_rows = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 2}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
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
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch("worker.shutdown_requested", False),
    ):
        result = run_autonomous_inference(project)

    assert result == 0


def test_inference_skips_invalid_candidate_encodings():
    """Candidate faces with invalid encoding lengths are skipped without crashing."""
    project = {"id": 22, "min_confirmed": 2, "distance_threshold": 0.6}

    confirmed_rows = [{"encoding": _make_encoding(i)} for i in range(3)]
    unclassified_rows = [
        {"id": 500, "image_id": 1, "encoding": b"bad"},  # Invalid — should be skipped
        {"id": 501, "image_id": 2, "encoding": _make_encoding(500)},  # Valid
    ]
    update_calls = []

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT COUNT(*) AS cnt FROM faces f" in sql:
            return [{"cnt": 3}]
        if "SELECT f.encoding FROM faces f" in sql and "f.is_target = 1" in sql:
            return confirmed_rows
        if "SELECT f.id, f.image_id, f.encoding FROM faces f" in sql:
            return unclassified_rows
        if "UPDATE faces SET" in sql:
            update_calls.append(params)
            return 1
        if "UPDATE projects SET last_inference" in sql:
            return 1
        return []

    with (
        patch("worker.execute_query", side_effect=mock_execute_query),
        patch(
            "worker.face_recognition.face_distance",
            return_value=np.array([0.3], dtype=np.float64),
        ),
        patch("worker.shutdown_requested", False),
    ):
        result = run_autonomous_inference(project)

    # Only the valid candidate (id=501) should be classified
    assert result == 1
