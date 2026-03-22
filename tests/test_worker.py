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
        process_images,
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
    """_claim_sdc_projects returns whatever the transaction closure produces."""
    fake_projects = [{"id": 7, "sdc_write_requested": 1}]

    with patch("worker.execute_transaction", return_value=fake_projects) as mock_txn:
        result = _claim_sdc_projects()

    assert result == fake_projects
    mock_txn.assert_called_once()


def test_claim_sdc_projects_returns_empty_on_db_error():
    """_claim_sdc_projects swallows DatabaseError and returns []."""
    from database import DatabaseError

    with patch("worker.execute_transaction", side_effect=DatabaseError("boom")):
        result = _claim_sdc_projects()

    assert result == []


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
