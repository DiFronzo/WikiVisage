import os

os.environ.setdefault("TOOL_TOOLSDB_USER", "testuser")
os.environ.setdefault("TOOL_TOOLSDB_PASSWORD", "testpass")
os.environ.setdefault("WIKIVISAGE_DB_NAME", "testdb")
os.environ.setdefault("TOOL_TOOLSDB_HOST", "localhost")

from unittest.mock import patch

import numpy as np

with patch("database.init_db"):
    from worker import _process_single_image, run_autonomous_inference


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
    """Single-face bootstrapped image should auto-classify as target and increment faces_confirmed."""
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
