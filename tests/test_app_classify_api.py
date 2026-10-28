"""Tests for the web app: classify api."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from tests._app_preamble import (
    _make_embedding,
    app_module,
)
from tests._app_support import (
    _auth_client_chunk4,
    _capture_render_template_chunk4,
    _make_authed_client,
    _set_csrf_chunk4,
)


@pytest.mark.integration
def test_api_classify_target(integration_client, seed_project, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_embedding(999)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": str(face_id),
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 1
    assert row["classified_by"] == "human"
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_classify_none(integration_client, seed_project, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_embedding(1000)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 0
    assert row["classified_by"] == "human"
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_classify_missing_fields(integration_client):
    client, _, _ = integration_client

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 400


@pytest.mark.integration
def test_api_classify_wrong_project(integration_client, seed_images):
    client, _, _ = integration_client

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": "999999",
            "image_id": str(seed_images[0]["id"]),
        },
    )

    assert response.status_code == 404


def test_classify_project_not_found(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 404


def test_classify_project_load_db_error(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        raise app_module.DatabaseError("oops")

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 500


def test_classify_skip_image_id_normal_mode_adds_session(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 15,
                    "file_title": "File:A.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 1, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=77")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_1"] == [77]
    assert captured["context"]["skipped_count"] == 1


def test_classify_skip_image_id_review_mode_adds_review_session(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 22,
                    "file_title": "File:B.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 9, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": 0.2}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=88&skip_reviewing=1")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_review_1"] == [88]
    assert captured["context"]["skipped_review_count"] == 1


def test_classify_skip_image_id_invalid_ignored(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=not-an-int")
    assert response.status_code == 200
    assert captured["context"]["skipped_count"] == 0


def test_classify_forced_image_review_mode_when_no_unclassified(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 44,
                    "file_title": "File:C.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": 0}]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 100, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=44")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"][0]["face_id"] == 100


def test_classify_forced_image_normal_mode_with_unclassified(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 45,
                    "file_title": "File:D.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": 2}]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 101, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=45")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is False
    assert captured["context"]["faces"][0]["face_id"] == 101


def test_classify_labels_only_number_first_nine_faces(monkeypatch):
    """Faces 10+ have no keyboard shortcut, so their "FACE N" tags are not
    rendered (they used to print "FACE 20" over the photo). The boxes stay
    clickable and keep an aria-label, and the labels can be toggled off."""
    faces = [
        {
            "face_id": 100 + i,
            "bbox_left": 1,
            "bbox_top": 2,
            "bbox_right": 20,
            "bbox_bottom": 40,
            "confidence": None,
        }
        for i in range(1, 13)
    ]

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 45,
                    "file_title": "File:D.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": len(faces)}]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return faces
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1/classify?image_id=45").get_data(as_text=True)

    assert body.count('<span class="face-label">') == 9
    assert ">Face 9</span>" in body
    assert ">Face 10</span>" not in body
    # Still announced to screen readers even though the visual tag is gone.
    assert 'aria-label="Face 12"' in body
    # Labels-off toggle (persisted, and present for the no-label touch default).
    assert 'id="btnToggleLabels"' in body
    assert "function toggleLabels()" in body
    assert "wikivisage-face-labels" in body
    assert ".image-wrapper.hide-labels .face-label" in body
    # "H" toggles them, and the hotkey row advertises it.
    assert "e.key === 'h' || e.key === 'H'" in body
    assert "<kbd>H</kbd> Labels" in body


def test_classify_single_face_has_no_label_toggle(monkeypatch):
    """A single face renders no "FACE N" tag, so there is nothing to toggle:
    neither the button nor the "H" hint belongs on that page (the single-face
    keydown branch has no H handler either)."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND i.id = %s" in sql:
            return [
                {
                    "image_id": 45,
                    "file_title": "File:D.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE image_id = %s AND is_target IS NULL" in sql:
            return [{"cnt": 1}]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {
                    "face_id": 101,
                    "bbox_left": 1,
                    "bbox_top": 2,
                    "bbox_right": 20,
                    "bbox_bottom": 40,
                    "confidence": None,
                }
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1/classify?image_id=45").get_data(as_text=True)

    assert "<kbd>Y</kbd>" in body  # single-face branch rendered
    assert 'id="btnToggleLabels"' not in body
    assert "<kbd>H</kbd>" not in body
    assert '<span class="face-label">' not in body


def test_classify_skipped_ids_path_uses_not_in(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[-1])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "i.id NOT IN" in sql and "f.is_target IS NULL" in sql:
            assert params == (1, 99)
            return [
                {
                    "image_id": 2,
                    "file_title": "File:E.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 3,
                    "file_title": "File:F.jpg",
                    "commons_page_id": 3,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 5, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_1"] = [99]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 3


def test_classify_normal_no_skips_uses_random_choice(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[1])

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "WHERE i.project_id = %s AND f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 20,
                    "file_title": "File:G.jpg",
                    "commons_page_id": 20,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 21,
                    "file_title": "File:H.jpg",
                    "commons_page_id": 21,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 7, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 21


def test_classify_fallback_to_model_review_without_skipped_review(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 55,
                    "file_title": "File:I.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 8, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["image"]["image_id"] == 55


def test_classify_fallback_with_skipped_review_ids(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[-1])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "i.id NOT IN" in sql and "f.classified_by = 'model'" in sql:
            assert params == (1, 5, 6)
            return [
                {
                    "image_id": 70,
                    "file_title": "File:J.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                },
                {
                    "image_id": 71,
                    "file_title": "File:K.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                },
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 81, "bbox_left": 1, "bbox_top": 2, "bbox_right": 20, "bbox_bottom": 40, "confidence": 0.1}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_review_1"] = [5, 6]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 71


def test_classify_image_loading_db_error_results_no_image(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            raise app_module.DatabaseError("load fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"] is None
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_normal(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 1,
                    "file_title": "File:X.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            raise app_module.DatabaseError("faces fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_review(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 1,
                    "file_title": "File:X.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            raise app_module.DatabaseError("faces fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"] == []


def test_classify_remaining_count_db_error(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            raise app_module.DatabaseError("count fail")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["remaining"] == 0
    assert captured["context"]["model_review_count"] == 0


def test_clear_skips_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/project/1/classify/clear-skips", data={"csrf_token": "wrong"})
    assert response.status_code == 403


def test_clear_skips_nonmember_returns_404(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post("/project/999/classify/clear-skips", data={"csrf_token": "testtoken"})
    assert response.status_code == 404


def test_clear_skips_clears_normal_and_review_keys(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_user_id": 123,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, execute_query_impl=eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_1"] = [1, 2]
        sess["skipped_images_review_1"] = [3]

    response = client.post("/project/1/classify/clear-skips", data={"csrf_token": "testtoken"})
    assert response.status_code == 302
    assert response.location.endswith("/project/1/classify")
    with client.session_transaction() as sess:
        assert "skipped_images_1" not in sess
        assert "skipped_images_review_1" not in sess


def test_api_classify_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/api/classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_classify_invalid_field_values(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "bad", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid field values"


def test_api_classify_ownership_check_fail(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 404
    assert response.get_json()["error"] == "Image not found or access denied"


def test_api_classify_ownership_check_db_error(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            raise app_module.DatabaseError("db")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_classify_invalid_selected_face_id(monkeypatch):
    queries = []

    def eq(sql, params=None, fetch=True):
        queries.append((sql, params, fetch))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "abc", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid face ID"
    assert any(
        "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in q[0]
        for q in queries
    )


def test_api_classify_none_normal_mode_sets_last_classify(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["manual_faces_22"] = [400]

    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    with client.session_transaction() as sess:
        assert sess["last_classify"]["action"] == "none"
        assert sess["last_classify"]["face_ids"] == [10, 11]
        assert sess["last_classify"]["manual_face_ids"] == [400]
        assert sess["last_classify"]["was_review"] is False
        assert "manual_faces_22" not in sess
    assert any("UPDATE faces SET is_target = 0" in sql for sql, _ in executed)


def test_api_classify_none_review_mode_updates_human_flags(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "none",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET classified_by = 'human'" in sql for sql, _ in executed)


def test_api_classify_none_bootstrapped_without_sibling_queues_removal(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET sdc_removal_pending = 1" in sql for sql, _ in executed)


def test_api_classify_none_bootstrapped_with_sibling_does_not_queue_removal(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 1}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert not any("UPDATE faces SET sdc_removal_pending = 1" in sql for sql, _ in executed)


def test_api_classify_target_normal_mode_updates_and_counter(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET is_target = 1" in sql for sql, _ in executed)
    assert any("id != %s AND is_target IS NULL" in sql for sql, _ in executed)
    assert any("UPDATE projects SET faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)
    # Re-targeting a face drops any queued P180 removal — the claim must stay.
    assert any("UPDATE faces SET is_target = 1" in sql and "sdc_removal_pending = 0" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["action"] == "target"
        assert sess["last_classify"]["selected_face_id"] == 10
        assert sess["last_classify"]["was_review"] is False


def test_api_classify_target_review_mode_updates_other_faces_human(monkeypatch):
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "AND is_target = 0 AND classified_by = 'model'" in sql and "SET is_target = 1" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "10",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    assert any(
        "AND is_target = 0 AND classified_by = 'model'" in sql and "SET is_target = 1" in sql for sql, _ in executed
    )
    assert any("AND is_target = 0 AND classified_by = 'model'" in sql and "id != %s" in sql for sql, _ in executed)
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["was_review"] is True


def test_api_classify_target_review_mode_does_not_use_is_target_null(monkeypatch):
    """Regression: review-mode target UPDATE must NOT use is_target IS NULL (would match 0 rows)."""
    executed = []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed.append((sql, params))
            if "SET is_target = 1" in sql:
                if "is_target IS NULL" in sql:
                    cursor.rowcount = 0
                elif "is_target = 0" in sql:
                    cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": "10",
            "project_id": "1",
            "image_id": "22",
            "reviewing_model": "1",
        },
    )
    assert response.status_code == 200
    target_sqls = [sql for sql, _ in executed if "SET is_target = 1" in sql]
    assert len(target_sqls) == 1
    assert "is_target IS NULL" not in target_sqls[0]
    assert "is_target = 0" in target_sqls[0]
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)


def test_api_classify_db_error_during_transaction(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("tx fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to save classification"


def test_classify_skip_sets_last_classify_with_skip_action(monkeypatch):
    """Skip sets session['last_classify'] with action='skip' so undo button appears."""
    captured = _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 15,
                    "file_title": "File:A.jpg",
                    "commons_page_id": 1,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target IS NULL" in sql:
            return [
                {"face_id": 1, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": None}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=77")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        last = sess.get("last_classify")
        assert last is not None
        assert last["action"] == "skip"
        assert last["project_id"] == 1
        assert last["image_id"] == 77
        assert last["was_review"] is False
    assert captured["context"]["has_undo"] is True


def test_classify_skip_review_sets_last_classify_with_was_review(monkeypatch):
    """Skip in review mode sets was_review=True in last_classify."""
    _capture_render_template_chunk4(monkeypatch)
    monkeypatch.setattr(app_module.random, "choice", lambda rows: rows[0])

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1}]
        if "f.is_target IS NULL" in sql and "LIMIT 200" in sql:
            return []
        if "f.is_target = 0 AND f.classified_by = 'model'" in sql and "LIMIT 200" in sql:
            return [
                {
                    "image_id": 22,
                    "file_title": "File:B.jpg",
                    "commons_page_id": 2,
                    "detection_width": 100,
                    "detection_height": 100,
                }
            ]
        if "WHERE f.image_id = %s AND f.is_target = 0" in sql:
            return [
                {"face_id": 9, "bbox_left": 10, "bbox_top": 1, "bbox_right": 20, "bbox_bottom": 30, "confidence": 0.2}
            ]
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=88&skip_reviewing=1")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        last = sess.get("last_classify")
        assert last is not None
        assert last["action"] == "skip"
        assert last["was_review"] is True
        assert last["image_id"] == 88


def test_api_progress_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 404


def test_api_progress_active_with_pending_and_stats(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 4, "images_total": 10, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 6}]
        if "COUNT(*) AS total_faces" in sql:
            return [
                {
                    "total_faces": 12,
                    "confirmed_matches": 5,
                    "confirmed_non_matches": 3,
                    "unclassified": 4,
                    "sdc_written": 2,
                    "by_human": 7,
                    "by_model": 4,
                    "by_bootstrap": 1,
                }
            ]
        if "AND f.is_target IS NULL" in sql and "inference_eligible" not in sql:
            # Faces with a queued P180 removal are not inference candidates.
            assert "AND f.sdc_removal_pending = 0" in sql
            return [{"cnt": 4}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["images_processed"] == 4
    assert payload["images_total"] == 10
    assert payload["pending_images"] == 6
    assert payload["complete"] is False
    assert payload["face_stats"]["total_faces"] == 12
    assert payload["inference_eligible"] == 4


def _progress_payload(monkeypatch, fake_user, project_row, *, live_claim=False):
    claim_queries = []

    def route_execute(sql, params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [project_row]
        if "worker_claimed_by IS NOT NULL" in sql:
            claim_queries.append(params)
            return [{"id": 1}] if live_claim else ()
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 0}]
        if "COUNT(*) AS total_faces" in sql:
            return [{}]
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    return client.get("/api/progress/1").get_json(), claim_queries


def test_api_progress_says_whether_a_worker_already_has_the_project(monkeypatch, fake_user):
    """With no images yet, "claimed" tells a project being read apart from one still queued."""
    base = {"id": 1, "user_id": 1, "images_processed": 0, "images_total": 0, "status": "active"}

    queued, _ = _progress_payload(monkeypatch, fake_user, base, live_claim=False)
    reading, queries = _progress_payload(monkeypatch, fake_user, base, live_claim=True)

    assert queued["claimed"] is False
    assert reading["claimed"] is True
    assert reading["images_total"] == 0 and reading["complete"] is False
    assert len(queries) == 1  # and it only asks when there is something to tell


def test_api_progress_never_claims_for_a_project_with_images_or_one_that_is_not_active(monkeypatch, fake_user):
    with_images = {"id": 1, "user_id": 1, "images_processed": 2, "images_total": 10, "status": "active"}
    paused = {"id": 1, "user_id": 1, "images_processed": 0, "images_total": 0, "status": "paused"}

    busy, busy_queries = _progress_payload(monkeypatch, fake_user, with_images, live_claim=True)
    idle, idle_queries = _progress_payload(monkeypatch, fake_user, paused, live_claim=True)

    assert busy["claimed"] is False and idle["claimed"] is False
    assert busy_queries == [] and idle_queries == []  # no database question at all


def test_api_progress_reports_the_last_completed_inference_run(monkeypatch, fake_user):
    from datetime import datetime as dt

    row = {
        "id": 1,
        "user_id": 1,
        "images_processed": 10,
        "images_total": 10,
        "status": "completed",
        "last_inference_at": dt(2026, 10, 28, 12, 0, 0),
        "last_inference_run": 5,
        "last_inference_matches": 7,
    }
    payload, _ = _progress_payload(monkeypatch, fake_user, row)
    never, _ = _progress_payload(monkeypatch, fake_user, {k: v for k, v in row.items() if "inference" not in k})

    assert payload["inference_run"] == 5  # the run counter, not a timestamp
    assert payload["inference_matches"] == 7
    assert never["inference_run"] == 0 and never["inference_matches"] is None


def test_api_progress_completed_project(monkeypatch, fake_user):
    seen_pending_query = {"called": False}

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 10, "images_total": 10, "status": "completed"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            seen_pending_query["called"] = True
            return [{"cnt": 1}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["complete"] is True
    assert payload["pending_images"] == 0
    assert seen_pending_query["called"] is False


def test_api_progress_face_stats_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 1, "images_total": 3, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 2}]
        if "COUNT(*) AS total_faces" in sql:
            raise app_module.DatabaseError("boom")
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 9}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["face_stats"] == {}
    assert payload["inference_eligible"] == 9


def test_api_progress_inference_eligible_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 2, "images_total": 3, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            return [{"cnt": 1}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5}]
        if "AND f.is_target IS NULL" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["inference_eligible"] == 0


def test_api_progress_pending_images_db_error_is_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "images_processed": 0, "images_total": 8, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM images" in sql:
            raise app_module.DatabaseError("boom")
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "AND f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 200
    assert response.get_json()["pending_images"] == 0


def test_api_progress_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_classify_target_marks_sdc_written_when_p180_exists(monkeypatch):
    executed_queries = []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            return [{"commons_page_id": 555, "wikidata_qid": "Q42"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed_queries.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: True)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)
    assert any("UPDATE images SET bootstrapped = 1" in sql for sql, _ in executed_queries)


def test_api_classify_target_no_sdc_written_when_p180_missing(monkeypatch):
    executed_queries = []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            return [{"commons_page_id": 555, "wikidata_qid": "Q42"}]
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            executed_queries.append((sql, params))
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: False)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert not any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)


def test_api_classify_target_p180_check_db_error_non_fatal(monkeypatch):
    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 1,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i" in sql:
            return [{"id": 22, "file_title": "File:Test.jpg", "status": "processed"}]
        if "SELECT i.commons_page_id, p.wikidata_qid" in sql:
            raise app_module.DatabaseError("meta lookup failed")
        return ()

    def tx(fn):
        cursor = MagicMock()

        def _execute(sql, params=None):
            if "WHERE id = %s AND image_id = %s AND is_target IS NULL" in sql:
                cursor.rowcount = 1

        cursor.execute.side_effect = _execute
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
