"""Tests for the web app: undo reclassify."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from tests._app_preamble import (
    _make_embedding,
    app_module,
    flask_app,
)
from tests._app_support import (
    _auth_client_chunk4,
    _authed_client,
    _fake_user_chunk5,
    _get_flashes,
    _set_csrf_chunk4,
)


def _undo_state_eq(state_rows, sibling=False, write_requested=0):
    """execute_query for the undo route's ownership, state and sibling lookups."""

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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        if "f.sdc_removal_pending" in sql and "FROM faces f" in sql:
            return list(state_rows)
        if "has_sibling" in sql:
            return [{"has_sibling": 1 if sibling else 0}]
        if "SELECT sdc_write_requested FROM projects WHERE id = %s" in sql:
            return [{"sdc_write_requested": write_requested}]
        return ()

    return eq


def _undo_session(client, **overrides):
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "target",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": False,
            **overrides,
        }


def _undo_tx(executed):
    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    return tx


def _default_reclassify_face_row(**overrides):
    row = {
        "id": 1,
        "image_id": 10,
        "old_is_target": 0,
        "sdc_written": 0,
        "classified_by": "model",
        "commons_page_id": 555,
        "bootstrapped": 0,
        "project_id": 1,
        "wikidata_qid": "Q42",
    }
    row.update(overrides)
    return row


def _reclassify_query_router(
    face_row=None, ownership_exists=True, sibling_match=0, ownership_error=False, sibling_error=False
):
    local_face_row = _default_reclassify_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            if ownership_error:
                raise app_module.DatabaseError("ownership failure")
            return [local_face_row] if ownership_exists else []
        if "has_sibling" in sql:
            if sibling_error:
                raise app_module.DatabaseError("sibling failure")
            return [{"has_sibling": sibling_match}]
        return []

    return _route_query


@pytest.mark.integration
def test_api_undo_classify(integration_client, seed_project, seed_images, db_conn):
    client, _, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 50, 200, 200, 50)",
            (seed_images[0]["id"], _make_embedding(1001)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    classify_response = client.post(
        "/api/classify",
        data={
            "csrf_token": "testtoken",
            "selected_face_id": str(face_id),
            "project_id": str(seed_project["id"]),
            "image_id": str(seed_images[0]["id"]),
        },
    )
    assert classify_response.status_code == 200

    undo_response = client.post(
        "/api/undo-classify",
        data={"csrf_token": "testtoken"},
    )

    assert undo_response.status_code == 200
    assert undo_response.get_json()["status"] == "ok"

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] is None
    assert row["classified_by"] is None
    assert row["classified_by_user_id"] is None


@pytest.mark.integration
def test_api_reclassify_approve(integration_client, seed_images, db_conn):
    client, user, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 50, 200, 200, 50, 0, 'model', NULL)",
            (seed_images[0]["id"], _make_embedding(1002)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/reclassify",
        data={
            "csrf_token": "testtoken",
            "face_id": str(face_id),
            "is_target": "1",
        },
    )

    assert response.status_code == 200

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT is_target, classified_by_user_id FROM faces WHERE id = %s",
            (face_id,),
        )
        row = cur.fetchone()

    assert row["is_target"] == 1
    assert row["classified_by_user_id"] == user["id"]


@pytest.mark.integration
def test_api_reclassify_reject(integration_client, seed_images, db_conn):
    client, _, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, bbox_bottom, bbox_left, "
            "is_target, classified_by, classified_by_user_id, sdc_written) "
            "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 50, 200, 200, 50, 1, 'model', NULL, 0)",
            (seed_images[0]["id"], _make_embedding(1003)),
        )
        face_id = cur.lastrowid

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        "/api/reclassify",
        data={
            "csrf_token": "testtoken",
            "face_id": str(face_id),
            "is_target": "0",
        },
    )

    assert response.status_code == 200

    with db_conn.cursor() as cur:
        cur.execute("SELECT is_target FROM faces WHERE id = %s", (face_id,))
        row = cur.fetchone()

    assert row["is_target"] == 0


def test_api_undo_classify_csrf_fail(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client, token="abc")
    response = client.post("/api/undo-classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_undo_classify_no_last_classify(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nothing to undo"


def test_api_undo_classify_empty_face_ids_clears_session(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nothing to undo"
    with client.session_transaction() as sess:
        assert "last_classify" not in sess


def test_api_undo_classify_ownership_check_fail(monkeypatch):
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
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [10], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 404
    assert response.get_json()["error"] == "Image not found or access denied"


def test_api_undo_classify_ownership_check_db_error(monkeypatch):
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
    with client.session_transaction() as sess:
        sess["last_classify"] = {"project_id": 1, "image_id": 2, "face_ids": [10], "manual_face_ids": []}
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_undo_classify_target_normal_decrements_counter(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "target",
            "face_ids": [10, 11],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("SET is_target = NULL" in sql for sql, _ in executed)
    assert any("faces_confirmed = GREATEST" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert "last_classify" not in sess


def test_api_undo_classify_none_review_restores_model(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "none",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": True,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("SET is_target = 0, classified_by = 'model'" in sql for sql, _ in executed)
    assert not any("faces_confirmed = GREATEST" in sql for sql, _ in executed)


def test_api_undo_classify_manual_face_deletion_and_exclusion(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "none",
            "face_ids": [10, 11, 12],
            "manual_face_ids": [11, 12],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    delete_sql = [sql for sql, _ in executed if sql.startswith("DELETE FROM faces")]
    assert delete_sql
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 0, 10, 2)


def test_api_undo_removes_written_claim_immediately(monkeypatch):
    """Undo of a written target removes its P180 claim right away when no sibling confirms it."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is True
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_called_once_with(555, "Q42", "valid-token")
    assert ("info", "SDC claim was removed.") in _get_flashes(client)
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 0, 10, 2)


def test_api_undo_removal_failure_queues_pending(monkeypatch):
    """When the immediate removal fails, undo proceeds and queues the worker instead."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", MagicMock(return_value=False))
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    assert ("info", "SDC claim removal queued.") in _get_flashes(client)
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (1, 1, 10, 2)


def test_api_undo_missing_token_queues_pending(monkeypatch):
    """Without a usable token the removal is queued for the worker, not lost."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: None)
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    assert ("info", "SDC claim removal queued.") in _get_flashes(client)
    remove_mock.assert_not_called()


def test_api_undo_sibling_target_keeps_claim(monkeypatch):
    """A sibling target still confirms the person, so undo must not touch the claim."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows, sibling=True))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    assert _get_flashes(client) == []
    remove_mock.assert_not_called()
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 0, 10, 2)


def test_api_undo_in_flight_write_queues_removal(monkeypatch):
    """An unwritten target is queued when a write run is active (claim may be in flight)."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 0,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows, write_requested=1))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 1, 10, 2)


def test_api_undo_no_in_flight_write_does_not_queue(monkeypatch):
    """An unwritten target with no write run active cannot have a claim, so nothing is queued."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 1,
            "sdc_written": 0,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows, write_requested=0))
    _set_csrf_chunk4(client)
    _undo_session(client)

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    assert _get_flashes(client) == []
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 0, 10, 2)


def test_api_undo_none_action_cancels_the_removal_it_queued(monkeypatch):
    """Undoing a rejection reverts its queued removal instead of confirming it.

    The "none" action queues sdc_removal_pending=1 for rejected faces on
    bootstrapped images; undoing that rejection must cancel the queue, not
    delete the (pre-existing) P180 claim.
    """
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 0,
            "sdc_written": 0,
            "sdc_removal_pending": 1,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
            "bootstrapped": 1,
        }
    ]
    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client, action="none")

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_not_called()
    assert _get_flashes(client) == []
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 0, 10, 2)


def test_api_undo_none_action_keeps_written_marker_without_touching_commons(monkeypatch):
    """A claim marker on record survives undoing the rejection that queued its removal."""
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 0,
            "sdc_written": 1,
            "sdc_removal_pending": 1,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
            "bootstrapped": 1,
        }
    ]
    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client, action="none")

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_not_called()
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (1, 0, 10, 2)


def test_api_undo_none_action_preserves_a_preexisting_worker_queue(monkeypatch):
    """Rejections only queue removals on bootstrapped images.

    On any other image a queued flag predates the rejection (an earlier undo
    racing a write run), so undoing the rejection keeps the queue for the
    worker to verify.
    """
    executed = []
    state_rows = [
        {
            "id": 10,
            "is_target": 0,
            "sdc_written": 0,
            "sdc_removal_pending": 1,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
            "bootstrapped": 0,
        }
    ]
    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client, action="none")

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    remove_mock.assert_not_called()
    assert ("info", "SDC claim removal queued.") in _get_flashes(client)
    update_params = [params for sql, params in executed if "SET is_target = NULL" in sql][0]
    assert update_params == (0, 1, 10, 2)


def test_api_undo_manual_face_claim_removal_failure_blocks_undo(monkeypatch):
    """A manual face that cannot lose its claim before deletion blocks the undo (no orphan)."""
    executed = []
    state_rows = [
        {
            "id": 11,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", MagicMock(return_value=False))
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client, action="manual_face", face_ids=[11], manual_face_ids=[11])

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 502
    assert not any(sql.startswith("DELETE FROM faces") for sql, _ in executed)
    assert _get_flashes(client) == []
    with client.session_transaction() as sess:
        assert "last_classify" in sess


def test_api_undo_manual_face_claim_removed_then_deleted(monkeypatch):
    """A manual face whose claim is removed first can be deleted as before."""
    executed = []
    state_rows = [
        {
            "id": 11,
            "is_target": 1,
            "sdc_written": 1,
            "sdc_removal_pending": 0,
            "commons_page_id": 555,
            "wikidata_qid": "Q42",
        }
    ]
    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)
    monkeypatch.setattr(app_module, "execute_transaction", _undo_tx(executed))

    client, _ = _auth_client_chunk4(monkeypatch, _undo_state_eq(state_rows))
    _set_csrf_chunk4(client)
    _undo_session(client, action="manual_face", face_ids=[11], manual_face_ids=[11])

    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sdc_removed"] is True
    remove_mock.assert_called_once_with(555, "Q42", "valid-token")
    assert ("info", "SDC claim was removed.") in _get_flashes(client)
    assert any(sql.startswith("DELETE FROM faces") for sql, _ in executed)


def test_api_undo_classify_manual_face_review_action_decrements(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "manual_face",
            "face_ids": [20],
            "manual_face_ids": [20],
            "was_review": True,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert any("faces_confirmed = GREATEST" in sql for sql, _ in executed)


def test_api_undo_classify_db_error_during_transaction(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("undo fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 1,
            "image_id": 2,
            "action": "target",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to undo classification"
    with client.session_transaction() as sess:
        assert "last_classify" in sess


def test_api_undo_classify_success_response_contains_ids(monkeypatch):
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
            return [{"id": 2, "file_title": "File:Test.jpg", "status": "processed"}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.return_value = None
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["last_classify"] = {
            "project_id": 9,
            "image_id": 77,
            "action": "none",
            "face_ids": [10],
            "manual_face_ids": [],
            "was_review": False,
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        "status": "ok",
        "project_id": 9,
        "image_id": 77,
        "sdc_removed": False,
        "sdc_removal_queued": False,
    }


def test_api_undo_skip_removes_from_skip_list(monkeypatch):
    """Undo of skip removes image from skip list and returns image_id for redirect."""

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
            return [{"id": 5, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_5"] = [10, 20, 30]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 5,
            "image_id": 20,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 5, "image_id": 20}
    with client.session_transaction() as sess:
        assert 20 not in sess["skipped_images_5"]
        assert sess["skipped_images_5"] == [10, 30]
        assert "last_classify" not in sess


def test_api_undo_skip_review_removes_from_review_skip_list(monkeypatch):
    """Undo of review-mode skip removes from the review skip list."""

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
            return [{"id": 3, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_review_3"] = [50, 60]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 3,
            "image_id": 60,
            "was_review": True,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 3, "image_id": 60}
    with client.session_transaction() as sess:
        assert sess["skipped_images_review_3"] == [50]
        assert "last_classify" not in sess


def test_api_undo_skip_idempotent_when_not_in_list(monkeypatch):
    """Undo of skip succeeds even if image_id is not in skip list (idempotent)."""

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
            return [{"id": 7, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_7"] = [100]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 7,
            "image_id": 999,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 7, "image_id": 999}
    with client.session_transaction() as sess:
        assert sess["skipped_images_7"] == [100]
        assert "last_classify" not in sess


def test_api_undo_skip_no_db_transaction_needed(monkeypatch):
    """Undo of skip does NOT call execute_transaction (session-only operation)."""
    tx_called = []

    def tx(_fn):
        tx_called.append(True)
        raise AssertionError("execute_transaction should not be called for skip undo")

    monkeypatch.setattr(app_module, "execute_transaction", tx)

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
            return [{"id": 2, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_2"] = [42]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 2,
            "image_id": 42,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    assert len(tx_called) == 0


def test_api_undo_skip_returns_404_when_not_owner_or_member(monkeypatch):
    """Undo of skip returns 404 if user is not the project owner or a member."""

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
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["skipped_images_99"] = [10]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 99,
            "image_id": 10,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 404
    with client.session_transaction() as sess:
        assert sess["skipped_images_99"] == [10]
        assert sess.get("last_classify") is not None


def test_api_undo_skip_works_for_project_member(monkeypatch):
    """Undo of skip succeeds when user is a project member (not owner)."""

    def eq(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [
                {
                    "id": 2,
                    "wiki_username": "tester",
                    "access_token": "token",
                    "refresh_token": "refresh",
                    "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
                }
            ]
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 5, "user_id": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    _set_csrf_chunk4(client)
    with client.session_transaction() as sess:
        sess["user_id"] = 2
        sess["skipped_images_5"] = [10, 20]
        sess["last_classify"] = {
            "action": "skip",
            "project_id": 5,
            "image_id": 20,
            "was_review": False,
            "face_ids": [],
            "manual_face_ids": [],
        }
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok", "project_id": 5, "image_id": 20}
    with client.session_transaction() as sess:
        assert sess["skipped_images_5"] == [10]
        assert "last_classify" not in sess


def test_api_reclassify_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_reclassify_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_reclassify_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "abc", "is_target": "1"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_reclassify_invalid_is_target_outside_allowed(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "2"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_reclassify_face_not_found(monkeypatch):
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(ownership_exists=False),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Face not found or access denied"


def test_api_reclassify_approve_from_model_face(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["is_target"] == 1
    assert payload["classified_by"] == "model"
    assert payload["sdc_written"] == 0
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    assert payload["updated_faces"] == []


def test_api_reclassify_approve_echoes_sdc_written_when_p180_exists(monkeypatch):
    """Approve echoes the post-action ``sdc_written`` so the gallery card can refresh its badge."""
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module, "_maybe_mark_sdc_written", MagicMock(return_value=True))
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["sdc_written"] == 1
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False


def test_api_reclassify_reject_bootstrap_no_sibling_sdc_removal_queued(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    assert payload["sdc_written"] == 0


def test_api_reclassify_reject_bootstrap_with_sibling_no_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=1),
    )

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False


def test_api_reclassify_reject_nonbootstrap_unwritten_no_sibling_queues_removal(monkeypatch):
    """A rejection with sdc_written=0 queues removal: a write may still be in flight."""
    executed = []
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=_default_reclassify_face_row(), sibling_match=0),
    )

    remove_mock = MagicMock()
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)

    def _transaction(fn):
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.rowcount = 1
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(mock_conn, cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is True
    assert payload["sdc_written"] == 0
    remove_mock.assert_not_called()
    update_params = [params for sql, params in executed if sql.startswith("UPDATE faces SET is_target")][0]
    # (is_target, user_id, sdc_written, sdc_removal_pending, face_id, user_id)
    assert update_params == (0, 1, 0, 1, 1, 1)


def test_api_reclassify_reject_nonbootstrap_sdc_written_no_sibling_immediate_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is True
    assert payload["sdc_removal_queued"] is False
    assert payload["sdc_written"] == 0
    remove_mock.assert_called_once_with(555, "Q42", "valid-token")


def test_api_reclassify_reject_nonbootstrap_sdc_written_with_sibling_no_removal(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=1),
    )

    remove_mock = MagicMock(return_value=True)
    monkeypatch.setattr(app_module, "_remove_sdc_claim", remove_mock)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sdc_removed"] is False
    assert payload["sdc_removal_queued"] is False
    remove_mock.assert_not_called()


def test_api_reclassify_reject_nonbootstrap_sdc_removal_fails(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    monkeypatch.setattr(app_module, "_get_valid_token", lambda: "valid-token")
    monkeypatch.setattr(app_module, "_remove_sdc_claim", lambda *_a, **_k: False)
    tx_mock = MagicMock()
    monkeypatch.setattr(app_module, "execute_transaction", tx_mock)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 502
    assert "Failed to remove SDC claim" in resp.get_json()["error"]
    tx_mock.assert_not_called()


def test_api_reclassify_reject_nonbootstrap_token_expired(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_match=0),
    )

    monkeypatch.setattr(app_module, "_get_valid_token", lambda: None)
    tx_mock = MagicMock()
    monkeypatch.setattr(app_module, "execute_transaction", tx_mock)

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 401
    assert "OAuth token expired" in resp.get_json()["error"]
    tx_mock.assert_not_called()


def test_api_reclassify_already_reviewed_by_another_user_returns_409(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": 999,
            "wiki_username": "OtherContributor",
        }
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 409
    body = resp.get_json()
    assert body["reason"] == "already_reviewed"
    # Friendly, actionable message that names the previous classifier
    assert "OtherContributor" in body["error"]


def test_api_reclassify_already_reviewed_unknown_user_returns_generic_409(monkeypatch):
    """If the previous classifier's username can't be resolved (e.g. user
    deleted, FK already SET NULL), still return a clean 409 with a generic
    actionable message."""
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": 999,
            "wiki_username": None,
        }
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 409
    body = resp.get_json()
    assert body["reason"] == "already_reviewed"
    assert "another contributor" in body["error"].lower()


def test_api_reclassify_same_user_reclick_noop_success(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {"classified_by_user_id": 1}
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_reclassify_counter_increment_on_approve(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0)
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any("faces_confirmed = faces_confirmed + 1" in sql for sql in executed_sql)
    assert any("UPDATE faces SET sdc_removal_pending = 0" in sql for sql in executed_sql)


def test_api_reclassify_counter_decrement_on_reject(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=1, sdc_written=0)
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 200
    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any("CAST(faces_confirmed AS SIGNED) - 1" in sql for sql in executed_sql)


def test_api_reclassify_db_error_on_ownership(monkeypatch):
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(ownership_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_sibling_lookup_bootstrap(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_sibling_lookup_nonbootstrap(monkeypatch):
    face_row = _default_reclassify_face_row(bootstrapped=0, sdc_written=1)
    client, _ = _authed_client(
        monkeypatch,
        route_execute_query=_reclassify_query_router(face_row=face_row, sibling_error=True),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "0"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_reclassify_db_error_on_execute_transaction(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save reclassification"


def test_api_reclassify_approve_marks_sdc_written_when_p180_exists(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    executed_queries = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        if "FROM images i JOIN projects p" in sql and "commons_page_id" in sql:
            return [{"commons_page_id": face_row["commons_page_id"], "wikidata_qid": face_row["wikidata_qid"]}]
        return []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: True)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)
    assert any("UPDATE images SET bootstrapped = 1" in sql for sql, _ in executed_queries)


def test_api_reclassify_approve_no_sdc_written_when_p180_missing(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    executed_queries = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        return []

    def eq(sql, params=None, fetch=True):
        del fetch
        executed_queries.append((sql, params))
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", lambda cpid, qid: False)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert not any("UPDATE faces SET sdc_written = 1" in sql for sql, _ in executed_queries)


def test_api_reclassify_approve_skips_p180_check_when_already_sdc_written(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model", sdc_written=1)
    p180_called = []

    def route_query(sql):
        if "FROM faces f " in sql and "old_is_target" in sql and "project_members" in sql:
            return [face_row]
        return []

    def eq(sql, params=None, fetch=True):
        del params, fetch
        if "FROM users WHERE id" in sql:
            return [_fake_user_chunk5()]
        return route_query(sql)

    def mock_check(cpid, qid):
        p180_called.append((cpid, qid))
        return True

    monkeypatch.setattr(app_module, "execute_query", eq)

    monkeypatch.setattr(app_module, "_check_p180_exists", mock_check)
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_cursor.fetchall.return_value = []
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    client = flask_app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["csrf_token"] = "testtoken"

    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    assert len(p180_called) == 0


def test_api_reclassify_approve_rejects_sibling_target_faces(monkeypatch):
    face_row = _default_reclassify_face_row(old_is_target=0, classified_by="model")
    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router(face_row=face_row))

        executed_sql = []

        def _transaction(fn):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.rowcount = 1
            mock_cursor.fetchall.return_value = [{"id": 100, "sdc_written": 0}, {"id": 101, "sdc_written": 1}]

            def capture_execute(sql, params=None):
                executed_sql.append(sql)
                mock_cursor.rowcount = 1

            mock_cursor.execute = capture_execute
            return fn(mock_conn, mock_cursor)

        monkeypatch.setattr(app_module, "execute_transaction", _transaction)
        resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})
    finally:
        flask_app.config["SERVER_NAME"] = None

    assert resp.status_code == 200

    payload = resp.get_json()
    assert len(payload["updated_faces"]) == 2
    assert {uf["face_id"] for uf in payload["updated_faces"]} == {100, 101}

    # Verify sdc_removal_queued matches pre-demotion sdc_written state per sibling
    sdc_by_face = {uf["face_id"]: uf["sdc_removal_queued"] for uf in payload["updated_faces"]}
    assert sdc_by_face[100] is False, "Face 100 had sdc_written=0, should not queue removal"
    assert sdc_by_face[101] is True, "Face 101 had sdc_written=1, should queue removal"

    sibling_reject_sqls = [s for s in executed_sql if "is_target = 0" in s and "image_id" in s and "id !=" in s]
    assert len(sibling_reject_sqls) > 0, "Expected UPDATE to reject sibling target faces"

    # Verify sdc_written=0 and sdc_removal_pending CASE are in the demotion UPDATE
    demotion_sql = sibling_reject_sqls[0]
    assert "sdc_written = 0" in demotion_sql, "Expected sdc_written reset on demoted siblings"
    assert "sdc_removal_pending" in demotion_sql, "Expected sdc_removal_pending handling on demoted siblings"

    counter_decrement_sqls = [s for s in executed_sql if "GREATEST" in s]
    assert len(counter_decrement_sqls) > 0, "Expected faces_confirmed decrement for rejected siblings"
