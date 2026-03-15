import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

os.environ.setdefault("TOOL_TOOLSDB_USER", "testuser")
os.environ.setdefault("TOOL_TOOLSDB_PASSWORD", "testpass")
os.environ.setdefault("WIKIVISAGE_DB_NAME", "testdb")
os.environ.setdefault("TOOL_TOOLSDB_HOST", "localhost")
os.environ.setdefault("OAUTH_CLIENT_ID", "test-client-id")
os.environ.setdefault("OAUTH_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

import numpy as np

with patch("database.init_db"):
    import app as app_module

    flask_app = app_module.app

import requests


def _fake_user():
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _authed_client(monkeypatch, route_execute_query=None, csrf_token=None):
    if csrf_token is None:
        csrf_token = "testtoken"
    fake_user = _fake_user()
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    monkeypatch.setattr(app_module.limiter, "enabled", False, raising=False)
    client = flask_app.test_client()

    def _query(sql, *args, **kwargs):
        del args, kwargs
        if "FROM users WHERE id" in sql:
            return [fake_user]
        if route_execute_query is not None:
            return route_execute_query(sql)
        return [fake_user]

    monkeypatch.setattr(app_module, "execute_query", _query)
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        if csrf_token is not None:
            sess["csrf_token"] = csrf_token

    return client, fake_user


def _manual_face_form(**overrides):
    payload = {
        "csrf_token": "testtoken",
        "project_id": "1",
        "image_id": "10",
        "bbox_top": "10",
        "bbox_right": "30",
        "bbox_bottom": "40",
        "bbox_left": "10",
    }
    payload.update(overrides)
    return payload


def _update_bbox_form(**overrides):
    payload = {
        "csrf_token": "testtoken",
        "face_id": "1",
        "bbox_top": "10",
        "bbox_right": "30",
        "bbox_bottom": "40",
        "bbox_left": "10",
    }
    payload.update(overrides)
    return payload


def _build_face_recognition_mock(encodings=None, face_encodings_side_effect=None):
    mock_fr = MagicMock()
    mock_fr.load_image_file.return_value = MagicMock()
    if face_encodings_side_effect is not None:
        mock_fr.face_encodings.side_effect = face_encodings_side_effect
    else:
        mock_fr.face_encodings.return_value = [] if encodings is None else encodings
    return mock_fr


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
        if "FROM faces f " in sql and "old_is_target" in sql:
            if ownership_error:
                raise app_module.DatabaseError("ownership failure")
            return [local_face_row] if ownership_exists else []
        if "has_sibling_match" in sql:
            if sibling_error:
                raise app_module.DatabaseError("sibling failure")
            return [{"has_sibling_match": sibling_match}]
        return []

    return _route_query


def _default_bbox_face_row(**overrides):
    row = {
        "id": 1,
        "image_id": 10,
        "is_target": 1,
        "classified_by": "model",
        "confidence": 0.23,
        "classified_by_user_id": None,
        "file_title": "File:Face.jpg",
        "project_id": 1,
    }
    row.update(overrides)
    return row


def _bbox_query_router(face_row=None, exists=True, ownership_error=False):
    local_row = _default_bbox_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "file_title" in sql:
            if ownership_error:
                raise app_module.DatabaseError("ownership failure")
            return [local_row] if exists else []
        return []

    return _route_query


def test_api_manual_face_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_manual_face_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data={"csrf_token": "testtoken"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_manual_face_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(project_id="abc"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_manual_face_invalid_bbox_dimensions(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(bbox_top="50", bbox_bottom="40"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid bounding box dimensions"


def test_api_manual_face_bbox_out_of_range_negative(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(bbox_left="-1"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_bbox_out_of_range_too_large(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/manual-face", data=_manual_face_form(bbox_right="10001"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_bbox_out_of_range_too_small_area(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(bbox_top="10", bbox_left="10", bbox_bottom="19", bbox_right="20"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_manual_face_ownership_check_fail(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return []
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Image not found or access denied"


def test_api_manual_face_no_encoding_result(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(encodings=[])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_manual_face_success_normal_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    with client.session_transaction() as sess:
        assert sess["manual_faces_10"] == [99]
        assert sess["last_classify"]["action"] == "manual_face"
        assert sess["last_classify"]["was_review"] is False
        assert sess["last_classify"]["face_ids"] == []
        assert sess["last_classify"]["manual_face_ids"] == [99]


def test_api_manual_face_success_review_mode_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 11}, {"id": 12}]
        mock_cursor.lastrowid = 100
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form(reviewing_model="1"))

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    with client.session_transaction() as sess:
        assert sess["manual_faces_10"] == [100]
        assert sess["last_classify"]["was_review"] is True
        assert sess["last_classify"]["face_ids"] == [11, 12]
        assert sess["last_classify"]["manual_face_ids"] == [100]


def test_api_manual_face_download_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("network")

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", _raise)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 502
    assert resp.get_json()["error"] == "Failed to download image from Commons"


def test_api_manual_face_db_error_on_ownership(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            raise app_module.DatabaseError("db")
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_manual_face_db_error_on_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )

    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_manual_face_unexpected_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    mock_fr = _build_face_recognition_mock(face_encodings_side_effect=RuntimeError("boom"))
    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


def test_remove_sdc_claim_success_claim_found_and_removed(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"success": 1}

    get_mock = MagicMock(side_effect=[claim_resp, token_resp])
    post_mock = MagicMock(return_value=remove_resp)
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", post_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 2
    assert post_mock.call_count == 1
    post_data = post_mock.call_args.kwargs["data"]
    assert post_data["action"] == "wbremoveclaims"
    assert post_data["claim"] == "M555$ABC"


def test_remove_sdc_claim_no_matching_claim_returns_true(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$NOTME", "mainsnak": {"datavalue": {"value": {"id": "Q1"}}}},
            ]
        }
    }

    get_mock = MagicMock(return_value=claim_resp)
    post_mock = MagicMock()
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", post_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 1
    assert post_mock.call_count == 0


def test_remove_sdc_claim_fetches_csrf_token(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"success": 1}

    get_mock = MagicMock(side_effect=[claim_resp, token_resp])
    monkeypatch.setattr(app_module.requests, "get", get_mock)
    monkeypatch.setattr(app_module.requests, "post", MagicMock(return_value=remove_resp))

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    second_get_params = get_mock.call_args_list[1].kwargs["params"]
    assert second_get_params["action"] == "query"
    assert second_get_params["meta"] == "tokens"
    assert second_get_params["type"] == "csrf"


def test_remove_sdc_claim_removal_api_returns_error(monkeypatch):
    claim_resp = MagicMock()
    claim_resp.raise_for_status.return_value = None
    claim_resp.json.return_value = {
        "claims": {
            "P180": [
                {"id": "M555$ABC", "mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
            ]
        }
    }

    token_resp = MagicMock()
    token_resp.raise_for_status.return_value = None
    token_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf-token"}}}

    remove_resp = MagicMock()
    remove_resp.raise_for_status.return_value = None
    remove_resp.json.return_value = {"error": {"code": "badtoken"}}

    monkeypatch.setattr(app_module.requests, "get", MagicMock(side_effect=[claim_resp, token_resp]))
    monkeypatch.setattr(app_module.requests, "post", MagicMock(return_value=remove_resp))

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False


def test_remove_sdc_claim_exception_during_api_calls(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        MagicMock(side_effect=requests.RequestException("network")),
    )

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False


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
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["is_target"] == 1
    assert payload["classified_by"] == "model"
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
        mock_cursor.fetchone.return_value = {"classified_by_user_id": 999}
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    resp = client.post("/api/reclassify", data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"})

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "This face has already been reviewed by another user"


def test_api_reclassify_same_user_reclick_noop_success(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_reclassify_query_router())

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.fetchone.return_value = {"classified_by_user_id": 1}
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


def test_api_update_face_bbox_csrf_fail(monkeypatch):
    client, _ = _authed_client(monkeypatch, csrf_token="different")

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid CSRF token"


def test_api_update_face_bbox_missing_fields(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data={"csrf_token": "testtoken"})

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing required fields"


def test_api_update_face_bbox_invalid_values(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(face_id="abc"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid field values"


def test_api_update_face_bbox_invalid_bbox_dimensions(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/update-face-bbox",
        data=_update_bbox_form(bbox_top="40", bbox_bottom="40"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Invalid bounding box dimensions"


def test_api_update_face_bbox_bbox_out_of_range_negative(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(bbox_top="-1"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_bbox_out_of_range_too_large(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form(bbox_right="20000"))

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_bbox_out_of_range_too_small_area(monkeypatch):
    client, _ = _authed_client(monkeypatch)

    resp = client.post(
        "/api/update-face-bbox",
        data=_update_bbox_form(bbox_top="10", bbox_left="10", bbox_bottom="19", bbox_right="20"),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Bounding box out of allowed range"


def test_api_update_face_bbox_face_not_found(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(exists=False))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Face not found or access denied"


def test_api_update_face_bbox_no_encoding_result(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(encodings=[])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_update_face_bbox_success_new_face_inserted_original_superseded(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 321
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload == {"status": "ok", "new_face_id": 321, "original_face_id": 1}

    executed_sql = [call.args[0] for call in captured["calls"]]
    assert any(sql.startswith("INSERT INTO faces") for sql in executed_sql)
    assert any("UPDATE faces SET superseded_by" in sql for sql in executed_sql)


def test_api_update_face_bbox_download_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(
        app_module,
        "_download_image",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.RequestException("network")),
    )

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 502
    assert resp.get_json()["error"] == "Failed to download image from Commons"


def test_api_update_face_bbox_db_error_on_ownership(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(ownership_error=True))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Database error"


def test_api_update_face_bbox_db_error_on_insert(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: (_ for _ in ()).throw(app_module.DatabaseError("db")),
    )
    fake_encoding = np.random.rand(128).astype(np.float64)
    mock_fr = _build_face_recognition_mock(encodings=[fake_encoding])

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_update_face_bbox_unexpected_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    mock_fr = _build_face_recognition_mock(face_encodings_side_effect=RuntimeError("boom"))

    with patch.dict("sys.modules", {"face_recognition": mock_fr}):
        resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"
