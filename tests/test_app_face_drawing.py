"""Tests for the web app: face drawing."""

from unittest.mock import MagicMock

import requests

from tests._app_preamble import (
    _make_embedding,
    app_module,
)
from tests._app_support import (
    _authed_client,
)


def _known_encoding(seed: int, *, aligned: bool = True):
    """What ``face_client.encode_known_face`` returns for a located face."""
    return app_module.face_client.KnownFaceEncoding(_make_embedding(seed), "scrfd500m+arcface_mbf", aligned)


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


def _default_bbox_face_row(**overrides):
    row = {
        "id": 1,
        "image_id": 10,
        "is_target": 1,
        "classified_by": "model",
        "confidence": 0.23,
        "classified_by_user_id": None,
        "sdc_written": 0,
        "file_title": "File:Face.jpg",
        "commons_page_id": 12345,
        "project_id": 1,
        "wikidata_qid": "Q42",
        "image_status": "processed",
        "prev_classifier_username": None,
    }
    row.update(overrides)
    return row


def _bbox_query_router(face_row=None, exists=True, ownership_error=False):
    local_row = _default_bbox_face_row() if face_row is None else face_row

    def _route_query(sql):
        if "FROM faces f " in sql and "file_title" in sql and "project_members" in sql:
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


def test_api_manual_face_dismiss_face_ids_too_many(monkeypatch):
    import json as _json

    client, _ = _authed_client(monkeypatch)
    oversized = _json.dumps(list(range(101)))
    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(dismiss_face_ids=oversized),
    )

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Too many face IDs in dismiss list"


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
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: None)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_manual_face_success_normal_insert(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
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
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(2001))

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    with client.session_transaction() as sess:
        assert sess["manual_faces_10"] == [99]
        assert sess["last_classify"]["action"] == "manual_face"
        assert sess["last_classify"]["was_review"] is False
        assert sess["last_classify"]["face_ids"] == []
        assert sess["last_classify"]["manual_face_ids"] == [99]


def test_api_manual_face_unaligned_encoding_is_stored_with_flag(monkeypatch):
    """L3: a drawn box the detector cannot confirm is embedded unaligned; the
    faces row records that so the effect is measurable later."""
    executed_sqls = []

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_cursor.rowcount = 1

        original_execute = mock_cursor.execute

        def tracking_execute(sql, params=None):
            executed_sqls.append((sql, params))
            return original_execute(sql, params)

        mock_cursor.execute = tracking_execute
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(
        app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(2100, aligned=False)
    )

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 200
    inserts = [(sql, params) for sql, params in executed_sqls if "INSERT INTO faces" in sql]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "aligned" in sql
    assert params[3] == 0  # aligned flag sits right after the embedding columns


def test_api_manual_face_face_service_rejected_returns_422(monkeypatch):
    """A malformed image is terminal for this request: 422, not a retryable 503.

    Only the HTTP transport is stubbed: the rejection travels through
    face_client's real parsing path, so the client's error classification is
    part of what is being tested.
    """

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _rejected(_instances):
        return [
            {
                "id": "0",
                "status": "error",
                "error": "could not decode image: broken data",
                "error_code": "invalid_image",
            }
        ]

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "_post", _rejected)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert resp.get_json()["error"] == "This image could not be read by the face detection service."


def test_api_manual_face_unusable_region_still_asks_for_a_new_box(monkeypatch):
    """A box-level rejection is not an image problem: the user should draw a
    different box, not be told the image is unreadable."""

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _rejected(_instances):
        return [{"id": "0", "status": "error", "error": "box lies outside", "error_code": "invalid_box"}]

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "_post", _rejected)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_manual_face_dismiss_faces_on_normal_insert(monkeypatch):
    """When dismiss_face_ids is sent, the API marks those faces as non-target
    in the same transaction, preventing the image from reappearing."""
    executed_sqls = []

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 99
        mock_cursor.rowcount = 1

        original_execute = mock_cursor.execute

        def tracking_execute(sql, params=None):
            executed_sqls.append((sql, params))
            return original_execute(sql, params)

        mock_cursor.execute = tracking_execute
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)

    import json

    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(2002))

    resp = client.post(
        "/api/manual-face",
        data=_manual_face_form(dismiss_face_ids=json.dumps([5, 7])),
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}

    dismiss_sqls = [(sql, params) for sql, params in executed_sqls if "UPDATE faces SET is_target = 0" in sql]
    assert len(dismiss_sqls) == 1
    sql, params = dismiss_sqls[0]
    assert "id IN (%s,%s)" in sql
    # params: (user_id, face_id_1, face_id_2, image_id)
    assert params[1] == 5
    assert params[2] == 7

    with client.session_transaction() as sess:
        assert sess["last_classify"]["face_ids"] == [5, 7]
        assert sess["last_classify"]["was_review"] is False
        assert sess["last_classify"]["manual_face_ids"] == [99]


def test_api_manual_face_review_mode_tracks_confirmed_ids(monkeypatch):
    def _review_route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _review_transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{"id": 11}, {"id": 12}]
        mock_cursor.lastrowid = 100
        mock_cursor.rowcount = 1
        return fn(mock_conn, mock_cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_review_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _review_transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(2003))

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
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
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
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _raise_db_error(*_args, **_kwargs):
        raise app_module.DatabaseError("db")

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _raise_db_error)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(2004))

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_manual_face_unexpected_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "encode_known_face", _raise_runtime_error)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


def test_api_manual_face_face_service_error(monkeypatch):
    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _raise_face_service_error(*_args, **_kwargs):
        raise app_module.FaceServiceError("service down")

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "encode_known_face", _raise_face_service_error)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "Face detection service is unavailable. Please try again in a moment."


def test_api_manual_face_image_not_processed(monkeypatch):
    """Return 409 when the image has not finished processing (status != 'processed')."""

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "pending"}]
        return []

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Image has not finished processing yet"


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
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: None)

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
        # In-tx re-check fetches the previous classifier; default row has
        # classified_by_user_id=None so the re-check should also see None.
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": None,
            "wiki_username": None,
        }
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3001))

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

    def _raise_db_error(*_args, **_kwargs):
        raise app_module.DatabaseError("db")

    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        _raise_db_error,
    )
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3002))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to save face"


def test_api_update_face_bbox_unexpected_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.face_client, "encode_known_face", _raise_runtime_error)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Failed to process face region"


def test_api_update_face_bbox_face_service_error(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _raise_face_service_error(*_args, **_kwargs):
        raise app_module.FaceServiceError("service down")

    monkeypatch.setattr(app_module.face_client, "encode_known_face", _raise_face_service_error)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "Face detection service is unavailable. Please try again in a moment."


def test_api_update_face_bbox_face_service_rejected_returns_422(monkeypatch):
    """Only the HTTP transport is stubbed; the rejection travels through
    face_client's real parsing path (see the manual-face counterpart)."""
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _rejected(_instances):
        return [
            {
                "id": "0",
                "status": "error",
                "error": "could not decode image: broken data",
                "error_code": "invalid_image",
            }
        ]

    monkeypatch.setattr(app_module.face_client, "_post", _rejected)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 422
    assert resp.get_json()["error"] == "This image could not be read by the face detection service."


def test_api_update_face_bbox_unusable_region_still_asks_for_a_new_box(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _rejected(_instances):
        return [{"id": "0", "status": "error", "error": "box lies outside", "error_code": "invalid_box"}]

    monkeypatch.setattr(app_module.face_client, "_post", _rejected)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 422
    assert "Could not compute face encoding" in resp.get_json()["error"]


def test_api_update_face_bbox_unaligned_encoding_records_flag(monkeypatch):
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router())
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    captured = {}

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 322
        mock_cursor.fetchone.return_value = {"classified_by_user_id": None, "wiki_username": None}
        result = fn(mock_conn, mock_cursor)
        captured["calls"] = mock_cursor.execute.call_args_list
        return result

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(
        app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3003, aligned=False)
    )

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    inserts = [call for call in captured["calls"] if call.args[0].startswith("INSERT INTO faces")]
    assert len(inserts) == 1
    sql, params = inserts[0].args[0], inserts[0].args[1]
    assert "aligned" in sql
    assert params[3] == 0


def test_api_update_face_bbox_image_not_processed(monkeypatch):
    face_row = _default_bbox_face_row(image_status="pending")
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))
    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Image has not finished processing yet"


def test_api_update_face_bbox_blocks_other_users_face_with_username(monkeypatch):
    """Pre-check: a member trying to edit a face classified by ANOTHER user
    is blocked with a 409 naming the previous classifier. The download +
    encoding pipeline must NOT run."""
    face_row = _default_bbox_face_row(
        classified_by_user_id=2,
        prev_classifier_username="OtherUser",
    )
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))

    download_called = {"hit": False}

    def _should_not_be_called(*_a, **_k):
        download_called["hit"] = True
        raise AssertionError("download must not run when guard rejects")

    monkeypatch.setattr(app_module, "_download_image", _should_not_be_called)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["reason"] == "already_reviewed"
    assert "OtherUser" in payload["error"]
    assert download_called["hit"] is False


def test_api_update_face_bbox_blocks_other_users_face_null_username_fallback(monkeypatch):
    """Pre-check: cross-user block still works when the previous classifier's
    wiki_username is NULL (e.g., user row deleted). Returns the generic
    fallback message instead of crashing."""
    face_row = _default_bbox_face_row(
        classified_by_user_id=2,
        prev_classifier_username=None,
    )
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["reason"] == "already_reviewed"
    assert payload["error"] == "This face has already been classified by another contributor."


def test_api_update_face_bbox_allows_self_edit(monkeypatch):
    """Members CAN edit a face they themselves classified."""
    face_row = _default_bbox_face_row(classified_by_user_id=1)  # same as logged-in user
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 999
        # In-tx re-check fetchone() must return the same owner so guard passes
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": 1,
            "wiki_username": "tester",
        }
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3003))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    assert resp.get_json()["new_face_id"] == 999


def test_api_update_face_bbox_allows_edit_of_unclassified_face(monkeypatch):
    """Members CAN edit a face whose classified_by_user_id is NULL
    (e.g., model-classified or bootstrap-classified faces)."""
    face_row = _default_bbox_face_row(classified_by_user_id=None)
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 777
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": None,
            "wiki_username": None,
        }
        return fn(mock_conn, mock_cursor)

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3004))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    assert resp.get_json()["new_face_id"] == 777


def test_api_update_face_bbox_intx_recheck_blocks_toctou_race(monkeypatch):
    """Defense-in-depth: pre-check passes (NULL classifier), but between the
    pre-check and the transaction another user classifies the face. The
    in-transaction re-check must catch this and return 409."""
    face_row = _default_bbox_face_row(classified_by_user_id=None)
    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=face_row))
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")

    def _transaction(fn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 0
        # Simulate race: another user (id=2) has classified between pre-check and now
        mock_cursor.fetchone.return_value = {
            "classified_by_user_id": 2,
            "wiki_username": "Racer",
        }
        # The function should raise ValueError before reaching INSERT
        try:
            fn(mock_conn, mock_cursor)
        except ValueError as exc:
            # _run_authorized_transaction wraps these in DatabaseError via
            # the cause chain — re-raise that wrapping to mimic real behavior
            raise app_module.DatabaseError("Transaction execution failed") from exc

    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", lambda *_a, **_k: _known_encoding(3005))

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 409
    payload = resp.get_json()
    assert payload["reason"] == "already_reviewed"
    assert "Racer" in payload["error"]


def test_api_manual_face_passes_stored_detection_space_and_stores_embedding_model(monkeypatch):
    seen = {}
    executed = []

    def _route_query(sql):
        if "FROM images i " in sql:
            return [
                {
                    "id": 10,
                    "file_title": "File:Face.jpg",
                    "status": "processed",
                    "detection_width": 960,
                    "detection_height": 720,
                }
            ]
        return []

    def _transaction(fn):
        cursor = MagicMock()
        cursor.lastrowid = 99
        cursor.rowcount = 1
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    def _encode(image_bytes, bbox, box_space=None):
        seen["bbox"], seen["box_space"] = bbox, box_space
        return _known_encoding(2101)

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", _encode)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 200
    # The drawn box is passed through untouched; the service rescales it from
    # the stored detection space onto whatever size was downloaded.
    assert seen["bbox"] == (10, 30, 40, 10)
    assert seen["box_space"] == (960, 720)

    inserts = [(sql, params) for sql, params in executed if sql.startswith("INSERT INTO faces")]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "embedding_model" in sql
    assert params[1] == _make_embedding(2101)
    assert params[2] == "scrfd500m+arcface_mbf"


def test_api_manual_face_omits_box_space_when_detection_size_unknown(monkeypatch):
    seen = {}

    def _route_query(sql):
        if "FROM images i " in sql:
            return [{"id": 10, "file_title": "File:Face.jpg", "status": "processed"}]
        return []

    def _encode(image_bytes, bbox, box_space=None):
        seen["box_space"] = box_space
        return None

    client, _ = _authed_client(monkeypatch, route_execute_query=_route_query)
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module.face_client, "encode_known_face", _encode)

    resp = client.post("/api/manual-face", data=_manual_face_form())

    assert resp.status_code == 422
    assert seen["box_space"] is None


def test_api_update_face_bbox_passes_detection_space_and_stores_embedding_model(monkeypatch):
    seen = {}
    executed = []
    row = _default_bbox_face_row()
    row["detection_width"] = 960
    row["detection_height"] = 640

    def _encode(image_bytes, bbox, box_space=None):
        seen["box_space"] = box_space
        return _known_encoding(3101)

    def _transaction(fn):
        cursor = MagicMock()
        cursor.lastrowid = 321
        cursor.fetchone.return_value = {"classified_by_user_id": None, "wiki_username": None}
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    client, _ = _authed_client(monkeypatch, route_execute_query=_bbox_query_router(face_row=row))
    monkeypatch.setattr(app_module, "_download_image", lambda *_a, **_k: b"image-bytes")
    monkeypatch.setattr(app_module, "execute_transaction", _transaction)
    monkeypatch.setattr(app_module.face_client, "encode_known_face", _encode)

    resp = client.post("/api/update-face-bbox", data=_update_bbox_form())

    assert resp.status_code == 200
    assert seen["box_space"] == (960, 640)
    inserts = [(sql, params) for sql, params in executed if sql.startswith("INSERT INTO faces")]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "embedding_model" in sql
    assert params[2] == "scrfd500m+arcface_mbf"
