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

with patch("database.init_db"):
    import app as app_module

    flask_app = app_module.app


def _auth_client(monkeypatch, execute_query_impl=None):
    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    flask_app.config["TESTING"] = True
    monkeypatch.setattr(app_module.limiter, "_check_request_limit", lambda *a, **k: None)
    client = flask_app.test_client()

    if execute_query_impl is None:

        def execute_query_impl(sql, params=None, fetch=True):
            del params, fetch
            if "FROM users WHERE id = %s" in sql:
                return [fake_user]
            return ()

    monkeypatch.setattr(app_module, "execute_query", execute_query_impl)
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _set_csrf(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _capture_render_template(monkeypatch):
    captured = {}

    def fake_render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return app_module.jsonify({"template": template})

    monkeypatch.setattr(app_module, "render_template", fake_render)
    return captured


def test_project_detail_db_error_loading_project(monkeypatch):
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
        raise app_module.DatabaseError("boom")

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 500


def test_project_detail_project_not_found(monkeypatch):
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return ()
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 404


def test_project_detail_full_success_with_lazy_p18_update(monkeypatch):
    captured = _capture_render_template(monkeypatch)
    updates = []

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "active"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((sql, params, fetch))
            return 1
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 10, "confirmed_matches": 2}]
        if "FROM faces f JOIN images i" in sql and "LIMIT 200" in sql:
            return [{"id": 5, "image_id": 9, "is_target": 0, "classified_by": "model"}]
        if "status = 'pending'" in sql:
            return [{"cnt": 7}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 4}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"https://thumb/{qid}.jpg")
    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb/Q42.jpg"
    assert captured["context"]["stats"]["total_faces"] == 10
    assert captured["context"]["model_faces"][0]["id"] == 5
    assert captured["context"]["pending_images"] == 7
    assert captured["context"]["inference_eligible"] == 4
    assert len(updates) == 1


def test_project_detail_lazy_p18_not_updated_when_fetch_empty(monkeypatch):
    captured = _capture_render_template(monkeypatch)
    seen_update = {"called": False}

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "completed"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            seen_update["called"] = True
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] is None
    assert seen_update["called"] is False


def test_project_detail_lazy_p18_update_db_error_ignored(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "completed"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("nope")
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "https://thumb.jpg")
    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb.jpg"


def test_project_detail_face_stats_db_error_returns_empty_stats(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            raise app_module.DatabaseError("stats fail")
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["stats"] == {}


def test_project_detail_model_faces_db_error_returns_empty_list(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 3}]
        if "LIMIT 200" in sql:
            raise app_module.DatabaseError("gallery fail")
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["model_faces"] == []


def test_project_detail_pending_images_only_for_active(monkeypatch):
    captured = _capture_render_template(monkeypatch)
    pending_calls = {"count": 0}

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "status = 'pending'" in sql:
            pending_calls["count"] += 1
            return [{"cnt": 99}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert pending_calls["count"] == 0
    assert captured["context"]["pending_images"] == 0


def test_project_detail_pending_images_db_error_defaults_zero(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "status = 'pending'" in sql:
            raise app_module.DatabaseError("pending fail")
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 3}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["pending_images"] == 0


def test_project_detail_inference_eligible_db_error_defaults_zero(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0}]
        if "LIMIT 200" in sql:
            return []
        if "status = 'pending'" in sql:
            return [{"cnt": 2}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            raise app_module.DatabaseError("eligible fail")
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["inference_eligible"] == 0


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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return ()
        return ()

    client, _ = _auth_client(monkeypatch, eq)
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 500


def test_classify_skip_image_id_normal_mode_adds_session(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=77")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_1"] == [77]
    assert captured["context"]["skipped_count"] == 1


def test_classify_skip_image_id_review_mode_adds_review_session(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=88&skip_reviewing=1")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert sess["skipped_images_review_1"] == [88]
    assert captured["context"]["skipped_review_count"] == 1


def test_classify_skip_image_id_invalid_ignored(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify?skip_image_id=not-an-int")
    assert response.status_code == 200
    assert captured["context"]["skipped_count"] == 0


def test_classify_forced_image_review_mode_when_no_unclassified(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=44")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"][0]["face_id"] == 100


def test_classify_forced_image_normal_mode_with_unclassified(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify?image_id=45")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is False
    assert captured["context"]["faces"][0]["face_id"] == 101


def test_classify_skipped_ids_path_uses_not_in(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_1"] = [99]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 3


def test_classify_normal_no_skips_uses_random_choice(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 21


def test_classify_fallback_to_model_review_without_skipped_review(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["image"]["image_id"] == 55


def test_classify_fallback_with_skipped_review_ids(monkeypatch):
    captured = _capture_render_template(monkeypatch)
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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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

    client, _ = _auth_client(monkeypatch, eq)
    with client.session_transaction() as sess:
        sess["skipped_images_review_1"] = [5, 6]
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"]["image_id"] == 71


def test_classify_image_loading_db_error_results_no_image(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            raise app_module.DatabaseError("load fail")
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["image"] is None
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_normal(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["faces"] == []


def test_classify_faces_loading_db_error_review(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
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
    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["reviewing_model"] is True
    assert captured["context"]["faces"] == []


def test_classify_remaining_count_db_error(monkeypatch):
    captured = _capture_render_template(monkeypatch)

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
        if "SELECT * FROM projects WHERE id = %s" in sql:
            return [{"id": 1, "user_id": 1}]
        if "LIMIT 200" in sql:
            return []
        if "COUNT(DISTINCT i.id) AS cnt" in sql:
            raise app_module.DatabaseError("count fail")
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    response = client.get("/project/1/classify")
    assert response.status_code == 200
    assert captured["context"]["remaining"] == 0
    assert captured["context"]["model_review_count"] == 0


def test_clear_skips_csrf_fail(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client, token="abc")
    response = client.post("/project/1/classify/clear-skips", data={"csrf_token": "wrong"})
    assert response.status_code == 403


def test_clear_skips_clears_normal_and_review_keys(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client)
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
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client, token="abc")
    response = client.post("/api/classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_classify_missing_fields(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing required fields"


def test_api_classify_invalid_field_values(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return ()
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            raise app_module.DatabaseError("db")
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "abc", "project_id": "1", "image_id": "2"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid face ID"
    assert any("SELECT i.id FROM images i" in q[0] for q in queries)


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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 0, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 0}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        cursor.fetchone.return_value = {"bootstrapped": 1, "has_sibling_match": 1}
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}, {"id": 11}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "10", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 200
    assert any("UPDATE faces SET is_target = 1" in sql for sql, _ in executed)
    assert any("id != %s AND is_target IS NULL" in sql for sql, _ in executed)
    assert any("UPDATE projects SET faces_confirmed = faces_confirmed + 1" in sql for sql, _ in executed)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        cursor.fetchall.return_value = [{"id": 10}]
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
    assert any("AND is_target = 0 AND classified_by = 'model'" in sql for sql, _ in executed)
    with client.session_transaction() as sess:
        assert sess["last_classify"]["was_review"] is True


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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 22}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("tx fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
    response = client.post(
        "/api/classify",
        data={"csrf_token": "testtoken", "selected_face_id": "none", "project_id": "1", "image_id": "22"},
    )
    assert response.status_code == 500
    assert response.get_json()["error"] == "Failed to save classification"


def test_api_undo_classify_csrf_fail(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client, token="abc")
    response = client.post("/api/undo-classify", data={"csrf_token": "wrong"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_undo_classify_no_last_classify(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client)
    response = client.post("/api/undo-classify", data={"csrf_token": "testtoken"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nothing to undo"


def test_api_undo_classify_empty_face_ids_clears_session(monkeypatch):
    client, _ = _auth_client(monkeypatch)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return ()
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            raise app_module.DatabaseError("db")
        return ()

    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
    assert update_params == (10, 2)


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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.side_effect = lambda sql, params=None: executed.append((sql, params))
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(_fn):
        raise app_module.DatabaseError("undo fail")

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
        if "SELECT i.id FROM images i" in sql:
            return [{"id": 2}]
        return ()

    def tx(fn):
        cursor = MagicMock()
        cursor.execute.return_value = None
        return fn(MagicMock(), cursor)

    monkeypatch.setattr(app_module, "execute_transaction", tx)
    client, _ = _auth_client(monkeypatch, eq)
    _set_csrf(client)
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
    assert payload == {"status": "ok", "project_id": 9, "image_id": 77}
