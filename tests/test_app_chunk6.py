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

import pytest
from flask import abort


@pytest.fixture
def fake_user():
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _make_authed_client(monkeypatch, fake_user, route_execute=None):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if route_execute is not None:
            return route_execute(sql, params, fetch)
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client


def _set_csrf(client, token=None):
    token = token or "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes(client):
    with client.session_transaction() as sess:
        return list(sess.get("_flashes", []))


def test_api_write_sdc_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_write_sdc_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_write_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_pending_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_already_requested(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 1}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 5, "removal_cnt": 2}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["message"] == "already_requested"
    assert payload["pending"] == 5
    assert payload["removal_pending"] == 2


def test_api_write_sdc_no_pending_writes(monkeypatch, fake_user):
    updates = []

    def route_execute(sql, _params, fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 0, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            updates.append((sql, fetch))
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 0, "removal_pending": 0}
    assert updates == []


def test_api_write_sdc_successful_flag_set(monkeypatch, fake_user):
    update_calls = []

    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 3, "removal_cnt": 1}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            update_calls.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    fake_file = MagicMock()
    with patch("builtins.open", return_value=fake_file):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 3, "removal_pending": 1}
    assert len(update_calls) == 1


def test_api_write_sdc_update_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 7, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_wakeup_file_oserror_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 1, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    with patch("builtins.open", side_effect=OSError("nope")):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_api_sdc_status_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested, sdc_write_error FROM projects" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_sdc_status_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested, sdc_write_error FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 1, "sdc_write_error": None}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            return [{"written": 9, "pending": 2, "removal_pending": 1}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {
        "status": "ok",
        "written": 9,
        "pending": 2,
        "removal_pending": 1,
        "in_progress": True,
        "error": None,
    }


def test_api_sdc_status_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested, sdc_write_error FROM projects" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_sdc_status_counts_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT id, sdc_write_requested, sdc_write_error FROM projects" in sql:
            return [{"id": 1, "sdc_write_requested": 0, "sdc_write_error": "oops"}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_progress_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 404


def test_api_progress_active_with_pending_and_stats(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return [{"images_processed": 4, "images_total": 10, "status": "active"}]
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


def test_api_progress_completed_project(monkeypatch, fake_user):
    seen_pending_query = {"called": False}

    def route_execute(sql, _params, _fetch):
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return [{"images_processed": 10, "images_total": 10, "status": "completed"}]
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
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return [{"images_processed": 1, "images_total": 3, "status": "active"}]
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
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return [{"images_processed": 2, "images_total": 3, "status": "active"}]
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
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            return [{"images_processed": 0, "images_total": 8, "status": "active"}]
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
        if "SELECT images_processed, images_total, status FROM projects" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/progress/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def _project_settings_base_row():
    return {
        "id": 1,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "commons_category": "Douglas_Adams",
        "label": "Douglas Adams",
        "status": "active",
        "distance_threshold": 0.6,
        "min_confirmed": 5,
    }


def test_project_settings_get_renders_form(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Settings" in response.data


def test_project_settings_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 404


def test_project_settings_csrf_fail_on_post(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client, "expected")

    response = client.post("/project/1/settings", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_settings_invalid_distance_threshold_string(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "abc",
            "min_confirmed": "5",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Distance threshold must be a number." in response.data


def test_project_settings_invalid_distance_threshold_out_of_range(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "1.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Distance threshold must be between 0.1 and 1.0." in response.data


def test_project_settings_invalid_min_confirmed_string(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "abc",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Minimum confirmed must be a whole number." in response.data


def test_project_settings_invalid_min_confirmed_below_one(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "0",
            "status": "active",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Minimum confirmed must be at least 1." in response.data


def test_project_settings_invalid_status(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "5",
            "status": "invalid",
            "label": "X",
        },
    )

    assert response.status_code == 200
    assert b"Invalid status." in response.data


def test_project_settings_successful_update_redirects(monkeypatch, fake_user):
    project = _project_settings_base_row()
    updates = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            updates.append((params, fetch))
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.55",
            "min_confirmed": "6",
            "status": "paused",
            "label": "New Label",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1")
    assert len(updates) == 1


def test_project_settings_success_with_warning_for_min_confirmed(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 1, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "8",
            "status": "active",
            "label": "New Label",
        },
    )

    assert response.status_code == 302
    flashes = _flashes(client)
    assert ("success", "Settings updated.") in flashes
    assert any(cat == "warning" and "Autonomous inference will not run" in msg for cat, msg in flashes)


def test_project_settings_db_error_on_update(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "8",
            "status": "active",
            "label": "New Label",
        },
    )

    assert response.status_code == 200
    assert b"Failed to update settings." in response.data


def test_project_rerun_inference_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_rerun_inference_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 404


def test_project_rerun_inference_settings_unchanged_noop(monkeypatch, fake_user):
    calls = {"reset_sql_seen": False}

    def route_execute(sql, _params, _fetch):
        if "FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "UPDATE faces f" in sql:
            calls["reset_sql_seen"] = True
            return 4
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    flashes = _flashes(client)
    assert (
        "info",
        "Settings have not changed since the last inference run. No re-run needed.",
    ) in flashes
    assert calls["reset_sql_seen"] is False


def test_project_rerun_inference_successful_reset_with_affected_faces(monkeypatch, fake_user):
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 3}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 4
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    assert any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)
    flashes = _flashes(client)
    assert any(cat == "success" and "Reset 4 model-classified faces." in msg for cat, msg in flashes)
    assert any(cat == "info" and "10/7 human-confirmed" in msg for cat, msg in flashes)


def test_project_rerun_inference_null_last_inference_values_still_resets(monkeypatch, fake_user):
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": None,
                    "last_inference_min_confirmed": None,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 3, "by_bootstrap": 2}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 2
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    assert any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)
    flashes = _flashes(client)
    # Below threshold (3/5) + has bootstrap faces → warning + bootstrap tip
    assert any(cat == "warning" and "3/5 human-confirmed" in msg for cat, msg in flashes)
    assert any(cat == "info" and "2 bootstrapped" in msg for cat, msg in flashes)


def test_project_rerun_inference_no_faces_to_reset(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if (
            "last_inference_threshold" in sql
            and "last_inference_min_confirmed" in sql
            and "AS human_confirmed" not in sql
        ):
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 8, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    executed_sql = []

    def fake_transaction(fn):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_cursor.execute = lambda sql, params=None: executed_sql.append(sql)
        mock_conn = MagicMock()
        return fn(mock_conn, mock_cursor)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("info", "No model-classified faces to reset.") in _flashes(client)
    # Project update must not run when affected=0
    assert not any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)


def test_project_rerun_inference_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if (
            "last_inference_threshold" in sql
            and "last_inference_min_confirmed" in sql
            and "AS human_confirmed" not in sql
        ):
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 0, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(_fn):
        raise app_module.DatabaseError("boom")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to re-run inference.") in _flashes(client)


def test_project_delete_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/delete", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_delete_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert ("warning", "Project deleted.") in _flashes(client)


def test_project_delete_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Project not found.") in _flashes(client)


def test_project_delete_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to delete project.") in _flashes(client)


def test_leaderboard_success_with_rows(monkeypatch):
    rows = [
        {"wiki_username": "alice", "classifications": 7, "sdc_tags": 3},
        {"wiki_username": "bob", "classifications": 2, "sdc_tags": 1},
    ]

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"Leaderboard" in response.data
    assert b"alice" in response.data
    assert b">9<" in response.data
    assert b">4<" in response.data


def test_leaderboard_db_error(monkeypatch):
    def boom(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        raise app_module.DatabaseError("boom")

    monkeypatch.setattr(app_module, "execute_query", boom)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"Failed to load leaderboard." in response.data


def test_leaderboard_empty_results(monkeypatch):
    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"No contributions yet" in response.data


def test_health_healthy_db(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 1}])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy", "database": "connected"}


def test_health_unhealthy_db_exception(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(app_module, "execute_query", boom)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy", "error": "database unavailable"}


def test_health_unhealthy_empty_rows(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy"}


def test_health_unhealthy_wrong_value(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 0}])

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {"status": "unhealthy"}


def test_error_handler_400(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: abort(400))

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/")

    assert response.status_code == 400
    assert b"400" in response.data


def test_error_handler_404():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404
    assert b"Page not found" in response.data


def test_error_handler_429(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: abort(429))

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/")

    assert response.status_code == 429
    assert b"Rate limit exceeded" in response.data


def test_error_handler_500(monkeypatch):
    monkeypatch.setitem(flask_app.view_functions, "index", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    previous_testing = flask_app.config.get("TESTING", False)
    previous_propagate = flask_app.config.get("PROPAGATE_EXCEPTIONS")
    flask_app.config["TESTING"] = False
    flask_app.config["PROPAGATE_EXCEPTIONS"] = False
    client = flask_app.test_client()

    response = client.get("/")

    flask_app.config["TESTING"] = previous_testing
    flask_app.config["PROPAGATE_EXCEPTIONS"] = previous_propagate
    assert response.status_code == 500
    assert b"Internal server error" in response.data


def test_commons_thumb_route_default_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 330)


def test_commons_thumb_route_custom_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg?width=500")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 500)


def test_create_app_returns_app():
    assert app_module.create_app() is flask_app
