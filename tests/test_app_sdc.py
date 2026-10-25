"""Tests for the web app: sdc."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tests._app_preamble import (
    app_module,
)
from tests._app_support import (
    _make_authed_client,
    _MockResponse,
    _set_csrf_chunk6,
)


@pytest.mark.integration
def test_api_sdc_status(integration_client, seed_project, seed_images, seed_faces):
    client, _, _ = integration_client

    response = client.get(f"/api/sdc-status/{seed_project['id']}")

    assert response.status_code == 200
    payload = response.get_json()
    assert "written" in payload
    assert "pending" in payload


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
    eq_mock = MagicMock(return_value=1)
    monkeypatch.setattr(app_module, "execute_query", eq_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 2
    assert post_mock.call_count == 1
    post_data = post_mock.call_args.kwargs["data"]
    assert post_data["action"] == "wbremoveclaims"
    assert post_data["claim"] == "M555$ABC"
    # The page/QID dedup row is dropped: a surviving written_at would make a
    # later write mark the face written without restoring the claim.
    eq_mock.assert_called_once()
    assert "DELETE FROM sdc_claims" in eq_mock.call_args.args[0]
    assert eq_mock.call_args.args[1] == (555, "Q42")


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
    eq_mock = MagicMock(return_value=1)
    monkeypatch.setattr(app_module, "execute_query", eq_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    assert get_mock.call_count == 1
    assert post_mock.call_count == 0
    # Claim already gone from Commons — the dedup row must go with it.
    eq_mock.assert_called_once()
    assert "DELETE FROM sdc_claims" in eq_mock.call_args.args[0]
    assert eq_mock.call_args.args[1] == (555, "Q42")


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
    eq_mock = MagicMock(return_value=1)
    monkeypatch.setattr(app_module, "execute_query", eq_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is True
    eq_mock.assert_called_once()
    assert "DELETE FROM sdc_claims" in eq_mock.call_args.args[0]
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
    eq_mock = MagicMock(return_value=1)
    monkeypatch.setattr(app_module, "execute_query", eq_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False
    # The claim is still on Commons: the dedup row must stay so the removal
    # is retried.
    eq_mock.assert_not_called()


def test_remove_sdc_claim_exception_during_api_calls(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        MagicMock(side_effect=requests.RequestException("network")),
    )
    eq_mock = MagicMock(return_value=1)
    monkeypatch.setattr(app_module, "execute_query", eq_mock)

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False
    eq_mock.assert_not_called()


def test_remove_sdc_claim_returns_false_when_dedup_row_cannot_be_cleared(monkeypatch):
    """A removal whose dedup row cannot be deleted is incomplete.

    Returning True would let a later send trust the surviving written_at and
    mark the face written without restoring the claim on Commons.
    """
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

    monkeypatch.setattr(app_module.requests, "get", MagicMock(side_effect=[claim_resp, token_resp]))
    monkeypatch.setattr(app_module.requests, "post", MagicMock(return_value=remove_resp))
    monkeypatch.setattr(
        app_module,
        "execute_query",
        MagicMock(side_effect=app_module.DatabaseError("pool down")),
    )

    ok = app_module._remove_sdc_claim(555, "Q42", "access-token")

    assert ok is False


def test_api_write_sdc_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_write_sdc_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_write_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_pending_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_already_requested(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 5, "removal_cnt": 2}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

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
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 0, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            updates.append((sql, fetch))
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 0, "removal_pending": 0}
    assert updates == []


def test_api_write_sdc_successful_flag_set(monkeypatch, fake_user):
    update_calls = []

    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 3, "removal_cnt": 1}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            update_calls.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    fake_file = MagicMock()
    with patch("builtins.open", return_value=fake_file):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "pending": 3, "removal_pending": 1}
    assert len(update_calls) == 1


def test_api_write_sdc_update_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 7, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_write_sdc_wakeup_file_oserror_ignored(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        if "AS write_cnt" in sql and "AS removal_cnt" in sql:
            return [{"write_cnt": 1, "removal_cnt": 0}]
        if "UPDATE projects SET sdc_write_requested = 1" in sql:
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    with patch("builtins.open", side_effect=OSError("nope")):
        response = client.post("/api/write-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_api_sdc_status_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_sdc_status_success(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1, "sdc_write_error": None}]
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
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_sdc_status_counts_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0, "sdc_write_error": "oops"}]
        if "AS written" in sql and "AS pending" in sql and "AS removal_pending" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/api/sdc-status/1")

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_stop_sdc_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid CSRF token"


def test_api_stop_sdc_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/123", data={"csrf_token": "testtoken"})

    assert response.status_code == 404
    assert "Project not found" in response.get_json()["error"]


def test_api_stop_sdc_project_query_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_api_stop_sdc_not_in_progress(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["message"] == "not_in_progress"


def test_api_stop_sdc_success(monkeypatch, fake_user):
    updates = []

    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            updates.append(sql)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "message" not in payload
    assert len(updates) == 1
    assert "sdc_write_error = NULL" in updates[0]


def test_api_stop_sdc_update_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active", "sdc_write_requested": 1}]
        if "UPDATE projects SET sdc_write_requested = 0" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/api/stop-sdc/1", data={"csrf_token": "testtoken"})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Database error"


def test_check_p180_exists_true_when_claim_matches(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse({"claims": {"P180": [{"mainsnak": {"datavalue": {"value": {"id": "Q42"}}}}]}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is True


def test_check_p180_exists_false_when_different_qid(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse({"claims": {"P180": [{"mainsnak": {"datavalue": {"value": {"id": "Q99"}}}}]}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_when_no_p180_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {}}))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_true_among_multiple_claims(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse(
            {
                "claims": {
                    "P180": [
                        {"mainsnak": {"datavalue": {"value": {"id": "Q99"}}}},
                        {"mainsnak": {"datavalue": {"value": {"id": "Q42"}}}},
                    ]
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._check_p180_exists(555, "Q42") is True


def test_check_p180_exists_false_on_missing_nested_keys(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P180": [{}]}}))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_false_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._check_p180_exists(555, "Q42") is False


def test_check_p180_exists_calls_commons_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P180": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._check_p180_exists(777, "Q76")

    assert captured["url"] == app_module.COMMONS_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "M777"
    assert captured["params"]["property"] == "P180"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10
