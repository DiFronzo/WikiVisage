"""Tests for the web app: project pages."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from database import DatabaseError
from tests._app_preamble import (
    _make_embedding,
    app_module,
    flask_app,
)
from tests._app_support import (
    _auth_client_chunk4,
    _authed_client,
    _capture_render,
    _capture_render_template_chunk4,
    _empty_prefilter_stats_row,
    _fake_user_chunk3,
    _flashes_chunk6,
    _liftwing_status_row,
    _make_authed_client,
    _make_authenticated_client_chunk3,
    _prefilter_stats_row,
    _project_settings_base_row,
    _set_csrf_chunk4,
    _set_csrf_chunk6,
)


def _user_aware_execute(fake_user, route_handler):
    def _execute_query(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return route_handler(sql, params, fetch)

    return _execute_query


def _set_csrf_chunk3(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes_chunk3(client):
    with client.session_transaction() as sess:
        return sess.get("_flashes", [])


class _FakeResponse_chunk3:
    def __init__(self, payload=None, raise_exc=None):
        self.payload = {} if payload is None else payload
        self.raise_exc = raise_exc

    def raise_for_status(self):
        if self.raise_exc:
            raise self.raise_exc

    def json(self):
        return self.payload


def _project_detail_banner_eq(project_overrides=None, stats_overrides=None, pending=0, eligible=0):
    """execute_query mock for real-render /project/<id> inference-banner tests."""
    project_row = {
        "id": 1,
        "user_id": 1,
        "wikidata_qid": "Q42",
        "p18_thumb_url": "https://thumb.jpg",
        "status": "active",
        "min_confirmed": 5,
        "images_total": 10,
        "images_processed": 10,
        "completion_reason": None,
    }
    project_row.update(project_overrides or {})
    stats_row = {
        "total_faces": 10,
        "confirmed_matches": 6,
        "confirmed_non_matches": 2,
        "unclassified": 2,
        "human_confirmed": 5,
        "sdc_written": 0,
        "sdc_pending": 1,
        "sdc_removal_pending_faces": 0,
        "sdc_removal_pending": 0,
        "by_human": 5,
        "by_model": 1,
        "by_bootstrap": 0,
    }
    stats_row.update(stats_overrides or {})

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
            return [project_row]
        if "COUNT(*) AS total_faces" in sql:
            return [stats_row]
        if "status = 'pending'" in sql:
            return [{"cnt": pending}]
        if "f.is_target IS NULL" in sql:
            return [{"cnt": eligible}]
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        if "LIMIT 200" in sql:
            return []
        return ()

    return eq


def _render_settings_with(monkeypatch, fake_user, **overrides):
    project = _project_settings_base_row()
    project.update(overrides)

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    return client.get("/project/1/settings").get_data(as_text=True)


def _fake_rerun_transaction(
    fn,
    *,
    status="completed",
    sdc_write_requested=0,
    min_confirmed=5,
    human_confirmed=10,
    by_bootstrap=0,
    rows_reset=2,
    executed_sql=None,
    stats_error=False,
):
    """Run a rerun transaction against a scripted cursor.

    ``_assert_project_access_locked`` consumes the first ``fetchone``; the
    project-row lock consumes the second and the gate stats the third.
    """
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        {"access": 1},
        {
            "status": status,
            "sdc_write_requested": sdc_write_requested,
            "min_confirmed": min_confirmed,
        },
        {"human_confirmed": human_confirmed, "by_bootstrap": by_bootstrap},
    ]
    cursor.rowcount = rows_reset

    def _execute(sql, params=None):
        if executed_sql is not None:
            executed_sql.append(sql)
        if stats_error and "AS human_confirmed" in sql:
            raise DatabaseError("stats query failed")

    cursor.execute = _execute
    return fn(MagicMock(), cursor)


def _flashes_tail(client):
    with client.session_transaction() as sess:
        return sess.get("_flashes", [])


def _make_project_new_base_execute(insert_side_effect):
    """Return an execute_query stub for /project/new that calls insert_side_effect on INSERT."""

    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return insert_side_effect()
        raise AssertionError(sql)

    return _execute_query


def _threshold_form_client(monkeypatch):
    return _authed_client(monkeypatch, route_execute_query=lambda sql: [])


def _prefilter_stats_client(monkeypatch, fake_user, project, stats_row):
    def route_execute(sql, _params=None, _fetch=True):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        if "prefilter_checked_at" in sql:
            if isinstance(stats_row, Exception):
                raise stats_row
            return [stats_row]
        raise AssertionError(f"Unexpected SQL: {sql}")

    return _make_authed_client(monkeypatch, fake_user, route_execute)


@pytest.mark.integration
def test_dashboard_empty(integration_client):
    client, _, _ = integration_client

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"WikiVisage" in response.data


@pytest.mark.integration
def test_project_detail_with_data(integration_client, seed_project, seed_images, seed_faces):
    client, _, _ = integration_client

    response = client.get(f"/project/{seed_project['id']}")

    assert response.status_code == 200
    assert b"Douglas Adams" in response.data or b"Q42" in response.data


def test_dashboard_defaults_page_and_totals(monkeypatch):
    captured = _capture_render(monkeypatch)
    calls = []
    fake_user = _fake_user_chunk3()

    def _handler(sql, params, fetch):
        calls.append((sql, params, fetch))
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["template"] == "dashboard.html"
    assert captured["context"]["page"] == 1
    assert captured["context"]["total_pages"] == 1
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["projects"] == []
    assert any("COUNT(DISTINCT p.id) AS cnt" in sql for sql, _params, _fetch in calls)
    assert any("SELECT DISTINCT p.*" in sql for sql, _params, _fetch in calls)


def test_dashboard_invalid_page_falls_back_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=abc")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_negative_page_clamped_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=-5")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_page_clamped_to_total_pages(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    select_params = {}

    def _handler(sql, params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 30}]
        if "SELECT DISTINCT p.*" in sql:
            select_params["params"] = params
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=99")

    assert response.status_code == 200
    assert captured["context"]["page"] == 2
    assert captured["context"]["total_pages"] == 2
    assert select_params["params"][2] == 25


def test_dashboard_count_database_error_sets_total_zero(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            raise app_module.DatabaseError("count failed")
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=3")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["page"] == 1


def test_dashboard_projects_database_error_flashes_and_uses_empty(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 10}]
        if "SELECT DISTINCT p.*" in sql:
            raise app_module.DatabaseError("projects failed")
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["projects"] == []
    assert any("Failed to load projects." in msg for _cat, msg in _flashes_chunk3(client))


def test_dashboard_count_row_empty_tuple(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return ()
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0


def test_dashboard_lazily_populates_missing_p18_and_updates_db(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    updates = []

    projects = [
        {"id": 101, "wikidata_qid": "Q42", "p18_thumb_url": None},
        {"id": 102, "wikidata_qid": "Q1", "p18_thumb_url": "existing"},
        {"id": 103, "wikidata_qid": None, "p18_thumb_url": None},
    ]

    def _handler(sql, params, fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 3}]
        if "SELECT DISTINCT p.*" in sql:
            return projects
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((params, fetch))
            return 1
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"thumb-{qid}")
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert projects[0]["p18_thumb_url"] == "thumb-Q42"
    assert len(updates) == 1
    assert updates[0][0] == ("thumb-Q42", 101)
    assert updates[0][1] is False


def test_dashboard_missing_thumb_but_fetch_returns_none_no_update(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()
    update_called = {"called": False}

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            update_called["called"] = True
            return 1
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert update_called["called"] is False
    assert captured["template"] == "dashboard.html"


def test_dashboard_p18_update_database_error_is_non_critical(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("update failed")
        if "project_members" in sql and "COUNT" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200


def test_dashboard_projects_tuple_skips_lazy_population(monkeypatch):
    fake_user = _fake_user_chunk3()
    fetch_calls = {"count": 0}

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return ()
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: fetch_calls.update(count=1))
    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert fetch_calls["count"] == 0


def test_api_category_info_missing_category_returns_400(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/api/category-info")

    assert response.status_code == 400
    assert response.get_json() == {"error": "missing category"}


@pytest.mark.parametrize(
    "category",
    [
        "Bad|Name",
        "Bad\nName",
        "x" * 201,
    ],
)
def test_api_category_info_invalid_category_name_returns_400(monkeypatch, category):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/api/category-info", query_string={"category": category})

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid category name"}


def test_api_category_info_root_category_not_found_returns_404(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse_chunk3({"query": {"pages": {"-1": {"missing": True}}}}),
    )

    response = client.get("/api/category-info?category=People")

    assert response.status_code == 404
    assert response.get_json() == {"error": "not_found"}


def test_api_category_info_root_no_subcats_returns_direct_counts(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(*_args, **_kwargs):
        return _FakeResponse_chunk3(
            {
                "query": {
                    "pages": {
                        "123": {
                            "title": "Category:People",
                            "categoryinfo": {"files": 7, "subcats": 0},
                        }
                    }
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=People")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 7
    assert body["subcats"] == 0
    assert body["categories_visited"] == 1
    assert body["approximate"] is False  # a strict bool: the page tests it for truthiness


def test_api_category_info_bfs_traversal_sums_subcategories(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 3, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 4, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:B"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:B":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "3": {
                                "title": "Category:B",
                                "categoryinfo": {"files": 6, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 13
    assert body["subcats"] == 2
    assert body["categories_visited"] == 3
    assert body["approximate"] is False  # a strict bool: the page tests it for truthiness


def test_api_category_info_sets_approximate_on_continue_from_root_subcats(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 2, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers":
            return _FakeResponse_chunk3(
                {
                    "query": {"categorymembers": [{"title": "Category:A"}]},
                    "continue": {"cmcontinue": "next"},
                }
            )
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 3, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_api_category_info_hits_max_categories_marks_approximate(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    many = [{"title": f"Category:C{i}"} for i in range(60)]

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 1, "subcats": len(many)},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": many}})
        if params.get("prop") == "categoryinfo" and "Category:C" in params.get("titles", ""):
            pages = {}
            for i, title in enumerate(params["titles"].split("|"), start=100):
                pages[str(i)] = {"title": title, "categoryinfo": {"files": 1, "subcats": 0}}
            return _FakeResponse_chunk3({"query": {"pages": pages}})
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["approximate"] is True
    assert body["categories_visited"] == 50


def test_api_category_info_timeout_marks_approximate(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 5, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        raise AssertionError(params)

    calls = {"n": 0}

    def _monotonic():
        calls["n"] += 1
        if calls["n"] == 1:
            return 0
        return 9

    monkeypatch.setattr(app_module.requests, "get", _get)
    monkeypatch.setattr(app_module.time, "monotonic", _monotonic)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_api_category_info_request_exception_returns_502(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *a, **kw: (_ for _ in ()).throw(requests.RequestException("boom"))
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_api_category_info_http_error_from_raise_for_status_returns_502(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse_chunk3(raise_exc=requests.HTTPError("bad status")),
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_project_new_get_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)

    response = client.get("/project/new")

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"


def test_project_new_post_invalid_csrf_returns_400(monkeypatch):
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client, "session-token")

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "bad-token",
            "wikidata_qid": "Q42",
            "commons_category": "People",
        },
    )

    assert response.status_code == 400


def test_project_new_post_missing_qid_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "commons_category": "People"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_qid_without_q_prefix_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_missing_category_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Commons category is required." in msg for _cat, msg in _flashes_chunk3(client))


@pytest.mark.parametrize("value", ["abc", "0.09", "1.1"])
def test_project_new_post_invalid_distance_threshold(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "distance_threshold": value,
        },
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Distance threshold" in msg for msg in messages)


@pytest.mark.parametrize("value", ["abc", "0", "-5"])
def test_project_new_post_invalid_min_confirmed(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "min_confirmed": value,
        },
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Minimum confirmed" in msg for msg in messages)


def test_project_new_post_multiple_validation_errors_are_all_flashed(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "", "commons_category": "", "distance_threshold": "oops"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes_chunk3(client)]
    assert any("Wikidata Q-ID must start" in msg for msg in messages)
    assert any("Commons category is required." in msg for msg in messages)
    assert any("Distance threshold must be a number." in msg for msg in messages)


def test_project_new_validation_short_circuits_remote_checks_on_basic_errors(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    is_human_called = {"called": False}
    category_called = {"called": False}
    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: is_human_called.update(called=True))
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _c: category_called.update(called=True))

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "bad", "commons_category": ""},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert is_human_called["called"] is False
    assert category_called["called"] is False


def test_project_new_post_non_human_qid_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: False)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q123", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("is not an instance of human" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_missing_commons_category_on_wikimedia_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: False)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "NotReal"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("does not exist" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_empty_commons_category_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client_chunk3(monkeypatch)
    _set_csrf_chunk3(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: False)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "EmptyCat"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("exists but contains no files" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_post_duplicate_project_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return [{"id": 99}]
        if "INSERT INTO projects" in sql:
            raise AssertionError("Should not insert duplicates")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People", "label": "My Label"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("already exists" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_duplicate_check_db_error_continues_to_create(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            raise app_module.DatabaseError("duplicate check failed")
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb-q42")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Douglas Adams")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert len(insert_calls) == 1


def test_project_new_join_while_banned_creates_own_project(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql and "user_id = %s" in sql:
            return ()
        if "u.wiki_username" in sql:
            return [{"id": 42, "label": "Existing Project", "wiki_username": "OtherUser"}]
        if "SELECT status FROM project_members" in sql:
            return [{"status": "banned"}]
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT" in sql:
            insert_calls.append(params)
            return 1
        raise AssertionError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert len(insert_calls) == 1


def test_project_new_successful_creation_with_label_fetch_and_p18(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return ()
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "https://thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Douglas Adams"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": " q42 ", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert label_calls["n"] == 1
    assert insert_calls[0][0][1] == "Q42"
    assert insert_calls[0][0][3] == "Douglas Adams"
    assert insert_calls[0][0][6] == "https://thumb"
    assert insert_calls[0][1] is False
    assert any("Project created successfully!" in msg for _cat, msg in _flashes_chunk3(client))


def test_project_new_successful_creation_with_user_label_skips_wikidata_label_fetch(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Should Not Be Used"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "People",
            "label": "Custom Label",
        },
    )

    assert response.status_code == 302
    assert label_calls["n"] == 0
    assert insert_calls[0][0][3] == "Custom Label"


def test_project_new_successful_creation_with_missing_p18_thumb(monkeypatch):
    fake_user = _fake_user_chunk3()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert insert_calls[0][0][6] is None


def test_project_new_wake_file_oserror_is_non_critical(monkeypatch):
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    def _raise_open(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise_open)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_project_new_db_error_on_insert_renders_form_with_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise app_module.DatabaseError("insert failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf_chunk3(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project. Please try again." in msg for _cat, msg in _flashes_chunk3(client))


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

    client, _ = _auth_client_chunk4(monkeypatch, eq)
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
        if "FROM projects p LEFT JOIN project_members" in sql:
            return ()
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 404


def test_project_detail_full_success_with_lazy_p18_update(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
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
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None, "status": "active"}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((sql, params, fetch))
            return 1
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 10, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 5}]
        if "status = 'pending'" in sql:
            return [{"cnt": 7}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            # Faces with a queued P180 removal are not inference candidates.
            assert "AND f.sdc_removal_pending = 0" in sql
            return [{"cnt": 4}]
        return ()

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"https://thumb/{qid}.jpg")
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb/Q42.jpg"
    assert captured["context"]["stats"]["total_faces"] == 10
    assert captured["context"]["gallery_total"] == 5
    assert captured["context"]["pending_images"] == 7
    assert captured["context"]["inference_eligible"] == 4
    assert len(updates) == 1


def test_project_detail_lazy_p18_not_updated_when_fetch_empty(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
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
        if "FROM projects p LEFT JOIN project_members" in sql:
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
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] is None
    assert seen_update["called"] is False


def test_project_detail_lazy_p18_update_db_error_ignored(monkeypatch):
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
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["project"]["p18_thumb_url"] == "https://thumb.jpg"


def test_project_detail_face_stats_db_error_returns_empty_stats(monkeypatch):
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
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            raise app_module.DatabaseError("stats fail")
        if "LIMIT 200" in sql:
            return []
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["stats"] == {}


def test_project_detail_gallery_count_db_error_returns_zero(monkeypatch):
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
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q1", "p18_thumb_url": "x", "status": "completed"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 3}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            raise app_module.DatabaseError("gallery fail")
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["gallery_total"] == 0


def test_project_detail_pending_images_only_for_active(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)
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
        if "FROM projects p LEFT JOIN project_members" in sql:
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

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert pending_calls["count"] == 0
    assert captured["context"]["pending_images"] == 0


def test_project_detail_pending_images_db_error_defaults_zero(monkeypatch):
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

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["pending_images"] == 0


def test_project_detail_inference_eligible_db_error_defaults_zero(monkeypatch):
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

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    assert captured["context"]["inference_eligible"] == 0


_PERMISSION_DENIED_MESSAGE = (
    "Your account lacks edit permissions. Please re-register the OAuth consumer with the 'Edit existing pages' grant."
)


def _project_detail_with_worker_error(monkeypatch, error):
    eq = _project_detail_banner_eq(project_overrides={"sdc_write_error": error, "sdc_write_requested": 0})
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")
    assert response.status_code == 200
    return response.get_data(as_text=True)


def test_project_detail_offers_reauthorize_for_a_stored_authorization_error(monkeypatch):
    """The status poller only runs during a write, so the stored error itself needs the button."""
    from config import SDC_AUTH_EXPIRED_MESSAGE

    body = _project_detail_with_worker_error(monkeypatch, SDC_AUTH_EXPIRED_MESSAGE)

    assert "Worker error: " in body
    assert 'href="/login?reauth=1&amp;next=/project/1"' in body


@pytest.mark.parametrize(
    "error",
    [None, "The Wikimedia servers are busy. Please try again later.", _PERMISSION_DENIED_MESSAGE],
)
def test_sdc_result_area_is_closed_before_the_action_buttons(monkeypatch, error):
    """A missing </div> once nested the buttons inside #sdcResult, which sdcShowMessage empties."""
    from html.parser import HTMLParser

    class Nesting(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.inside_result = {}

        def handle_starttag(self, tag, attrs):
            attr = dict(attrs)
            if tag not in ("br", "img", "input", "meta", "link", "hr"):
                self.stack.append((tag, attr.get("id")))
            if attr.get("id") in ("sdcWriteBtn", "sdcStopBtn"):
                self.inside_result[attr["id"]] = any(i == "sdcResult" for _t, i in self.stack[:-1])

        def handle_endtag(self, tag):
            for index in range(len(self.stack) - 1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]
                    break

    parser = Nesting()
    parser.feed(_project_detail_with_worker_error(monkeypatch, error))

    assert parser.inside_result == {"sdcWriteBtn": False, "sdcStopBtn": False}


@pytest.mark.parametrize(
    "error",
    ["The Wikimedia servers are busy. Please try again later.", _PERMISSION_DENIED_MESSAGE],
)
def test_project_detail_offers_no_reauthorize_for_other_worker_errors(monkeypatch, error):
    body = _project_detail_with_worker_error(monkeypatch, error)

    assert "Worker error: " in body
    # The page-wide REAUTH_URL constant always mentions the route; only a rendered anchor
    # for this error would offer the button.
    assert 'href="/login?reauth=1' not in body


def test_project_detail_inference_banner_hidden_while_processing(monkeypatch):
    """Queued image work with no eligible faces → banner exists hidden so the poller can reveal it."""
    eq = _project_detail_banner_eq(
        project_overrides={"images_processed": 3, "images_total": 10},
        pending=4,
        eligible=0,
    )
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1").get_data(as_text=True)

    assert 'id="inferenceBanner" role="status" aria-live="polite" style="display:none"' in body
    assert "pollInference" in body
    # Regression pins for the poller's stop/finalize rules (Copilot review, PR #118):
    # a removed banner must stop the timer, and the observed-work marker must persist.
    assert "if (!document.getElementById('inferenceBanner')) {" in body
    assert "var sawWork" in body


def test_project_detail_inference_banner_absent_when_idle(monkeypatch):
    """Processing finished and no eligible faces → no banner and no polling at all."""
    eq = _project_detail_banner_eq()
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1").get_data(as_text=True)

    assert 'id="inferenceBanner"' not in body
    assert "pollInference" not in body


def test_project_detail_inference_banner_visible_while_running(monkeypatch):
    """Eligible faces at load → visible banner, and the poller keeps the removal count consistent."""
    eq = _project_detail_banner_eq(eligible=7)
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1").get_data(as_text=True)

    assert '<div class="card inference-banner" id="inferenceBanner" role="status" aria-live="polite">' in body
    assert "Model inference in progress — 7 faces remaining…" in body
    assert "(fs.sdc_pending || 0) + (fs.sdc_removal_pending_faces || 0)" in body


def test_project_detail_inference_banner_renders_for_rerun_done_state(monkeypatch):
    """?inference_triggered=1 with nothing to classify still shows the completion banner."""
    eq = _project_detail_banner_eq()
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1?inference_triggered=1").get_data(as_text=True)

    assert "inference-banner-done" in body
    assert "Model inference complete." in body


def test_project_detail_inference_completion_counts_only_model_matches(monkeypatch):
    """The completion note counts new matches; ``by_model`` also counts non-matches."""
    eq = _project_detail_banner_eq(stats_overrides={"model_matches": 3, "by_model": 4})
    client, _ = _auth_client_chunk4(monkeypatch, eq)
    body = client.get("/project/1?inference_triggered=1").get_data(as_text=True)

    assert "var initialModelMatches = 3;" in body
    assert "var newMatches = Math.max(0, (fs.model_matches || 0) - initialModelMatches);" in body


def test_project_settings_get_renders_form(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Settings" in response.data


def test_project_settings_disables_rerun_while_paused(monkeypatch, fake_user):
    body = _render_settings_with(monkeypatch, fake_user, status="paused")

    assert "disabled>Re-run Inference</button>" in body
    assert "This project is paused. Set it to Active or Completed before re-running inference." in body


def test_project_settings_disables_rerun_during_sdc_write(monkeypatch, fake_user):
    body = _render_settings_with(monkeypatch, fake_user, sdc_write_requested=1)

    assert "disabled>Re-run Inference</button>" in body
    assert "A Commons edit run is in progress. Wait for it to finish (or stop it) before re-running inference." in body


def test_project_settings_keeps_rerun_enabled_when_runnable(monkeypatch, fake_user):
    body = _render_settings_with(monkeypatch, fake_user, status="completed", sdc_write_requested=0)

    assert "disabled>Re-run Inference</button>" not in body


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
    _set_csrf_chunk6(client, "expected")

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
    _set_csrf_chunk6(client)

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
    _set_csrf_chunk6(client)

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
    _set_csrf_chunk6(client)

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
    _set_csrf_chunk6(client)

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
    _set_csrf_chunk6(client)

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
    _set_csrf_chunk6(client)

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
    assert response.headers["Location"].endswith("/project/1/settings")
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
    _set_csrf_chunk6(client)

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
    flashes = _flashes_chunk6(client)
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
    _set_csrf_chunk6(client)

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


def test_project_settings_completed_status_preserves_completion_reason(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["status"] = "completed"
    sqls_seen = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            sqls_seen.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "completed",
            "label": "My Project",
        },
    )

    assert response.status_code == 302
    assert len(sqls_seen) == 1
    # Must NOT include completion_reason = NULL when keeping completed status
    assert "completion_reason" not in sqls_seen[0]


def test_project_settings_reactivation_clears_completion_reason(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["status"] = "completed"
    sqls_seen = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            sqls_seen.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "My Project",
        },
    )

    assert response.status_code == 302
    assert len(sqls_seen) == 1
    # MUST include completion_reason = NULL when reactivating
    assert "completion_reason" in sqls_seen[0]


def test_project_rerun_inference_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "wrong"})

    assert response.status_code == 400
    assert b"Invalid CSRF token" in response.data


def test_project_rerun_inference_project_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 404


def test_project_rerun_inference_runs_even_when_settings_match(monkeypatch, fake_user):
    """The button must not no-op on an unreliable 'last settings' marker.

    A background inference pass records the current threshold/min_confirmed even
    when it only classified new faces, so the marker cannot prove that the
    existing model results used those settings. Clicking the button is an
    explicit request to reset and reclassify.
    """
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "status": "completed",
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(fn, rows_reset=2, executed_sql=executed_sql),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "inference_triggered=1" in response.headers["Location"]
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    flashes = _flashes_chunk6(client)
    assert any(cat == "success" and "Reset 2 model-classified faces." in msg for cat, msg in flashes)


def test_project_rerun_inference_successful_reset_with_affected_faces(monkeypatch, fake_user):
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(
            fn,
            min_confirmed=7,
            human_confirmed=10,
            by_bootstrap=3,
            rows_reset=4,
            executed_sql=executed_sql,
        ),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any("UPDATE faces f" in sql for sql in executed_sql)
    assert any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)
    flashes = _flashes_chunk6(client)
    assert any(cat == "success" and "Reset 4 model-classified faces." in msg for cat, msg in flashes)
    assert any(cat == "info" and "10/7 human-confirmed" in msg for cat, msg in flashes)


def test_project_rerun_inference_below_threshold_does_not_reset(monkeypatch, fake_user):
    """Below the human-confirmed gate, refuse instead of wiping model results
    that the worker would not be allowed to recompute."""
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "status": "completed",
                    "distance_threshold": 0.6,
                    "min_confirmed": 5,
                    "last_inference_threshold": None,
                    "last_inference_min_confirmed": None,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(
            fn,
            human_confirmed=3,
            by_bootstrap=2,
            rows_reset=0,
            executed_sql=executed_sql,
        ),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert not any("UPDATE faces f" in sql for sql in executed_sql)
    flashes = _flashes_chunk6(client)
    # Below threshold (3/5) + has bootstrap faces → warning + bootstrap tip,
    # and explicitly no reset.
    assert any(cat == "warning" and "3/5 human-confirmed" in msg for cat, msg in flashes)
    assert any(cat == "info" and "2 bootstrapped" in msg for cat, msg in flashes)
    assert ("info", "No faces were reset.") in flashes
    assert not any(cat == "success" for cat, _msg in flashes)


def test_project_rerun_inference_paused_project_does_not_reset(monkeypatch, fake_user):
    """Inference never runs for paused projects, so the reset would be a dead end."""
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "status": "paused",
                    "distance_threshold": 0.55,
                    "min_confirmed": 5,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(fn, status="paused", rows_reset=0, executed_sql=executed_sql),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert not any("UPDATE faces f" in sql for sql in executed_sql)
    assert (
        "warning",
        "This project is paused. Set it to Active or Completed before re-running inference.",
    ) in _flashes_chunk6(client)


def test_project_rerun_inference_refuses_during_sdc_write(monkeypatch, fake_user):
    """A reset during a Commons write can strand a P180 claim the worker adds."""
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [{"id": 1, "status": "completed", "distance_threshold": 0.6, "min_confirmed": 5}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(fn, sdc_write_requested=1, rows_reset=0, executed_sql=executed_sql),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert not any("UPDATE faces f" in sql for sql in executed_sql)
    assert (
        "warning",
        "A Commons edit run is in progress. Wait for it to finish (or stop it) before re-running inference.",
    ) in _flashes_chunk6(client)


def test_project_rerun_inference_no_faces_to_reset(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    executed_sql = []

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(
            fn, min_confirmed=7, human_confirmed=8, rows_reset=0, executed_sql=executed_sql
        ),
    )
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert "inference_triggered=1" in response.headers["Location"]
    assert ("info", "No model-classified faces to reset.") in _flashes_chunk6(client)
    # Project update must not run when affected=0
    assert not any("UPDATE projects SET last_inference_threshold = NULL" in sql for sql in executed_sql)


def test_project_rerun_inference_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "status": "completed",
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fake_transaction(_fn):
        raise app_module.DatabaseError("boom")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    monkeypatch.setattr(app_module, "execute_transaction", fake_transaction)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to re-run inference.") in _flashes_chunk6(client)


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
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert ("warning", "Project deleted.") in _flashes_chunk6(client)


def test_project_delete_not_found(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            return 0
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Project not found.") in _flashes_chunk6(client)


def test_project_delete_db_error(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "UPDATE projects SET status" in sql and "deleted" in sql:
            raise app_module.DatabaseError("boom")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post("/project/1/delete", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert ("error", "Failed to delete project.") in _flashes_chunk6(client)


def test_leaderboard_success_with_rows(monkeypatch):
    rows = [
        {"wiki_username": "alice", "classifications": 7, "sdc_tags": 3},
        {"wiki_username": "bob", "classifications": 2, "sdc_tags": 1},
    ]

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
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
            return [{"fresh": 1, "stopping": 0, "total": 1}]
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
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"No contributions yet" in response.data


def test_leaderboard_period_month(monkeypatch):
    """Month filter uses time-filtered SQL without user_stats."""
    captured = {}
    rows = [{"wiki_username": "alice", "classifications": 3, "sdc_tags": 1}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=month")

    assert response.status_code == 200
    assert b"alice" in response.data
    assert "classified_at" in captured["sql"]
    assert "user_stats" not in captured["sql"]
    assert captured["params"] is not None
    assert len(captured["params"]) == 2


def test_leaderboard_period_daily(monkeypatch):
    """Daily filter uses time-filtered SQL without user_stats."""
    captured = {}
    rows = [{"wiki_username": "bob", "classifications": 1, "sdc_tags": 0}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=daily")

    assert response.status_code == 200
    assert b"bob" in response.data
    assert "classified_at" in captured["sql"]
    assert captured["params"] is not None
    assert len(captured["params"]) == 2


def test_leaderboard_period_all_includes_user_stats(monkeypatch):
    """All-time filter includes user_stats for archived project totals."""
    captured = {}
    rows = [{"wiki_username": "carol", "classifications": 10, "sdc_tags": 5}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=all")

    assert response.status_code == 200
    assert b"carol" in response.data
    assert "user_stats" in captured["sql"]
    assert captured["params"] is None


def test_leaderboard_invalid_period_defaults_to_all(monkeypatch):
    """Invalid period value falls back to all-time query."""
    captured = {}
    rows = [{"wiki_username": "dave", "classifications": 2, "sdc_tags": 0}]

    def execute_query_mock(sql, params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured["sql"] = sql
            captured["params"] = params
            return rows
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard?period=bogus")

    assert response.status_code == 200
    assert b"dave" in response.data
    assert "user_stats" in captured["sql"]
    assert captured["params"] is None


def test_leaderboard_filter_bar_active_state(monkeypatch):
    """Filter bar renders with correct active class for each period."""

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    # Default (all)
    resp = client.get("/leaderboard")
    assert b'class="active"' in resp.data
    assert b"All Time" in resp.data

    # Month
    resp = client.get("/leaderboard?period=month")
    html = resp.data.decode()
    assert "period=month" in html

    # Daily
    resp = client.get("/leaderboard?period=daily")
    html = resp.data.decode()
    assert "period=daily" in html


def test_account_settings_get(monkeypatch, fake_user):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "SELECT leaderboard_opt_out" in sql:
            return [{"leaderboard_opt_out": 0}]
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert b"Account Settings" in response.data
    assert b"leaderboard_opt_out" in response.data


def test_account_settings_get_opted_out(monkeypatch, fake_user):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, params=None, fetch=True):
        if "SELECT leaderboard_opt_out" in sql:
            return [{"leaderboard_opt_out": 1}]
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return [] if fetch else 0

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert b"checked" in response.data


def test_account_settings_post_opt_out(monkeypatch, fake_user):
    updated = {}

    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            updated["opt_out"] = params[0]
            return 1
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 302
    assert updated["opt_out"] == 1
    assert ("success", "Settings saved.") in _flashes_chunk6(client)


def test_account_settings_post_opt_in(monkeypatch, fake_user):
    updated = {}

    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            updated["opt_out"] = params[0]
            return 1
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 302
    assert updated["opt_out"] == 0


def test_account_settings_post_csrf_fail(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "wrong", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 400


def test_account_settings_post_db_error(monkeypatch, fake_user):
    def route_execute(sql, params, fetch):
        if "UPDATE users SET leaderboard_opt_out" in sql:
            raise app_module.DatabaseError("boom")
        return [] if fetch else 0

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/account/settings",
        data={"csrf_token": "testtoken", "leaderboard_opt_out": "1"},
    )

    assert response.status_code == 302
    assert ("error", "Failed to save settings.") in _flashes_chunk6(client)


def test_account_settings_requires_login(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    response = client.get("/account/settings")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_leaderboard_excludes_opted_out_users(monkeypatch):
    captured_sql = {}

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured_sql["leaderboard"] = sql
            return [{"wiki_username": "visible", "classifications": 5, "sdc_tags": 2}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    assert b"visible" in response.data
    assert "leaderboard_opt_out = 0" in captured_sql["leaderboard"]


def test_leaderboard_sdc_tags_attributed_to_classifier_not_project_owner(monkeypatch):
    """SDC tag count must be grouped by faces.classified_by_user_id, matching
    the archival logic in worker.py, not by projects.user_id (project owner)."""
    captured_sql = {}

    def execute_query_mock(sql, _params=None, _fetch=True):
        if "FROM worker_heartbeat" in sql:
            return [{"fresh": 1, "stopping": 0, "total": 1}]
        if "FROM users u" in sql and "LEFT JOIN faces f" in sql:
            captured_sql["leaderboard"] = sql
            return [{"wiki_username": "alice", "classifications": 3, "sdc_tags": 1}]
        return []

    monkeypatch.setattr(app_module, "execute_query", execute_query_mock)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/leaderboard")

    assert response.status_code == 200
    sql = captured_sql["leaderboard"]
    # Live query must credit the face classifier, not the project owner
    assert "classified_by_user_id" in sql
    assert "JOIN projects p ON sc.project_id = p.id" not in sql


def test_api_category_info_breaks_when_batched_queue_only_contains_visited(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)
    repeated_subcats = [{"title": "Category:A"} for _ in range(51)]

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 4, "subcats": 51},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": repeated_subcats}})
        if params.get("prop") == "categoryinfo" and "Category:A" in params.get("titles", ""):
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 3, "subcats": 0},
                            }
                        }
                    }
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["files"] == 7
    assert body["categories_visited"] == 2


def test_api_category_info_breaks_subcategory_fetch_when_deadline_exceeded(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 1, "subcats": 2},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3(
                {"query": {"categorymembers": [{"title": "Category:A"}, {"title": "Category:B"}]}}
            )
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A|Category:B":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "10": {"title": "Category:A", "categoryinfo": {"files": 2, "subcats": 1}},
                            "11": {"title": "Category:B", "categoryinfo": {"files": 3, "subcats": 1}},
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3({"query": {"categorymembers": []}})
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:B":
            raise AssertionError("Should break before requesting Category:B")
        raise AssertionError(params)

    monotonic_values = iter([0, 1, 1, 9, 9, 9])

    def _monotonic():
        return next(monotonic_values, 9)

    monkeypatch.setattr(app_module.requests, "get", _get)
    monkeypatch.setattr(app_module.time, "monotonic", _monotonic)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["files"] == 6


def test_api_category_info_marks_approximate_when_continue_in_subcategory_fetch(monkeypatch):
    client, _ = _auth_client_chunk4(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "Category:Root",
                                "categoryinfo": {"files": 2, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:Root":
            return _FakeResponse_chunk3({"query": {"categorymembers": [{"title": "Category:A"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {
                        "pages": {
                            "2": {
                                "title": "Category:A",
                                "categoryinfo": {"files": 5, "subcats": 1},
                            }
                        }
                    }
                }
            )
        if params.get("list") == "categorymembers" and params.get("cmtitle") == "Category:A":
            return _FakeResponse_chunk3(
                {
                    "query": {"categorymembers": []},
                    "continue": {"cmcontinue": "next-page"},
                }
            )
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    assert response.get_json()["approximate"] is True


def test_project_new_duplicate_entry_detected_via_exception_cause(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    cause = Exception(1062, "Duplicate entry")
    db_exc = DatabaseError("duplicate")
    db_exc.__cause__ = cause

    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("still being cleaned up" in msg for _cat, msg in _flashes_tail(client))


def test_project_new_duplicate_error_with_non_numeric_cause_args_falls_back(monkeypatch):
    captured = _capture_render_template_chunk4(monkeypatch)

    cause = Exception("not-a-number")
    db_exc = DatabaseError("insert failed")
    db_exc.__cause__ = cause

    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT id FROM projects" in sql:
            return []
        if "u.wiki_username" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise db_exc
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project. Please try again." in msg for _cat, msg in _flashes_tail(client))


def test_project_new_invite_code_collision_retries_and_succeeds(monkeypatch):
    """First INSERT raises invite_code collision; second attempt succeeds."""
    call_count = {"n": 0}

    def _insert():
        call_count["n"] += 1
        if call_count["n"] == 1:
            cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
            exc = DatabaseError("dup invite_code")
            exc.__cause__ = cause
            raise exc
        return 1

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _make_project_new_base_execute(_insert))
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 302
    assert call_count["n"] == 2
    assert any("successfully" in msg.lower() for _cat, msg in _flashes_tail(client))


def test_project_new_invite_code_collision_fails_after_max_retries(monkeypatch):
    """All retry attempts raise invite_code collisions; shows 'Failed to create project'."""
    captured = _capture_render_template_chunk4(monkeypatch)

    def _insert():
        cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
        exc = DatabaseError("dup invite_code")
        exc.__cause__ = cause
        raise exc

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _auth_client_chunk4(monkeypatch, _make_project_new_base_execute(_insert))
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project" in msg for _cat, msg in _flashes_tail(client))

    def _execute_query(sql, params=None, fetch=True):
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
        if "FROM faces f" in sql and "WHERE f.id = %s" in sql and "project_members" in sql:
            return [
                {
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
            ]
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda _fn: (_ for _ in ()).throw(ValueError("unexpected")),
    )

    with pytest.raises(ValueError, match="unexpected"):
        client.post(
            "/api/reclassify",
            data={"csrf_token": "testtoken", "face_id": "1", "is_target": "1"},
        )


def test_project_settings_get_db_error_returns_500(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            raise DatabaseError("boom")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)

    response = client.get("/project/1/settings")

    assert response.status_code == 500


def test_project_settings_post_shows_bootstrap_tip(monkeypatch):
    project = {
        "id": 1,
        "user_id": 1,
        "distance_threshold": 0.6,
        "min_confirmed": 5,
        "status": "active",
        "label": "Existing",
    }

    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 1, "by_bootstrap": 2}]
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "Updated",
        },
    )

    assert response.status_code == 302
    flashes = _flashes_tail(client)
    assert any(cat == "info" and "bootstrapped matches" in msg for cat, msg in flashes)


def test_project_settings_post_ignores_stats_query_db_error(monkeypatch):
    project = {
        "id": 1,
        "user_id": 1,
        "distance_threshold": 0.6,
        "min_confirmed": 5,
        "status": "active",
        "label": "Existing",
    }

    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            return 1
        if "AS human_confirmed" in sql:
            raise DatabaseError("stats failed")
        raise AssertionError(sql)

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.5",
            "min_confirmed": "5",
            "status": "active",
            "label": "Updated",
        },
    )

    assert response.status_code == 302
    assert ("success", "Settings updated.") in _flashes_tail(client)


def test_project_rerun_inference_db_error_loading_project_returns_500(monkeypatch):
    def _execute_query(sql, params=None, fetch=True):
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
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            raise DatabaseError("load failed")
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)
    _set_csrf_chunk4(client)

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 500


def test_project_rerun_inference_ignores_wake_file_oserror(monkeypatch, fake_user):
    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(sql)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(fn, min_confirmed=7, human_confirmed=8, rows_reset=3),
    )
    monkeypatch.setattr("builtins.open", MagicMock(side_effect=OSError("disk error")))

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert any(cat == "success" and "Reset 3 model-classified faces." in msg for cat, msg in _flashes_chunk6(client))


def test_project_rerun_inference_stats_query_db_error_resets_nothing(monkeypatch, fake_user):
    """If the gate cannot be checked, fail closed: do not wipe model results."""
    executed_sql = []

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects WHERE id = %s AND user_id = %s" in sql:
            return [
                {
                    "id": 1,
                    "status": "completed",
                    "distance_threshold": 0.55,
                    "min_confirmed": 7,
                    "last_inference_threshold": 0.6,
                    "last_inference_min_confirmed": 5,
                }
            ]
        raise AssertionError(sql)

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)
    monkeypatch.setattr(
        app_module,
        "execute_transaction",
        lambda fn: _fake_rerun_transaction(fn, stats_error=True, executed_sql=executed_sql),
    )
    monkeypatch.setattr("builtins.open", mock_open())

    response = client.post("/project/1/rerun-inference", data={"csrf_token": "testtoken"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/project/1/settings")
    assert not any("UPDATE faces f" in sql for sql in executed_sql)
    assert ("error", "Failed to re-run inference.") in _flashes_chunk6(client)
    assert not any(cat == "success" for cat, _msg in _flashes_chunk6(client))


def test_account_settings_get_db_error_falls_back_to_opt_out_zero(monkeypatch):
    captured = {}

    def _fake_render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return app_module.jsonify(context)

    def _execute_query(sql, params=None, fetch=True):
        del params, fetch
        if "SELECT leaderboard_opt_out" in sql:
            raise DatabaseError("cannot load")
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
        return ()

    monkeypatch.setattr(app_module, "render_template", _fake_render)
    client, _ = _auth_client_chunk4(monkeypatch, _execute_query)

    response = client.get("/account/settings")

    assert response.status_code == 200
    assert captured["template"] == "account_settings.html"
    assert captured["context"]["leaderboard_opt_out"] == 0


def test_project_settings_get_shows_members_list(monkeypatch, fake_user):
    project = _project_settings_base_row()
    members = [
        {
            "user_id": 5,
            "role": "member",
            "joined_at": datetime(2025, 1, 10, 12, 0, 0),
            "wiki_username": "Alice",
            "status": "active",
        },
        {
            "user_id": 8,
            "role": "member",
            "joined_at": datetime(2025, 2, 15, 9, 30, 0),
            "wiki_username": "Bob",
            "status": "active",
        },
    ]

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql and "JOIN users" in sql:
            return members
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Alice" in response.data
    assert b"Bob" in response.data
    assert b"Project Members" in response.data


def test_project_settings_get_empty_members_list(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Members" in response.data


def test_project_settings_get_members_db_error_returns_empty(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            raise app_module.DatabaseError("members failed")
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Project Settings" in response.data


def test_project_settings_shows_banned_members(monkeypatch, fake_user):
    project = _project_settings_base_row()
    members = [
        {
            "user_id": 5,
            "role": "member",
            "joined_at": datetime(2025, 1, 10, 12, 0, 0),
            "wiki_username": "Alice",
            "status": "active",
        },
        {
            "user_id": 8,
            "role": "member",
            "joined_at": datetime(2025, 2, 15, 9, 30, 0),
            "wiki_username": "Eve",
            "status": "banned",
        },
    ]

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql and "JOIN users" in sql:
            return members
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Alice" in response.data
    assert b"Eve" in response.data
    assert b"Banned" in response.data
    assert b"Unban" in response.data


def test_dashboard_member_counts_passed_to_template(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, params, fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 2}]
        if "SELECT DISTINCT p.*" in sql:
            return [
                {"id": 10, "wikidata_qid": "Q42", "p18_thumb_url": "t1"},
                {"id": 20, "wikidata_qid": "Q1", "p18_thumb_url": "t2"},
            ]
        if "project_members" in sql and "COUNT" in sql:
            return [{"project_id": 10, "cnt": 3}, {"project_id": 20, "cnt": 1}]
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {10: 3, 20: 1}


def test_dashboard_member_counts_db_error_is_non_critical(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT DISTINCT p.*" in sql:
            return [{"id": 10, "wikidata_qid": "Q42", "p18_thumb_url": "t1"}]
        if "project_members" in sql and "COUNT" in sql:
            raise app_module.DatabaseError("members count failed")
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {}


def test_dashboard_member_counts_empty_when_no_projects(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user_chunk3()

    def _handler(sql, _params, _fetch):
        if "COUNT(DISTINCT p.id) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT DISTINCT p.*" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client_chunk3(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["member_counts"] == {}


def test_project_detail_member_count_passed_to_template(monkeypatch):
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
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": "t.jpg", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS cnt" in sql and "project_members" in sql:
            return [{"cnt": 4}]
        if "status = 'pending'" in sql:
            return [{"cnt": 1}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["context"]["member_count"] == 4


def test_project_detail_member_count_db_error_defaults_zero(monkeypatch):
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
            return [{"id": 1, "user_id": 1, "wikidata_qid": "Q42", "p18_thumb_url": "t.jpg", "status": "active"}]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 5, "confirmed_matches": 2}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS cnt" in sql and "project_members" in sql:
            raise app_module.DatabaseError("member count failed")
        if "status = 'pending'" in sql:
            return [{"cnt": 1}]
        if "f.is_target IS NULL" in sql and "classified_by_user_id IS NULL" in sql:
            return [{"cnt": 2}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["context"]["member_count"] == 0


def test_project_detail_shows_no_faces_banner(monkeypatch):
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
            return [
                {
                    "id": 1,
                    "user_id": 1,
                    "wikidata_qid": "Q42",
                    "p18_thumb_url": "https://thumb.jpg",
                    "status": "completed",
                    "completion_reason": "no_faces",
                    "images_total": 50,
                    "images_processed": 50,
                }
            ]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 0, "confirmed_matches": 0}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 0}]
        if "status = 'pending'" in sql:
            return [{"cnt": 0}]
        if "f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["completion_reason"] == "no_faces"


def test_project_detail_shows_insufficient_faces_banner(monkeypatch):
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
            return [
                {
                    "id": 1,
                    "user_id": 1,
                    "wikidata_qid": "Q42",
                    "p18_thumb_url": "https://thumb.jpg",
                    "status": "completed",
                    "completion_reason": "insufficient_faces",
                    "images_total": 50,
                    "images_processed": 50,
                }
            ]
        if "COUNT(*) AS total_faces" in sql:
            return [{"total_faces": 3, "confirmed_matches": 0}]
        if "COUNT(*) AS cnt" in sql and "classified_by IN" in sql:
            return [{"cnt": 0}]
        if "status = 'pending'" in sql:
            return [{"cnt": 0}]
        if "f.is_target IS NULL" in sql:
            return [{"cnt": 0}]
        return ()

    client, _ = _auth_client_chunk4(monkeypatch, eq)
    response = client.get("/project/1")

    assert response.status_code == 200
    assert captured["template"] == "project_detail.html"
    assert captured["context"]["project"]["completion_reason"] == "insufficient_faces"


def test_project_settings_clears_completion_reason_on_update(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["completion_reason"] = "no_faces"
    project["status"] = "completed"
    update_sqls = []

    def route_execute(sql, params, fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "UPDATE projects SET distance_threshold" in sql:
            update_sqls.append(sql)
            return 1
        if "AS human_confirmed" in sql:
            return [{"human_confirmed": 10, "by_bootstrap": 0}]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.55",
            "min_confirmed": "3",
            "status": "active",
            "label": "Reactivated",
        },
    )

    assert response.status_code == 302
    assert len(update_sqls) == 1
    assert "completion_reason = NULL" in update_sqls[0]


def test_project_new_cross_user_duplicate_already_member_redirects(monkeypatch):
    """If user is already a member of a cross-user project, redirect to that project."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 2,
        "wiki_user_id": 200,
        "wiki_username": "joiner",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": None,
    }

    def mock_execute_query(sql, params=None, fetch=True):
        # before_request loads user
        if "SELECT id, wiki_user_id, wiki_username, access_token" in sql:
            return [fake_user]
        # worker heartbeat check
        if "worker_heartbeat" in sql:
            return []
        # Own-duplicate check: user has no own project
        if "SELECT id FROM projects WHERE user_id" in sql:
            return []
        # Cross-user: another user (id=1) has the same project
        if "SELECT p.id, p.label, u.wiki_username" in sql:
            return [{"id": 10, "label": "Test", "wiki_username": "owner1"}]
        # Membership check: user IS already a member
        if "SELECT status FROM project_members" in sql:
            return [{"status": "member"}]
        return []

    monkeypatch.setattr(app_module, "execute_query", mock_execute_query)
    monkeypatch.setattr(app_module, "decrypt_token", lambda t: t)
    monkeypatch.setattr(app_module, "_is_human_entity", lambda qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda cat: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda cat: True)

    with client.session_transaction() as sess:
        sess["user_id"] = 2
        sess["csrf_token"] = "tok"

    response = client.post(
        "/project/new",
        data={
            "wikidata_qid": "Q42",
            "commons_category": "Test Category",
            "label": "Test",
            "distance_threshold": "0.6",
            "min_confirmed": "5",
            "csrf_token": "tok",
        },
    )

    # Should redirect to the existing project
    assert response.status_code == 302
    assert "/project/10" in response.headers["Location"]


def test_project_new_cross_user_duplicate_shows_join_option(monkeypatch):
    """If another user has the project and current user is not a member, show join option."""
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 3,
        "wiki_user_id": 300,
        "wiki_username": "newcomer",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": None,
    }

    def mock_execute_query(sql, params=None, fetch=True):
        if "SELECT id, wiki_user_id, wiki_username, access_token" in sql:
            return [fake_user]
        if "worker_heartbeat" in sql:
            return []
        # Own-duplicate check: no own project
        if "SELECT id FROM projects WHERE user_id" in sql:
            return []
        # Cross-user: another user owns the same project
        if "SELECT p.id, p.label, u.wiki_username" in sql:
            return [{"id": 20, "label": "Test", "wiki_username": "otherowner"}]
        # Membership check: NOT a member
        if "SELECT status FROM project_members" in sql:
            return []
        return []

    monkeypatch.setattr(app_module, "execute_query", mock_execute_query)
    monkeypatch.setattr(app_module, "decrypt_token", lambda t: t)
    monkeypatch.setattr(app_module, "_is_human_entity", lambda qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda cat: True)
    monkeypatch.setattr(app_module, "_commons_category_has_files", lambda cat: True)

    with client.session_transaction() as sess:
        sess["user_id"] = 3
        sess["csrf_token"] = "tok"

    response = client.post(
        "/project/new",
        data={
            "wikidata_qid": "Q42",
            "commons_category": "Test Category",
            "label": "Test",
            "distance_threshold": "0.6",
            "min_confirmed": "5",
            "csrf_token": "tok",
        },
    )

    assert response.status_code == 200
    # Join option message should appear in the rendered template
    assert b"otherowner" in response.data or b"project_new" in response.data


def test_api_gallery_filtered_sdc_pending_with_source_filter(monkeypatch, fake_user):
    call_log = []

    def route_execute(sql, params, fetch):
        call_log.append(sql)
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM faces" in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS total" in sql and "sdc_pending" not in sql:
            return [{"cnt": 3}]
        if "COUNT(*) AS total" in sql and "sdc_pending" in sql:
            return [
                {
                    "total": 10,
                    "matches": 6,
                    "non_matches": 3,
                    "rejected": 1,
                    "source_model": 4,
                    "source_bootstrap": 2,
                    "source_human": 4,
                    "sdc_pending": 5,
                }
            ]
        if "sdc_pending" in sql and "COUNT(*) AS total" not in sql:
            return [{"sdc_pending": 2}]
        if "SELECT f.id" in sql:
            return []
        return [] if fetch else 0

    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client = _make_authed_client(monkeypatch, fake_user, route_execute)
        response = client.get("/api/project/1/gallery?page=1&source=model")
        payload = response.get_json()
    finally:
        flask_app.config["SERVER_NAME"] = None

    assert response.status_code == 200
    assert "counts" in payload
    assert payload["counts"]["sdc_pending"] == 5
    assert payload["counts"]["filtered_sdc_pending"] == 2


def test_api_gallery_filtered_sdc_pending_no_filter_equals_global(monkeypatch, fake_user):
    def route_execute(sql, params, fetch):
        if "FROM projects p LEFT JOIN project_members" in sql:
            return [{"id": 1, "user_id": 1, "status": "active"}]
        if "SELECT COUNT(*) AS cnt FROM faces" in sql:
            return [{"cnt": 5}]
        if "COUNT(*) AS total" in sql:
            return [
                {
                    "total": 10,
                    "matches": 6,
                    "non_matches": 3,
                    "rejected": 1,
                    "source_model": 4,
                    "source_bootstrap": 2,
                    "source_human": 4,
                    "sdc_pending": 5,
                }
            ]
        if "SELECT f.id" in sql:
            return []
        return [] if fetch else 0

    flask_app.config["SERVER_NAME"] = "localhost"
    try:
        client = _make_authed_client(monkeypatch, fake_user, route_execute)
        response = client.get("/api/project/1/gallery?page=1")
        payload = response.get_json()
    finally:
        flask_app.config["SERVER_NAME"] = None

    assert response.status_code == 200
    assert "counts" in payload
    assert payload["counts"]["filtered_sdc_pending"] == payload["counts"]["sdc_pending"]


@pytest.mark.parametrize("value,ok", [("0.5", True), ("0.1", True), ("1.0", True), ("0.09", False), ("1.01", False)])
def test_project_new_distance_threshold_range_is_unchanged(monkeypatch, value, ok):
    client, _ = _threshold_form_client(monkeypatch)
    monkeypatch.setattr(app_module, "_is_human_entity", lambda *_a, **_k: False)

    resp = client.post(
        "/project/new",
        data={
            "csrf_token": "testtoken",
            "wikidata_qid": "Q42",
            "commons_category": "Douglas Adams",
            "distance_threshold": value,
            "min_confirmed": "5",
        },
    )

    body = resp.get_data(as_text=True)
    assert ("Distance threshold must be between 0.1 and 1.0." in body) is (not ok)


def test_project_new_form_defaults_to_the_cosine_threshold():
    assert app_module.DEFAULT_DISTANCE_THRESHOLD == 0.5
    with flask_app.test_request_context("/"):
        html = app_module.render_template("project_new.html")
    assert 'value="0.5"' in html
    assert "(default 0.5)" in html
    assert "0.6" not in html.split('id="distance_threshold"', 1)[1].split("</div>", 1)[0]


def test_project_settings_invalid_prefilter_mode(monkeypatch, fake_user):
    project = _project_settings_base_row()

    def route_execute(sql, _params, _fetch):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.6",
            "min_confirmed": "5",
            "status": "active",
            "label": "X",
            "prefilter_mode": "enforce",
        },
    )

    assert response.status_code == 200
    assert b"Invalid metadata pre-filter mode." in response.data


def test_project_settings_success_persists_prefilter_mode(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "off"
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
    _set_csrf_chunk6(client)

    response = client.post(
        "/project/1/settings",
        data={
            "csrf_token": "testtoken",
            "distance_threshold": "0.55",
            "min_confirmed": "6",
            "status": "paused",
            "label": "New Label",
            "prefilter_mode": "shadow",
        },
    )

    assert response.status_code == 302
    assert len(updates) == 1
    assert updates[0][0][4] == "shadow"


def test_project_settings_shows_liftwing_banner_and_status(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    def route_execute(sql, _params=None, _fetch=True):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        if "SELECT status, last_checked, last_ok" in sql:
            return [_liftwing_status_row("down")]
        if "prefilter_checked_at" in sql:
            return [_prefilter_stats_row()]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"The LiftWing AI service used for the metadata filter" in response.data
    assert b"unreachable" in response.data


def test_project_settings_hides_liftwing_banner_when_mode_off(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "off"

    def route_execute(sql, _params=None, _fetch=True):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        if "SELECT status, last_checked, last_ok" in sql:
            return [_liftwing_status_row("down")]
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)
    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"The LiftWing AI service used for the metadata filter" not in response.data


def test_project_settings_prefilter_stats_in_shadow_mode(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    client = _prefilter_stats_client(monkeypatch, fake_user, project, _prefilter_stats_row())

    body = client.get("/project/1/settings").get_data(as_text=True)

    assert "Pre-filter results" in body
    assert "Files checked" in body
    assert 'stat-value">250</div>' in body
    assert "Would skip 8.0%" in body
    assert 'stat-value">20</div>' in body
    assert "Confirmed matches it would drop" in body
    assert "Skips that had a face" in body
    assert 'stat-value">3</div>' in body
    assert "It would not have dropped any of the 7 images you confirmed contain the target." in body
    assert "It kept 5 checked files in which no face was found" in body
    assert "last verdict 2026-10-17 09:30 UTC" in body
    assert 'class="stat-box stat-box-success"' in body


def test_project_settings_prefilter_stats_warns_on_confirmed_match_drop(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    client = _prefilter_stats_client(monkeypatch, fake_user, project, _prefilter_stats_row(human_targets_dropped=2))

    body = client.get("/project/1/settings").get_data(as_text=True)

    assert "It would have dropped 2 of the 7 images you confirmed contain the target." in body
    assert "Do not enforce this filter on this project." in body
    assert 'class="prefilter-note prefilter-note-bad"' in body
    assert 'class="stat-box stat-box-danger"' in body


def test_project_settings_prefilter_stats_no_human_confirmed_data(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    client = _prefilter_stats_client(
        monkeypatch,
        fake_user,
        project,
        _prefilter_stats_row(human_targets_total=0, kept_without_faces=0),
    )

    body = client.get("/project/1/settings").get_data(as_text=True)

    assert "No human-confirmed target images are among the checked files yet" in body
    assert 'class="prefilter-note prefilter-note-ok"' not in body


def test_project_settings_prefilter_stats_empty_state(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    client = _prefilter_stats_client(monkeypatch, fake_user, project, _empty_prefilter_stats_row())

    body = client.get("/project/1/settings").get_data(as_text=True)

    assert "No verdicts recorded yet." in body
    assert "Files checked" not in body
    assert "last verdict" not in body


def test_project_settings_prefilter_stats_hidden_when_mode_off(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "off"

    def route_execute(sql, _params=None, _fetch=True):
        if "SELECT * FROM projects" in sql:
            return [project.copy()]
        if "project_members" in sql:
            return []
        raise AssertionError(f"Unexpected SQL: {sql}")

    client = _make_authed_client(monkeypatch, fake_user, route_execute)

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Pre-filter results" not in response.data


def test_project_settings_prefilter_stats_db_error_keeps_page(monkeypatch, fake_user):
    project = _project_settings_base_row()
    project["prefilter_mode"] = "shadow"

    client = _prefilter_stats_client(monkeypatch, fake_user, project, DatabaseError("stats failed"))

    response = client.get("/project/1/settings")

    assert response.status_code == 200
    assert b"Pre-filter stats are unavailable right now." in response.data


@pytest.mark.integration
def test_project_settings_prefilter_stats_from_db(integration_client, db_conn, seed_project, seed_images, seed_faces):
    client, _, _ = integration_client

    with db_conn.cursor() as cur:
        cur.execute("UPDATE projects SET prefilter_mode = 'shadow' WHERE id = %s", (seed_project["id"],))
        for image, keep in zip(seed_images, (1, 1, 0, 0, 0)):
            cur.execute(
                "UPDATE images SET prefilter_keep = %s, prefilter_reason = 'test verdict', "
                "prefilter_checked_at = NOW() WHERE id = %s",
                (keep, image["id"]),
            )
        # The middle dropped image has no human-confirmed target: deleting its
        # target face must keep it out of the safety counter.
        cur.execute("DELETE FROM faces WHERE image_id = %s AND is_target = 1", (seed_images[2]["id"],))

    response = client.get(f"/project/{seed_project['id']}/settings")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Pre-filter results" in body
    assert 'stat-value">5</div>' in body  # checked
    assert 'stat-value">3</div>' in body  # would skip
    assert "Would skip 60.0%" in body
    assert 'stat-value">2</div>' in body  # confirmed matches it would drop
    assert "It would have dropped 2 of the 4 images you confirmed contain the target." in body


@pytest.mark.integration
def test_project_rerun_inference_resets_and_worker_reclassifies(
    integration_client, db_conn, seed_project, seed_images, seed_faces
):
    """End-to-end: the route resets model faces and the worker resolves them.

    Every image already has a human-confirmed target, so the worker must mark
    the reset faces as non-matches instead of creating a second target.
    """
    client, _, _ = integration_client
    model_face_ids = []

    with db_conn.cursor() as cur:
        for image in seed_images[:2]:
            cur.execute(
                "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, "
                "bbox_bottom, bbox_left, is_target, classified_by, confidence, sdc_written) "
                "VALUES (%s, %s, 'scrfd500m+arcface_mbf', 100, 300, 300, 100, 1, 'model', 0.2, 0)",
                (image["id"], _make_embedding(300 + image["id"])),
            )
            model_face_ids.append(cur.lastrowid)
        cur.execute(
            "SELECT id, min_confirmed, distance_threshold FROM projects WHERE id = %s",
            (seed_project["id"],),
        )
        project = cur.fetchone()

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    response = client.post(
        f"/project/{seed_project['id']}/rerun-inference",
        data={"csrf_token": "testtoken"},
    )

    assert response.status_code == 302
    assert "inference_triggered=1" in response.headers["Location"]

    with db_conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(model_face_ids))
        cur.execute(
            "SELECT id, is_target, classified_by FROM faces WHERE id IN (" + placeholders + ") ORDER BY id",
            tuple(model_face_ids),
        )
        rows = cur.fetchall()
    assert len(rows) == 2
    assert all(row["is_target"] is None and row["classified_by"] is None for row in rows)

    import worker

    with patch("worker.shutdown_requested", False):
        classified = worker.run_autonomous_inference(
            {
                "id": project["id"],
                "min_confirmed": project["min_confirmed"],
                "distance_threshold": float(project["distance_threshold"]),
            }
        )

    assert classified == 2
    with db_conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(model_face_ids))
        cur.execute(
            "SELECT id, is_target, classified_by FROM faces WHERE id IN (" + placeholders + ") ORDER BY id",
            tuple(model_face_ids),
        )
        rows = cur.fetchall()
    assert all(row["is_target"] == 0 for row in rows)
    assert all(row["classified_by"] == "model" for row in rows)
