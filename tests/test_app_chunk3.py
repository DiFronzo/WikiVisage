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
import requests


def _fake_user(username="tester"):
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": username,
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _make_authenticated_client(monkeypatch, execute_query_fn=None):
    fake_user = _fake_user()
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    monkeypatch.setattr(app_module.limiter, "enabled", False)
    client = flask_app.test_client()

    if execute_query_fn is None:

        def _default_execute_query(*_a, **_kw):
            return [fake_user]

        execute_query_fn = _default_execute_query

    monkeypatch.setattr(app_module, "execute_query", execute_query_fn)
    monkeypatch.setattr(app_module, "_load_whitelist", lambda: {"tester"})
    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _user_aware_execute(fake_user, route_handler):
    def _execute_query(sql, params=None, fetch=True):
        if "FROM users WHERE id = %s" in sql:
            return [fake_user]
        if "FROM worker_heartbeat" in sql:
            return [{"is_stale": 0}]
        if "SHOW TABLES LIKE 'worker_heartbeat'" in sql:
            return [{"1": 1}]
        return route_handler(sql, params, fetch)

    return _execute_query


def _capture_render(monkeypatch):
    captured = {}

    def _render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return f"rendered:{template}"

    monkeypatch.setattr(app_module, "render_template", _render)
    return captured


def _set_csrf(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes(client):
    with client.session_transaction() as sess:
        return sess.get("_flashes", [])


class _FakeResponse:
    def __init__(self, payload=None, raise_exc=None):
        self.payload = {} if payload is None else payload
        self.raise_exc = raise_exc

    def raise_for_status(self):
        if self.raise_exc:
            raise self.raise_exc

    def json(self):
        return self.payload


def test_index_not_logged_in_renders_index_template(monkeypatch):
    flask_app.config["TESTING"] = True
    captured = _capture_render(monkeypatch)
    client = flask_app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "rendered:index.html"
    assert captured["template"] == "index.html"


def test_index_logged_in_redirects_to_dashboard(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

    response = client.get("/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_dashboard_defaults_page_and_totals(monkeypatch):
    captured = _capture_render(monkeypatch)
    calls = []
    fake_user = _fake_user()

    def _handler(sql, params, fetch):
        calls.append((sql, params, fetch))
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT * FROM projects" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["template"] == "dashboard.html"
    assert captured["context"]["page"] == 1
    assert captured["context"]["total_pages"] == 1
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["projects"] == []
    assert any("COUNT(*) AS cnt" in sql for sql, _params, _fetch in calls)
    assert any("SELECT * FROM projects" in sql for sql, _params, _fetch in calls)


def test_dashboard_invalid_page_falls_back_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=abc")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_negative_page_clamped_to_one(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        return []

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=-5")

    assert response.status_code == 200
    assert captured["context"]["page"] == 1


def test_dashboard_page_clamped_to_total_pages(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()
    select_params = {}

    def _handler(sql, params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 30}]
        if "SELECT * FROM projects" in sql:
            select_params["params"] = params
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=99")

    assert response.status_code == 200
    assert captured["context"]["page"] == 2
    assert captured["context"]["total_pages"] == 2
    assert select_params["params"][2] == 25


def test_dashboard_count_database_error_sets_total_zero(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            raise app_module.DatabaseError("count failed")
        if "SELECT * FROM projects" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard?page=3")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0
    assert captured["context"]["page"] == 1


def test_dashboard_projects_database_error_flashes_and_uses_empty(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 10}]
        if "SELECT * FROM projects" in sql:
            raise app_module.DatabaseError("projects failed")
        raise AssertionError(sql)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["projects"] == []
    assert any("Failed to load projects." in msg for _cat, msg in _flashes(client))


def test_dashboard_count_row_empty_tuple(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return ()
        if "SELECT * FROM projects" in sql:
            return []
        raise AssertionError(sql)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert captured["context"]["total_projects"] == 0


def test_dashboard_lazily_populates_missing_p18_and_updates_db(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user()
    updates = []

    projects = [
        {"id": 101, "wikidata_qid": "Q42", "p18_thumb_url": None},
        {"id": 102, "wikidata_qid": "Q1", "p18_thumb_url": "existing"},
        {"id": 103, "wikidata_qid": None, "p18_thumb_url": None},
    ]

    def _handler(sql, params, fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 3}]
        if "SELECT * FROM projects" in sql:
            return projects
        if "UPDATE projects SET p18_thumb_url" in sql:
            updates.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda qid: f"thumb-{qid}")
    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert projects[0]["p18_thumb_url"] == "thumb-Q42"
    assert len(updates) == 1
    assert updates[0][0] == ("thumb-Q42", 101)
    assert updates[0][1] is False


def test_dashboard_missing_thumb_but_fetch_returns_none_no_update(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()
    update_called = {"called": False}

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT * FROM projects" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            update_called["called"] = True
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert update_called["called"] is False
    assert captured["template"] == "dashboard.html"


def test_dashboard_p18_update_database_error_is_non_critical(monkeypatch):
    _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 1}]
        if "SELECT * FROM projects" in sql:
            return [{"id": 1, "wikidata_qid": "Q42", "p18_thumb_url": None}]
        if "UPDATE projects SET p18_thumb_url" in sql:
            raise app_module.DatabaseError("update failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200


def test_dashboard_projects_tuple_skips_lazy_population(monkeypatch):
    fake_user = _fake_user()
    fetch_calls = {"count": 0}

    def _handler(sql, _params, _fetch):
        if "COUNT(*) AS cnt" in sql:
            return [{"cnt": 0}]
        if "SELECT * FROM projects" in sql:
            return ()
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: fetch_calls.update(count=1))
    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert fetch_calls["count"] == 0


def test_api_category_info_missing_category_returns_400(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

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
    client, _ = _make_authenticated_client(monkeypatch)

    response = client.get("/api/category-info", query_string={"category": category})

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid category name"}


def test_api_category_info_root_category_not_found_returns_404(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse({"query": {"pages": {"-1": {"missing": True}}}}),
    )

    response = client.get("/api/category-info?category=People")

    assert response.status_code == 404
    assert response.get_json() == {"error": "not_found"}


def test_api_category_info_root_no_subcats_returns_direct_counts(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

    def _get(*_args, **_kwargs):
        return _FakeResponse(
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
    assert not body["approximate"]


def test_api_category_info_bfs_traversal_sums_subcategories(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse(
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
            return _FakeResponse({"query": {"categorymembers": [{"title": "Category:A"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse(
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
            return _FakeResponse({"query": {"categorymembers": [{"title": "Category:B"}]}})
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:B":
            return _FakeResponse(
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
    assert not body["approximate"]


def test_api_category_info_sets_approximate_on_continue_from_root_subcats(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse(
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
            return _FakeResponse(
                {
                    "query": {"categorymembers": [{"title": "Category:A"}]},
                    "continue": {"cmcontinue": "next"},
                }
            )
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:A":
            return _FakeResponse(
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
    client, _ = _make_authenticated_client(monkeypatch)
    many = [{"title": f"Category:C{i}"} for i in range(60)]

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse(
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
            return _FakeResponse({"query": {"categorymembers": many}})
        if params.get("prop") == "categoryinfo" and "Category:C" in params.get("titles", ""):
            pages = {}
            for i, title in enumerate(params["titles"].split("|"), start=100):
                pages[str(i)] = {"title": title, "categoryinfo": {"files": 1, "subcats": 0}}
            return _FakeResponse({"query": {"pages": pages}})
        raise AssertionError(params)

    monkeypatch.setattr(app_module.requests, "get", _get)

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 200
    body = response.get_json()
    assert body["approximate"] is True
    assert body["categories_visited"] == 50


def test_api_category_info_timeout_marks_approximate(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)

    def _get(_url, params, **_kwargs):
        if params.get("prop") == "categoryinfo" and params.get("titles") == "Category:Root":
            return _FakeResponse(
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
            return _FakeResponse({"query": {"categorymembers": [{"title": "Category:A"}]}})
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
    client, _ = _make_authenticated_client(monkeypatch)
    monkeypatch.setattr(
        app_module.requests, "get", lambda *a, **kw: (_ for _ in ()).throw(requests.RequestException("boom"))
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_api_category_info_http_error_from_raise_for_status_returns_502(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *a, **kw: _FakeResponse(raise_exc=requests.HTTPError("bad status")),
    )

    response = client.get("/api/category-info?category=Root")

    assert response.status_code == 502
    assert response.get_json() == {"error": "api_error"}


def test_project_new_get_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)

    response = client.get("/project/new")

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"


def test_project_new_post_invalid_csrf_returns_400(monkeypatch):
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client, "session-token")

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
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "commons_category": "People"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes(client))


def test_project_new_post_qid_without_q_prefix_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Wikidata Q-ID must start" in msg for _cat, msg in _flashes(client))


def test_project_new_post_missing_category_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    response = client.post("/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42"})

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Commons category is required." in msg for _cat, msg in _flashes(client))


@pytest.mark.parametrize("value", ["abc", "0.09", "1.1"])
def test_project_new_post_invalid_distance_threshold(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

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
    messages = [msg for _cat, msg in _flashes(client)]
    assert any("Distance threshold" in msg for msg in messages)


@pytest.mark.parametrize("value", ["abc", "0", "-5"])
def test_project_new_post_invalid_min_confirmed(monkeypatch, value):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

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
    messages = [msg for _cat, msg in _flashes(client)]
    assert any("Minimum confirmed" in msg for msg in messages)


def test_project_new_post_multiple_validation_errors_are_all_flashed(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "", "commons_category": "", "distance_threshold": "oops"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    messages = [msg for _cat, msg in _flashes(client)]
    assert any("Wikidata Q-ID must start" in msg for msg in messages)
    assert any("Commons category is required." in msg for msg in messages)
    assert any("Distance threshold must be a number." in msg for msg in messages)


def test_project_new_validation_short_circuits_remote_checks_on_basic_errors(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

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
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: False)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q123", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("is not an instance of human" in msg for _cat, msg in _flashes(client))


def test_project_new_post_missing_commons_category_on_wikimedia_flashes_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    client, _ = _make_authenticated_client(monkeypatch)
    _set_csrf(client)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: False)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "NotReal"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("does not exist" in msg for _cat, msg in _flashes(client))


def test_project_new_post_duplicate_project_renders_form(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return [{"id": 99}]
        if "INSERT INTO projects" in sql:
            raise AssertionError("Should not insert duplicates")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People", "label": "My Label"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("already exists" in msg for _cat, msg in _flashes(client))


def test_project_new_duplicate_check_db_error_continues_to_create(monkeypatch):
    fake_user = _fake_user()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            raise app_module.DatabaseError("duplicate check failed")
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb-q42")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Douglas Adams")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    assert len(insert_calls) == 1


def test_project_new_successful_creation_with_label_fetch_and_p18(monkeypatch):
    fake_user = _fake_user()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return ()
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "https://thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Douglas Adams"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

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
    assert any("Project created successfully!" in msg for _cat, msg in _flashes(client))


def test_project_new_successful_creation_with_user_label_skips_wikidata_label_fetch(monkeypatch):
    fake_user = _fake_user()
    insert_calls = []
    label_calls = {"n": 0}

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")

    def _fetch_label(_qid):
        label_calls["n"] += 1
        return "Should Not Be Used"

    monkeypatch.setattr(app_module, "_fetch_wikidata_label", _fetch_label)
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

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
    fake_user = _fake_user()
    insert_calls = []

    def _handler(sql, params, fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            insert_calls.append((params, fetch))
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: None)
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")
    monkeypatch.setattr("builtins.open", MagicMock())

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert insert_calls[0][0][6] is None


def test_project_new_wake_file_oserror_is_non_critical(monkeypatch):
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            return 1
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    def _raise_open(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise_open)

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

    response = client.post(
        "/project/new", data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_project_new_db_error_on_insert_renders_form_with_error(monkeypatch):
    captured = _capture_render(monkeypatch)
    fake_user = _fake_user()

    def _handler(sql, _params, _fetch):
        if "SELECT id FROM projects" in sql:
            return []
        if "DELETE FROM projects" in sql:
            return 0
        if "INSERT INTO projects" in sql:
            raise app_module.DatabaseError("insert failed")
        raise AssertionError(sql)

    monkeypatch.setattr(app_module, "_is_human_entity", lambda _qid: True)
    monkeypatch.setattr(app_module, "_commons_category_exists", lambda _category: True)
    monkeypatch.setattr(app_module, "_fetch_p18_thumb_url", lambda _qid: "thumb")
    monkeypatch.setattr(app_module, "_fetch_wikidata_label", lambda _qid: "Label")

    client, _ = _make_authenticated_client(monkeypatch, _user_aware_execute(fake_user, _handler))
    _set_csrf(client)

    response = client.post(
        "/project/new",
        data={"csrf_token": "testtoken", "wikidata_qid": "Q42", "commons_category": "People"},
    )

    assert response.status_code == 200
    assert captured["template"] == "project_new.html"
    assert any("Failed to create project. Please try again." in msg for _cat, msg in _flashes(client))
