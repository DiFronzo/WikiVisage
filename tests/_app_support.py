"""Helpers shared by more than one of the split test_app_*.py modules."""

from datetime import UTC, datetime, timedelta

from tests._app_preamble import (
    app_module,
    flask_app,
)


class _MockResponse:
    def __init__(self, json_data=None, should_raise=False):
        self._json_data = {} if json_data is None else json_data
        self._should_raise = should_raise

    def raise_for_status(self):
        if self._should_raise:
            raise RuntimeError("http error")

    def json(self):
        return self._json_data


def _get_flashes(client):
    with client.session_transaction() as sess:
        return list(sess.get("_flashes", []))


def _fake_user_chunk3(username="tester"):
    return {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": username,
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC) + timedelta(hours=4),
    }


def _make_authenticated_client_chunk3(monkeypatch, execute_query_fn=None):
    fake_user = _fake_user_chunk3()
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False
    monkeypatch.setattr(app_module.limiter, "enabled", False)
    client = flask_app.test_client()

    if execute_query_fn is None:

        def _default_execute_query(*_a, **_kw):
            return [fake_user]

        execute_query_fn = _default_execute_query

    monkeypatch.setattr(app_module, "execute_query", execute_query_fn)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _capture_render(monkeypatch):
    captured = {}

    def _render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return f"rendered:{template}"

    monkeypatch.setattr(app_module, "render_template", _render)
    return captured


def _auth_client_chunk4(monkeypatch, execute_query_impl=None):
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

        def _default_execute_query_impl(sql, params=None, fetch=True):
            del params, fetch
            if "FROM users WHERE id = %s" in sql:
                return [fake_user]
            return ()

        execute_query_impl = _default_execute_query_impl

    monkeypatch.setattr(app_module, "execute_query", execute_query_impl)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client, fake_user


def _set_csrf_chunk4(client, token=None):
    if token is None:
        token = "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _capture_render_template_chunk4(monkeypatch):
    captured = {}

    def fake_render(template, **context):
        captured["template"] = template
        captured["context"] = context
        return app_module.jsonify({"template": template})

    monkeypatch.setattr(app_module, "render_template", fake_render)
    return captured


def _fake_user_chunk5():
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
    fake_user = _fake_user_chunk5()
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

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        if csrf_token is not None:
            sess["csrf_token"] = csrf_token

    return client, fake_user


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

    with client.session_transaction() as sess:
        sess["user_id"] = 1
    return client


def _set_csrf_chunk6(client, token=None):
    token = token or "testtoken"
    with client.session_transaction() as sess:
        sess["csrf_token"] = token


def _flashes_chunk6(client):
    with client.session_transaction() as sess:
        return list(sess.get("_flashes", []))


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


def _make_invite_code_collision_error():
    """Build a DatabaseError that looks like a MySQL 1062 on the invite_code index."""
    cause = Exception(1062, "Duplicate entry 'ABCD1234' for key 'idx_projects_invite_code'")
    exc = app_module.DatabaseError("duplicate invite_code")
    exc.__cause__ = cause
    return exc


def _liftwing_status_row(status="down", is_stale=0):
    return {
        "status": status,
        "is_stale": is_stale,
        "last_checked": datetime.now(UTC),
        "last_ok": None,
    }


def _prefilter_stats_row(**overrides):
    row = {
        "checked": 250,
        "kept": 230,
        "dropped": 20,
        "dropped_with_faces": 3,
        "kept_without_faces": 5,
        "human_targets_dropped": 0,
        "human_targets_total": 7,
        "last_checked": datetime(2026, 10, 17, 9, 30),
    }
    row.update(overrides)
    return row


def _empty_prefilter_stats_row():
    return _prefilter_stats_row(
        checked=0,
        kept=0,
        dropped=0,
        dropped_with_faces=0,
        kept_without_faces=0,
        human_targets_total=0,
        last_checked=None,
    )
