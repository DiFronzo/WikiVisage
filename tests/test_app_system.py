"""Tests for the web app: system."""

import os
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
from flask import abort

from tests._app_preamble import (
    app_module,
    flask_app,
)
from tests._app_support import (
    _liftwing_status_row,
)


def _csp_directive(csp: str, name: str) -> str:
    for part in csp.split(";"):
        part = part.strip()
        if part.startswith(name + " ") or part == name:
            return part
    raise AssertionError(f"{name} missing from CSP: {csp}")


def _run_main_guard(env: dict[str, str]) -> dict:
    import runpy

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

    with patch.dict(os.environ, env, clear=False):
        with patch("database.init_db"):
            with patch("flask.app.Flask.run") as run_mock:
                runpy.run_path(app_path, run_name="__main__")

    run_mock.assert_called_once()
    return run_mock.call_args.kwargs


def _health_execute_query(liftwing_usage, liftwing_row):
    def route(sql, params=None, fetch=True):
        if "SELECT 1 AS ok" in sql:
            return [{"ok": 1}]
        if "COUNT(*) AS cnt FROM projects" in sql:
            return [{"cnt": liftwing_usage}]
        if "SELECT status, last_checked, last_ok" in sql:
            return [liftwing_row] if liftwing_row else []
        raise AssertionError(f"Unexpected SQL: {sql}")

    return route


@pytest.mark.integration
def test_health_endpoint(integration_client):
    client, _, _ = integration_client

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"
    assert payload["face_service"] in ("reachable", "unreachable")
    assert payload["limiter"] in ("redis", "memory")
    assert isinstance(payload["degraded"], bool)


def test_inject_i18n_helpers_ltr_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    data = app_module.inject_i18n_helpers()

    assert data["current_locale"] == "en"
    assert data["languages"] == app_module.LANGUAGES
    assert data["text_direction"] == "ltr"
    assert data["app_version"] == app_module.APP_VERSION


def test_inject_i18n_helpers_rtl_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "ar")
    data = app_module.inject_i18n_helpers()
    assert data["text_direction"] == "rtl"


def test_inject_worker_status_stale_heartbeat(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"is_stale": 1}])
    assert app_module.inject_worker_status() == {"worker_down": True}


def test_inject_worker_status_fresh_heartbeat(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_kw: [{"is_stale": 0}])
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_worker_status_missing_row_table_exists(monkeypatch):
    calls = {"count": 0}

    def _query(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [{"1": 1}]

    monkeypatch.setattr(app_module, "execute_query", _query)
    assert app_module.inject_worker_status() == {"worker_down": True}


def test_inject_worker_status_missing_row_table_missing(monkeypatch):
    calls = {"count": 0}

    def _query(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return []

    monkeypatch.setattr(app_module, "execute_query", _query)
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_worker_status_exception_returns_not_down(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "execute_query", _raise)
    assert app_module.inject_worker_status() == {"worker_down": False}


def test_inject_csrf_token_exposes_callable():
    with flask_app.test_request_context("/"):
        data = app_module.inject_csrf_token()
        assert "csrf_token" in data
        token = data["csrf_token"]()
        assert isinstance(token, str)
        assert len(token) == 64


def test_csp_header_contains_required_directives():
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)
        csp = result.headers["Content-Security-Policy"]

        assert "default-src 'self'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp
        assert "img-src 'self' https://*.wikimedia.org data:" in csp
        assert "connect-src 'self'" in csp
        assert "font-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp


def test_csp_script_src_uses_the_request_nonce_and_no_unsafe_inline():
    """Scripts are allowed by nonce only; inline handler attributes are forbidden."""
    with flask_app.test_request_context():
        from flask import g

        g.csp_nonce = "test-nonce-value"
        response = flask_app.make_response(("ok", 200))
        csp = app_module.set_security_headers(response).headers["Content-Security-Policy"]

        assert _csp_directive(csp, "script-src") == "script-src 'self' 'nonce-test-nonce-value'"
        assert "'unsafe-inline'" not in _csp_directive(csp, "script-src")
        assert _csp_directive(csp, "script-src-attr") == "script-src-attr 'none'"


def test_health_healthy_db(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 1}])
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **k: True)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    # Round-2 M1 added `limiter` and `degraded` fields. Assert the stable subset
    # rather than full equality so future fields don't break this test.
    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"
    assert payload["limiter"] in ("redis", "memory")
    assert payload["face_service"] == "reachable"
    assert payload["degraded"] is (payload["limiter"] == "memory")


def test_health_degraded_when_face_service_unhealthy(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **k: [{"ok": 1}])
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **k: False)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"
    assert payload["face_service"] == "unreachable"
    assert payload["degraded"] is True


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
    # The 404 page renders the branded "Face Not Found" UI rather than the raw
    # werkzeug "Page not found" description (which was removed as redundant —
    # users see the title, scanner animation, and ERR_TARGET_ESCAPED_CLASSIFICATION
    # chip; the literal HTTP description added no information).
    assert b"Face Not Found" in response.data
    assert b"ERR_TARGET_ESCAPED_CLASSIFICATION" in response.data
    assert b"404" in response.data


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


def test_create_app_returns_app():
    # Compare within the module: an integration-mode test may have reloaded
    # app and rebound app_module.app after flask_app was captured at import.
    assert app_module.create_app() is app_module.app


def test_service_worker_route_sets_required_headers():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/sw.js")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/javascript"
    assert response.headers["Service-Worker-Allowed"] == "/"
    assert response.headers["Cache-Control"] == "no-cache"


def test_chrome_devtools_json_returns_empty_json():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/.well-known/appspecific/com.chrome.devtools.json")

    assert response.status_code == 200
    assert response.get_json() == {}


def test_main_guard_runs_app_on_loopback_in_debug(monkeypatch):
    """The Werkzeug debugger allows code execution, so debug never binds beyond loopback."""
    kwargs = _run_main_guard({"PORT": "8765", "FLASK_DEBUG": "1"})

    assert kwargs["debug"] is True
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8765


def test_main_guard_binds_all_interfaces_without_debug(monkeypatch):
    kwargs = _run_main_guard({"PORT": "8766", "FLASK_DEBUG": "0"})

    assert kwargs["debug"] is False
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8766


def test_robots_txt_returns_plain_text():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.content_type == "text/plain; charset=utf-8"
    body = response.data.decode()
    assert "User-agent: *" in body
    assert "Allow: /$" in body
    assert "Allow: /leaderboard" in body
    assert "Disallow: /" in body
    assert "Sitemap:" in body
    assert "sitemap.xml" in body


def test_sitemap_xml_returns_valid_xml():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/sitemap.xml")

    assert response.status_code == 200
    assert response.content_type == "application/xml; charset=utf-8"
    body = response.data.decode()

    # Parse XML to ensure it is well-formed and has the expected structure.
    root = ET.fromstring(body)
    assert root.tag.endswith("urlset")

    loc_texts = [(elem.text or "") for elem in root.iter() if elem.tag.endswith("loc")]
    assert any("/leaderboard" in text for text in loc_texts)


def test_sitemap_xml_escapes_url_root():
    flask_app.config["TESTING"] = True
    flask_app.config["SERVER_NAME"] = "evil.com/<script>"
    try:
        client = flask_app.test_client()
        response = client.get("/sitemap.xml", base_url="http://evil.com/<script>/")
        body = response.data.decode()

        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        root = ET.fromstring(body)
        assert root.tag.endswith("urlset")
    finally:
        flask_app.config["SERVER_NAME"] = None


def test_robots_txt_escapes_url_root():
    flask_app.config["TESTING"] = True
    flask_app.config["SERVER_NAME"] = "evil.com/<script>"
    try:
        client = flask_app.test_client()
        response = client.get("/robots.txt", base_url="http://evil.com/<script>/")
        body = response.data.decode()

        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert "Sitemap:" in body
    finally:
        flask_app.config["SERVER_NAME"] = None


def test_inject_liftwing_status_up(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [_liftwing_status_row("up")])

    data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is False
    assert data["liftwing_status"] == "up"


def test_inject_liftwing_status_down(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [_liftwing_status_row("down")])

    data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is True
    assert data["liftwing_status"] == "down"
    assert data["liftwing_last_checked"] is not None


def test_inject_liftwing_status_stale_down_reads_unknown(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [_liftwing_status_row("down", is_stale=1)])

    data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is False
    assert data["liftwing_status"] == "unknown"


def test_inject_liftwing_status_missing_row_is_unknown(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [])

    data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is False
    assert data["liftwing_status"] == "unknown"
    assert data["liftwing_last_checked"] is None


def test_inject_liftwing_status_skips_lookup_on_non_project_pages(monkeypatch):
    """L4: only project detail/settings show the banner; other pages skip the query."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("the context processor must not query on this page")

    monkeypatch.setattr(app_module, "execute_query", _boom)

    with app_module.app.test_request_context("/dashboard"):
        data = app_module.inject_liftwing_status()

    assert data == {"liftwing_down": False, "liftwing_status": "unknown", "liftwing_last_checked": None}


def test_inject_liftwing_status_still_queries_on_project_pages(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [_liftwing_status_row("down")])

    with app_module.app.test_request_context("/project/1"):
        data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is True


def test_inject_liftwing_status_db_error_is_unknown(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("database down")

    monkeypatch.setattr(app_module, "execute_query", boom)

    data = app_module.inject_liftwing_status()

    assert data["liftwing_down"] is False
    assert data["liftwing_status"] == "unknown"


def test_health_liftwing_disabled_when_unused(monkeypatch):
    monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", True)
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
    monkeypatch.setattr(app_module, "execute_query", _health_execute_query(0, None))

    response = flask_app.test_client().get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["liftwing"] == "disabled"
    assert data["degraded"] is False


def test_health_liftwing_reachable(monkeypatch):
    monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", True)
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
    monkeypatch.setattr(app_module, "execute_query", _health_execute_query(1, _liftwing_status_row("up")))

    data = flask_app.test_client().get("/health").get_json()

    assert data["liftwing"] == "reachable"
    assert data["degraded"] is False


def test_health_liftwing_unreachable_does_not_degrade(monkeypatch):
    monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", True)
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
    monkeypatch.setattr(app_module, "execute_query", _health_execute_query(1, _liftwing_status_row("down")))

    data = flask_app.test_client().get("/health").get_json()

    assert data["liftwing"] == "unreachable"
    assert data["degraded"] is False


def test_health_liftwing_disabled_when_status_stale(monkeypatch):
    monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", True)
    monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
    row = _liftwing_status_row("down", is_stale=1)
    monkeypatch.setattr(app_module, "execute_query", _health_execute_query(1, row))

    data = flask_app.test_client().get("/health").get_json()

    assert data["liftwing"] == "disabled"
