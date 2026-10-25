"""Tests for the web app: misc."""

import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from database import DatabaseError
from tests._app_preamble import (
    _is_safe_url,
    _make_embedding,
    app_module,
    commons_thumb_url,
    flask_app,
    require_sample_env,
)
from tests._app_support import (
    _auth_client_chunk4,
    _empty_prefilter_stats_row,
    _make_invite_code_collision_error,
    _prefilter_stats_row,
    _set_csrf_chunk4,
)


def test_require_sample_env_accepts_only_explicit_affirmatives(monkeypatch):
    """L2: WIKIVISAGE_REQUIRE_SAMPLE=0 must mean "not required", not truthy."""
    for value, expected in [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
    ]:
        if value is None:
            monkeypatch.delenv("WIKIVISAGE_REQUIRE_SAMPLE", raising=False)
        else:
            monkeypatch.setenv("WIKIVISAGE_REQUIRE_SAMPLE", value)
        assert require_sample_env() is expected, value


def test_safe_relative_path():
    with flask_app.test_request_context():
        assert _is_safe_url("/dashboard") is True


def test_safe_relative_with_query():
    with flask_app.test_request_context():
        assert _is_safe_url("/project/1?tab=model") is True


def test_unsafe_absolute_url():
    with flask_app.test_request_context():
        assert _is_safe_url("https://evil.com/phish") is False


def test_unsafe_protocol_relative():
    with flask_app.test_request_context():
        assert _is_safe_url("//evil.com/phish") is False


def test_empty_string():
    with flask_app.test_request_context():
        assert _is_safe_url("") is False


def test_safe_plain_path():
    with flask_app.test_request_context():
        assert _is_safe_url("dashboard") is True


def test_basic_jpg():
    url = commons_thumb_url("File:Example.jpg", 330)
    assert "/330px-Example.jpg" in url


def test_strips_file_prefix():
    with_prefix = commons_thumb_url("File:Test.jpg", 330)
    without_prefix = commons_thumb_url("Test.jpg", 330)
    assert with_prefix == without_prefix


def test_spaces_to_underscores():
    url = commons_thumb_url("File:My Photo.jpg", 330)
    assert "My_Photo.jpg" in url


def test_svg_becomes_png():
    url = commons_thumb_url("File:Logo.svg", 330)
    assert url.endswith(".png")


def test_tiff_becomes_jpg():
    url = commons_thumb_url("File:Scan.tiff", 330)
    assert url.endswith(".jpg")


def test_webm_double_dash():
    url = commons_thumb_url("File:Video.webm", 330)
    assert "px--Video.webm.jpg" in url


def test_width_snapped():
    url = commons_thumb_url("File:X.jpg", 100)
    assert "/120px-" in url


def test_default_width():
    url = commons_thumb_url("File:X.jpg")
    assert "/330px-" in url


def test_ogv_video():
    url = commons_thumb_url("File:Clip.ogv", 330)
    assert "px--Clip.ogv.jpg" in url


def test_tif_extension():
    url = commons_thumb_url("File:Doc.tif", 330)
    assert url.endswith(".jpg")


def test_session_cleared_on_login_callback(monkeypatch):
    """OAuth callback should clear session before setting user_id (session fixation prevention)."""
    client = flask_app.test_client()

    fake_token = {"access_token": "tok", "refresh_token": "ref", "expires_at": 9999999999, "expires_in": 14400}
    fake_userinfo = {"sub": 42, "username": "TestUser"}
    fake_user_row = [
        {
            "id": 1,
            "wiki_user_id": 42,
            "wiki_username": "TestUser",
            "access_token": "tok",
            "refresh_token": "ref",
            "token_expires_at": datetime.now(UTC),
        }
    ]

    monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: fake_user_row)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_userinfo

    monkeypatch.setattr(app_module.requests, "get", lambda *a, **kw: FakeResp())

    class FakeOAuth:
        def fetch_token(self, *a, **kw):
            return fake_token

    monkeypatch.setattr(app_module, "_make_oauth_session", lambda state=None: FakeOAuth())

    with client.session_transaction() as sess:
        sess["oauth_state"] = "fakestate"
        sess["login_next"] = "/dashboard"
        sess["attacker_data"] = "should_be_removed"

    client.get("/auth/callback?state=fakestate&code=fakecode")

    with client.session_transaction() as sess:
        assert sess.get("user_id") == 1
        assert "attacker_data" not in sess


def test_wake_file_path_consistent():
    """Both app.py and worker.py should import WAKE_FILE_PATH from config."""
    import inspect

    import config as config_module

    # config.py must define the canonical path using '.worker-wake-up'
    config_source = inspect.getsource(config_module)
    assert ".worker-wake-up" in config_source, "config.py must define WAKE_FILE_PATH with '.worker-wake-up'"

    # app.py must import WAKE_FILE_PATH from config (not define its own)
    app_source = inspect.getsource(app_module)
    assert re.search(r"^from config import .*\bWAKE_FILE_PATH\b", app_source, re.M), (
        "app.py must import WAKE_FILE_PATH from config"
    )


def test_set_security_headers_sets_all_required_headers():
    with flask_app.test_request_context():
        response = flask_app.make_response(("ok", 200))
        result = app_module.set_security_headers(response)

        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-Frame-Options"] == "DENY"
        assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert result.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
        assert (
            "Strict-Transport-Security" not in result.headers
            or result.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
        )


def test_hsts_header_not_set_in_debug_mode():
    # set_security_headers reads the module-level app, and an integration-mode
    # test may have reloaded the module, so both must be addressed through
    # app_module rather than the module-level flask_app reference.
    application = app_module.app
    original_debug = application.debug
    application.debug = True
    try:
        with application.test_request_context():
            response = application.make_response(("ok", 200))
            result = app_module.set_security_headers(response)
            assert "Strict-Transport-Security" not in result.headers
    finally:
        application.debug = original_debug


def test_hsts_header_set_when_not_debug():
    application = app_module.app
    original_debug = application.debug
    application.debug = False
    try:
        with application.test_request_context():
            response = application.make_response(("ok", 200))
            result = app_module.set_security_headers(response)
            assert result.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    finally:
        application.debug = original_debug


def test_app_allowlist_still_rejects_non_wikimedia_hosts():
    """Adding thumb.wikimedia.org must not widen the allowlist to lookalikes."""
    expected = frozenset({"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"})
    assert expected == app_module._ALLOWED_DOWNLOAD_HOSTS


def test_make_oauth_session_uses_expected_params(monkeypatch):
    captured = {}

    class _FakeOAuth2Session:
        def __init__(self, client_id, redirect_uri=None, state=None):
            captured["client_id"] = client_id
            captured["redirect_uri"] = redirect_uri
            captured["state"] = state
            self.headers = {}

    monkeypatch.setattr(app_module, "OAuth2Session", _FakeOAuth2Session)
    sess = app_module._make_oauth_session(state="abc")

    assert captured["client_id"] == app_module.OAUTH_CLIENT_ID
    assert captured["redirect_uri"] == app_module.OAUTH_REDIRECT_URI
    assert captured["state"] == "abc"
    assert sess.headers["User-Agent"] == app_module.USER_AGENT


def test_propertymock_is_used():
    class _Holder:
        @property
        def value(self):
            return "old"

    obj = _Holder()
    with patch.object(_Holder, "value", new_callable=PropertyMock, return_value="new"):
        assert obj.value == "new"


def test_templates_never_put_tojson_in_a_double_quoted_attribute():
    """``|tojson`` emits raw double quotes, so inside a double-quoted attribute the
    rendered value closes the attribute and injects markup (owner-only forms) or
    breaks the inline handler (translated strings with spaces). Inline handlers must
    single-quote the attribute and double-quote the JS placeholder instead."""
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    pattern = re.compile(r'="[^"]*tojson[^"]*"', re.DOTALL)
    offenders = []
    for path in sorted(templates_dir.glob("*.html")):
        source = path.read_text(encoding="utf-8")
        for match in pattern.finditer(source):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}: {match.group(0)[:120]}")
    assert offenders == []


class TestChunk4ApiClassify:
    def test_api_classify_missing_fields(self, monkeypatch):
        client, _ = _auth_client_chunk4(monkeypatch)
        _set_csrf_chunk4(client)
        response = client.post(
            "/api/classify",
            data={"csrf_token": "testtoken", "project_id": "1", "image_id": "2"},
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == "Missing required fields"


def test_is_invite_code_collision_true():
    exc = _make_invite_code_collision_error()
    assert app_module._is_invite_code_collision(exc)


def test_is_invite_code_collision_false_wrong_key():
    cause = Exception(1062, "Duplicate entry '...' for key 'idx_projects_user_qid_cat'")
    exc = app_module.DatabaseError("duplicate project")
    exc.__cause__ = cause
    assert not app_module._is_invite_code_collision(exc)


def test_is_invite_code_collision_false_not_1062():
    cause = Exception(1064, "Syntax error")
    exc = app_module.DatabaseError("syntax error")
    exc.__cause__ = cause
    assert not app_module._is_invite_code_collision(exc)


def test_app_refresh_token_rowcount_zero_reads_fresh(monkeypatch):
    """When UPDATE rowcount==0 in app token refresh, fresh token is re-read from DB."""
    flask_app.config["TESTING"] = True

    # Import the private refresh function
    from datetime import UTC, datetime, timedelta

    from app import _refresh_access_token

    past = datetime.now(UTC) - timedelta(hours=1)
    user = {
        "id": 1,
        "wiki_username": "tester",
        "access_token": "old-token",
        "refresh_token": "old-refresh",
        "token_expires_at": past,
    }

    query_calls = [0]

    # First SELECT: the credentials read again under the refresh lock (still the stale row).
    # Second SELECT: the post-CAS re-read, after the peer's refresh landed.
    select_results = [
        [{"access_token": "old-token", "refresh_token": "old-refresh", "token_expires_at": past}],
        [{"access_token": "fresh-token", "refresh_token": "fresh-refresh", "token_expires_at": None}],
    ]

    def mock_execute_query(sql, params=None, fetch=True):
        query_calls[0] += 1
        if "UPDATE users SET access_token" in sql:
            return 0  # rowcount == 0 — another process refreshed first
        if "SELECT access_token, refresh_token, token_expires_at FROM users" in sql:
            return select_results.pop(0)
        return []

    new_token_data = {
        "access_token": "new-token",
        "refresh_token": "new-refresh",
        "expires_in": 14400,
    }

    with (
        flask_app.test_request_context(),
        patch("app.execute_query", side_effect=mock_execute_query),
        patch("app.encrypt_token", side_effect=lambda t: t),
        patch("app.decrypt_token", side_effect=lambda t: t),
        patch("app.OAuth2Session") as mock_oauth_cls,
    ):
        mock_oauth_instance = MagicMock()
        mock_oauth_instance.refresh_token.return_value = new_token_data
        mock_oauth_cls.return_value = mock_oauth_instance

        result = _refresh_access_token(user)

    # When rowcount==0, fresh token from DB should be returned via the user dict
    assert result is not None
    assert result.get("access_token") == "fresh-token"


class TestAssertProjectAccessLocked:
    """``_assert_project_access_locked`` issues a ``SELECT ... FOR UPDATE`` and
    raises ``AccessRevoked`` if the actor no longer has access. This closes a
    TOCTOU window where a project owner could revoke a member between the
    pre-transaction permission check and the actual mutation."""

    def test_raises_when_no_row(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        with pytest.raises(app_module.AccessRevoked):
            app_module._assert_project_access_locked(cursor, project_id=42, user_id=7)
        # Verify FOR UPDATE locking was used
        sql = cursor.execute.call_args[0][0]
        assert "FOR UPDATE" in sql
        assert "project_members" in sql
        # Bind order: (user_id, project_id, user_id)
        assert cursor.execute.call_args[0][1] == (7, 42, 7)

    def test_passes_when_row_returned(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (1,)
        # Should not raise
        app_module._assert_project_access_locked(cursor, project_id=42, user_id=7)


class TestIsAccessRevoked:
    def test_direct_access_revoked(self):
        exc = app_module.AccessRevoked("nope")
        assert app_module._is_access_revoked(exc) is True

    def test_wrapped_in_database_error(self):
        from database import DatabaseError

        cause = app_module.AccessRevoked("nope")
        try:
            raise DatabaseError("boom") from cause
        except DatabaseError as e:
            assert app_module._is_access_revoked(e) is True

    def test_unrelated_exception(self):
        assert app_module._is_access_revoked(ValueError("x")) is False
        from database import DatabaseError

        assert app_module._is_access_revoked(DatabaseError("x")) is False

    def test_no_infinite_loop_on_self_referential_cause(self):
        e = ValueError("loop")
        e.__cause__ = e  # pathological self-cycle
        # Should terminate, not hang
        assert app_module._is_access_revoked(e) is False


class TestRunAuthorizedTransaction:
    def test_passes_through_normal_result(self, monkeypatch):
        monkeypatch.setattr(app_module, "execute_transaction", lambda op: "result")
        assert app_module._run_authorized_transaction(lambda c, cur: None) == "result"

    def test_aborts_403_on_access_revoked(self, monkeypatch):
        from database import DatabaseError

        cause = app_module.AccessRevoked("revoked")

        def fake_tx(_op):
            raise DatabaseError("wrapped") from cause

        monkeypatch.setattr(app_module, "execute_transaction", fake_tx)
        # abort(403) raises a Werkzeug HTTPException with code 403
        from werkzeug.exceptions import HTTPException

        with pytest.raises(HTTPException) as excinfo:
            app_module._run_authorized_transaction(lambda c, cur: None)
        assert excinfo.value.code == 403

    def test_reraises_other_database_errors(self, monkeypatch):
        from database import DatabaseError

        def fake_tx(_op):
            raise DatabaseError("real db error")

        monkeypatch.setattr(app_module, "execute_transaction", fake_tx)
        with pytest.raises(DatabaseError):
            app_module._run_authorized_transaction(lambda c, cur: None)

    def test_unwraps_value_error_from_cause_chain(self, monkeypatch):
        """Regression: ``execute_transaction`` wraps every exception in
        ``DatabaseError(...) from e``. Route handlers signal control flow via
        ``ValueError`` (e.g. ``ValueError("already_reviewed")`` in
        ``_reclassify``); the helper must unwrap it so the route's
        ``except ValueError`` handler runs and returns 4xx instead of 500."""
        from database import DatabaseError

        cause = ValueError("already_reviewed")

        def fake_tx(_op):
            raise DatabaseError(f"Transaction execution failed: {cause}") from cause

        monkeypatch.setattr(app_module, "execute_transaction", fake_tx)
        with pytest.raises(ValueError) as excinfo:
            app_module._run_authorized_transaction(lambda c, cur: None)
        assert str(excinfo.value) == "already_reviewed"
        # Unwrapped ValueError must keep DatabaseError as its cause for tracing
        assert isinstance(excinfo.value.__cause__, DatabaseError)

    def test_unwraps_value_error_through_nested_cause_chain(self, monkeypatch):
        """ValueError nested deeper than one level is still recovered."""
        from database import DatabaseError

        original = ValueError("custom_signal")
        intermediate = RuntimeError("middle")
        intermediate.__cause__ = original

        def fake_tx(_op):
            raise DatabaseError("wrapped twice") from intermediate

        monkeypatch.setattr(app_module, "execute_transaction", fake_tx)
        with pytest.raises(ValueError) as excinfo:
            app_module._run_authorized_transaction(lambda c, cur: None)
        assert str(excinfo.value) == "custom_signal"

    def test_access_revoked_takes_precedence_over_value_error(self, monkeypatch):
        """If both AccessRevoked and ValueError are in the chain, 403 wins
        (security > control-flow)."""
        from werkzeug.exceptions import HTTPException

        from database import DatabaseError

        revoked = app_module.AccessRevoked("revoked")
        value_err = ValueError("would_be_409")
        revoked.__cause__ = value_err

        def fake_tx(_op):
            raise DatabaseError("wrapped") from revoked

        monkeypatch.setattr(app_module, "execute_transaction", fake_tx)
        with pytest.raises(HTTPException) as excinfo:
            app_module._run_authorized_transaction(lambda c, cur: None)
        assert excinfo.value.code == 403


class TestFindInCauseChain:
    def test_returns_match_at_top(self):
        e = ValueError("hit")
        assert app_module._find_in_cause_chain(e, ValueError) is e

    def test_walks_chain_to_find_type(self):
        from database import DatabaseError

        original = ValueError("deep")
        wrapper = DatabaseError("outer")
        wrapper.__cause__ = original
        found = app_module._find_in_cause_chain(wrapper, ValueError)
        assert found is original

    def test_returns_none_when_type_absent(self):
        from database import DatabaseError

        e = DatabaseError("no value error here")
        assert app_module._find_in_cause_chain(e, ValueError) is None

    def test_no_infinite_loop_on_self_referential_cause(self):
        e = RuntimeError("loop")
        e.__cause__ = e
        # Should terminate, not hang. RuntimeError is not ValueError.
        assert app_module._find_in_cause_chain(e, ValueError) is None

    def test_handles_none_input(self):
        # Defensive: walking None should just return None
        assert app_module._find_in_cause_chain(None, ValueError) is None  # type: ignore[arg-type]


def test_schema_default_threshold_matches_app_and_migration_default():
    import migrate

    schema = (Path(__file__).resolve().parent.parent / "schema.sql").read_text(encoding="utf-8")
    assert f"DEFAULT {app_module.DEFAULT_DISTANCE_THRESHOLD} COMMENT 'Cosine distance threshold" in schema
    assert migrate.DEFAULT_DISTANCE_THRESHOLD == app_module.DEFAULT_DISTANCE_THRESHOLD


def test_get_prefilter_stats_computes_skip_pct(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_k: [_prefilter_stats_row(checked=3, dropped=1)])

    stats = app_module._get_prefilter_stats(1)

    assert stats["checked"] == 3
    assert stats["skip_pct"] == 33.3
    assert stats["last_checked"] == datetime(2026, 10, 17, 9, 30)


def test_get_prefilter_stats_zero_checked(monkeypatch):
    monkeypatch.setattr(app_module, "execute_query", lambda *_a, **_k: [_empty_prefilter_stats_row()])

    stats = app_module._get_prefilter_stats(1)

    assert stats["skip_pct"] == 0.0
    assert stats["last_checked"] is None


def test_get_prefilter_stats_db_error_returns_empty(monkeypatch):
    def boom(*_a, **_k):
        raise DatabaseError("boom")

    monkeypatch.setattr(app_module, "execute_query", boom)

    assert app_module._get_prefilter_stats(1) == {}


@pytest.mark.integration
def test_inference_preserves_sibling_approved_after_candidate_selection(
    integration_client, db_conn, seed_project, seed_images, seed_faces, monkeypatch
):
    """A gallery approval during inference must not leave two targets on an image."""
    client, _, module = integration_client
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO images (project_id, commons_page_id, file_title, status, face_count, "
            "detection_width, detection_height) VALUES (%s, 999001, 'File:Race.jpg', 'processed', 2, 800, 600)",
            (seed_project["id"],),
        )
        image_id = cur.lastrowid
        face_ids = []
        for seed in (700, 701):
            cur.execute(
                "INSERT INTO faces (image_id, encoding, embedding_model, bbox_top, bbox_right, "
                "bbox_bottom, bbox_left) VALUES (%s, %s, 'scrfd500m+arcface_mbf', 100, 300, 300, 100)",
                (image_id, _make_embedding(seed)),
            )
            face_ids.append(cur.lastrowid)

    with client.session_transaction() as sess:
        sess["csrf_token"] = "testtoken"

    import worker

    real_query = worker.execute_query
    approved = False

    def approve_after_snapshot(sql, params=None, fetch=True):
        nonlocal approved
        rows = real_query(sql, params, fetch)
        if "SELECT f.id, f.image_id, f.encoding" in sql:
            assert {row["id"] for row in rows} == set(face_ids)
            response = client.post(
                "/api/reclassify",
                data={"face_id": face_ids[1], "is_target": "1", "csrf_token": "testtoken"},
            )
            assert response.status_code == 200
            approved = True
            return sorted(rows, key=lambda row: row["id"])
        return rows

    monkeypatch.setattr(module, "_maybe_mark_sdc_written", lambda *args: False)
    monkeypatch.setattr(worker, "execute_query", approve_after_snapshot)
    monkeypatch.setattr(worker, "_face_distance", lambda *args: 0.1)
    monkeypatch.setattr(worker, "shutdown_requested", False)
    classified = worker.run_autonomous_inference(seed_project)

    assert approved
    assert classified == 1  # The newly human-approved face is no longer a candidate.
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, is_target, classified_by FROM faces WHERE image_id = %s ORDER BY id", (image_id,))
        rows = cur.fetchall()
    assert [(row["is_target"], row["classified_by"]) for row in rows] == [(0, "model"), (1, "human")]
