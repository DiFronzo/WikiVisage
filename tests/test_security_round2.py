"""Round-2 security patch tests.

Covers (in order):
- H1: CSP nonce-only script-src, script-src-attr 'none', object-src/base-uri/frame-ancestors
- H2: OAuth refresh single-flight Redis lock (redis_lock.single_flight)
- M1: /health surfaces Redis fallback as `degraded: true`
- M2: PKCE (S256) on the OAuth authorize/token flow
- M3: Per-face cancel re-check inside SDC write batch
- M4: _remove_sdc_claim treats already-gone claims as success

All tests are unit-level: pure mocks, no DB or Redis required.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Match the env bootstrap pattern used by tests/test_app.py and tests/test_worker.py
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
    import worker as worker_module

import redis_lock


@pytest.fixture(autouse=True)
def _normalize_server_name():
    """Flask's default config has ``SERVER_NAME = None``. Some other tests in
    the suite explicitly ``pop()`` the key (which makes ``app.config["SERVER_NAME"]``
    raise KeyError inside ``create_url_adapter`` when the test client builds
    URLs). Restore the proper default before each test in this file."""
    flask_app = app_module.app
    prior_present = "SERVER_NAME" in flask_app.config
    prior_value = flask_app.config.get("SERVER_NAME")
    # Force Flask's documented default so ``create_url_adapter`` doesn't blow up.
    flask_app.config["SERVER_NAME"] = None
    try:
        yield
    finally:
        if prior_present:
            flask_app.config["SERVER_NAME"] = prior_value
        else:
            flask_app.config.pop("SERVER_NAME", None)


# ---------------------------------------------------------------------------
# H1 — CSP hardening (object-src / base-uri / frame-ancestors)
# ---------------------------------------------------------------------------


class TestCSPHardening:
    """Round-2 H1: object-src / base-uri / frame-ancestors, plus a nonce-only
    script-src.

    Scripts are allowed only by the per-request nonce (`script-src 'self'
    'nonce-...'`) and inline event-handler attributes are forbidden by
    `script-src-attr 'none'`. Per CSP Level 3 a nonce makes browsers ignore
    `'unsafe-inline'` in the same directive, so `script-src` must never carry
    both. `style-src` keeps `'unsafe-inline'` (many style="" attributes remain)
    and has no nonce, so it is unaffected.
    """

    def test_csp_header_contains_required_directives(self):
        client = app_module.app.test_client()
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "object-src 'none'" in csp, "must block <object>/<embed>/<applet>"
        assert "base-uri 'none'" in csp, "must block <base> XSS escalation"
        assert "frame-ancestors 'none'" in csp, "must block clickjacking"
        assert "default-src 'self'" in csp, "baseline restriction"

    @staticmethod
    def _directive(csp, name):
        for part in csp.split(";"):
            part = part.strip()
            if part.startswith(name + " "):
                return part
        raise AssertionError(f"{name} missing from CSP: {csp}")

    def test_script_src_is_nonce_only_without_unsafe_inline(self):
        client = app_module.app.test_client()
        csp = client.get("/health").headers.get("Content-Security-Policy", "")
        script_src = self._directive(csp, "script-src")
        assert "'unsafe-inline'" not in script_src
        assert re.fullmatch(r"script-src 'self' 'nonce-[A-Za-z0-9_-]{32,}'", script_src), script_src

    def test_script_src_attr_forbids_inline_handlers(self):
        client = app_module.app.test_client()
        csp = client.get("/health").headers.get("Content-Security-Policy", "")
        assert self._directive(csp, "script-src-attr") == "script-src-attr 'none'"

    def test_style_src_keeps_unsafe_inline_for_style_attributes(self):
        client = app_module.app.test_client()
        csp = client.get("/health").headers.get("Content-Security-Policy", "")
        assert self._directive(csp, "style-src") == "style-src 'self' 'unsafe-inline'"

    def test_header_nonce_differs_per_request(self):
        client = app_module.app.test_client()
        nonces = {
            re.search(r"'nonce-([^']+)'", client.get("/health").headers["Content-Security-Policy"]).group(1)
            for _ in range(3)
        }
        assert len(nonces) == 3

    def test_header_nonce_matches_the_nonce_on_rendered_script_tags(self):
        """The nonce in the header must be the very value templates render."""
        client = app_module.app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        header_nonce = re.search(r"'nonce-([^']+)'", resp.headers["Content-Security-Policy"]).group(1)
        html = resp.get_data(as_text=True)
        script_nonces = set(re.findall(r'<script\b[^>]*\bnonce="([^"]+)"', html))
        assert script_nonces == {header_nonce}
        assert len(re.findall(r"<script\b", html)) == len(re.findall(r'<script\b[^>]*\bnonce="', html))

    def test_per_request_csp_nonce_still_generated_for_templates(self):
        """`g.csp_nonce` is populated on every request; the header and the
        `{{ csp_nonce }}` template variable both read it."""
        with app_module.app.test_request_context("/"):
            # Trigger before_request manually
            for func in app_module.app.before_request_funcs.get(None, []):
                func()
            from flask import g

            assert hasattr(g, "csp_nonce"), "g.csp_nonce must be set per request"
            assert isinstance(g.csp_nonce, str)
            assert len(g.csp_nonce) >= 32, "nonce must have >=32 chars of entropy"


class TestTemplatesAreCSPClean:
    """Static guard: with `script-src-attr 'none'` and nonce-only `script-src`,
    a single inline handler or un-nonced <script> silently stops working in
    production. Check the Jinja sources so a regression fails in CI."""

    TEMPLATES = sorted((Path(__file__).resolve().parent.parent / "templates").glob("*.html"))

    @staticmethod
    def _markup_without_script_bodies(source):
        source = re.sub(r"\{#.*?#\}", "", source, flags=re.DOTALL)
        return re.sub(r"(<script\b[^>]*>).*?(</script\b[^>]*>)", r"\1\2", source, flags=re.DOTALL | re.IGNORECASE)

    def test_templates_found(self):
        assert len(self.TEMPLATES) >= 10

    def test_no_inline_event_handler_attributes(self):
        pattern = re.compile(r"""[\s"'/]on[a-z]{3,}\s*=""", re.IGNORECASE)
        offenders = []
        for path in self.TEMPLATES:
            markup = self._markup_without_script_bodies(path.read_text(encoding="utf-8"))
            for match in pattern.finditer(markup):
                line = markup[: match.start()].count("\n") + 1
                offenders.append(f"{path.name}: {markup.splitlines()[line - 1].strip()[:100]}")
        assert not offenders, "inline event handlers are blocked by the CSP:\n" + "\n".join(offenders)

    def test_no_javascript_urls(self):
        for path in self.TEMPLATES:
            assert not re.search(r"javascript\s*:", path.read_text(encoding="utf-8"), re.IGNORECASE), path.name

    def test_every_script_tag_carries_the_nonce(self):
        offenders = []
        for path in self.TEMPLATES:
            for tag in re.findall(r"<script\b[^>]*>", path.read_text(encoding="utf-8"), re.IGNORECASE):
                if 'nonce="{{ csp_nonce }}"' not in tag:
                    offenders.append(f"{path.name}: {tag}")
        assert not offenders, "<script> without nonce is blocked by the CSP:\n" + "\n".join(offenders)

    def test_no_window_alert_calls(self):
        for path in self.TEMPLATES:
            assert not re.search(r"(?<![\w.])alert\(", path.read_text(encoding="utf-8")), path.name


# ---------------------------------------------------------------------------
# H2 — Redis single-flight lock (redis_lock.single_flight)
# ---------------------------------------------------------------------------


class TestSingleFlightLock:
    def test_no_redis_yields_true(self):
        """`redis_client=None` (Redis disabled at startup) must yield True so
        callers proceed unprotected — preserves liveness in dev / outage."""
        with redis_lock.single_flight(None, "any", ttl_seconds=10) as got:
            assert got is True

    def test_acquires_when_setnx_succeeds(self):
        client = MagicMock()
        client.set.return_value = True  # NX SET succeeded
        with redis_lock.single_flight(client, "user:1", ttl_seconds=15) as got:
            assert got is True
        # Verify the SET call used NX + EX
        args, kwargs = client.set.call_args
        assert kwargs.get("nx") is True
        assert kwargs.get("ex") == 15
        assert args[0] == "wikivisage:lock:user:1"

    def test_returns_false_when_lock_held_by_peer(self):
        client = MagicMock()
        client.set.return_value = None  # NX SET refused — peer holds it
        with redis_lock.single_flight(client, "user:1") as got:
            assert got is False
        # Must NOT call delete on a lock we never owned
        client.delete.assert_not_called()

    def test_redis_outage_during_acquire_yields_true(self):
        """If Redis raises during SETNX we fail-open (yield True) so DB CAS
        remains the only safety net. Better than hard-failing the request."""
        client = MagicMock()
        client.set.side_effect = ConnectionError("redis down")
        with redis_lock.single_flight(client, "user:1") as got:
            assert got is True

    def test_releases_only_own_token(self):
        """The release path must use an atomic Lua compare-and-delete — prevents
        releasing a lock that already expired and was reacquired by a peer."""
        from redis_lock import _LUA_RELEASE

        client = MagicMock()
        client.set.return_value = True
        captured_token = {}

        def capture_set(*args, **kwargs):
            captured_token["v"] = args[1]
            return True

        client.set.side_effect = capture_set
        with redis_lock.single_flight(client, "user:1") as got:
            assert got is True
        # Verify the Lua script is invoked with the correct key and token.
        client.eval.assert_called_once_with(_LUA_RELEASE, 1, "wikivisage:lock:user:1", captured_token["v"])
        # Direct DEL should never be called — the Lua script performs it atomically.
        client.delete.assert_not_called()

    def test_does_not_release_foreign_token(self):
        """If the lock expired and was reacquired by a peer, we must not DEL it.
        The Lua script handles the comparison atomically — direct DEL is never used."""
        client = MagicMock()
        client.set.return_value = True
        with redis_lock.single_flight(client, "user:1") as _got:
            pass
        # Lua eval is called (the script handles the compare internally).
        assert client.eval.call_count == 1
        # Direct DEL must never be issued from Python code.
        client.delete.assert_not_called()

    def test_release_redis_outage_is_swallowed(self):
        client = MagicMock()
        client.set.return_value = True
        client.eval.side_effect = ConnectionError("redis down at release")
        # Must not raise out of the with-block
        with redis_lock.single_flight(client, "user:1") as got:
            assert got is True


# ---------------------------------------------------------------------------
# M1 — /health surfaces Redis fallback
# ---------------------------------------------------------------------------


class TestHealthRedisFallback:
    def test_health_reports_limiter_and_degraded(self, monkeypatch):
        # Pretend Redis-backed limiter is up
        monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", True)
        monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [{"ok": 1}])
        monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
        client = app_module.app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["limiter"] == "redis"
        assert data["degraded"] is False

    def test_health_reports_degraded_when_redis_down(self, monkeypatch):
        monkeypatch.setattr(app_module, "_LIMITER_REDIS_OK", False)
        monkeypatch.setattr(app_module, "execute_query", lambda *a, **kw: [{"ok": 1}])
        monkeypatch.setattr(app_module.face_client, "is_healthy", lambda *a, **kw: True)
        client = app_module.app.test_client()
        resp = client.get("/health")
        # Still 200 OK — DB is healthy; ops scrape `degraded` for alerting.
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["limiter"] == "memory"
        assert data["degraded"] is True

    def test_health_returns_503_when_db_down(self, monkeypatch):
        from database import DatabaseError

        def boom(*_a, **_kw):
            raise DatabaseError("db unreachable")

        monkeypatch.setattr(app_module, "execute_query", boom)
        client = app_module.app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# M2 — PKCE (S256) on OAuth flow
# ---------------------------------------------------------------------------


class TestPKCE:
    def test_login_stores_code_verifier_and_redirects_with_challenge(self):
        client = app_module.app.test_client()
        resp = client.get("/login")
        # /login redirects to the OAuth provider
        assert resp.status_code in (301, 302, 303, 307, 308)
        location = resp.headers["Location"]
        # Authorize URL must include code_challenge + S256 method
        assert "code_challenge=" in location
        assert "code_challenge_method=S256" in location
        # Verifier is in the session
        with client.session_transaction() as sess:
            verifier = sess.get("oauth_code_verifier")
            assert verifier is not None
            # secrets.token_urlsafe(64) → ~86 chars of [A-Za-z0-9_-]
            assert len(verifier) >= 43, "PKCE verifier must be 43-128 chars (RFC 7636)"
            assert len(verifier) <= 128

        # And the challenge in the URL must be SHA256(verifier) base64url-no-pad
        import re
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(location).query)
        challenge = qs["code_challenge"][0]
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        )
        assert challenge == expected
        # And challenge contains only base64url chars (no '=' padding)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", challenge)

    def test_callback_passes_code_verifier_to_token_exchange(self, monkeypatch):
        """The /auth/callback must forward the stored verifier as `code_verifier`
        on the token exchange — otherwise the provider rejects the code."""
        client = app_module.app.test_client()
        # Seed session as if /login just ran
        with client.session_transaction() as sess:
            sess["oauth_state"] = "state-abc"
            sess["oauth_code_verifier"] = "verifier-xyz"

        captured = {}

        class FakeOAuth:
            def __init__(self, *a, **kw):
                pass

            def fetch_token(self, _url, **kwargs):
                captured.update(kwargs)
                return {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}

        monkeypatch.setattr(app_module, "_make_oauth_session", lambda **_kw: FakeOAuth())

        # Make the profile request fail so we short-circuit before DB writes —
        # we only care that fetch_token received `code_verifier`.
        class _BoomResp:
            def raise_for_status(self):
                raise RuntimeError("stop here")

        monkeypatch.setattr(app_module.requests, "get", lambda *a, **kw: _BoomResp())

        resp = client.get("/auth/callback?code=dummy&state=state-abc")
        # We don't care about the final HTTP code — only that the verifier was
        # forwarded.
        assert resp.status_code in (200, 302)
        assert captured.get("code_verifier") == "verifier-xyz"

    def test_callback_without_state_redirects_with_flash(self):
        client = app_module.app.test_client()
        resp = client.get("/auth/callback?code=dummy")
        # No oauth_state in session → redirect to index (302)
        assert resp.status_code in (301, 302, 303, 307, 308)


# ---------------------------------------------------------------------------
# M3 — Per-face cancel re-check inside SDC write batch
# ---------------------------------------------------------------------------


class TestSDCMidBatchCancel:
    """`write_sdc_claims` re-checks `sdc_write_requested` every
    `_PER_FACE_CANCEL_CHECK_EVERY` faces. Without this, a user clicking Stop
    could still see up to SDC_BATCH (=50) extra Commons writes happen before
    the next outer-loop check fires.

    The behaviour (a stop is noticed within five faces; a failed check never
    stops the run) is exercised end to end in tests/test_worker_sdc.py
    (`test_stop_button_is_noticed_within_five_faces` and
    `test_a_failed_mid_batch_cancel_check_does_not_stop_the_write`). These
    tests only pin the configuration those rely on."""

    def test_per_face_cancel_interval_is_small_relative_to_the_batch(self):
        """An interval at or above the batch size would make the inner check useless."""
        assert 1 <= worker_module._PER_FACE_CANCEL_CHECK_EVERY < worker_module._SDC_WRITE_BATCH
        assert worker_module._PER_FACE_CANCEL_CHECK_EVERY <= 5


# ---------------------------------------------------------------------------
# M4 — _remove_sdc_claim treats "already gone" as success
# ---------------------------------------------------------------------------


class TestRemoveSdcClaimIdempotent:
    """If the entity or claim is already absent on Commons (file deleted, peer
    removed it, etc.), the user's desired end-state is satisfied. Surfacing an
    error would just confuse operators. A successful removal also deletes the
    page/QID deduplication row — a surviving ``written_at`` would make a later
    send trust it and skip the actual Commons write."""

    @pytest.fixture(autouse=True)
    def _mock_execute_query(self, monkeypatch):
        """Capture the dedup-row delete; these unit tests carry no database."""
        mock = MagicMock(return_value=1)
        monkeypatch.setattr(app_module, "execute_query", mock)
        return mock

    @staticmethod
    def _assert_dedup_row_cleared(mock, page_id=1, qid="Q42"):
        mock.assert_called_once()
        assert "DELETE FROM sdc_claims" in mock.call_args.args[0]
        assert mock.call_args.args[1] == (page_id, qid)

    @staticmethod
    def _fake_response(payload, status=200):
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status.return_value = None
        r.status_code = status
        return r

    def test_returns_true_when_entity_missing(self, monkeypatch, _mock_execute_query):
        """`wbgetclaims` returns {error: {code: 'no-such-entity'}} when the
        Commons file has no SDC entity at all. We treat this as success."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"error": {"code": "no-such-entity", "info": "M1 not found"}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True
        self._assert_dedup_row_cleared(_mock_execute_query)

    def test_returns_true_when_p180_list_empty(self, monkeypatch, _mock_execute_query):
        """No P180 claims for this QID on the file → nothing to remove → success."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"claims": {"P180": []}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True
        self._assert_dedup_row_cleared(_mock_execute_query)

    def test_returns_true_on_race_during_remove(self, monkeypatch, _mock_execute_query):
        """A peer removed the claim between our wbgetclaims and wbremoveclaims.
        Wikibase returns `no-such-claim` / `no-such-statement`; treat as success."""
        get_calls = {"n": 0}

        def fake_get(url, **_kw):
            get_calls["n"] += 1
            if get_calls["n"] == 1:
                # First GET: wbgetclaims returns a real claim with our QID.
                return self._fake_response(
                    {
                        "claims": {
                            "P180": [
                                {
                                    "id": "M1$abcd",
                                    "mainsnak": {"datavalue": {"value": {"id": "Q42"}}},
                                }
                            ]
                        }
                    }
                )
            # Second GET: CSRF token fetch
            return self._fake_response({"query": {"tokens": {"csrftoken": "csrf-tok"}}})

        def fake_post(_url, **_kw):
            # wbremoveclaims fails because the peer already removed it.
            return self._fake_response({"error": {"code": "no-such-claim", "info": "gone"}})

        monkeypatch.setattr(app_module.requests, "get", fake_get)
        monkeypatch.setattr(app_module.requests, "post", fake_post)
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True
        self._assert_dedup_row_cleared(_mock_execute_query)

    def test_returns_false_on_real_error(self, monkeypatch, _mock_execute_query):
        """A genuine error code (e.g. permissiondenied) must NOT be silently
        swallowed — operators need to know."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"error": {"code": "permissiondenied", "info": "no perm"}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is False
        _mock_execute_query.assert_not_called()

    def test_returns_false_on_network_exception(self, monkeypatch, _mock_execute_query):
        """Unhandled exceptions (timeout, DNS failure) → False."""

        def boom(*_a, **_kw):
            raise ConnectionError("network down")

        monkeypatch.setattr(app_module.requests, "get", boom)
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is False
        _mock_execute_query.assert_not_called()

    def test_happy_path_returns_true(self, monkeypatch, _mock_execute_query):
        """Normal: claim exists, CSRF fetched, removal succeeds."""
        get_calls = {"n": 0}

        def fake_get(*_a, **_kw):
            get_calls["n"] += 1
            if get_calls["n"] == 1:
                return self._fake_response(
                    {
                        "claims": {
                            "P180": [
                                {
                                    "id": "M1$abcd",
                                    "mainsnak": {"datavalue": {"value": {"id": "Q42"}}},
                                }
                            ]
                        }
                    }
                )
            return self._fake_response({"query": {"tokens": {"csrftoken": "csrf-tok"}}})

        monkeypatch.setattr(app_module.requests, "get", fake_get)
        monkeypatch.setattr(app_module.requests, "post", lambda *a, **kw: self._fake_response({"success": 1}))
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True
        self._assert_dedup_row_cleared(_mock_execute_query)

    def test_returns_false_when_dedup_row_delete_fails(self, monkeypatch, _mock_execute_query):
        """The removal is incomplete while the stale written_at row may survive."""
        _mock_execute_query.side_effect = app_module.DatabaseError("pool down")
        get_calls = {"n": 0}

        def fake_get(*_a, **_kw):
            get_calls["n"] += 1
            if get_calls["n"] == 1:
                return self._fake_response(
                    {
                        "claims": {
                            "P180": [
                                {
                                    "id": "M1$abcd",
                                    "mainsnak": {"datavalue": {"value": {"id": "Q42"}}},
                                }
                            ]
                        }
                    }
                )
            return self._fake_response({"query": {"tokens": {"csrftoken": "csrf-tok"}}})

        monkeypatch.setattr(app_module.requests, "get", fake_get)
        monkeypatch.setattr(app_module.requests, "post", lambda *a, **kw: self._fake_response({"success": 1}))
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is False
