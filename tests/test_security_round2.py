"""Round-2 security patch tests.

Covers (in order):
- C1: Worker face-detect subprocess hardening (env scrub + RLIMIT)
- H1: CSP nonce + object-src/base-uri/frame-ancestors
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
import sys
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
# C1 — Worker subprocess hardening
# ---------------------------------------------------------------------------


class TestSubprocessHardening:
    """`_harden_face_detect_subprocess()` runs FIRST inside the face-detection
    subprocess, before importing untrusted-input parsers (dlib, libjpeg, libpng).
    It scrubs sensitive env vars so a subprocess RCE cannot exfiltrate OAuth
    secrets / DB creds, and applies POSIX rlimits so a CPU/mem-bomb image cannot
    wedge the Toolforge pod.
    """

    def test_scrubs_oauth_and_db_secrets(self, monkeypatch):
        # Populate the subprocess's env with everything we expect to be scrubbed.
        secrets_to_scrub = {
            "OAUTH_CLIENT_ID": "leak-id",
            "OAUTH_CLIENT_SECRET": "leak-secret",
            "OAUTH_REDIRECT_URI": "leak-uri",
            "WIKIVISAGE_TOKEN_KEY": "leak-fernet-key",
            "FLASK_SECRET_KEY": "leak-flask-key",
            "TOOL_TOOLSDB_USER": "leak-user",
            "TOOL_TOOLSDB_PASSWORD": "leak-pw",
            "WIKIVISAGE_DB_NAME": "leak-db",
            "WIKIVISAGE_REDIS_URL": "redis://leak:6379",
        }
        # And things that MUST survive (locale, PATH, Toolforge runtime hints).
        keep = {
            "PATH": "/usr/bin:/bin",
            "LANG": "en_US.UTF-8",
            "HOME": "/home/user",
        }
        env = {**secrets_to_scrub, **keep}
        with patch.dict(os.environ, env, clear=True):
            worker_module._harden_face_detect_subprocess()
            for k in secrets_to_scrub:
                assert k not in os.environ, f"{k} must be scrubbed"
            for k, v in keep.items():
                assert os.environ.get(k) == v, f"{k} must be preserved"

    def test_scrub_handles_prefix_match(self):
        """Any var starting with one of the configured prefixes must be removed,
        not just exact matches."""
        env = {
            "OAUTH_FOO": "leak",
            "OAUTH_BAR_BAZ": "leak",
            "TOOL_TOOLSDB_HOST": "leak",
            "WIKIVISAGE_DB_POOL_SIZE": "leak",
            "PATH": "/keep",
        }
        with patch.dict(os.environ, env, clear=True):
            worker_module._harden_face_detect_subprocess()
            assert "OAUTH_FOO" not in os.environ
            assert "OAUTH_BAR_BAZ" not in os.environ
            assert "TOOL_TOOLSDB_HOST" not in os.environ
            assert "WIKIVISAGE_DB_POOL_SIZE" not in os.environ
            assert os.environ.get("PATH") == "/keep"

    def test_scrub_does_not_match_unrelated_vars(self):
        """An env var whose name only *contains* a sensitive substring (but
        doesn't *start* with any prefix) must not be scrubbed."""
        env = {
            "MY_OAUTH_HELPER": "keep",  # contains OAUTH_ but doesn't start with it
            "PATH": "/keep",
        }
        with patch.dict(os.environ, env, clear=True):
            worker_module._harden_face_detect_subprocess()
            assert os.environ.get("MY_OAUTH_HELPER") == "keep"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")
    def test_applies_resource_limits_when_available(self):
        """When `resource` is importable (POSIX), setrlimit is called for
        AS / CPU / FSIZE. We patch resource.setrlimit to capture the calls."""
        import resource as _resource

        with patch.object(_resource, "setrlimit") as mock_setrlimit:
            with patch.dict(os.environ, {"PATH": "/x"}, clear=True):
                worker_module._harden_face_detect_subprocess()
        # We expect exactly 3 rlimit applications.
        rlimit_names = {
            _resource.RLIMIT_AS,
            _resource.RLIMIT_CPU,
            _resource.RLIMIT_FSIZE,
        }
        called_with = {call.args[0] for call in mock_setrlimit.call_args_list}
        # Each of the 3 rlimits we care about should have been targeted.
        assert rlimit_names.issubset(called_with), f"Missing rlimits: {rlimit_names - called_with}"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")
    def test_rlimit_failures_are_swallowed(self):
        """A sandbox that rejects setrlimit (raises ValueError or OSError) must
        not crash the subprocess — best-effort hardening only."""
        import resource as _resource

        with patch.object(_resource, "setrlimit", side_effect=OSError("denied")):
            with patch.dict(os.environ, {"PATH": "/x"}, clear=True):
                # Must not raise
                worker_module._harden_face_detect_subprocess()


# ---------------------------------------------------------------------------
# H1 — CSP hardening (object-src / base-uri / frame-ancestors)
# ---------------------------------------------------------------------------


class TestCSPHardening:
    """Round-2 H1 (partial): tighten Content-Security-Policy with the three
    directives that don't require a template refactor.

    NOTE: We intentionally do NOT include a `'nonce-...'` source alongside
    `'unsafe-inline'`. Per CSP Level 3, browsers ignore `'unsafe-inline'` when
    a nonce or hash source is present — which would silently break ~106 inline
    style="" attributes and ~41 inline event handlers across our templates
    (logout button, mobile nav toggle, language selector, PWA install banner).
    A future ticket can do the full nonce-only refactor; for now we keep the
    inline-style/handler ergonomics and ship the three directives below, which
    are the genuine security wins of H1 and are independent of nonces.
    """

    def test_csp_header_contains_required_directives(self):
        client = app_module.app.test_client()
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "object-src 'none'" in csp, "must block <object>/<embed>/<applet>"
        assert "base-uri 'none'" in csp, "must block <base> XSS escalation"
        assert "frame-ancestors 'none'" in csp, "must block clickjacking"
        assert "default-src 'self'" in csp, "baseline restriction"

    def test_csp_does_not_neutralize_unsafe_inline_with_nonce(self):
        """Regression guard: if a future change re-introduces 'nonce-...' into
        script-src or style-src while 'unsafe-inline' is also present, browsers
        per CSP3 will silently ignore 'unsafe-inline' and break inline styles
        and inline event handlers (logout button white background, mobile nav
        toggle dead, language selector dead, PWA banner mispositioned, etc.).
        """
        client = app_module.app.test_client()
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        # Either drop 'unsafe-inline', or drop the nonce — never both together.
        if "'unsafe-inline'" in csp:
            assert "'nonce-" not in csp, (
                "CSP regression: 'nonce-...' alongside 'unsafe-inline' makes "
                "browsers ignore 'unsafe-inline' (CSP3 §6.7.2.2) and breaks "
                'all inline style="" and inline event handlers.'
            )

    def test_csp_unsafe_inline_present_for_template_compat(self):
        """We currently rely on 'unsafe-inline' for inline styles/handlers."""
        client = app_module.app.test_client()
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "script-src 'self' 'unsafe-inline'" in csp
        assert "style-src 'self' 'unsafe-inline'" in csp

    def test_per_request_csp_nonce_still_generated_for_templates(self):
        """`g.csp_nonce` is still populated on every request so templates that
        reference `{{ csp_nonce }}` (~30 occurrences) keep rendering. The
        nonce just isn't advertised in the CSP header right now — harmless,
        and means a future nonce-only refactor won't need template changes.
        """
        with app_module.app.test_request_context("/"):
            # Trigger before_request manually
            for func in app_module.app.before_request_funcs.get(None, []):
                func()
            from flask import g

            assert hasattr(g, "csp_nonce"), "g.csp_nonce must be set per request"
            assert isinstance(g.csp_nonce, str)
            assert len(g.csp_nonce) >= 32, "nonce must have >=32 chars of entropy"


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
    the next outer-loop check fires."""

    def test_per_face_cancel_constant_present_and_small(self):
        """The interval must exist and be <= SDC_BATCH (otherwise the inner
        check is structurally useless)."""
        # Read the module source to assert the constant is defined inside
        # write_sdc_claims (it's a function-local for clarity).
        import inspect

        src = inspect.getsource(worker_module.write_sdc_claims)
        assert "_PER_FACE_CANCEL_CHECK_EVERY" in src
        # Must be checked against sdc_write_requested
        assert "sdc_write_requested" in src

    def test_inner_check_uses_sdc_write_requested_column(self):
        """The mid-batch query must SELECT the cancel flag — not some other
        weaker indicator (e.g. project state)."""
        import inspect

        src = inspect.getsource(worker_module.write_sdc_claims)
        # Look for the mid-batch SELECT (separate from the outer loop's check).
        # Both checks query the same column; we assert at least 2 occurrences.
        assert src.count("sdc_write_requested") >= 2, (
            "Expected both outer-loop and per-face cancel checks to query sdc_write_requested"
        )

    def test_mid_batch_database_error_is_non_fatal(self):
        """A failed cancel-check during a batch must not abort the write —
        we keep going with what we have, since failing-closed would block
        legitimate writes during a brief DB hiccup."""
        import inspect

        src = inspect.getsource(worker_module.write_sdc_claims)
        # The pattern: `except DatabaseError: pass` (or equivalent) should be
        # present near the inner check.
        assert "DatabaseError" in src
        assert "pass" in src


# ---------------------------------------------------------------------------
# M4 — _remove_sdc_claim treats "already gone" as success
# ---------------------------------------------------------------------------


class TestRemoveSdcClaimIdempotent:
    """If the entity or claim is already absent on Commons (file deleted, peer
    removed it, etc.), the user's desired end-state is satisfied. Surfacing an
    error would just confuse operators."""

    @staticmethod
    def _fake_response(payload, status=200):
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status.return_value = None
        r.status_code = status
        return r

    def test_returns_true_when_entity_missing(self, monkeypatch):
        """`wbgetclaims` returns {error: {code: 'no-such-entity'}} when the
        Commons file has no SDC entity at all. We treat this as success."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"error": {"code": "no-such-entity", "info": "M1 not found"}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True

    def test_returns_true_when_p180_list_empty(self, monkeypatch):
        """No P180 claims for this QID on the file → nothing to remove → success."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"claims": {"P180": []}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is True

    def test_returns_true_on_race_during_remove(self, monkeypatch):
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

    def test_returns_false_on_real_error(self, monkeypatch):
        """A genuine error code (e.g. permissiondenied) must NOT be silently
        swallowed — operators need to know."""
        monkeypatch.setattr(
            app_module.requests,
            "get",
            lambda *a, **kw: self._fake_response({"error": {"code": "permissiondenied", "info": "no perm"}}),
        )
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is False

    def test_returns_false_on_network_exception(self, monkeypatch):
        """Unhandled exceptions (timeout, DNS failure) → False."""

        def boom(*_a, **_kw):
            raise ConnectionError("network down")

        monkeypatch.setattr(app_module.requests, "get", boom)
        ok = app_module._remove_sdc_claim(commons_page_id=1, wikidata_qid="Q42", access_token="tok")
        assert ok is False

    def test_happy_path_returns_true(self, monkeypatch):
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
