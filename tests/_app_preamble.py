"""Shared setup for the test_app_*.py modules: environment defaults and the app import.

The app is imported once, with database initialisation patched out and the rate limiter
disabled (a full run otherwise exhausts the global 200/hour quota part-way through and turns
unrelated assertions into 429s; tests that need limiting turn it back on themselves).

These tests used to live in one 13,000-line module that did this at import time. Importing this
module first (every test_app_*.py does) keeps the order of events the same.
"""

import os
from unittest.mock import patch

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
    # The limiter is initialised with its config, so tests must disable the
    # instance (not the config). Tests that need limiting turn it back on themselves.
    app_module.limiter.enabled = False
    _snap_thumb_width = app_module._snap_thumb_width
    _is_safe_url = app_module._is_safe_url
    commons_thumb_url = app_module.commons_thumb_url
    _validate_csrf = app_module._validate_csrf
    _THUMB_STEPS = app_module._THUMB_STEPS

try:
    from conftest import _make_embedding, require_sample_env
except ModuleNotFoundError:
    from tests.conftest import _make_embedding, require_sample_env

flask_app.config["TESTING"] = True
flask_app.config["RATELIMIT_ENABLED"] = False

__all__ = [
    "_THUMB_STEPS",
    "_is_safe_url",
    "_make_embedding",
    "_snap_thumb_width",
    "_validate_csrf",
    "app_module",
    "commons_thumb_url",
    "flask_app",
    "require_sample_env",
]
