"""
WikiVisage — Flask web application.

Active-learning facial recognition tool for Wikimedia Commons.
Provides OAuth 2.0 authentication, project management, and an active
learning interface for classifying detected faces.
"""

import tomllib
from pathlib import Path

with Path(__file__).parent.joinpath("pyproject.toml").open("rb") as _f:
    APP_VERSION: str = tomllib.load(_f)["project"]["version"]

import hashlib
import io
import json
import logging
import math
import os
import random
import re
import secrets
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from dotenv import load_dotenv

load_dotenv()

import requests
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_babel import Babel
from flask_babel import gettext as _
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from requests_oauthlib import OAuth2Session

from config import WAKE_FILE_PATH
from database import (
    DatabaseError,
    execute_query,
    execute_transaction,
    init_db,
)
from token_crypto import TokenDecryptionError, decrypt_token, encrypt_token

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Trust reverse-proxy headers (Toolforge nginx → gunicorn).
# x_for=1, x_proto=1, x_host=1 so Flask sees the real client IP,
# HTTPS scheme, and correct Host — required for secure cookies and
# OAuth redirect URLs.
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# OAuth 2.0 configuration (Wikimedia Meta)
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_AUTHORIZE_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/authorize"
OAUTH_TOKEN_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/access_token"
OAUTH_PROFILE_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/resource/profile"
OAUTH_REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", "")

# Shared constants
MAX_BBOX_PX = 10000  # Maximum bounding box coordinate value
MIN_BBOX_AREA = 100  # Minimum bounding box area in pixels (10×10)
MAX_DISMISS_FACE_IDS = 100  # Maximum number of face IDs accepted in a single dismiss request
PROJECTS_PER_PAGE = 25
MAX_CATEGORY_TRAVERSAL = 50  # Max subcategories to visit in BFS
CATEGORY_API_TIMEOUT = 8  # Seconds for category info API calls
COMMONS_API_LIMIT = "500"  # MediaWiki API cmlimit

# Beta whitelist — fetched from GitHub every 5 minutes, falls back to local file
_WHITELIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelist.txt")
_WHITELIST_URL = "https://raw.githubusercontent.com/DiFronzo/WikiVisage/main/whitelist.txt"
_WHITELIST_LOCAL_ONLY = os.environ.get("WIKIVISAGE_WHITELIST_LOCAL") == "1"
_WHITELIST_CACHE_TTL = 300  # seconds
_whitelist_cache: set[str] = set()
_whitelist_cache_time: float = 0.0


def _parse_whitelist(text: str) -> set[str]:
    """Parse whitelist text into a set of usernames (NFKC-normalized)."""
    return {
        unicodedata.normalize("NFKC", s) for line in text.splitlines() if (s := line.strip()) and not s.startswith("#")
    }


def _load_whitelist() -> set[str]:
    """Return cached whitelist, refreshing from GitHub every 5 minutes.

    Fetch order: GitHub raw → local file → last-known-good cache.
    """
    global _whitelist_cache, _whitelist_cache_time

    now = time.monotonic()
    if _whitelist_cache and (now - _whitelist_cache_time) < _WHITELIST_CACHE_TTL:
        return _whitelist_cache

    # Try GitHub first (skip if local-only mode)
    if not _WHITELIST_LOCAL_ONLY:
        try:
            resp = requests.get(_WHITELIST_URL, timeout=5)
            resp.raise_for_status()
            fresh = _parse_whitelist(resp.text)
            if fresh:
                _whitelist_cache = fresh
                _whitelist_cache_time = now
                return _whitelist_cache
        except Exception:
            logger.debug("Failed to fetch whitelist from GitHub, trying local file")

    # Fall back to local file
    try:
        with open(_WHITELIST_PATH, encoding="utf-8") as f:
            fresh = _parse_whitelist(f.read())
            if fresh:
                _whitelist_cache = fresh
                _whitelist_cache_time = now
                return _whitelist_cache
    except FileNotFoundError:
        pass

    # Return last-known-good (may be empty on first boot if both fail)
    if not _whitelist_cache:
        logger.warning("Whitelist is empty — all authenticated access will be denied (fail-closed)")
    return _whitelist_cache


ALLOWED_USERS = _load_whitelist()

# Session configuration
app.config["SESSION_COOKIE_SECURE"] = not app.debug
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

# Rate limiter — uses Redis on Toolforge for shared state across gunicorn workers.
# Falls back to memory:// for local development where Redis may not be available.
_REDIS_URL = os.environ.get("WIKIVISAGE_REDIS_URL", "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379")
_limiter_storage_uri = _REDIS_URL

try:
    import redis as _redis_mod

    _r = _redis_mod.from_url(_REDIS_URL, socket_connect_timeout=2)
    _r.ping()
except Exception:
    logger.warning("Redis unavailable at %s — rate limiter using per-process memory storage", _REDIS_URL)
    _limiter_storage_uri = "memory://"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri=_limiter_storage_uri,
    key_prefix="wikivisage:",
)

# ---------------------------------------------------------------------------
# Internationalization (i18n)
# ---------------------------------------------------------------------------

LANGUAGES = {"en": "English", "nb": "Norsk bokmål", "es": "Español", "fr": "Français"}
RTL_LANGUAGES = {"ar", "he", "fa", "ur"}

app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_DEFAULT_TIMEZONE"] = "UTC"


def get_locale() -> str:
    """Select locale: cookie → Accept-Language header → default."""
    locale = request.cookies.get("locale")
    if locale and locale in LANGUAGES:
        return locale
    best = request.accept_languages.best_match(LANGUAGES.keys())
    return best or "en"


babel = Babel(app, locale_selector=get_locale)


@app.route("/set-language/<lang>")
def set_language(lang: str):
    """Set the user's preferred language via cookie (if consent given)."""
    if lang not in LANGUAGES:
        lang = "en"
    referrer = request.referrer or ""
    if not referrer or not _is_safe_url(referrer):
        referrer = url_for("index")
    resp = redirect(referrer)
    if not request.args.get("nocookie"):
        resp.set_cookie("locale", lang, max_age=60 * 60 * 24 * 365, httponly=True, samesite="Lax")
    else:
        # Delete any existing locale cookie when consent is withdrawn
        resp.delete_cookie("locale")
    return resp


@app.context_processor
def inject_i18n_helpers() -> dict[str, Any]:
    """Make i18n helpers available in all templates."""
    locale = get_locale()
    return {
        "current_locale": locale,
        "languages": LANGUAGES,
        "text_direction": "rtl" if locale in RTL_LANGUAGES else "ltr",
        "app_version": APP_VERSION,
    }


# Token expiry buffer (refresh 5 minutes before actual expiry)
TOKEN_REFRESH_BUFFER = 300  # seconds

# Wikimedia Commons file URL pattern (used for manual face detection)
FILE_PATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{file_title}?width=1024"
USER_AGENT = "WikiVisage/1.0 (https://github.com/DiFronzo/WikiVisage)"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# Maximum image download size (50 MB) — prevents OOM on abnormally large files
MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.before_request
def before_request() -> None:
    """Load current user from session into g.user before each request."""
    g.user = None
    user_id = session.get("user_id")
    if user_id:
        try:
            rows = execute_query(
                "SELECT id, wiki_user_id, wiki_username, access_token, "
                "refresh_token, token_expires_at FROM users WHERE id = %s",
                (user_id,),
            )
            if rows:
                g.user = rows[0]
                # Normalize token types (PyMySQL may return BLOB columns as bytes)
                for _tk in ("access_token", "refresh_token"):
                    if isinstance(g.user.get(_tk), bytes):
                        g.user[_tk] = g.user[_tk].decode("utf-8")
                    g.user[_tk] = decrypt_token(g.user[_tk])
                # Enforce whitelist on every request (not just login).
                # Fail-closed: empty whitelist = deny all (prevents bypass if both sources fail).
                allowed = _load_whitelist()
                if not allowed or unicodedata.normalize("NFKC", g.user["wiki_username"]) not in allowed:
                    logger.warning(f"Session revoked for user not on whitelist: {g.user['wiki_username']}")
                    session.clear()
                    g.user = None
        except TokenDecryptionError:
            logger.warning("Token decryption failed for user %s — clearing session to force re-auth", user_id)
            session.clear()
            g.user = None
        except DatabaseError:
            logger.exception("Failed to load user from session")
            session.clear()


@app.teardown_appcontext
def teardown_appcontext(exception: BaseException | None = None) -> None:
    """Clean up per-request resources."""
    pass  # Connection pool handles cleanup via context managers


def _is_safe_url(target: str) -> bool:
    """Check that a redirect URL is safe (same-host relative path only)."""
    if not target:
        return False
    # Canonicalize to absolute URL against our host, then verify
    # the result points back to the same host. This catches edge cases
    # like "///evil.com" which urlparse alone misclassifies.
    ref_url = request.host_url
    test_url = urljoin(ref_url, target)
    parsed = urlparse(test_url)
    return parsed.scheme in ("http", "https") and parsed.netloc == urlparse(ref_url).netloc


_ALLOWED_DOWNLOAD_HOSTS = frozenset({"commons.wikimedia.org", "upload.wikimedia.org"})


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Raises ValueError if the URL points to an untrusted host or if the
    response exceeds max_bytes.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or parsed_url.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Blocked download from untrusted host: {parsed_url.hostname}")
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        stream=True,
    )
    resp.raise_for_status()

    # Check Content-Length header first (fast reject)
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        resp.close()
        raise ValueError(f"Image too large: {int(content_length)} bytes (limit {max_bytes})")

    # Stream with enforced cap
    chunks: list[bytes] = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=65536):
        downloaded += len(chunk)
        if downloaded > max_bytes:
            resp.close()
            raise ValueError(f"Image download exceeded {max_bytes} bytes limit")
        chunks.append(chunk)

    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def login_required(f):
    """Decorator to require authentication."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash(_("Please log in to continue."), "warning")
            return redirect(url_for("login", next=request.full_path))
        return f(*args, **kwargs)

    return decorated_function


def _make_oauth_session(state: str | None = None) -> OAuth2Session:
    """Create an OAuth2Session with the configured client."""
    sess = OAuth2Session(
        client_id=OAUTH_CLIENT_ID,
        redirect_uri=OAUTH_REDIRECT_URI,
        state=state,
    )
    sess.headers["User-Agent"] = "WikiVisage/1.0 (https://github.com/DiFronzo/WikiVisage)"
    return sess


def _refresh_access_token(user: dict[str, Any]) -> dict[str, Any] | None:
    """
    Refresh the user's access token if it is expired or about to expire.

    Uses optimistic concurrency: after refreshing, the DB UPDATE compares
    the old token_expires_at value.  If another process already refreshed
    (changing token_expires_at), the UPDATE affects 0 rows and we re-read
    the freshly-refreshed credentials from the DB instead.

    Returns updated user dict or None if refresh failed.
    """
    expires_at = user["token_expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    if (expires_at - now).total_seconds() > TOKEN_REFRESH_BUFFER:
        return user  # Still valid

    old_expires_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Refreshing access token for user {user['wiki_username']}")

    try:
        oauth = OAuth2Session(client_id=OAUTH_CLIENT_ID)
        new_token = oauth.refresh_token(
            OAUTH_TOKEN_URL,
            refresh_token=user["refresh_token"],
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET,
        )

        new_expires_at = datetime.now(UTC) + timedelta(seconds=new_token.get("expires_in", 14400))

        # Optimistic concurrency: only update if no other process refreshed first
        rowcount = execute_query(
            "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s "
            "WHERE id = %s AND token_expires_at = %s",
            (
                encrypt_token(new_token["access_token"]),
                encrypt_token(new_token.get("refresh_token", user["refresh_token"])),
                new_expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                user["id"],
                old_expires_str,
            ),
            fetch=False,
        )

        if rowcount == 0:
            # Another process already refreshed — re-read from DB
            logger.info(f"Token already refreshed by another process for user {user['wiki_username']}")
            fresh = execute_query(
                "SELECT access_token, refresh_token, token_expires_at FROM users WHERE id = %s",
                (user["id"],),
                fetch=True,
            )
            if fresh:
                row = fresh[0]
                at = row["access_token"]
                if isinstance(at, bytes):
                    at = at.decode("utf-8")
                rt = row["refresh_token"]
                if isinstance(rt, bytes):
                    rt = rt.decode("utf-8")
                user["access_token"] = decrypt_token(at)
                user["refresh_token"] = decrypt_token(rt)
                user["token_expires_at"] = row["token_expires_at"]
                return user
            return None

        user["access_token"] = new_token["access_token"]
        user["refresh_token"] = new_token.get("refresh_token", user["refresh_token"])
        user["token_expires_at"] = new_expires_at
        return user

    except Exception:
        logger.exception("Failed to refresh access token")
        return None


def _get_valid_token() -> str | None:
    """Get a valid access token for the current user, refreshing if needed."""
    if g.user is None:
        return None

    refreshed = _refresh_access_token(g.user)
    if refreshed is None:
        session.clear()
        return None

    g.user = refreshed
    return refreshed["access_token"]


def _csrf_token() -> str:
    """Generate or return the current CSRF token for forms."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def _validate_csrf() -> bool:
    """Validate CSRF token from form submission or X-CSRFToken header."""
    token = request.form.get("csrf_token", "") or request.headers.get("X-CSRFToken", "")
    expected = session.get("csrf_token", "")
    if not expected or not token:
        return False
    return secrets.compare_digest(token, expected)


@app.context_processor
def inject_csrf_token() -> dict[str, Any]:
    """Make csrf_token() available in all templates."""
    return {"csrf_token": _csrf_token}


@app.context_processor
def inject_worker_status() -> dict[str, Any]:
    """Check worker heartbeat and inject worker_down flag into all templates."""
    try:
        rows = execute_query(
            "SELECT last_seen < NOW() - INTERVAL 5 MINUTE AS is_stale FROM worker_heartbeat WHERE id = 1",
        )
        if rows and isinstance(rows, list):
            worker_down = bool(rows[0]["is_stale"])
        else:
            # Row missing = worker never wrote a heartbeat since deploy.
            # Check if the table exists at all (fresh install → not down).
            table_check = execute_query(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = DATABASE() "
                "AND table_name = 'worker_heartbeat'",
            )
            worker_down = bool(table_check and isinstance(table_check, list))
    except Exception:
        app.logger.exception("Worker heartbeat check failed")
        worker_down = False
    return {"worker_down": worker_down}


@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' https://*.wikimedia.org data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response


def _is_human_entity(qid: str) -> bool:
    """
    Check whether a Wikidata entity has instance of (P31) set to human (Q5).

    Returns True if Q5 is among the P31 values, False otherwise (including
    on API errors — fail-open would allow non-human entities, so we fail-closed).
    """
    try:
        resp = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetclaims",
                "entity": qid,
                "property": "P31",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        claims = data.get("claims", {}).get("P31", [])
        for claim in claims:
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if value == "Q5":
                return True
        return False
    except Exception:
        logger.debug(f"Failed to check P31 for {qid}", exc_info=True)
        return False


def _commons_category_exists(category: str) -> bool:
    """Check whether a category exists on Wikimedia Commons.

    Returns True if the category page exists, False otherwise (including
    on API errors — fail-closed to prevent projects with invalid categories).
    """
    try:
        resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "query",
                "titles": f"Category:{category}",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        # If the only key is "-1", the page does not exist
        return "-1" not in pages
    except Exception:
        logger.debug(f"Failed to check Commons category: {category}", exc_info=True)
        return False


def _commons_category_has_files(category: str) -> bool:
    """Check whether a Commons category contains any files (directly or in subcategories).

    Uses the ``categoryinfo`` property — a single lightweight API call.
    Returns False if the category has 0 files and 0 subcategories, or on
    API errors (fail-closed).
    """
    try:
        resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "query",
                "titles": f"Category:{category}",
                "prop": "categoryinfo",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        if "-1" in pages:
            return False
        page = next(iter(pages.values()))
        info = page.get("categoryinfo", {})
        return info.get("files", 0) > 0 or info.get("subcats", 0) > 0
    except Exception:
        logger.debug(f"Failed to check Commons category files: {category}", exc_info=True)
        return False


def _check_p180_exists(commons_page_id: int, qid: str) -> bool:
    """Check whether a P180 (depicts) claim for *qid* already exists on a Commons media item.

    Uses the ``wbgetclaims`` API (read-only, no OAuth required).
    Returns ``True`` if the claim exists, ``False`` otherwise (including on API errors).
    """
    try:
        resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "wbgetclaims",
                "entity": f"M{commons_page_id}",
                "property": "P180",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        claims = data.get("claims", {}).get("P180", [])
        for claim in claims:
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if value == qid:
                return True
        return False
    except Exception:
        logger.debug(f"Failed to check P180 for M{commons_page_id}/{qid}", exc_info=True)
        return False


def _maybe_mark_sdc_written(face_id: int, image_id: int, project_id: int) -> bool:
    """Check if a P180 (depicts) claim already exists on Commons for the image/QID pair.

    If a matching claim is found, marks the face as ``sdc_written=1`` and the image
    as ``bootstrapped=1`` to avoid showing the face as "SDC Pending".

    Returns ``True`` if the face was marked as written, ``False`` otherwise.
    Non-critical: swallows all exceptions with a debug-level log.
    """
    try:
        meta = execute_query(
            "SELECT i.commons_page_id, p.wikidata_qid "
            "FROM images i JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = %s AND p.id = %s",
            (image_id, project_id),
        )
        if meta and meta[0]["commons_page_id"]:
            cpid = meta[0]["commons_page_id"]
            qid = meta[0]["wikidata_qid"]
            if _check_p180_exists(cpid, qid):
                execute_query(
                    "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                    (face_id,),
                    fetch=False,
                )
                execute_query(
                    "UPDATE images SET bootstrapped = 1 WHERE id = %s AND bootstrapped = 0",
                    (image_id,),
                    fetch=False,
                )
                return True
    except Exception:
        logger.debug("P180 existence check failed (non-critical)", exc_info=True)
    return False


def _fetch_p18_thumb_url(qid: str, width: int = 250) -> str | None:
    """
    Fetch P18 (image) property from Wikidata and return a Commons thumbnail URL.

    Args:
        qid: Wikidata entity ID (e.g., "Q42").
        width: Desired thumbnail width in pixels.

    Returns:
        Thumbnail URL string, or None if no P18 exists or API call fails.
    """
    try:
        resp = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetclaims",
                "entity": qid,
                "property": "P18",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        claims = data.get("claims", {}).get("P18", [])
        if not claims:
            return None

        filename = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")
        if not filename:
            return None

        return commons_thumb_url(filename, width)
    except Exception:
        logger.debug(f"Failed to fetch P18 for {qid}", exc_info=True)
        return None


def _fetch_wikidata_label(qid: str) -> str | None:
    """
    Fetch the label for a Wikidata entity.

    Tries the user's current locale first, falls back to English.

    Args:
        qid: Wikidata entity ID (e.g., "Q42").

    Returns:
        Entity label string, or None if unavailable or API call fails.
    """
    try:
        locale = get_locale()
        lang = str(locale) if locale else "en"
        languages = f"{lang}|en" if lang != "en" else "en"

        resp = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "labels",
                "languages": languages,
                "languagefallback": "1",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        entity = data.get("entities", {}).get(qid, {})
        labels = entity.get("labels", {})

        # Prefer user's locale, fall back to English
        label_obj = labels.get(lang) or labels.get("en")
        if label_obj:
            return label_obj.get("value")
        return None
    except Exception:
        logger.debug(f"Failed to fetch label for {qid}", exc_info=True)
        return None


# Wikimedia Commons enforces standard thumbnail step sizes ($wgThumbnailSteps).
# Requests for non-standard widths return 429. Snap to the nearest allowed step.
# https://www.mediawiki.org/wiki/Common_thumbnail_sizes
_THUMB_STEPS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)

# Extensions that need special thumbnail URL patterns on Commons
_VIDEO_EXTENSIONS = {".webm", ".ogv", ".ogg"}
_CONVERT_TO_JPG_EXTENSIONS = {".tif", ".tiff"}
_CONVERT_TO_PNG_EXTENSIONS = {".svg"}


def _snap_thumb_width(width: int) -> int:
    """Snap a requested width to the nearest Commons thumbnail step (≥ width)."""
    for step in _THUMB_STEPS:
        if step >= width:
            return step
    return _THUMB_STEPS[-1]


_MAX_INVITE_CODE_RETRIES = 5


def _generate_invite_code() -> str:
    return secrets.token_urlsafe(6)[:8]


def _is_mysql_1062(exc: Exception) -> bool:
    """Return True if *exc* (or its __cause__) is a MySQL 1062 (ER_DUP_ENTRY) error."""
    for candidate in (getattr(exc, "__cause__", None), exc):
        if candidate is None:
            continue
        args = getattr(candidate, "args", None)
        if args:
            try:
                if int(args[0]) == 1062:
                    return True
            except (ValueError, TypeError, IndexError):
                pass
    # Last resort: require both 1062 and "Duplicate entry" to avoid false positives
    # from error messages that incidentally contain the digits 1062.
    exc_str = str(exc)
    return bool(re.search(r"\b1062\b", exc_str)) and "Duplicate entry" in exc_str


def _is_invite_code_collision(exc: Exception) -> bool:
    """Return True if *exc* is a MySQL 1062 error on the invite_code UNIQUE index."""
    if not _is_mysql_1062(exc):
        return False
    # Also verify the collision is specifically on the invite_code index.
    for candidate in (getattr(exc, "__cause__", None), exc):
        if candidate is None:
            continue
        args = getattr(candidate, "args", None)
        if args and len(args) > 1 and "invite_code" in str(args[1]):
            return True
    return "invite_code" in str(exc)


def commons_thumb_url(file_title: str, width: int = 330) -> str:
    """Build a Wikimedia Commons thumbnail URL for any file type.

    The width is snapped to the nearest standard thumbnail step size
    enforced by Wikimedia Commons (see ``_THUMB_STEPS``).

    Handles special cases:
    - Video (.webm, .ogv): ``{width}px--{filename}.jpg``
    - TIFF (.tif, .tiff): ``{width}px-{filename}.jpg``
    - SVG: ``{width}px-{filename}.png``
    - Everything else: ``{width}px-{filename}``
    """
    width = _snap_thumb_width(width)
    # Strip "File:" prefix if present, normalise spaces to underscores
    clean = file_title.replace("File:", "").replace(" ", "_")
    md5 = hashlib.md5(clean.encode("utf-8")).hexdigest()
    ext = os.path.splitext(clean)[1].lower()

    # URL-encode the filename for safe use in path segments / redirect headers
    encoded = quote(clean, safe="")

    base = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5[0]}/{md5[0:2]}/{encoded}"

    if ext in _VIDEO_EXTENSIONS:
        return f"{base}/{width}px--{encoded}.jpg"
    elif ext in _CONVERT_TO_JPG_EXTENSIONS:
        return f"{base}/{width}px-{encoded}.jpg"
    elif ext in _CONVERT_TO_PNG_EXTENSIONS:
        return f"{base}/{width}px-{encoded}.png"
    else:
        return f"{base}/{width}px-{encoded}"


# Make helper available in all Jinja2 templates
app.jinja_env.globals["commons_thumb_url"] = commons_thumb_url


# ---------------------------------------------------------------------------
# OAuth 2.0 routes
# ---------------------------------------------------------------------------


@app.route("/login")
@limiter.limit("10 per minute")
def login():
    """Initiate OAuth 2.0 authorization flow."""
    if g.user:
        return redirect(url_for("dashboard"))

    oauth = _make_oauth_session()
    authorization_url, state = oauth.authorization_url(OAUTH_AUTHORIZE_URL)
    session["oauth_state"] = state
    session["login_next"] = request.args.get("next", "")
    return redirect(authorization_url)


@app.route("/auth/callback")
@limiter.limit("10 per minute")
def oauth_callback():
    """Handle OAuth 2.0 callback and create/update user record."""
    stored_state = session.pop("oauth_state", None)
    if not stored_state:
        flash(_("Invalid OAuth state. Please try again."), "error")
        return redirect(url_for("index"))

    oauth = _make_oauth_session(state=stored_state)

    try:
        token = oauth.fetch_token(
            OAUTH_TOKEN_URL,
            client_secret=OAUTH_CLIENT_SECRET,
            authorization_response=request.url,
        )
    except Exception:
        logger.exception("Failed to fetch OAuth token")
        flash(_("Authentication failed. Please try again."), "error")
        return redirect(url_for("index"))

    # Fetch user profile
    try:
        resp = requests.get(
            OAUTH_PROFILE_URL,
            headers={"Authorization": f"Bearer {token['access_token']}"},
            timeout=10,
        )
        resp.raise_for_status()
        profile = resp.json()
    except Exception:
        logger.exception("Failed to fetch user profile")
        flash(_("Could not retrieve your profile. Please try again."), "error")
        return redirect(url_for("index"))

    wiki_user_id = profile.get("sub")
    wiki_username = profile.get("username", "")

    if not wiki_user_id:
        flash(_("Invalid profile data received."), "error")
        return redirect(url_for("index"))

    # Beta whitelist check — fail-closed: empty whitelist = deny all
    allowed = _load_whitelist()
    if not allowed or wiki_username not in allowed:
        logger.warning(f"Login denied for user not on whitelist: {wiki_username}")
        flash(_("Access is currently restricted to approved testers."), "warning")
        return redirect(url_for("index"))

    expires_at = datetime.now(UTC) + timedelta(seconds=token.get("expires_in", 14400))
    expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

    # Upsert user
    try:
        existing = execute_query("SELECT id FROM users WHERE wiki_user_id = %s", (wiki_user_id,))

        if existing:
            execute_query(
                "UPDATE users SET wiki_username = %s, access_token = %s, "
                "refresh_token = %s, token_expires_at = %s WHERE wiki_user_id = %s",
                (
                    wiki_username,
                    encrypt_token(token["access_token"]),
                    encrypt_token(token.get("refresh_token", "")),
                    expires_at_str,
                    wiki_user_id,
                ),
                fetch=False,
            )
            user_id = existing[0]["id"]
        else:
            execute_query(
                "INSERT INTO users (wiki_user_id, wiki_username, access_token, "
                "refresh_token, token_expires_at) VALUES (%s, %s, %s, %s, %s)",
                (
                    wiki_user_id,
                    wiki_username,
                    encrypt_token(token["access_token"]),
                    encrypt_token(token.get("refresh_token", "")),
                    expires_at_str,
                ),
                fetch=False,
            )
            rows = execute_query("SELECT id FROM users WHERE wiki_user_id = %s", (wiki_user_id,))
            user_id = rows[0]["id"]

    except DatabaseError:
        logger.exception("Failed to upsert user")
        flash(_("Database error. Please try again."), "error")
        return redirect(url_for("index"))

    # Prevent session fixation: clear old session data before establishing
    # the authenticated session. Preserve CSRF token for continuity.
    csrf = session.get("csrf_token")
    login_next = session.pop("login_next", "")
    session.clear()
    if csrf:
        session["csrf_token"] = csrf
    session.permanent = True
    session["user_id"] = user_id

    next_url = login_next
    if not _is_safe_url(next_url):
        next_url = url_for("dashboard")
    return redirect(next_url)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    """Clear session and log out."""
    if not _validate_csrf():
        abort(403)
    session.clear()
    flash(_("You have been logged out."), "info")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# DB helpers — deduplicated query patterns
# ---------------------------------------------------------------------------


def get_project_for_user(project_id: int, user_id: int) -> dict | None:
    """Fetch a non-deleted project owned by *user_id*, or return ``None``."""
    rows = execute_query(
        "SELECT * FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
        (project_id, user_id),
    )
    return rows[0] if rows else None


def get_project_for_actor(project_id: int, user_id: int, *, require_owner: bool = False) -> dict | None:
    """Fetch a non-deleted project accessible to *user_id*.

    When *require_owner* is True, only the project owner can access it.
    Otherwise, both the owner and any member in ``project_members`` qualify.
    """
    if require_owner:
        return get_project_for_user(project_id, user_id)
    rows = execute_query(
        "SELECT p.* FROM projects p "
        "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
        "WHERE p.id = %s AND p.status != 'deleted' AND (p.user_id = %s OR pm.user_id IS NOT NULL)",
        (user_id, project_id, user_id),
    )
    return rows[0] if rows else None


def verify_image_access(image_id: int, project_id: int, user_id: int) -> dict | None:
    """Confirm *image_id* belongs to a project accessible to *user_id* (owner or member).

    Returns the image row (``id``, ``file_title``, ``status``) or ``None``.
    """
    rows = execute_query(
        "SELECT i.id, i.file_title, i.status FROM images i "
        "JOIN projects p ON i.project_id = p.id "
        "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
        "WHERE i.id = %s AND p.id = %s AND p.status != 'deleted' "
        "AND (p.user_id = %s OR pm.user_id IS NOT NULL)",
        (user_id, image_id, project_id, user_id),
    )
    return rows[0] if rows else None


def has_sibling_match(image_id: int, exclude_face_id: int) -> bool:
    """Return ``True`` if another target-match face exists on *image_id*
    (excluding *exclude_face_id* and superseded faces)."""
    rows = execute_query(
        "SELECT EXISTS("
        "  SELECT 1 FROM faces f2 "
        "  WHERE f2.image_id = %s AND f2.is_target = 1 "
        "  AND f2.superseded_by IS NULL AND f2.id != %s"
        ") AS has_sibling",
        (image_id, exclude_face_id),
    )
    return bool(rows[0]["has_sibling"]) if rows else False


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Landing page."""
    if g.user:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard showing all projects."""
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (ValueError, TypeError):
        page = 1

    offset = (page - 1) * PROJECTS_PER_PAGE

    try:
        count_row = execute_query(
            "SELECT COUNT(DISTINCT p.id) AS cnt FROM projects p "
            "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
            "WHERE (p.user_id = %s OR pm.user_id IS NOT NULL) AND p.status != 'deleted'",
            (g.user["id"], g.user["id"]),
        )
        total = count_row[0]["cnt"] if count_row else 0
    except DatabaseError:
        logger.exception("Failed to count projects")
        total = 0

    total_pages = max(1, -(-total // PROJECTS_PER_PAGE))  # ceil division
    page = min(page, total_pages)
    offset = (page - 1) * PROJECTS_PER_PAGE

    try:
        projects = execute_query(
            "SELECT DISTINCT p.* FROM projects p "
            "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
            "WHERE (p.user_id = %s OR pm.user_id IS NOT NULL) AND p.status != 'deleted' "
            "ORDER BY p.updated_at DESC LIMIT %s OFFSET %s",
            (g.user["id"], g.user["id"], PROJECTS_PER_PAGE, offset),
        )
    except DatabaseError:
        logger.exception("Failed to load projects")
        projects = []
        flash(_("Failed to load projects."), "error")

    # Add member counts for these projects
    member_counts = {}
    if projects:
        try:
            project_ids = [p["id"] for p in projects]
            placeholders = ", ".join(["%s"] * len(project_ids))
            count_rows = execute_query(
                f"SELECT project_id, COUNT(*) AS cnt FROM project_members "
                f"WHERE project_id IN ({placeholders}) AND status = 'active' GROUP BY project_id",
                tuple(project_ids),
            )
            for row in count_rows:
                member_counts[row["project_id"]] = row["cnt"]
        except DatabaseError:
            pass  # Non-critical

    # Lazily populate P18 thumbnails for projects missing them (cap to avoid slow page loads)
    if isinstance(projects, list):
        fetched = 0
        for proj in projects:
            if not proj.get("p18_thumb_url") and proj.get("wikidata_qid"):
                if fetched >= 3:
                    break
                thumb = _fetch_p18_thumb_url(proj["wikidata_qid"])
                fetched += 1
                if thumb:
                    proj["p18_thumb_url"] = thumb
                    try:
                        execute_query(
                            "UPDATE projects SET p18_thumb_url = %s WHERE id = %s",
                            (thumb, proj["id"]),
                            fetch=False,
                        )
                    except DatabaseError:
                        pass  # Non-critical — will retry next page load

    return render_template(
        "dashboard.html",
        projects=projects,
        member_counts=member_counts,
        current_user_id=g.user["id"],
        page=page,
        total_pages=total_pages,
        total_projects=total,
    )


@app.route("/api/category-info")
@login_required
@limiter.limit("5 per minute")
def api_category_info():
    """Return total file count for a Commons category (including subcategories).

    Does a BFS traversal of subcategories, batch-fetching categoryinfo to sum
    file counts. Bounded to MAX_CATEGORY_TRAVERSAL categories and a wall-clock timeout to
    stay responsive.
    """
    category = request.args.get("category", "").strip()
    if not category:
        return jsonify({"error": "missing category"}), 400

    if len(category) > 200 or any(c in category for c in "|\n\r\x00"):
        return jsonify({"error": "invalid category name"}), 400

    deadline = time.monotonic() + CATEGORY_API_TIMEOUT

    root_title = f"Category:{category}"

    try:
        # 1. Check root category exists
        resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "query",
                "titles": root_title,
                "prop": "categoryinfo",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        if "-1" in pages:
            return jsonify({"error": "not_found"}), 404

        page = next(iter(pages.values()))
        root_info = page.get("categoryinfo", {})
        total_files = root_info.get("files", 0)
        total_subcats = root_info.get("subcats", 0)

        # 2. BFS subcategory traversal to sum file counts
        cat_queue = []
        visited = {root_title}
        approximate = False

        # Seed queue with subcategories of root
        if total_subcats > 0:
            sub_resp = requests.get(
                COMMONS_API_URL,
                params={
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": root_title,
                    "cmtype": "subcat",
                    "cmlimit": COMMONS_API_LIMIT,
                    "format": "json",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=3,
            )
            sub_resp.raise_for_status()
            sub_data = sub_resp.json()
            for m in sub_data.get("query", {}).get("categorymembers", []):
                if m["title"] not in visited:
                    cat_queue.append(m["title"])
            if "continue" in sub_data:
                approximate = True

        while cat_queue and len(visited) < MAX_CATEGORY_TRAVERSAL and time.monotonic() < deadline:
            # Batch up to 50 titles for categoryinfo
            batch = []
            while cat_queue and len(batch) < 50 and len(visited) + len(batch) < MAX_CATEGORY_TRAVERSAL:
                title = cat_queue.pop(0)
                if title not in visited:
                    batch.append(title)

            if not batch:
                break

            for title in batch:
                visited.add(title)

            info_resp = requests.get(
                COMMONS_API_URL,
                params={
                    "action": "query",
                    "titles": "|".join(batch),
                    "prop": "categoryinfo",
                    "format": "json",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=3,
            )
            info_resp.raise_for_status()
            info_pages = info_resp.json().get("query", {}).get("pages", {})

            subcats_to_fetch = []
            for p in info_pages.values():
                ci = p.get("categoryinfo", {})
                total_files += ci.get("files", 0)
                total_subcats += ci.get("subcats", 0)
                if ci.get("subcats", 0) > 0:
                    subcats_to_fetch.append(p["title"])

            for sub_title in subcats_to_fetch:
                if time.monotonic() >= deadline or len(visited) >= MAX_CATEGORY_TRAVERSAL:
                    break
                sub_resp = requests.get(
                    COMMONS_API_URL,
                    params={
                        "action": "query",
                        "list": "categorymembers",
                        "cmtitle": sub_title,
                        "cmtype": "subcat",
                        "cmlimit": COMMONS_API_LIMIT,
                        "format": "json",
                    },
                    headers={"User-Agent": USER_AGENT},
                    timeout=3,
                )
                sub_resp.raise_for_status()
                sub_data = sub_resp.json()
                for m in sub_data.get("query", {}).get("categorymembers", []):
                    if m["title"] not in visited:
                        cat_queue.append(m["title"])
                if "continue" in sub_data:
                    approximate = True

        approximate = (
            approximate or len(visited) >= MAX_CATEGORY_TRAVERSAL or (cat_queue and time.monotonic() >= deadline)
        )

        return jsonify(
            {
                "files": total_files,
                "subcats": total_subcats,
                "categories_visited": len(visited),
                "approximate": approximate,
            }
        )
    except Exception:
        logger.debug(f"Failed to fetch category info: {category}", exc_info=True)
        return jsonify({"error": "api_error"}), 502


@app.route("/project/new", methods=["GET", "POST"])
@login_required
@limiter.limit("20 per hour")
def project_new():
    """Create a new project."""
    if request.method == "GET":
        return render_template("project_new.html")

    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    wikidata_qid = request.form.get("wikidata_qid", "").strip().upper()
    commons_category = request.form.get("commons_category", "").strip()
    label = request.form.get("label", "").strip()
    distance_threshold = request.form.get("distance_threshold", "0.6")
    min_confirmed = request.form.get("min_confirmed", "5")

    # Validation
    errors = []
    if not wikidata_qid or not wikidata_qid.startswith("Q"):
        errors.append(_("Wikidata Q-ID must start with 'Q' (e.g., Q42)."))
    if not commons_category:
        errors.append(_("Commons category is required."))

    try:
        distance_threshold = float(distance_threshold)
        if not 0.1 <= distance_threshold <= 1.0:
            errors.append(_("Distance threshold must be between 0.1 and 1.0."))
    except ValueError:
        errors.append(_("Distance threshold must be a number."))
        distance_threshold = 0.6

    try:
        min_confirmed = int(min_confirmed)
        if min_confirmed < 1:
            errors.append(_("Minimum confirmed must be at least 1."))
    except ValueError:
        errors.append(_("Minimum confirmed must be a whole number."))
        min_confirmed = 5

    # Validate Q-ID refers to a human (P31 = Q5) on Wikidata
    if not errors and wikidata_qid:
        if not _is_human_entity(wikidata_qid):
            errors.append(
                _(
                    "The Wikidata entity %(qid)s is not an instance of human (Q5). Only human entities are supported.",
                    qid=wikidata_qid,
                )
            )

    # Validate Commons category exists and has files
    if not errors and commons_category:
        if not _commons_category_exists(commons_category):
            errors.append(
                _(
                    'The Commons category "%(category)s" does not exist.',
                    category=commons_category,
                )
            )
        elif not _commons_category_has_files(commons_category):
            errors.append(
                _(
                    'The Commons category "%(category)s" exists but contains no files.',
                    category=commons_category,
                )
            )

    if errors:
        for err in errors:
            flash(err, "error")
        return render_template(
            "project_new.html",
            wikidata_qid=wikidata_qid,
            commons_category=commons_category,
            label=label,
            distance_threshold=distance_threshold,
            min_confirmed=min_confirmed,
        )

    # Check for duplicate
    try:
        existing = execute_query(
            "SELECT id FROM projects WHERE user_id = %s AND wikidata_qid = %s AND commons_category = %s "
            "AND status != 'deleted'",
            (g.user["id"], wikidata_qid, commons_category),
        )
        if existing:
            flash(_("A project with this Q-ID and category already exists."), "error")
            return render_template(
                "project_new.html",
                wikidata_qid=wikidata_qid,
                commons_category=commons_category,
                label=label,
                distance_threshold=distance_threshold,
                min_confirmed=min_confirmed,
            )
    except DatabaseError:
        logger.exception("Failed to check for duplicate project")

    # Cross-user duplicate: inform user if another project exists for same QID + category
    action = request.form.get("action", "create")
    try:
        other_project = execute_query(
            "SELECT p.id, p.label, u.wiki_username FROM projects p "
            "JOIN users u ON p.user_id = u.id "
            "WHERE p.wikidata_qid = %s AND p.commons_category = %s "
            "AND p.status != 'deleted' AND p.user_id != %s LIMIT 1",
            (wikidata_qid, commons_category, g.user["id"]),
        )
        if other_project:
            op = other_project[0]
            existing_membership = execute_query(
                "SELECT status FROM project_members WHERE project_id = %s AND user_id = %s",
                (op["id"], g.user["id"]),
            )
            is_banned = existing_membership and existing_membership[0]["status"] == "banned"
            if existing_membership and not is_banned:
                flash(_("You are already a member of this project."), "info")
                return redirect(url_for("project_detail", project_id=op["id"]))
            if not is_banned and action != "create_anyway":
                flash(
                    _(
                        "%(username)s already has a project for this Q-ID and category. "
                        "You can ask them for an invite code to join, or create your own.",
                        username=op["wiki_username"],
                    ),
                    "info",
                )
                return render_template(
                    "project_new.html",
                    wikidata_qid=wikidata_qid,
                    commons_category=commons_category,
                    label=label,
                    distance_threshold=distance_threshold,
                    min_confirmed=min_confirmed,
                    existing_project=op,
                )
    except DatabaseError:
        logger.exception("Failed to check for joinable projects")

    # Create project
    try:
        p18_thumb_url = _fetch_p18_thumb_url(wikidata_qid)

        if not label:
            label = _fetch_wikidata_label(wikidata_qid) or ""

        invite_code = _generate_invite_code()
        for _attempt in range(_MAX_INVITE_CODE_RETRIES):
            try:
                execute_query(
                    "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
                    "distance_threshold, min_confirmed, p18_thumb_url, invite_code) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        g.user["id"],
                        wikidata_qid,
                        commons_category,
                        label,
                        distance_threshold,
                        min_confirmed,
                        p18_thumb_url,
                        invite_code,
                    ),
                    fetch=False,
                )
                break
            except DatabaseError as _exc:
                if _is_invite_code_collision(_exc) and _attempt < _MAX_INVITE_CODE_RETRIES - 1:
                    logger.debug("invite_code collision on attempt %d, retrying", _attempt + 1)
                    invite_code = _generate_invite_code()
                else:
                    raise
        flash(_("Project created successfully!"), "success")

        # Signal the worker to wake up and process the new project immediately
        try:
            with open(WAKE_FILE_PATH, "w") as f:
                f.write("")
        except OSError:
            pass  # Non-critical — worker will pick it up on next poll

        return redirect(url_for("dashboard"))
    except DatabaseError as exc:
        # MySQL error 1062 (ER_DUP_ENTRY) means a soft-deleted row with the same
        # user_id + wikidata_qid + commons_category still exists.  The background
        # worker will hard-delete it on the next poll cycle (≤60 s).
        if _is_invite_code_collision(exc):
            # The retry loop exhausted all _MAX_INVITE_CODE_RETRIES attempts to find
            # a unique invite code and re-raised the last collision — treat as a
            # generic creation failure (extremely unlikely in practice).
            logger.exception("Failed to create project: invite_code collision after all retries")
            flash(_("Failed to create project. Please try again."), "error")
        elif _is_mysql_1062(exc):
            logger.info("Project creation blocked by pending soft-deleted row: %s", exc)
            flash(
                _(
                    "A previously deleted project with this Q-ID and category is still "
                    "being cleaned up. Please try again in a minute."
                ),
                "error",
            )
        else:
            logger.exception("Failed to create project")
            flash(_("Failed to create project. Please try again."), "error")
        return render_template(
            "project_new.html",
            wikidata_qid=wikidata_qid,
            commons_category=commons_category,
            label=label,
            distance_threshold=distance_threshold,
            min_confirmed=min_confirmed,
        )


@app.route("/project/<int:project_id>")
@login_required
def project_detail(project_id: int):
    """Project detail page with progress and stats."""
    try:
        project = get_project_for_actor(project_id, g.user["id"])
    except DatabaseError:
        logger.exception("Failed to load project")
        abort(500)

    if not project:
        abort(404)

    # Lazily populate P18 thumbnail if missing
    if not project.get("p18_thumb_url") and project.get("wikidata_qid"):
        thumb = _fetch_p18_thumb_url(project["wikidata_qid"])
        if thumb:
            project["p18_thumb_url"] = thumb
            try:
                execute_query(
                    "UPDATE projects SET p18_thumb_url = %s WHERE id = %s",
                    (thumb, project["id"]),
                    fetch=False,
                )
            except DatabaseError:
                pass  # Non-critical

    # Get face stats (totals + classification method breakdown)
    try:
        stats = execute_query(
            "SELECT "
            "  COUNT(*) AS total_faces, "
            "  SUM(CASE WHEN f.is_target = 1 THEN 1 ELSE 0 END) AS confirmed_matches, "
            "  SUM(CASE WHEN f.is_target = 0 THEN 1 ELSE 0 END) AS confirmed_non_matches, "
            "  SUM(CASE WHEN f.is_target IS NULL THEN 1 ELSE 0 END) AS unclassified, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS human_confirmed, "
            "  SUM(CASE WHEN f.sdc_written = 1 THEN 1 ELSE 0 END) AS sdc_written, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 1 ELSE 0 END) AS sdc_pending, "
            "  SUM(CASE WHEN f.sdc_removal_pending = 1 "
            "    AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
            "    AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id) "
            "    THEN 1 ELSE 0 END) AS sdc_removal_pending_faces, "
            "  COUNT(DISTINCT CASE WHEN f.sdc_removal_pending = 1 "
            "    AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
            "    AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id) "
            "    THEN f.image_id END) AS sdc_removal_pending, "
            "  SUM(CASE WHEN f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS by_human, "
            "  SUM(CASE WHEN f.classified_by = 'model' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_model, "
            "  SUM(CASE WHEN f.classified_by = 'bootstrap' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_bootstrap "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.superseded_by IS NULL",
            (project_id,),
        )
        face_stats = stats[0] if stats else {}
    except DatabaseError:
        logger.exception("Failed to load face stats")
        face_stats = {}

    gallery_total = 0
    try:
        rows = execute_query(
            "SELECT COUNT(*) AS cnt "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.superseded_by IS NULL "
            "  AND (f.classified_by IN ('model', 'bootstrap') "
            "       OR (f.classified_by = 'human' AND f.classified_by_user_id IS NOT NULL)) "
            "  AND LOWER(i.file_title) NOT REGEXP '\\\\.(webm|ogv|ogg|mp3|wav|flac|opus|mid|oga)$'",
            (project_id,),
        )
        gallery_total = rows[0]["cnt"] if rows else 0
    except DatabaseError:
        logger.exception("Failed to count gallery faces")

    pending_images = 0
    if project.get("status") == "active":
        try:
            pending_row = execute_query(
                "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s AND status = 'pending'",
                (project_id,),
            )
            pending_images = pending_row[0]["cnt"] if pending_row else 0
        except DatabaseError:
            pass

    # Auto-complete when processing is done but no usable faces exist.
    # Mirrors the worker's check so the user gets immediate feedback.
    total_faces = face_stats.get("total_faces") or 0
    images_total = project.get("images_total") or 0
    if (
        project.get("status") == "active"
        and pending_images == 0
        and images_total > 0
        and not project.get("completion_reason")
    ):
        min_confirmed = project.get("min_confirmed") or 5
        reason = None
        if total_faces == 0:
            reason = "no_faces"
        elif total_faces <= 5 and total_faces < min_confirmed:
            reason = "insufficient_faces"

        if reason:
            # Reflect completion status only in-memory for this response.
            # The worker is responsible for persisting this transition.
            project["status"] = "completed"
            project["completion_reason"] = reason

    # Count faces eligible for model inference (mirrors worker's filter)
    inference_eligible = 0
    try:
        eligible_row = execute_query(
            "SELECT COUNT(*) AS cnt FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.is_target IS NULL "
            "  AND f.superseded_by IS NULL "
            "  AND f.classified_by_user_id IS NULL",
            (project_id,),
        )
        inference_eligible = eligible_row[0]["cnt"] if eligible_row else 0
    except DatabaseError:
        pass

    # Count project members (excluding owner who is implicit)
    member_count = 0
    try:
        mc_row = execute_query(
            "SELECT COUNT(*) AS cnt FROM project_members WHERE project_id = %s AND status = 'active'",
            (project_id,),
        )
        member_count = mc_row[0]["cnt"] if mc_row else 0
    except DatabaseError:
        pass  # Non-critical

    inference_triggered = request.args.get("inference_triggered") == "1"

    is_owner = project["user_id"] == g.user["id"]

    return render_template(
        "project_detail.html",
        project=project,
        stats=face_stats,
        gallery_total=gallery_total,
        member_count=member_count,
        pending_images=pending_images,
        inference_eligible=inference_eligible,
        inference_triggered=inference_triggered,
        is_owner=is_owner,
    )


@app.route("/project/<int:project_id>/classify")
@login_required
def classify(project_id: int):
    """Active learning classification interface."""
    try:
        project = get_project_for_actor(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    # Handle skipped images (stored in session, reset when project changes)
    skip_key = f"skipped_images_{project_id}"
    skip_key_review = f"skipped_images_review_{project_id}"
    is_skip_review = request.args.get("skip_reviewing") == "1"
    if request.args.get("skip_image_id"):
        try:
            skip_id = int(request.args["skip_image_id"])
            active_skip_key = skip_key_review if is_skip_review else skip_key
            skipped = session.get(active_skip_key, [])
            if skip_id not in skipped:
                skipped.append(skip_id)
            session[active_skip_key] = skipped
            session["last_classify"] = {
                "action": "skip",
                "project_id": project_id,
                "image_id": skip_id,
                "was_review": is_skip_review,
                "face_ids": [],
                "manual_face_ids": [],
            }
        except (ValueError, TypeError):
            pass

    skipped_ids = session.get(skip_key, [])

    # If a specific image_id is requested (e.g. after undo), force-show it
    forced_image_id = request.args.get("image_id", type=int)

    # Get next image that has unclassified faces (excluding skipped)
    # Priority: forced image_id > unclassified (is_target IS NULL) > model not-target review
    reviewing_model = False
    try:
        if forced_image_id:
            # Force-load a specific image (used after undo to return to the undone image)
            image_row = execute_query(
                "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, "
                "i.detection_width, i.detection_height "
                "FROM images i "
                "JOIN faces f ON f.image_id = i.id "
                "WHERE i.project_id = %s AND i.id = %s "
                "AND f.superseded_by IS NULL "
                "AND (f.is_target IS NULL OR (f.is_target = 0 AND f.classified_by = 'model')) "
                "LIMIT 1",
                (project_id, forced_image_id),
            )
            # Check if this image has model not-target faces (for review mode)
            if image_row:
                unclassified_check = execute_query(
                    "SELECT COUNT(*) AS cnt FROM faces "
                    "WHERE image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
                    (forced_image_id,),
                )
                if not unclassified_check or unclassified_check[0]["cnt"] == 0:
                    reviewing_model = True
        elif skipped_ids:
            placeholders = ",".join(["%s"] * len(skipped_ids))
            image_row = execute_query(
                "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, "
                "i.detection_width, i.detection_height "
                "FROM images i "
                "JOIN faces f ON f.image_id = i.id "
                f"WHERE i.project_id = %s AND f.is_target IS NULL AND f.superseded_by IS NULL AND i.id NOT IN ({placeholders}) "
                "LIMIT 200",
                (project_id, *skipped_ids),
            )
            if image_row:
                image_row = [random.choice(image_row)]
        else:
            image_row = execute_query(
                "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, "
                "i.detection_width, i.detection_height "
                "FROM images i "
                "JOIN faces f ON f.image_id = i.id "
                "WHERE i.project_id = %s AND f.is_target IS NULL AND f.superseded_by IS NULL "
                "LIMIT 200",
                (project_id,),
            )
            if image_row:
                image_row = [random.choice(image_row)]

        # Fallback: if no unclassified faces, try model not-target faces for review
        if not image_row:
            reviewing_model = True
            skipped_review_ids = session.get(skip_key_review, [])

            if skipped_review_ids:
                placeholders = ",".join(["%s"] * len(skipped_review_ids))
                image_row = execute_query(
                    "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, "
                    "i.detection_width, i.detection_height "
                    "FROM images i "
                    "JOIN faces f ON f.image_id = i.id "
                    f"WHERE i.project_id = %s AND f.is_target = 0 AND f.classified_by = 'model' "
                    f"AND f.classified_by_user_id IS NULL AND f.superseded_by IS NULL AND i.id NOT IN ({placeholders}) "
                    "LIMIT 200",
                    (project_id, *skipped_review_ids),
                )
                if image_row:
                    image_row = [random.choice(image_row)]
            else:
                image_row = execute_query(
                    "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, "
                    "i.detection_width, i.detection_height "
                    "FROM images i "
                    "JOIN faces f ON f.image_id = i.id "
                    "WHERE i.project_id = %s AND f.is_target = 0 AND f.classified_by = 'model' "
                    "AND f.classified_by_user_id IS NULL AND f.superseded_by IS NULL "
                    "LIMIT 200",
                    (project_id,),
                )
                if image_row:
                    image_row = [random.choice(image_row)]
    except DatabaseError:
        logger.exception("Failed to load image for classification")
        image_row = []

    image = image_row[0] if image_row else None
    faces_list = []

    if image:
        if reviewing_model:
            # Get model not-target faces for review
            try:
                faces_list = execute_query(
                    "SELECT f.id AS face_id, f.bbox_top, f.bbox_right, "
                    "f.bbox_bottom, f.bbox_left, f.confidence "
                    "FROM faces f "
                    "WHERE f.image_id = %s AND f.is_target = 0 AND f.classified_by = 'model' "
                    "AND f.superseded_by IS NULL "
                    "ORDER BY f.bbox_left ASC",
                    (image["image_id"],),
                )
            except DatabaseError:
                logger.exception("Failed to load faces for model review")
                faces_list = []
        else:
            # Get ALL unclassified faces for this image
            try:
                faces_list = execute_query(
                    "SELECT f.id AS face_id, f.bbox_top, f.bbox_right, "
                    "f.bbox_bottom, f.bbox_left, f.confidence "
                    "FROM faces f "
                    "WHERE f.image_id = %s AND f.is_target IS NULL AND f.superseded_by IS NULL "
                    "ORDER BY f.bbox_left ASC",
                    (image["image_id"],),
                )
            except DatabaseError:
                logger.exception("Failed to load faces for classification")
                faces_list = []

    # Count remaining images (not faces) — both unclassified and model review
    try:
        remaining = execute_query(
            "SELECT COUNT(DISTINCT i.id) AS cnt "
            "FROM images i "
            "JOIN faces f ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.is_target IS NULL AND f.superseded_by IS NULL",
            (project_id,),
        )
        remaining_count = remaining[0]["cnt"] if remaining else 0
    except DatabaseError:
        remaining_count = 0

    try:
        model_remaining = execute_query(
            "SELECT COUNT(DISTINCT i.id) AS cnt "
            "FROM images i "
            "JOIN faces f ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.is_target = 0 AND f.classified_by = 'model' "
            "AND f.classified_by_user_id IS NULL AND f.superseded_by IS NULL",
            (project_id,),
        )
        model_review_count = model_remaining[0]["cnt"] if model_remaining else 0
    except DatabaseError:
        model_review_count = 0

    return render_template(
        "classify.html",
        project=project,
        image=image,
        faces=faces_list,
        remaining=remaining_count,
        model_review_count=model_review_count,
        reviewing_model=reviewing_model,
        has_undo=bool(session.get("last_classify")),
        skipped_count=len(session.get(skip_key, [])),
        skipped_review_count=len(session.get(skip_key_review, [])),
    )


@app.route("/project/<int:project_id>/classify/clear-skips", methods=["POST"])
@login_required
def clear_skips(project_id: int):
    """Clear skipped images and redirect back to classify."""
    if not _validate_csrf():
        abort(403)
    skip_key = f"skipped_images_{project_id}"
    skip_key_review = f"skipped_images_review_{project_id}"
    session.pop(skip_key, None)
    session.pop(skip_key_review, None)
    return redirect(url_for("classify", project_id=project_id))


@app.route("/api/classify", methods=["POST"])
@login_required
@limiter.limit("120 per minute")
def api_classify():
    """API endpoint to classify faces for an image. Accepts the selected
    target face_id or 'none' to mark all faces as non-target."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    selected_face_id = request.form.get("selected_face_id")  # face_id or "none"
    project_id = request.form.get("project_id")
    image_id = request.form.get("image_id")
    is_review_mode = request.form.get("reviewing_model") == "1"

    if selected_face_id is None or not project_id or not image_id:
        return jsonify({"error": _("Missing required fields")}), 400

    try:
        project_id = int(project_id)
        image_id = int(image_id)
    except (ValueError, TypeError):
        return jsonify({"error": _("Invalid field values")}), 400

    # Verify ownership: image belongs to a project owned by this user
    try:
        img = verify_image_access(image_id, project_id, g.user["id"])
        if not img:
            return jsonify({"error": _("Image not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    # In review mode, we target model-classified not-target faces
    # In normal mode, we target unclassified faces
    if is_review_mode:
        face_filter_sql = (
            "is_target = 0 AND classified_by = 'model' AND classified_by_user_id IS NULL AND superseded_by IS NULL"
        )
    else:
        face_filter_sql = "is_target IS NULL AND superseded_by IS NULL"

    try:
        if selected_face_id == "none":

            def _classify_none(conn, cursor):
                cursor.execute(
                    f"SELECT id FROM faces WHERE image_id = %s AND {face_filter_sql}",
                    (image_id,),
                )
                rows = cursor.fetchall()
                ids = [r["id"] for r in rows] if rows else []

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE image_id = %s AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id),
                    )

                # Queue P180 removal for rejected faces on bootstrapped images
                if ids:
                    cursor.execute(
                        "SELECT i.bootstrapped, "
                        "EXISTS(SELECT 1 FROM faces f2 WHERE f2.image_id = %s "
                        "AND f2.is_target = 1 AND f2.superseded_by IS NULL "
                        "AND f2.id NOT IN ({placeholders})) AS has_sibling_match "
                        "FROM images i WHERE i.id = %s".format(placeholders=",".join(["%s"] * len(ids))),
                        (image_id, *ids, image_id),
                    )
                    img_row = cursor.fetchone()
                    if img_row and img_row["bootstrapped"] and not img_row["has_sibling_match"]:
                        id_placeholders = ",".join(["%s"] * len(ids))
                        cursor.execute(
                            f"UPDATE faces SET sdc_removal_pending = 1 "
                            f"WHERE id IN ({id_placeholders}) AND is_target = 0",
                            tuple(ids),
                        )

                return ids

            affected_ids = execute_transaction(_classify_none)

            session["last_classify"] = {
                "project_id": project_id,
                "image_id": image_id,
                "action": "none",
                "face_ids": affected_ids,
                "manual_face_ids": session.pop(f"manual_faces_{image_id}", []),
                "was_review": is_review_mode,
            }
        else:
            try:
                selected_face_id = int(selected_face_id)
            except (ValueError, TypeError):
                return jsonify({"error": _("Invalid face ID")}), 400

            def _classify_target(conn, cursor):
                cursor.execute(
                    f"SELECT id FROM faces WHERE image_id = %s AND {face_filter_sql}",
                    (image_id,),
                )
                rows = cursor.fetchall()
                ids = [r["id"] for r in rows] if rows else []

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET is_target = 1, classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE id = %s AND image_id = %s "
                        "AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], selected_face_id, image_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 1, classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE id = %s AND image_id = %s AND is_target IS NULL",
                        (g.user["id"], selected_face_id, image_id),
                    )
                target_updated = cursor.rowcount

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE image_id = %s AND id != %s "
                        "AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id, selected_face_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        "classified_by_user_id = %s "
                        "WHERE image_id = %s AND id != %s AND is_target IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id, selected_face_id),
                    )

                if target_updated > 0:
                    cursor.execute(
                        "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
                        (project_id,),
                    )

                # Do NOT queue P180 removal here — the selected face confirms
                # the person IS depicted, so the P180 claim must stay.

                return ids

            affected_ids = execute_transaction(_classify_target)

            if affected_ids:
                _maybe_mark_sdc_written(selected_face_id, image_id, project_id)

            session["last_classify"] = {
                "project_id": project_id,
                "image_id": image_id,
                "action": "target",
                "selected_face_id": selected_face_id,
                "face_ids": affected_ids,
                "manual_face_ids": session.pop(f"manual_faces_{image_id}", []),
                "was_review": is_review_mode,
            }
    except DatabaseError:
        logger.exception("Failed to classify faces")
        return jsonify({"error": _("Failed to save classification")}), 500

    return jsonify({"status": "ok"})


@app.route("/api/undo-classify", methods=["POST"])
@login_required
@limiter.limit("60 per minute")
def api_undo_classify():
    """Undo the last face classification, resetting affected faces to unclassified."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    last = session.get("last_classify")
    if not last:
        return jsonify({"error": _("Nothing to undo")}), 400

    project_id = last["project_id"]
    image_id = last["image_id"]

    # Handle undo of skip — no DB changes, just remove from skip list
    if last.get("action") == "skip":
        try:
            proj = get_project_for_actor(project_id, g.user["id"])
            if not proj:
                return jsonify({"error": _("Project not found")}), 404
        except DatabaseError:
            return jsonify({"error": _("Database error")}), 500

        was_review = last.get("was_review", False)
        active_skip_key = f"skipped_images_review_{project_id}" if was_review else f"skipped_images_{project_id}"
        skipped = session.get(active_skip_key, [])
        if image_id in skipped:
            skipped.remove(image_id)
            session[active_skip_key] = skipped
        session.pop("last_classify", None)
        g._skip_undo = True
        return jsonify({"status": "ok", "project_id": project_id, "image_id": image_id})

    face_ids = last.get("face_ids", [])
    manual_face_ids = last.get("manual_face_ids", [])

    if not face_ids and not manual_face_ids:
        session.pop("last_classify", None)
        return jsonify({"error": _("Nothing to undo")}), 400

    # Verify ownership
    try:
        img = verify_image_access(image_id, project_id, g.user["id"])
        if not img:
            return jsonify({"error": _("Image not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    try:
        was_review = last.get("was_review", False)
        decrement_counter = last.get("action") == "target" or (
            last.get("action") == "manual_face" and last.get("was_review")
        )

        def _undo(conn, cursor):
            if manual_face_ids:
                m_placeholders = ",".join(["%s"] * len(manual_face_ids))
                cursor.execute(
                    f"DELETE FROM faces WHERE id IN ({m_placeholders}) AND image_id = %s",
                    (*manual_face_ids, image_id),
                )

            original_ids = [fid for fid in face_ids if fid not in manual_face_ids]
            if original_ids:
                placeholders = ",".join(["%s"] * len(original_ids))
                if was_review:
                    cursor.execute(
                        f"UPDATE faces SET is_target = 0, classified_by = 'model', "
                        f"classified_by_user_id = NULL, sdc_removal_pending = 0 "
                        f"WHERE id IN ({placeholders}) AND image_id = %s",
                        (*original_ids, image_id),
                    )
                else:
                    cursor.execute(
                        f"UPDATE faces SET is_target = NULL, classified_by = NULL, "
                        f"classified_by_user_id = NULL, sdc_removal_pending = 0 "
                        f"WHERE id IN ({placeholders}) AND image_id = %s",
                        (*original_ids, image_id),
                    )

            if decrement_counter:
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = GREATEST(faces_confirmed - 1, 0) WHERE id = %s",
                    (project_id,),
                )

        execute_transaction(_undo)
        session.pop("last_classify", None)

    except DatabaseError:
        logger.exception("Failed to undo classification")
        return jsonify({"error": _("Failed to undo classification")}), 500

    return jsonify({"status": "ok", "project_id": project_id, "image_id": image_id})


@app.route("/api/manual-face", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_manual_face():
    """Accept a manually drawn bounding box, compute face encoding, and insert
    the face into the database so it can be classified."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    project_id_str = request.form.get("project_id")
    image_id_str = request.form.get("image_id")
    bbox_top_str = request.form.get("bbox_top")
    bbox_right_str = request.form.get("bbox_right")
    bbox_bottom_str = request.form.get("bbox_bottom")
    bbox_left_str = request.form.get("bbox_left")

    if not all(
        [
            project_id_str,
            image_id_str,
            bbox_top_str,
            bbox_right_str,
            bbox_bottom_str,
            bbox_left_str,
        ]
    ):
        return jsonify({"error": _("Missing required fields")}), 400

    try:
        project_id = int(project_id_str)  # type: ignore[arg-type]
        image_id = int(image_id_str)  # type: ignore[arg-type]
        bbox_top = int(bbox_top_str)  # type: ignore[arg-type]
        bbox_right = int(bbox_right_str)  # type: ignore[arg-type]
        bbox_bottom = int(bbox_bottom_str)  # type: ignore[arg-type]
        bbox_left = int(bbox_left_str)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return jsonify({"error": _("Invalid field values")}), 400

    if bbox_top >= bbox_bottom or bbox_left >= bbox_right:
        return jsonify({"error": _("Invalid bounding box dimensions")}), 400

    if (
        bbox_top < 0
        or bbox_left < 0
        or bbox_bottom > MAX_BBOX_PX
        or bbox_right > MAX_BBOX_PX
        or (bbox_bottom - bbox_top) * (bbox_right - bbox_left) < MIN_BBOX_AREA
    ):
        return jsonify({"error": _("Bounding box out of allowed range")}), 400

    # In review mode, user is drawing a face the model missed — auto-classify
    is_review_mode = request.form.get("reviewing_model") == "1"

    # In normal classify mode, dismiss currently displayed faces as non-target
    dismiss_face_ids: list[int] = []
    if not is_review_mode:
        dismiss_raw = request.form.get("dismiss_face_ids", "")
        if dismiss_raw:
            try:
                parsed = json.loads(dismiss_raw)
                if isinstance(parsed, list):
                    if len(parsed) > MAX_DISMISS_FACE_IDS:
                        return jsonify({"error": _("Too many face IDs in dismiss list")}), 400
                    dismiss_face_ids = [int(fid) for fid in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    try:
        img = verify_image_access(image_id, project_id, g.user["id"])
        if not img:
            return jsonify({"error": _("Image not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    if img.get("status") != "processed":
        return jsonify({"error": _("Image has not finished processing yet")}), 409

    file_title = img["file_title"]
    clean_title = file_title[5:] if file_title.startswith("File:") else file_title
    url = FILE_PATH_URL.format(file_title=clean_title)

    try:
        image_bytes = _download_image(url)

        import face_recognition  # lazy: dlib is ~130 MB; avoid loading at startup

        image_data = face_recognition.load_image_file(io.BytesIO(image_bytes))

        face_location = [(bbox_top, bbox_right, bbox_bottom, bbox_left)]
        encodings = face_recognition.face_encodings(image_data, face_location)

        if not encodings:
            return jsonify(
                {
                    "error": _(
                        "Could not compute face encoding for the selected region. Try drawing a slightly larger box."
                    )
                }
            ), 422

        encoding_bytes = encodings[0].tobytes()

        def _insert_manual_face(conn, cursor):
            review_confirmed_ids = []
            if is_review_mode:
                cursor.execute(
                    "SELECT id FROM faces "
                    "WHERE image_id = %s AND is_target = 0 AND classified_by = 'model' "
                    "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                    (image_id,),
                )
                rows = cursor.fetchall()
                review_confirmed_ids = [r["id"] for r in rows] if rows else []

                cursor.execute(
                    "INSERT INTO faces "
                    "(image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
                    "is_target, classified_by, classified_by_user_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 1, 'human', %s)",
                    (
                        image_id,
                        encoding_bytes,
                        bbox_top,
                        bbox_right,
                        bbox_bottom,
                        bbox_left,
                        g.user["id"],
                    ),
                )
                new_face_id = cursor.lastrowid

                cursor.execute(
                    "UPDATE faces SET classified_by = 'human', "
                    "classified_by_user_id = %s "
                    "WHERE image_id = %s AND is_target = 0 AND classified_by = 'model' "
                    "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                    (g.user["id"], image_id),
                )
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
                    (project_id,),
                )
            else:
                cursor.execute(
                    "INSERT INTO faces "
                    "(image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
                    "is_target, classified_by, classified_by_user_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 1, 'human', %s)",
                    (
                        image_id,
                        encoding_bytes,
                        bbox_top,
                        bbox_right,
                        bbox_bottom,
                        bbox_left,
                        g.user["id"],
                    ),
                )
                new_face_id = cursor.lastrowid

                # Dismiss currently displayed faces as non-target so the image
                # doesn't reappear in the classify queue endlessly.
                if dismiss_face_ids:
                    placeholders = ",".join(["%s"] * len(dismiss_face_ids))
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        f"classified_by_user_id = %s WHERE id IN ({placeholders}) "
                        "AND image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
                        (g.user["id"], *dismiss_face_ids, image_id),
                    )

            return new_face_id, review_confirmed_ids

        new_face_id, review_confirmed_ids = execute_transaction(_insert_manual_face)

        if new_face_id:
            _maybe_mark_sdc_written(new_face_id, image_id, project_id)

            manual_key = f"manual_faces_{image_id}"
            manual_list = session.get(manual_key, [])
            manual_list.append(new_face_id)
            session[manual_key] = manual_list

            session["last_classify"] = {
                "project_id": project_id,
                "image_id": image_id,
                "action": "manual_face",
                "face_ids": review_confirmed_ids if is_review_mode else dismiss_face_ids,
                "manual_face_ids": [new_face_id],
                "was_review": is_review_mode,
            }

        return jsonify({"status": "ok"})

    except requests.RequestException as e:
        logger.error(f"Failed to download image for manual face: {e}")
        return jsonify({"error": _("Failed to download image from Commons")}), 502
    except DatabaseError:
        logger.exception("Failed to insert manual face")
        return jsonify({"error": _("Failed to save face")}), 500
    except Exception:
        logger.exception("Unexpected error in manual face detection")
        return jsonify({"error": _("Failed to process face region")}), 500


def _remove_sdc_claim(commons_page_id: int, wikidata_qid: str, access_token: str) -> bool:
    """Remove a P180 depicts claim for a specific QID from a Commons file.
    Returns True if claim was removed or didn't exist, False on error."""
    mid = f"M{commons_page_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
    }

    try:
        # 1. Find the specific claim GUID
        claim_resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "wbgetclaims",
                "entity": mid,
                "property": "P180",
                "format": "json",
            },
            headers=headers,
            timeout=30,
        )
        claim_resp.raise_for_status()
        claim_data = claim_resp.json()

        claims = claim_data.get("claims", {}).get("P180", [])
        target_guid = None
        for claim in claims:
            snak = claim.get("mainsnak", {})
            if snak.get("datavalue", {}).get("value", {}).get("id") == wikidata_qid:
                target_guid = claim.get("id")
                break

        if not target_guid:
            # Claim doesn't exist on Commons — nothing to remove
            return True

        # 2. Get CSRF token
        token_resp = requests.get(
            COMMONS_API_URL,
            params={
                "action": "query",
                "meta": "tokens",
                "type": "csrf",
                "format": "json",
            },
            headers=headers,
            timeout=30,
        )
        token_resp.raise_for_status()
        csrf_token = token_resp.json()["query"]["tokens"]["csrftoken"]

        # 3. Remove the claim
        remove_resp = requests.post(
            COMMONS_API_URL,
            data={
                "action": "wbremoveclaims",
                "claim": target_guid,
                "token": csrf_token,
                "summary": f"WikiVisage: Removing depicts (P180) claim for {wikidata_qid} (human review)",
                "format": "json",
                "bot": "1",
                "maxlag": "5",
            },
            headers=headers,
            timeout=30,
        )
        remove_resp.raise_for_status()
        result = remove_resp.json()

        if "error" in result:
            logger.error(f"SDC removal error for {mid}/{wikidata_qid}: {result['error']}")
            return False

        return True

    except Exception:
        logger.exception(f"Failed to remove SDC claim for {mid}/{wikidata_qid}")
        return False


@app.route("/api/reclassify", methods=["POST"])
@login_required
@limiter.limit("60 per minute")
def api_reclassify():
    """Reclassify a model/bootstrap-classified face as human-verified.
    If rejecting a bootstrap face, queues P180 removal for the background worker.
    If rejecting a non-bootstrap face with sdc_written=1, removes the P180 claim
    from Commons immediately."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    face_id_str = request.form.get("face_id")
    is_target_str = request.form.get("is_target")  # "1" or "0"

    if not face_id_str or is_target_str is None:
        return jsonify({"error": _("Missing required fields")}), 400

    try:
        face_id = int(face_id_str)
        is_target = int(is_target_str)
        if is_target not in (0, 1):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": _("Invalid field values")}), 400

    # Verify ownership: face → image → project → user (or member)
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target AS old_is_target, f.sdc_written, "
            "  f.classified_by, "
            "  i.commons_page_id, i.bootstrapped, "
            "  p.id AS project_id, p.wikidata_qid "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
            "WHERE f.id = %s AND (p.user_id = %s OR pm.user_id IS NOT NULL) "
            "AND f.superseded_by IS NULL AND p.status != 'deleted'",
            (g.user["id"], face_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Face not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    face_row = rows[0]
    old_is_target = face_row["old_is_target"]
    sdc_written = face_row["sdc_written"]
    commons_page_id = face_row["commons_page_id"]
    wikidata_qid = face_row["wikidata_qid"]
    is_bootstrapped = face_row["bootstrapped"]

    # If rejecting (is_target=0): for bootstrap images, queue removal via
    # sdc_removal_pending (worker handles actual API call) — but ONLY if no
    # other face on the same image confirms the target (sibling guard).
    # For non-bootstrap faces with sdc_written=1, remove the claim immediately.
    sdc_removed = False
    sdc_removal_queued = False
    if is_target == 0 and is_bootstrapped:
        # Check if another face on the same image is confirmed as target.
        # If so, the P180 claim must stay — don't queue removal.
        try:
            has_sibling = has_sibling_match(face_row["image_id"], face_id)
        except DatabaseError:
            return jsonify({"error": _("Database error")}), 500

        if not has_sibling:
            sdc_removal_queued = True
    elif is_target == 0 and sdc_written == 1:
        # Check if another face on the same image is confirmed as target.
        # If so, the P180 claim must stay — only clear sdc_written on this face.
        try:
            has_sibling = has_sibling_match(face_row["image_id"], face_id)
        except DatabaseError:
            return jsonify({"error": _("Database error")}), 500

        if not has_sibling:
            access_token = _get_valid_token()
            if not access_token:
                return jsonify({"error": _("OAuth token expired. Please log in again.")}), 401

            sdc_removed = _remove_sdc_claim(commons_page_id, wikidata_qid, access_token)
            if not sdc_removed:
                return jsonify(
                    {"error": _("Failed to remove SDC claim from Commons. The face was not reclassified.")}
                ), 502

    try:

        def _reclassify(conn, cursor):
            cursor.execute(
                "UPDATE faces SET is_target = %s, classified_by = 'human', "
                "classified_by_user_id = %s, sdc_written = %s, "
                "sdc_removal_pending = %s "
                "WHERE id = %s AND (classified_by_user_id IS NULL OR classified_by_user_id = %s)",
                (
                    is_target,
                    g.user["id"],
                    0 if is_target == 0 else sdc_written,
                    1 if sdc_removal_queued else 0,
                    face_id,
                    g.user["id"],
                ),
            )
            if cursor.rowcount == 0:
                # rowcount=0 means either another user owns this face, or the
                # same user clicked the same action again (no columns changed).
                # Re-check to distinguish the two cases.
                cursor.execute(
                    "SELECT classified_by_user_id FROM faces WHERE id = %s",
                    (face_id,),
                )
                check = cursor.fetchone()
                if check and check["classified_by_user_id"] == g.user["id"]:
                    # Same user, same value — treat as no-op success
                    pass
                else:
                    raise ValueError("already_reviewed")

            if is_target == 1 and old_is_target != 1:
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
                    (face_row["project_id"],),
                )
                cursor.execute(
                    "UPDATE faces SET sdc_removal_pending = 0 "
                    "WHERE image_id = %s AND sdc_removal_pending = 1 "
                    "AND superseded_by IS NULL AND id != %s",
                    (face_row["image_id"], face_id),
                )
            elif is_target == 0 and old_is_target == 1:
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = "
                    "GREATEST(0, CAST(faces_confirmed AS SIGNED) - 1) WHERE id = %s",
                    (face_row["project_id"],),
                )

        execute_transaction(_reclassify)

        # When approving, check if P180 already exists on Commons
        if is_target == 1 and commons_page_id and not sdc_written:
            if _maybe_mark_sdc_written(face_id, face_row["image_id"], face_row["project_id"]):
                sdc_written = 1

    except ValueError as e:
        if str(e) == "already_reviewed":
            return jsonify({"error": _("This face has already been reviewed by another user")}), 409
        raise
    except DatabaseError:
        logger.exception("Failed to reclassify face")
        return jsonify({"error": _("Failed to save reclassification")}), 500

    return jsonify(
        {
            "status": "ok",
            "face_id": face_id,
            "is_target": is_target,
            "classified_by": face_row["classified_by"],
            "sdc_removed": sdc_removed,
            "sdc_removal_queued": sdc_removal_queued,
        }
    )


@app.route("/api/update-face-bbox", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_update_face_bbox():
    """Update a face's bounding box and recompute encoding. Creates a new
    face row while keeping the original (both faces coexist)."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    face_id_str = request.form.get("face_id")
    bbox_top_str = request.form.get("bbox_top")
    bbox_right_str = request.form.get("bbox_right")
    bbox_bottom_str = request.form.get("bbox_bottom")
    bbox_left_str = request.form.get("bbox_left")

    if not all([face_id_str, bbox_top_str, bbox_right_str, bbox_bottom_str, bbox_left_str]):
        return jsonify({"error": _("Missing required fields")}), 400

    try:
        face_id = int(face_id_str)  # type: ignore[arg-type]
        bbox_top = int(bbox_top_str)  # type: ignore[arg-type]
        bbox_right = int(bbox_right_str)  # type: ignore[arg-type]
        bbox_bottom = int(bbox_bottom_str)  # type: ignore[arg-type]
        bbox_left = int(bbox_left_str)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return jsonify({"error": _("Invalid field values")}), 400

    if bbox_top >= bbox_bottom or bbox_left >= bbox_right:
        return jsonify({"error": _("Invalid bounding box dimensions")}), 400

    if (
        bbox_top < 0
        or bbox_left < 0
        or bbox_bottom > MAX_BBOX_PX
        or bbox_right > MAX_BBOX_PX
        or (bbox_bottom - bbox_top) * (bbox_right - bbox_left) < MIN_BBOX_AREA
    ):
        return jsonify({"error": _("Bounding box out of allowed range")}), 400

    # Verify ownership: face → image → project → user (or member)
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target, f.classified_by, f.confidence, "
            "  f.classified_by_user_id, f.sdc_written, i.file_title, i.commons_page_id, "
            "  i.status AS image_status, "
            "  p.id AS project_id, p.wikidata_qid "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
            "WHERE f.id = %s AND (p.user_id = %s OR pm.user_id IS NOT NULL) "
            "AND f.superseded_by IS NULL AND p.status != 'deleted'",
            (g.user["id"], face_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Face not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    face_row = rows[0]
    if face_row.get("image_status") != "processed":
        return jsonify({"error": _("Image has not finished processing yet")}), 409
    image_id = face_row["image_id"]
    orig_is_target = face_row["is_target"]
    orig_confidence = face_row["confidence"]
    orig_sdc_written = face_row["sdc_written"]
    file_title = face_row["file_title"]
    clean_title = file_title[5:] if file_title.startswith("File:") else file_title
    url = FILE_PATH_URL.format(file_title=clean_title)

    try:
        image_bytes = _download_image(url)

        import face_recognition  # lazy: dlib is ~130 MB; avoid loading at startup

        image_data = face_recognition.load_image_file(io.BytesIO(image_bytes))
        face_location = [(bbox_top, bbox_right, bbox_bottom, bbox_left)]
        encodings = face_recognition.face_encodings(image_data, face_location)

        if not encodings:
            return jsonify(
                {
                    "error": _(
                        "Could not compute face encoding for the selected region. Try drawing a slightly larger box."
                    )
                }
            ), 422

        encoding_bytes = encodings[0].tobytes()

        def _update_bbox(conn, cursor):
            cursor.execute(
                "INSERT INTO faces "
                "(image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
                " is_target, classified_by, confidence, classified_by_user_id, sdc_written) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'human', %s, %s, %s)",
                (
                    image_id,
                    encoding_bytes,
                    bbox_top,
                    bbox_right,
                    bbox_bottom,
                    bbox_left,
                    orig_is_target,
                    orig_confidence,
                    g.user["id"],
                    orig_sdc_written,
                ),
            )
            new_face_id = cursor.lastrowid
            # Link original face to its replacement — superseded faces are
            # excluded from all stats, inference, and SDC queries.
            cursor.execute(
                "UPDATE faces SET superseded_by = %s WHERE id = %s",
                (new_face_id, face_id),
            )
            return new_face_id

        new_face_id = execute_transaction(_update_bbox)

        if new_face_id and orig_is_target == 1 and not orig_sdc_written:
            _maybe_mark_sdc_written(new_face_id, image_id, face_row["project_id"])

        return jsonify(
            {
                "status": "ok",
                "new_face_id": new_face_id,
                "original_face_id": face_id,
            }
        )

    except requests.RequestException as e:
        logger.error(f"Failed to download image for bbox update: {e}")
        return jsonify({"error": _("Failed to download image from Commons")}), 502
    except DatabaseError:
        logger.exception("Failed to insert updated face")
        return jsonify({"error": _("Failed to save face")}), 500
    except Exception:
        logger.exception("Unexpected error in bbox update")
        return jsonify({"error": _("Failed to process face region")}), 500


@app.route("/api/write-sdc/<int:project_id>", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_write_sdc(project_id: int):
    """Request the background worker to write P180 depicts claims for all
    unwritten approved faces. Sets a flag on the project; the worker picks
    it up on the next poll cycle."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    # Verify project access (owner or member)
    try:
        project = get_project_for_actor(project_id, g.user["id"])
        if not project:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    # Query current counts (needed for both already_requested and normal paths)
    try:
        pending = execute_query(
            "SELECT "
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 0 "
            "    AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 1 ELSE 0 END) AS write_cnt, "
            "  COUNT(DISTINCT CASE WHEN f.sdc_removal_pending = 1 "
            "    AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
            "    AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id) "
            "    THEN f.image_id END) AS removal_cnt "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.superseded_by IS NULL",
            (project_id,),
            fetch=True,
        )
        write_count = (pending[0]["write_cnt"] or 0) if pending else 0
        removal_count = (pending[0]["removal_cnt"] or 0) if pending else 0
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    if project["sdc_write_requested"] == 1:
        return jsonify(
            {
                "status": "ok",
                "message": "already_requested",
                "pending": write_count,
                "removal_pending": removal_count,
            }
        )

    if write_count == 0 and removal_count == 0:
        return jsonify({"status": "ok", "pending": 0, "removal_pending": 0})

    # Refresh token now so the worker has a valid one for API calls
    token = _get_valid_token()
    if token is None:
        return jsonify({"error": _("Your session has expired. Please log out and log in again to re-authorize.")}), 401

    # Set the flag for the worker to pick up
    try:
        execute_query(
            "UPDATE projects SET sdc_write_requested = 1, sdc_write_error = NULL WHERE id = %s",
            (project_id,),
            fetch=False,
        )
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    # Touch wake-up file to reduce latency (worker checks every 60s)
    try:
        with open(WAKE_FILE_PATH, "w") as f:
            f.write("sdc")
    except OSError:
        pass

    return jsonify({"status": "ok", "pending": write_count, "removal_pending": removal_count})


@app.route("/api/sdc-status/<int:project_id>", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def api_sdc_status(project_id: int):
    """Poll endpoint for SDC write progress. Returns counts of written,
    pending, and whether the worker is actively writing."""
    try:
        project = get_project_for_actor(project_id, g.user["id"])
        if not project:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    try:
        counts = execute_query(
            "SELECT "
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 1 THEN 1 ELSE 0 END) AS written, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 1 ELSE 0 END) AS pending, "
            "  COUNT(DISTINCT CASE WHEN f.sdc_removal_pending = 1 "
            "    AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
            "    AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id) "
            "    THEN f.image_id END) AS removal_pending "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.superseded_by IS NULL",
            (project_id,),
            fetch=True,
        )
        written = counts[0]["written"] or 0 if counts else 0
        pending = counts[0]["pending"] or 0 if counts else 0
        removal_pending = counts[0]["removal_pending"] or 0 if counts else 0
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    return jsonify(
        {
            "status": "ok",
            "written": written,
            "pending": pending,
            "removal_pending": removal_pending,
            "in_progress": bool(project["sdc_write_requested"]),
            "error": project["sdc_write_error"],
        }
    )


@app.route("/api/stop-sdc/<int:project_id>", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_stop_sdc(project_id: int):
    """Cancel an in-progress SDC write.  Clears sdc_write_requested so the
    worker stops at the next batch boundary."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    try:
        project = get_project_for_actor(project_id, g.user["id"])
        if not project:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    if not project["sdc_write_requested"]:
        return jsonify({"status": "ok", "message": "not_in_progress"})

    try:
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_error = NULL WHERE id = %s",
            (project_id,),
            fetch=False,
        )
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    return jsonify({"status": "ok"})


@app.route("/api/project/<int:project_id>/gallery", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def api_gallery(project_id: int):
    """Paginated JSON API for the Classification Results gallery.

    Query params:
        page        -- 1-based page number (default 1)
        per_page    -- items per page, capped at 100 (default 27)
        result      -- filter by result type: match | non-match | rejected | all (default all)
        source      -- filter by classification source: model | bootstrap | human | all (default all)
        sdc         -- filter by SDC status: sdc-pending | all (default all)
    """
    try:
        project = get_project_for_actor(project_id, g.user["id"])
        if not project:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 27))))
    except (ValueError, TypeError):
        return jsonify({"error": _("Invalid pagination parameters")}), 400

    result_filter = request.args.get("result", "all")
    source_filter = request.args.get("source", "all")
    sdc_filter = request.args.get("sdc", "all")
    sort_param = request.args.get("sort", "threshold-asc")

    base_where = (
        "i.project_id = %s "
        "AND f.superseded_by IS NULL "
        "AND (f.classified_by IN ('model', 'bootstrap') "
        "     OR (f.classified_by = 'human' AND f.classified_by_user_id IS NOT NULL)) "
        "AND LOWER(i.file_title) NOT REGEXP '\\\\.(webm|ogv|ogg|mp3|wav|flac|opus|mid|oga)$'"
    )
    params: list = [project_id]

    filter_clauses: list[str] = []
    if result_filter == "match":
        filter_clauses.append("f.is_target = 1")
    elif result_filter == "non-match":
        filter_clauses.append("f.is_target != 1")
        filter_clauses.append("NOT (f.is_target = 0 AND f.classified_by_user_id IS NOT NULL)")
    elif result_filter == "rejected":
        filter_clauses.append("f.is_target = 0")
        filter_clauses.append("f.classified_by_user_id IS NOT NULL")

    if source_filter == "model":
        filter_clauses.append("f.classified_by = 'model'")
        filter_clauses.append("f.classified_by_user_id IS NULL")
    elif source_filter == "bootstrap":
        filter_clauses.append("f.classified_by = 'bootstrap'")
        filter_clauses.append("f.classified_by_user_id IS NULL")
    elif source_filter == "human":
        filter_clauses.append("f.classified_by_user_id IS NOT NULL")

    if sdc_filter == "sdc-pending":
        filter_clauses.append(
            "(f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0"
            " OR f.sdc_removal_pending = 1)"
        )

    where = base_where
    if filter_clauses:
        where += " AND " + " AND ".join(filter_clauses)

    try:
        count_rows = execute_query(
            f"SELECT COUNT(*) AS cnt FROM faces f JOIN images i ON f.image_id = i.id WHERE {where}",  # noqa: S608
            tuple(params),
        )
        total = count_rows[0]["cnt"] if count_rows else 0
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    counts = {}
    if page == 1:
        try:
            count_data = execute_query(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN f.is_target = 1 THEN 1 ELSE 0 END) AS matches, "
                "  SUM(CASE WHEN f.is_target != 1 AND NOT (f.is_target = 0 AND f.classified_by_user_id IS NOT NULL) THEN 1 ELSE 0 END) AS non_matches, "
                "  SUM(CASE WHEN f.is_target = 0 AND f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS rejected, "
                "  SUM(CASE WHEN f.classified_by = 'model' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS source_model, "
                "  SUM(CASE WHEN f.classified_by = 'bootstrap' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS source_bootstrap, "
                "  SUM(CASE WHEN f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS source_human, "
                "  SUM(CASE WHEN (f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0) "
                "            OR (f.sdc_removal_pending = 1 AND NOT EXISTS ("
                "                  SELECT 1 FROM faces f2 "
                "                  WHERE f2.image_id = f.image_id "
                "                    AND f2.is_target = 1 "
                "                    AND f2.id != f.id"
                "            )) THEN 1 ELSE 0 END) AS sdc_pending "
                f"FROM faces f JOIN images i ON f.image_id = i.id WHERE {base_where}",  # noqa: S608
                (project_id,),
            )
            if count_data:
                c = count_data[0]
                counts = {
                    "total": c["total"] or 0,
                    "matches": c["matches"] or 0,
                    "non_matches": c["non_matches"] or 0,
                    "rejected": c["rejected"] or 0,
                    "source_model": c["source_model"] or 0,
                    "source_bootstrap": c["source_bootstrap"] or 0,
                    "source_human": c["source_human"] or 0,
                    "sdc_pending": c["sdc_pending"] or 0,
                }
        except DatabaseError:
            pass

    sort_options = {
        "threshold-asc": "COALESCE(f.confidence, 999) ASC",
        "threshold-desc": "COALESCE(f.confidence, 999) DESC",
    }
    sort_clause = sort_options.get(sort_param, sort_options["threshold-asc"])

    offset = (page - 1) * per_page
    try:
        face_rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target, f.confidence, f.classified_by, "
            "  f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left, "
            "  f.sdc_written, f.sdc_removal_pending, f.classified_by_user_id, "
            "  i.file_title, i.commons_page_id, "
            "  i.detection_width, i.detection_height, i.bootstrapped, "
            "  EXISTS ("
            "    SELECT 1 FROM faces f2 "
            "    WHERE f2.image_id = f.image_id "
            "      AND f2.id <> f.id "
            "      AND f2.is_target = 1 "
            "      AND f2.sdc_removal_pending = 0"
            "  ) AS has_confirmed_target_sibling "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            f"WHERE {where} "  # noqa: S608
            "ORDER BY "
            "  (CASE WHEN f.is_target = 1 AND f.sdc_written = 0 "
            "        AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 0 ELSE 1 END), "
            f"  {sort_clause}, f.id ASC "  # noqa: S608
            "LIMIT %s OFFSET %s",
            (*params, per_page, offset),
        )
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    total_pages = max(1, math.ceil(total / per_page))

    faces = []
    for face in face_rows if face_rows else []:
        is_sdc_pending = (
            face["is_target"] == 1
            and face["sdc_written"] == 0
            and face["classified_by"] != "bootstrap"
            and face["bootstrapped"] == 0
        ) or (face["sdc_removal_pending"] == 1 and not face["has_confirmed_target_sibling"])
        if face["is_target"] == 0 and face["classified_by_user_id"]:
            result_type = "rejected"
        elif face["is_target"] == 1:
            result_type = "match"
        else:
            result_type = "non-match"

        faces.append(
            {
                "id": face["id"],
                "image_id": face["image_id"],
                "is_target": face["is_target"],
                "confidence": float(face["confidence"]) if face["confidence"] is not None else None,
                "classified_by": face["classified_by"],
                "classified_by_user_id": face["classified_by_user_id"],
                "bbox_top": face["bbox_top"],
                "bbox_right": face["bbox_right"],
                "bbox_bottom": face["bbox_bottom"],
                "bbox_left": face["bbox_left"],
                "sdc_written": face["sdc_written"],
                "bootstrapped": face["bootstrapped"],
                "file_title": face["file_title"],
                "detection_width": face["detection_width"] or 0,
                "detection_height": face["detection_height"] or 0,
                "thumb_url": commons_thumb_url(face["file_title"], 330),
                "result": result_type,
                "sdc_pending": is_sdc_pending,
            }
        )

    result = {
        "status": "ok",
        "faces": faces,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }
    if counts:
        result["counts"] = counts

    return jsonify(result)


@app.route("/api/progress/<int:project_id>", methods=["GET"])
@login_required
@limiter.limit("30 per minute")
def api_progress(project_id: int):
    """Poll endpoint for image processing progress."""
    try:
        project = get_project_for_actor(project_id, g.user["id"])
        if not project:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    images_processed = project["images_processed"] or 0
    images_total = project["images_total"] or 0

    pending_images = 0
    if project["status"] == "active" and images_total > 0:
        try:
            pending_row = execute_query(
                "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s AND status = 'pending'",
                (project_id,),
            )
            pending_images = pending_row[0]["cnt"] if pending_row else 0
        except DatabaseError:
            pass

    face_stats = {}
    try:
        stat_rows = execute_query(
            "SELECT "
            "  COUNT(*) AS total_faces, "
            "  SUM(CASE WHEN f.is_target = 1 THEN 1 ELSE 0 END) AS confirmed_matches, "
            "  SUM(CASE WHEN f.is_target = 0 THEN 1 ELSE 0 END) AS confirmed_non_matches, "
            "  SUM(CASE WHEN f.is_target IS NULL THEN 1 ELSE 0 END) AS unclassified, "
            "  SUM(CASE WHEN f.sdc_written = 1 THEN 1 ELSE 0 END) AS sdc_written, "
            "  SUM(CASE WHEN (f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0) OR (f.sdc_removal_pending = 1 AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id AND f2.superseded_by IS NULL AND f2.is_target = 1)) THEN 1 ELSE 0 END) AS sdc_pending, "
            "  SUM(CASE WHEN f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS by_human, "
            "  SUM(CASE WHEN f.classified_by = 'model' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_model, "
            "  SUM(CASE WHEN f.classified_by = 'bootstrap' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_bootstrap "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.superseded_by IS NULL",
            (project_id,),
            fetch=True,
        )
        if stat_rows:
            face_stats = {k: (v or 0) for k, v in stat_rows[0].items()}
    except DatabaseError:
        pass

    inference_eligible = 0
    try:
        eligible_row = execute_query(
            "SELECT COUNT(*) AS cnt FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.is_target IS NULL "
            "  AND f.superseded_by IS NULL "
            "  AND f.classified_by_user_id IS NULL",
            (project_id,),
        )
        inference_eligible = eligible_row[0]["cnt"] if eligible_row else 0
    except DatabaseError:
        pass

    return jsonify(
        {
            "status": "ok",
            "images_processed": images_processed,
            "images_total": images_total,
            "pending_images": pending_images,
            "complete": images_total > 0 and images_processed >= images_total,
            "face_stats": face_stats,
            "inference_eligible": inference_eligible,
        }
    )


@app.route("/project/<int:project_id>/settings", methods=["GET", "POST"])
@login_required
def project_settings(project_id: int):
    """Edit project settings."""
    try:
        project = get_project_for_user(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    if request.method == "GET":
        # Fetch project members with usernames
        members = []
        try:
            members = execute_query(
                "SELECT pm.user_id, pm.role, pm.status, pm.joined_at, u.wiki_username "
                "FROM project_members pm "
                "JOIN users u ON pm.user_id = u.id "
                "WHERE pm.project_id = %s "
                "ORDER BY pm.status ASC, pm.joined_at ASC",
                (project_id,),
            )
        except DatabaseError:
            logger.exception("Failed to load project members")
        return render_template("project_settings.html", project=project, members=members, is_owner=True)

    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    distance_threshold = request.form.get("distance_threshold", str(project["distance_threshold"]))
    min_confirmed = request.form.get("min_confirmed", str(project["min_confirmed"]))
    status = request.form.get("status", project["status"])
    label = request.form.get("label", project["label"])

    errors = []
    try:
        distance_threshold = float(distance_threshold)
        if not 0.1 <= distance_threshold <= 1.0:
            errors.append(_("Distance threshold must be between 0.1 and 1.0."))
    except ValueError:
        errors.append(_("Distance threshold must be a number."))
        distance_threshold = project["distance_threshold"]

    try:
        min_confirmed = int(min_confirmed)
        if min_confirmed < 1:
            errors.append(_("Minimum confirmed must be at least 1."))
    except ValueError:
        errors.append(_("Minimum confirmed must be a whole number."))
        min_confirmed = project["min_confirmed"]

    if status not in ("active", "paused", "completed"):
        errors.append(_("Invalid status."))
        status = project["status"]

    if errors:
        for err in errors:
            flash(err, "error")
        project["distance_threshold"] = distance_threshold
        project["min_confirmed"] = min_confirmed
        project["status"] = status
        project["label"] = label
        return render_template("project_settings.html", project=project, members=[], is_owner=True)

    try:
        execute_query(
            "UPDATE projects SET distance_threshold = %s, min_confirmed = %s, "
            "status = %s, label = %s, completion_reason = NULL WHERE id = %s AND user_id = %s",
            (
                distance_threshold,
                min_confirmed,
                status,
                label,
                project_id,
                g.user["id"],
            ),
            fetch=False,
        )
        flash(_("Settings updated."), "success")

        try:
            stats = execute_query(
                "SELECT "
                "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by_user_id IS NOT NULL "
                "      THEN 1 ELSE 0 END) AS human_confirmed, "
                "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by = 'bootstrap' "
                "      AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_bootstrap "
                "FROM faces f "
                "JOIN images i ON f.image_id = i.id "
                "WHERE i.project_id = %s AND f.superseded_by IS NULL",
                (project_id,),
            )
            human_confirmed = (stats[0]["human_confirmed"] or 0) if stats else 0
            by_bootstrap = (stats[0]["by_bootstrap"] or 0) if stats else 0
            if human_confirmed < min_confirmed:
                flash(
                    _(
                        "Warning: You currently have %(confirmed)d/%(min)d human-confirmed target faces. "
                        "Autonomous inference will not run until this threshold is met.",
                        confirmed=human_confirmed,
                        min=min_confirmed,
                    ),
                    "warning",
                )
                if by_bootstrap > 0:
                    flash(
                        _(
                            "Tip: You have %(count)d bootstrapped matches from existing Commons depicts claims. "
                            "Approve them in Model Results to count them as human-confirmed.",
                            count=by_bootstrap,
                        ),
                        "info",
                    )
        except DatabaseError:
            logger.debug("Non-critical: failed to fetch stats after settings update for project %s", project_id)

        return redirect(url_for("project_settings", project_id=project_id))
    except DatabaseError:
        logger.exception("Failed to update project settings")
        flash(_("Failed to update settings."), "error")
        return render_template("project_settings.html", project=project, members=[], is_owner=True)


@app.route("/project/<int:project_id>/settings/remove-member", methods=["POST"])
@login_required
def project_remove_member(project_id: int):
    """Remove a member from the project (owner-only)."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        project = get_project_for_user(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    member_user_id = request.form.get("member_user_id")
    if not member_user_id:
        flash(_("No member specified."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    try:
        member_user_id = int(member_user_id)
    except (ValueError, TypeError):
        flash(_("Invalid member."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    # Don't allow removing yourself (the owner)
    if member_user_id == g.user["id"]:
        flash(_("You cannot remove yourself from your own project."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    try:
        affected = execute_query(
            "UPDATE project_members SET status = 'banned' WHERE project_id = %s AND user_id = %s AND status = 'active'",
            (project_id, member_user_id),
            fetch=False,
        )
        if affected:
            flash(_("Member removed and banned from rejoining."), "success")
        else:
            flash(_("Member not found."), "error")
    except DatabaseError:
        logger.exception("Failed to remove member from project %s", project_id)
        flash(_("Failed to remove member."), "error")

    return redirect(url_for("project_settings", project_id=project_id))


@app.route("/project/<int:project_id>/settings/unban-member", methods=["POST"])
@login_required
def project_unban_member(project_id: int):
    """Unban a member so they can rejoin the project (owner-only)."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        project = get_project_for_user(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    member_user_id = request.form.get("member_user_id")
    if not member_user_id:
        flash(_("No member specified."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    try:
        member_user_id = int(member_user_id)
    except (ValueError, TypeError):
        flash(_("Invalid member."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    try:
        affected = execute_query(
            "DELETE FROM project_members WHERE project_id = %s AND user_id = %s AND status = 'banned'",
            (project_id, member_user_id),
            fetch=False,
        )
        if affected:
            flash(_("Member unbanned."), "success")
        else:
            flash(_("Member not found."), "error")
    except DatabaseError:
        logger.exception("Failed to unban member from project %s", project_id)
        flash(_("Failed to unban member."), "error")

    return redirect(url_for("project_settings", project_id=project_id))


@app.route("/project/<int:project_id>/invite-code", methods=["POST"])
@login_required
def project_invite_code(project_id: int):
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        project = get_project_for_user(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    action = request.form.get("action")

    if action == "generate":
        code = _generate_invite_code()
        for _attempt in range(_MAX_INVITE_CODE_RETRIES):
            try:
                execute_query(
                    "UPDATE projects SET invite_code = %s WHERE id = %s AND user_id = %s",
                    (code, project_id, g.user["id"]),
                    fetch=False,
                )
                flash(_("Invite code generated."), "success")
                break
            except DatabaseError as exc:
                if _is_invite_code_collision(exc) and _attempt < _MAX_INVITE_CODE_RETRIES - 1:
                    logger.debug(
                        "invite_code collision on attempt %d for project %s, retrying",
                        _attempt + 1,
                        project_id,
                    )
                    code = _generate_invite_code()
                else:
                    logger.exception("Failed to generate invite code for project %s", project_id)
                    flash(_("Failed to generate invite code."), "error")
                    break
    elif action == "revoke":
        try:
            execute_query(
                "UPDATE projects SET invite_code = NULL WHERE id = %s AND user_id = %s",
                (project_id, g.user["id"]),
                fetch=False,
            )
            flash(_("Invite code revoked."), "success")
        except DatabaseError:
            logger.exception("Failed to revoke invite code for project %s", project_id)
            flash(_("Failed to revoke invite code."), "error")
    else:
        flash(_("Invalid action."), "error")

    return redirect(url_for("project_settings", project_id=project_id))


@app.route("/join", methods=["POST"])
@login_required
def project_join():
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    code = (request.form.get("invite_code") or "").strip()
    if not code:
        flash(_("Please enter an invite code."), "error")
        return redirect(url_for("dashboard"))

    try:
        project = execute_query(
            "SELECT p.id, p.label, p.user_id, u.wiki_username FROM projects p "
            "JOIN users u ON p.user_id = u.id "
            "WHERE p.invite_code = %s AND p.status != 'deleted' LIMIT 1",
            (code,),
        )
    except DatabaseError:
        logger.exception("Failed to look up invite code")
        flash(_("Something went wrong. Please try again."), "error")
        return redirect(url_for("dashboard"))

    if not project:
        flash(_("Invalid invite code."), "error")
        return redirect(url_for("dashboard"))

    proj = project[0]

    if proj["user_id"] == g.user["id"]:
        flash(_("You are the owner of this project."), "info")
        return redirect(url_for("project_detail", project_id=proj["id"]))

    try:
        existing = execute_query(
            "SELECT status FROM project_members WHERE project_id = %s AND user_id = %s",
            (proj["id"], g.user["id"]),
        )
    except DatabaseError:
        logger.exception("Failed to check membership for project %s", proj["id"])
        flash(_("Something went wrong. Please try again."), "error")
        return redirect(url_for("dashboard"))

    if existing:
        if existing[0]["status"] == "banned":
            flash(_("You have been banned from this project."), "error")
            return redirect(url_for("dashboard"))
        flash(_("You are already a member of this project."), "info")
        return redirect(url_for("project_detail", project_id=proj["id"]))

    try:
        execute_query(
            "INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'member')",
            (proj["id"], g.user["id"]),
            fetch=False,
        )
        flash(
            _(
                'Joined %(username)s\'s project "%(label)s" successfully!',
                username=proj["wiki_username"],
                label=proj["label"] or proj["id"],
            ),
            "success",
        )
        return redirect(url_for("project_detail", project_id=proj["id"]))
    except DatabaseError:
        logger.exception("Failed to join project %s", proj["id"])
        flash(_("Failed to join project."), "error")
        return redirect(url_for("dashboard"))


@app.route("/project/<int:project_id>/rerun-inference", methods=["POST"])
@login_required
def project_rerun_inference(project_id: int):
    """Reset model-classified faces so the worker re-runs inference with current settings."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        project = get_project_for_user(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    if (
        project["last_inference_threshold"] is not None
        and project["last_inference_min_confirmed"] is not None
        and abs(float(project["distance_threshold"]) - float(project["last_inference_threshold"])) < 1e-6
        and int(project["min_confirmed"]) == int(project["last_inference_min_confirmed"])
    ):
        flash(_("Settings have not changed since the last inference run. No re-run needed."), "info")
        return redirect(url_for("project_settings", project_id=project_id))

    affected = 0
    inference_will_run = False

    try:

        def _reset_inference(conn, cursor):
            cursor.execute(
                "UPDATE faces f "
                "JOIN images i ON f.image_id = i.id "
                "SET f.is_target = NULL, f.classified_by = NULL, f.confidence = NULL "
                "WHERE i.project_id = %s "
                "AND f.classified_by = 'model' "
                "AND f.classified_by_user_id IS NULL "
                "AND f.sdc_written = 0 "
                "AND f.superseded_by IS NULL",
                (project_id,),
            )
            rows_reset = cursor.rowcount
            if rows_reset:
                cursor.execute(
                    "UPDATE projects SET last_inference_threshold = NULL, last_inference_min_confirmed = NULL "
                    "WHERE id = %s",
                    (project_id,),
                )
            return rows_reset

        affected = execute_transaction(_reset_inference)
        if affected:
            flash(
                _(
                    "Reset %(num)d model-classified faces. The worker will re-classify them on the next cycle.",
                    num=affected,
                ),
                "success",
            )
            # Signal the worker to wake up and re-run inference immediately
            try:
                with open(WAKE_FILE_PATH, "w") as f:
                    f.write("")
            except OSError:
                pass  # Non-critical — worker will pick it up on next poll
        else:
            flash(_("No model-classified faces to reset."), "info")

        try:
            stats = execute_query(
                "SELECT "
                "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by_user_id IS NOT NULL "
                "      THEN 1 ELSE 0 END) AS human_confirmed, "
                "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by = 'bootstrap' "
                "      AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS by_bootstrap "
                "FROM faces f "
                "JOIN images i ON f.image_id = i.id "
                "WHERE i.project_id = %s AND f.superseded_by IS NULL",
                (project_id,),
            )
            human_confirmed = (stats[0]["human_confirmed"] or 0) if stats else 0
            by_bootstrap = (stats[0]["by_bootstrap"] or 0) if stats else 0
            min_confirmed = int(project["min_confirmed"])
            if human_confirmed >= min_confirmed:
                inference_will_run = True
                flash(
                    _(
                        "You have %(confirmed)d/%(min)d human-confirmed target faces — "
                        "inference will run on the next worker cycle.",
                        confirmed=human_confirmed,
                        min=min_confirmed,
                    ),
                    "info",
                )
            else:
                flash(
                    _(
                        "Warning: You currently have %(confirmed)d/%(min)d human-confirmed target faces. "
                        "Autonomous inference will not run until this threshold is met.",
                        confirmed=human_confirmed,
                        min=min_confirmed,
                    ),
                    "warning",
                )
                if by_bootstrap > 0:
                    flash(
                        _(
                            "Tip: You have %(count)d bootstrapped matches from existing Commons depicts claims. "
                            "Approve them in Model Results to count them as human-confirmed.",
                            count=by_bootstrap,
                        ),
                        "info",
                    )
        except DatabaseError:
            logger.debug("Non-critical: failed to fetch stats after inference reset for project %s", project_id)

    except DatabaseError:
        logger.exception("Failed to reset model-classified faces")
        flash(_("Failed to re-run inference."), "error")

    if affected and inference_will_run:
        return redirect(url_for("project_detail", project_id=project_id, inference_triggered="1"))
    return redirect(url_for("project_detail", project_id=project_id))


@app.route("/project/<int:project_id>/delete", methods=["POST"])
@login_required
def project_delete(project_id: int):
    """Soft-delete a project by setting status to 'deleted'.

    The background worker hard-deletes soft-deleted projects (and their
    cascaded images/faces) once it confirms it is no longer processing them.
    This prevents FK constraint violations from the worker trying to INSERT
    faces for images that were cascade-deleted mid-processing.
    """
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        affected = execute_query(
            "UPDATE projects SET status = 'deleted' WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
            fetch=False,
        )
        if affected:
            flash(_("Project deleted."), "warning")
        else:
            flash(_("Project not found."), "error")
    except DatabaseError as exc:
        logger.exception("Failed to delete project %s: %s", project_id, exc)
        flash(_("Failed to delete project."), "error")

    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Account settings
# ---------------------------------------------------------------------------


@app.route("/account/settings", methods=["GET", "POST"])
@login_required
def account_settings():
    """User account settings page (leaderboard opt-out, etc.)."""
    if request.method == "POST":
        if not _validate_csrf():
            abort(400, _("Invalid CSRF token"))

        opt_out = 1 if request.form.get("leaderboard_opt_out") else 0
        try:
            execute_query(
                "UPDATE users SET leaderboard_opt_out = %s WHERE id = %s",
                (opt_out, g.user["id"]),
                fetch=False,
            )
            flash(_("Settings saved."), "success")
        except DatabaseError:
            logger.exception("Failed to save account settings for user %s", g.user["id"])
            flash(_("Failed to save settings."), "error")

        return redirect(url_for("account_settings"))

    # GET — load current preference
    try:
        rows = execute_query(
            "SELECT leaderboard_opt_out FROM users WHERE id = %s",
            (g.user["id"],),
        )
        opt_out = rows[0]["leaderboard_opt_out"] if rows else 0
    except DatabaseError:
        logger.exception("Failed to load account settings for user %s", g.user["id"])
        opt_out = 0

    return render_template("account_settings.html", leaderboard_opt_out=opt_out)


@app.route("/leaderboard")
def leaderboard():
    """Community leaderboard ranking users by classifications and SDC tags."""
    try:
        rows = execute_query(
            "SELECT "
            "  u.wiki_username, "
            "  COUNT(f.id) + COALESCE(us.classifications, 0) "
            "    AS classifications, "
            "  COUNT(CASE WHEN f.sdc_written = 1 THEN 1 END) "
            "    + COALESCE(us.sdc_tags, 0) AS sdc_tags "
            "FROM users u "
            "LEFT JOIN faces f ON f.classified_by_user_id = u.id "
            "  AND f.superseded_by IS NULL "
            "LEFT JOIN user_stats us ON us.user_id = u.id "
            "WHERE u.leaderboard_opt_out = 0 "
            "  AND (f.id IS NOT NULL OR us.user_id IS NOT NULL) "
            "GROUP BY u.id, u.wiki_username, us.classifications, us.sdc_tags "
            "ORDER BY (COUNT(f.id) + COALESCE(us.classifications, 0) "
            "        + COUNT(CASE WHEN f.sdc_written = 1 THEN 1 END) "
            "        + COALESCE(us.sdc_tags, 0)) DESC, "
            "         (COUNT(f.id) + COALESCE(us.classifications, 0)) DESC "
            "LIMIT 100",
        )
    except DatabaseError:
        logger.exception("Failed to load leaderboard")
        rows = []
        flash(_("Failed to load leaderboard."), "error")

    totals = {"classifications": 0, "sdc_tags": 0}
    for row in rows:
        totals["classifications"] += row["classifications"]
        totals["sdc_tags"] += row["sdc_tags"]

    return render_template("leaderboard.html", rows=rows, totals=totals)


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------


@app.route("/sw.js")
@limiter.exempt
def service_worker():
    """Serve the service worker from the root scope."""
    response = app.send_static_file("sw.js")
    response.headers["Content-Type"] = "application/javascript"
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
@limiter.exempt
def chrome_devtools_json():
    """Silence Chrome DevTools auto-request."""
    return jsonify({}), 200


@app.route("/health")
@limiter.exempt
def health():
    """Health check endpoint for Toolforge monitoring."""
    try:
        rows = execute_query("SELECT 1 AS ok")
        if rows and rows[0].get("ok") == 1:
            return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception:
        logger.exception("Health check failed")
        return jsonify({"status": "unhealthy", "error": "database unavailable"}), 503

    return jsonify({"status": "unhealthy"}), 503


@app.route("/robots.txt")
@limiter.exempt
def robots_txt():
    """Serve robots.txt to override Toolforge default (Disallow: /)."""
    lines = [
        "User-agent: *",
        "Allow: /$",
        "Allow: /leaderboard",
        "Disallow: /",
        "",
        f"Sitemap: {request.url_root}sitemap.xml",
    ]
    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/sitemap.xml")
@limiter.exempt
def sitemap_xml():
    """Serve a minimal XML sitemap listing publicly crawlable pages."""
    base = request.url_root.rstrip("/")
    urls = ["/", "/leaderboard"]
    xml_urls = "".join(f"<url><loc>{base}{path}</loc></url>" for path in urls)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{xml_urls}"
        "</urlset>"
    )
    return xml, 200, {"Content-Type": "application/xml; charset=utf-8"}


@app.route("/commons-thumb/<path:file_title>")
@login_required
def commons_thumb_route(file_title: str):
    """
    Redirect to the correct Wikimedia Commons thumbnail URL for a file.

    Accepts an optional ``width`` query parameter (default 330).
    Width is snapped to the nearest standard Commons thumbnail step.
    Handles video, TIFF, and SVG files that need non-standard thumb URLs.
    """
    width = request.args.get("width", 330, type=int)
    thumb_url = commons_thumb_url(file_title, width)
    return redirect(thumb_url)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors."""
    msg = e.description if hasattr(e, "description") and isinstance(e.description, str) else _("Bad request")
    return render_template("error.html", code=400, message=msg), 400


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template("error.html", code=404, message=_("Page not found")), 404


@app.errorhandler(429)
def rate_limited(e):
    """Handle rate limit exceeded."""
    return (
        render_template(
            "error.html",
            code=429,
            message=_("Rate limit exceeded. Please try again later."),
        ),
        429,
    )


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return (
        render_template("error.html", code=500, message=_("Internal server error")),
        500,
    )


# ---------------------------------------------------------------------------
# App factory / startup
# ---------------------------------------------------------------------------

# Initialize DB pool at import time so gunicorn `app:app` works without a
# factory call.  The `create_app()` factory is kept for backwards compat
# (e.g. tests, one-off scripts) but is no longer required for production.
init_db(pool_size=2)


def create_app() -> Flask:
    """Application factory for programmatic usage and testing."""
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
