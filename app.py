"""
WikiVisage — Flask web application.

Active-learning facial recognition tool for Wikimedia Commons.
Provides OAuth 2.0 authentication, project management, and an active
learning interface for classifying detected faces.
"""

APP_VERSION = "0.3.4"

import hashlib
import io
import logging
import os
import random
import secrets
import time
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any
from urllib.parse import urlparse

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

from database import (
    DatabaseError,
    execute_query,
    execute_transaction,
    init_db,
)

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

# Beta whitelist — fetched from GitHub every 5 minutes, falls back to local file
_WHITELIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whitelist.txt")
_WHITELIST_URL = "https://raw.githubusercontent.com/DiFronzo/WikiVisage/main/whitelist.txt"
_WHITELIST_CACHE_TTL = 300  # seconds
_whitelist_cache: set[str] = set()
_whitelist_cache_time: float = 0.0


def _parse_whitelist(text: str) -> set[str]:
    """Parse whitelist text into a set of usernames."""
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def _load_whitelist() -> set[str]:
    """Return cached whitelist, refreshing from GitHub every 5 minutes.

    Fetch order: GitHub raw → local file → last-known-good cache.
    """
    global _whitelist_cache, _whitelist_cache_time

    now = time.monotonic()
    if _whitelist_cache and (now - _whitelist_cache_time) < _WHITELIST_CACHE_TTL:
        return _whitelist_cache

    # Try GitHub first
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

# Rate limiter — memory:// is per-process (not shared across gunicorn workers).
# With 2 gunicorn workers, effective limits are doubled (e.g., 200/hr becomes ~400/hr).
# Acceptable for a whitelisted beta tool on Toolforge with limited user count.
# For broader deployments, switch to Redis or memcached storage.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
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
    if referrer:
        ref_parsed = urlparse(referrer)
        # Only allow same-host referrer redirects
        if ref_parsed.netloc and ref_parsed.netloc != request.host:
            referrer = url_for("index")
    if not referrer:
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
                # Enforce whitelist on every request (not just login).
                # Fail-closed: empty whitelist = deny all (prevents bypass if both sources fail).
                allowed = _load_whitelist()
                if not allowed or g.user["wiki_username"] not in allowed:
                    logger.warning(f"Session revoked for user not on whitelist: {g.user['wiki_username']}")
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
    """Check that a redirect URL is safe (relative, no scheme/netloc)."""
    if not target:
        return False
    parsed = urlparse(target)
    return parsed.scheme == "" and parsed.netloc == ""


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Raises ValueError if the response exceeds max_bytes.
    """
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

        execute_query(
            "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s WHERE id = %s",
            (
                new_token["access_token"],
                new_token.get("refresh_token", user["refresh_token"]),
                new_expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                user["id"],
            ),
            fetch=False,
        )

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

    base = f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5[0]}/{md5[0:2]}/{clean}"

    if ext in _VIDEO_EXTENSIONS:
        return f"{base}/{width}px--{clean}.jpg"
    elif ext in _CONVERT_TO_JPG_EXTENSIONS:
        return f"{base}/{width}px-{clean}.jpg"
    elif ext in _CONVERT_TO_PNG_EXTENSIONS:
        return f"{base}/{width}px-{clean}.png"
    else:
        return f"{base}/{width}px-{clean}"


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
                    token["access_token"],
                    token.get("refresh_token", ""),
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
                    token["access_token"],
                    token.get("refresh_token", ""),
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
def logout():
    """Clear session and log out."""
    if not _validate_csrf():
        abort(403)
    session.clear()
    flash(_("You have been logged out."), "info")
    return redirect(url_for("index"))


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
    PROJECTS_PER_PAGE = 25

    try:
        page = max(1, int(request.args.get("page", 1)))
    except (ValueError, TypeError):
        page = 1

    offset = (page - 1) * PROJECTS_PER_PAGE

    try:
        count_row = execute_query(
            "SELECT COUNT(*) AS cnt FROM projects WHERE user_id = %s AND status != 'deleted'",
            (g.user["id"],),
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
            "SELECT * FROM projects WHERE user_id = %s AND status != 'deleted' ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            (g.user["id"], PROJECTS_PER_PAGE, offset),
        )
    except DatabaseError:
        logger.exception("Failed to load projects")
        projects = []
        flash(_("Failed to load projects."), "error")

    # Lazily populate P18 thumbnails for projects missing them
    if isinstance(projects, list):
        for proj in projects:
            if not proj.get("p18_thumb_url") and proj.get("wikidata_qid"):
                thumb = _fetch_p18_thumb_url(proj["wikidata_qid"])
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
    file counts. Bounded to MAX_CATS categories and a wall-clock timeout to
    stay responsive.
    """
    category = request.args.get("category", "").strip()
    if not category:
        return jsonify({"error": "missing category"}), 400

    if len(category) > 200 or any(c in category for c in "|\n\r\x00"):
        return jsonify({"error": "invalid category name"}), 400

    MAX_CATS = 50
    TIMEOUT = 8

    deadline = time.monotonic() + TIMEOUT

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
                    "cmlimit": "500",
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

        while cat_queue and len(visited) < MAX_CATS and time.monotonic() < deadline:
            # Batch up to 50 titles for categoryinfo
            batch = []
            while cat_queue and len(batch) < 50 and len(visited) + len(batch) < MAX_CATS:
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
                if time.monotonic() >= deadline or len(visited) >= MAX_CATS:
                    break
                sub_resp = requests.get(
                    COMMONS_API_URL,
                    params={
                        "action": "query",
                        "list": "categorymembers",
                        "cmtitle": sub_title,
                        "cmtype": "subcat",
                        "cmlimit": "500",
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

        approximate = approximate or len(visited) >= MAX_CATS or (cat_queue and time.monotonic() >= deadline)

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

    # Validate Commons category exists
    if not errors and commons_category:
        if not _commons_category_exists(commons_category):
            errors.append(
                _(
                    'The Commons category "%(category)s" does not exist.',
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

    # Create project
    try:
        p18_thumb_url = _fetch_p18_thumb_url(wikidata_qid)

        if not label:
            label = _fetch_wikidata_label(wikidata_qid) or ""

        execute_query(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
            "distance_threshold, min_confirmed, p18_thumb_url) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                g.user["id"],
                wikidata_qid,
                commons_category,
                label,
                distance_threshold,
                min_confirmed,
                p18_thumb_url,
            ),
            fetch=False,
        )
        flash(_("Project created successfully!"), "success")

        # Signal the worker to wake up and process the new project immediately
        try:
            wake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")
            with open(wake_file, "w") as f:
                f.write("")
        except OSError:
            pass  # Non-critical — worker will pick it up on next poll

        return redirect(url_for("dashboard"))
    except DatabaseError as exc:
        # MySQL error 1062 (ER_DUP_ENTRY) means a soft-deleted row with the same
        # user_id + wikidata_qid + commons_category still exists.  The background
        # worker will hard-delete it on the next poll cycle (≤60 s).
        def _is_duplicate_entry_error(db_exc: Exception) -> bool:
            """
            Return True if the given database exception represents a MySQL
            duplicate-entry (error code 1062) condition.
            """
            # SQLAlchemy-style wrappers often expose the underlying DB-API error
            # via an `orig` attribute.
            orig = getattr(db_exc, "orig", None)
            if orig is not None and getattr(orig, "args", None):
                try:
                    return int(orig.args[0]) == 1062
                except (ValueError, TypeError, IndexError):
                    pass
            # Fall back to checking the exception's own args, as used by many
            # MySQL DB-API drivers where args[0] is the numeric error code.
            if getattr(db_exc, "args", None):
                try:
                    return int(db_exc.args[0]) == 1062
                except (ValueError, TypeError, IndexError):
                    pass
            return False

        if _is_duplicate_entry_error(exc):
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
        rows = execute_query(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        logger.exception("Failed to load project")
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

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

    # Get model-classified faces for the results gallery (sorted by confidence)
    # Exclude video/audio files — only show images with renderable thumbnails
    model_faces: list = []
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target, f.confidence, f.classified_by, "
            "  f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left, "
            "  f.sdc_written, f.classified_by_user_id, "
            "  i.file_title, i.commons_page_id, "
            "  i.detection_width, i.detection_height, i.bootstrapped "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.superseded_by IS NULL "
            "  AND (f.classified_by IN ('model', 'bootstrap') "
            "       OR (f.is_target = 0 AND f.classified_by = 'human' AND f.classified_by_user_id IS NOT NULL)) "
            "  AND LOWER(i.file_title) NOT REGEXP '\\\\.(webm|ogv|ogg|mp3|wav|flac|opus|mid|oga)$' "
            "ORDER BY f.is_target DESC, COALESCE(f.confidence, 999) ASC "
            "LIMIT 200",
            (project_id,),
        )
        if isinstance(rows, list):
            model_faces = rows
    except DatabaseError:
        logger.exception("Failed to load model faces")

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

    return render_template(
        "project_detail.html",
        project=project,
        stats=face_stats,
        model_faces=model_faces,
        pending_images=pending_images,
        inference_eligible=inference_eligible,
    )


@app.route("/project/<int:project_id>/classify")
@login_required
def classify(project_id: int):
    """Active learning classification interface."""
    # Verify project ownership
    try:
        rows = execute_query(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

    # Handle skipped images (stored in session, reset when project changes)
    skip_key = f"skipped_images_{project_id}"
    skip_key_review = f"skipped_images_review_{project_id}"
    is_skip_review = request.args.get("skip_reviewing") == "1"
    if request.args.get("skip_image_id"):
        try:
            skip_id = int(request.args.get("skip_image_id"))
            active_skip_key = skip_key_review if is_skip_review else skip_key
            skipped = session.get(active_skip_key, [])
            if skip_id not in skipped:
                skipped.append(skip_id)
            session[active_skip_key] = skipped
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
        check = execute_query(
            "SELECT i.id FROM images i "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = %s AND p.id = %s AND p.user_id = %s AND p.status != 'deleted'",
            (image_id, project_id, g.user["id"]),
        )
        if not check:
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

                cursor.execute(
                    "UPDATE faces SET is_target = 1, classified_by = 'human', "
                    "classified_by_user_id = %s "
                    "WHERE id = %s AND image_id = %s",
                    (g.user["id"], selected_face_id, image_id),
                )

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

                cursor.execute(
                    "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
                    (project_id,),
                )

                # Do NOT queue P180 removal here — the selected face confirms
                # the person IS depicted, so the P180 claim must stay.

                return ids

            affected_ids = execute_transaction(_classify_target)

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
    face_ids = last.get("face_ids", [])
    manual_face_ids = last.get("manual_face_ids", [])

    if not face_ids and not manual_face_ids:
        session.pop("last_classify", None)
        return jsonify({"error": _("Nothing to undo")}), 400

    # Verify ownership
    try:
        check = execute_query(
            "SELECT i.id FROM images i "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = %s AND p.id = %s AND p.user_id = %s AND p.status != 'deleted'",
            (image_id, project_id, g.user["id"]),
        )
        if not check:
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

    # Constrain bbox to reasonable image bounds
    MAX_BBOX_PX = 10000
    MIN_BBOX_AREA = 100  # 10×10 minimum
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

    try:
        check = execute_query(
            "SELECT i.id, i.file_title FROM images i "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE i.id = %s AND p.id = %s AND p.user_id = %s AND p.status != 'deleted'",
            (image_id, project_id, g.user["id"]),
        )
        if not check:
            return jsonify({"error": _("Image not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    file_title = check[0]["file_title"]
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
                    "(image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        image_id,
                        encoding_bytes,
                        bbox_top,
                        bbox_right,
                        bbox_bottom,
                        bbox_left,
                    ),
                )
                new_face_id = cursor.lastrowid

            return new_face_id, review_confirmed_ids

        new_face_id, review_confirmed_ids = execute_transaction(_insert_manual_face)

        if new_face_id:
            manual_key = f"manual_faces_{image_id}"
            manual_list = session.get(manual_key, [])
            manual_list.append(new_face_id)
            session[manual_key] = manual_list

            session["last_classify"] = {
                "project_id": project_id,
                "image_id": image_id,
                "action": "manual_face",
                "face_ids": review_confirmed_ids,
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

    # Verify ownership: face → image → project → user
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target AS old_is_target, f.sdc_written, "
            "  f.classified_by, "
            "  i.commons_page_id, i.bootstrapped, "
            "  p.id AS project_id, p.wikidata_qid "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE f.id = %s AND p.user_id = %s AND f.superseded_by IS NULL AND p.status != 'deleted'",
            (face_id, g.user["id"]),
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
            sibling_rows = execute_query(
                "SELECT EXISTS("
                "  SELECT 1 FROM faces f2 "
                "  WHERE f2.image_id = %s AND f2.is_target = 1 "
                "  AND f2.superseded_by IS NULL AND f2.id != %s"
                ") AS has_sibling_match",
                (face_row["image_id"], face_id),
            )
            has_sibling = sibling_rows[0]["has_sibling_match"] if sibling_rows else False
        except DatabaseError:
            return jsonify({"error": _("Database error")}), 500

        if not has_sibling:
            sdc_removal_queued = True
    elif is_target == 0 and sdc_written == 1:
        # Check if another face on the same image is confirmed as target.
        # If so, the P180 claim must stay — only clear sdc_written on this face.
        try:
            sibling_rows = execute_query(
                "SELECT EXISTS("
                "  SELECT 1 FROM faces f2 "
                "  WHERE f2.image_id = %s AND f2.is_target = 1 "
                "  AND f2.superseded_by IS NULL AND f2.id != %s"
                ") AS has_sibling_match",
                (face_row["image_id"], face_id),
            )
            has_sibling = sibling_rows[0]["has_sibling_match"] if sibling_rows else False
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
                "UPDATE faces SET is_target = %s, "
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

    # Constrain bbox to reasonable image bounds
    MAX_BBOX_PX = 10000
    MIN_BBOX_AREA = 100  # 10×10 minimum
    if (
        bbox_top < 0
        or bbox_left < 0
        or bbox_bottom > MAX_BBOX_PX
        or bbox_right > MAX_BBOX_PX
        or (bbox_bottom - bbox_top) * (bbox_right - bbox_left) < MIN_BBOX_AREA
    ):
        return jsonify({"error": _("Bounding box out of allowed range")}), 400

    # Verify ownership: face → image → project → user
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target, f.classified_by, f.confidence, "
            "  f.classified_by_user_id, i.file_title, p.id AS project_id "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE f.id = %s AND p.user_id = %s AND f.superseded_by IS NULL AND p.status != 'deleted'",
            (face_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Face not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    face_row = rows[0]
    image_id = face_row["image_id"]
    orig_is_target = face_row["is_target"]
    orig_confidence = face_row["confidence"]
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
            # Insert new face row with classification carried over from original
            cursor.execute(
                "INSERT INTO faces "
                "(image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, "
                " is_target, classified_by, confidence, classified_by_user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'human', %s, %s)",
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

    # Verify project ownership
    try:
        rows = execute_query(
            "SELECT id, sdc_write_requested FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    project = rows[0]

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
        wake_up_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")
        with open(wake_up_path, "w") as f:
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
        rows = execute_query(
            "SELECT id, sdc_write_requested, sdc_write_error FROM projects WHERE id = %s AND user_id = %s "
            "AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    project = rows[0]

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


@app.route("/api/progress/<int:project_id>", methods=["GET"])
@login_required
@limiter.limit("30 per minute")
def api_progress(project_id: int):
    """Poll endpoint for image processing progress."""
    try:
        rows = execute_query(
            "SELECT images_processed, images_total, status FROM projects "
            "WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
        if not rows:
            return jsonify({"error": _("Project not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    project = rows[0]
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
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 1 ELSE 0 END) AS sdc_pending, "
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
        rows = execute_query(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

    if request.method == "GET":
        return render_template("project_settings.html", project=project)

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
        return render_template("project_settings.html", project=project)

    try:
        execute_query(
            "UPDATE projects SET distance_threshold = %s, min_confirmed = %s, "
            "status = %s, label = %s WHERE id = %s AND user_id = %s",
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
            pass

        return redirect(url_for("project_detail", project_id=project_id))
    except DatabaseError:
        logger.exception("Failed to update project settings")
        flash(_("Failed to update settings."), "error")
        return render_template("project_settings.html", project=project)


@app.route("/project/<int:project_id>/rerun-inference", methods=["POST"])
@login_required
def project_rerun_inference(project_id: int):
    """Reset model-classified faces so the worker re-runs inference with current settings."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        rows = execute_query(
            "SELECT id, distance_threshold, min_confirmed, "
            "last_inference_threshold, last_inference_min_confirmed "
            "FROM projects WHERE id = %s AND user_id = %s AND status != 'deleted'",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

    if (
        project["last_inference_threshold"] is not None
        and project["last_inference_min_confirmed"] is not None
        and abs(float(project["distance_threshold"]) - float(project["last_inference_threshold"])) < 1e-6
        and int(project["min_confirmed"]) == int(project["last_inference_min_confirmed"])
    ):
        flash(_("Settings have not changed since the last inference run. No re-run needed."), "info")
        return redirect(url_for("project_settings", project_id=project_id))

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
                wake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")
                with open(wake_file, "w") as f:
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
            pass

    except DatabaseError:
        logger.exception("Failed to reset model-classified faces")
        flash(_("Failed to re-run inference."), "error")

    return redirect(url_for("project_settings", project_id=project_id))


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
            "WHERE (f.id IS NOT NULL OR us.user_id IS NOT NULL) "
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
        return (
            app.send_static_file("sw.js"),
            200,
            {"Content-Type": "application/javascript", "Service-Worker-Allowed": "/"},
        )

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
    return render_template("error.html", code=400, message=str(e)), 400


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
