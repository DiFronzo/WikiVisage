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

import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any
from urllib.parse import quote, urljoin, urlparse, urlunparse
from xml.sax.saxutils import escape as xml_escape

from dotenv import load_dotenv

load_dotenv()

import requests
from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask.sessions import SecureCookieSessionInterface
from flask_babel import Babel
from flask_babel import gettext as _
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from itsdangerous import BadSignature, SignatureExpired
from requests_oauthlib import OAuth2Session

import face_client
import oauth_refresh
from api_errors import ApiError
from config import (
    COMMONS_THUMB_STEPS,
    DETECTION_WIDTH,
    LIFTWING_PROBE_INTERVAL,
    SDC_REAUTH_MESSAGES,
    SESSION_IDLE_DAYS,
    SESSION_MAX_AGE_DAYS,
    USER_AGENT,
    WAKE_FILE_PATH,
    WORKER_STALE_SECONDS,
)
from database import (
    DatabaseError,
    execute_query,
    execute_transaction,
    init_db,
)
from face_client import FaceServiceError, FaceServiceRejected
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
_flask_secret = os.environ.get("FLASK_SECRET_KEY")
if not _flask_secret:
    _flask_secret = secrets.token_hex(32)
    logging.getLogger(__name__).warning(
        "FLASK_SECRET_KEY not set — using a random key. "
        "Sessions will not survive restarts or work across gunicorn workers."
    )
app.secret_key = _flask_secret
app.debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}


def _insecure_http_enabled() -> bool:
    """Whether the LAN dev mode is on: the env flag, never on Toolforge (its envvars are tool-wide)."""
    if os.environ.get("WIKIVISAGE_INSECURE_HTTP", "").strip().lower() not in {"1", "true", "yes"}:
        return False
    if os.environ.get("TOOL_TOOLFORGE_API_URL"):
        logger.error("WIKIVISAGE_INSECURE_HTTP is set on Toolforge: ignoring it, cookies keep the Secure flag.")
        return False
    return True


# Local-only escape hatch for testing from another device (a phone, a second laptop) over plain
# http. Leave it unset everywhere else: with it on, the session and locale cookies lose the
# Secure flag, so anyone on the same network can read them. Production runs HTTPS and never
# sets it. The server binds beyond loopback when FLASK_DEBUG is off (see ``__main__``).
INSECURE_HTTP = _insecure_http_enabled()
if INSECURE_HTTP:
    logger.warning(
        "WIKIVISAGE_INSECURE_HTTP is set: cookies are sent without Secure and the app may be "
        "reachable from the local network. Use this for local testing only."
    )


def _cookie_secure() -> bool:
    """Whether cookies carry the Secure flag: off in debug and in the local-network dev mode."""
    return not app.debug and not INSECURE_HTTP


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

# Session configuration
app.config["SESSION_COOKIE_SECURE"] = _cookie_secure()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Sliding idle timeout: every response re-sends the (permanent) cookie, so the lifetime only
# runs down while the app is closed. SESSION_MAX_AGE_DAYS is the absolute cap since login.
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=SESSION_IDLE_DAYS)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
_SESSION_MAX_AGE_SECONDS = SESSION_MAX_AGE_DAYS * 86400
# Sampled INFO line explaining why an anonymous request arrived without a session.
_LOG_SESSION_LOSS = os.environ.get("WIKIVISAGE_LOG_SESSION_LOSS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_SESSION_LOSS_SAMPLE_RATE = 0.05
_SESSION_LOSS_SKIP_PREFIXES = (
    "/static/",
    "/health",
    "/sw.js",
    "/robots.txt",
    "/sitemap.xml",
    "/llms.txt",
    "/commons-thumb/",
    "/.well-known/",
)
LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# Rate limiter — uses Redis on Toolforge for shared state across gunicorn workers.
# Falls back to memory:// for local development where Redis may not be available.
_REDIS_URL = os.environ.get("WIKIVISAGE_REDIS_URL", "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379")
_limiter_storage_uri = _REDIS_URL
# `redis_client` is shared with the OAuth single-flight lock (see redis_lock.py).
# None means we degraded to memory:// — the lock becomes a no-op (best-effort).
redis_client: Any = None
# True iff the rate limiter is using shared Redis storage. Surfaced in /health
# so monitoring can alert when we're degraded to per-process memory limits.
_LIMITER_REDIS_OK = False

try:
    import redis as _redis_mod

    _r = _redis_mod.from_url(_REDIS_URL, socket_connect_timeout=2)
    _r.ping()
    redis_client = _r
    _LIMITER_REDIS_OK = True
except Exception:
    logger.error("Redis unavailable at %s — rate limiter using per-process memory storage", _REDIS_URL)
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
    """Set the user's preferred language via cookie (persistent only with consent)."""
    if lang not in LANGUAGES:
        lang = "en"
    referrer = (request.referrer or "").replace("\\", "")
    if referrer:
        parsed = urlparse(referrer)
        if parsed.netloc:
            if parsed.netloc == urlparse(request.host_url).netloc:
                referrer = parsed.path or "/"
                if parsed.query:
                    referrer = f"{referrer}?{parsed.query}"
            else:
                referrer = ""
    if not referrer or not _is_safe_redirect_target(referrer):
        referrer = url_for("index")
    # Guard: only redirect to relative paths (no scheme, no netloc, no //)
    parsed_ref = urlparse(referrer)
    if parsed_ref.scheme or parsed_ref.netloc or referrer.startswith("//"):
        referrer = url_for("index")
    resp = redirect(referrer)
    # Without consent (?nocookie=1) the cookie is session-scoped, so switching language
    # works without having to accept the banner first. Nothing is remembered afterwards.
    _set_locale_cookie(resp, lang, persistent=not request.args.get("nocookie"))
    return resp


# Exact values the ``locale`` cookie may hold. The input reaches _set_locale_cookie() from a
# URL segment and from the client-controlled cookie, so the sink maps it to a literal instead
# of writing request text (CodeQL's cookie-injection query flags anything else).
_COOKIE_LOCALES = {"en": "en", "nb": "nb", "es": "es", "fr": "fr"}


def _set_locale_cookie(resp: Any, lang: str, *, persistent: bool) -> None:
    """Write the (HttpOnly) language cookie, kept for a year only when *persistent*."""
    resp.set_cookie(
        "locale",
        _COOKIE_LOCALES.get(lang, "en"),
        max_age=LOCALE_COOKIE_MAX_AGE if persistent else None,
        httponly=True,
        samesite="Lax",
        secure=_cookie_secure(),
    )


@app.route("/cookie-consent", methods=["POST"])
@limiter.limit("20 per minute")
def cookie_consent():
    """Apply the visitor's cookie choice to the language cookie (HttpOnly, so JS cannot).

    ``choice=accepted`` keeps an existing ``locale`` cookie for a year, ``declined`` turns it
    back into a session-scoped one. Without a ``locale`` cookie there is nothing to change.
    """
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))
    choice = request.form.get("choice", "")
    if choice not in ("accepted", "declined"):
        abort(400, _("Bad request"))
    resp = Response(status=204)
    locale = request.cookies.get("locale")
    if locale in LANGUAGES:
        _set_locale_cookie(resp, locale, persistent=choice == "accepted")
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


# Wikimedia Commons file URL pattern (used for manual face detection). Same width
# as the worker's detection (WIKIVISAGE_DETECTION_WIDTH), so a drawn box and the
# stored detection space differ by a scale factor of about 1.
FILE_PATH_URL = f"https://commons.wikimedia.org/wiki/Special:FilePath/{{file_title}}?width={DETECTION_WIDTH}"

# Default cosine-distance threshold for new projects. Provisional until calibrated
# on real data; keep in step with schema.sql.
DEFAULT_DISTANCE_THRESHOLD = 0.5
# Files that cannot show a face. The crawler no longer inserts them, but rows
# from older crawls may exist, so the gallery still excludes them. Comparing the
# last dot-separated segment is equivalent to the former `REGEXP '\\.(ext)$'`
# without running a regex over every row of every page.
_NON_VISUAL_EXTENSIONS = ("webm", "ogv", "ogg", "mp3", "wav", "flac", "opus", "mid", "oga")
_EXCLUDE_NON_VISUAL_SQL = "SUBSTRING_INDEX(LOWER(i.file_title), '.', -1) NOT IN ({})".format(
    ", ".join(f"'{ext}'" for ext in _NON_VISUAL_EXTENSIONS)
)
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
    # Set when the user row could not be read (database trouble): the session is kept and
    # login_required answers 503 instead of bouncing the user through a pointless login.
    g.user_load_failed = False
    # Set when a token refresh failed for a reason that retrying may fix (see _get_valid_token).
    g.token_refresh_transient = False
    user_id = session.get("user_id")
    if not user_id:
        _log_session_loss()
        return
    if not _session_within_max_age():
        _end_session(_("Your session has expired. Please log in again."))
        return
    _load_session_user(user_id)


def _session_within_max_age() -> bool:
    """Enforce the absolute session age; sessions that predate ``auth_at`` are stamped, not logged out."""
    now = int(time.time())
    auth_at = session.get("auth_at")
    if not isinstance(auth_at, int):
        session["auth_at"] = now
        return True
    return now - auth_at <= _SESSION_MAX_AGE_SECONDS


def _end_session(message: str | None = None) -> None:
    """Drop everything this browser holds in its session, optionally explaining why."""
    session.clear()
    session.permanent = True  # the flash below must not travel in a session-scoped cookie
    if message:
        flash(message, "warning")


def _load_session_user(user_id: int) -> None:
    """Fill g.user from the database; a revoked epoch or undecryptable tokens end the session."""
    try:
        rows = execute_query(
            "SELECT id, wiki_user_id, wiki_username, access_token, "
            "refresh_token, token_expires_at, session_epoch FROM users WHERE id = %s",
            (user_id,),
        )
        if not rows:
            return
        user = rows[0]
        # A missing key means 0 on both sides, so sessions created before the epoch existed survive.
        if session.get("epoch", 0) != user.get("session_epoch", 0):
            _end_session(_("Your session has expired. Please log in again."))
            return
        # Normalize token types (PyMySQL may return BLOB columns as bytes)
        for _tk in ("access_token", "refresh_token"):
            if isinstance(user.get(_tk), bytes):
                user[_tk] = user[_tk].decode("utf-8")
            user[_tk] = decrypt_token(user[_tk])
        g.user = user
    except TokenDecryptionError:
        logger.warning("Token decryption failed for user %s — clearing session to force re-auth", user_id)
        _end_session()
    except DatabaseError:
        # Transient: keep the session so a database hiccup does not log the user out.
        logger.exception("Failed to load user from session")
        g.user = None
        g.user_load_failed = True


def _client_bucket(user_agent: str) -> str:
    """Coarse client class for the session-loss log line (an installed app looks like its browser)."""
    ua = user_agent.lower()
    if any(word in ua for word in ("bot", "crawler", "spider")):
        return "other"
    if any(word in ua for word in ("iphone", "ipad", "ipod")):
        return "ios"
    if "android" in ua:
        return "android"
    if any(word in ua for word in ("windows", "macintosh", "x11", "linux", "cros")):
        return "desktop"
    return "other"


def _session_loss_reason() -> str:
    """Why an anonymous request has no session: no cookie, expired, bad signature, or valid but empty."""
    raw = request.cookies.get("session")
    if not raw:
        return "no cookie sent"
    serializer = app.session_interface.get_signing_serializer(app)  # type: ignore[attr-defined]
    if serializer is None:
        return "bad signature"
    try:
        serializer.loads(raw, max_age=int(app.permanent_session_lifetime.total_seconds()))
    except SignatureExpired:
        return "expired"
    except BadSignature:
        return "bad signature"
    return "valid but empty"


def _log_session_loss() -> None:
    """Sampled INFO line that makes "I have to log in again" a countable reason in the web logs."""
    if not _LOG_SESSION_LOSS or request.path.startswith(_SESSION_LOSS_SKIP_PREFIXES):
        return
    if random.random() >= _SESSION_LOSS_SAMPLE_RATE:  # noqa: S311 - sampling, not security
        return
    logger.info(
        "session loss: reason=%s had_cookie=%s client=%s path=%s",
        _session_loss_reason(),
        "session" in request.cookies,
        _client_bucket(request.headers.get("User-Agent", "")),
        request.path,
    )


@app.after_request
def _keep_session_cookie_persistent(response: Response) -> Response:
    """Make every session cookie we send persistent.

    iOS home-screen apps reset session-scoped cookies (WebKit bug 272325), so even the
    pre-login cookie (OAuth state, CSRF token, flashes) must carry an expiry. Runs before
    Flask saves the session, so one check here covers every writer.
    """
    if session and not session.permanent:
        session.permanent = True
    return response


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


def _is_safe_redirect_target(target: str) -> bool:
    """Verify *target* is a relative URL with no scheme or host.

    Backslashes are stripped first to neutralise browser quirks where
    ``\\`` is treated like ``/`` (which could smuggle a netloc).  The
    URL is then parsed and accepted only when both ``scheme`` and
    ``netloc`` are empty — i.e. it is a plain relative path.
    """
    if not target:
        return False
    cleaned = target.replace("\\", "")
    parsed = urlparse(cleaned)
    return not parsed.scheme and not parsed.netloc


def _wikimedia_api_get(url: str, params: dict[str, str], timeout: int = 10) -> dict[str, Any]:
    """Make a GET request to a Wikimedia API endpoint and return parsed JSON.

    Sets the standard User-Agent header, calls ``raise_for_status()``, and
    parses the JSON response.  Callers handle their own exceptions.
    """
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# Commons downloads can legitimately resolve to these hosts: `Special:FilePath`
# with `?width=N` redirects to `thumb.wikimedia.org`, while the original file
# URL still lands on `upload.wikimedia.org`. Keep `thumb.wikimedia.org` or
# thumbnail downloads fail with "Redirect to untrusted host".
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "commons.wikimedia.org",
        "upload.wikimedia.org",
        "thumb.wikimedia.org",
    }
)


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Raises ValueError if the URL points to an untrusted host, uses a
    non-HTTPS scheme, or if the response exceeds max_bytes.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Blocked download from untrusted host: {parsed_url.hostname}")
    resp = None
    current_url = url
    max_redirects = 5
    try:
        for _hop in range(max_redirects + 1):
            resp = requests.get(
                current_url,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
                stream=True,
                allow_redirects=False,
            )
            if not resp.is_redirect:
                break
            # Validate redirect target before following
            resp.close()
            location = resp.headers.get("Location", "")
            parsed_redirect = urlparse(location)
            if parsed_redirect.scheme != "https" or parsed_redirect.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
                raise ValueError(f"Redirect to untrusted host: {parsed_redirect.hostname}")
            current_url = location
        else:
            raise ValueError(f"Too many redirects ({max_redirects}) from allowed download hosts")

        resp.raise_for_status()

        # Check Content-Length header first (fast reject)
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"Image too large: {int(content_length)} bytes (limit {max_bytes})")

        # Stream with enforced cap
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError(f"Image download exceeded {max_bytes} bytes limit")
            chunks.append(chunk)
    finally:
        if resp is not None:
            resp.close()

    return b"".join(chunks)


def _validate_bbox(bbox_top: int, bbox_right: int, bbox_bottom: int, bbox_left: int) -> str | None:
    """Validate bounding box coordinates.

    Returns an error message string if validation fails, or ``None`` if valid.
    Checks ordering (top < bottom, left < right), range (0..MAX_BBOX_PX),
    and minimum area (MIN_BBOX_AREA).
    """
    if bbox_top >= bbox_bottom or bbox_left >= bbox_right:
        return _("Invalid bounding box dimensions")

    if (
        bbox_top < 0
        or bbox_left < 0
        or bbox_bottom > MAX_BBOX_PX
        or bbox_right > MAX_BBOX_PX
        or (bbox_bottom - bbox_top) * (bbox_right - bbox_left) < MIN_BBOX_AREA
    ):
        return _("Bounding box out of allowed range")

    return None


def _read_box_request(*id_fields: str) -> tuple[tuple[int, ...], tuple[int, int, int, int]]:
    """Read the integer id fields and a drawn box (top, right, bottom, left) from the POST form.

    Raises ``ApiError`` (400) for a missing field, a non-integer value or an invalid box.
    """
    names = (*id_fields, "bbox_top", "bbox_right", "bbox_bottom", "bbox_left")
    raw = [request.form.get(name) for name in names]
    if not all(raw):
        raise ApiError(_("Missing required fields"))
    try:
        values = [int(value) for value in raw]  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise ApiError(_("Invalid field values")) from None
    ids, box = tuple(values[: len(id_fields)]), (values[-4], values[-3], values[-2], values[-1])
    bbox_error = _validate_bbox(*box)
    if bbox_error:
        raise ApiError(bbox_error)
    return ids, box


def _read_dismiss_ids() -> list[int]:
    """Face ids the classify page asks to dismiss as non-target (``dismiss_face_ids``, a JSON list).

    Anything that is not a well-formed list is ignored; a list over MAX_DISMISS_FACE_IDS is refused.
    """
    raw = request.form.get("dismiss_face_ids", "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        if len(parsed) > MAX_DISMISS_FACE_IDS:
            raise ApiError(_("Too many face IDs in dismiss list"))
        return [int(face_id) for face_id in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _commons_file_url(file_title: str) -> str:
    """Detection-width thumbnail URL for a ``File:`` title."""
    clean_title = file_title[5:] if file_title.startswith("File:") else file_title
    return FILE_PATH_URL.format(file_title=clean_title)


def _encode_drawn_box(image_row: dict[str, Any], box: tuple[int, int, int, int], *, purpose: str):
    """Download the image and embed a drawn box with the face service.

    *purpose* completes the log lines ("encoding manual face", "re-encoding face bbox").
    Raises ``ApiError``: 422 when the service cannot read the image or the box yields no
    usable embedding, 503 when the service is unavailable. A failed download raises
    ``requests.RequestException`` for the route to turn into a 502.
    """
    image_bytes = _download_image(_commons_file_url(image_row["file_title"]))
    try:
        encoded = face_client.encode_known_face(
            image_bytes,
            box,
            box_space=_detection_space(image_row.get("detection_width"), image_row.get("detection_height")),
        )
    except FaceServiceRejected:
        logger.warning("Face service rejected the image while %s", purpose)
        raise ApiError(_("This image could not be read by the face detection service."), 422) from None
    except FaceServiceError:
        logger.exception("Face service unavailable while %s", purpose)
        raise ApiError(_("Face detection service is unavailable. Please try again in a moment."), 503) from None
    if not encoded:
        raise ApiError(
            _("Could not compute face encoding for the selected region. Try drawing a slightly larger box."), 422
        )
    return encoded


def _already_reviewed_error(other_username: str) -> ApiError:
    """409 for a face another contributor classified; only they may change their own classification."""
    if other_username:
        message = _(
            "This face was already classified by %(user)s. Only they can change their own classification.",
            user=other_username,
        )
    else:
        message = _("This face has already been classified by another contributor.")
    return ApiError(message, 409, reason="already_reviewed")


def _insert_human_target_face(
    cursor: Any, image_id: int, encoded: Any, box: tuple[int, int, int, int], user_id: int
) -> int:
    """Insert a manually drawn face as a human-confirmed target and return its id."""
    cursor.execute(
        "INSERT INTO faces "
        "(image_id, encoding, embedding_model, aligned, bbox_top, bbox_right, bbox_bottom, bbox_left, "
        "is_target, classified_by, classified_by_user_id, classified_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, 'human', %s, NOW())",
        (image_id, encoded.encoding, encoded.model_id, 1 if encoded.aligned else 0, *box, user_id),
    )
    return cursor.lastrowid


def _get_face_stats(project_id: int) -> dict[str, Any]:
    """Return aggregate face statistics for a project.

    Columns returned: total_faces, confirmed_matches, confirmed_non_matches,
    unclassified, human_confirmed, model_matches, sdc_written, sdc_pending,
    sdc_removal_pending_faces, sdc_removal_pending, by_human, by_model, by_bootstrap.
    All values are coerced to ``int`` (NULL → 0).
    Returns an empty dict on database error.
    """
    try:
        rows = execute_query(
            "SELECT "
            "  COUNT(*) AS total_faces, "
            "  SUM(CASE WHEN f.is_target = 1 THEN 1 ELSE 0 END) AS confirmed_matches, "
            "  SUM(CASE WHEN f.is_target = 0 THEN 1 ELSE 0 END) AS confirmed_non_matches, "
            "  SUM(CASE WHEN f.is_target IS NULL THEN 1 ELSE 0 END) AS unclassified, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS human_confirmed, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.classified_by = 'model' AND f.classified_by_user_id IS NULL THEN 1 ELSE 0 END) AS model_matches, "
            "  SUM(CASE WHEN f.sdc_written = 1 AND sc.face_id IS NOT NULL THEN 1 ELSE 0 END) AS sdc_written, "
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
            "LEFT JOIN (SELECT DISTINCT face_id FROM sdc_claims WHERE written_at IS NOT NULL) sc ON sc.face_id = f.id "
            "WHERE i.project_id = %s AND f.superseded_by IS NULL",
            (project_id,),
        )
        if rows:
            return {k: int(v or 0) for k, v in rows[0].items()}
    except DatabaseError:
        logger.exception("Failed to load face stats for project %s", project_id)
    return {}


def _get_prefilter_stats(project_id: int) -> dict[str, Any]:
    """Return shadow-mode Metadata Pre-Filter results for a project.

    Columns: checked, kept, dropped, dropped_with_faces, kept_without_faces,
    human_targets_dropped, human_targets_total, plus ``skip_pct`` and the
    timestamp of the last stored verdict (``last_checked``). Counts are
    coerced to ``int`` (NULL → 0). Face-dependent counters only cover
    ``status='processed'`` images, mirroring ``scripts/eval_prefilter.py``.
    Returns an empty dict on database error.
    """
    try:
        rows = execute_query(
            "SELECT "
            "  COUNT(*) AS checked, "
            "  COALESCE(SUM(i.prefilter_keep = 1), 0) AS kept, "
            "  COALESCE(SUM(i.prefilter_keep = 0), 0) AS dropped, "
            "  COALESCE(SUM(i.prefilter_keep = 0 AND i.status = 'processed' "
            "    AND COALESCE(f.face_count, 0) > 0), 0) AS dropped_with_faces, "
            "  COALESCE(SUM(i.prefilter_keep = 1 AND i.status = 'processed' "
            "    AND COALESCE(f.face_count, 0) = 0), 0) AS kept_without_faces, "
            "  COALESCE(SUM(i.prefilter_keep = 0 AND COALESCE(f.human_targets, 0) > 0), 0) AS human_targets_dropped, "
            "  COALESCE(SUM(COALESCE(f.human_targets, 0) > 0), 0) AS human_targets_total, "
            "  MAX(i.prefilter_checked_at) AS last_checked "
            "FROM images i "
            "LEFT JOIN ("
            "  SELECT image_id, COUNT(*) AS face_count, "
            "    SUM(CASE WHEN is_target = 1 AND classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END) AS human_targets "
            "  FROM faces WHERE superseded_by IS NULL GROUP BY image_id"
            ") f ON f.image_id = i.id "
            "WHERE i.project_id = %s AND i.prefilter_checked_at IS NOT NULL",
            (project_id,),
        )
        if rows:
            row = dict(rows[0])
            last_checked = row.pop("last_checked", None)
            stats = {k: int(v or 0) for k, v in row.items()}
            stats["last_checked"] = last_checked
            stats["skip_pct"] = round(100.0 * stats["dropped"] / stats["checked"], 1) if stats["checked"] else 0.0
            return stats
    except DatabaseError:
        logger.exception("Failed to load pre-filter stats for project %s", project_id)
    return {}


def _prefilter_stats_for(project: dict[str, Any]) -> dict[str, Any]:
    """Load pre-filter stats for the settings page; empty when the mode is off."""
    if project.get("prefilter_mode", "off") == "off":
        return {}
    return _get_prefilter_stats(int(project["id"]))


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _service_unavailable():
    """503 for a request whose user could not be loaded (the session itself is still fine)."""
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"error": _("The service is temporarily unavailable. Please try again in a moment.")}), 503
    abort(503)


def login_required(f):
    """Decorator to require authentication."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            if getattr(g, "user_load_failed", False):
                return _service_unavailable()
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
    sess.headers["User-Agent"] = USER_AGENT
    return sess


def _oauth_exchange(refresh_token: str) -> dict[str, Any]:
    """Trade a refresh token for new tokens at the Wikimedia OAuth endpoint."""
    oauth = OAuth2Session(client_id=OAUTH_CLIENT_ID)
    return oauth.refresh_token(
        OAUTH_TOKEN_URL,
        refresh_token=refresh_token,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        timeout=10,  # Must complete well within the single-flight lock's TTL.
    )


def _note_refresh_failure(permanent: bool) -> None:
    """Remember on the current request whether a failed refresh is worth retrying."""
    if has_request_context():
        g.token_refresh_transient = not permanent


def _refresh_access_token(user: dict[str, Any]) -> dict[str, Any] | None:
    """Refresh the user's access token if it is expired or about to expire.

    The protocol (Redis single-flight lock, DB compare-and-swap, re-read after a lost
    race) lives in ``oauth_refresh``, shared with the worker. This only supplies the
    web side: the credentials already loaded for the request, the OAuth2 exchange, and
    carrying the result back onto the caller's ``user`` dict.

    Returns the updated user dict, or None if the refresh failed.
    """
    current = oauth_refresh.TokenSet(
        user["access_token"],
        user["refresh_token"],
        oauth_refresh.as_utc(user["token_expires_at"]),
    )
    try:
        tokens = oauth_refresh.refresh_tokens(
            user["id"],
            label=f"user {user['wiki_username']}",
            query=execute_query,
            get_redis=lambda: redis_client,
            exchange=_oauth_exchange,
            current=current,
            on_failure=_note_refresh_failure,
        )
    except DatabaseError:
        logger.exception("Database error while refreshing the token of %s", user["wiki_username"])
        _note_refresh_failure(False)
        return None
    if tokens is None:
        return None
    if tokens is not current:  # refreshed here, or taken over from a peer's refresh
        user["access_token"] = tokens.access_token
        user["refresh_token"] = tokens.refresh_token
        user["token_expires_at"] = tokens.expires_at
    return user


def _token_unavailable_error() -> ApiError:
    """The error for a missing token: log in again when the grant is gone, retry later otherwise."""
    if g.get("token_refresh_transient"):
        return ApiError(_("Could not reach the Wikimedia login service. Please try again in a moment."), 503)
    return ApiError(
        _("Your Wikimedia authorization has expired or was revoked. Please re-authorize to continue."),
        401,
        reauth_url=_reauth_url(),
    )


def _reauth_url() -> str:
    """The re-authorize target an API answer carries as ``reauth_url``.

    It only tells the page *whether* to offer the button; the link the page shows is the one
    the server renders for that page (``_reauth_login_url``, exposed as ``reauth_login_url``).
    """
    return url_for("login", reauth=1)


def _reauth_login_url() -> str:
    """The re-authorize link for the page being rendered: log in again, then come back here."""
    return url_for("login", reauth=1, next=request.full_path.rstrip("?"))


@app.context_processor
def inject_reauth_login_url() -> dict[str, str]:
    """Expose ``reauth_login_url`` to templates, so none repeats the ``next`` expression."""
    return {"reauth_login_url": _reauth_login_url() if has_request_context() else ""}


def _get_valid_token() -> str | None:
    """Get a valid access token for the current user, refreshing if needed.

    Returns None when that is not possible. The session is left alone: the browser login is
    still good, only the Commons write needs a fresh authorisation. ``g.token_refresh_transient``
    tells callers whether to ask the user to log in again (False) or to simply retry (True).
    """
    if g.user is None:
        return None

    refreshed = _refresh_access_token(g.user)
    if refreshed is None:
        return None

    g.user = refreshed
    return refreshed["access_token"]


def _csrf_token() -> str:
    """Generate or return the current CSRF token for forms."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
        session.permanent = True  # an anonymous session-scoped cookie is dropped by iOS home-screen apps
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


class _SessionInterface(SecureCookieSessionInterface):
    """Secure flag from ``_cookie_secure()``, so debug and the LAN dev mode are honoured per request."""

    def get_cookie_secure(self, app: Flask) -> bool:
        return _cookie_secure()


app.session_interface = _SessionInterface()


@app.before_request
def _generate_csp_nonce() -> None:
    """Generate a per-request CSP nonce and store it on g.

    Must run as a before_request so that g.csp_nonce is available to
    set_security_headers() (after_request) and to templates via the
    inject_csp_nonce context processor. The same value is emitted as
    'nonce-<value>' in the script-src directive and must be placed on every
    inline <script> via {{ csp_nonce }}; inline event-handler attributes are
    forbidden (script-src-attr 'none'), so page behaviour is wired with
    addEventListener from those nonce'd scripts.
    """
    g.csp_nonce = secrets.token_urlsafe(32)


@app.context_processor
def inject_csp_nonce() -> dict[str, str]:
    """Expose the per-request CSP nonce to all templates."""
    if not hasattr(g, "csp_nonce"):
        # Defensive fallback: should already be set by _generate_csp_nonce.
        g.csp_nonce = secrets.token_urlsafe(32)
    return {"csp_nonce": g.csp_nonce}


@app.context_processor
def inject_worker_status() -> dict[str, Any]:
    """Inject the ``worker_down`` flag into all templates.

    The banner is for a worker that is really gone, never for one that is working or being
    restarted on purpose, so it needs every "alive" signal to be absent (see ``_workers_alive``).
    A failing check means "not down": an unreadable database must not paint a scary banner.
    """
    try:
        worker_down = not _workers_alive()
    except Exception:
        app.logger.exception("Worker heartbeat check failed")
        worker_down = False
    return {"worker_down": worker_down}


def _worker_heartbeat_counts() -> dict[str, int] | None:
    """Count worker rows by state: ``fresh``, ``stopping`` (planned stop still within its grace),
    ``stale`` and ``total``. None when the query returns no row (nothing to judge by)."""
    rows = execute_query(
        "SELECT COALESCE(SUM(last_seen >= NOW() - INTERVAL %s SECOND), 0) AS fresh, "
        "COALESCE(SUM(last_seen < NOW() - INTERVAL %s SECOND AND expected_back_until > NOW()), 0) AS stopping, "
        "COUNT(*) AS total FROM worker_heartbeats",
        (WORKER_STALE_SECONDS, WORKER_STALE_SECONDS),
    )
    if not rows or not isinstance(rows, list):
        return None
    fresh, stopping, total = (int(rows[0][key]) for key in ("fresh", "stopping", "total"))
    return {"fresh": fresh, "stopping": stopping, "stale": total - fresh - stopping, "total": total}


def _recent_project_claim() -> bool:
    """True when a worker refreshed a project claim lately (it does so every minute while busy)."""
    rows = execute_query(
        "SELECT 1 AS ok FROM projects WHERE worker_claimed_at > NOW() - INTERVAL %s SECOND LIMIT 1",
        (WORKER_STALE_SECONDS,),
    )
    return bool(rows)


def _workers_alive() -> bool:
    """True unless every signal says the workers are gone.

    Alive means any of: a worker heartbeat newer than ``WORKER_STALE_SECONDS``; a worker that
    announced a planned stop (deploy, restart) whose grace period has not run out; a project
    claim refreshed within the same window, which still proves work is going on when a
    heartbeat write fails. Nothing recorded yet (a fresh install, or the minutes between a
    deploy's migration and the workers' first beat) is not evidence of an outage either.
    """
    counts = _worker_heartbeat_counts()
    if counts is None or counts["total"] == 0 or counts["fresh"] or counts["stopping"]:
        return True
    return _recent_project_claim()


def _workers_health_value() -> dict[str, int] | str:
    """Per-state worker counts for ``/health``; ``"unknown"`` when they cannot be read."""
    try:
        counts = _worker_heartbeat_counts()
    except Exception:
        logger.exception("Worker status for /health failed")
        return "unknown"
    if counts is None:
        return "unknown"
    return {key: counts[key] for key in ("fresh", "stopping", "stale")}


# A LiftWing status older than three probe intervals reads as unknown, so a
# stopped worker hides the banner instead of showing a stale one. With the
# probe disabled (interval 0) every status is stale, which is correct.
_LIFTWING_STALE_SECONDS = LIFTWING_PROBE_INTERVAL * 3


def _liftwing_status_snapshot() -> dict[str, Any]:
    """Last-known LiftWing state from ``service_status``; never makes a live call.

    Never raises: the banner and /health must not depend on the optional
    experiment being reachable, or on the table having been migrated yet.
    """
    unknown: dict[str, Any] = {"status": "unknown", "last_checked": None, "last_ok": None}
    try:
        rows = execute_query(
            "SELECT status, last_checked, last_ok, "
            "(last_checked IS NULL OR last_checked < NOW() - INTERVAL %s SECOND) AS is_stale "
            "FROM service_status WHERE name = 'liftwing'",
            (_LIFTWING_STALE_SECONDS,),
        )
        if not rows:
            return unknown
        row = rows[0]
        is_stale = bool(row.get("is_stale"))
        status = "unknown" if is_stale else str(row.get("status") or "unknown")
        if status not in ("up", "down"):
            status = "unknown"
        return {"status": status, "last_checked": row.get("last_checked"), "last_ok": row.get("last_ok")}
    except Exception:
        logger.debug("LiftWing status lookup failed", exc_info=True)
        return unknown


# Endpoints that can show the LiftWing banner or status line. Every other page
# skips the service_status lookup entirely.
_LIFTWING_STATUS_ENDPOINTS = ("project_detail", "project_settings")


@app.context_processor
def inject_liftwing_status() -> dict[str, Any]:
    """Expose the experimental LiftWing pre-filter status to templates.

    The banner and the settings status line only exist on project detail and
    settings pages, so other endpoints return defaults without a DB query.
    (Outside a request context — unit tests — the lookup still runs.)
    """
    if has_request_context() and request.endpoint not in _LIFTWING_STATUS_ENDPOINTS:
        return {"liftwing_down": False, "liftwing_status": "unknown", "liftwing_last_checked": None}
    snapshot = _liftwing_status_snapshot()
    return {
        "liftwing_down": snapshot["status"] == "down",
        "liftwing_status": snapshot["status"],
        "liftwing_last_checked": snapshot["last_checked"],
    }


@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if not app.debug:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # CSP notes:
    # - script-src allows only same-origin files and inline <script> tags carrying this
    #   request's nonce (g.csp_nonce, the same value templates render as {{ csp_nonce }}).
    #   There is no 'unsafe-inline' for scripts: a nonce makes browsers ignore it anyway.
    # - script-src-attr 'none' forbids inline event-handler attributes (onclick=...) and
    #   javascript: URLs outright; templates use addEventListener instead.
    # - style-src keeps 'unsafe-inline' because many style="" attributes remain.
    # - object-src 'none' blocks <object>/<embed>/<applet> plugin loads.
    # - base-uri 'none' prevents an injected <base> tag from rewriting all relative URLs
    #   on the page, which is a known XSS escalation vector.
    # - frame-ancestors 'none' makes X-Frame-Options redundant but both are kept for
    #   older browsers.
    nonce = getattr(g, "csp_nonce", None)
    if not nonce:
        # Responses built before _generate_csp_nonce ran (e.g. an early abort) render no
        # nonce'd script, so a fresh value is fine; keep it on g for any later reader.
        nonce = g.csp_nonce = secrets.token_urlsafe(32)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "script-src-attr 'none'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' https://*.wikimedia.org data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
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
        data = _wikimedia_api_get(
            WIKIDATA_API_URL,
            {"action": "wbgetclaims", "entity": qid, "property": "P31", "format": "json"},
        )

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
        data = _wikimedia_api_get(
            COMMONS_API_URL,
            {"action": "query", "titles": f"Category:{category}", "format": "json"},
        )
        pages = data.get("query", {}).get("pages", {})
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
        data = _wikimedia_api_get(
            COMMONS_API_URL,
            {"action": "query", "titles": f"Category:{category}", "prop": "categoryinfo", "format": "json"},
        )
        pages = data.get("query", {}).get("pages", {})
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
        data = _wikimedia_api_get(
            COMMONS_API_URL,
            {"action": "wbgetclaims", "entity": f"M{commons_page_id}", "property": "P180", "format": "json"},
        )
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
        data = _wikimedia_api_get(
            WIKIDATA_API_URL,
            {"action": "wbgetclaims", "entity": qid, "property": "P18", "format": "json"},
        )

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

        data = _wikimedia_api_get(
            WIKIDATA_API_URL,
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "labels",
                "languages": languages,
                "languagefallback": "1",
                "format": "json",
            },
        )

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
_THUMB_STEPS = COMMONS_THUMB_STEPS

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


# Widest thumbnail a gallery card may swap to. Keep equal to FACE_VIEW_MAX_SOURCE_WIDTH in
# static/wikivisage-common.js, whose faceViewSourceWidth() applies the same limit in the browser.
_GALLERY_HIRES_MAX = 960
_GALLERY_BASE_WIDTH = 330


def _gallery_hires_steps(detection_width: int) -> list[int]:
    """Standard thumbnail steps above the gallery's 330 px thumbnail that a card may swap to.

    Commons refuses thumbnails wider than the original and a detection width below 960 is the
    original's width, so nothing above the largest step <= detection_width is offered. Unknown
    detection widths (legacy rows) offer none.
    """
    if detection_width <= 0:
        return []
    limit = min(_GALLERY_HIRES_MAX, detection_width)
    return [step for step in _THUMB_STEPS if _GALLERY_BASE_WIDTH < step <= limit]


_MAX_INVITE_CODE_RETRIES = 5
_INVITE_CODE_TTL_DAYS = 7


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
    """Initiate OAuth 2.0 authorization flow.

    Uses PKCE (RFC 7636, S256) as defense-in-depth: even if the authorization
    code is intercepted (e.g. via a referrer leak or a malicious redirect_uri
    on a misconfigured consumer), the attacker cannot exchange it for tokens
    without also knowing the per-flow `code_verifier` we keep in the session.
    """
    if g.user and not request.args.get("reauth"):
        # ``?reauth=1`` is the one-click re-authorize: the browser login is still good but
        # Wikimedia no longer accepts the stored token, so run the OAuth round trip again.
        return redirect(url_for("dashboard"))
    if g.user and request.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
        # A state-changing GET must not be triggerable by a link on another site.
        return redirect(url_for("dashboard"))
    if g.get("user_load_failed"):
        # The session is fine, the database is not: a login round trip would not help.
        return _service_unavailable()

    # Generate PKCE pair. The verifier MUST be 43-128 chars from the unreserved
    # URL set; secrets.token_urlsafe(64) yields ~86 chars from [A-Za-z0-9_-].
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    )

    oauth = _make_oauth_session()
    authorization_url, state = oauth.authorization_url(
        OAUTH_AUTHORIZE_URL,
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )
    # Persistent from the first byte: iOS home-screen apps reset session-scoped cookies, which
    # would drop oauth_state while the user is away at the OAuth page.
    session.permanent = True
    session["oauth_state"] = state
    session["oauth_code_verifier"] = code_verifier
    session["login_next"] = request.args.get("next", "")
    return redirect(authorization_url)


@app.route("/auth/callback")
@limiter.limit("10 per minute")
def oauth_callback():
    """Handle OAuth 2.0 callback and create/update user record."""
    stored_state = session.pop("oauth_state", None)
    code_verifier = session.pop("oauth_code_verifier", None)
    if not stored_state:
        flash(_("Invalid OAuth state. Please try again."), "error")
        return redirect(url_for("index"))

    oauth = _make_oauth_session(state=stored_state)

    try:
        # Pass code_verifier so the provider can validate against the
        # code_challenge we sent at /login. Wikimedia's OAuth2 endpoint
        # accepts this as a standard PKCE parameter; consumers that haven't
        # opted into PKCE simply ignore the extra field.
        token_kwargs: dict[str, Any] = {
            "client_secret": OAUTH_CLIENT_SECRET,
            "authorization_response": request.url,
        }
        if code_verifier:
            token_kwargs["code_verifier"] = code_verifier
        token = oauth.fetch_token(OAUTH_TOKEN_URL, **token_kwargs)
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

    expires_at = datetime.now(UTC) + timedelta(seconds=token.get("expires_in", 14400))
    expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

    # Upsert user
    try:
        existing = execute_query("SELECT id, session_epoch FROM users WHERE wiki_user_id = %s", (wiki_user_id,))

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
            epoch = existing[0].get("session_epoch", 0)
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
            rows = execute_query("SELECT id, session_epoch FROM users WHERE wiki_user_id = %s", (wiki_user_id,))
            user_id = rows[0]["id"]
            epoch = rows[0].get("session_epoch", 0)

    except DatabaseError:
        logger.exception("Failed to upsert user")
        flash(_("Database error. Please try again."), "error")
        return redirect(url_for("index"))

    # Prevent session fixation: clear old session data and regenerate the CSRF
    # token on login. Preserving the pre-login CSRF token would let an attacker
    # who fixated a session before login retain a valid CSRF token afterwards.
    # The exception is a same-account re-authorize: that session was already
    # authenticated as this user, so keeping its work state (undo record, skips,
    # manual faces) and CSRF token is not fixation, and it keeps other open tabs working.
    login_next = session.pop("login_next", "")
    carried: dict[str, Any] = {}
    if session.get("user_id") == user_id:
        carried = {
            key: value
            for key, value in session.items()
            if key in ("csrf_token", "last_classify") or key.startswith(("skipped_images_", "manual_faces_"))
        }
    session.clear()
    session.update(carried)
    session.setdefault("csrf_token", secrets.token_hex(32))
    session.permanent = True
    session["user_id"] = user_id
    session["auth_at"] = int(time.time())
    session["epoch"] = epoch

    next_url = login_next
    if not _is_safe_url(next_url):
        next_url = url_for("dashboard")
    return redirect(next_url)


@app.route("/logout", methods=["POST"])
def logout():
    """Clear session and log out.

    Not ``login_required``: it only clears the cookie, so it must keep working while the
    database is down (``g.user`` could not be loaded then).
    """
    if "user_id" not in session:
        return redirect(url_for("index"))
    if not _validate_csrf():
        abort(403)
    # This device only: other sessions stay valid (see account_logout_everywhere).
    session.clear()
    session.permanent = True  # keep the flash below out of a session-scoped cookie
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

    Returns the image row (``id``, ``file_title``, ``status``, ``detection_width``,
    ``detection_height``) or ``None``.
    """
    rows = execute_query(
        "SELECT i.id, i.file_title, i.status, i.detection_width, i.detection_height FROM images i "
        "JOIN projects p ON i.project_id = p.id "
        "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
        "WHERE i.id = %s AND p.id = %s AND p.status != 'deleted' "
        "AND (p.user_id = %s OR pm.user_id IS NOT NULL)",
        (user_id, image_id, project_id, user_id),
    )
    return rows[0] if rows else None


class AccessRevoked(Exception):
    """Raised inside a transaction when a user's access to a project was
    revoked between the initial pre-transaction permission check and the
    actual mutation. Callers should translate this to HTTP 403."""


def _assert_project_access_locked(cursor: Any, project_id: int, user_id: int) -> None:
    """Re-verify project access INSIDE an open transaction with row locking.

    Closes a TOCTOU window: an owner could revoke a user's membership
    between ``get_project_for_actor`` / ``verify_image_access`` (which
    run on a separate connection before the transaction) and the actual
    INSERT/UPDATE. By taking ``FOR UPDATE`` locks on the project row and
    membership row, we guarantee the user's access is still valid at
    commit time, or the revoking transaction will block until we finish.

    Raises:
        AccessRevoked: if the user is neither the owner nor an active
            member of a non-deleted project at this instant.
    """
    cursor.execute(
        "SELECT 1 FROM projects p "
        "LEFT JOIN project_members pm "
        "  ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
        "WHERE p.id = %s AND p.status != 'deleted' "
        "  AND (p.user_id = %s OR pm.user_id IS NOT NULL) "
        "FOR UPDATE",
        (user_id, project_id, user_id),
    )
    if cursor.fetchone() is None:
        raise AccessRevoked(f"User {user_id} no longer has access to project {project_id}")


def _is_access_revoked(exc: BaseException) -> bool:
    """Return True if *exc* (or its cause chain) is an AccessRevoked error.

    ``execute_transaction`` wraps every exception in ``DatabaseError(...) from e``,
    so we walk ``__cause__`` to detect a TOCTOU access-revocation that fired
    inside the transaction.
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, AccessRevoked):
            return True
        seen.add(id(cur))
        cur = cur.__cause__
    return False


def _find_in_cause_chain(exc: BaseException, exc_type: type) -> BaseException | None:
    """Walk ``__cause__`` chain looking for an exception of *exc_type*.

    ``execute_transaction`` wraps every exception in ``DatabaseError(...) from e``,
    so callers that signal control flow via ``ValueError`` (or similar) inside a
    transaction need this to recover the original exception."""
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, exc_type):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
    return None


def _run_authorized_transaction(operations: Callable[[Any, Any], Any]) -> Any:
    """Run *operations* in a DB transaction, translating an in-transaction
    ``AccessRevoked`` into HTTP 403 and re-raising any in-transaction
    ``ValueError`` unwrapped so route handlers can catch it normally.
    Other ``DatabaseError`` exceptions propagate so callers can handle them
    as before."""
    try:
        return execute_transaction(operations)
    except DatabaseError as e:
        if _is_access_revoked(e):
            logger.warning("Access revoked mid-transaction; aborting with 403")
            abort(403)
        original_value_error = _find_in_cause_chain(e, ValueError)
        if original_value_error is not None:
            raise original_value_error from e
        raise


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
    if g.get("user_load_failed"):
        return _service_unavailable()
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


def _fetch_subcategories(category_title: str) -> tuple[list[str], bool]:
    """A category's direct subcategories, and whether Commons truncated the list (more pages exist)."""
    data = _wikimedia_api_get(
        COMMONS_API_URL,
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "subcat",
            "cmlimit": COMMONS_API_LIMIT,
            "format": "json",
        },
        timeout=3,
    )
    titles = [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
    return titles, "continue" in data


def _category_info_pages(titles: list[str]) -> dict[str, Any]:
    """The ``categoryinfo`` (file and subcategory counts) of up to 50 categories."""
    return (
        _wikimedia_api_get(
            COMMONS_API_URL,
            {"action": "query", "titles": "|".join(titles), "prop": "categoryinfo", "format": "json"},
            timeout=3,
        )
        .get("query", {})
        .get("pages", {})
    )


def _count_category_tree(root_title: str, deadline: float) -> dict[str, Any] | None:
    """Breadth-first count of the files under a Commons category and its subcategories.

    Bounded by MAX_CATEGORY_TRAVERSAL categories and the *deadline* (``time.monotonic()``), so the
    result may be approximate. Returns None when the root category does not exist.
    """
    pages = _category_info_pages([root_title])
    if "-1" in pages:
        return None

    root_info = next(iter(pages.values())).get("categoryinfo", {})
    total_files = root_info.get("files", 0)
    total_subcats = root_info.get("subcats", 0)

    visited = {root_title}
    queue: list[str] = []
    approximate = False

    if total_subcats > 0:
        titles, truncated = _fetch_subcategories(root_title)
        queue.extend(title for title in titles if title not in visited)
        approximate = truncated

    while queue and len(visited) < MAX_CATEGORY_TRAVERSAL and time.monotonic() < deadline:
        batch: list[str] = []
        while queue and len(batch) < 50 and len(visited) + len(batch) < MAX_CATEGORY_TRAVERSAL:
            title = queue.pop(0)
            if title not in visited:
                batch.append(title)
        if not batch:
            break
        visited.update(batch)

        subcats_to_fetch = []
        for info_page in _category_info_pages(batch).values():
            counts = info_page.get("categoryinfo", {})
            total_files += counts.get("files", 0)
            total_subcats += counts.get("subcats", 0)
            if counts.get("subcats", 0) > 0:
                subcats_to_fetch.append(info_page["title"])

        for sub_title in subcats_to_fetch:
            if time.monotonic() >= deadline or len(visited) >= MAX_CATEGORY_TRAVERSAL:
                break
            titles, truncated = _fetch_subcategories(sub_title)
            queue.extend(title for title in titles if title not in visited)
            approximate = approximate or truncated

    # A strict bool: the page reads it as `data.approximate ? ... : ...`, where an empty
    # list (what `queue and ...` used to leave behind) is truthy.
    approximate = approximate or len(visited) >= MAX_CATEGORY_TRAVERSAL or bool(queue and time.monotonic() >= deadline)
    return {
        "files": total_files,
        "subcats": total_subcats,
        "categories_visited": len(visited),
        "approximate": approximate,
    }


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

    try:
        counts = _count_category_tree(f"Category:{category}", time.monotonic() + CATEGORY_API_TIMEOUT)
    except Exception:
        logger.debug(f"Failed to fetch category info: {category}", exc_info=True)
        return jsonify({"error": "api_error"}), 502
    if counts is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(counts)


@dataclass
class _NewProjectForm:
    """The "new project" form after parsing and validation. ``errors`` is empty when it is usable."""

    wikidata_qid: str
    commons_category: str
    label: str
    distance_threshold: float
    min_confirmed: int
    errors: list[str]

    def template_context(self) -> dict[str, Any]:
        return {
            "wikidata_qid": self.wikidata_qid,
            "commons_category": self.commons_category,
            "label": self.label,
            "distance_threshold": self.distance_threshold,
            "min_confirmed": self.min_confirmed,
        }

    def render(self, **extra: Any):
        return render_template("project_new.html", **self.template_context(), **extra)


def _read_new_project_form() -> _NewProjectForm:
    """Parse the form and validate it, including the Wikidata and Commons lookups."""
    wikidata_qid = request.form.get("wikidata_qid", "").strip().upper()
    commons_category = request.form.get("commons_category", "").strip()
    label = request.form.get("label", "").strip()
    raw_threshold = request.form.get("distance_threshold", str(DEFAULT_DISTANCE_THRESHOLD))
    raw_min_confirmed = request.form.get("min_confirmed", "5")

    errors: list[str] = []
    if not wikidata_qid or not wikidata_qid.startswith("Q"):
        errors.append(_("Wikidata Q-ID must start with 'Q' (e.g., Q42)."))
    if not commons_category:
        errors.append(_("Commons category is required."))

    distance_threshold: float
    try:
        distance_threshold = float(raw_threshold)
        if not 0.1 <= distance_threshold <= 1.0:
            errors.append(_("Distance threshold must be between 0.1 and 1.0."))
    except ValueError:
        errors.append(_("Distance threshold must be a number."))
        distance_threshold = DEFAULT_DISTANCE_THRESHOLD

    min_confirmed: int
    try:
        min_confirmed = int(raw_min_confirmed)
        if min_confirmed < 1:
            errors.append(_("Minimum confirmed must be at least 1."))
    except ValueError:
        errors.append(_("Minimum confirmed must be a whole number."))
        min_confirmed = 5

    # The network checks only run once the cheap ones passed.
    if not errors and wikidata_qid and not _is_human_entity(wikidata_qid):
        errors.append(
            _(
                "The Wikidata entity %(qid)s is not an instance of human (Q5). Only human entities are supported.",
                qid=wikidata_qid,
            )
        )
    if not errors and commons_category:
        if not _commons_category_exists(commons_category):
            errors.append(_('The Commons category "%(category)s" does not exist.', category=commons_category))
        elif not _commons_category_has_files(commons_category):
            errors.append(
                _('The Commons category "%(category)s" exists but contains no files.', category=commons_category)
            )

    return _NewProjectForm(wikidata_qid, commons_category, label, distance_threshold, min_confirmed, errors)


def _user_already_has_project(form: _NewProjectForm) -> bool:
    """True if this user has a live project with the same Q-ID and category. A failed lookup reads as False."""
    try:
        return bool(
            execute_query(
                "SELECT id FROM projects WHERE user_id = %s AND wikidata_qid = %s AND commons_category = %s "
                "AND status != 'deleted'",
                (g.user["id"], form.wikidata_qid, form.commons_category),
            )
        )
    except DatabaseError:
        logger.exception("Failed to check for duplicate project")
        return False


def _respond_to_other_users_project(form: _NewProjectForm):
    """If another user already has this Q-ID + category, redirect a member to it or offer to join.

    Returns a response, or None to go on creating the project: no such project, the user was
    banned from it, or they chose "create anyway". A failed lookup also reads as None.
    """
    action = request.form.get("action", "create")
    try:
        other_project = execute_query(
            "SELECT p.id, p.label, u.wiki_username FROM projects p "
            "JOIN users u ON p.user_id = u.id "
            "WHERE p.wikidata_qid = %s AND p.commons_category = %s "
            "AND p.status != 'deleted' AND p.user_id != %s LIMIT 1",
            (form.wikidata_qid, form.commons_category, g.user["id"]),
        )
        if not other_project:
            return None
        other = other_project[0]
        membership = execute_query(
            "SELECT status FROM project_members WHERE project_id = %s AND user_id = %s",
            (other["id"], g.user["id"]),
        )
        is_banned = bool(membership) and membership[0]["status"] == "banned"
        if membership and not is_banned:
            flash(_("You are already a member of this project."), "info")
            return redirect(url_for("project_detail", project_id=other["id"]))
        if not is_banned and action != "create_anyway":
            flash(
                _(
                    "%(username)s already has a project for this Q-ID and category. "
                    "You can ask them for an invite code to join, or create your own.",
                    username=other["wiki_username"],
                ),
                "info",
            )
            return form.render(existing_project=other)
    except DatabaseError:
        logger.exception("Failed to check for joinable projects")
    return None


def _insert_project(form: _NewProjectForm, p18_thumb_url: str | None) -> None:
    """Insert the project with a fresh invite code, retrying on a code collision. Raises ``DatabaseError``."""
    invite_code = _generate_invite_code()
    for attempt in range(_MAX_INVITE_CODE_RETRIES):
        try:
            execute_query(
                "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
                "distance_threshold, min_confirmed, p18_thumb_url, invite_code, invite_code_created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                (
                    g.user["id"],
                    form.wikidata_qid,
                    form.commons_category,
                    form.label,
                    form.distance_threshold,
                    form.min_confirmed,
                    p18_thumb_url,
                    invite_code,
                ),
                fetch=False,
            )
            return
        except DatabaseError as exc:
            if _is_invite_code_collision(exc) and attempt < _MAX_INVITE_CODE_RETRIES - 1:
                logger.debug("invite_code collision on attempt %d, retrying", attempt + 1)
                invite_code = _generate_invite_code()
            else:
                raise


def _create_project(form: _NewProjectForm):
    """Create the project and wake the worker; on failure flash why and show the form again."""
    try:
        p18_thumb_url = _fetch_p18_thumb_url(form.wikidata_qid)
        if not form.label:
            form.label = _fetch_wikidata_label(form.wikidata_qid) or ""

        _insert_project(form, p18_thumb_url)
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
            # The retry loop exhausted all attempts to find a unique invite code and
            # re-raised the last collision — treat it as a generic creation failure.
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
        return form.render()


@app.route("/project/new", methods=["GET", "POST"])
@login_required
@limiter.limit("20 per hour")
def project_new():
    """Create a new project."""
    if request.method == "GET":
        return render_template("project_new.html")

    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    form = _read_new_project_form()
    if form.errors:
        for err in form.errors:
            flash(err, "error")
        return form.render()

    if _user_already_has_project(form):
        flash(_("A project with this Q-ID and category already exists."), "error")
        return form.render()

    response = _respond_to_other_users_project(form)
    if response is not None:
        return response

    return _create_project(form)


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
    face_stats = _get_face_stats(project_id)

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
            f"  AND {_EXCLUDE_NON_VISUAL_SQL}",
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
            "  AND f.classified_by_user_id IS NULL "
            "  AND f.sdc_removal_pending = 0",
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
        sdc_needs_reauth=bool(project.get("sdc_write_error")) and project["sdc_write_error"] in SDC_REAUTH_MESSAGES,
    )


_CLASSIFY_IMAGE_SELECT = (
    "SELECT DISTINCT i.id AS image_id, i.file_title, i.commons_page_id, i.detection_width, i.detection_height "
    "FROM images i JOIN faces f ON f.image_id = i.id WHERE "
)
_UNCLASSIFIED_FACE_CONDITIONS = "i.project_id = %s AND f.is_target IS NULL AND f.superseded_by IS NULL"
_MODEL_REJECTED_FACE_CONDITIONS = (
    "i.project_id = %s AND f.is_target = 0 AND f.classified_by = 'model' "
    "AND f.classified_by_user_id IS NULL AND f.superseded_by IS NULL"
)


def _record_skip_request(project_id: int) -> None:
    """Remember ``?skip_image_id=`` in the session so the image is not offered again, and make it undoable."""
    if not request.args.get("skip_image_id"):
        return
    is_skip_review = request.args.get("skip_reviewing") == "1"
    try:
        skip_id = int(request.args["skip_image_id"])
    except (ValueError, TypeError):
        return
    key = f"skipped_images_review_{project_id}" if is_skip_review else f"skipped_images_{project_id}"
    skipped = session.get(key, [])
    if skip_id not in skipped:
        skipped.append(skip_id)
    session[key] = skipped
    session["last_classify"] = {
        "action": "skip",
        "project_id": project_id,
        "image_id": skip_id,
        "was_review": is_skip_review,
        "face_ids": [],
        "manual_face_ids": [],
    }


def _random_candidate_image(conditions: str, params: tuple, excluded: list[int]) -> list[dict[str, Any]]:
    """One random image matching *conditions* (out of at most 200), skipping the *excluded* image ids."""
    sql = _CLASSIFY_IMAGE_SELECT + conditions
    if excluded:
        sql += f" AND i.id NOT IN ({','.join(['%s'] * len(excluded))})"
        params = (*params, *excluded)
    rows = execute_query(sql + " LIMIT 200", params)
    return [random.choice(rows)] if rows else rows


def _forced_candidate_image(project_id: int, image_id: int) -> tuple[list[dict[str, Any]], bool]:
    """Load one specific image (used after an undo). Returns (rows, reviewing_model).

    Reviewing mode applies when the image has no unclassified face left, only model rejections.
    """
    rows = execute_query(
        _CLASSIFY_IMAGE_SELECT + "i.project_id = %s AND i.id = %s "
        "AND f.superseded_by IS NULL "
        "AND (f.is_target IS NULL OR (f.is_target = 0 AND f.classified_by = 'model')) "
        "LIMIT 1",
        (project_id, image_id),
    )
    if not rows:
        return rows, False
    unclassified = execute_query(
        "SELECT COUNT(*) AS cnt FROM faces WHERE image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
        (image_id,),
    )
    return rows, not unclassified or unclassified[0]["cnt"] == 0


def _pick_image_to_classify(
    project_id: int, forced_image_id: int | None, skipped_ids: list[int], skipped_review_ids: list[int]
) -> tuple[dict[str, Any] | None, bool]:
    """Choose the next image: (image or None, reviewing_model).

    Priority: the forced image (after an undo), then an image with unclassified faces, then an
    image whose faces the model rejected and no human has reviewed. Skipped images are left out.
    """
    reviewing_model = False
    try:
        if forced_image_id:
            image_rows, reviewing_model = _forced_candidate_image(project_id, forced_image_id)
        else:
            image_rows = _random_candidate_image(_UNCLASSIFIED_FACE_CONDITIONS, (project_id,), skipped_ids)

        if not image_rows:
            reviewing_model = True
            image_rows = _random_candidate_image(_MODEL_REJECTED_FACE_CONDITIONS, (project_id,), skipped_review_ids)
    except DatabaseError:
        logger.exception("Failed to load image for classification")
        image_rows = []
    return (image_rows[0] if image_rows else None), reviewing_model


def _faces_to_classify(image_id: int, reviewing_model: bool) -> list[dict[str, Any]]:
    """The faces shown for an image: the model's rejections when reviewing, else the unclassified ones."""
    if reviewing_model:
        where, failure = (
            "f.is_target = 0 AND f.classified_by = 'model' AND f.superseded_by IS NULL",
            ("Failed to load faces for model review"),
        )
    else:
        where, failure = "f.is_target IS NULL AND f.superseded_by IS NULL", "Failed to load faces for classification"
    try:
        return execute_query(
            "SELECT f.id AS face_id, f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left, f.confidence "
            f"FROM faces f WHERE f.image_id = %s AND {where} ORDER BY f.bbox_left ASC",
            (image_id,),
        )
    except DatabaseError:
        logger.exception(failure)
        return []


def _count_images(conditions: str, project_id: int) -> int:
    """Number of distinct images of the project with a face matching *conditions* (0 if the lookup fails)."""
    try:
        rows = execute_query(
            f"SELECT COUNT(DISTINCT i.id) AS cnt FROM images i JOIN faces f ON f.image_id = i.id WHERE {conditions}",
            (project_id,),
        )
        return rows[0]["cnt"] if rows else 0
    except DatabaseError:
        return 0


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

    # Skipped images live in the session and are reset when the project changes.
    _record_skip_request(project_id)
    skip_key = f"skipped_images_{project_id}"
    skip_key_review = f"skipped_images_review_{project_id}"

    image, reviewing_model = _pick_image_to_classify(
        project_id,
        request.args.get("image_id", type=int),
        session.get(skip_key, []),
        session.get(skip_key_review, []),
    )
    faces_list = _faces_to_classify(image["image_id"], reviewing_model) if image else []

    return render_template(
        "classify.html",
        project=project,
        image=image,
        faces=faces_list,
        # Remaining images (not faces): unclassified ones, and those awaiting a review of the model's rejections.
        remaining=_count_images(_UNCLASSIFIED_FACE_CONDITIONS, project_id),
        model_review_count=_count_images(_MODEL_REJECTED_FACE_CONDITIONS, project_id),
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
    project = get_project_for_actor(project_id, g.user["id"])
    if not project:
        abort(404)
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
                _assert_project_access_locked(cursor, project_id, g.user["id"])
                cursor.execute(
                    f"SELECT id FROM faces WHERE image_id = %s AND {face_filter_sql}",
                    (image_id,),
                )
                rows = cursor.fetchall()
                ids = [r["id"] for r in rows] if rows else []

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET classified_by = 'human', "
                        "classified_by_user_id = %s, classified_at = NOW() "
                        "WHERE image_id = %s AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        "classified_by_user_id = %s, classified_at = NOW() "
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

            affected_ids = _run_authorized_transaction(_classify_none)

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
                _assert_project_access_locked(cursor, project_id, g.user["id"])
                cursor.execute(
                    f"SELECT id FROM faces WHERE image_id = %s AND {face_filter_sql}",
                    (image_id,),
                )
                rows = cursor.fetchall()
                ids = [r["id"] for r in rows] if rows else []

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET is_target = 1, classified_by = 'human', "
                        "classified_by_user_id = %s, sdc_removal_pending = 0, classified_at = NOW() "
                        "WHERE id = %s AND image_id = %s "
                        "AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], selected_face_id, image_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 1, classified_by = 'human', "
                        "classified_by_user_id = %s, sdc_removal_pending = 0, classified_at = NOW() "
                        "WHERE id = %s AND image_id = %s AND is_target IS NULL",
                        (g.user["id"], selected_face_id, image_id),
                    )
                target_updated = cursor.rowcount

                if is_review_mode:
                    cursor.execute(
                        "UPDATE faces SET classified_by = 'human', "
                        "classified_by_user_id = %s, classified_at = NOW() "
                        "WHERE image_id = %s AND id != %s "
                        "AND is_target = 0 AND classified_by = 'model' "
                        "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                        (g.user["id"], image_id, selected_face_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        "classified_by_user_id = %s, classified_at = NOW() "
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

            affected_ids = _run_authorized_transaction(_classify_target)

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


@dataclass
class _UndoClaimContext:
    """What an undo must know about Commons claims: read once, before anything changes."""

    face_states: dict[int, dict[str, Any]]
    sibling_keeps_claim: bool  # another target on the image keeps the P180 claim valid
    write_in_flight: bool  # a send-to-Commons run is active, so a write for a face may be in progress


def _undo_skip(last: dict[str, Any]):
    """Undo a skip: no database change, the image just leaves the session's skip list."""
    project_id, image_id = last["project_id"], last["image_id"]
    try:
        if not get_project_for_actor(project_id, g.user["id"]):
            return jsonify({"error": _("Project not found")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    active_skip_key = (
        f"skipped_images_review_{project_id}" if last.get("was_review", False) else f"skipped_images_{project_id}"
    )
    skipped = session.get(active_skip_key, [])
    if image_id in skipped:
        skipped.remove(image_id)
        session[active_skip_key] = skipped
    session.pop("last_classify", None)
    return jsonify({"status": "ok", "project_id": project_id, "image_id": image_id})


def _read_undo_claim_context(image_id: int, project_id: int, touched_ids: list[int]) -> _UndoClaimContext:
    """Read the claim-related state of the faces an undo touches. Raises ``DatabaseError``."""
    placeholders = ",".join(["%s"] * len(touched_ids))
    state_rows = execute_query(
        f"SELECT f.id, f.is_target, f.sdc_written, f.sdc_removal_pending, "
        f"  i.commons_page_id, i.bootstrapped, p.wikidata_qid "
        f"FROM faces f "
        f"JOIN images i ON f.image_id = i.id "
        f"JOIN projects p ON i.project_id = p.id "
        f"WHERE f.image_id = %s AND f.id IN ({placeholders})",
        (image_id, *touched_ids),
    )
    sibling_rows = execute_query(
        f"SELECT EXISTS("
        f"  SELECT 1 FROM faces f2 WHERE f2.image_id = %s "
        f"  AND f2.is_target = 1 AND f2.superseded_by IS NULL "
        f"  AND f2.id NOT IN ({placeholders})"
        f") AS has_sibling",
        (image_id, *touched_ids),
    )
    project_rows = execute_query("SELECT sdc_write_requested FROM projects WHERE id = %s", (project_id,))
    return _UndoClaimContext(
        face_states={int(r["id"]): r for r in (state_rows or ())},
        sibling_keeps_claim=bool(sibling_rows and sibling_rows[0]["has_sibling"]),
        write_in_flight=bool(project_rows and project_rows[0]["sdc_write_requested"]),
    )


def _remove_claims_before_undo(
    context: _UndoClaimContext, manual_face_ids: list[int], *, undoing_rejection: bool
) -> tuple[bool, bool]:
    """Take a P180 claim off Commons before its face is reverted. Returns (removed, removal_queued).

    A claim on record (sdc_written=1 or a queued removal) is removed now, or queued for the
    worker to verify when removal is not possible. Undoing a rejection is the opposite case and
    never removes a claim. Faces that get deleted (manual draws) cannot carry a queued removal,
    so for them the claim must be gone before the delete: the undo is refused (``ApiError``)
    rather than orphaning a claim.
    """
    claim_faces = [
        face_id
        for face_id, state in context.face_states.items()
        if not undoing_rejection and (state["sdc_written"] == 1 or state["sdc_removal_pending"] == 1)
    ]
    if not claim_faces or context.sibling_keeps_claim:
        return False, False

    claim = context.face_states[claim_faces[0]]
    access_token = _get_valid_token()
    removed = False
    if access_token and claim["commons_page_id"]:
        removed = _remove_sdc_claim(claim["commons_page_id"], claim["wikidata_qid"], access_token)
    if not removed and any(face_id in manual_face_ids for face_id in claim_faces):
        if not access_token:
            raise _token_unavailable_error()
        raise ApiError(_("Failed to remove SDC claim from Commons. The classification was not undone."), 502)
    return removed, not removed


def _undo_claim_flags(
    state: dict[str, Any] | None,
    *,
    undoing_rejection: bool,
    sibling_keeps_claim: bool,
    sdc_removed: bool,
    write_in_flight: bool,
) -> tuple[int, int]:
    """The (sdc_removal_pending, sdc_written) a reverted face ends up with.

    Undoing a target must not strand a P180 claim on Commons, and undoing a rejection must
    never delete one. The rules, in order:

    * a face that is unknown, or whose claim a sibling target keeps, simply reverts: (0, 0);
    * undoing a rejection cancels the removal the rejection queued. Only a rejection on a
      bootstrapped image ever sets that flag; on other images a queued flag predates the
      rejection (an earlier undo racing a write run), so it is kept for the worker to verify
      the possibly in-flight claim. The written marker is never touched;
    * a face with a claim on record is cleared if it was just removed, else it queues a
      removal and keeps its written marker;
    * a target without a claim on record queues a removal only while a write run is active,
      the one window in which a claim can be in flight;
    * anything else has nothing to clean up.
    """
    if state is None or sibling_keeps_claim:
        return 0, 0
    if undoing_rejection:
        queued = 0 if state.get("bootstrapped") else int(state["sdc_removal_pending"] or 0)
        return queued, int(state["sdc_written"] or 0)
    if state["sdc_written"] == 1 or state["sdc_removal_pending"] == 1:
        return (0, 0) if sdc_removed else (1, int(state["sdc_written"] or 0))
    if state["is_target"] == 1 and write_in_flight:
        return 1, 0
    return 0, 0


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

    if last.get("action") == "skip":
        return _undo_skip(last)

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

    undoing_rejection = last.get("action") == "none"
    touched_ids = list(dict.fromkeys([*face_ids, *manual_face_ids]))
    try:
        context = _read_undo_claim_context(image_id, project_id, touched_ids)
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    sdc_removed, sdc_removal_queued = _remove_claims_before_undo(
        context, manual_face_ids, undoing_rejection=undoing_rejection
    )

    try:
        was_review = last.get("was_review", False)
        decrement_counter = last.get("action") == "target" or (
            last.get("action") == "manual_face" and last.get("was_review")
        )

        def _undo(conn, cursor) -> bool:
            """Revert the faces; True when any of them now carries a queued removal."""
            _assert_project_access_locked(cursor, project_id, g.user["id"])
            if manual_face_ids:
                m_placeholders = ",".join(["%s"] * len(manual_face_ids))
                cursor.execute(
                    f"DELETE FROM faces WHERE id IN ({m_placeholders}) AND image_id = %s",
                    (*manual_face_ids, image_id),
                )

            queued_any = False
            for fid in [fid for fid in face_ids if fid not in manual_face_ids]:
                pending, written = _undo_claim_flags(
                    context.face_states.get(int(fid)),
                    undoing_rejection=undoing_rejection,
                    sibling_keeps_claim=context.sibling_keeps_claim,
                    sdc_removed=sdc_removed,
                    write_in_flight=context.write_in_flight,
                )
                queued_any = queued_any or bool(pending)

                if was_review:
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'model', "
                        "classified_by_user_id = NULL, sdc_written = %s, "
                        "sdc_removal_pending = %s, classified_at = NULL "
                        "WHERE id = %s AND image_id = %s",
                        (written, pending, fid, image_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE faces SET is_target = NULL, classified_by = NULL, "
                        "classified_by_user_id = NULL, sdc_written = %s, "
                        "sdc_removal_pending = %s, classified_at = NULL "
                        "WHERE id = %s AND image_id = %s",
                        (written, pending, fid, image_id),
                    )

            if decrement_counter:
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = GREATEST(faces_confirmed - 1, 0) WHERE id = %s",
                    (project_id,),
                )
            return queued_any

        sdc_removal_queued = _run_authorized_transaction(_undo) or sdc_removal_queued
        session.pop("last_classify", None)

    except DatabaseError:
        logger.exception("Failed to undo classification")
        return jsonify({"error": _("Failed to undo classification")}), 500

    # The classify page reloads right after an undo; surface the (otherwise
    # invisible) effect the undo had on Commons so the user is not surprised.
    if sdc_removed:
        flash(_("SDC claim was removed."), "info")
    elif sdc_removal_queued:
        flash(_("SDC claim removal queued."), "info")

    return jsonify(
        {
            "status": "ok",
            "project_id": project_id,
            "image_id": image_id,
            "sdc_removed": sdc_removed,
            "sdc_removal_queued": sdc_removal_queued,
        }
    )


def _detection_space(width: Any, height: Any) -> tuple[int, int] | None:
    """The stored ``images.detection_width/height`` as a ``box_space``, or None if unknown.

    Drawn boxes are in this pixel space (the UI scales them into it); passing it
    lets the face service map them onto whatever size we download.
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


@app.route("/api/manual-face", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def api_manual_face():
    """Accept a manually drawn bounding box, compute face encoding, and insert
    the face into the database so it can be classified."""
    if not _validate_csrf():
        return jsonify({"error": _("Invalid CSRF token")}), 400

    (project_id, image_id), box = _read_box_request("project_id", "image_id")

    # In review mode, user is drawing a face the model missed — auto-classify
    is_review_mode = request.form.get("reviewing_model") == "1"

    # In normal classify mode, dismiss currently displayed faces as non-target
    dismiss_face_ids = [] if is_review_mode else _read_dismiss_ids()

    try:
        img = verify_image_access(image_id, project_id, g.user["id"])
        if not img:
            return jsonify({"error": _("Image not found or access denied")}), 404
    except DatabaseError:
        return jsonify({"error": _("Database error")}), 500

    if img.get("status") != "processed":
        return jsonify({"error": _("Image has not finished processing yet")}), 409

    try:
        encoded = _encode_drawn_box(img, box, purpose=f"encoding manual face (image {image_id})")
        if not encoded.aligned:
            logger.info("Manual face on image %s embedded without landmark alignment", image_id)

        def _insert_manual_face(conn, cursor):
            _assert_project_access_locked(cursor, project_id, g.user["id"])
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

                new_face_id = _insert_human_target_face(cursor, image_id, encoded, box, g.user["id"])

                cursor.execute(
                    "UPDATE faces SET classified_by = 'human', "
                    "classified_by_user_id = %s, classified_at = NOW() "
                    "WHERE image_id = %s AND is_target = 0 AND classified_by = 'model' "
                    "AND classified_by_user_id IS NULL AND superseded_by IS NULL",
                    (g.user["id"], image_id),
                )
                cursor.execute(
                    "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
                    (project_id,),
                )
            else:
                new_face_id = _insert_human_target_face(cursor, image_id, encoded, box, g.user["id"])

                # Dismiss currently displayed faces as non-target so the image
                # doesn't reappear in the classify queue endlessly.
                if dismiss_face_ids:
                    placeholders = ",".join(["%s"] * len(dismiss_face_ids))
                    cursor.execute(
                        "UPDATE faces SET is_target = 0, classified_by = 'human', "
                        f"classified_by_user_id = %s, classified_at = NOW() WHERE id IN ({placeholders}) "
                        "AND image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
                        (g.user["id"], *dismiss_face_ids, image_id),
                    )

            return new_face_id, review_confirmed_ids

        new_face_id, review_confirmed_ids = _run_authorized_transaction(_insert_manual_face)

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

    except ApiError:
        raise
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

    Returns True on success — including the "benign" cases where the claim or
    the entire entity already doesn't exist (someone else removed it, the file
    was deleted on Commons, etc.). The user's intent ("this claim should not
    exist") is satisfied either way, so surfacing an error to the UI just
    confuses operators.

    Returns False on errors we cannot interpret as "already in the desired
    state" (auth failures, network errors, maxlag, unexpected API responses)
    and when the page/QID deduplication row cannot be cleared afterwards.
    """
    mid = f"M{commons_page_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
    }
    # Wikibase error codes that mean "claim/entity is already gone" — caller's
    # desired end state is reached, so we report success.
    _ALREADY_GONE_CODES = {
        "no-such-entity",
        "no-such-claim",
        "no-such-statement",
        "notfound",
    }

    def _clear_dedup_row() -> bool:
        """Delete the page/QID deduplication row once the claim is off Commons.

        Leaving the row behind makes a later ``write_sdc_claims`` run trust its
        ``written_at`` and mark the face written without restoring the claim,
        so a removal only counts as complete once the row is gone. Returns
        False when the row may still exist; the callers then treat the removal
        as incomplete and queue a worker removal instead of reporting success.
        """
        try:
            execute_query(
                "DELETE FROM sdc_claims WHERE commons_page_id = %s AND wikidata_qid = %s",
                (commons_page_id, wikidata_qid),
                fetch=False,
            )
            return True
        except DatabaseError:
            logger.exception(f"Failed to clear SDC dedup row for {mid}/{wikidata_qid}")
            return False

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

        # Entity-level miss: the M-id has no SDC entity at all (file deleted /
        # never had structured data). Nothing to remove.
        if "error" in claim_data:
            err_code = claim_data["error"].get("code", "")
            if err_code in _ALREADY_GONE_CODES:
                logger.info(f"SDC entity {mid} already gone ({err_code}); treating removal as success")
                return _clear_dedup_row()
            logger.error(f"SDC wbgetclaims error for {mid}: {claim_data['error']}")
            return False

        claims = claim_data.get("claims", {}).get("P180", [])
        target_guid = None
        for claim in claims:
            snak = claim.get("mainsnak", {})
            if snak.get("datavalue", {}).get("value", {}).get("id") == wikidata_qid:
                target_guid = claim.get("id")
                break

        if not target_guid:
            # Claim doesn't exist on Commons — nothing to remove
            return _clear_dedup_row()

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
            err_code = result["error"].get("code", "")
            if err_code in _ALREADY_GONE_CODES:
                # Race: another actor removed the claim between our wbgetclaims
                # and wbremoveclaims. The end state is what we wanted.
                logger.info(
                    f"SDC claim {target_guid} for {mid}/{wikidata_qid} already removed by peer "
                    f"({err_code}); treating as success"
                )
                return _clear_dedup_row()
            logger.error(f"SDC removal error for {mid}/{wikidata_qid}: {result['error']}")
            return False

        return _clear_dedup_row()

    except Exception:
        logger.exception(f"Failed to remove SDC claim for {mid}/{wikidata_qid}")
        return False


def _read_reclassify_request() -> tuple[int, int]:
    """(face_id, is_target) from the POST form. Raises ``ApiError`` (400) when missing or invalid."""
    face_id_str = request.form.get("face_id")
    is_target_str = request.form.get("is_target")  # "1" or "0"
    if not face_id_str or is_target_str is None:
        raise ApiError(_("Missing required fields"))
    try:
        face_id = int(face_id_str)
        is_target = int(is_target_str)
        if is_target not in (0, 1):
            raise ValueError
    except (ValueError, TypeError):
        raise ApiError(_("Invalid field values")) from None
    return face_id, is_target


def _load_face_for_reclassify(face_id: int) -> dict[str, Any]:
    """The face with its image and project, if the user may act on it. Raises ``ApiError`` (404/500)."""
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
    except DatabaseError:
        raise ApiError(_("Database error"), 500) from None
    if not rows:
        raise ApiError(_("Face not found or access denied"), 404)
    return rows[0]


def _settle_claim_for_rejection(face_row: dict[str, Any], face_id: int) -> tuple[bool, bool]:
    """Decide what happens to the P180 claim when a face is rejected: (removed_now, removal_queued).

    A claim must not survive a rejection unless another face on the same image still confirms
    the target (the sibling guard). A claim on record for a non-bootstrap face is removed
    immediately with the caller's token. Every other case is queued for the worker: bootstrap
    faces, and faces whose claim may be written by a write run still in flight (sdc_written=0).
    The worker re-verifies each face right before writing; without the queue, a rejection racing
    an in-flight write left the claim on Commons. Raises ``ApiError`` (401/502/500).
    """
    try:
        has_sibling = has_sibling_match(face_row["image_id"], face_id)
    except DatabaseError:
        raise ApiError(_("Database error"), 500) from None
    if has_sibling:
        return False, False

    if face_row["bootstrapped"] or face_row["sdc_written"] != 1:
        return False, True

    access_token = _get_valid_token()
    if not access_token:
        raise _token_unavailable_error()
    if not _remove_sdc_claim(face_row["commons_page_id"], face_row["wikidata_qid"], access_token):
        raise ApiError(_("Failed to remove SDC claim from Commons. The face was not reclassified."), 502)
    return True, False


def _require_same_reviewer(cursor: Any, face_id: int, user_id: int) -> None:
    """After an UPDATE that touched no row: a repeat click by the same user is fine, anyone else is not.

    Raises ``ValueError("already_reviewed:<username>")``, which ``_run_authorized_transaction``
    re-raises unwrapped so the route can build the 409.
    """
    cursor.execute(
        "SELECT f.classified_by_user_id, u.wiki_username "
        "FROM faces f LEFT JOIN users u ON u.id = f.classified_by_user_id "
        "WHERE f.id = %s",
        (face_id,),
    )
    check = cursor.fetchone()
    if check and check["classified_by_user_id"] == user_id:
        return  # same user, same value: a no-op success
    raise ValueError(f"already_reviewed:{(check or {}).get('wiki_username') or ''}")


def _demote_sibling_targets(
    cursor: Any, face_row: dict[str, Any], face_id: int, user_id: int
) -> tuple[list[int], dict[int, int]]:
    """A newly approved face replaces the image's other targets. Returns (demoted ids, their sdc_written)."""
    image_id = face_row["image_id"]
    cursor.execute(
        "UPDATE faces SET sdc_removal_pending = 0 "
        "WHERE image_id = %s AND sdc_removal_pending = 1 "
        "AND superseded_by IS NULL AND id != %s",
        (image_id, face_id),
    )
    # Lock sibling target faces to prevent concurrent reclassify races
    cursor.execute(
        "SELECT id, sdc_written FROM faces "
        "WHERE image_id = %s AND id != %s AND is_target = 1 "
        "AND superseded_by IS NULL FOR UPDATE",
        (image_id, face_id),
    )
    sibling_rows = cursor.fetchall()
    if not sibling_rows:
        return [], {}

    cursor.execute(
        "UPDATE faces SET is_target = 0, classified_by = 'human', "
        "classified_by_user_id = %s, classified_at = NOW(), "
        "sdc_removal_pending = CASE WHEN sdc_written = 1 THEN 1 ELSE sdc_removal_pending END, "
        "sdc_written = 0 "
        "WHERE image_id = %s AND id != %s AND is_target = 1 "
        "AND superseded_by IS NULL",
        (user_id, image_id, face_id),
    )
    demoted_count = cursor.rowcount
    if demoted_count > 0:
        cursor.execute(
            "UPDATE projects SET faces_confirmed = GREATEST(0, CAST(faces_confirmed AS SIGNED) - %s) WHERE id = %s",
            (demoted_count, face_row["project_id"]),
        )
    return [row["id"] for row in sibling_rows], {row["id"]: row["sdc_written"] for row in sibling_rows}


def _apply_reclassification(
    cursor: Any,
    face_row: dict[str, Any],
    face_id: int,
    is_target: int,
    *,
    sdc_written: int,
    removal_queued: bool,
    user_id: int,
) -> tuple[list[int], dict[int, int]]:
    """The transaction body of a reclassification. Returns (demoted sibling ids, their sdc_written)."""
    cursor.execute(
        "UPDATE faces SET is_target = %s, classified_by = 'human', "
        "classified_by_user_id = %s, sdc_written = %s, "
        "sdc_removal_pending = %s, classified_at = NOW() "
        "WHERE id = %s AND (classified_by_user_id IS NULL OR classified_by_user_id = %s)",
        (is_target, user_id, 0 if is_target == 0 else sdc_written, 1 if removal_queued else 0, face_id, user_id),
    )
    if cursor.rowcount == 0:
        # Either another user owns this face, or the same user repeated the action (no column changed).
        _require_same_reviewer(cursor, face_id, user_id)

    old_is_target = face_row["old_is_target"]
    if is_target == 1 and old_is_target != 1:
        cursor.execute(
            "UPDATE projects SET faces_confirmed = faces_confirmed + 1 WHERE id = %s",
            (face_row["project_id"],),
        )
        return _demote_sibling_targets(cursor, face_row, face_id, user_id)
    if is_target == 0 and old_is_target == 1:
        cursor.execute(
            "UPDATE projects SET faces_confirmed = GREATEST(0, CAST(faces_confirmed AS SIGNED) - 1) WHERE id = %s",
            (face_row["project_id"],),
        )
    return [], {}


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

    face_id, is_target = _read_reclassify_request()
    face_row = _load_face_for_reclassify(face_id)
    sdc_written = face_row["sdc_written"]
    commons_page_id = face_row["commons_page_id"]

    sdc_removed, sdc_removal_queued = (False, False)
    if is_target == 0:
        sdc_removed, sdc_removal_queued = _settle_claim_for_rejection(face_row, face_id)

    try:

        def _reclassify(conn, cursor):
            _assert_project_access_locked(cursor, face_row["project_id"], g.user["id"])
            return _apply_reclassification(
                cursor,
                face_row,
                face_id,
                is_target,
                sdc_written=sdc_written,
                removal_queued=sdc_removal_queued,
                user_id=g.user["id"],
            )

        demoted_ids, demoted_sdc = _run_authorized_transaction(_reclassify)

        # When approving, check if P180 already exists on Commons
        if is_target == 1 and commons_page_id and not sdc_written:
            if _maybe_mark_sdc_written(face_id, face_row["image_id"], face_row["project_id"]):
                sdc_written = 1

    except ValueError as e:
        msg = str(e)
        if msg.startswith("already_reviewed"):
            raise _already_reviewed_error(msg.split(":", 1)[1] if ":" in msg else "") from None
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
            # Post-action state, so the gallery card can refresh its SDC badge
            # without a reload. Reject always clears the marker; approve may set
            # it when the P180 check above found the claim already on Commons.
            "sdc_written": 0 if is_target == 0 else (1 if sdc_written else 0),
            "sdc_removed": sdc_removed,
            "sdc_removal_queued": sdc_removal_queued,
            "updated_faces": [
                {"face_id": fid, "is_target": 0, "sdc_removal_queued": bool(demoted_sdc.get(fid))}
                for fid in demoted_ids
            ],
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

    (face_id,), box = _read_box_request("face_id")

    # Verify ownership: face → image → project → user (or member).
    # Also fetch previous classifier's wiki_username so we can build a friendly
    # cross-user 409 message without a second query.
    try:
        rows = execute_query(
            "SELECT f.id, f.image_id, f.is_target, f.classified_by, f.confidence, "
            "  f.classified_by_user_id, f.sdc_written, i.file_title, i.commons_page_id, "
            "  i.status AS image_status, i.detection_width, i.detection_height, "
            "  p.id AS project_id, p.wikidata_qid, "
            "  u.wiki_username AS prev_classifier_username "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "LEFT JOIN project_members pm ON pm.project_id = p.id AND pm.user_id = %s AND pm.status = 'active' "
            "LEFT JOIN users u ON u.id = f.classified_by_user_id "
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

    # Pre-check cross-user guard BEFORE downloading the image and running the
    # face-service encoding. Mirrors the in-transaction guard in reclassify: members can
    # only edit faces they classified themselves (or NULL/model/bootstrap).
    # An in-transaction re-check below closes the TOCTOU window.
    orig_classified_by_user_id = face_row.get("classified_by_user_id")
    if orig_classified_by_user_id is not None and orig_classified_by_user_id != g.user["id"]:
        raise _already_reviewed_error(face_row.get("prev_classifier_username") or "")

    image_id = face_row["image_id"]
    orig_is_target = face_row["is_target"]
    orig_confidence = face_row["confidence"]
    orig_sdc_written = face_row["sdc_written"]

    try:
        encoded = _encode_drawn_box(face_row, box, purpose=f"re-encoding face bbox (face {face_id})")
        if not encoded.aligned:
            logger.info("Edited face %s embedded without landmark alignment", face_id)

        def _update_bbox(conn, cursor):
            _assert_project_access_locked(cursor, face_row["project_id"], g.user["id"])

            # In-transaction re-check of the cross-user guard. Closes the
            # TOCTOU window between the pre-check above and the INSERT below:
            # another user could have classified this face in between.
            cursor.execute(
                "SELECT f.classified_by_user_id, u.wiki_username "
                "FROM faces f LEFT JOIN users u ON u.id = f.classified_by_user_id "
                "WHERE f.id = %s AND f.superseded_by IS NULL",
                (face_id,),
            )
            check = cursor.fetchone()
            if not check:
                # Face was superseded or deleted while we were downloading
                raise ValueError("face_gone")
            current_owner = check["classified_by_user_id"]
            if current_owner is not None and current_owner != g.user["id"]:
                other_user = check.get("wiki_username") or ""
                raise ValueError(f"already_reviewed:{other_user}")

            cursor.execute(
                "INSERT INTO faces "
                "(image_id, encoding, embedding_model, aligned, bbox_top, bbox_right, bbox_bottom, bbox_left, "
                " is_target, classified_by, confidence, classified_by_user_id, sdc_written, classified_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'human', %s, %s, %s, NOW())",
                (
                    image_id,
                    encoded.encoding,
                    encoded.model_id,
                    1 if encoded.aligned else 0,
                    *box,
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

        new_face_id = _run_authorized_transaction(_update_bbox)

        if new_face_id and orig_is_target == 1 and not orig_sdc_written:
            _maybe_mark_sdc_written(new_face_id, image_id, face_row["project_id"])

        return jsonify(
            {
                "status": "ok",
                "new_face_id": new_face_id,
                "original_face_id": face_id,
            }
        )

    except ApiError:
        raise
    except requests.RequestException as e:
        logger.error(f"Failed to download image for bbox update: {e}")
        return jsonify({"error": _("Failed to download image from Commons")}), 502
    except ValueError as e:
        msg = str(e)
        if msg.startswith("already_reviewed"):
            raise _already_reviewed_error(msg.split(":", 1)[1] if ":" in msg else "") from None
        if msg == "face_gone":
            return jsonify({"error": _("Face not found or access denied")}), 404
        raise
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
        error = _token_unavailable_error()
        return jsonify({"error": error.message, **error.extra}), error.status

    # Set the flag for the worker to pick up
    try:
        execute_query(
            "UPDATE projects SET sdc_write_requested = 1, sdc_write_user_id = %s, sdc_write_error = NULL WHERE id = %s",
            (g.user["id"], project_id),
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
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 1 AND sc.face_id IS NOT NULL THEN 1 ELSE 0 END) AS written, "
            "  SUM(CASE WHEN f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0 THEN 1 ELSE 0 END) AS pending, "
            "  COUNT(DISTINCT CASE WHEN f.sdc_removal_pending = 1 "
            "    AND NOT EXISTS (SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
            "    AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id) "
            "    THEN f.image_id END) AS removal_pending "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "LEFT JOIN sdc_claims sc ON sc.face_id = f.id AND sc.written_at IS NOT NULL "
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
            "reauth_url": _reauth_url() if project["sdc_write_error"] in SDC_REAUTH_MESSAGES else None,
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
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = NULL WHERE id = %s",
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
        f"AND {_EXCLUDE_NON_VISUAL_SQL}"
    )
    params: list = [project_id]

    _RESULT_FILTERS: dict[str, list[str]] = {
        "match": ["f.is_target = 1"],
        "non-match": ["f.is_target != 1", "NOT (f.is_target = 0 AND f.classified_by_user_id IS NOT NULL)"],
        "rejected": ["f.is_target = 0", "f.classified_by_user_id IS NOT NULL"],
    }
    _SOURCE_FILTERS: dict[str, list[str]] = {
        "model": ["f.classified_by = 'model'", "f.classified_by_user_id IS NULL"],
        "bootstrap": ["f.classified_by = 'bootstrap'", "f.classified_by_user_id IS NULL"],
        "human": ["f.classified_by_user_id IS NOT NULL"],
    }

    result_source_clauses: list[str] = []
    result_source_clauses.extend(_RESULT_FILTERS.get(result_filter, []))
    result_source_clauses.extend(_SOURCE_FILTERS.get(source_filter, []))

    filter_clauses: list[str] = list(result_source_clauses)
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
                "                    AND f2.superseded_by IS NULL "
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
            if result_source_clauses:
                filtered_where = base_where + " AND " + " AND ".join(result_source_clauses)
                filtered_data = execute_query(
                    "SELECT "
                    "  SUM(CASE WHEN (f.is_target = 1 AND f.sdc_written = 0 AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0) "
                    "            OR (f.sdc_removal_pending = 1 AND NOT EXISTS ("
                    "                  SELECT 1 FROM faces f2 "
                    "                  WHERE f2.image_id = f.image_id "
                    "                    AND f2.is_target = 1 "
                    "                    AND f2.superseded_by IS NULL "
                    "                    AND f2.id != f.id"
                    "            )) THEN 1 ELSE 0 END) AS sdc_pending "
                    f"FROM faces f JOIN images i ON f.image_id = i.id WHERE {filtered_where}",  # noqa: S608
                    (project_id,),
                )
                if filtered_data:
                    counts["filtered_sdc_pending"] = filtered_data[0]["sdc_pending"] or 0
                else:
                    counts["filtered_sdc_pending"] = 0
            else:
                counts["filtered_sdc_pending"] = counts.get("sdc_pending", 0)
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
        is_sdc_send = (
            face["is_target"] == 1
            and face["sdc_written"] == 0
            and face["classified_by"] != "bootstrap"
            and face["bootstrapped"] == 0
        )
        is_sdc_removal = face["sdc_removal_pending"] == 1 and not face["has_confirmed_target_sibling"]
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
                "thumb_url": commons_thumb_url(face["file_title"], _GALLERY_BASE_WIDTH),
                # Direct Commons URLs for the one-time hi-res swap, so the browser never goes
                # through the rate-limited /commons-thumb proxy for it.
                "thumb_urls": {
                    str(step): commons_thumb_url(face["file_title"], step)
                    for step in _gallery_hires_steps(face["detection_width"] or 0)
                },
                "result": result_type,
                "sdc_pending": is_sdc_send or is_sdc_removal,
                "sdc_removal": is_sdc_removal,
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

    face_stats = _get_face_stats(project_id)

    inference_eligible = 0
    try:
        eligible_row = execute_query(
            "SELECT COUNT(*) AS cnt FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.is_target IS NULL "
            "  AND f.superseded_by IS NULL "
            "  AND f.classified_by_user_id IS NULL "
            "  AND f.sdc_removal_pending = 0",
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
        prefilter_stats = _prefilter_stats_for(project)
        return render_template(
            "project_settings.html", project=project, members=members, is_owner=True, prefilter_stats=prefilter_stats
        )

    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    distance_threshold = request.form.get("distance_threshold", str(project["distance_threshold"]))
    min_confirmed = request.form.get("min_confirmed", str(project["min_confirmed"]))
    status = request.form.get("status", project["status"])
    label = request.form.get("label", project["label"])
    prefilter_mode = request.form.get("prefilter_mode", project.get("prefilter_mode", "off"))

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

    # 'enforce' exists in the column for forward compatibility but is not built
    # yet, so only the two implemented modes can be selected.
    if prefilter_mode not in ("off", "shadow"):
        errors.append(_("Invalid metadata pre-filter mode."))
        prefilter_mode = project.get("prefilter_mode", "off")

    if errors:
        for err in errors:
            flash(err, "error")
        project["distance_threshold"] = distance_threshold
        project["min_confirmed"] = min_confirmed
        project["status"] = status
        project["label"] = label
        project["prefilter_mode"] = prefilter_mode
        return render_template(
            "project_settings.html",
            project=project,
            members=[],
            is_owner=True,
            prefilter_stats=_prefilter_stats_for(project),
        )

    try:
        completion_reason_clause = ", completion_reason = NULL" if status != "completed" else ""
        execute_query(
            "UPDATE projects SET distance_threshold = %s, min_confirmed = %s, "
            f"status = %s, label = %s, prefilter_mode = %s{completion_reason_clause} "
            "WHERE id = %s AND user_id = %s",
            (
                distance_threshold,
                min_confirmed,
                status,
                label,
                prefilter_mode,
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


@app.route("/project/<int:project_id>/leave", methods=["POST"])
@login_required
def project_leave(project_id: int):
    """Leave a project (member-only, not the owner)."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))

    try:
        project = get_project_for_actor(project_id, g.user["id"])
    except DatabaseError:
        abort(500)

    if not project:
        abort(404)

    if project["user_id"] == g.user["id"]:
        flash(_("You cannot leave a project you own. Delete it instead."), "error")
        return redirect(url_for("project_detail", project_id=project_id))

    try:
        affected = execute_query(
            "DELETE FROM project_members WHERE project_id = %s AND user_id = %s AND status = 'active'",
            (project_id, g.user["id"]),
            fetch=False,
        )
        if affected:
            flash(_("You have left the project."), "success")
        else:
            flash(_("You are not a member of this project."), "error")
    except DatabaseError:
        logger.exception("Failed to leave project %s", project_id)
        flash(_("Failed to leave the project."), "error")
        return redirect(url_for("project_detail", project_id=project_id))

    return redirect(url_for("dashboard"))


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
                    "UPDATE projects SET invite_code = %s, invite_code_created_at = NOW() "
                    "WHERE id = %s AND user_id = %s",
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
                "UPDATE projects SET invite_code = NULL, invite_code_created_at = NULL WHERE id = %s AND user_id = %s",
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
            "WHERE p.invite_code = %s AND p.status != 'deleted' "
            "AND (p.invite_code_created_at IS NULL "
            "     OR p.invite_code_created_at >= NOW() - INTERVAL %s DAY) "
            "LIMIT 1",
            (code, _INVITE_CODE_TTL_DAYS),
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

    # All gate checks and the reset itself run in one transaction under the
    # project-row lock, so a status change, a queued SDC write or a concurrent
    # undo cannot land between the pre-check and the UPDATE.
    try:

        def _reset_inference(conn, cursor):
            _assert_project_access_locked(cursor, project_id, g.user["id"])
            cursor.execute(
                "SELECT status, sdc_write_requested, min_confirmed FROM projects WHERE id = %s FOR UPDATE",
                (project_id,),
            )
            locked = cursor.fetchone()
            if not locked:
                return {"affected": 0, "blocked": "not_found"}

            # Inference is only claimed for active and completed projects.
            # Resetting a paused project would wipe its model results and leave
            # nothing to recompute them.
            if locked["status"] == "paused":
                return {"affected": 0, "blocked": "paused"}

            # A reset during a write run can land between the worker's live
            # re-check and its wbeditentity call, leaving a P180 claim on
            # Commons for a face that is no longer a target.
            if locked["sdc_write_requested"]:
                return {"affected": 0, "blocked": "sdc_in_progress"}

            cursor.execute(
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
            stats = cursor.fetchone() or {}
            human_confirmed = int(stats.get("human_confirmed") or 0)
            by_bootstrap = int(stats.get("by_bootstrap") or 0)
            min_confirmed = int(locked["min_confirmed"])

            # Below the gate the worker would not be allowed to reclassify the
            # faces, so wiping the existing model results would be destructive.
            if human_confirmed < min_confirmed:
                return {
                    "affected": 0,
                    "blocked": "below_gate",
                    "human_confirmed": human_confirmed,
                    "by_bootstrap": by_bootstrap,
                    "min_confirmed": min_confirmed,
                }

            cursor.execute(
                "UPDATE faces f "
                "JOIN images i ON f.image_id = i.id "
                "SET f.is_target = NULL, f.classified_by = NULL, f.confidence = NULL, "
                "f.classified_at = NULL "
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
            return {
                "affected": rows_reset,
                "blocked": None,
                "human_confirmed": human_confirmed,
                "by_bootstrap": by_bootstrap,
                "min_confirmed": min_confirmed,
            }

        result = _run_authorized_transaction(_reset_inference)
    except DatabaseError:
        logger.exception("Failed to reset model-classified faces")
        flash(_("Failed to re-run inference."), "error")
        return redirect(url_for("project_settings", project_id=project_id))

    blocked = result.get("blocked")
    if blocked == "not_found":
        abort(404)
    if blocked == "paused":
        flash(
            _("This project is paused. Set it to Active or Completed before re-running inference."),
            "warning",
        )
        return redirect(url_for("project_settings", project_id=project_id))
    if blocked == "sdc_in_progress":
        flash(
            _("A Commons edit run is in progress. Wait for it to finish (or stop it) before re-running inference."),
            "warning",
        )
        return redirect(url_for("project_settings", project_id=project_id))
    if blocked == "below_gate":
        flash(
            _(
                "Warning: You currently have %(confirmed)d/%(min)d human-confirmed target faces. "
                "Autonomous inference will not run until this threshold is met.",
                confirmed=result["human_confirmed"],
                min=result["min_confirmed"],
            ),
            "warning",
        )
        if result["by_bootstrap"] > 0:
            flash(
                _(
                    "Tip: You have %(count)d bootstrapped matches from existing Commons depicts claims. "
                    "Approve them in Model Results to count them as human-confirmed.",
                    count=result["by_bootstrap"],
                ),
                "info",
            )
        flash(_("No faces were reset."), "info")
        return redirect(url_for("project_settings", project_id=project_id))

    affected = result["affected"]
    if affected:
        flash(
            _(
                "Reset %(num)d model-classified faces. The worker will re-classify them on the next cycle.",
                num=affected,
            ),
            "success",
        )
    else:
        flash(_("No model-classified faces to reset."), "info")

    # Wake the worker even when no face needed a reset: unclassified faces
    # may still be waiting, and the user explicitly asked for a run.
    try:
        with open(WAKE_FILE_PATH, "w") as f:
            f.write("")
    except OSError:
        pass  # Non-critical — worker will pick it up on next poll

    flash(
        _(
            "You have %(confirmed)d/%(min)d human-confirmed target faces — "
            "inference will run on the next worker cycle.",
            confirmed=result["human_confirmed"],
            min=result["min_confirmed"],
        ),
        "info",
    )
    return redirect(url_for("project_detail", project_id=project_id, inference_triggered="1"))


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


@app.route("/account/logout-everywhere", methods=["POST"])
@limiter.limit("5 per hour")
@login_required
def account_logout_everywhere():
    """End every session of this user, on all browsers and devices, including this one."""
    if not _validate_csrf():
        abort(400, _("Invalid CSRF token"))
    try:
        execute_query(
            "UPDATE users SET session_epoch = session_epoch + 1 WHERE id = %s",
            (g.user["id"],),
            fetch=False,
        )
    except DatabaseError:
        logger.exception("Failed to bump session_epoch for user %s", g.user["id"])
        flash(_("Could not log you out on all devices. Please try again."), "error")
        return redirect(url_for("account_settings"))
    _end_session()
    flash(_("You have been logged out on all devices."), "info")
    return redirect(url_for("index"))


@app.route("/leaderboard")
def leaderboard():
    """Community leaderboard ranking users by classifications and SDC tags."""
    period = request.args.get("period", "all")
    if period not in ("all", "month", "daily"):
        period = "all"

    try:
        if period == "all":
            rows = execute_query(
                "SELECT "
                "  u.wiki_username, "
                "  COUNT(f.id) + COALESCE(us.classifications, 0) "
                "    AS classifications, "
                "  COALESCE(sdc.sdc_count, 0) "
                "    + COALESCE(us.sdc_tags, 0) AS sdc_tags "
                "FROM users u "
                "LEFT JOIN faces f ON f.classified_by_user_id = u.id "
                "  AND f.superseded_by IS NULL "
                "LEFT JOIN user_stats us ON us.user_id = u.id "
                "LEFT JOIN ("
                "  SELECT f2.classified_by_user_id AS user_id, COUNT(*) AS sdc_count "
                "  FROM sdc_claims sc "
                "  JOIN faces f2 ON f2.id = sc.face_id "
                "  WHERE sc.written_at IS NOT NULL "
                "    AND f2.classified_by_user_id IS NOT NULL "
                "  GROUP BY f2.classified_by_user_id"
                ") sdc ON sdc.user_id = u.id "
                "WHERE u.leaderboard_opt_out = 0 "
                "  AND (f.id IS NOT NULL OR us.user_id IS NOT NULL "
                "       OR sdc.user_id IS NOT NULL) "
                "GROUP BY u.id, u.wiki_username, us.classifications, "
                "         us.sdc_tags, sdc.sdc_count "
                "ORDER BY (COUNT(f.id) + COALESCE(us.classifications, 0) "
                "        + COALESCE(sdc.sdc_count, 0) "
                "        + COALESCE(us.sdc_tags, 0)) DESC, "
                "         (COUNT(f.id) + COALESCE(us.classifications, 0)) DESC "
                "LIMIT 100",
            )
        else:
            if period == "month":
                cutoff = datetime.now(UTC) - timedelta(days=30)
            else:
                cutoff = datetime.now(UTC) - timedelta(hours=24)
            rows = execute_query(
                "SELECT "
                "  u.wiki_username, "
                "  COUNT(f.id) AS classifications, "
                "  COALESCE(sdc.sdc_count, 0) AS sdc_tags "
                "FROM users u "
                "LEFT JOIN faces f ON f.classified_by_user_id = u.id "
                "  AND f.superseded_by IS NULL "
                "  AND f.classified_at >= %s "
                "LEFT JOIN ("
                "  SELECT f2.classified_by_user_id AS user_id, COUNT(*) AS sdc_count "
                "  FROM sdc_claims sc "
                "  JOIN faces f2 ON f2.id = sc.face_id "
                "  WHERE sc.written_at IS NOT NULL "
                "    AND sc.written_at >= %s "
                "    AND f2.classified_by_user_id IS NOT NULL "
                "  GROUP BY f2.classified_by_user_id"
                ") sdc ON sdc.user_id = u.id "
                "WHERE u.leaderboard_opt_out = 0 "
                "  AND (f.id IS NOT NULL OR sdc.user_id IS NOT NULL) "
                "GROUP BY u.id, u.wiki_username, sdc.sdc_count "
                "ORDER BY (COUNT(f.id) + COALESCE(sdc.sdc_count, 0)) DESC, "
                "         COUNT(f.id) DESC "
                "LIMIT 100",
                (cutoff, cutoff),
            )
    except DatabaseError:
        logger.exception("Failed to load leaderboard")
        rows = []
        flash(_("Failed to load leaderboard."), "error")

    totals = {"classifications": 0, "sdc_tags": 0}
    for row in rows:
        totals["classifications"] += row["classifications"]
        totals["sdc_tags"] += row["sdc_tags"]

    return render_template("leaderboard.html", rows=rows, totals=totals, period=period)


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


def _liftwing_health_value() -> str:
    """``reachable`` / ``unreachable`` / ``disabled`` for /health.

    Reads the worker's last probe from ``service_status`` (never a live call)
    and reports ``disabled`` when no project uses the pre-filter or the status
    is unknown. Deliberately does not affect ``degraded``: LiftWing is an
    optional, fail-open experiment, and only the face service degrades health.
    """
    try:
        in_use = execute_query(
            "SELECT COUNT(*) AS cnt FROM projects WHERE status != 'deleted' AND prefilter_mode != 'off'"
        )
        if not in_use or not in_use[0]["cnt"]:
            return "disabled"
    except Exception:
        return "disabled"
    status = _liftwing_status_snapshot()["status"]
    if status == "up":
        return "reachable"
    if status == "down":
        return "unreachable"
    return "disabled"


@app.route("/health")
@limiter.exempt
def health():
    """Health check endpoint for Toolforge monitoring.

    Reports degraded (200 OK with `degraded: true`) when the rate limiter fell
    back to per-process memory storage at startup or when the face service is
    unreachable. In both cases the database is still healthy, so we don't
    return 503, ops can scrape `degraded` and alert. The ``liftwing`` and ``workers``
    fields are informational and never set ``degraded``.
    """
    try:
        rows = execute_query("SELECT 1 AS ok")
        if rows and rows[0].get("ok") == 1:
            face_service_ok = face_client.is_healthy()
            payload = {
                "status": "healthy",
                "database": "connected",
                "limiter": "redis" if _LIMITER_REDIS_OK else "memory",
                "face_service": "reachable" if face_service_ok else "unreachable",
                "liftwing": _liftwing_health_value(),
                "workers": _workers_health_value(),
                "degraded": not _LIMITER_REDIS_OK or not face_service_ok,
            }
            return jsonify(payload), 200
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
        "Allow: /llms.txt",
        "Disallow: /",
        "",
        f"Sitemap: {xml_escape(request.url_root)}sitemap.xml",
    ]
    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/sitemap.xml")
@limiter.exempt
def sitemap_xml():
    """Serve a minimal XML sitemap listing publicly crawlable pages."""
    base = xml_escape(request.url_root.rstrip("/"))
    urls = ["/", "/leaderboard"]
    xml_urls = "".join(f"<url><loc>{base}{xml_escape(path)}</loc></url>" for path in urls)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{xml_urls}"
        "</urlset>"
    )
    return xml, 200, {"Content-Type": "application/xml; charset=utf-8"}


@app.route("/llms.txt")
@limiter.exempt
def llms_txt():
    """Serve llms.txt (https://llmstxt.org/): what this site is, for LLM crawlers."""
    base = request.url_root.rstrip("/")
    lines = [
        "# WikiVisage",
        "",
        "> Active-learning facial recognition for Wikimedia Commons. Train a model on a person,",
        "> review its matches, and add P180 (depicts) statements to the matching files' Structured",
        "> Data on Commons.",
        "",
        "WikiVisage is an open-source Flask tool for Wikimedia Commons contributors. A project",
        "pairs a Wikidata item (normally a person) with a Commons category. A background worker",
        "detects faces in the category's images, and the contributor confirms or rejects candidates",
        "in a yes/no interface. Once enough faces are confirmed, the tool classifies the remaining",
        "faces with a centroid-distance model. Approved matches can then be written to Commons as",
        "P180 (depicts) statements using the contributor's own Wikimedia OAuth authorization.",
        "Edits are only made when the contributor asks for them; nothing is written automatically.",
        "",
        "Most of the site requires signing in with Wikimedia OAuth, so only the pages below are",
        "publicly readable.",
        "",
        "## Pages",
        "",
        f"- [Home]({base}/): What the tool does and how to sign in.",
        f"- [Leaderboard]({base}/leaderboard): Top contributors by classifications and P180 tags.",
        f"- [Log in]({base}/login): Wikimedia OAuth 2.0 sign-in.",
        f"- [Health]({base}/health): JSON status of the database and background services.",
        "",
        "## Source and documentation",
        "",
        "- [Source code](https://github.com/DiFronzo/WikiVisage): The Flask app, background worker",
        "  and face service.",
        "- [README](https://github.com/DiFronzo/WikiVisage/blob/main/README.md): Setup, architecture",
        "  and screenshots.",
        "- [Contributing](https://github.com/DiFronzo/WikiVisage/blob/main/CONTRIBUTING.md):",
        "  Development conventions and tests.",
        "- [License](https://github.com/DiFronzo/WikiVisage/blob/main/LICENSE): MIT (the bundled",
        "  face-recognition weights have separate non-commercial terms).",
        "",
    ]
    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


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
    parsed = urlparse(thumb_url)
    if parsed.scheme != "https" or parsed.netloc != "upload.wikimedia.org":
        abort(400)
    # Reconstruct from validated components to break taint flow (CodeQL py/url-redirection)
    safe_url = urlunparse(("https", "upload.wikimedia.org", parsed.path, "", "", ""))
    return redirect(safe_url)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
# Security note: 400/403 handlers render e.description which comes from
# abort() calls.  All abort() descriptions in this codebase are translated
# constants (never user input), and Jinja2 auto-escapes {{ message }}.
# If adding new abort() calls, never pass unsanitised user input as the
# description argument.
# ---------------------------------------------------------------------------


@app.errorhandler(ApiError)
def _handle_api_error(error: ApiError):
    """Render a rejected API request as the JSON body the front end expects."""
    return jsonify({"error": error.message, **error.extra}), error.status


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 errors."""
    msg = e.description if hasattr(e, "description") and isinstance(e.description, str) else _("Bad request")
    return render_template("error.html", code=400, message=msg), 400


@app.errorhandler(403)
def forbidden(e):
    """Handle 403 errors."""
    msg = e.description if hasattr(e, "description") and isinstance(e.description, str) else _("Forbidden")
    return render_template("error.html", code=403, message=msg), 403


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


@app.errorhandler(503)
def service_unavailable(e):
    """Handle 503: the database (or another dependency) is briefly unavailable; the session is kept."""
    return (
        render_template(
            "error.html",
            code=503,
            message=_("The service is temporarily unavailable. Please try again in a moment."),
        ),
        503,
        {"Retry-After": "5"},
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
# Web pool per gunicorn worker. Not WIKIVISAGE_DB_POOL_SIZE: Toolforge envvars are
# tool-wide, so that variable already sizes the background workers' pools.
# ToolsDB allows 20 connections for the whole tool: 2 web workers x 3 here plus
# 2 background workers x 6 = 18, leaving two spare for migrations and one-off scripts.
WEB_DB_POOL_SIZE = int(os.environ.get("WIKIVISAGE_WEB_DB_POOL_SIZE", "3"))
init_db(pool_size=WEB_DB_POOL_SIZE)


def create_app() -> Flask:
    """Application factory for programmatic usage and testing."""
    return app


if __name__ == "__main__":
    if INSECURE_HTTP and app.debug:
        logger.warning(
            "FLASK_DEBUG is on, so the server still binds to 127.0.0.1 only. "
            "Unset FLASK_DEBUG to reach it from another device."
        )
    # The Werkzeug debugger allows code execution: never expose it beyond loopback.
    app.run(
        debug=app.debug,
        host="127.0.0.1" if app.debug else "0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
