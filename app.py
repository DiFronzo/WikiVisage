"""
WikiVisage — Flask web application.

Active-learning facial recognition tool for Wikimedia Commons.
Provides OAuth 2.0 authentication, project management, and an active
learning interface for classifying detected faces.
"""

import hashlib
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Optional

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
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from requests_oauthlib import OAuth2Session

from database import (
    DatabaseError,
    close_pool,
    execute_query,
    get_connection,
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

# OAuth 2.0 configuration (Wikimedia Meta)
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_AUTHORIZE_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/authorize"
OAUTH_TOKEN_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/access_token"
OAUTH_PROFILE_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/resource/profile"
OAUTH_REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", "")

# Session configuration
app.config["SESSION_COOKIE_SECURE"] = not app.debug
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

# Rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

# Token expiry buffer (refresh 5 minutes before actual expiry)
TOKEN_REFRESH_BUFFER = 300  # seconds


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
        except DatabaseError:
            logger.exception("Failed to load user from session")
            session.clear()


@app.teardown_appcontext
def teardown_appcontext(exception: Optional[BaseException] = None) -> None:
    """Clean up per-request resources."""
    pass  # Connection pool handles cleanup via context managers


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def login_required(f):
    """Decorator to require authentication."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def _make_oauth_session(state: Optional[str] = None) -> OAuth2Session:
    """Create an OAuth2Session with the configured client."""
    sess = OAuth2Session(
        client_id=OAUTH_CLIENT_ID,
        redirect_uri=OAUTH_REDIRECT_URI,
        state=state,
    )
    sess.headers["User-Agent"] = (
        "WikiVisage/1.0 (https://github.com/DiFronzo/WikiVisage)"
    )
    return sess


def _refresh_access_token(user: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Refresh the user's access token if it is expired or about to expire.

    Returns updated user dict or None if refresh failed.
    """
    expires_at = user["token_expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
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

        new_expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=new_token.get("expires_in", 14400)
        )

        execute_query(
            "UPDATE users SET access_token = %s, refresh_token = %s, "
            "token_expires_at = %s WHERE id = %s",
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


def _get_valid_token() -> Optional[str]:
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
    """Validate CSRF token from form submission."""
    token = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not expected or not token:
        return False
    return secrets.compare_digest(token, expected)


@app.context_processor
def inject_csrf_token() -> dict[str, Any]:
    """Make csrf_token() available in all templates."""
    return {"csrf_token": _csrf_token}


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
        flash("Invalid OAuth state. Please try again.", "error")
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
        flash("Authentication failed. Please try again.", "error")
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
        flash("Could not retrieve your profile. Please try again.", "error")
        return redirect(url_for("index"))

    wiki_user_id = profile.get("sub")
    wiki_username = profile.get("username", "")

    if not wiki_user_id:
        flash("Invalid profile data received.", "error")
        return redirect(url_for("index"))

    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=token.get("expires_in", 14400)
    )
    expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

    # Upsert user
    try:
        existing = execute_query(
            "SELECT id FROM users WHERE wiki_user_id = %s", (wiki_user_id,)
        )

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
            rows = execute_query(
                "SELECT id FROM users WHERE wiki_user_id = %s", (wiki_user_id,)
            )
            user_id = rows[0]["id"]

    except DatabaseError:
        logger.exception("Failed to upsert user")
        flash("Database error. Please try again.", "error")
        return redirect(url_for("index"))

    session.permanent = True
    session["user_id"] = user_id

    next_url = session.pop("login_next", "") or url_for("dashboard")
    return redirect(next_url)


@app.route("/logout")
def logout():
    """Clear session and log out."""
    session.clear()
    flash("You have been logged out.", "info")
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
    try:
        projects = execute_query(
            "SELECT * FROM projects WHERE user_id = %s ORDER BY updated_at DESC",
            (g.user["id"],),
        )
    except DatabaseError:
        logger.exception("Failed to load projects")
        projects = []
        flash("Failed to load projects.", "error")

    return render_template("dashboard.html", projects=projects)


@app.route("/project/new", methods=["GET", "POST"])
@login_required
@limiter.limit("20 per hour")
def project_new():
    """Create a new project."""
    if request.method == "GET":
        return render_template("project_new.html")

    if not _validate_csrf():
        abort(400, "Invalid CSRF token")

    wikidata_qid = request.form.get("wikidata_qid", "").strip().upper()
    commons_category = request.form.get("commons_category", "").strip()
    label = request.form.get("label", "").strip()
    distance_threshold = request.form.get("distance_threshold", "0.6")
    min_confirmed = request.form.get("min_confirmed", "5")

    # Validation
    errors = []
    if not wikidata_qid or not wikidata_qid.startswith("Q"):
        errors.append("Wikidata Q-ID must start with 'Q' (e.g., Q42).")
    if not commons_category:
        errors.append("Commons category is required.")

    try:
        distance_threshold = float(distance_threshold)
        if not 0.1 <= distance_threshold <= 1.0:
            errors.append("Distance threshold must be between 0.1 and 1.0.")
    except ValueError:
        errors.append("Distance threshold must be a number.")
        distance_threshold = 0.6

    try:
        min_confirmed = int(min_confirmed)
        if min_confirmed < 1:
            errors.append("Minimum confirmed must be at least 1.")
    except ValueError:
        errors.append("Minimum confirmed must be a whole number.")
        min_confirmed = 5

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
            "SELECT id FROM projects WHERE user_id = %s AND wikidata_qid = %s "
            "AND commons_category = %s",
            (g.user["id"], wikidata_qid, commons_category),
        )
        if existing:
            flash("A project with this Q-ID and category already exists.", "error")
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
        execute_query(
            "INSERT INTO projects (user_id, wikidata_qid, commons_category, label, "
            "distance_threshold, min_confirmed) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                g.user["id"],
                wikidata_qid,
                commons_category,
                label,
                distance_threshold,
                min_confirmed,
            ),
            fetch=False,
        )
        flash("Project created successfully!", "success")
        return redirect(url_for("dashboard"))
    except DatabaseError:
        logger.exception("Failed to create project")
        flash("Failed to create project. Please try again.", "error")
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
            "SELECT * FROM projects WHERE id = %s AND user_id = %s",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        logger.exception("Failed to load project")
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

    # Get face stats
    try:
        stats = execute_query(
            "SELECT "
            "  COUNT(*) AS total_faces, "
            "  SUM(CASE WHEN f.is_target = 1 THEN 1 ELSE 0 END) AS confirmed_matches, "
            "  SUM(CASE WHEN f.is_target = 0 THEN 1 ELSE 0 END) AS confirmed_non_matches, "
            "  SUM(CASE WHEN f.is_target IS NULL THEN 1 ELSE 0 END) AS unclassified, "
            "  SUM(CASE WHEN f.sdc_written = 1 THEN 1 ELSE 0 END) AS sdc_written "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s",
            (project_id,),
        )
        face_stats = stats[0] if stats else {}
    except DatabaseError:
        logger.exception("Failed to load face stats")
        face_stats = {}

    return render_template("project_detail.html", project=project, stats=face_stats)


@app.route("/project/<int:project_id>/classify")
@login_required
def classify(project_id: int):
    """Active learning classification interface."""
    # Verify project ownership
    try:
        rows = execute_query(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s",
            (project_id, g.user["id"]),
        )
    except DatabaseError:
        abort(500)

    if not rows:
        abort(404)

    project = rows[0]

    # Get next unclassified face
    try:
        faces = execute_query(
            "SELECT f.id AS face_id, f.bbox_top, f.bbox_right, f.bbox_bottom, "
            "f.bbox_left, f.confidence, i.file_title, i.commons_page_id "
            "FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.is_target IS NULL "
            "ORDER BY f.confidence ASC "
            "LIMIT 1",
            (project_id,),
        )
    except DatabaseError:
        logger.exception("Failed to load face for classification")
        faces = []

    face = faces[0] if faces else None

    # Count remaining
    try:
        remaining = execute_query(
            "SELECT COUNT(*) AS cnt FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s AND f.is_target IS NULL",
            (project_id,),
        )
        remaining_count = remaining[0]["cnt"] if remaining else 0
    except DatabaseError:
        remaining_count = 0

    return render_template(
        "classify.html",
        project=project,
        face=face,
        remaining=remaining_count,
    )


@app.route("/api/classify", methods=["POST"])
@login_required
@limiter.limit("120 per minute")
def api_classify():
    """API endpoint to classify a face as target/non-target."""
    if not _validate_csrf():
        return jsonify({"error": "Invalid CSRF token"}), 400

    face_id = request.form.get("face_id")
    is_target = request.form.get("is_target")
    project_id = request.form.get("project_id")

    if not face_id or is_target is None or not project_id:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        face_id = int(face_id)
        is_target_val = 1 if is_target == "1" else 0
        project_id = int(project_id)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid field values"}), 400

    # Verify ownership: face belongs to a project owned by this user
    try:
        check = execute_query(
            "SELECT f.id FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "JOIN projects p ON i.project_id = p.id "
            "WHERE f.id = %s AND p.id = %s AND p.user_id = %s",
            (face_id, project_id, g.user["id"]),
        )
        if not check:
            return jsonify({"error": "Face not found or access denied"}), 404
    except DatabaseError:
        return jsonify({"error": "Database error"}), 500

    # Update face classification
    try:
        execute_query(
            "UPDATE faces SET is_target = %s, classified_by = 'human' WHERE id = %s",
            (is_target_val, face_id),
            fetch=False,
        )

        # Update project counters if confirmed match
        if is_target_val == 1:
            execute_query(
                "UPDATE projects SET faces_confirmed = faces_confirmed + 1 "
                "WHERE id = %s",
                (project_id,),
                fetch=False,
            )

    except DatabaseError:
        logger.exception("Failed to classify face")
        return jsonify({"error": "Failed to save classification"}), 500

    return jsonify({"status": "ok", "face_id": face_id, "is_target": is_target_val})


@app.route("/project/<int:project_id>/settings", methods=["GET", "POST"])
@login_required
def project_settings(project_id: int):
    """Edit project settings."""
    try:
        rows = execute_query(
            "SELECT * FROM projects WHERE id = %s AND user_id = %s",
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
        abort(400, "Invalid CSRF token")

    distance_threshold = request.form.get(
        "distance_threshold", str(project["distance_threshold"])
    )
    min_confirmed = request.form.get("min_confirmed", str(project["min_confirmed"]))
    status = request.form.get("status", project["status"])
    label = request.form.get("label", project["label"])

    errors = []
    try:
        distance_threshold = float(distance_threshold)
        if not 0.1 <= distance_threshold <= 1.0:
            errors.append("Distance threshold must be between 0.1 and 1.0.")
    except ValueError:
        errors.append("Distance threshold must be a number.")
        distance_threshold = project["distance_threshold"]

    try:
        min_confirmed = int(min_confirmed)
        if min_confirmed < 1:
            errors.append("Minimum confirmed must be at least 1.")
    except ValueError:
        errors.append("Minimum confirmed must be a whole number.")
        min_confirmed = project["min_confirmed"]

    if status not in ("active", "paused", "completed"):
        errors.append("Invalid status.")
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
        flash("Settings updated.", "success")
        return redirect(url_for("project_detail", project_id=project_id))
    except DatabaseError:
        logger.exception("Failed to update project settings")
        flash("Failed to update settings.", "error")
        return render_template("project_settings.html", project=project)


@app.route("/project/<int:project_id>/delete", methods=["POST"])
@login_required
def project_delete(project_id: int):
    """Delete a project and all associated data (cascades via FK)."""
    if not _validate_csrf():
        abort(400, "Invalid CSRF token")

    try:
        affected = execute_query(
            "DELETE FROM projects WHERE id = %s AND user_id = %s",
            (project_id, g.user["id"]),
            fetch=False,
        )
        if affected:
            flash("Project deleted.", "info")
        else:
            flash("Project not found.", "error")
    except DatabaseError:
        logger.exception("Failed to delete project")
        flash("Failed to delete project.", "error")

    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------


@app.route("/health")
@limiter.exempt
def health():
    """Health check endpoint for Toolforge monitoring."""
    try:
        rows = execute_query("SELECT 1 AS ok")
        if rows and rows[0].get("ok") == 1:
            return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

    return jsonify({"status": "unhealthy"}), 503


@app.route("/commons-thumb/<path:file_title>")
@login_required
def commons_thumb(file_title: str):
    """
    Generate a Wikimedia Commons thumbnail URL for a file.

    Returns JSON with the thumbnail URL. Used by the classification UI
    to display face images without directly proxying Commons content.
    """
    # Compute MD5 hash for thumb URL construction
    md5 = hashlib.md5(file_title.encode("utf-8")).hexdigest()

    # Standard Wikimedia thumbnail URL pattern
    thumb_url = (
        f"https://upload.wikimedia.org/wikipedia/commons/thumb/"
        f"{md5[0]}/{md5[0:2]}/{file_title}/300px-{file_title}"
    )

    return jsonify({"url": thumb_url, "file_title": file_title})


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
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(429)
def rate_limited(e):
    """Handle rate limit exceeded."""
    return (
        render_template(
            "error.html",
            code=429,
            message="Rate limit exceeded. Please try again later.",
        ),
        429,
    )


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return (
        render_template("error.html", code=500, message="Internal server error"),
        500,
    )


# ---------------------------------------------------------------------------
# App factory / startup
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Application factory for programmatic usage and testing."""
    init_db()
    return app


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
