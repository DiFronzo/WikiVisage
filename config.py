"""Shared configuration constants used by both app.py and worker.py."""

import os
import tomllib

# Version from pyproject.toml. Wikimedia's User-Agent policy wants
# ``<client>/<version> (<contact>)``, so worker and LiftWing client read it
# here instead of hard-coding "1.0". Falls back if the file is unreadable.
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"), "rb") as _f:
        PROJECT_VERSION = str(tomllib.load(_f)["project"]["version"])
except (OSError, KeyError, tomllib.TOMLDecodeError):  # pragma: no cover - pyproject ships with the repo
    PROJECT_VERSION = "1.0"

# The one User-Agent every outbound client sends (web app, worker, LiftWing and
# face-service clients).
USER_AGENT = f"WikiVisage/{PROJECT_VERSION} (https://github.com/DiFronzo/WikiVisage)"

# Base directory for heartbeat files touched by each worker instance.
# Each worker appends its own worker-id to form a unique file path,
# so independent Toolforge health checks can detect per-worker staleness.
HEARTBEAT_FILE_DIR = os.path.expanduser("~")

# A project claim older than this no longer holds the project: the other worker may take it. The web
# app uses the same figure to tell "a worker is on it" from "a worker died holding the claim".
CLAIM_EXPIRY_MINUTES = 15

# Standard thumbnail widths Commons will serve (``$wgThumbnailSteps``); any other
# width is answered with HTTP 429. Shared so every thumbnail URL is built from
# one list.
COMMONS_THUMB_STEPS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)

# Seconds between LiftWing liveness probes by the worker (0 disables probing).
# The web app reads it too: a status older than three intervals is treated as
# unknown, so a stopped worker hides the banner instead of showing a stale one.
LIFTWING_PROBE_INTERVAL = int(os.environ.get("WIKIVISAGE_LIFTWING_PROBE_INTERVAL", "300"))


def _detection_width() -> int:
    """Width of the Commons thumbnail face detection runs on (``WIKIVISAGE_DETECTION_WIDTH``).

    The worker detects on it and the web app downloads the same width for manual
    drawing, so stored boxes and drawn boxes live in (nearly) the same pixel
    space. Must be a standard Commons step: anything else would 429 every image,
    so a bad value fails at startup instead.
    """
    width = int(os.environ.get("WIKIVISAGE_DETECTION_WIDTH", "960"))
    if width not in COMMONS_THUMB_STEPS:
        raise ValueError(f"WIKIVISAGE_DETECTION_WIDTH={width} is not a Commons thumbnail step {COMMONS_THUMB_STEPS}")
    return width


DETECTION_WIDTH = _detection_width()

# Login sessions. The cookie is a sliding "idle" lifetime that every request renews, and
# ``auth_at`` (set at login) enforces an absolute cap no matter how active the user is.
SESSION_IDLE_DAYS = int(os.environ.get("WIKIVISAGE_SESSION_IDLE_DAYS", "30"))
SESSION_MAX_AGE_DAYS = int(os.environ.get("WIKIVISAGE_SESSION_MAX_AGE_DAYS", "90"))

# Worker liveness, shared by the worker (writes) and the web app (reads).
# A worker row older than WORKER_STALE_SECONDS no longer counts as alive. A worker that is
# stopped on purpose (SIGTERM on a deploy or restart) announces that it will be back within
# WORKER_RESTART_GRACE_SECONDS, which keeps the offline banner hidden for planned restarts.
WORKER_STALE_SECONDS = 300
WORKER_RESTART_GRACE_SECONDS = int(os.environ.get("WIKIVISAGE_WORKER_RESTART_GRACE", "1200"))

# What the worker stores in ``projects.sdc_write_error`` when a Commons write run ended because the
# user's authorization is no longer accepted (token refused, revoked or expired). The web app
# recognises it (and the wordings older releases stored) to offer a one-click re-authorize.
SDC_AUTH_EXPIRED_MESSAGE = "Your Wikimedia authorization has expired or was revoked. Please re-authorize to continue."

# Stored instead when the refresh failed for a transient reason (login service timeout, 5xx, database
# trouble). Deliberately NOT in SDC_REAUTH_MESSAGES: sending the edits again is the fix, not re-authorizing.
SDC_AUTH_UNAVAILABLE_MESSAGE = (
    "Could not reach the Wikimedia login service. Please try sending the edits again in a moment."
)
SDC_REAUTH_MESSAGES = frozenset(
    {
        SDC_AUTH_EXPIRED_MESSAGE,
        "Token expired or user not found. Please log in again.",
        "Your session has expired. Please log in again.",
        "Authentication token expired. Please try again or log in again.",
    }
)
