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

# Path to the file used to wake the background worker immediately.
# Both the web process (writes it) and the worker (reads/deletes it) use
# this path, so it must be defined in one place to stay consistent.
WAKE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")

# Base directory for heartbeat files touched by each worker instance.
# Each worker appends its own worker-id to form a unique file path,
# so independent Toolforge health checks can detect per-worker staleness.
HEARTBEAT_FILE_DIR = os.path.expanduser("~")

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
