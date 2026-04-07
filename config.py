"""Shared configuration constants used by both app.py and worker.py."""

import os

# Path to the file used to wake the background worker immediately.
# Both the web process (writes it) and the worker (reads/deletes it) use
# this path, so it must be defined in one place to stay consistent.
WAKE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")

# Base directory for heartbeat files touched by each worker instance.
# Each worker appends its own worker-id to form a unique file path,
# so independent Toolforge health checks can detect per-worker staleness.
HEARTBEAT_FILE_DIR = os.path.expanduser("~")
