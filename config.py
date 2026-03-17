"""Shared configuration constants used by both app.py and worker.py."""

import os

# Path to the file used to wake the background worker immediately.
# Both the web process (writes it) and the worker (reads/deletes it) use
# this path, so it must be defined in one place to stay consistent.
WAKE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up")
