"""Wake the background workers through the database.

The web app and the workers run in different Toolforge containers (the web service mounts
nothing, every worker is its own job), so a file written by one is never seen by the other.
They do share the database, so the "wake up" request is a counter there: the web app bumps
it, and every worker that sees a value it has not handled yet starts its next cycle at once
instead of waiting out the poll interval.

A counter rather than a flag, because each worker keeps its own baseline: two workers both
wake (the project claim is atomic, so they never take the same project), and a request that
arrives while a worker is busy is still seen on its next check instead of being lost.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from database import DatabaseError, execute_query

logger = logging.getLogger(__name__)

# How often a worker looks at the counter. One indexed single-row read, so cheap; 0 checks on
# every call (tests).
WAKE_CHECK_INTERVAL = float(os.environ.get("WIKIVISAGE_WAKE_CHECK_INTERVAL", 2))


def request_wake() -> bool:
    """Ask the workers to run a cycle now. Best effort: never raises, False when it did not land.

    A failure is harmless, the workers still pick the work up on their next poll.
    """
    try:
        execute_query(
            "INSERT INTO worker_wake (id, seq) VALUES (1, 1) ON DUPLICATE KEY UPDATE seq = seq + 1",
            fetch=False,
        )
        return True
    except DatabaseError as exc:
        logger.warning("Could not wake the worker (it will notice on its next poll): %s", exc)
        return False


class WakeWatcher:
    """One worker's view of the wake counter: ``consume()`` is True once per new request."""

    def __init__(self, interval: float | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = WAKE_CHECK_INTERVAL if interval is None else interval
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: int | None = None
        self._checked_at: float | None = None

    def _read(self) -> int | None:
        try:
            rows = execute_query("SELECT seq FROM worker_wake WHERE id = 1", fetch=True)
        except DatabaseError as exc:
            logger.debug("Wake check skipped: %s", exc)
            return None
        return int(rows[0]["seq"]) if rows else 0

    def prime(self) -> None:
        """Take the current counter as handled. Call before the first cycle: it looks at everything."""
        with self._lock:
            current = self._read()
            if current is not None:
                self._seen = current

    def consume(self) -> bool:
        """True when a request arrived since the last call that returned True (or since ``prime``)."""
        with self._lock:
            now = self._clock()
            if self._checked_at is not None and now - self._checked_at < self._interval:
                return False
            self._checked_at = now
            current = self._read()
            if current is None:
                return False
            if self._seen is None:
                # Never primed (the database was down at startup): adopt the value, wake next time.
                self._seen = current
                return False
            if current != self._seen:
                self._seen = current
                return True
            return False
