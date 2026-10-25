"""Exceptions the JSON API raises to reject a request.

Kept out of ``app.py`` on purpose: the Flask error handler is registered for this
class, and a handler is matched by class identity. Test fixtures ``importlib.reload``
the ``app`` module, which would create a new class each time and orphan the handler
of any app object created earlier. A module nobody reloads has one class for the
life of the process.
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """A request the JSON API rejects: carries the status and the body the client receives.

    Raising it from a helper (instead of returning a response tuple) lets validation and
    face-service steps be shared between routes. ``app._handle_api_error`` turns it into the
    ``{"error": ...}`` response the routes used to build inline. Inside a ``try`` with a broad
    ``except Exception``, re-raise it first (``except ApiError: raise``).
    """

    def __init__(self, message: str, status: int = 400, **extra: Any):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra
