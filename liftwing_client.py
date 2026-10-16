"""HTTP client for the experimental LiftWing Qwen metadata pre-filter.

Phase 2 of ``docs/plans/face-model-upgrade.md``. Standalone by design: no ML
dependencies, no database access, no imports from ``worker.py`` — just one
OpenAI-compatible chat-completions endpoint. The worker owns the pipeline and
the ``service_status`` table; this module only talks HTTP.

Every call **fails open**. ``filter_metadata`` returns ``None`` instead of
raising when the service is slow, down, or unparseable, so the caller keeps
every file and simply retries later. Files whose MIME type cannot depict a
human face (audio, video, PDF, SVG, DjVu) are settled deterministically as
``keep=false`` without a model call; anything else, including a missing MIME
type, still goes to the model. A small circuit breaker (per process)
backs the pre-filter off after repeated failures, mirroring the worker's
database-level probe hysteresis. A ``429`` opens the breaker immediately and
honours ``Retry-After`` when the gateway sends it.

Configured via ``WIKIVISAGE_LIFTWING_*`` environment variables; the module
constants below carry the defaults. Set ``WIKIVISAGE_LIFTWING_API_TOKEN`` to
send ``Authorization: Bearer <token>`` (OAuth 2.0 / personal API token from
https://api.wikimedia.org), which lifts the anonymous LiftWing rate limit.
"""

from __future__ import annotations

import email.utils
import json
import logging
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from config import PROJECT_VERSION

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://api.wikimedia.org/service/lw/inference/v1/models/llm-qwen36-27b/openai/v1/chat/completions"

# Wikimedia's User-Agent policy wants ``<client>/<version> (<contact>)``. Keep
# this in step with the worker's own UA (same string, same version).
USER_AGENT = os.environ.get(
    "WIKIVISAGE_LIFTWING_USER_AGENT",
    f"WikiVisage/{PROJECT_VERSION} (https://github.com/DiFronzo/WikiVisage)",
)

# A whole batch must finish inside this. Measured 2026-10-17: a 40-file batch
# took ~40 s and a 20-file batch ~20 s (a verdict per file, each with a reason
# then; reasons now come only on drops). The worker's LIFTWING_BATCH_SIZE
# defaults to 20, so 90 s leaves ~4x headroom.
REQUEST_TIMEOUT = float(os.environ.get("WIKIVISAGE_LIFTWING_TIMEOUT", 90))
# The probe is one tiny completion; 20 s is generous for a loaded replica.
PROBE_TIMEOUT = float(os.environ.get("WIKIVISAGE_LIFTWING_PROBE_TIMEOUT", 20))
PROBE_MAX_TOKENS = 8

# Verdicts are short; 2000 tokens leaves room for ~50 of them plus the model's
# reasoning preamble. Truncation would break the JSON and fail open, which is
# safe but useless, so keep this roomy.
MAX_TOKENS = int(os.environ.get("WIKIVISAGE_LIFTWING_MAX_TOKENS", 2000))

# Circuit breaker: after this many consecutive failures — or on the first
# HTTP 429 — stop calling the service for the cooldown (at least as long as
# Retry-After says) and let the worker probe take over.
CIRCUIT_FAILURE_THRESHOLD = int(os.environ.get("WIKIVISAGE_LIFTWING_CIRCUIT_FAILURES", 3))
CIRCUIT_COOLDOWN = float(os.environ.get("WIKIVISAGE_LIFTWING_CIRCUIT_COOLDOWN", 600))

MAX_REASON_CHARS = 160
MAX_DESCRIPTION_CHARS = 250
MAX_CATEGORIES_PER_FILE = 15

# Media that cannot depict a human face. Settled as ``keep=false`` without a
# model call: nothing is gained by asking, and the round trip is saved. Any
# other MIME type — or a missing one — still goes to the model, so a rare
# visual format is judged rather than dropped by accident.
NON_VISUAL_MIME_PREFIXES = ("audio/", "video/")
NON_VISUAL_MIME_TYPES = frozenset({"application/pdf", "application/ogg", "image/svg+xml", "image/vnd.djvu"})

# Qwen3 may emit a reasoning block before the answer; the JSON is outside it.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You pre-filter files for WikiVisage, a tool that classifies faces in Wikimedia Commons files. "
    "You receive a JSON array of files, each with a page_id, title, description and categories. "
    "For every file decide whether it could plausibly depict a visible human face in any medium: "
    "photographs, paintings, drawings, engravings, busts, sculptures, coins, stamps and posters all count. "
    "Set keep=true whenever you are unsure, when a face might be present, and when the metadata is sparse. "
    "Set keep=false only when the file clearly cannot depict a face, such as signatures, autographs, "
    "handwritten letters, text scans, certificates, gravestones or tombs without portraits, memorial "
    "plaques, buildings, residences, streets, landscapes, maps, diagrams, charts, coats of arms, flags "
    "or logos. "
    'Answer with ONLY a raw JSON array of objects: [{"page_id": <int>, "keep": <true|false>}]. '
    'For keep=false entries add a "reason" of at most four words; omit it for keep=true entries. '
    "Do not wrap the output in markdown code fences and do not add any other text."
)


def is_visual_media(mime: str) -> bool:
    """Whether a file of this MIME type could show a face at all.

    Unknown or empty types return ``True`` (fail open): only the explicitly
    non-visual types above are settled without asking the model.
    """
    mime = (mime or "").strip().lower()
    return not (mime.startswith(NON_VISUAL_MIME_PREFIXES) or mime in NON_VISUAL_MIME_TYPES)


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class LiftWingError(Exception):
    """Base class for LiftWing failures."""


class LiftWingUnavailable(LiftWingError):
    """The service could not be reached, timed out, or answered unusably.

    Retryable in principle: the caller keeps its files and tries again later.
    ``status_code`` and ``retry_after`` are filled in for HTTP failures so the
    circuit breaker can honour a ``429``'s ``Retry-After``.
    """

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class MetadataItem:
    """One Commons file as sent to the model. Never contains image bytes."""

    page_id: int
    title: str
    description: str = ""
    categories: tuple[str, ...] = ()
    mime: str = ""


@dataclass(frozen=True)
class Verdict:
    """The model's verdict for one file."""

    page_id: int
    keep: bool
    reason: str


def _coerce_bool(value: Any) -> bool | None:
    """Interpret the model's ``keep`` value leniently; ``None`` means unusable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "keep"):
            return True
        if lowered in ("false", "0", "no", "drop"):
            return False
    return None


def parse_verdicts(content: str) -> dict[int, Verdict]:
    """Parse the model's answer into ``{page_id: Verdict}``.

    Tolerates a ``<think>…</think>`` reasoning block before the answer and a
    Markdown code fence around it. Raises :class:`LiftWingUnavailable` when no
    usable JSON array can be extracted — the caller then fails open.
    """
    cleaned = _THINK_BLOCK_RE.sub(" ", content)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1)

    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        raise LiftWingUnavailable("no JSON array in the completion")
    try:
        raw = json.loads(cleaned[start : end + 1])
    except ValueError as exc:
        raise LiftWingUnavailable(f"completion is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise LiftWingUnavailable("completion JSON is not an array")

    verdicts: dict[int, Verdict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            page_id = int(entry["page_id"])
        except (KeyError, TypeError, ValueError):
            continue
        keep = _coerce_bool(entry.get("keep"))
        if keep is None:
            continue
        reason = entry.get("reason")
        verdicts[page_id] = Verdict(
            page_id=page_id,
            keep=keep,
            reason=str(reason)[:MAX_REASON_CHARS] if reason is not None else "",
        )
    if not verdicts:
        raise LiftWingUnavailable("completion contained no usable verdicts")
    return verdicts


class LiftWingClient:
    """One shared instance per worker process. Requests are serialised.

    The pre-filter is deliberately low-throughput: one request at a time per
    worker (``concurrency 1``), a single attempt per call, and a circuit
    breaker that stops calls for a while after repeated failures.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float | None = None,
        probe_timeout: float | None = None,
        max_tokens: int | None = None,
        circuit_failure_threshold: int | None = None,
        circuit_cooldown: float | None = None,
        user_agent: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self.url = url or os.environ.get("WIKIVISAGE_LIFTWING_URL", DEFAULT_URL)
        self.model = model or os.environ.get("WIKIVISAGE_LIFTWING_MODEL", "llm-qwen36-27b")
        self.timeout = REQUEST_TIMEOUT if timeout is None else timeout
        self.probe_timeout = PROBE_TIMEOUT if probe_timeout is None else probe_timeout
        self.max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
        self.circuit_failure_threshold = (
            CIRCUIT_FAILURE_THRESHOLD if circuit_failure_threshold is None else circuit_failure_threshold
        )
        self.circuit_cooldown = CIRCUIT_COOLDOWN if circuit_cooldown is None else circuit_cooldown
        self.user_agent = user_agent or USER_AGENT
        # Optional OAuth 2.0 / personal API token; never logged.
        self.api_token = api_token if api_token is not None else os.environ.get("WIKIVISAGE_LIFTWING_API_TOKEN", "")

        self._session = session if session is not None else requests.Session()
        # Serialises HTTP calls so a worker never runs two pre-filter requests
        # at once, whatever its project-level thread pool is doing.
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._disabled_until = 0.0

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        try:
            self._session.close()
        except Exception:  # pragma: no cover - close() must never raise
            pass

    # -- circuit breaker ---------------------------------------------------

    def _circuit_open(self) -> bool:
        return time.monotonic() < self._disabled_until

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._disabled_until = 0.0

    def _record_failure(self, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        self._consecutive_failures += 1
        cooldown = self.circuit_cooldown
        if retry_after is not None:
            cooldown = max(cooldown, retry_after)
        # A 429 is an explicit "come back later": open immediately and honour
        # Retry-After instead of hammering the gateway two more times.
        if status_code == 429 or self._consecutive_failures >= self.circuit_failure_threshold:
            self._disabled_until = time.monotonic() + cooldown
            logger.warning(
                "LiftWing circuit breaker opened after %d consecutive failure(s)%s; paused for %.0f s",
                self._consecutive_failures,
                " (HTTP 429)" if status_code == 429 else "",
                cooldown,
            )

    # -- HTTP --------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """One chat completion; returns the assistant's content or raises."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        try:
            with self._lock:
                response = self._session.post(
                    self.url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout if timeout is None else timeout,
                )
        except requests.exceptions.RequestException as exc:
            raise LiftWingUnavailable(f"request failed: {exc}") from exc

        if response.status_code != 200:
            raise LiftWingUnavailable(
                f"HTTP {response.status_code}",
                status_code=response.status_code,
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LiftWingUnavailable("response is not JSON") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LiftWingUnavailable("response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LiftWingUnavailable("completion is empty")
        return content

    def _build_messages(self, items: Sequence[MetadataItem]) -> list[dict[str, str]]:
        files = [
            {
                "page_id": item.page_id,
                "title": item.title,
                "description": item.description[:MAX_DESCRIPTION_CHARS],
                "categories": [
                    category.removeprefix("Category:") for category in item.categories[:MAX_CATEGORIES_PER_FILE]
                ],
            }
            for item in items
        ]
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(files, ensure_ascii=False)},
        ]

    # -- public API ----------------------------------------------------------

    def filter_metadata(self, items: Sequence[MetadataItem]) -> dict[int, Verdict] | None:
        """Ask for verdicts on a batch of files.

        Returns ``{page_id: Verdict}`` on success. Files with a non-visual MIME
        type are settled here as ``keep=false`` without a model call; the rest
        go to the model in one request. Returns ``None`` — meaning "keep
        everything, try again later" — when the circuit is open or the request
        or its answer was unusable. Never raises.
        """
        items = list(items)
        if not items:
            return {}
        visual = [item for item in items if is_visual_media(item.mime)]
        settled = {
            item.page_id: Verdict(
                page_id=item.page_id,
                keep=False,
                reason=f"non-visual media ({item.mime})"[:MAX_REASON_CHARS],
            )
            for item in items
            if not is_visual_media(item.mime)
        }
        if not visual:
            logger.info("LiftWing pre-filter settled %d non-visual file(s) without a call", len(settled))
            return settled
        if self._circuit_open():
            logger.info("LiftWing circuit breaker is open; keeping %d file(s) unfiltered", len(visual))
            return None

        try:
            content = self._chat(self._build_messages(visual))
        except LiftWingUnavailable as exc:
            self._record_failure(status_code=exc.status_code, retry_after=exc.retry_after)
            logger.warning("LiftWing pre-filter request failed (%s); keeping %d file(s)", exc, len(visual))
            return None

        try:
            verdicts = parse_verdicts(content)
        except LiftWingUnavailable as exc:
            self._record_failure()
            logger.warning("LiftWing answer unusable (%s); keeping %d file(s)", exc, len(visual))
            return None

        self._record_success()
        verdicts.update(settled)
        return verdicts

    def probe(self) -> tuple[bool, str | None]:
        """Check whether LiftWing can answer a real completion. Never raises.

        Deliberately runs a minimal chat completion (``max_tokens=8``) rather
        than the cheap ``GET <base>/models``: the models endpoint can answer
        200 while completions are timing out or rate limited, which would hide
        a broken pre-filter from the status banner and wrongly close the
        circuit breaker. A 429 counts as a failure and its ``Retry-After``
        feeds the breaker cooldown.
        """
        try:
            self._chat(
                [{"role": "user", "content": "Reply with the single word: pong"}],
                max_tokens=PROBE_MAX_TOKENS,
                timeout=self.probe_timeout,
            )
        except LiftWingUnavailable as exc:
            self._record_failure(status_code=exc.status_code, retry_after=exc.retry_after)
            return False, str(exc)
        except Exception as exc:  # defensive: a probe must never take the worker down
            logger.warning("LiftWing probe crashed: %s", exc)
            return False, str(exc)
        self._record_success()
        return True, None


_CLIENT: LiftWingClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> LiftWingClient:
    """The process-wide client (one per worker process)."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = LiftWingClient()
    return _CLIENT


def reset_client() -> None:
    """Drop and close the process-wide client (worker shutdown, tests)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None
