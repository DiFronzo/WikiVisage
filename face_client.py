"""HTTP client for the WikiVisage face detection service.

Shared by ``app.py`` and ``worker.py``. Has no ML dependencies by design —
importing this module must never pull in dlib. Encodings come back as raw
bytes, ready to write straight into the ``faces.encoding`` BLOB.

Configured via ``WIKIVISAGE_FACE_SERVICE_*`` environment variables; the module
constants below carry the defaults. See TESTING.md for the wire contract.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

FACE_SERVICE_URL = os.environ.get("WIKIVISAGE_FACE_SERVICE_URL", "http://localhost:8080").rstrip("/")
FACE_SERVICE_MODEL = os.environ.get("WIKIVISAGE_FACE_SERVICE_MODEL", "wikivisage")
FACE_SERVICE_TOKEN = os.environ.get("WIKIVISAGE_FACE_SERVICE_TOKEN", "")

FACE_SERVICE_BATCH_SIZE = int(os.environ.get("WIKIVISAGE_FACE_SERVICE_BATCH_SIZE", 8))
FACE_SERVICE_MAX_BATCH_BYTES = int(os.environ.get("WIKIVISAGE_FACE_SERVICE_MAX_BATCH_BYTES", 24 * 1024 * 1024))

FACE_SERVICE_TIMEOUT = float(os.environ.get("WIKIVISAGE_FACE_SERVICE_TIMEOUT", 30))
FACE_SERVICE_TIMEOUT_PER_IMAGE = float(os.environ.get("WIKIVISAGE_FACE_SERVICE_TIMEOUT_PER_IMAGE", 15))
FACE_SERVICE_CONNECT_TIMEOUT = 10.0

USER_AGENT = "WikiVisage/1.0 (Wikimedia Toolforge; https://toolsadmin.wikimedia.org)"

# A 128D float64 encoding is exactly this many bytes. Enforced on receipt so a
# misconfigured or wrong-version server cannot poison the database with
# vectors that would silently break centroid math downstream.
ENCODING_BYTES = 128 * 8


class FaceServiceError(Exception):
    """Base class for all face service failures."""


class FaceServiceUnavailable(FaceServiceError):
    """The service could not be reached, timed out, or returned a bad response.

    Retryable in principle — the caller should leave work pending rather than
    marking it permanently failed.
    """


class FaceServiceRejected(FaceServiceError):
    """The service processed the request but refused this specific image.

    Not retryable: the image is malformed, oversized, or the bounding box is
    unusable. The caller should record a terminal error for this item.
    """


@dataclass(frozen=True)
class DetectedFace:
    """One detected face: CSS-order bounding box plus its 128D encoding."""

    top: int
    right: int
    bottom: int
    left: int
    encoding: bytes = field(repr=False)

    @property
    def location(self) -> tuple[int, int, int, int]:
        """Return the box in ``face_recognition`` CSS order."""
        return (self.top, self.right, self.bottom, self.left)


@dataclass(frozen=True)
class DetectionResult:
    """Detection output for a single image."""

    width: int
    height: int
    faces: list[DetectedFace]

    @property
    def locations(self) -> list[tuple[int, int, int, int]]:
        return [face.location for face in self.faces]

    @property
    def encodings(self) -> list[bytes]:
        return [face.encoding for face in self.faces]

    def __len__(self) -> int:
        return len(self.faces)


_session: requests.Session | None = None
_session_lock = threading.Lock()


def _build_session() -> requests.Session:
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if FACE_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {FACE_SERVICE_TOKEN}"
    session.headers.update(headers)

    # Detection is idempotent, so retrying POSTs is safe. Backoff is short:
    # the caller has its own outer poll loop and would rather fail fast and
    # leave images pending than block a worker thread for minutes.
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_session() -> requests.Session:
    """Return the shared session, creating it on first use."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = _build_session()
    return _session


def reset_session() -> None:
    """Drop the cached session. Used by tests and after config changes."""
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
        _session = None


def predict_url() -> str:
    """Return the KServe V1 predict endpoint for the configured model."""
    return f"{FACE_SERVICE_URL}/v1/models/{FACE_SERVICE_MODEL}:predict"


def health_url() -> str:
    """Return the KServe V1 model-readiness endpoint."""
    return f"{FACE_SERVICE_URL}/v1/models/{FACE_SERVICE_MODEL}"


def _auth_probe_url() -> str:
    """A cheap endpoint the service only answers when the token is accepted."""
    return f"{FACE_SERVICE_URL}/v1/models"


def is_healthy(timeout: float = 5.0) -> bool:
    """Best-effort readiness probe. Never raises.

    Readiness alone is not enough: the readiness path is deliberately open so
    the platform health check works, so a missing or wrong token would still
    report ready while every predict call fails with 401. The second request
    hits a token-protected path to catch that before workers are started.
    """
    try:
        session = get_session()
        resp = session.get(health_url(), timeout=timeout)
        if resp.status_code != 200 or not resp.json().get("ready", False):
            return False
        auth = session.get(_auth_probe_url(), timeout=timeout)
        if auth.status_code == 401:
            logger.error("Face service rejected WIKIVISAGE_FACE_SERVICE_TOKEN (HTTP 401) — token missing or wrong")
            return False
        return auth.status_code == 200
    except Exception:
        return False


def _request_timeout(batch_len: int) -> tuple[float, float]:
    """Scale the read timeout with batch size; connect timeout stays fixed."""
    read = FACE_SERVICE_TIMEOUT + FACE_SERVICE_TIMEOUT_PER_IMAGE * max(1, batch_len)
    return (FACE_SERVICE_CONNECT_TIMEOUT, read)


def _post(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """POST a batch and return the raw per-instance prediction dicts.

    Raises FaceServiceUnavailable for any transport- or envelope-level problem.
    Per-instance failures are left in the returned dicts for the caller to
    classify.
    """
    try:
        resp = get_session().post(
            predict_url(),
            json={"instances": instances},
            timeout=_request_timeout(len(instances)),
        )
    except requests.RequestException as exc:
        raise FaceServiceUnavailable(f"face service request failed: {exc}") from exc

    if resp.status_code >= 400:
        detail = resp.text[:500]
        raise FaceServiceUnavailable(f"face service returned HTTP {resp.status_code}: {detail}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise FaceServiceUnavailable(f"face service returned non-JSON response: {exc}") from exc

    predictions = body.get("predictions") if isinstance(body, dict) else None
    if not isinstance(predictions, list):
        raise FaceServiceUnavailable("face service response is missing a 'predictions' list")
    if len(predictions) != len(instances):
        raise FaceServiceUnavailable(
            f"face service returned {len(predictions)} predictions for {len(instances)} instances"
        )
    return predictions


def _decode_encoding(raw: Any) -> bytes:
    """Base64-decode one encoding and enforce the exact 1024-byte width."""
    if not isinstance(raw, str):
        raise FaceServiceUnavailable("face encoding is missing or not a string")
    try:
        encoding = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceServiceUnavailable(f"face encoding is not valid base64: {exc}") from exc
    if len(encoding) != ENCODING_BYTES:
        raise FaceServiceUnavailable(f"face encoding is {len(encoding)} bytes, expected {ENCODING_BYTES}")
    return encoding


def _parse_prediction(prediction: dict[str, Any]) -> DetectionResult:
    """Convert one prediction dict into a DetectionResult.

    Raises FaceServiceRejected if the server reported a per-image error, or
    FaceServiceUnavailable if the payload is structurally wrong.
    """
    status = prediction.get("status")
    if status == "error":
        raise FaceServiceRejected(str(prediction.get("error", "unknown face service error")))
    if status != "ok":
        raise FaceServiceUnavailable(f"unexpected prediction status {status!r}")

    raw_faces = prediction.get("faces")
    if not isinstance(raw_faces, list):
        raise FaceServiceUnavailable("prediction is missing a 'faces' list")

    faces: list[DetectedFace] = []
    for raw_face in raw_faces:
        if not isinstance(raw_face, dict):
            raise FaceServiceUnavailable("each face must be an object")
        box = raw_face.get("box")
        if not isinstance(box, dict):
            raise FaceServiceUnavailable("each face must carry a 'box' object")
        try:
            faces.append(
                DetectedFace(
                    top=int(box["top"]),
                    right=int(box["right"]),
                    bottom=int(box["bottom"]),
                    left=int(box["left"]),
                    encoding=_decode_encoding(raw_face.get("encoding")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FaceServiceUnavailable(f"malformed face box: {exc}") from exc

    try:
        width = int(prediction.get("width", 0))
        height = int(prediction.get("height", 0))
    except (TypeError, ValueError) as exc:
        raise FaceServiceUnavailable(f"malformed image dimensions: {exc}") from exc

    return DetectionResult(width=width, height=height, faces=faces)


def _chunk(items: Sequence[tuple[str, bytes]]) -> Iterable[list[tuple[str, bytes]]]:
    """Split items into batches bounded by both count and total raw bytes.

    A single item larger than the byte budget still gets its own batch — the
    server enforces its own per-image ceiling and will reject it cleanly.
    """
    batch: list[tuple[str, bytes]] = []
    batch_bytes = 0
    for key, image_bytes in items:
        size = len(image_bytes)
        if batch and (len(batch) >= FACE_SERVICE_BATCH_SIZE or batch_bytes + size > FACE_SERVICE_MAX_BATCH_BYTES):
            yield batch
            batch, batch_bytes = [], 0
        batch.append((key, image_bytes))
        batch_bytes += size
    if batch:
        yield batch


def detect_faces_batch(items: Sequence[tuple[str, bytes]]) -> dict[str, DetectionResult | FaceServiceError]:
    """Detect faces in several images, splitting them across batched requests.

    ``items`` is a sequence of ``(key, image_bytes)``. The returned mapping has
    one entry per key: either a :class:`DetectionResult` or the
    :class:`FaceServiceError` explaining why that image failed. Transport
    failures fail every key in the affected batch but leave other batches
    intact, so one flaky request does not discard a whole run's work.
    """
    results: dict[str, DetectionResult | FaceServiceError] = {}
    if not items:
        return results

    for batch in _chunk(items):
        instances = [
            {"id": key, "task": "detect", "image": base64.b64encode(image_bytes).decode("ascii")}
            for key, image_bytes in batch
        ]
        try:
            predictions = _post(instances)
        except FaceServiceError as exc:
            logger.warning("Face service batch of %d failed: %s", len(batch), exc)
            for key, _bytes in batch:
                results[key] = exc
            continue

        # The server preserves instance order; ids are echoed back for defence
        # in depth but positional pairing is the contract.
        for (key, _bytes), prediction in zip(batch, predictions):
            try:
                results[key] = _parse_prediction(prediction)
            except FaceServiceError as exc:
                results[key] = exc

    return results


def detect_faces(image_bytes: bytes) -> DetectionResult:
    """Detect all faces in a single image.

    Raises :class:`FaceServiceRejected` if the image itself is unusable, or
    :class:`FaceServiceUnavailable` if the service could not be reached.
    """
    predictions = _post([{"id": "0", "task": "detect", "image": base64.b64encode(image_bytes).decode("ascii")}])
    return _parse_prediction(predictions[0])


def encode_known_face(image_bytes: bytes, bbox: tuple[int, int, int, int]) -> bytes | None:
    """Compute the 128D encoding for one known bounding box.

    ``bbox`` is ``(top, right, bottom, left)``. Returns the raw 1024-byte
    encoding, or ``None`` if the service could not produce one for that region
    (typically because the box is too small or contains no usable face) — which
    callers surface to the user as "try drawing a slightly larger box".
    """
    top, right, bottom, left = bbox
    predictions = _post(
        [
            {
                "id": "0",
                "task": "encode",
                "image": base64.b64encode(image_bytes).decode("ascii"),
                "boxes": [{"top": top, "right": right, "bottom": bottom, "left": left}],
            }
        ]
    )
    try:
        result = _parse_prediction(predictions[0])
    except FaceServiceRejected as exc:
        logger.info("Face service rejected manual bbox %s: %s", bbox, exc)
        return None
    if not result.faces:
        return None
    return result.faces[0].encoding
