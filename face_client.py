"""HTTP client for the WikiVisage face detection service.

Shared by ``app.py`` and ``worker.py``. Has no ML dependencies by design —
importing this module must never pull in onnxruntime or OpenCV. Embeddings come
back as raw bytes (``dim`` little-endian float32 values, L2-normalised), ready
to write straight into the ``faces.encoding`` BLOB.

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
from typing import Any, NamedTuple

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from config import USER_AGENT

logger = logging.getLogger(__name__)

FACE_SERVICE_URL = os.environ.get("WIKIVISAGE_FACE_SERVICE_URL", "http://localhost:8080").rstrip("/")
FACE_SERVICE_MODEL = os.environ.get("WIKIVISAGE_FACE_SERVICE_MODEL", "wikivisage")
FACE_SERVICE_TOKEN = os.environ.get("WIKIVISAGE_FACE_SERVICE_TOKEN", "")

FACE_SERVICE_BATCH_SIZE = int(os.environ.get("WIKIVISAGE_FACE_SERVICE_BATCH_SIZE", 8))
FACE_SERVICE_MAX_BATCH_BYTES = int(os.environ.get("WIKIVISAGE_FACE_SERVICE_MAX_BATCH_BYTES", 24 * 1024 * 1024))
# Mirrors the server's WIKIVISAGE_MAX_IMAGE_BYTES. Larger images are rejected
# here: the server would refuse them anyway, and above ~30 MiB the base64 body
# exceeds its 40 MiB request cap, which surfaces as a retryable HTTP 413 and
# would leave the image pending forever.
FACE_SERVICE_MAX_IMAGE_BYTES = int(os.environ.get("WIKIVISAGE_FACE_SERVICE_MAX_IMAGE_BYTES", 20 * 1024 * 1024))

FACE_SERVICE_TIMEOUT = float(os.environ.get("WIKIVISAGE_FACE_SERVICE_TIMEOUT", 30))
FACE_SERVICE_TIMEOUT_PER_IMAGE = float(os.environ.get("WIKIVISAGE_FACE_SERVICE_TIMEOUT_PER_IMAGE", 15))
FACE_SERVICE_CONNECT_TIMEOUT = 10.0

# The embedding the database holds and the inference math expects. Enforced on
# receipt — dimension, byte width and model id — so a misconfigured or
# wrong-version server cannot poison the database with vectors that would
# silently break centroid math downstream.
EXPECTED_MODEL_ID = os.environ.get("WIKIVISAGE_FACE_MODEL_ID", "scrfd500m+arcface_mbf")
EMBEDDING_DIM = int(os.environ.get("WIKIVISAGE_FACE_EMBEDDING_DIM", 512))
BYTES_PER_COMPONENT = 4  # little-endian float32

# The only per-instance ``error_code`` this client interprets: the service's
# image bytes cannot be decoded or are over the size limit. Only the ``encode``
# path special-cases it (see :func:`encode_known_face`); the full code set is
# documented in TESTING.md and produced by model-server/model.py (a test pins
# the values together).
ERROR_CODE_INVALID_IMAGE = "invalid_image"


class FaceServiceError(Exception):
    """Base class for all face service failures."""


class FaceServiceUnavailable(FaceServiceError):
    """The service could not be reached, timed out, or returned a bad response.

    Retryable in principle — the caller should leave work pending rather than
    marking it permanently failed.
    """


class FaceServiceModelMismatch(FaceServiceUnavailable):
    """The service reports an embedding model or width this client was not built for.

    Still retryable (a rolling deploy may be in progress), but unlike a plain
    outage it will not fix itself by waiting, so startup checks treat it as fatal.
    """


class FaceServiceRejected(FaceServiceError):
    """The service processed the request but refused this specific image.

    Not retryable: the image is malformed, oversized, or the bounding box is
    unusable. The caller should record a terminal error for this item.

    ``code`` is the service's machine-readable ``error_code`` when present.
    :data:`ERROR_CODE_INVALID_IMAGE` means the image itself can never be
    embedded; callers that accept user-drawn boxes re-raise that and treat every
    other rejection as an unusable region (retry with a different box).
    """

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DetectedFace:
    """One face: CSS-order bounding box, its embedding, and how it was found.

    ``score`` is the detector's confidence (``None`` for user-drawn boxes whose
    face was not located). ``aligned`` is False when a drawn box was embedded
    by plain resize instead of landmark alignment.
    """

    top: int
    right: int
    bottom: int
    left: int
    encoding: bytes = field(repr=False)
    score: float | None = None
    aligned: bool = True

    @property
    def location(self) -> tuple[int, int, int, int]:
        """Return the box in CSS order: ``(top, right, bottom, left)``."""
        return (self.top, self.right, self.bottom, self.left)


@dataclass(frozen=True)
class DetectionResult:
    """Detection output for a single image.

    ``model_id`` names the embedding space; it is stored with every face so
    vectors from different models are never compared. ``dim`` is the embedding
    length.
    """

    width: int
    height: int
    faces: list[DetectedFace]
    model_id: str
    dim: int

    @property
    def locations(self) -> list[tuple[int, int, int, int]]:
        return [face.location for face in self.faces]

    @property
    def encodings(self) -> list[bytes]:
        return [face.encoding for face in self.faces]

    def __len__(self) -> int:
        return len(self.faces)


_session: requests.Session | None = None
_probe_session: requests.Session | None = None
_session_lock = threading.Lock()


def _build_session(*, retry: bool = True) -> requests.Session:
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if FACE_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {FACE_SERVICE_TOKEN}"
    session.headers.update(headers)

    # Detection is idempotent, so retrying POSTs is safe. Backoff is short:
    # the caller has its own outer poll loop and would rather fail fast and
    # leave images pending than block a worker thread for minutes.
    retries = Retry(
        total=3 if retry else 0,
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


def get_probe_session() -> requests.Session:
    """Return the shared session for readiness probes: same headers, no retries.

    A probe must answer "is it up right now" quickly. With the retrying session an
    unreachable service made ``/health`` (and the deploy preflight polling it) take
    about 12 s per call instead of failing at once.
    """
    global _probe_session
    if _probe_session is None:
        with _session_lock:
            if _probe_session is None:
                _probe_session = _build_session(retry=False)
    return _probe_session


def reset_session() -> None:
    """Drop the cached sessions. Used by tests and after config changes."""
    global _session, _probe_session
    with _session_lock:
        for cached in (_session, _probe_session):
            if cached is not None:
                cached.close()
        _session = None
        _probe_session = None


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
        session = get_probe_session()
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


def _decode_encoding(raw: Any, dim: int) -> bytes:
    """Base64-decode one embedding and enforce the exact ``dim * 4`` byte width."""
    if not isinstance(raw, str):
        raise FaceServiceUnavailable("face encoding is missing or not a string")
    try:
        encoding = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceServiceUnavailable(f"face encoding is not valid base64: {exc}") from exc
    expected = dim * BYTES_PER_COMPONENT
    if len(encoding) != expected:
        raise FaceServiceUnavailable(f"face encoding is {len(encoding)} bytes, expected {expected}")
    return encoding


def _parse_model(prediction: dict[str, Any]) -> tuple[str, int]:
    """Read and validate ``model_id`` and ``dim`` from an ``ok`` prediction.

    A server that predates the embedding switch sends neither field (and
    1024-byte encodings); a differently configured one sends another model or
    width. Both must fail here, as retryable, rather than reach the database.
    """
    model_id = prediction.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise FaceServiceUnavailable("prediction is missing a non-empty 'model_id'")
    if model_id != EXPECTED_MODEL_ID:
        raise FaceServiceModelMismatch(f"face service model_id is {model_id!r}, expected {EXPECTED_MODEL_ID!r}")
    dim = prediction.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim != EMBEDDING_DIM:
        raise FaceServiceModelMismatch(f"face service embedding dim is {dim!r}, expected {EMBEDDING_DIM}")
    return model_id, dim


def _parse_prediction(prediction: dict[str, Any]) -> DetectionResult:
    """Convert one prediction dict into a DetectionResult.

    Raises FaceServiceRejected if the server reported a per-image error, or
    FaceServiceUnavailable if the payload is structurally wrong.
    """
    if not isinstance(prediction, dict):
        raise FaceServiceUnavailable("each prediction must be an object")
    status = prediction.get("status")
    if status == "error":
        code = prediction.get("error_code")
        raise FaceServiceRejected(
            str(prediction.get("error", "unknown face service error")),
            code=code if isinstance(code, str) else None,
        )
    if status != "ok":
        raise FaceServiceUnavailable(f"unexpected prediction status {status!r}")

    model_id, dim = _parse_model(prediction)

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
        score = raw_face.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise FaceServiceUnavailable("face 'score' must be a number or null")
        aligned = raw_face.get("aligned", True)
        if not isinstance(aligned, bool):
            raise FaceServiceUnavailable("face 'aligned' must be a boolean")
        try:
            faces.append(
                DetectedFace(
                    top=int(box["top"]),
                    right=int(box["right"]),
                    bottom=int(box["bottom"]),
                    left=int(box["left"]),
                    encoding=_decode_encoding(raw_face.get("encoding"), dim),
                    score=None if score is None else float(score),
                    aligned=aligned,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FaceServiceUnavailable(f"malformed face box: {exc}") from exc

    try:
        width = int(prediction.get("width", 0))
        height = int(prediction.get("height", 0))
    except (TypeError, ValueError) as exc:
        raise FaceServiceUnavailable(f"malformed image dimensions: {exc}") from exc

    return DetectionResult(width=width, height=height, faces=faces, model_id=model_id, dim=dim)


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


def _check_image_size(image_bytes: bytes) -> None:
    if len(image_bytes) > FACE_SERVICE_MAX_IMAGE_BYTES:
        raise FaceServiceRejected(
            f"image is {len(image_bytes)} bytes, limit is {FACE_SERVICE_MAX_IMAGE_BYTES}",
            code=ERROR_CODE_INVALID_IMAGE,
        )


def detect_faces_batch(items: Sequence[tuple[str, bytes]]) -> dict[str, DetectionResult | FaceServiceError]:
    """Detect faces in several images, splitting them across batched requests.

    ``items`` is a sequence of ``(key, image_bytes)``. The returned mapping has
    one entry per key: either a :class:`DetectionResult` or the
    :class:`FaceServiceError` explaining why that image failed. Transport
    failures fail every key in the affected batch but leave other batches
    intact, so one flaky request does not discard a whole run's work.
    """
    results: dict[str, DetectionResult | FaceServiceError] = {}
    sendable = []
    for key, image_bytes in items:
        try:
            _check_image_size(image_bytes)
        except FaceServiceRejected as exc:
            results[key] = exc
        else:
            sendable.append((key, image_bytes))

    for batch in _chunk(sendable):
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
    _check_image_size(image_bytes)
    predictions = _post([{"id": "0", "task": "detect", "image": base64.b64encode(image_bytes).decode("ascii")}])
    return _parse_prediction(predictions[0])


class KnownFaceEncoding(NamedTuple):
    """Result of :func:`encode_known_face`; unpacks as ``(encoding, model_id, aligned)``."""

    encoding: bytes
    model_id: str
    aligned: bool


def encode_known_face(
    image_bytes: bytes,
    bbox: tuple[int, int, int, int],
    box_space: tuple[int, int] | None = None,
) -> KnownFaceEncoding | None:
    """Compute the embedding for one known bounding box.

    ``bbox`` is ``(top, right, bottom, left)``. ``box_space`` is the
    ``(width, height)`` of the image the box was measured on (the stored
    ``images.detection_width/height``); the service scales the box to the image
    it decodes, so the bytes sent may have a different size. Without it the box
    is taken to be in pixels of ``image_bytes``.

    Returns ``(encoding, model_id, aligned)``, or ``None`` if the service could
    not produce one for that region (typically because the box is too small or
    lies outside the image) — which callers surface to the user as "try drawing
    a slightly larger box". Raises :class:`FaceServiceRejected` when the image
    itself is unusable (over the size limit, or the service cannot decode it):
    no box the user draws can help there. ``aligned`` is False when no face
    could be located inside the box and it was embedded by plain resize.
    """
    _check_image_size(image_bytes)
    top, right, bottom, left = bbox
    instance: dict[str, Any] = {
        "id": "0",
        "task": "encode",
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "boxes": [{"top": top, "right": right, "bottom": bottom, "left": left}],
    }
    if box_space is not None:
        instance["box_space"] = {"width": box_space[0], "height": box_space[1]}
    predictions = _post([instance])
    try:
        result = _parse_prediction(predictions[0])
    except FaceServiceRejected as exc:
        if exc.code == ERROR_CODE_INVALID_IMAGE:
            raise
        logger.info("Face service rejected manual bbox %s: %s", bbox, exc)
        return None
    if not result.faces:
        return None
    face = result.faces[0]
    return KnownFaceEncoding(face.encoding, result.model_id, face.aligned)


# A 1x1 PNG, so the startup probe needs no image library.
_PROBE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGOoqKgAAALUAWneSzH7AAAAAElFTkSuQmCC"
)


def check_service_model() -> tuple[str, int]:
    """Ask the service which embedding model it runs, via one tiny real request.

    Returns ``(model_id, dim)`` as reported, after the same validation every
    prediction gets. Raises :class:`FaceServiceError` if the service cannot be
    reached or reports a model other than ``EXPECTED_MODEL_ID`` or the wrong
    width — callers use this at startup to refuse to run against the wrong model.
    """
    predictions = _post([{"id": "probe", "task": "detect", "image": base64.b64encode(_PROBE_PNG).decode("ascii")}])
    result = _parse_prediction(predictions[0])
    return result.model_id, result.dim
