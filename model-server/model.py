"""WikiVisage face detection + embedding model server (KServe V1 protocol).

The only place ML runs. ``app.py`` and ``worker.py`` reach it via
``face_client.py`` and ship without any ML dependency. See TESTING.md to build
and run it. In production it runs on Toolforge as its own tool, published at
``https://<tool>.toolforge.org`` behind ``guard.RequestGuard``.

The models (SCRFD-500M detector, ArcFace MobileFaceNet embedder, on ONNX
Runtime) live in ``backend.py``; this module is the KServe envelope around them.

``POST /v1/models/<name>:predict``::

    {"instances": [
      {"id": "42", "task": "detect", "image": "<base64 image bytes>"},
      {"id": "43", "task": "encode", "image": "<base64 image bytes>",
       "boxes": [{"top": 10, "right": 90, "bottom": 90, "left": 10}],
       "box_space": {"width": 960, "height": 720}}
    ]}

    {"predictions": [
      {"id": "42", "status": "ok", "width": 960, "height": 720,
       "model_id": "scrfd500m+arcface_mbf", "dim": 512,
       "faces": [{"box": {...}, "score": 0.91,
                  "encoding": "<base64 of dim little-endian float32 values>"}]},
      {"id": "43", "status": "ok", "width": 1024, "height": 768,
       "model_id": "scrfd500m+arcface_mbf", "dim": 512,
       "faces": [{"box": {...}, "score": 0.88, "aligned": true, "encoding": "..."}]},
      {"id": "44", "status": "error", "error": "...", "error_code": "invalid_image"}
    ]}

``predictions`` always matches ``instances`` in length and order; one bad image
yields a ``status: "error"`` entry (with a machine-readable ``error_code``)
without affecting its siblings.

``box_space`` (``encode`` only, optional) is the pixel size of the image the
caller measured its boxes in. The server scales the boxes to the image it
actually decoded, so callers need not know the size of the bytes they sent.

Design constraints worth knowing before editing:

* **ONNX Runtime sessions are thread-safe.** One session per model is shared by
  all executor threads; ``run()`` releases the GIL, so threads overlap real
  inference. Throughput scales with ``FACE_WORKERS`` (set it to the vCPU count).
  There is deliberately no lock and no multiprocessing.
* **Images arrive base64, never as URLs.** The server makes no outbound network
  calls, which removes SSRF surface and keeps it deployable behind default-deny
  egress.
* **The weights are verified at load.** ``backend.verify_weights`` pins their
  SHA-256; a mismatch stops the process before it can report ready, so a bad
  deploy fails the platform readiness wait instead of serving wrong embeddings.
* **Embeddings are model-specific.** Anything stored by the database is only
  comparable with vectors from the same ``model_id``; changing the weights, the
  alignment template or the preprocessing means re-embedding everything.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import kserve
import numpy as np
from backend import (
    BackendConfig,
    FaceBackend,
    FaceBackendError,
    ScrfdArcFaceBackend,
    encode_boxes,
)
from guard import MIN_TOKEN_LENGTH, RequestGuard
from kserve.errors import InvalidInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wikivisage-model-server")

DEFAULT_MODEL_NAME = "wikivisage"

# Callers send Commons thumbnails (?width=960), typically a few hundred KB.
# These ceilings are generous but bound worst-case memory per request.
MAX_IMAGE_BYTES = int(os.environ.get("WIKIVISAGE_MAX_IMAGE_BYTES", 20 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("WIKIVISAGE_MAX_IMAGE_PIXELS", 100_000_000))
MAX_BATCH_SIZE = int(os.environ.get("WIKIVISAGE_MAX_BATCH_SIZE", 32))
MAX_BOXES_PER_INSTANCE = int(os.environ.get("WIKIVISAGE_MAX_BOXES", 64))

# Threads handling instances. Each runs ONNX inference on the shared sessions
# (thread-safe, one intra-op thread per call), so throughput scales with this up
# to the CPU count. The Toolforge job gets 2 vCPUs (--cpu 2).
FACE_WORKERS = int(os.environ.get("WIKIVISAGE_FACE_WORKERS", 2))

# Largest request body accepted. face_client caps a batch at 24 MiB of raw
# image bytes, which base64 inflates to 32 MiB plus a little JSON.
MAX_REQUEST_BYTES = int(os.environ.get("WIKIVISAGE_MAX_REQUEST_BYTES", 40 * 1024 * 1024))

# Shared secret with face_client. Required whenever the service is reachable
# from anywhere but localhost; see main() for the Toolforge fail-closed check.
FACE_SERVICE_TOKEN = os.environ.get("WIKIVISAGE_FACE_SERVICE_TOKEN", "")

# Mirrors app.py's _validate_bbox.
MAX_BBOX_PX = 10000
MIN_BBOX_AREA = 100

# Largest image side a box_space may claim; far above any real image.
MAX_BOX_SPACE_PX = 65535

_TASK_DETECT = "detect"
_TASK_ENCODE = "encode"
_VALID_TASKS = frozenset({_TASK_DETECT, _TASK_ENCODE})

# Machine-readable codes for per-instance error entries ("error_code").
# face_client only special-cases ERROR_CODE_INVALID_IMAGE: an unusable image can
# never be embedded, while every other rejection means the request (usually the
# box) should change. Keep in step with face_client's constants (a test pins
# the values together).
ERROR_CODE_INVALID_IMAGE = "invalid_image"
ERROR_CODE_INVALID_BOX = "invalid_box"
ERROR_CODE_INVALID_REQUEST = "invalid_request"
ERROR_CODE_SERVICE_ERROR = "service_error"


class ImageRejected(ValueError):
    """Raised for per-instance input problems that should not fail the batch.

    ``code`` classifies the problem for the caller (sent as ``error_code`` on
    the wire); it defaults to ``invalid_request`` so a raise site that forgets a
    code is never mistaken for an unusable image.
    """

    def __init__(self, message: str, code: str = ERROR_CODE_INVALID_REQUEST):
        super().__init__(message)
        self.code = code


def _decode_image_field(raw: Any) -> bytes:
    """Base64-decode an instance's ``image`` field with strict validation."""
    if not isinstance(raw, str) or not raw:
        raise ImageRejected("'image' must be a non-empty base64 string", code=ERROR_CODE_INVALID_IMAGE)
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRejected(f"'image' is not valid base64: {exc}", code=ERROR_CODE_INVALID_IMAGE) from exc
    if not image_bytes:
        raise ImageRejected("'image' decoded to zero bytes", code=ERROR_CODE_INVALID_IMAGE)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageRejected(
            f"image is {len(image_bytes)} bytes, limit is {MAX_IMAGE_BYTES}", code=ERROR_CODE_INVALID_IMAGE
        )
    return image_bytes


def _parse_box(raw: Any) -> tuple[int, int, int, int]:
    """Normalise one bounding box into a ``(top, right, bottom, left)`` tuple.

    Accepts either the object form ``{"top": .., "right": .., ...}`` or the
    positional 4-element list form, matching how callers naturally serialise
    a CSS-order tuple.
    """
    values: tuple[Any, ...]
    if isinstance(raw, dict):
        try:
            values = (raw["top"], raw["right"], raw["bottom"], raw["left"])
        except KeyError as exc:
            raise ImageRejected(f"box is missing key {exc}", code=ERROR_CODE_INVALID_BOX) from exc
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = tuple(raw)
    else:
        raise ImageRejected(
            "box must be an object with top/right/bottom/left, or a 4-element list", code=ERROR_CODE_INVALID_BOX
        )

    try:
        top, right, bottom, left = (int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ImageRejected("box coordinates must be integers", code=ERROR_CODE_INVALID_BOX) from exc

    if min(top, right, bottom, left) < 0 or max(top, right, bottom, left) > MAX_BBOX_PX:
        raise ImageRejected(f"box coordinates must be within 0..{MAX_BBOX_PX}", code=ERROR_CODE_INVALID_BOX)
    if right <= left or bottom <= top:
        raise ImageRejected("box must have right > left and bottom > top", code=ERROR_CODE_INVALID_BOX)
    if (right - left) * (bottom - top) < MIN_BBOX_AREA:
        raise ImageRejected(f"box area must be at least {MIN_BBOX_AREA} pixels", code=ERROR_CODE_INVALID_BOX)
    return top, right, bottom, left


def _parse_box_space(raw: Any) -> tuple[int, int] | None:
    """Parse the optional ``box_space`` ``{"width": .., "height": ..}`` of an ``encode`` instance."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ImageRejected("'box_space' must be an object with width and height", code=ERROR_CODE_INVALID_BOX)
    try:
        width, height = int(raw["width"]), int(raw["height"])
    except KeyError as exc:
        raise ImageRejected(f"'box_space' is missing key {exc}", code=ERROR_CODE_INVALID_BOX) from exc
    except (TypeError, ValueError) as exc:
        raise ImageRejected("'box_space' width and height must be integers", code=ERROR_CODE_INVALID_BOX) from exc
    if not (0 < width <= MAX_BOX_SPACE_PX and 0 < height <= MAX_BOX_SPACE_PX):
        raise ImageRejected(
            f"'box_space' width and height must be within 1..{MAX_BOX_SPACE_PX}", code=ERROR_CODE_INVALID_BOX
        )
    return width, height


def _fit_box(
    box: tuple[int, int, int, int],
    box_space: tuple[int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Map a ``(top, right, bottom, left)`` box into the decoded image and clip it.

    ``box_space`` is the size of the image the caller drew the box on. Without
    it the box is taken to be in decoded-image pixels already. The result is
    clipped to the image; a box that ends up empty is rejected.
    """
    top, right, bottom, left = box
    if box_space is not None:
        space_w, space_h = box_space
        scale_x, scale_y = width / space_w, height / space_h
        top, bottom = round(top * scale_y), round(bottom * scale_y)
        left, right = round(left * scale_x), round(right * scale_x)
    top, left = max(0, top), max(0, left)
    bottom, right = min(height, bottom), min(width, right)
    if right <= left or bottom <= top:
        raise ImageRejected(f"box {box} lies outside the {width}x{height} image", code=ERROR_CODE_INVALID_BOX)
    return top, right, bottom, left


def _box_json(box: tuple[int, int, int, int]) -> dict[str, int]:
    top, right, bottom, left = box
    return {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)}


def _encode_vector(vector: np.ndarray, dim: int) -> str:
    """Serialise one embedding to base64 of its raw little-endian ``float32`` bytes.

    The client writes these bytes straight into the ``faces.encoding`` BLOB, so
    the wire format is exactly ``dim * 4`` bytes. ``astype("<f4")`` pins byte
    order so a big-endian host would not silently emit vectors the client would
    misread.
    """
    raw = np.ascontiguousarray(vector, dtype=np.float32).astype("<f4", copy=False).tobytes()
    if len(raw) != dim * 4:
        raise ImageRejected(
            f"unexpected embedding length {len(raw)}, expected {dim * 4}", code=ERROR_CODE_SERVICE_ERROR
        )
    return base64.b64encode(raw).decode("ascii")


class WikiVisageModel(kserve.Model):
    """KServe predictor exposing SCRFD face detection and ArcFace embedding."""

    def __init__(
        self,
        name: str = DEFAULT_MODEL_NAME,
        face_workers: int = FACE_WORKERS,
        backend: FaceBackend | None = None,
    ):
        super().__init__(name)
        self.name = name
        self._face_workers = max(1, face_workers)
        self._executor: ThreadPoolExecutor | None = None
        # Bounds how many requests can queue up for the executor at once, so a
        # burst of concurrent callers cannot grow its backlog without limit.
        self._semaphore: asyncio.Semaphore | None = None
        # Injectable so tests can run the envelope without the ONNX weights.
        self._backend: FaceBackend | None = backend
        self.ready = False

    def load(self) -> bool:
        """Verify and load the ONNX models, warm them up, and mark the server ready.

        Raises :class:`backend.WeightsError` if a weights file is missing or does
        not match its pinned SHA-256, so the process never becomes ready on
        unknown weights. ``self.ready`` must end up True: ``ModelServer.start()``
        raises ``NoModelReady`` and exits if no registered model reports ready.
        """
        from PIL import Image as PILImage
        from PIL import ImageFile

        # Treat decompression bombs as hard errors instead of warnings.
        PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        # Truncated JPEGs are common on Commons. Decoding them (rather than
        # rejecting) keeps images processable that always were.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self._backend is None:
            self._backend = ScrfdArcFaceBackend.load(BackendConfig.from_env())
        self._executor = ThreadPoolExecutor(
            max_workers=self._face_workers,
            thread_name_prefix="face",
        )
        self._semaphore = asyncio.Semaphore(self._face_workers * 2)

        self.ready = True
        logger.info(
            "Model '%s' loaded (model_id=%s, dim=%d, face_workers=%d)",
            self.name,
            self._backend.model_id,
            self._backend.dim,
            self._face_workers,
        )
        return self.ready

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        super().stop()

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image as PILImage

        try:
            probe = PILImage.open(io.BytesIO(image_bytes))
            width, height = probe.size
        except PILImage.DecompressionBombError as exc:
            raise ImageRejected(f"image rejected as decompression bomb: {exc}", code=ERROR_CODE_INVALID_IMAGE) from exc
        except Exception as exc:
            raise ImageRejected(f"could not read image: {exc}", code=ERROR_CODE_INVALID_IMAGE) from exc

        if width * height > MAX_IMAGE_PIXELS:
            raise ImageRejected(
                f"image is {width}x{height} ({width * height:,} px), limit is {MAX_IMAGE_PIXELS:,}",
                code=ERROR_CODE_INVALID_IMAGE,
            )

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as image:
                return np.array(image.convert("RGB"))
        except PILImage.DecompressionBombError as exc:
            raise ImageRejected(f"image rejected as decompression bomb: {exc}", code=ERROR_CODE_INVALID_IMAGE) from exc
        except Exception as exc:
            raise ImageRejected(f"could not decode image: {exc}", code=ERROR_CODE_INVALID_IMAGE) from exc

    def _run_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Detect or encode faces for one instance. Never raises."""
        instance_id = instance.get("id")
        try:
            task = instance.get("task", _TASK_DETECT)
            if task not in _VALID_TASKS:
                raise ImageRejected(f"unknown task {task!r}, expected one of {sorted(_VALID_TASKS)}")

            # Validate the cheap structural fields before decoding the image, so
            # a malformed request is rejected without paying for a full decode.
            boxes: list[tuple[int, int, int, int]] | None = None
            box_space: tuple[int, int] | None = None
            if task == _TASK_ENCODE:
                raw_boxes = instance.get("boxes")
                if not isinstance(raw_boxes, list) or not raw_boxes:
                    raise ImageRejected("task 'encode' requires a non-empty 'boxes' list")
                if len(raw_boxes) > MAX_BOXES_PER_INSTANCE:
                    raise ImageRejected(f"too many boxes ({len(raw_boxes)}), limit is {MAX_BOXES_PER_INSTANCE}")
                boxes = [_parse_box(box) for box in raw_boxes]
                box_space = _parse_box_space(instance.get("box_space"))

            image_bytes = _decode_image_field(instance.get("image"))
            image_data = self._load_image(image_bytes)
            height, width = image_data.shape[:2]

            backend = self._backend
            if backend is None:
                raise ImageRejected("model is not loaded", code=ERROR_CODE_SERVICE_ERROR)

            faces: list[dict[str, Any]]
            if boxes is None:
                detections = backend.detect(image_data)
                vectors = backend.embed(image_data, detections)
                faces = [
                    {
                        "box": _box_json(detection.box),
                        "score": round(detection.score, 4),
                        "encoding": _encode_vector(vector, backend.dim),
                    }
                    for detection, vector in zip(detections, vectors)
                ]
            else:
                fitted = [_fit_box(box, box_space, width, height) for box in boxes]
                faces = [
                    {
                        "box": _box_json(result.box),
                        "score": None if result.score is None else round(result.score, 4),
                        "aligned": result.aligned,
                        "encoding": _encode_vector(result.vector, backend.dim),
                    }
                    for result in encode_boxes(backend, image_data, fitted)
                ]

            return {
                "id": instance_id,
                "status": "ok",
                "width": int(width),
                "height": int(height),
                "model_id": backend.model_id,
                "dim": backend.dim,
                "faces": faces,
            }
        except ImageRejected as exc:
            return {"id": instance_id, "status": "error", "error": str(exc), "error_code": exc.code}
        except FaceBackendError as exc:
            logger.exception("Face backend failed for instance id=%r", instance_id)
            return {
                "id": instance_id,
                "status": "error",
                "error": f"face backend failed: {exc}",
                "error_code": ERROR_CODE_SERVICE_ERROR,
            }
        except Exception as exc:  # noqa: BLE001 — a bad image must not kill the batch
            logger.exception("Face processing failed for instance id=%r", instance_id)
            return {
                "id": instance_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "error_code": ERROR_CODE_SERVICE_ERROR,
            }

    async def predict(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Handle a V1 predict request.

        Only parsing happens on the event loop; all image decoding and inference
        is dispatched to the thread pool so concurrent requests are not
        serialised behind blocking work.
        """
        if not isinstance(payload, dict):
            raise InvalidInput("request body must be a JSON object")

        instances = payload.get("instances")
        if not isinstance(instances, list) or not instances:
            raise InvalidInput("request must contain a non-empty 'instances' list")
        if len(instances) > MAX_BATCH_SIZE:
            raise InvalidInput(f"batch of {len(instances)} exceeds limit of {MAX_BATCH_SIZE} instances")
        if not all(isinstance(item, dict) for item in instances):
            raise InvalidInput("every entry in 'instances' must be an object")
        executor, semaphore = self._executor, self._semaphore
        if executor is None or semaphore is None:
            raise InvalidInput("model is not loaded")

        loop = asyncio.get_running_loop()

        async def _dispatch(instance: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await loop.run_in_executor(executor, self._run_instance, instance)

        predictions = await asyncio.gather(*(_dispatch(item) for item in instances))
        return {"predictions": list(predictions)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--face_workers",
        type=int,
        default=FACE_WORKERS,
        help="Threads running detection and embedding concurrently (default: 2).",
    )
    return parser


def _check_token() -> None:
    if FACE_SERVICE_TOKEN and len(FACE_SERVICE_TOKEN) < MIN_TOKEN_LENGTH:
        logger.critical("WIKIVISAGE_FACE_SERVICE_TOKEN must be at least %d characters", MIN_TOKEN_LENGTH)
        sys.exit(1)
    # Toolforge sets TOOL_TOOLFORGE_API_URL in every job. There the service is
    # published on the internet, so running without a token is never intended.
    if not FACE_SERVICE_TOKEN and os.environ.get("TOOL_TOOLFORGE_API_URL"):
        logger.critical("Running on Toolforge without WIKIVISAGE_FACE_SERVICE_TOKEN; refusing to start")
        sys.exit(1)
    if not FACE_SERVICE_TOKEN:
        logger.warning("WIKIVISAGE_FACE_SERVICE_TOKEN is not set; the service accepts unauthenticated requests")


def main() -> None:
    _check_token()
    args, _unknown = _build_parser().parse_known_args()
    model_name = getattr(args, "model_name", None) or DEFAULT_MODEL_NAME

    # KServe serves one module-level FastAPI app; middleware must be added
    # before ModelServer.start() builds it.
    kserve.model_server.app.add_middleware(
        RequestGuard,
        token=FACE_SERVICE_TOKEN,
        max_body_bytes=MAX_REQUEST_BYTES,
        open_paths=[f"/v1/models/{model_name}"],
    )

    model = WikiVisageModel(model_name, face_workers=args.face_workers)
    try:
        model.load()
    except FaceBackendError as exc:
        # Includes WeightsError (missing or wrong SHA-256). Exiting here means the
        # platform readiness wait fails and no workers are pointed at this service.
        logger.critical("Face backend failed to load, refusing to start: %s", exc)
        sys.exit(1)
    # gRPC would listen on a second port that bypasses RequestGuard, and
    # nothing here uses it.
    kserve.ModelServer(enable_grpc=False).start([model])


if __name__ == "__main__":
    main()
