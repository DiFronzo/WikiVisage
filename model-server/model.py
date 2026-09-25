"""WikiVisage face detection + encoding model server (KServe V1 protocol).

The only place dlib runs. ``app.py`` and ``worker.py`` reach it via
``face_client.py`` and ship without any ML dependency. See TESTING.md to build
and run it. In production it runs on Toolforge as its own tool, published at
``https://<tool>.toolforge.org`` behind ``guard.RequestGuard``.

``POST /v1/models/<name>:predict``::

    {"instances": [
      {"id": "42", "task": "detect", "image": "<base64 image bytes>"},
      {"id": "43", "task": "encode", "image": "<base64 image bytes>",
       "boxes": [{"top": 10, "right": 90, "bottom": 90, "left": 10}]}
    ]}

    {"predictions": [
      {"id": "42", "status": "ok", "width": 500, "height": 375,
       "faces": [{"box": {...}, "encoding": "<base64 1024 raw float64 bytes>"}]},
      {"id": "43", "status": "error", "error": "..."}
    ]}

``predictions`` always matches ``instances`` in length and order; one bad image
yields a ``status: "error"`` entry without affecting its siblings.

Design constraints worth knowing before editing:

* **dlib is not thread-safe.** The detector, pose predictor and encoder are
  shared objects; driving them from several threads races on C++ state and
  segfaults the whole process (SIGSEGV, reproduced with a single batch of 8).
  Every dlib call is therefore serialised under ``_dlib_lock``. Threads still
  overlap base64 and image decoding, but detection runs one image at a time —
  scale with replicas.
* **Images arrive base64, never as URLs.** The server makes no outbound network
  calls, which removes SSRF surface and keeps it deployable behind default-deny
  egress.
* **Encoding compatibility is load-bearing.** Every vector in the database was
  produced by ``face_recognition`` 1.3.0. ``_Dlib`` reproduces exactly the dlib
  calls it made (HOG detector with one upsample, the *5-point* landmark model,
  one jitter) without depending on the package, whose ``dlib`` requirement
  cannot be installed by a buildpack. Changing any of those, the ``dlib-bin``
  version, or the ``face_recognition_models`` weights requires a full re-encode.
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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import kserve
import numpy as np
from guard import MIN_TOKEN_LENGTH, RequestGuard
from kserve.errors import InvalidInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wikivisage-model-server")

DEFAULT_MODEL_NAME = "wikivisage"

# Callers send Commons thumbnails (?width=500), typically well under 300 KB.
# These ceilings are generous but bound worst-case memory per request.
MAX_IMAGE_BYTES = int(os.environ.get("WIKIVISAGE_MAX_IMAGE_BYTES", 20 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("WIKIVISAGE_MAX_IMAGE_PIXELS", 100_000_000))
MAX_BATCH_SIZE = int(os.environ.get("WIKIVISAGE_MAX_BATCH_SIZE", 32))
MAX_BOXES_PER_INSTANCE = int(os.environ.get("WIKIVISAGE_MAX_BOXES", 64))

# Threads handling instances. dlib itself is serialised under _dlib_lock (it is
# not thread-safe), so extra threads only overlap base64 and image decoding.
# Scale throughput with more replicas, not more threads.
FACE_WORKERS = int(os.environ.get("WIKIVISAGE_FACE_WORKERS", 1))

# Largest request body accepted. face_client caps a batch at 24 MiB of raw
# image bytes, which base64 inflates to 32 MiB plus a little JSON.
MAX_REQUEST_BYTES = int(os.environ.get("WIKIVISAGE_MAX_REQUEST_BYTES", 40 * 1024 * 1024))

# Shared secret with face_client. Required whenever the service is reachable
# from anywhere but localhost; see main() for the Toolforge fail-closed check.
FACE_SERVICE_TOKEN = os.environ.get("WIKIVISAGE_FACE_SERVICE_TOKEN", "")

# Mirrors app.py's _validate_bbox.
MAX_BBOX_PX = 10000
MIN_BBOX_AREA = 100

ENCODING_BYTES = 128 * 8

_TASK_DETECT = "detect"
_TASK_ENCODE = "encode"
_VALID_TASKS = frozenset({_TASK_DETECT, _TASK_ENCODE})


class ImageRejected(ValueError):
    """Raised for per-instance input problems that should not fail the batch."""


class _Dlib:
    """The dlib calls ``face_recognition`` 1.3.0 made, without the package.

    Box order is ``(top, right, bottom, left)`` throughout, as in the original.
    """

    def __init__(self) -> None:
        import dlib
        import face_recognition_models as weights

        self._rectangle = dlib.rectangle
        self._detector = dlib.get_frontal_face_detector()
        self._pose_5_point = dlib.shape_predictor(weights.pose_predictor_five_point_model_location())
        self._encoder = dlib.face_recognition_model_v1(weights.face_recognition_model_location())

    def face_locations(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """``face_recognition.face_locations(image, 1, "hog")``."""
        height, width = image.shape[:2]
        return [
            (max(rect.top(), 0), min(rect.right(), width), min(rect.bottom(), height), max(rect.left(), 0))
            for rect in self._detector(image, 1)
        ]

    def face_encodings(self, image: np.ndarray, locations: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        """``face_recognition.face_encodings(image, locations, num_jitters=1, model="small")``."""
        encodings = []
        for top, right, bottom, left in locations:
            landmarks = self._pose_5_point(image, self._rectangle(left, top, right, bottom))
            encodings.append(np.array(self._encoder.compute_face_descriptor(image, landmarks, 1)))
        return encodings


def _decode_image_field(raw: Any) -> bytes:
    """Base64-decode an instance's ``image`` field with strict validation."""
    if not isinstance(raw, str) or not raw:
        raise ImageRejected("'image' must be a non-empty base64 string")
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRejected(f"'image' is not valid base64: {exc}") from exc
    if not image_bytes:
        raise ImageRejected("'image' decoded to zero bytes")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageRejected(f"image is {len(image_bytes)} bytes, limit is {MAX_IMAGE_BYTES}")
    return image_bytes


def _parse_box(raw: Any) -> tuple[int, int, int, int]:
    """Normalise one bounding box into a ``(top, right, bottom, left)`` tuple.

    Accepts either the object form ``{"top": .., "right": .., ...}`` or the
    positional 4-element list form, matching how callers naturally serialise
    the ``face_recognition`` CSS-order tuple.
    """
    values: tuple[Any, ...]
    if isinstance(raw, dict):
        try:
            values = (raw["top"], raw["right"], raw["bottom"], raw["left"])
        except KeyError as exc:
            raise ImageRejected(f"box is missing key {exc}") from exc
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = tuple(raw)
    else:
        raise ImageRejected("box must be an object with top/right/bottom/left, or a 4-element list")

    try:
        top, right, bottom, left = (int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ImageRejected("box coordinates must be integers") from exc

    if min(top, right, bottom, left) < 0 or max(top, right, bottom, left) > MAX_BBOX_PX:
        raise ImageRejected(f"box coordinates must be within 0..{MAX_BBOX_PX}")
    if right <= left or bottom <= top:
        raise ImageRejected("box must have right > left and bottom > top")
    if (right - left) * (bottom - top) < MIN_BBOX_AREA:
        raise ImageRejected(f"box area must be at least {MIN_BBOX_AREA} pixels")
    return top, right, bottom, left


def _encode_vector(vector: np.ndarray) -> str:
    """Serialise a 128D encoding to base64 of its raw little-endian float64 bytes.

    The client writes these bytes straight into the ``faces.encoding`` BLOB, so
    the wire *format* must stay exactly what ``numpy.ndarray.tobytes()``
    produced when detection ran in-process: 1024 raw bytes, little-endian
    float64. ``astype("<f8")`` pins byte order so a big-endian host would not
    silently emit vectors the client would misread.

    Note this guarantees format, not bit-exact values. dlib's ResNet runs in
    float32, so the same image encoded on different CPU/BLAS combinations
    differs by ~1e-7 per component (measured: 7.7e-07 L2, against a 0.6
    classification threshold). That drift is ~1e-6 of the decision margin and
    is not specific to this service — it exists between any two machines.
    """
    raw = np.ascontiguousarray(vector, dtype=np.float64).astype("<f8", copy=False).tobytes()
    if len(raw) != ENCODING_BYTES:
        raise ImageRejected(f"unexpected encoding length {len(raw)}, expected {ENCODING_BYTES}")
    return base64.b64encode(raw).decode("ascii")


class WikiVisageModel(kserve.Model):
    """KServe predictor exposing dlib HOG face detection and 128D encoding."""

    def __init__(self, name: str = DEFAULT_MODEL_NAME, face_workers: int = FACE_WORKERS):
        super().__init__(name)
        self.name = name
        self._face_workers = max(1, face_workers)
        self._executor: ThreadPoolExecutor | None = None
        # Bounds how many requests can queue up against dlib at once, so a burst
        # of concurrent callers cannot grow the executor backlog without limit.
        self._semaphore: asyncio.Semaphore | None = None
        # Serialises dlib. Non-negotiable: see the module docstring.
        self._dlib_lock = threading.Lock()
        self._dlib: _Dlib | None = None
        self.ready = False

    def load(self) -> bool:
        """Load the dlib models, warm them up, and mark the server ready.

        The dlib weights ship inside the ``face_recognition_models`` wheel, so
        there is nothing to download or mount — importing the package is the
        load step. A warm-up detection on a synthetic image forces lazy model
        deserialisation to happen now rather than on the first real request.

        ``self.ready`` must end up True: ``ModelServer.start()`` raises
        ``NoModelReady`` and exits if no registered model reports ready.
        """
        from PIL import Image as PILImage
        from PIL import ImageFile

        # Treat decompression bombs as hard errors instead of warnings.
        PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        # face_recognition 1.3.0 set this on import. Keeping it means images
        # that decoded before (truncated JPEGs are common on Commons) still do.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self._dlib = _Dlib()
        self._executor = ThreadPoolExecutor(
            max_workers=self._face_workers,
            thread_name_prefix="dlib",
        )
        self._semaphore = asyncio.Semaphore(self._face_workers * 2)

        warmup = np.zeros((64, 64, 3), dtype=np.uint8)
        self._dlib.face_encodings(warmup, self._dlib.face_locations(warmup))

        self.ready = True
        logger.info("Model '%s' loaded (face_workers=%d)", self.name, self._face_workers)
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
            raise ImageRejected(f"image rejected as decompression bomb: {exc}") from exc
        except Exception as exc:
            raise ImageRejected(f"could not read image: {exc}") from exc

        if width * height > MAX_IMAGE_PIXELS:
            raise ImageRejected(f"image is {width}x{height} ({width * height:,} px), limit is {MAX_IMAGE_PIXELS:,}")

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as image:
                return np.array(image.convert("RGB"))
        except PILImage.DecompressionBombError as exc:
            raise ImageRejected(f"image rejected as decompression bomb: {exc}") from exc
        except Exception as exc:
            raise ImageRejected(f"could not decode image: {exc}") from exc

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
            if task == _TASK_ENCODE:
                raw_boxes = instance.get("boxes")
                if not isinstance(raw_boxes, list) or not raw_boxes:
                    raise ImageRejected("task 'encode' requires a non-empty 'boxes' list")
                if len(raw_boxes) > MAX_BOXES_PER_INSTANCE:
                    raise ImageRejected(f"too many boxes ({len(raw_boxes)}), limit is {MAX_BOXES_PER_INSTANCE}")
                boxes = [_parse_box(box) for box in raw_boxes]

            image_bytes = _decode_image_field(instance.get("image"))
            image_data = self._load_image(image_bytes)
            height, width = image_data.shape[:2]

            if self._dlib is None:
                raise ImageRejected("model is not loaded")

            # Decoding above is safe to run concurrently; dlib is not. Hold the
            # lock across both calls so no two threads are ever inside dlib.
            with self._dlib_lock:
                locations = boxes if boxes is not None else self._dlib.face_locations(image_data)
                encodings = self._dlib.face_encodings(image_data, locations)

            faces = [
                {
                    "box": {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)},
                    "encoding": _encode_vector(vector),
                }
                for (top, right, bottom, left), vector in zip(locations, encodings)
            ]

            return {
                "id": instance_id,
                "status": "ok",
                "width": int(width),
                "height": int(height),
                "faces": faces,
            }
        except ImageRejected as exc:
            return {"id": instance_id, "status": "error", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — a bad image must not kill the batch
            logger.exception("Face processing failed for instance id=%r", instance_id)
            return {"id": instance_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    async def predict(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Handle a V1 predict request.

        Only parsing happens on the event loop; every dlib call is dispatched to
        the thread pool so concurrent requests are not serialised behind
        blocking C++ work.
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
        help="Threads decoding images; dlib itself always runs one at a time (default: 1).",
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
    model.load()
    # gRPC would listen on a second port that bypasses RequestGuard, and
    # nothing here uses it.
    kserve.ModelServer(enable_grpc=False).start([model])


if __name__ == "__main__":
    main()
