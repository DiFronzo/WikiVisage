"""Face detection and embedding backend for the WikiVisage face service.

SCRFD-500M (detector, 5 landmarks) plus ArcFace MobileFaceNet (512D embedding),
both run on ONNX Runtime. Everything here is synchronous, stateless per call
and thread-safe: the two ``InferenceSession`` objects are shared and
``InferenceSession.run()`` may be called from several threads at once. This
module has no KServe dependency, so it can be tested on its own.

Conventions used throughout:

* Images are ``uint8`` RGB arrays of shape ``(height, width, 3)``.
* Boxes are ``(top, right, bottom, left)`` in pixels of the image passed in.
* Embeddings are L2-normalised ``float32`` vectors of length ``EMBEDDING_DIM``,
  so cosine similarity is a plain dot product.

The weights are third-party and non-commercial; see ``weights/LICENSE-NOTICE``.
``verify_weights`` pins their SHA-256 so the service never runs on a pointer
file, a truncated download or a swapped model.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

# OpenCV spawns its own thread pool by default. Parallelism here comes from the
# caller's executor threads, so a second layer would only oversubscribe the CPU.
cv2.setNumThreads(1)

logger = logging.getLogger("wikivisage-model-server")

MODEL_ID = "scrfd500m+arcface_mbf"
EMBEDDING_DIM = 512

DETECTOR_FILE = "det_500m.onnx"
EMBEDDER_FILE = "w600k_mbf.onnx"

# Pinned SHA-256 of the two files from InsightFace's ``buffalo_sc.zip``. Kept in
# code (not read from SHA256SUMS next to the weights) so a tampered directory
# cannot vouch for itself; tests assert this agrees with the committed manifest.
WEIGHT_SHA256: dict[str, str] = {
    DETECTOR_FILE: "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    EMBEDDER_FILE: "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
}

# Canonical ArcFace 112x112 landmark template (left eye, right eye, nose,
# left mouth corner, right mouth corner).
ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
CROP_SIZE = 112

_STRIDES = (8, 16, 32)
_ANCHORS_PER_LOCATION = 2
_DETECTOR_OUTPUTS = 9  # scores, boxes and keypoints for each of the three strides
_PAD_MULTIPLE = 32

# Faces embedded per ONNX call. An image's faces are batched, but a crowd photo
# can hold hundreds; chunking bounds activation memory on the 3 GiB face job.
EMBED_CHUNK = 32

# SCRFD-500M misses very large faces: on a 960x1307 portrait whose face was 532 px
# tall it found nothing at full size, yet found it at 0.78x (faces up to ~430 px
# were fine). Tight portraits are common on Commons, so a second, half-scale pass
# runs when the working image is at least ``coarse_min_side`` on its long side.
# Only faces at least ``COARSE_MIN_FACE_PX`` (in source pixels) are taken from it:
# smaller ones are already found, more precisely, by the full-size pass.
COARSE_MIN_FACE_PX = 240

# ``encode`` task: the detector runs on the user's box grown by this fraction
# (0.5 = 1.5x the width and height, a quarter margin per side), and a detection
# is accepted only if it overlaps the user's box at least this much.
ENCODE_BOX_EXPANSION = 0.5
ENCODE_MIN_IOU = 0.3


class FaceBackendError(RuntimeError):
    """The backend could not be loaded or produced an unusable result."""


class WeightsError(FaceBackendError):
    """A weights file is missing or does not match its pinned SHA-256."""


@dataclass(frozen=True)
class BackendConfig:
    """Tunables of the SCRFD + ArcFace backend. ``from_env`` reads the service's environment."""

    weights_dir: Path
    det_max_side: int = 1280
    score_threshold: float = 0.50
    nms_threshold: float = 0.40
    min_face_px: int = 20
    max_faces: int = 300
    coarse_min_side: int = 640  # 0 disables the half-scale pass

    @classmethod
    def from_env(cls) -> BackendConfig:
        default_dir = Path(__file__).resolve().parent / "weights"
        return cls(
            weights_dir=Path(os.environ.get("WIKIVISAGE_WEIGHTS_DIR", default_dir)),
            det_max_side=int(os.environ.get("WIKIVISAGE_DET_MAX_SIDE", cls.det_max_side)),
            min_face_px=int(os.environ.get("WIKIVISAGE_MIN_FACE_PX", cls.min_face_px)),
            max_faces=int(os.environ.get("WIKIVISAGE_MAX_FACES", cls.max_faces)),
            coarse_min_side=int(os.environ.get("WIKIVISAGE_DET_COARSE_MIN_SIDE", cls.coarse_min_side)),
        )


@dataclass(frozen=True)
class Detection:
    """One detected face in pixels of the image that was searched."""

    box: tuple[int, int, int, int]  # top, right, bottom, left
    score: float
    landmarks: np.ndarray  # float32 (5, 2), (x, y) per point

    def shifted(self, dx: int, dy: int) -> Detection:
        """Return this detection translated, e.g. from crop to full-image coordinates."""
        top, right, bottom, left = self.box
        return Detection(
            box=(top + dy, right + dx, bottom + dy, left + dx),
            score=self.score,
            landmarks=self.landmarks + np.array([dx, dy], dtype=np.float32),
        )


@dataclass(frozen=True)
class EncodedBox:
    """Embedding for one user-supplied box (``encode`` task)."""

    box: tuple[int, int, int, int]
    vector: np.ndarray
    score: float | None  # detector score when the face was located, else None
    aligned: bool  # False when the box was resized without landmark alignment


class FaceBackend(Protocol):
    """What ``model.py`` needs from a detector/embedder pair.

    Kept narrow so another pair (for example YuNet + SFace) can be dropped in
    without touching the KServe envelope, the client or the database.
    """

    model_id: str
    dim: int

    def detect(self, rgb: np.ndarray, *, min_face_px: int | None = None) -> list[Detection]:
        """Find faces, best first. ``min_face_px`` overrides the configured minimum short side."""
        ...

    def embed(self, rgb: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """Aligned, L2-normalised ``float32`` embeddings, shape ``(len(detections), dim)``."""
        ...

    def embed_unaligned(self, rgb: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
        """Embeddings of plain resized box crops (no landmarks), shape ``(len(boxes), dim)``."""
        ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(weights_dir: Path) -> dict[str, str]:
    """Check both weight files against ``WEIGHT_SHA256``; return the observed hashes.

    Raises :class:`WeightsError` on the first missing file or mismatch. A Git LFS
    pointer file or a truncated download fails here with both hashes in the message.
    """
    observed: dict[str, str] = {}
    for name, expected in WEIGHT_SHA256.items():
        path = weights_dir / name
        if not path.is_file():
            raise WeightsError(f"weights file missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise WeightsError(
                f"{name}: SHA-256 mismatch (expected {expected}, got {actual}, {path.stat().st_size} bytes)"
            )
        observed[name] = actual
    return observed


def _make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # The detector's declared output shapes come from a 640x640 trace, so ORT
    # warns (VerifyOutputSizes) on every call with any other input size. The
    # results are correct; level 3 keeps real errors and drops the flood.
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression on ``(x1, y1, x2, y2)`` boxes, inclusive-pixel areas."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        inter_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        inter_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = inter_w * inter_h
        order = rest[inter / (areas[i] + areas[rest] - inter) <= threshold]
    return keep


def decode_scrfd(
    outputs: Sequence[np.ndarray], padded_h: int, padded_w: int, score_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SCRFD's nine output tensors into candidates in padded-input pixels.

    Output order is scores for strides 8/16/32, then box distances (4) for
    8/16/32, then keypoint offsets (10) for 8/16/32. Each grid location carries
    two anchors, stored consecutively, and distances are in stride units.
    Returns ``boxes (N, 4)`` as x1, y1, x2, y2, ``scores (N,)`` and
    ``landmarks (N, 5, 2)``.
    """
    if len(outputs) != _DETECTOR_OUTPUTS:
        raise FaceBackendError(f"detector returned {len(outputs)} outputs, expected {_DETECTOR_OUTPUTS}")

    all_boxes, all_scores, all_kps = [], [], []
    for level, stride in enumerate(_STRIDES):
        scores = outputs[level].reshape(-1)
        distances = outputs[level + 3].reshape(-1, 4) * stride
        offsets = outputs[level + 6].reshape(-1, 10) * stride

        grid_h, grid_w = padded_h // stride, padded_w // stride
        centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        centers = np.repeat(centers, _ANCHORS_PER_LOCATION, axis=0)
        if len(centers) != len(scores):
            raise FaceBackendError(f"stride {stride}: {len(scores)} scores for {len(centers)} anchors")

        keep = scores >= score_threshold
        c, d = centers[keep], distances[keep]
        all_boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
        all_scores.append(scores[keep])
        all_kps.append(offsets[keep].reshape(-1, 5, 2) + c[:, None, :])

    return np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kps)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) from ``src`` to ``dst``.

    Returns the 2x3 matrix ``cv2.warpAffine`` expects. Raises
    :class:`FaceBackendError` if ``src`` has no spread (all points identical).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mean_src, dst - mean_dst
    variance = src_c.var(axis=0).sum()
    if variance < 1e-12:
        raise FaceBackendError("degenerate landmarks: no spread")

    u, singular, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    scale = float((singular * np.diag(sign)).sum() / variance)
    translation = mean_dst - scale * rotation @ mean_src
    return np.hstack([scale * rotation, translation[:, None]]).astype(np.float32)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two ``(top, right, bottom, left)`` boxes."""
    top, right, bottom, left = max(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    union = (a[1] - a[3]) * (a[2] - a[0]) + (b[1] - b[3]) * (b[2] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box: tuple[int, int, int, int], fraction: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow a box by ``fraction`` of its size (split over both sides), clipped to the image."""
    top, right, bottom, left = box
    pad_y = (bottom - top) * fraction / 2
    pad_x = (right - left) * fraction / 2
    return (
        max(0, int(top - pad_y)),
        min(width, int(np.ceil(right + pad_x))),
        min(height, int(np.ceil(bottom + pad_y))),
        max(0, int(left - pad_x)),
    )


class ScrfdArcFaceBackend:
    """SCRFD-500M detector + ArcFace MobileFaceNet embedder on ONNX Runtime."""

    model_id = MODEL_ID
    dim = EMBEDDING_DIM

    def __init__(self, detector: ort.InferenceSession, embedder: ort.InferenceSession, config: BackendConfig):
        self._detector = detector
        self._embedder = embedder
        self._config = config
        self._detector_input = detector.get_inputs()[0].name
        self._embedder_input = embedder.get_inputs()[0].name

    @classmethod
    def load(cls, config: BackendConfig) -> ScrfdArcFaceBackend:
        """Verify the weights, open both sessions and run a warm-up pass."""
        hashes = verify_weights(config.weights_dir)
        backend = cls(
            _make_session(config.weights_dir / DETECTOR_FILE),
            _make_session(config.weights_dir / EMBEDDER_FILE),
            config,
        )
        logger.info(
            "Weights verified: %s",
            ", ".join(f"{name} sha256={digest}" for name, digest in hashes.items()),
        )
        backend._warm_up()
        return backend

    def _warm_up(self) -> None:
        """Run both models once so lazy initialisation happens now, and check the embedding width."""
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.detect(blank)
        vectors = self.embed_unaligned(blank, [(0, 64, 64, 0)])
        if vectors.shape != (1, self.dim):
            raise FaceBackendError(f"embedder returned shape {vectors.shape}, expected (1, {self.dim})")

    def _candidates(self, rgb: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the detector at ``scale`` and return raw candidates in pixels of ``rgb``.

        Boxes are ``(x1, y1, x2, y2)`` ``float32``, landmarks ``(N, 5, 2)``; no NMS yet.
        """
        height, width = rgb.shape[:2]
        if scale < 1.0:
            small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
        else:
            small = rgb
        small_h, small_w = small.shape[:2]
        scale_x, scale_y = small_w / width, small_h / height

        # Bottom/right zero padding to a multiple of 32; the detector's height
        # and width are dynamic, so no fixed square is needed.
        padded_h = -(-small_h // _PAD_MULTIPLE) * _PAD_MULTIPLE
        padded_w = -(-small_w // _PAD_MULTIPLE) * _PAD_MULTIPLE
        padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded[:small_h, :small_w] = small
        blob = np.ascontiguousarray(((padded.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None])

        outputs = self._detector.run(None, {self._detector_input: blob})
        boxes, scores, landmarks = decode_scrfd(outputs, padded_h, padded_w, self._config.score_threshold)

        # Back to the pixels of the image we were given.
        boxes = boxes / np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        landmarks = landmarks / np.array([scale_x, scale_y], dtype=np.float32)
        return boxes, scores, landmarks

    def detect(self, rgb: np.ndarray, *, min_face_px: int | None = None) -> list[Detection]:
        cfg = self._config
        min_px = cfg.min_face_px if min_face_px is None else min_face_px
        height, width = rgb.shape[:2]

        scale = min(1.0, cfg.det_max_side / max(height, width))
        boxes, scores, landmarks = self._candidates(rgb, scale)

        if cfg.coarse_min_side and max(height, width) * scale >= cfg.coarse_min_side:
            c_boxes, c_scores, c_landmarks = self._candidates(rgb, scale / 2)
            big = np.minimum(c_boxes[:, 2] - c_boxes[:, 0], c_boxes[:, 3] - c_boxes[:, 1]) >= COARSE_MIN_FACE_PX
            boxes = np.concatenate([boxes, c_boxes[big]])
            scores = np.concatenate([scores, c_scores[big]])
            landmarks = np.concatenate([landmarks, c_landmarks[big]])

        if not len(scores):
            return []

        detections: list[Detection] = []
        for i in _nms(boxes, scores, cfg.nms_threshold):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            if min(x2 - x1, y2 - y1) < min_px:
                continue
            box = (max(0, round(y1)), min(width, round(x2)), min(height, round(y2)), max(0, round(x1)))
            if box[1] <= box[3] or box[2] <= box[0]:
                continue
            detections.append(Detection(box=box, score=float(scores[i]), landmarks=landmarks[i].astype(np.float32)))
            if len(detections) >= cfg.max_faces:
                logger.warning("Detection capped at %d faces (WIKIVISAGE_MAX_FACES)", cfg.max_faces)
                break
        return detections

    def embed(self, rgb: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        crops = []
        for detection in detections:
            try:
                matrix = umeyama(detection.landmarks, ARCFACE_SRC)
            except FaceBackendError:
                crops.append(self._box_crop(rgb, detection.box))
                continue
            crops.append(cv2.warpAffine(rgb, matrix, (CROP_SIZE, CROP_SIZE), borderValue=0.0))
        return self._embed_crops(crops)

    def embed_unaligned(self, rgb: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
        return self._embed_crops([self._box_crop(rgb, box) for box in boxes])

    @staticmethod
    def _box_crop(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        top, right, bottom, left = box
        region = rgb[max(0, top) : bottom, max(0, left) : right]
        if region.size == 0:
            raise FaceBackendError(f"box {box} lies outside the image")
        return cv2.resize(region, (CROP_SIZE, CROP_SIZE))

    def _embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.dim), dtype=np.float32)
        chunks = []
        for start in range(0, len(crops), EMBED_CHUNK):
            batch = np.stack(crops[start : start + EMBED_CHUNK])
            blob = np.ascontiguousarray(((batch.astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2))
            chunks.append(self._embedder.run(None, {self._embedder_input: blob})[0])
        vectors = np.concatenate(chunks).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise FaceBackendError(f"embedder returned shape {vectors.shape}, expected (N, {self.dim})")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms < 1e-6).any():
            raise FaceBackendError("embedder produced a non-finite or zero vector")
        return vectors / norms


def encode_boxes(backend: FaceBackend, rgb: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> list[EncodedBox]:
    """Embed user-drawn boxes. A drawn box carries no landmarks, so recover them.

    For each box the detector runs on the box grown by ``ENCODE_BOX_EXPANSION``
    and clipped to the image; the detection with the highest IoU against the
    drawn box wins if it reaches ``ENCODE_MIN_IOU``, and the face is embedded
    aligned. Otherwise the box is resized to the crop size and embedded without
    alignment (``aligned=False``), which is noticeably weaker.

    Each result keeps the box the caller drew. The minimum face size is not
    applied here: the user chose this region deliberately.
    """
    height, width = rgb.shape[:2]
    located: list[tuple[int, Detection]] = []
    fallback: list[int] = []
    for index, box in enumerate(boxes):
        top, right, bottom, left = expand_box(box, ENCODE_BOX_EXPANSION, width, height)
        crop = rgb[top:bottom, left:right]
        best: Detection | None = None
        best_iou = ENCODE_MIN_IOU
        if crop.size:
            for detection in backend.detect(crop, min_face_px=0):
                shifted = detection.shifted(left, top)
                overlap = iou(shifted.box, box)
                if overlap >= best_iou:
                    best, best_iou = shifted, overlap
        if best is None:
            fallback.append(index)
        else:
            located.append((index, best))

    results: dict[int, EncodedBox] = {}
    if located:
        vectors = backend.embed(rgb, [detection for _, detection in located])
        for (index, detection), vector in zip(located, vectors):
            results[index] = EncodedBox(box=boxes[index], vector=vector, score=detection.score, aligned=True)
    if fallback:
        vectors = backend.embed_unaligned(rgb, [boxes[index] for index in fallback])
        for index, vector in zip(fallback, vectors):
            results[index] = EncodedBox(box=boxes[index], vector=vector, score=None, aligned=False)
    return [results[index] for index in range(len(boxes))]
