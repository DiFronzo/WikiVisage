"""Unit tests for ``model-server/backend.py`` (SCRFD + ArcFace on ONNX Runtime).

``backend.py`` has no KServe dependency, so it is imported straight from the
``model-server`` directory. It does need ``onnxruntime`` and ``opencv``, which
the web/worker environment deliberately lacks, so these tests are skipped there
and run in the face-service CI job (and in any environment that installed
``model-server/requirements.txt``).

Most tests use fake ONNX sessions so they exercise the surrounding logic —
decoding, scaling, alignment, chunking, box recovery — without needing a real
face. A few load the committed weights for real. Tests that need a real photo of
a face use ``tests/data/portrait_sample.jpg``; see ``tests/data/README.md``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

MODEL_SERVER = Path(__file__).resolve().parent.parent / "model-server"
if str(MODEL_SERVER) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVER))

import backend as be  # noqa: E402

try:
    from conftest import require_sample_env
except ModuleNotFoundError:
    from tests.conftest import require_sample_env

WEIGHTS = MODEL_SERVER / "weights"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Input:
    def __init__(self, name: str):
        self.name = name


def synth_outputs(padded_h: int, padded_w: int, faces: list[dict]) -> list[np.ndarray]:
    """Build SCRFD's nine output tensors so that ``decode_scrfd`` yields ``faces``.

    Each face is ``{"box": (x1, y1, x2, y2), "score": s, "kps": (5, 2) | None}`` in
    padded-input pixels. It is written onto the stride-32 anchor nearest its
    centre (first of the two anchors at that location).
    """
    strides = (8, 16, 32)
    scores = [np.zeros((((padded_h // s) * (padded_w // s)) * 2, 1), np.float32) for s in strides]
    boxes = [np.zeros((len(sc), 4), np.float32) for sc in scores]
    kps = [np.zeros((len(sc), 10), np.float32) for sc in scores]
    level = 2
    stride = strides[level]
    grid_w = padded_w // stride
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        col = min(int(((x1 + x2) / 2) // stride), grid_w - 1)
        row = min(int(((y1 + y2) / 2) // stride), padded_h // stride - 1)
        index = (row * grid_w + col) * 2
        if scores[level][index, 0] != 0:  # first anchor of this cell is taken: use the second
            index += 1
        ax, ay = col * stride, row * stride
        scores[level][index, 0] = face["score"]
        boxes[level][index] = [(ax - x1) / stride, (ay - y1) / stride, (x2 - ax) / stride, (y2 - ay) / stride]
        points = face.get("kps")
        if points is None:
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            points = np.array(
                [
                    [cx - 0.2 * w, cy - 0.1 * h],
                    [cx + 0.2 * w, cy - 0.1 * h],
                    [cx, cy + 0.05 * h],
                    [cx - 0.15 * w, cy + 0.25 * h],
                    [cx + 0.15 * w, cy + 0.25 * h],
                ]
            )
        kps[level][index] = ((np.asarray(points) - [ax, ay]) / stride).reshape(-1)
    return [*scores, *boxes, *kps]


class FakeDetector:
    """Stands in for the SCRFD session; ``script(blob)`` returns faces in padded pixels."""

    def __init__(self, script):
        self.script = script
        self.calls: list[tuple[int, ...]] = []
        self.blobs: list[np.ndarray] = []

    def get_inputs(self):
        return [_Input("input.1")]

    def run(self, _outputs, feed):
        blob = feed["input.1"]
        self.calls.append(blob.shape)
        self.blobs.append(blob)
        padded_h, padded_w = blob.shape[2], blob.shape[3]
        return synth_outputs(padded_h, padded_w, self.script(blob))


class FakeEmbedder:
    """Returns a deterministic, deliberately *unnormalised* 512-vector per crop."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.batch_sizes: list[int] = []
        self.blobs: list[np.ndarray] = []

    def get_inputs(self):
        return [_Input("input.1")]

    def run(self, _outputs, feed):
        blob = feed["input.1"]
        self.batch_sizes.append(len(blob))
        self.blobs.append(blob.copy())
        out = np.empty((len(blob), self.dim), np.float32)
        for i, crop in enumerate(blob):
            rng = np.random.default_rng(int(abs(float(crop.sum())) * 1000) % (2**32))
            out[i] = rng.standard_normal(self.dim) * 3.0
        return [out]


def make_backend(script=None, config: be.BackendConfig | None = None, embedder: FakeEmbedder | None = None):
    config = config or be.BackendConfig(weights_dir=WEIGHTS)
    detector = FakeDetector(script or (lambda _blob: []))
    embedder = embedder or FakeEmbedder()
    return be.ScrfdArcFaceBackend(detector, embedder, config), detector, embedder


def landmarks_for(box: tuple[float, float, float, float]) -> np.ndarray:
    """A plausible, non-degenerate 5-point set inside ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array(
        [
            [x1 + 0.3 * w, y1 + 0.4 * h],
            [x1 + 0.7 * w, y1 + 0.4 * h],
            [x1 + 0.5 * w, y1 + 0.6 * h],
            [x1 + 0.35 * w, y1 + 0.8 * h],
            [x1 + 0.65 * w, y1 + 0.8 * h],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Weights: committed files, pinned hashes, fail-closed verification
# ---------------------------------------------------------------------------


def _manifest() -> dict[str, str]:
    entries = {}
    for line in (WEIGHTS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip().lstrip("*")] = digest
    return entries


def test_pinned_hashes_match_the_committed_manifest():
    assert _manifest() == be.WEIGHT_SHA256


def test_pinned_hashes_match_the_values_recorded_in_the_plan():
    assert be.WEIGHT_SHA256 == {
        "det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
        "w600k_mbf.onnx": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    }


def test_committed_weights_are_real_files_not_lfs_pointers():
    for name, size in (("det_500m.onnx", 2_524_817), ("w600k_mbf.onnx", 13_616_099)):
        path = WEIGHTS / name
        assert path.stat().st_size == size
        assert not path.read_bytes()[:64].startswith(b"version https://git-lfs")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == be.WEIGHT_SHA256[name]


def test_license_notice_ships_with_the_weights():
    notice = (WEIGHTS / "LICENSE-NOTICE").read_text()
    assert "NON-COMMERCIAL" in notice
    assert "buffalo_sc" in notice


def test_verify_weights_accepts_the_committed_files():
    assert be.verify_weights(WEIGHTS) == be.WEIGHT_SHA256


def _copy_weights(tmp_path: Path) -> Path:
    for name in be.WEIGHT_SHA256:
        (tmp_path / name).write_bytes((WEIGHTS / name).read_bytes())
    return tmp_path


def test_verify_weights_rejects_a_missing_file(tmp_path):
    _copy_weights(tmp_path)
    (tmp_path / be.EMBEDDER_FILE).unlink()
    with pytest.raises(be.WeightsError, match="missing"):
        be.verify_weights(tmp_path)


def test_verify_weights_rejects_a_flipped_byte_and_reports_both_hashes(tmp_path):
    _copy_weights(tmp_path)
    path = tmp_path / be.DETECTOR_FILE
    data = bytearray(path.read_bytes())
    data[1000] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(be.WeightsError) as exc:
        be.verify_weights(tmp_path)
    assert be.WEIGHT_SHA256[be.DETECTOR_FILE] in str(exc.value)
    assert hashlib.sha256(bytes(data)).hexdigest() in str(exc.value)


def test_verify_weights_rejects_a_git_lfs_pointer_file(tmp_path):
    _copy_weights(tmp_path)
    (tmp_path / be.EMBEDDER_FILE).write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 13616099\n"
    )
    with pytest.raises(be.WeightsError, match="mismatch"):
        be.verify_weights(tmp_path)


def test_verify_weights_rejects_a_truncated_download(tmp_path):
    _copy_weights(tmp_path)
    path = tmp_path / be.DETECTOR_FILE
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(be.WeightsError, match="mismatch"):
        be.verify_weights(tmp_path)


def test_load_refuses_to_open_sessions_when_weights_do_not_verify(tmp_path, monkeypatch):
    _copy_weights(tmp_path)
    (tmp_path / be.DETECTOR_FILE).write_bytes(b"not a model")
    opened = []
    monkeypatch.setattr(be, "_make_session", lambda path: opened.append(path))

    with pytest.raises(be.WeightsError):
        be.ScrfdArcFaceBackend.load(be.BackendConfig(weights_dir=tmp_path))

    assert opened == []


def test_load_logs_the_verified_hashes_once(caplog):
    with caplog.at_level("INFO", logger="wikivisage-model-server"):
        be.ScrfdArcFaceBackend.load(be.BackendConfig(weights_dir=WEIGHTS))

    line = next(r.getMessage() for r in caplog.records if "Weights verified" in r.getMessage())
    for name, digest in be.WEIGHT_SHA256.items():
        assert f"{name} sha256={digest}" in line


# ---------------------------------------------------------------------------
# Session options (plan 4.2)
# ---------------------------------------------------------------------------


def test_sessions_are_single_threaded_sequential_cpu_and_quiet(monkeypatch):
    seen = {}

    class _Session:
        def __init__(self, path, options, providers):
            seen.update(path=path, options=options, providers=providers)

    monkeypatch.setattr(be.ort, "InferenceSession", _Session)
    be._make_session(WEIGHTS / be.DETECTOR_FILE)

    options = seen["options"]
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == be.ort.ExecutionMode.ORT_SEQUENTIAL
    # 3 = ERROR: silences the per-call VerifyOutputSizes warning of non-640x640 inputs.
    assert options.log_severity_level == 3
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_importing_the_backend_limits_opencv_to_one_thread(monkeypatch):
    """Executor threads provide the parallelism; OpenCV's own pool would oversubscribe the CPU."""
    import importlib.util

    import cv2

    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: calls.append(n))
    spec = importlib.util.spec_from_file_location("backend_import_probe", MODEL_SERVER / "backend.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look their module up here
    spec.loader.exec_module(module)

    assert calls == [1]


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WIKIVISAGE_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("WIKIVISAGE_DET_MAX_SIDE", "1024")
    monkeypatch.setenv("WIKIVISAGE_MIN_FACE_PX", "16")
    monkeypatch.setenv("WIKIVISAGE_MAX_FACES", "50")
    monkeypatch.setenv("WIKIVISAGE_DET_COARSE_MIN_SIDE", "0")

    cfg = be.BackendConfig.from_env()

    assert (cfg.weights_dir, cfg.det_max_side, cfg.min_face_px, cfg.max_faces, cfg.coarse_min_side) == (
        tmp_path,
        1024,
        16,
        50,
        0,
    )


def test_config_defaults_match_the_plan(monkeypatch):
    for var in (
        "WIKIVISAGE_WEIGHTS_DIR",
        "WIKIVISAGE_DET_MAX_SIDE",
        "WIKIVISAGE_MIN_FACE_PX",
        "WIKIVISAGE_MAX_FACES",
        "WIKIVISAGE_DET_COARSE_MIN_SIDE",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = be.BackendConfig.from_env()

    assert cfg.weights_dir == WEIGHTS
    assert (cfg.det_max_side, cfg.min_face_px, cfg.score_threshold, cfg.nms_threshold) == (1280, 20, 0.5, 0.4)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _similarity(scale: float, angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty]])


def test_umeyama_recovers_an_exact_similarity_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (5, 2))
    truth = _similarity(1.7, np.deg2rad(30), 12.0, -8.0)
    dst = src @ truth[:, :2].T + truth[:, 2]

    assert np.allclose(be.umeyama(src, dst), truth, atol=1e-3)


def test_umeyama_of_the_template_onto_itself_is_the_identity():
    m = be.umeyama(be.ARCFACE_SRC, be.ARCFACE_SRC)
    assert np.allclose(m, np.hstack([np.eye(2), np.zeros((2, 1))]), atol=1e-4)


def test_umeyama_never_returns_a_reflection():
    src = be.ARCFACE_SRC.astype(np.float64)
    mirrored = src * np.array([-1.0, 1.0])
    m = be.umeyama(src, mirrored)
    assert np.linalg.det(m[:, :2]) > 0


def test_umeyama_minimises_squared_error_for_noisy_points():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, (5, 2))
    truth = _similarity(0.6, np.deg2rad(-12), 3.0, 5.0)
    dst = src @ truth[:, :2].T + truth[:, 2] + rng.normal(0, 0.5, (5, 2))

    def error(m: np.ndarray) -> float:
        return float(((src @ m[:, :2].T + m[:, 2] - dst) ** 2).sum())

    best = be.umeyama(src, dst).astype(np.float64)
    for delta in (np.array([0.01, 0, 0, 0, 0, 0]), np.array([0, 0, 0.5, 0, 0, 0]), np.array([0, 0, 0, 0, 0, -0.5])):
        assert error(best) <= error(best + delta.reshape(2, 3))


def test_umeyama_rejects_degenerate_landmarks():
    with pytest.raises(be.FaceBackendError, match="degenerate"):
        be.umeyama(np.full((5, 2), 7.0), be.ARCFACE_SRC)


def test_iou():
    assert be.iou((0, 10, 10, 0), (0, 10, 10, 0)) == 1.0
    assert be.iou((0, 10, 10, 0), (20, 30, 30, 20)) == 0.0
    assert be.iou((0, 10, 10, 0), (0, 15, 10, 5)) == pytest.approx(50 / 150)


def test_expand_box_grows_by_the_fraction_and_clips():
    # 100x100 box grown by 0.5 -> 150x150 (25 px per side)
    assert be.expand_box((100, 200, 200, 100), 0.5, 1000, 1000) == (75, 225, 225, 75)
    assert be.expand_box((10, 60, 60, 10), 0.5, 70, 65) == (0, 70, 65, 0)


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], np.float32)
    scores = np.array([0.6, 0.9, 0.7], np.float32)
    assert be._nms(boxes, scores, 0.4) == [1, 2]


def test_nms_keeps_a_loosely_overlapping_box():
    boxes = np.array([[0, 0, 100, 100], [70, 0, 170, 100]], np.float32)  # IoU ~ 0.18
    assert be._nms(boxes, np.array([0.9, 0.8], np.float32), 0.4) == [0, 1]


# ---------------------------------------------------------------------------
# SCRFD output decoding
# ---------------------------------------------------------------------------


def test_decode_scrfd_inverts_the_anchor_encoding():
    outputs = synth_outputs(96, 128, [{"box": (40.0, 20.0, 90.0, 70.0), "score": 0.93}])

    boxes, scores, kps = be.decode_scrfd(outputs, 96, 128, 0.5)

    assert scores.tolist() == pytest.approx([0.93])
    assert boxes[0].tolist() == pytest.approx([40.0, 20.0, 90.0, 70.0])
    assert kps.shape == (1, 5, 2)


def test_decode_scrfd_drops_candidates_below_the_score_threshold():
    outputs = synth_outputs(64, 64, [{"box": (8.0, 8.0, 40.0, 40.0), "score": 0.49}])
    boxes, scores, _ = be.decode_scrfd(outputs, 64, 64, 0.5)
    assert len(scores) == 0 and boxes.shape == (0, 4)


def test_decode_scrfd_rejects_the_wrong_number_of_outputs():
    with pytest.raises(be.FaceBackendError, match="expected 9"):
        be.decode_scrfd(synth_outputs(64, 64, [])[:6], 64, 64, 0.5)


def test_decode_scrfd_rejects_an_anchor_count_mismatch():
    outputs = synth_outputs(64, 64, [])
    with pytest.raises(be.FaceBackendError, match="anchors"):
        be.decode_scrfd(outputs, 96, 64, 0.5)


# ---------------------------------------------------------------------------
# detect(): preprocessing, scaling, filtering
# ---------------------------------------------------------------------------


def _image(width: int, height: int, value: int = 100) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_detect_pads_to_a_multiple_of_32_and_normalises_like_scrfd():
    backend, detector, _ = make_backend()
    image = _image(100, 70, value=127)
    image[0, 0] = (255, 0, 127)

    backend.detect(image)

    blob = detector.blobs[0]
    assert blob.shape == (1, 3, 96, 128)
    assert blob.dtype == np.float32
    assert blob[0, :, 0, 0] == pytest.approx([(255 - 127.5) / 128, (0 - 127.5) / 128, (127 - 127.5) / 128])
    assert blob[0, 0, 5, 5] == pytest.approx((127 - 127.5) / 128)
    # Padding is black *pixels* (0) on the bottom/right, which normalise to -127.5/128.
    assert np.allclose(blob[0, :, 70:, :], -127.5 / 128) and np.allclose(blob[0, :, :, 100:], -127.5 / 128)


def test_detect_returns_boxes_in_the_pixels_of_the_input_image():
    face = {"box": (40.0, 20.0, 90.0, 70.0), "score": 0.9}
    backend, _, _ = make_backend(lambda _b: [face], be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0))

    [detection] = backend.detect(_image(100, 70))

    assert detection.box == (20, 90, 70, 40)  # top, right, bottom, left
    assert detection.score == pytest.approx(0.9)
    assert detection.landmarks.shape == (5, 2) and detection.landmarks.dtype == np.float32


def test_detect_downscales_large_images_and_maps_back():
    config = be.BackendConfig(weights_dir=WEIGHTS, det_max_side=1280, coarse_min_side=0)
    # In the 1280-wide detector input the face is at x 320..640, y 160..480.
    backend, detector, _ = make_backend(lambda _b: [{"box": (320.0, 160.0, 640.0, 480.0), "score": 0.9}], config)

    [detection] = backend.detect(_image(2560, 1280))

    assert detector.calls[0][2:] == (640, 1280)  # long side capped at 1280, aspect kept
    assert detection.box == (320, 1280, 960, 640)  # twice the detector-space coordinates


def test_detect_never_upscales_small_images():
    backend, detector, _ = make_backend()
    backend.detect(_image(200, 100))
    assert detector.calls[0][2:] == (128, 224)


def test_detect_clips_boxes_to_the_image():
    backend, _, _ = make_backend(
        lambda _b: [{"box": (-30.0, -20.0, 130.0, 90.0), "score": 0.9}],
        be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0),
    )
    [detection] = backend.detect(_image(100, 70))
    assert detection.box == (0, 100, 70, 0)


def test_detect_drops_faces_below_the_minimum_short_side():
    faces = [
        {"box": (10.0, 10.0, 28.0, 60.0), "score": 0.9},  # 18 px wide: too small
        {"box": (50.0, 10.0, 74.0, 60.0), "score": 0.8},  # 24 px wide: kept
    ]
    backend, _, _ = make_backend(lambda _b: faces, be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0))

    result = backend.detect(_image(128, 96))

    assert [d.box[3] for d in result] == [50]


def test_detect_min_face_px_can_be_overridden_per_call():
    faces = [{"box": (10.0, 10.0, 28.0, 60.0), "score": 0.9}]
    backend, _, _ = make_backend(lambda _b: faces, be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0))

    assert backend.detect(_image(128, 96)) == []
    assert len(backend.detect(_image(128, 96), min_face_px=0)) == 1


def test_detect_returns_best_face_first_and_suppresses_duplicates():
    faces = [
        {"box": (10.0, 10.0, 60.0, 60.0), "score": 0.6},
        {"box": (14.0, 12.0, 64.0, 62.0), "score": 0.9},  # same face
        {"box": (200.0, 10.0, 250.0, 60.0), "score": 0.7},
    ]
    backend, _, _ = make_backend(lambda _b: faces, be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0))

    result = backend.detect(_image(256, 96))

    assert [round(d.score, 2) for d in result] == [0.9, 0.7]


def test_detect_caps_the_number_of_faces(caplog):
    faces = [{"box": (float(10 + 40 * i), 10.0, float(40 + 40 * i), 60.0), "score": 0.9 - i * 0.01} for i in range(6)]
    config = be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0, max_faces=3)
    backend, _, _ = make_backend(lambda _b: faces, config)

    with caplog.at_level("WARNING", logger="wikivisage-model-server"):
        result = backend.detect(_image(320, 96))

    assert len(result) == 3
    assert [round(d.score, 2) for d in result] == [0.9, 0.89, 0.88]
    assert any("capped" in r.getMessage() for r in caplog.records)


def test_detect_on_an_image_with_no_faces_returns_an_empty_list():
    backend, _, _ = make_backend()
    assert backend.detect(_image(128, 128)) == []


def test_detect_runs_a_second_half_scale_pass_to_catch_very_large_faces():
    """SCRFD-500M misses faces > ~480 px; the coarse pass finds them at half size."""
    calls = []

    def script(blob):
        calls.append(blob.shape[2:])
        if blob.shape[3] >= 1000:  # full-size pass: the face is too big to see
            return []
        return [{"box": (100.0, 50.0, 400.0, 350.0), "score": 0.9}]  # half-size pass

    backend, _, _ = make_backend(script)

    [detection] = backend.detect(_image(1280, 800))

    assert calls == [(800, 1280), (416, 640)]  # half-size pass is padded to a multiple of 32 too
    assert detection.box == (100, 800, 700, 200)  # mapped back to source pixels
    assert detection.score == pytest.approx(0.9)


def test_coarse_pass_duplicate_of_a_face_the_full_pass_found_is_suppressed():
    def script(blob):
        if blob.shape[3] >= 1000:
            return [{"box": (400.0, 200.0, 760.0, 560.0), "score": 0.95}]
        # The same 360 px face also found at half size (180 px there), less precisely.
        return [{"box": (196.0, 96.0, 384.0, 276.0), "score": 0.9}]

    backend, _, _ = make_backend(script)

    result = backend.detect(_image(1280, 800))

    assert len(result) == 1
    assert result[0].box == (200, 760, 560, 400)  # the full-size box, not the coarse one


def test_coarse_pass_only_contributes_large_faces():
    def script(blob):
        if blob.shape[3] >= 1000:
            return []
        return [{"box": (100.0, 50.0, 180.0, 130.0), "score": 0.9}]  # 160 px in source: the full pass owns this size

    backend, _, _ = make_backend(script)

    assert backend.detect(_image(1280, 800)) == []


def test_coarse_pass_is_skipped_for_small_images_and_when_disabled():
    backend, detector, _ = make_backend()
    backend.detect(_image(500, 375))
    assert len(detector.calls) == 1

    backend, detector, _ = make_backend(config=be.BackendConfig(weights_dir=WEIGHTS, coarse_min_side=0))
    backend.detect(_image(2000, 1500))
    assert len(detector.calls) == 1


# ---------------------------------------------------------------------------
# embed(): alignment, batching, normalisation
# ---------------------------------------------------------------------------


def test_embed_aligns_with_the_arcface_template_and_normalises_like_arcface():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
    backend, _, embedder = make_backend()
    detection = be.Detection(box=(0, 112, 112, 0), score=0.9, landmarks=be.ARCFACE_SRC.copy())

    backend.embed(image, [detection])

    # Landmarks equal to the template -> identity warp -> the crop is the image itself.
    expected = ((image.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)
    assert embedder.blobs[0].shape == (1, 3, 112, 112)
    assert np.allclose(embedder.blobs[0][0], expected, atol=1e-5)


def test_embed_returns_unit_norm_float32_vectors():
    backend, _, _ = make_backend()
    image = np.random.default_rng(4).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    detections = [
        be.Detection(box=(0, 100, 100, 0), score=0.9, landmarks=landmarks_for((20, 20, 120, 120))),
        be.Detection(box=(0, 100, 100, 0), score=0.8, landmarks=landmarks_for((60, 60, 190, 190))),
    ]

    vectors = backend.embed(image, detections)

    assert vectors.shape == (2, 512) and vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert not np.allclose(vectors[0], vectors[1])


def test_embed_batches_all_faces_of_an_image_into_one_call_up_to_the_chunk_size():
    backend, _, embedder = make_backend()
    image = np.random.default_rng(5).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    detection = be.Detection(box=(0, 100, 100, 0), score=0.9, landmarks=landmarks_for((20, 20, 120, 120)))

    backend.embed(image, [detection] * 10)
    assert embedder.batch_sizes == [10]

    embedder.batch_sizes.clear()
    vectors = backend.embed(image, [detection] * (be.EMBED_CHUNK * 2 + 5))
    assert embedder.batch_sizes == [be.EMBED_CHUNK, be.EMBED_CHUNK, 5]
    assert len(vectors) == be.EMBED_CHUNK * 2 + 5


def test_embed_of_no_detections_makes_no_model_call():
    backend, _, embedder = make_backend()
    assert backend.embed(_image(50, 50), []).shape == (0, 512)
    assert embedder.batch_sizes == []


def test_embed_falls_back_to_a_box_crop_for_degenerate_landmarks():
    backend, _, embedder = make_backend()
    image = np.random.default_rng(6).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    detection = be.Detection(box=(10, 60, 60, 10), score=0.9, landmarks=np.full((5, 2), 30.0, np.float32))

    vectors = backend.embed(image, [detection])

    assert vectors.shape == (1, 512)
    assert embedder.blobs[0].shape == (1, 3, 112, 112)


def test_embed_unaligned_resizes_the_box_to_112():
    backend, _, embedder = make_backend()
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[20:80, 40:100] = 255

    backend.embed_unaligned(image, [(20, 100, 80, 40)])

    assert embedder.blobs[0].shape == (1, 3, 112, 112)
    assert embedder.blobs[0].min() == pytest.approx(1.0)  # all-white region -> (255 - 127.5) / 127.5


def test_embed_unaligned_rejects_a_box_outside_the_image():
    backend, _, _ = make_backend()
    with pytest.raises(be.FaceBackendError, match="outside"):
        backend.embed_unaligned(_image(50, 50), [(100, 120, 120, 100)])


@pytest.mark.parametrize("bad", ["nan", "zero", "shape"])
def test_embedder_misbehaviour_is_an_error_not_a_stored_vector(bad):
    class Broken(FakeEmbedder):
        def run(self, _outputs, feed):
            (out,) = super().run(_outputs, feed)
            if bad == "nan":
                out[0, 0] = np.nan
            elif bad == "zero":
                out[0] = 0
            else:
                out = out[:, :128]
            return [out]

    backend, _, _ = make_backend(embedder=Broken())
    with pytest.raises(be.FaceBackendError):
        backend.embed_unaligned(_image(60, 60), [(0, 60, 60, 0)])


# ---------------------------------------------------------------------------
# encode_boxes(): landmark recovery and the unaligned fallback
# ---------------------------------------------------------------------------


class StubBackend:
    """A ``FaceBackend`` whose detector output is scripted per crop."""

    model_id = "stub"
    dim = 4

    def __init__(self, detections_for_crop):
        self.detections_for_crop = detections_for_crop
        self.detect_calls: list[tuple[tuple[int, int], int | None]] = []
        self.embedded: list[list[tuple[int, int, int, int]]] = []
        self.unaligned: list[list[tuple[int, int, int, int]]] = []

    def detect(self, rgb, *, min_face_px=None):
        self.detect_calls.append((rgb.shape[:2], min_face_px))
        return self.detections_for_crop(rgb)

    def embed(self, rgb, detections):
        self.embedded.append([d.box for d in detections])
        return np.array([[1.0, i, 0.0, 0.0] for i, _ in enumerate(detections)], np.float32)

    def embed_unaligned(self, rgb, boxes):
        self.unaligned.append(list(boxes))
        return np.array([[0.0, 0.0, 1.0, i] for i, _ in enumerate(boxes)], np.float32)


def _det(box, score=0.9):
    t, r, b, left = box
    return be.Detection(box=box, score=score, landmarks=landmarks_for((left, t, r, b)))


def test_encode_recovers_landmarks_from_the_detection_with_the_best_overlap():
    drawn = (100, 200, 200, 100)
    # In crop coordinates the crop starts at (75, 75), so these are the full-image boxes minus 75.
    crop_detections = [_det((25, 125, 125, 25), 0.7), _det((300, 400, 400, 300), 0.99)]
    backend = StubBackend(lambda rgb: crop_detections)

    [result] = be.encode_boxes(backend, _image(1000, 1000), [drawn])

    assert result.aligned is True
    assert result.score == pytest.approx(0.7)
    assert result.box == drawn  # the caller's box, not the detection's
    assert backend.embedded == [[(100, 200, 200, 100)]]  # shifted back to full-image coordinates
    assert backend.unaligned == []


def test_encode_runs_the_detector_on_the_box_grown_by_half_and_without_a_minimum_size():
    backend = StubBackend(lambda rgb: [])

    be.encode_boxes(backend, _image(1000, 1000), [(100, 200, 200, 100)])

    assert backend.detect_calls == [((150, 150), 0)]


def test_encode_clips_the_search_region_to_the_image():
    backend = StubBackend(lambda rgb: [])
    be.encode_boxes(backend, _image(120, 100), [(10, 110, 90, 20)])
    height, width = backend.detect_calls[0][0]
    assert width <= 120 and height <= 100


def test_encode_falls_back_to_an_unaligned_crop_when_nothing_overlaps_enough():
    drawn = (100, 200, 200, 100)
    far = _det((130, 330, 210, 270))  # overlaps nothing (crop starts at 75)
    backend = StubBackend(lambda rgb: [far])

    [result] = be.encode_boxes(backend, _image(1000, 1000), [drawn])

    assert result.aligned is False
    assert result.score is None
    assert result.box == drawn
    assert backend.unaligned == [[drawn]]
    assert backend.embedded == []


def test_encode_requires_an_iou_of_at_least_0_3():
    drawn = (100, 200, 200, 100)  # 100x100
    # IoU with a box shifted 60 px right: 4000 / 16000 = 0.25 -> rejected; 50 px: 5000/15000 = 0.33 -> accepted.
    rejected = StubBackend(lambda rgb: [_det((25, 185, 125, 85))])  # crop coords for x 160..260
    accepted = StubBackend(lambda rgb: [_det((25, 175, 125, 75))])  # x 150..250

    assert be.encode_boxes(rejected, _image(1000, 1000), [drawn])[0].aligned is False
    assert be.encode_boxes(accepted, _image(1000, 1000), [drawn])[0].aligned is True


def test_encode_keeps_result_order_with_a_mix_of_located_and_fallback_boxes():
    boxes = [(100, 200, 200, 100), (400, 500, 500, 400), (700, 800, 800, 700)]
    image = _image(1000, 1000, value=0)
    image[375:525, 375:525] = 7  # only the middle box's search region is marked

    def script(rgb):
        return [_det((25, 125, 125, 25))] if rgb[0, 0, 0] == 7 else []

    backend = StubBackend(script)
    results = be.encode_boxes(backend, image, boxes)

    assert [r.box for r in results] == boxes
    assert [r.aligned for r in results] == [False, True, False]
    assert backend.embedded == [[(400, 500, 500, 400)]]
    assert backend.unaligned == [[boxes[0], boxes[2]]]
    # Vectors follow their boxes: located boxes came from embed(), fallbacks from embed_unaligned().
    assert results[1].vector.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert results[0].vector.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert results[2].vector.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_encode_returns_one_result_per_box_with_matching_vectors():
    backend = StubBackend(lambda rgb: [])
    boxes = [(10, 60, 60, 10), (100, 160, 160, 100)]

    results = be.encode_boxes(backend, _image(300, 300), boxes)

    assert len(results) == 2
    assert [r.vector.tolist() for r in results] == [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]


# ---------------------------------------------------------------------------
# Real weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_backend():
    return be.ScrfdArcFaceBackend.load(be.BackendConfig(weights_dir=WEIGHTS))


def _noise(seed: int, size: int = 160) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_real_models_load_and_report_the_plan_constants(real_backend):
    assert real_backend.model_id == "scrfd500m+arcface_mbf"
    assert real_backend.dim == 512


def test_real_detector_finds_no_face_in_a_flat_image(real_backend):
    assert real_backend.detect(_image(320, 240, 128)) == []


def test_real_embedder_returns_deterministic_unit_float32_vectors(real_backend):
    image = _noise(1)
    first = real_backend.embed_unaligned(image, [(0, 160, 160, 0)])
    second = real_backend.embed_unaligned(image, [(0, 160, 160, 0)])

    assert first.shape == (1, 512) and first.dtype == np.float32
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(first, second)
    assert not np.allclose(first, real_backend.embed_unaligned(_noise(2), [(0, 160, 160, 0)]))


def test_real_sessions_are_safe_to_share_across_threads(real_backend):
    """The service runs several executor threads on one session pair (no lock)."""
    images = [_noise(i) for i in range(8)]
    expected = [real_backend.embed_unaligned(img, [(0, 160, 160, 0)])[0] for img in images]
    results: dict[int, np.ndarray] = {}
    errors: list[Exception] = []

    def work(index: int) -> None:
        try:
            for _ in range(5):
                results[index] = real_backend.embed_unaligned(images[index], [(0, 160, 160, 0)])[0]
                real_backend.detect(images[index])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for i, vector in results.items():
        assert np.array_equal(vector, expected[i])


def test_real_encode_of_a_box_without_a_face_uses_the_unaligned_fallback(real_backend):
    [result] = be.encode_boxes(real_backend, _noise(3, 300), [(60, 200, 200, 60)])

    assert result.aligned is False
    assert result.vector.shape == (512,)
    assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-5)


# A public-domain photo with exactly one frontal face. Override the path with
# WIKIVISAGE_TEST_SAMPLE_IMAGE to try a different photo without committing it.
SAMPLE = Path(
    os.environ.get("WIKIVISAGE_TEST_SAMPLE_IMAGE", Path(__file__).resolve().parent / "data" / "portrait_sample.jpg")
)


@pytest.fixture(scope="module")
def portrait() -> np.ndarray:
    if not SAMPLE.exists():
        if require_sample_env():
            pytest.fail(f"{SAMPLE} is missing but WIKIVISAGE_REQUIRE_SAMPLE is set (see tests/data/README.md)")
        pytest.skip(f"{SAMPLE} not committed yet (see tests/data/README.md)")
    from PIL import Image

    return np.array(Image.open(SAMPLE).convert("RGB"))


def test_real_portrait_has_exactly_one_confident_face_and_a_usable_embedding(real_backend, portrait):
    detections = real_backend.detect(portrait)

    assert len(detections) == 1
    assert detections[0].score > 0.6
    vector = real_backend.embed(portrait, detections)[0]
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_real_portrait_drawn_box_recovers_landmarks_and_matches_the_detected_embedding(real_backend, portrait):
    [detection] = real_backend.detect(portrait)
    top, right, bottom, left = detection.box
    drawn = (top + 8, right - 6, bottom - 4, left + 6)  # a slightly sloppy hand-drawn box

    [encoded] = be.encode_boxes(real_backend, portrait, [drawn])
    reference = real_backend.embed(portrait, [detection])[0]

    assert encoded.aligned is True
    assert float(encoded.vector @ reference) > 0.9
