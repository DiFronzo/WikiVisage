"""Unit tests for ``model-server/model.py``: the KServe V1 envelope around the backend.

Needs ``kserve`` and the ONNX stack (``model-server/requirements.txt``), which the
web/worker environment lacks, so it is skipped there and runs in the face-service
CI job. A scripted fake backend stands in for the models, so no weights or real
faces are involved (two tests load the committed weights for real).
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kserve")
pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

MODEL_SERVER = Path(__file__).resolve().parent.parent / "model-server"
if str(MODEL_SERVER) not in sys.path:
    sys.path.insert(0, str(MODEL_SERVER))

import backend as be  # noqa: E402
import model as ms  # noqa: E402
from PIL import Image  # noqa: E402

DIM = 512
BOX = {"top": 10, "right": 60, "bottom": 60, "left": 10}


def _png(width: int = 200, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _vector(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeBackend:
    """Scripted ``FaceBackend``: detections are fixed, embeddings derive from the box."""

    model_id = "fake-model"
    dim = DIM

    def __init__(self, detections: list[be.Detection] | None = None):
        self.detections = detections or []
        self.calls: list[tuple] = []
        self.thread_ids: set[int] = set()

    def detect(self, rgb, *, min_face_px=None):
        self.thread_ids.add(threading.get_ident())
        self.calls.append(("detect", rgb.shape[:2], min_face_px))
        return list(self.detections)

    def embed(self, rgb, detections):
        self.calls.append(("embed", [d.box for d in detections]))
        if not detections:
            return np.empty((0, DIM), np.float32)
        return np.stack([_vector(d.box[0] + d.box[3]) for d in detections])

    def embed_unaligned(self, rgb, boxes):
        self.calls.append(("embed_unaligned", list(boxes)))
        return np.stack([_vector(b[0] + b[3] + 1000) for b in boxes])


def _detection(box=(10, 60, 60, 10), score=0.91) -> be.Detection:
    top, _right, _bottom, left = box
    landmarks = np.array(
        [
            [left + 15, top + 20],
            [left + 35, top + 20],
            [left + 25, top + 30],
            [left + 18, top + 40],
            [left + 32, top + 40],
        ],
        dtype=np.float32,
    )
    return be.Detection(box=box, score=score, landmarks=landmarks)


@pytest.fixture
def model():
    m = ms.WikiVisageModel("wikivisage", face_workers=2, backend=FakeBackend([_detection()]))
    m.load()
    yield m
    m.stop()


def _run(model, instance):
    return model._run_instance(instance)


def _detect_instance(image: bytes | None = None, **extra):
    return {"id": "1", "task": "detect", "image": _b64(image or _png()), **extra}


def _encode_instance(boxes, image: bytes | None = None, **extra):
    return {"id": "2", "task": "encode", "image": _b64(image or _png()), "boxes": boxes, **extra}


# ---------------------------------------------------------------------------
# No dlib, no lock, no multiprocessing, no system packages
# ---------------------------------------------------------------------------


def test_service_no_longer_references_dlib():
    source = (MODEL_SERVER / "model.py").read_text()
    for forbidden in ("import dlib", "face_recognition", "_dlib_lock", "_Dlib", "import multiprocessing"):
        assert forbidden not in source
    assert "dlib" not in sys.modules


def test_requirements_drop_dlib_and_pin_the_onnx_stack():
    text = (MODEL_SERVER / "requirements.txt").read_text()
    pins = {line.split("#")[0].strip() for line in text.splitlines()} - {""}
    assert {"kserve==0.15.2", "onnxruntime==1.31.0", "opencv-python-headless==5.0.0.93"} <= pins
    assert not any(p.startswith(("dlib", "face_recognition", "setuptools")) for p in pins)


def test_no_deb_packages_are_needed():
    assert not (MODEL_SERVER / "project.toml").exists()
    for name in ("Dockerfile", "blubber.yaml"):
        text = (MODEL_SERVER / name).read_text()
        assert "openblas" not in text and "liblapack" not in text


def test_dockerfile_and_dockerignore_ship_the_backend_and_weights():
    dockerfile = (MODEL_SERVER / "Dockerfile").read_text()
    assert "COPY model.py backend.py guard.py" in dockerfile
    assert "COPY weights" in dockerfile
    ignore = (MODEL_SERVER / ".dockerignore").read_text().split()
    for entry in ("!backend.py", "!weights/", "!weights/det_500m.onnx", "!weights/w600k_mbf.onnx"):
        assert entry in ignore


def test_default_face_workers_match_the_two_vcpu_job():
    assert ms.FACE_WORKERS == 2


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def test_detect_response_carries_model_id_dim_score_and_float32_encoding(model):
    result = _run(model, _detect_instance())

    assert result["status"] == "ok"
    assert (result["width"], result["height"]) == (200, 100)
    assert result["model_id"] == "fake-model" and result["dim"] == DIM
    [face] = result["faces"]
    assert face["box"] == BOX
    assert face["score"] == 0.91
    raw = base64.b64decode(face["encoding"], validate=True)
    assert len(raw) == DIM * 4
    assert np.array_equal(np.frombuffer(raw, dtype="<f4"), _vector(20))
    assert "aligned" not in face


def test_detect_with_no_faces_is_an_empty_list(model):
    model._backend.detections = []
    result = _run(model, _detect_instance())
    assert result["status"] == "ok" and result["faces"] == []


def test_encode_vector_is_little_endian_float32_of_exactly_dim_values():
    vec = np.arange(DIM, dtype=np.float64) / DIM
    raw = base64.b64decode(ms._encode_vector(vec, DIM))
    assert len(raw) == DIM * 4
    assert np.array_equal(np.frombuffer(raw, dtype="<f4"), vec.astype(np.float32))


def test_encode_vector_rejects_a_wrong_width():
    with pytest.raises(ms.ImageRejected, match="unexpected embedding length"):
        ms._encode_vector(np.zeros(128, dtype=np.float32), DIM)


def test_response_never_carries_the_old_float64_encoding_width(model):
    face = _run(model, _detect_instance())["faces"][0]
    assert len(base64.b64decode(face["encoding"])) != 1024


# ---------------------------------------------------------------------------
# encode task
# ---------------------------------------------------------------------------


def test_encode_returns_the_callers_box_with_score_and_aligned_flag(model):
    model._backend.detections = [_detection((10, 60, 60, 10), 0.77)]

    result = _run(model, _encode_instance([BOX]))

    assert result["status"] == "ok"
    [face] = result["faces"]
    assert face["box"] == BOX
    assert face["aligned"] is True
    assert face["score"] == 0.77
    assert len(base64.b64decode(face["encoding"])) == DIM * 4


def test_encode_fallback_reports_unaligned_and_null_score(model):
    model._backend.detections = []

    [face] = _run(model, _encode_instance([BOX]))["faces"]

    assert face["aligned"] is False
    assert face["score"] is None


def test_encode_scales_boxes_from_box_space_to_the_decoded_image(model):
    """The service decodes a 200x100 image; the caller measured in 400x200 (2x)."""
    model._backend.detections = []
    box = {"top": 20, "right": 120, "bottom": 120, "left": 40}

    [face] = _run(model, _encode_instance([box], box_space={"width": 400, "height": 200}))["faces"]

    assert face["box"] == {"top": 10, "right": 60, "bottom": 60, "left": 20}
    assert ("embed_unaligned", [(10, 60, 60, 20)]) in model._backend.calls


def test_encode_scales_upwards_from_a_smaller_box_space(model):
    model._backend.detections = []
    box = {"top": 10, "right": 20, "bottom": 20, "left": 10}  # 10x10 in a 50x25 space

    [face] = _run(model, _encode_instance([box], box_space={"width": 50, "height": 25}))["faces"]

    assert face["box"] == {"top": 40, "right": 80, "bottom": 80, "left": 40}  # 200/50 = 100/25 = 4x


def test_fit_box_scales_axes_independently():
    assert ms._fit_box((10, 20, 20, 10), (100, 100), 200, 50) == (5, 40, 10, 20)


def test_encode_without_box_space_uses_boxes_as_decoded_pixels(model):
    model._backend.detections = []
    [face] = _run(model, _encode_instance([BOX]))["faces"]
    assert face["box"] == BOX


def test_encode_clips_a_box_that_overhangs_the_image(model):
    model._backend.detections = []
    box = {"top": 10, "right": 900, "bottom": 900, "left": 10}

    [face] = _run(model, _encode_instance([box]))["faces"]

    assert face["box"] == {"top": 10, "right": 200, "bottom": 100, "left": 10}


def test_encode_rejects_a_box_entirely_outside_the_image(model):
    box = {"top": 500, "right": 700, "bottom": 600, "left": 600}

    result = _run(model, _encode_instance([box]))

    assert result["status"] == "error"
    assert "outside" in result["error"]


@pytest.mark.parametrize(
    "box_space,message",
    [
        ("big", "must be an object"),
        ({"width": 100}, "missing key"),
        ({"width": "a", "height": 10}, "integers"),
        ({"width": 0, "height": 10}, "within 1"),
        ({"width": 10, "height": -5}, "within 1"),
        ({"width": 10**6, "height": 10}, "within 1"),
    ],
)
def test_invalid_box_space_is_rejected_per_instance(model, box_space, message):
    result = _run(model, _encode_instance([BOX], box_space=box_space))
    assert result["status"] == "error" and message in result["error"]


def test_box_space_is_ignored_for_detect(model):
    result = _run(model, _detect_instance(box_space={"width": 1, "height": 1}))
    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "box,message",
    [
        ({"top": 10, "right": 15, "bottom": 15, "left": 10}, "area must be at least"),  # 5x5
        ({"top": 10, "right": 5, "bottom": 60, "left": 20}, "right > left"),
        ({"top": -1, "right": 60, "bottom": 60, "left": 10}, "within 0.."),
        ({"top": 10, "right": 20000, "bottom": 60, "left": 10}, "within 0.."),
    ],
)
def test_existing_box_validation_is_unchanged(model, box, message):
    result = _run(model, _encode_instance([box]))
    assert result["status"] == "error" and message in result["error"]


def test_encode_requires_boxes_and_enforces_the_per_instance_limit(model):
    assert "requires a non-empty" in _run(model, _encode_instance([]))["error"]
    too_many = [BOX] * (ms.MAX_BOXES_PER_INSTANCE + 1)
    assert "too many boxes" in _run(model, _encode_instance(too_many))["error"]


def test_encode_handles_several_boxes_in_order(model):
    model._backend.detections = []
    boxes = [BOX, {"top": 20, "right": 90, "bottom": 90, "left": 20}]

    result = _run(model, _encode_instance(boxes))

    assert [f["box"] for f in result["faces"]] == boxes


# ---------------------------------------------------------------------------
# Per-instance error isolation (unchanged contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance,message",
    [
        ({"id": "x", "task": "detect", "image": "!!!"}, "not valid base64"),
        ({"id": "x", "task": "detect", "image": _b64(b"not an image")}, "could not read image"),
        ({"id": "x", "task": "detect", "image": ""}, "non-empty"),
        ({"id": "x", "task": "frobnicate", "image": _b64(b"x")}, "unknown task"),
    ],
)
def test_bad_instances_become_error_entries(model, instance, message):
    result = _run(model, instance)
    assert result["status"] == "error" and message in result["error"]


def test_backend_failure_is_reported_for_that_instance_only(model):
    def boom(*_a, **_k):
        raise be.FaceBackendError("embedder produced a non-finite or zero vector")

    model._backend.embed = boom

    result = _run(model, _detect_instance())

    assert result["status"] == "error" and "face backend failed" in result["error"]


def test_oversized_image_bytes_are_rejected(model, monkeypatch):
    monkeypatch.setattr(ms, "MAX_IMAGE_BYTES", 10)
    assert "limit is 10" in _run(model, _detect_instance())["error"]


def test_pixel_limit_is_enforced(model, monkeypatch):
    monkeypatch.setattr(ms, "MAX_IMAGE_PIXELS", 1000)
    assert _run(model, _detect_instance())["status"] == "error"


def test_exif_orientation_is_not_applied(model):
    """Behaviour must match the dlib service: the decoded array is used as stored."""
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees"
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (1, 2, 3)).save(buf, format="JPEG", exif=exif)

    result = _run(model, _detect_instance(buf.getvalue()))

    assert (result["width"], result["height"]) == (200, 100)


def test_truncated_jpegs_still_decode(model):
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (200, 10, 10)).save(buf, format="JPEG")

    assert _run(model, _detect_instance(buf.getvalue()[:-40]))["status"] == "ok"


# ---------------------------------------------------------------------------
# predict(): batching, ordering, concurrency
# ---------------------------------------------------------------------------


def test_predict_matches_instances_in_length_and_order(model):
    payload = {
        "instances": [
            {**_detect_instance(), "id": "a"},
            {"id": "b", "task": "detect", "image": "%%%"},
            {**_encode_instance([BOX]), "id": "c"},
        ]
    }

    result = asyncio.run(model.predict(payload))

    assert [p["id"] for p in result["predictions"]] == ["a", "b", "c"]
    assert [p["status"] for p in result["predictions"]] == ["ok", "error", "ok"]


def test_predict_rejects_malformed_envelopes(model):
    from kserve.errors import InvalidInput

    for payload in ({}, {"instances": []}, {"instances": ["x"]}, [], {"instances": [{}] * (ms.MAX_BATCH_SIZE + 1)}):
        with pytest.raises(InvalidInput):
            asyncio.run(model.predict(payload))


def test_predict_runs_instances_on_executor_threads(model):
    payload = {"instances": [{**_detect_instance(), "id": str(i)} for i in range(8)]}

    result = asyncio.run(model.predict(payload))

    assert all(p["status"] == "ok" for p in result["predictions"])
    assert threading.get_ident() not in model._backend.thread_ids


def test_executor_overlaps_inference_across_threads():
    """Two workers must be inside the backend at the same time (no serialising lock)."""
    barrier = threading.Barrier(2, timeout=5)

    class Rendezvous(FakeBackend):
        def detect(self, rgb, *, min_face_px=None):
            barrier.wait()  # only passes if two instances are in detect() concurrently
            return []

    m = ms.WikiVisageModel("wikivisage", face_workers=2, backend=Rendezvous())
    m.load()
    try:
        payload = {"instances": [{**_detect_instance(), "id": str(i)} for i in range(2)]}
        result = asyncio.run(m.predict(payload))
    finally:
        m.stop()

    assert [p["status"] for p in result["predictions"]] == ["ok", "ok"]


# ---------------------------------------------------------------------------
# load() and startup
# ---------------------------------------------------------------------------


def test_load_marks_ready_with_an_injected_backend():
    m = ms.WikiVisageModel("wikivisage", backend=FakeBackend())
    assert m.ready is False
    assert m.load() is True and m.ready is True
    m.stop()


def test_load_with_real_weights_is_ready_and_reports_the_model(monkeypatch):
    monkeypatch.delenv("WIKIVISAGE_WEIGHTS_DIR", raising=False)
    m = ms.WikiVisageModel("wikivisage", face_workers=1)
    try:
        assert m.load() is True
        result = _run(m, _detect_instance())
        assert result["status"] == "ok"
        assert result["model_id"] == "scrfd500m+arcface_mbf" and result["dim"] == 512
        assert result["faces"] == []
    finally:
        m.stop()


def test_load_fails_closed_on_tampered_weights(tmp_path, monkeypatch):
    for name in be.WEIGHT_SHA256:
        (tmp_path / name).write_bytes((MODEL_SERVER / "weights" / name).read_bytes())
    (tmp_path / be.EMBEDDER_FILE).write_bytes(b"tampered")
    monkeypatch.setenv("WIKIVISAGE_WEIGHTS_DIR", str(tmp_path))

    m = ms.WikiVisageModel("wikivisage")
    with pytest.raises(be.WeightsError):
        m.load()

    assert m.ready is False
    assert m._executor is None


def test_main_exits_nonzero_and_never_starts_the_server_when_weights_do_not_verify(monkeypatch, tmp_path):
    started = []
    monkeypatch.setenv("WIKIVISAGE_WEIGHTS_DIR", str(tmp_path))  # empty: weights missing
    monkeypatch.setattr(ms, "_check_token", lambda: None)
    monkeypatch.setattr(ms.kserve, "ModelServer", lambda **_k: started.append("server"))
    monkeypatch.setattr(sys, "argv", ["model.py", "--model_name", "wikivisage", "--http_port", "8000"])

    with pytest.raises(SystemExit) as exc:
        ms.main()

    assert exc.value.code == 1
    assert started == []


def test_main_keeps_grpc_disabled_and_the_guard_in_front(monkeypatch):
    seen = {}

    class _Server:
        def __init__(self, **kwargs):
            seen["server_kwargs"] = kwargs

        def start(self, models):
            seen["models"] = models

    monkeypatch.setattr(ms, "_check_token", lambda: None)
    monkeypatch.setattr(ms.kserve, "ModelServer", _Server)
    monkeypatch.setattr(ms.WikiVisageModel, "load", lambda self: setattr(self, "ready", True) or True)
    monkeypatch.setattr(
        ms.kserve.model_server.app, "add_middleware", lambda cls, **kw: seen.update(middleware=cls, mw=kw)
    )
    monkeypatch.setattr(sys, "argv", ["model.py", "--model_name", "wikivisage", "--http_port", "8000"])

    ms.main()

    assert seen["server_kwargs"] == {"enable_grpc": False}
    assert seen["middleware"] is ms.RequestGuard
    assert seen["mw"]["open_paths"] == ["/v1/models/wikivisage"]
    assert len(seen["models"]) == 1


def test_token_checks_are_unchanged(monkeypatch):
    monkeypatch.setattr(ms, "FACE_SERVICE_TOKEN", "short")
    with pytest.raises(SystemExit):
        ms._check_token()
    monkeypatch.setattr(ms, "FACE_SERVICE_TOKEN", "")
    monkeypatch.setenv("TOOL_TOOLFORGE_API_URL", "https://example.invalid")
    with pytest.raises(SystemExit):
        ms._check_token()
