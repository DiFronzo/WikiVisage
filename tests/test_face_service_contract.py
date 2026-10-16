"""Contract tests: real `face_client` against a real running face service.

Everything else in the suite mocks the boundary. These tests exercise both
sides of it for real, which is the only way to catch a broken Dockerfile, an
unresolvable dependency, a KServe API change, or a wire-format drift between
`face_client.py` and `model-server/model.py`.

Skipped unless ``WIKIVISAGE_FACE_SERVICE_CONTRACT=1`` and a service is
reachable at ``WIKIVISAGE_FACE_SERVICE_URL``. CI starts the container and sets
both; locally::

    cd model-server && docker compose up --build
    WIKIVISAGE_FACE_SERVICE_CONTRACT=1 pytest tests/test_face_service_contract.py -v

Synthetic images validate the transport, parsing, dimensions, embedding shape
and error contract. Tests that need a real face use a committed public-domain
portrait, ``tests/data/portrait_sample.jpg`` (see ``tests/data/README.md``); set
``WIKIVISAGE_TEST_SAMPLE_IMAGE`` to try another photo. If the file is absent they
skip locally, but fail when ``WIKIVISAGE_REQUIRE_SAMPLE=1`` (as CI sets it), so a
missing sample can never silently turn the face tests off.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import pytest
import requests

import face_client
from face_client import DetectionResult, FaceServiceRejected, FaceServiceUnavailable

try:
    from conftest import require_sample_env
except ModuleNotFoundError:
    from tests.conftest import require_sample_env

pytestmark = pytest.mark.contract


def _synthetic_png(width: int = 320, height: int = 240) -> bytes:
    """Build a valid PNG in-process so the tests need no network or fixtures."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _noise_png(width: int = 320, height: int = 240, seed: int = 0) -> bytes:
    """A textured image, so different regions embed differently (a flat one would not)."""
    from PIL import Image

    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


SAMPLE = Path(
    os.environ.get("WIKIVISAGE_TEST_SAMPLE_IMAGE", Path(__file__).resolve().parent / "data" / "portrait_sample.jpg")
)


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    return _synthetic_png()


@pytest.fixture(scope="module")
def portrait() -> bytes:
    """A public-domain photo with exactly one frontal face."""
    if SAMPLE.exists():
        return SAMPLE.read_bytes()
    if require_sample_env():
        pytest.fail(
            f"{SAMPLE} is missing but WIKIVISAGE_REQUIRE_SAMPLE is set — commit the sample (tests/data/README.md)"
        )
    pytest.skip(f"{SAMPLE} not committed yet (tests/data/README.md)")


def _unit_norm(encoding: bytes) -> float:
    return float(np.linalg.norm(np.frombuffer(encoding, dtype="<f4")))


def _post_instances(instances: list[dict]) -> list[dict]:
    resp = face_client.get_session().post(face_client.predict_url(), json={"instances": instances}, timeout=120)
    assert resp.status_code == 200, resp.text
    return resp.json()["predictions"]


def test_service_reports_ready():
    assert face_client.is_healthy(), f"face service not ready at {face_client.health_url()} — is the container up?"


def test_detect_returns_parsed_result_with_correct_dimensions(image_bytes):
    result = face_client.detect_faces(image_bytes)

    assert isinstance(result, DetectionResult)
    assert (result.width, result.height) == (320, 240)
    # A flat grey image has no faces; the point is that the envelope parses.
    assert result.faces == []
    assert result.locations == []
    assert result.encodings == []


def test_prediction_reports_model_id_and_dim(image_bytes):
    result = face_client.detect_faces(image_bytes)
    assert result.model_id == face_client.EXPECTED_MODEL_ID == "scrfd500m+arcface_mbf"
    assert result.dim == face_client.EMBEDDING_DIM == 512


def test_raw_prediction_carries_model_id_dim_and_no_legacy_fields(image_bytes):
    [prediction] = _post_instances([{"id": "0", "task": "detect", "image": _b64(image_bytes)}])
    assert prediction["model_id"] == "scrfd500m+arcface_mbf"
    assert prediction["dim"] == 512
    assert prediction["faces"] == []


def test_startup_probe_reports_the_expected_model():
    assert face_client.check_service_model() == ("scrfd500m+arcface_mbf", 512)


def test_every_returned_encoding_is_dim_float32_values(image_bytes):
    """Guards the DB BLOB contract from both sides at once: 512 x float32 = 2048 bytes."""
    result = face_client.detect_faces(image_bytes)
    for encoding in result.encodings:
        assert len(encoding) == face_client.EMBEDDING_DIM * 4 == 2048


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_corrupt_image_is_rejected_not_retried(image_bytes):
    """A terminal rejection must not masquerade as a transient outage.

    The worker branches on exactly this distinction: FaceServiceRejected marks
    the image 'error', FaceServiceUnavailable leaves it pending forever.
    """
    with pytest.raises(FaceServiceRejected):
        face_client.detect_faces(b"this is definitely not an image")


def test_batch_isolates_a_bad_image_from_its_siblings(image_bytes):
    items = [("good-1", image_bytes), ("bad", b"not an image"), ("good-2", image_bytes)]

    results = face_client.detect_faces_batch(items)

    assert set(results) == {"good-1", "bad", "good-2"}
    assert isinstance(results["good-1"], DetectionResult)
    assert isinstance(results["good-2"], DetectionResult)
    assert isinstance(results["bad"], FaceServiceRejected)


def test_batch_larger_than_one_request_still_returns_every_key(image_bytes):
    """Exercises the client's chunking against the server's batch ceiling."""
    count = face_client.FACE_SERVICE_BATCH_SIZE * 2 + 1
    items = [(str(i), image_bytes) for i in range(count)]

    results = face_client.detect_faces_batch(items)

    assert set(results) == {str(i) for i in range(count)}
    assert all(isinstance(v, DetectionResult) for v in results.values())


def test_encode_known_box_returns_none_for_an_unusable_region(image_bytes):
    """Drives the HTTP 422 path in /api/manual-face and /api/update-face-bbox."""
    assert face_client.encode_known_face(image_bytes, (10, 15, 15, 10)) is None


def test_encode_without_a_face_falls_back_to_an_unaligned_unit_vector():
    encoded = face_client.encode_known_face(_noise_png(), (20, 120, 120, 20))

    assert encoded is not None
    encoding, model_id, aligned = encoded
    assert len(encoding) == 2048
    assert _unit_norm(encoding) == pytest.approx(1.0, abs=1e-4)
    assert model_id == "scrfd500m+arcface_mbf"
    assert aligned is False


def test_encode_box_space_scales_the_box_onto_the_decoded_image():
    """The same region, described in a 2x larger space, must embed identically."""
    image = _noise_png(320, 240)
    direct = face_client.encode_known_face(image, (20, 120, 120, 20))
    scaled = face_client.encode_known_face(image, (40, 240, 240, 40), box_space=(640, 480))
    unscaled = face_client.encode_known_face(image, (40, 240, 240, 40))

    assert direct is not None and scaled is not None and unscaled is not None
    assert scaled.encoding == direct.encoding
    # Without box_space the same numbers name a different region, hence a different embedding.
    assert unscaled.encoding != direct.encoding


def test_encode_returns_the_box_in_decoded_image_pixels():
    image = _noise_png(320, 240)
    box = {"top": 40, "right": 240, "bottom": 240, "left": 40}
    [prediction] = _post_instances(
        [
            {
                "id": "0",
                "task": "encode",
                "image": _b64(image),
                "boxes": [box],
                "box_space": {"width": 640, "height": 480},
            }
        ]
    )
    assert (prediction["width"], prediction["height"]) == (320, 240)
    assert prediction["faces"][0]["box"] == {"top": 20, "right": 120, "bottom": 120, "left": 20}
    assert prediction["faces"][0]["aligned"] is False
    assert prediction["faces"][0]["score"] is None


def test_encode_rejects_a_malformed_box_space(image_bytes):
    [prediction] = _post_instances(
        [
            {
                "id": "0",
                "task": "encode",
                "image": _b64(image_bytes),
                "boxes": [{"top": 10, "right": 60, "bottom": 60, "left": 10}],
                "box_space": {"width": 0, "height": 10},
            }
        ]
    )
    assert prediction["status"] == "error"


def test_encode_rejects_a_box_outside_the_image(image_bytes):
    box = {"top": 2000, "right": 3000, "bottom": 2100, "left": 2900}
    [prediction] = _post_instances([{"id": "0", "task": "encode", "image": _b64(image_bytes), "boxes": [box]}])
    assert prediction["status"] == "error"


def test_detect_finds_exactly_one_face_in_the_portrait(portrait):
    result = face_client.detect_faces(portrait)

    assert len(result.faces) == 1
    face = result.faces[0]
    assert face.score is not None and face.score > 0.6
    assert len(face.encoding) == 2048
    assert _unit_norm(face.encoding) == pytest.approx(1.0, abs=1e-4)
    assert 0 <= face.left < face.right <= result.width and 0 <= face.top < face.bottom <= result.height


def test_encode_recovers_landmarks_for_a_hand_drawn_box_around_the_face(portrait):
    detected = face_client.detect_faces(portrait)
    face = detected.faces[0]
    drawn = (face.top + 8, face.right - 6, face.bottom - 4, face.left + 6)  # a slightly sloppy box

    encoded = face_client.encode_known_face(portrait, drawn, box_space=(detected.width, detected.height))

    assert encoded is not None and encoded.aligned is True
    a = np.frombuffer(encoded.encoding, dtype="<f4")
    b = np.frombuffer(face.encoding, dtype="<f4")
    assert float(a @ b) > 0.9


def test_encode_falls_back_to_unaligned_when_the_box_holds_no_face(portrait):
    detected = face_client.detect_faces(portrait)
    face = detected.faces[0]
    # A box well clear of the face, in the opposite corner of the image.
    size = min(face.right - face.left, face.bottom - face.top) // 2
    if face.left > detected.width - face.right:
        box = (detected.height - size - 5, size + 5, detected.height - 5, 5)
    else:
        box = (detected.height - size - 5, detected.width - 5, detected.height - 5, detected.width - size - 5)

    encoded = face_client.encode_known_face(portrait, box, box_space=(detected.width, detected.height))

    assert encoded is not None and encoded.aligned is False


def test_server_rejects_an_oversized_batch(image_bytes):
    """The server's own MAX_BATCH_SIZE must reject, not silently truncate."""
    payload = {
        "instances": [
            {"id": str(i), "task": "detect", "image": base64.b64encode(image_bytes).decode("ascii")}
            for i in range(face_client.FACE_SERVICE_BATCH_SIZE + 64)
        ]
    }
    resp = face_client.get_session().post(face_client.predict_url(), json=payload, timeout=60)
    assert resp.status_code == 400


def test_unreachable_service_raises_unavailable(image_bytes, monkeypatch):
    """Transport failure must surface as the retryable error type."""
    monkeypatch.setattr(face_client, "FACE_SERVICE_URL", "http://127.0.0.1:9")
    face_client.reset_session()
    try:
        with pytest.raises(FaceServiceUnavailable):
            face_client.detect_faces(image_bytes)
    finally:
        face_client.reset_session()


requires_token = pytest.mark.skipif(
    not face_client.FACE_SERVICE_TOKEN,
    reason="set WIKIVISAGE_FACE_SERVICE_TOKEN on both the container and the client to test auth",
)


@requires_token
def test_health_path_is_open_without_a_token():
    """Toolforge's HTTP health check sends no headers and must still pass."""
    resp = requests.get(face_client.health_url(), timeout=10)
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@requires_token
def test_predict_without_a_token_is_rejected(image_bytes):
    payload = {"instances": [{"id": "0", "task": "detect", "image": base64.b64encode(image_bytes).decode("ascii")}]}
    resp = requests.post(face_client.predict_url(), json=payload, timeout=10)
    assert resp.status_code == 401


@requires_token
def test_model_unload_without_a_token_is_rejected():
    """KServe's unload endpoint would otherwise be a public off switch."""
    resp = requests.post(f"{face_client.FACE_SERVICE_URL}/v2/repository/models/wikivisage/unload", timeout=10)
    assert resp.status_code == 401
    assert face_client.is_healthy()


@requires_token
def test_wrong_token_makes_the_client_report_unhealthy(monkeypatch):
    monkeypatch.setattr(face_client, "FACE_SERVICE_TOKEN", "w" * 40)
    face_client.reset_session()
    try:
        assert face_client.is_healthy() is False
    finally:
        face_client.reset_session()


def test_oversized_request_body_is_refused():
    resp = face_client.get_session().post(
        face_client.predict_url(),
        data=b"x" * (41 * 1024 * 1024),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    assert resp.status_code == 413
