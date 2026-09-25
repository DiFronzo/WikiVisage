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

No real faces are needed — a synthetic image validates the transport, parsing,
dimensions, and error contract. Encoding-value equivalence against in-process
dlib is a separate local check (see TESTING.md), since it needs dlib installed.
"""

from __future__ import annotations

import base64
import io

import pytest
import requests

import face_client
from face_client import DetectionResult, FaceServiceRejected, FaceServiceUnavailable

pytestmark = pytest.mark.contract


def _synthetic_png(width: int = 320, height: int = 240) -> bytes:
    """Build a valid PNG in-process so the tests need no network or fixtures."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    return _synthetic_png()


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


def test_every_returned_encoding_is_exactly_1024_bytes(image_bytes):
    """Guards the DB BLOB contract from both sides at once."""
    result = face_client.detect_faces(image_bytes)
    for encoding in result.encodings:
        assert len(encoding) == face_client.ENCODING_BYTES


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
