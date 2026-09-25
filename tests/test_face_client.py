import base64
from unittest.mock import MagicMock, patch

import pytest
import requests

import face_client


def _encoding_bytes(seed: int = 1) -> bytes:
    return bytes([(seed + i) % 256 for i in range(face_client.ENCODING_BYTES)])


def _encoded(seed: int = 1) -> str:
    return base64.b64encode(_encoding_bytes(seed)).decode("ascii")


def _ok_prediction(*, key: str = "0", faces=None, width: int = 120, height: int = 80):
    if faces is None:
        faces = [
            {
                "box": {"top": 1, "right": 5, "bottom": 9, "left": 2},
                "encoding": _encoded(),
            }
        ]
    return {"id": key, "status": "ok", "width": width, "height": height, "faces": faces}


def _response(*, status_code=200, json_body=None, text="", json_exc=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_exc is not None:
        resp.json.side_effect = json_exc
    else:
        resp.json.return_value = json_body
    return resp


def test_detect_faces_parses_successful_response():
    session = MagicMock()
    session.post.return_value = _response(json_body={"predictions": [_ok_prediction()]})

    with patch("face_client.get_session", return_value=session):
        result = face_client.detect_faces(b"image-bytes")

    assert result.width == 120
    assert result.height == 80
    assert len(result) == 1
    assert result.locations == [(1, 5, 9, 2)]
    assert result.encodings == [_encoding_bytes()]


def test_detect_faces_rejected_prediction_raises_faceservicerejected():
    session = MagicMock()
    session.post.return_value = _response(
        json_body={"predictions": [{"id": "0", "status": "error", "error": "bad image"}]}
    )

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceRejected, match="bad image"):
            face_client.detect_faces(b"image-bytes")


def test_detect_faces_http_500_raises_unavailable():
    session = MagicMock()
    session.post.return_value = _response(status_code=500, text="server exploded")

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceUnavailable, match="HTTP 500"):
            face_client.detect_faces(b"image-bytes")


def test_detect_faces_non_json_response_raises_unavailable():
    session = MagicMock()
    session.post.return_value = _response(json_exc=ValueError("not json"))

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceUnavailable, match="non-JSON"):
            face_client.detect_faces(b"image-bytes")


def test_detect_faces_prediction_count_mismatch_raises_unavailable():
    session = MagicMock()
    session.post.return_value = _response(json_body={"predictions": []})

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceUnavailable, match="returned 0 predictions for 1 instances"):
            face_client.detect_faces(b"image-bytes")


def test_detect_faces_wrong_encoding_length_raises_unavailable():
    bad_encoding = base64.b64encode(b"too-short").decode("ascii")
    session = MagicMock()
    session.post.return_value = _response(
        json_body={
            "predictions": [
                _ok_prediction(
                    faces=[{"box": {"top": 1, "right": 2, "bottom": 3, "left": 4}, "encoding": bad_encoding}]
                )
            ]
        }
    )

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceUnavailable, match="expected 1024"):
            face_client.detect_faces(b"image-bytes")


def test_detect_faces_invalid_base64_raises_unavailable():
    session = MagicMock()
    session.post.return_value = _response(
        json_body={
            "predictions": [
                _ok_prediction(faces=[{"box": {"top": 1, "right": 2, "bottom": 3, "left": 4}, "encoding": "%%%"}])
            ]
        }
    )

    with patch("face_client.get_session", return_value=session):
        with pytest.raises(face_client.FaceServiceUnavailable, match="not valid base64"):
            face_client.detect_faces(b"image-bytes")


def test_chunk_splits_on_count_limit_and_byte_limit():
    items = [(str(i), b"x" * size) for i, size in enumerate([5, 5, 7, 4, 9], start=1)]

    with (
        patch.object(face_client, "FACE_SERVICE_BATCH_SIZE", 2),
        patch.object(face_client, "FACE_SERVICE_MAX_BATCH_BYTES", 10),
    ):
        batches = list(face_client._chunk(items))

    assert [[key for key, _ in batch] for batch in batches] == [["1", "2"], ["3"], ["4"], ["5"]]


def test_detect_faces_batch_maps_transport_failure_per_batch_only():
    items = [("1", b"a"), ("2", b"b"), ("3", b"c")]

    with (
        patch.object(face_client, "FACE_SERVICE_BATCH_SIZE", 2),
        patch(
            "face_client._post",
            side_effect=[
                face_client.FaceServiceUnavailable("transport down"),
                [{"id": "3", "status": "ok", "width": 9, "height": 7, "faces": []}],
            ],
        ),
    ):
        result = face_client.detect_faces_batch(items)

    assert isinstance(result["1"], face_client.FaceServiceUnavailable)
    assert isinstance(result["2"], face_client.FaceServiceUnavailable)
    assert result["1"] is result["2"]
    assert isinstance(result["3"], face_client.DetectionResult)
    assert result["3"].width == 9
    assert result["3"].height == 7


def test_encode_known_face_returns_none_on_empty_faces():
    with patch("face_client._post", return_value=[{"id": "0", "status": "ok", "width": 10, "height": 20, "faces": []}]):
        result = face_client.encode_known_face(b"image", (1, 2, 3, 4))

    assert result is None


def test_encode_known_face_returns_none_on_rejected():
    with patch(
        "face_client._post",
        return_value=[{"id": "0", "status": "error", "error": "box unusable"}],
    ):
        result = face_client.encode_known_face(b"image", (1, 2, 3, 4))

    assert result is None


def test_is_healthy_false_on_exception():
    session = MagicMock()
    session.get.side_effect = requests.RequestException("boom")

    with patch("face_client.get_session", return_value=session):
        assert face_client.is_healthy() is False


def test_is_healthy_false_on_ready_false():
    session = MagicMock()
    session.get.return_value = _response(json_body={"ready": False})

    with patch("face_client.get_session", return_value=session):
        assert face_client.is_healthy() is False


def test_is_healthy_true_when_ready_and_token_accepted():
    session = MagicMock()
    session.get.side_effect = [_response(json_body={"ready": True}), _response(json_body={"models": []})]

    with patch("face_client.get_session", return_value=session):
        assert face_client.is_healthy() is True

    probed = [call.args[0] for call in session.get.call_args_list]
    assert probed == [face_client.health_url(), f"{face_client.FACE_SERVICE_URL}/v1/models"]


def test_is_healthy_false_when_token_rejected_even_though_ready():
    """The readiness path is open, so only the second probe can catch a bad token."""
    session = MagicMock()
    session.get.side_effect = [_response(json_body={"ready": True}), _response(status_code=401)]

    with patch("face_client.get_session", return_value=session):
        assert face_client.is_healthy() is False


def test_is_healthy_false_when_auth_probe_errors():
    session = MagicMock()
    session.get.side_effect = [_response(json_body={"ready": True}), _response(status_code=503)]

    with patch("face_client.get_session", return_value=session):
        assert face_client.is_healthy() is False


def test_get_session_caches_and_reset_session_rebuilds():
    built_sessions = [MagicMock(), MagicMock()]

    with patch("face_client.requests.Session", side_effect=built_sessions):
        face_client.reset_session()
        first = face_client.get_session()
        second = face_client.get_session()
        assert first is second
        face_client.reset_session()
        third = face_client.get_session()

    assert third is not first
    assert built_sessions[0].mount.call_count == 2
    built_sessions[0].close.assert_called_once()


def test_non_object_prediction_is_unavailable_not_a_crash():
    with patch("face_client._post", return_value=[None]):
        results = face_client.detect_faces_batch([("a", b"img")])

    assert isinstance(results["a"], face_client.FaceServiceUnavailable)


def test_oversized_image_is_rejected_without_a_request(monkeypatch):
    """Above the server's cap the body would hit HTTP 413, which is retryable."""
    monkeypatch.setattr(face_client, "FACE_SERVICE_MAX_IMAGE_BYTES", 10)
    with patch("face_client._post", return_value=[_ok_prediction(key="ok")]) as post:
        results = face_client.detect_faces_batch([("big", b"x" * 11), ("ok", b"x" * 10)])

    assert isinstance(results["big"], face_client.FaceServiceRejected)
    assert isinstance(results["ok"], face_client.DetectionResult)
    assert [i["id"] for i in post.call_args.args[0]] == ["ok"]


def test_encode_known_face_returns_none_for_oversized_image(monkeypatch):
    monkeypatch.setattr(face_client, "FACE_SERVICE_MAX_IMAGE_BYTES", 10)
    with patch("face_client._post") as post:
        assert face_client.encode_known_face(b"x" * 11, (1, 2, 3, 4)) is None
    post.assert_not_called()
