"""Tests for the web app: commons."""

from datetime import UTC, datetime

import pytest
import requests

from tests._app_preamble import (
    _THUMB_STEPS,
    _snap_thumb_width,
    app_module,
    commons_thumb_url,
    flask_app,
)
from tests._app_support import (
    _make_authed_client,
    _MockResponse,
)


class _FakeResponse:
    def __init__(
        self, headers=None, chunks=None, error=None, url="https://upload.wikimedia.org/file.jpg", is_redirect=False
    ):
        self.headers = headers or {}
        self._chunks = chunks or []
        self._error = error
        self.closed = False
        self.url = url
        self.is_redirect = is_redirect

    def raise_for_status(self):
        if self._error:
            raise self._error

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True


def test_snap_exact_match():
    assert _snap_thumb_width(330) == 330


def test_snap_rounds_up():
    assert _snap_thumb_width(100) == 120


def test_snap_minimum():
    assert _snap_thumb_width(1) == 20


def test_snap_maximum():
    assert _snap_thumb_width(5000) == 3840


def test_snap_boundary():
    assert _snap_thumb_width(20) == 20
    assert _snap_thumb_width(21) == 40


def test_snap_all_steps():
    for step in _THUMB_STEPS:
        assert _snap_thumb_width(step) == step


def test_snap_between_steps():
    assert _snap_thumb_width(251) == 330
    assert _snap_thumb_width(961) == 1280


def test_commons_thumb_route_requires_login_redirects_to_login():
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_commons_thumb_route_logged_in_redirects_to_generated_thumb(monkeypatch):
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    fake_user = {
        "id": 1,
        "wiki_user_id": 123,
        "wiki_username": "tester",
        "access_token": "token",
        "refresh_token": "refresh",
        "token_expires_at": datetime.now(UTC),
    }

    monkeypatch.setattr(app_module, "execute_query", lambda *args, **kwargs: [fake_user])

    with client.session_transaction() as sess:
        sess["user_id"] = 1

    response = client.get("/commons-thumb/File:X.jpg?width=100")

    assert response.status_code == 302
    assert response.headers["Location"] == commons_thumb_url("File:X.jpg", 100)


def test_download_image_success(monkeypatch):
    resp = _FakeResponse(headers={"Content-Length": "6"}, chunks=[b"ab", b"cd", b"ef"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    data = app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert data == b"abcdef"


def test_download_image_content_length_too_large(monkeypatch):
    resp = _FakeResponse(headers={"Content-Length": "11"}, chunks=[b"abc"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="Image too large"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_stream_exceeds_limit(monkeypatch):
    resp = _FakeResponse(headers={}, chunks=[b"12345", b"67890", b"x"])
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="exceeded"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert resp.closed is True


def test_download_image_raises_http_error(monkeypatch):
    resp = _FakeResponse(error=requests.HTTPError("bad response"))
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(requests.HTTPError):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")


def test_download_image_blocks_untrusted_host():
    with pytest.raises(ValueError, match="Blocked download from untrusted host"):
        app_module._download_image("https://evil.example.com/file.jpg")


def test_download_image_rejects_http_scheme():
    with pytest.raises(ValueError, match="Blocked download from untrusted host"):
        app_module._download_image("http://upload.wikimedia.org/file.jpg")


def test_download_image_rejects_redirect_to_untrusted_host(monkeypatch):
    resp = _FakeResponse(
        headers={"Location": "https://evil.example.com/redirected.jpg"},
        chunks=[],
        is_redirect=True,
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="Redirect to untrusted host"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")
    assert resp.closed is True


def test_download_image_rejects_redirect_to_http_scheme(monkeypatch):
    resp = _FakeResponse(
        headers={"Location": "http://upload.wikimedia.org/redirected.jpg"},
        chunks=[],
        is_redirect=True,
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: resp)

    with pytest.raises(ValueError, match="Redirect to untrusted host"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")
    assert resp.closed is True


def test_download_image_allows_redirect_to_trusted_host(monkeypatch):
    redirect_resp = _FakeResponse(
        headers={"Location": "https://commons.wikimedia.org/redirected.jpg"},
        chunks=[],
        is_redirect=True,
    )
    content_resp = _FakeResponse(
        headers={"Content-Length": "3"},
        chunks=[b"abc"],
    )
    call_count = 0

    def _fake_get(*_a, **_k):
        nonlocal call_count
        call_count += 1
        return redirect_resp if call_count == 1 else content_resp

    monkeypatch.setattr(app_module.requests, "get", _fake_get)

    data = app_module._download_image("https://upload.wikimedia.org/file.jpg", max_bytes=10)
    assert data == b"abc"


def test_download_image_multi_hop_redirect_succeeds(monkeypatch):
    hop1 = _FakeResponse(
        headers={"Location": "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/X.jpg"},
        chunks=[],
        is_redirect=True,
    )
    hop2 = _FakeResponse(
        headers={"Location": "https://upload.wikimedia.org/wikipedia/commons/a/X.jpg"},
        chunks=[],
        is_redirect=True,
    )
    content = _FakeResponse(headers={"Content-Length": "4"}, chunks=[b"data"])
    responses = iter([hop1, hop2, content])

    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: next(responses))

    data = app_module._download_image("https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg", max_bytes=100)
    assert data == b"data"
    assert hop1.closed is True
    assert hop2.closed is True


def test_download_image_follows_redirect_to_thumb_host(monkeypatch):
    """Commons serves ?width=N thumbnails from thumb.wikimedia.org.

    /api/manual-face and /api/update-face-bbox both fetch via Special:FilePath,
    so rejecting that host breaks manual face drawing and bbox editing.
    """
    hop1 = _FakeResponse(
        headers={"Location": "https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/X.jpg"},
        chunks=[],
        is_redirect=True,
    )
    hop2 = _FakeResponse(
        headers={"Location": "https://thumb.wikimedia.org/wikipedia/commons/thumb/6/61/X.jpg/1024px-X.jpg"},
        chunks=[],
        is_redirect=True,
    )
    content = _FakeResponse(headers={"Content-Length": "4"}, chunks=[b"data"])
    responses = iter([hop1, hop2, content])

    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: next(responses))

    data = app_module._download_image(
        "https://commons.wikimedia.org/wiki/Special:FilePath/X.jpg?width=1024", max_bytes=100
    )
    assert data == b"data"


def test_download_image_too_many_redirects_raises(monkeypatch):
    redirect = _FakeResponse(
        headers={"Location": "https://upload.wikimedia.org/loop"},
        chunks=[],
        is_redirect=True,
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_k: redirect)

    with pytest.raises(ValueError, match="Too many redirects"):
        app_module._download_image("https://upload.wikimedia.org/file.jpg")


def test_is_human_entity_true_when_p31_contains_q5(monkeypatch):
    def mock_get(*_a, **_kw):
        return _MockResponse(
            {
                "claims": {
                    "P31": [
                        {"mainsnak": {"datavalue": {"value": {"id": "Q95074"}}}},
                        {"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}},
                    ]
                }
            }
        )

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._is_human_entity("Q42") is True


def test_is_human_entity_false_when_no_p31_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {}}))
    assert app_module._is_human_entity("Q1") is False


def test_is_human_entity_false_when_p31_without_q5(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q215627"}}}}]}}
        ),
    )
    assert app_module._is_human_entity("Q2") is False


def test_is_human_entity_false_on_missing_nested_keys(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P31": [{}]}}))
    assert app_module._is_human_entity("Q3") is False


def test_is_human_entity_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._is_human_entity("Q4") is False


def test_is_human_entity_false_on_request_exception(monkeypatch):
    def mock_get(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    assert app_module._is_human_entity("Q5") is False


def test_is_human_entity_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P31": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._is_human_entity("Q123")

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "Q123"
    assert captured["params"]["property"] == "P31"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_category_exists_true_when_page_exists(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"title": "Category:People"}}}}),
    )
    assert app_module._commons_category_exists("People") is True


def test_commons_category_exists_false_when_only_minus_one_page(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"-1": {"missing": ""}}}}),
    )
    assert app_module._commons_category_exists("Missing") is False


def test_commons_category_exists_true_with_empty_pages_dict(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"query": {"pages": {}}}))
    assert app_module._commons_category_exists("EdgeCase") is True


def test_commons_category_exists_false_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._commons_category_exists("People") is False


def test_commons_category_exists_false_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._commons_category_exists("People") is False


def test_commons_category_exists_calls_commons_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"query": {"pages": {"123": {}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._commons_category_exists("Human faces")

    assert captured["url"] == app_module.COMMONS_API_URL
    assert captured["params"]["action"] == "query"
    assert captured["params"]["titles"] == "Category:Human faces"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_category_has_files_true_when_files_present(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 5, "subcats": 0}}}}}),
    )
    assert app_module._commons_category_has_files("People") is True


def test_commons_category_has_files_true_when_only_subcats(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 0, "subcats": 3}}}}}),
    )
    assert app_module._commons_category_has_files("People") is True


def test_commons_category_has_files_false_when_empty(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {"categoryinfo": {"files": 0, "subcats": 0}}}}}),
    )
    assert app_module._commons_category_has_files("EmptyCat") is False


def test_commons_category_has_files_false_when_no_categoryinfo(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"123": {}}}}),
    )
    assert app_module._commons_category_has_files("NoCatInfo") is False


def test_commons_category_has_files_false_on_missing_page(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"query": {"pages": {"-1": {"missing": ""}}}}),
    )
    assert app_module._commons_category_has_files("Missing") is False


def test_commons_category_has_files_false_on_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._commons_category_has_files("People") is False


def test_fetch_p18_thumb_url_returns_none_when_no_claims(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"claims": {"P18": []}}))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_when_filename_missing(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"claims": {"P18": [{"mainsnak": {"datavalue": {"value": None}}}]}}),
    )
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_returns_none_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._fetch_p18_thumb_url("Q42") is None


def test_fetch_p18_thumb_url_builds_commons_thumb_from_first_claim(monkeypatch):
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {
                "claims": {
                    "P18": [
                        {"mainsnak": {"datavalue": {"value": "First.jpg"}}},
                        {"mainsnak": {"datavalue": {"value": "Second.jpg"}}},
                    ]
                }
            }
        ),
    )
    monkeypatch.setattr(app_module, "commons_thumb_url", lambda file_title, width: f"u:{file_title}:{width}")
    assert app_module._fetch_p18_thumb_url("Q42", width=500) == "u:First.jpg:500"


def test_fetch_p18_thumb_url_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"claims": {"P18": []}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._fetch_p18_thumb_url("Q99", width=120)

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetclaims"
    assert captured["params"]["entity"] == "Q99"
    assert captured["params"]["property"] == "P18"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_fetch_wikidata_label_prefers_current_locale(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "nb")
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse(
            {"entities": {"Q42": {"labels": {"nb": {"value": "Douglas Adams NB"}, "en": {"value": "Douglas Adams"}}}}}
        ),
    )
    assert app_module._fetch_wikidata_label("Q42") == "Douglas Adams NB"


def test_fetch_wikidata_label_falls_back_to_english(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "fr")
    monkeypatch.setattr(
        app_module.requests,
        "get",
        lambda *_a, **_kw: _MockResponse({"entities": {"Q42": {"labels": {"en": {"value": "Douglas Adams"}}}}}),
    )
    assert app_module._fetch_wikidata_label("Q42") == "Douglas Adams"


def test_fetch_wikidata_label_returns_none_when_no_labels(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "es")
    monkeypatch.setattr(
        app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"entities": {"Q42": {"labels": {}}}})
    )
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_returns_none_when_entity_missing(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({"entities": {}}))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_uses_en_when_locale_is_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(app_module, "get_locale", lambda: None)

    def mock_get(url, **kwargs):
        captured.update(kwargs)
        return _MockResponse({"entities": {"Q1": {"labels": {"en": {"value": "Label"}}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    result = app_module._fetch_wikidata_label("Q1")

    assert result == "Label"
    assert captured["params"]["languages"] == "en"


def test_fetch_wikidata_label_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: _MockResponse({}, should_raise=True))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_returns_none_on_request_exception(monkeypatch):
    monkeypatch.setattr(app_module, "get_locale", lambda: "en")
    monkeypatch.setattr(app_module.requests, "get", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app_module._fetch_wikidata_label("Q42") is None


def test_fetch_wikidata_label_calls_wikidata_api_with_expected_params(monkeypatch):
    captured = {}
    monkeypatch.setattr(app_module, "get_locale", lambda: "nb")

    def mock_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _MockResponse({"entities": {"Q99": {"labels": {"nb": {"value": "x"}}}}})

    monkeypatch.setattr(app_module.requests, "get", mock_get)
    app_module._fetch_wikidata_label("Q99")

    assert captured["url"] == app_module.WIKIDATA_API_URL
    assert captured["params"]["action"] == "wbgetentities"
    assert captured["params"]["ids"] == "Q99"
    assert captured["params"]["props"] == "labels"
    assert captured["params"]["languages"] == "nb|en"
    assert captured["params"]["languagefallback"] == "1"
    assert captured["params"]["format"] == "json"
    assert captured["headers"] == {"User-Agent": app_module.USER_AGENT}
    assert captured["timeout"] == 10


def test_commons_thumb_url_regular_jpg():
    url = app_module.commons_thumb_url("File:Example.jpg", 330)
    assert "/330px-Example.jpg" in url


def test_commons_thumb_url_webm_video_uses_double_dash():
    url = app_module.commons_thumb_url("File:Movie.webm", 330)
    assert url.endswith("/330px--Movie.webm.jpg")


def test_commons_thumb_url_ogv_video_uses_double_dash():
    url = app_module.commons_thumb_url("File:Clip.ogv", 330)
    assert url.endswith("/330px--Clip.ogv.jpg")


def test_commons_thumb_url_tif_converts_to_jpg():
    url = app_module.commons_thumb_url("File:Scan.tif", 330)
    assert url.endswith("/330px-Scan.tif.jpg")


def test_commons_thumb_url_tiff_converts_to_jpg():
    url = app_module.commons_thumb_url("File:Scan.tiff", 330)
    assert url.endswith("/330px-Scan.tiff.jpg")


def test_commons_thumb_url_svg_converts_to_png():
    url = app_module.commons_thumb_url("File:Logo.svg", 330)
    assert url.endswith("/330px-Logo.svg.png")


def test_commons_thumb_url_handles_uppercase_extension():
    url = app_module.commons_thumb_url("File:Logo.SVG", 330)
    assert url.endswith("/330px-Logo.SVG.png")


def test_commons_thumb_url_snaps_width_up_to_next_step():
    url = app_module.commons_thumb_url("File:X.jpg", 100)
    assert "/120px-" in url


def test_commons_thumb_url_caps_width_at_max_step():
    url = app_module.commons_thumb_url("File:X.jpg", 9999)
    assert "/3840px-" in url


def test_commons_thumb_url_normalizes_file_prefix_and_spaces():
    url = app_module.commons_thumb_url("File:My Photo.jpg", 330)
    assert "My_Photo.jpg" in url


def test_commons_thumb_route_default_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 330)


def test_commons_thumb_route_custom_width(monkeypatch, fake_user):
    client = _make_authed_client(monkeypatch, fake_user)

    response = client.get("/commons-thumb/File:Example.jpg?width=500")

    assert response.status_code == 302
    assert response.headers["Location"] == app_module.commons_thumb_url("File:Example.jpg", 500)


def test_detection_space_uses_stored_dimensions():
    assert app_module._detection_space(960, 640) == (960, 640)
    assert app_module._detection_space("960", "640") == (960, 640)


@pytest.mark.parametrize("width,height", [(None, 640), (960, None), (0, 640), (960, -1), ("x", 640)])
def test_detection_space_is_none_when_dimensions_unknown(width, height):
    assert app_module._detection_space(width, height) is None


def test_file_path_url_uses_the_shared_detection_width():
    import config
    import worker

    assert config.DETECTION_WIDTH == 960
    assert app_module.FILE_PATH_URL == worker.FILE_PATH_URL
    assert app_module.FILE_PATH_URL.endswith("?width=960")
    assert "{file_title}" in app_module.FILE_PATH_URL
    assert config.DETECTION_WIDTH in app_module._THUMB_STEPS
