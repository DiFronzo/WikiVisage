"""Unit tests for the experimental LiftWing pre-filter client.

Pure mocks: no network, no database. The contract under test is "fail open":
every failure returns ``None`` (or ``(False, reason)`` from ``probe``) instead
of raising, and a circuit breaker stops calls after repeated failures.
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

import liftwing_client
from liftwing_client import (
    LiftWingClient,
    LiftWingUnavailable,
    MetadataItem,
    Verdict,
    parse_verdicts,
)


def _response(status_code: int = 200, payload=None, json_error: bool = False, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers if headers is not None else {}
    if json_error:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


def _client(session: MagicMock, **kwargs) -> LiftWingClient:
    kwargs.setdefault("url", "https://example.invalid/openai/v1/chat/completions")
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("api_token", "")  # tests opt in to auth explicitly
    kwargs.setdefault("min_call_interval", 0)  # request spacing is tested explicitly
    return LiftWingClient(session=session, **kwargs)


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# ---------------------------------------------------------------------------
# parse_verdicts
# ---------------------------------------------------------------------------


def test_parse_verdicts_plain_array():
    verdicts = parse_verdicts('[{"page_id": 1, "keep": true, "reason": "portrait"}]')

    assert verdicts == {1: Verdict(page_id=1, keep=True, reason="portrait")}


def test_parse_verdicts_strips_think_block_and_fence():
    content = '<think>let me reason about this</think>\n```json\n[{"page_id": 7, "keep": false, "reason": "map"}]\n```'

    verdicts = parse_verdicts(content)

    assert verdicts == {7: Verdict(page_id=7, keep=False, reason="map")}


def test_parse_verdicts_strips_stray_think_tags():
    verdicts = parse_verdicts('</think> [{"page_id": 3, "keep": true, "reason": "ok"}]')

    assert 3 in verdicts


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), ("true", True), ("No", False), ("keep", True)],
)
def test_parse_verdicts_coerces_keep_values(value, expected):
    content = json.dumps([{"page_id": 5, "keep": value, "reason": "r"}])

    verdicts = parse_verdicts(content)

    assert verdicts[5].keep is expected


def test_parse_verdicts_skips_invalid_entries_and_truncates_reason():
    content = (
        "["
        '{"page_id": "not-a-number", "keep": true},'
        '{"page_id": 9},'
        '"just a string",'
        f'{{"page_id": 11, "keep": false, "reason": "{"x" * 300}"}}'
        "]"
    )

    verdicts = parse_verdicts(content)

    assert set(verdicts) == {11}
    assert len(verdicts[11].reason) == liftwing_client.MAX_REASON_CHARS


@pytest.mark.parametrize(
    "content",
    [
        "no json at all",
        "[not valid json}",
        '{"page_id": 1, "keep": true}',
        "[]",
        '[{"page_id": 1, "note": "no keep value"}]',
    ],
)
def test_parse_verdicts_raises_when_unusable(content):
    with pytest.raises(LiftWingUnavailable):
        parse_verdicts(content)


# ---------------------------------------------------------------------------
# filter_metadata — success path
# ---------------------------------------------------------------------------


def test_filter_metadata_returns_verdicts():
    session = MagicMock()
    session.post.return_value = _response(
        payload=_completion(
            '[{"page_id": 501, "keep": true, "reason": "person"}, {"page_id": 502, "keep": false, "reason": "a map"}]'
        )
    )
    client = _client(session)
    items = [
        MetadataItem(
            page_id=501, title="File:A.jpg", description="a person", categories=("Category:X",), mime="image/jpeg"
        ),
        MetadataItem(page_id=502, title="File:B.png", description="a map", categories=(), mime="image/png"),
    ]

    verdicts = client.filter_metadata(items)

    assert verdicts == {
        501: Verdict(page_id=501, keep=True, reason="person"),
        502: Verdict(page_id=502, keep=False, reason="a map"),
    }
    body = session.post.call_args.kwargs["json"]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["temperature"] == 0
    user_content = body["messages"][1]["content"]
    assert "File:A.jpg" in user_content
    assert "a person" in user_content
    # The Category: prefix is stripped and the MIME type is not sent at all.
    assert '"X"' in user_content
    assert "Category:" not in user_content
    assert "mime" not in user_content


def test_filter_metadata_settles_non_visual_media_without_a_model_call():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 3, "keep": true, "reason": "person"}]'))
    client = _client(session)
    items = [
        MetadataItem(page_id=1, title="File:Sig.svg", mime="image/svg+xml"),
        MetadataItem(page_id=2, title="File:S.ogg", mime="application/ogg"),
        MetadataItem(page_id=3, title="File:A.jpg", mime="image/jpeg"),
    ]

    verdicts = client.filter_metadata(items)

    assert verdicts == {
        1: Verdict(page_id=1, keep=False, reason="non-visual media (image/svg+xml)"),
        2: Verdict(page_id=2, keep=False, reason="non-visual media (application/ogg)"),
        3: Verdict(page_id=3, keep=True, reason="person"),
    }
    # Only the visual file went over the wire.
    payload = session.post.call_args.kwargs["json"]
    sent = json.loads(payload["messages"][1]["content"])
    assert [entry["page_id"] for entry in sent] == [3]


def test_filter_metadata_non_visual_batch_makes_no_request():
    session = MagicMock()
    client = _client(session)
    items = [
        MetadataItem(page_id=1, title="File:Sig.svg", mime="image/svg+xml"),
        MetadataItem(page_id=2, title="File:Doc.pdf", mime="application/pdf"),
    ]

    verdicts = client.filter_metadata(items)

    assert verdicts == {
        1: Verdict(page_id=1, keep=False, reason="non-visual media (image/svg+xml)"),
        2: Verdict(page_id=2, keep=False, reason="non-visual media (application/pdf)"),
    }
    session.post.assert_not_called()


def test_filter_metadata_unknown_mime_still_goes_to_the_model():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 1, "keep": true}]'))
    client = _client(session)

    verdicts = client.filter_metadata([MetadataItem(page_id=1, title="File:A", mime="")])

    assert verdicts == {1: Verdict(page_id=1, keep=True, reason="")}
    session.post.assert_called_once()


def test_build_messages_truncates_descriptions_and_strips_category_prefix():
    client = _client(MagicMock())
    long_description = "x" * (liftwing_client.MAX_DESCRIPTION_CHARS + 50)

    messages = client._build_messages(
        [
            MetadataItem(
                page_id=1,
                title="File:A.jpg",
                description=long_description,
                categories=("Category:People",),
                mime="image/jpeg",
            )
        ]
    )

    entry = json.loads(messages[1]["content"])[0]
    assert len(entry["description"]) == liftwing_client.MAX_DESCRIPTION_CHARS
    assert entry["categories"] == ["People"]


def test_filter_metadata_empty_items_makes_no_request():
    session = MagicMock()
    client = _client(session)

    assert client.filter_metadata([]) == {}
    session.post.assert_not_called()


# ---------------------------------------------------------------------------
# filter_metadata — fail open
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        _response(status_code=500, payload={}),
        _response(status_code=429, payload={}),
        _response(status_code=200, json_error=True),
        _response(status_code=200, payload={"choices": []}),
        _response(status_code=200, payload=_completion("")),
        _response(status_code=200, payload=_completion("I cannot answer that.")),
    ],
)
def test_filter_metadata_fails_open(response):
    session = MagicMock()
    session.post.return_value = response
    client = _client(session)

    assert client.filter_metadata([MetadataItem(page_id=1, title="File:A.jpg")]) is None


def test_filter_metadata_fails_open_on_transport_error():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout("too slow")
    client = _client(session)

    assert client.filter_metadata([MetadataItem(page_id=1, title="File:A.jpg")]) is None


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_opens_after_threshold_and_stops_calling():
    session = MagicMock()
    session.post.return_value = _response(status_code=500, payload={})
    client = _client(session, circuit_failure_threshold=2, circuit_cooldown=600)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is None
    assert client.filter_metadata([item]) is None  # threshold reached, breaker opens
    assert client.filter_metadata([item]) is None  # breaker open: no HTTP call

    assert session.post.call_count == 2


def test_circuit_resets_after_success():
    session = MagicMock()
    session.post.side_effect = [
        _response(status_code=500, payload={}),
        _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]')),
        _response(status_code=500, payload={}),
    ]
    client = _client(session, circuit_failure_threshold=2, circuit_cooldown=600)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is None  # failure 1
    assert client.filter_metadata([item]) == {1: Verdict(page_id=1, keep=True, reason="ok")}
    assert client.filter_metadata([item]) is None  # failure counter was reset, breaker stays closed

    assert session.post.call_count == 3


def test_queued_calls_see_the_breaker_the_running_call_opens():
    """Regression: the breaker used to be checked outside the request lock, so
    threads already inside filter_metadata each sent their own request after the
    429 that should have stopped them — exactly the hammering the breaker is
    there to prevent."""
    session = MagicMock()
    in_request = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def _post(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            in_request.set()
            assert release.wait(timeout=5)
        return _response(status_code=429, payload={}, headers={"Retry-After": "600"})

    session.post.side_effect = _post
    client = _client(session, circuit_failure_threshold=5)
    item = MetadataItem(page_id=1, title="File:A.jpg")
    results: list[dict | None] = []

    threads = [threading.Thread(target=lambda: results.append(client.filter_metadata([item])))]
    threads[0].start()
    assert in_request.wait(timeout=5)  # the first call is inside its HTTP request
    threads += [threading.Thread(target=lambda: results.append(client.filter_metadata([item]))) for _ in range(7)]
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.2)  # let the queued threads reach the breaker check
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert len(calls) == 1  # only the first call ever reached the wire
    assert results == [None] * 8


# ---------------------------------------------------------------------------
# Auth and Retry-After (H4)
# ---------------------------------------------------------------------------


def test_requests_send_bearer_token_when_configured():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]'))
    client = _client(session, api_token="secret-token")

    assert client.filter_metadata([MetadataItem(page_id=1, title="File:A.jpg")]) is not None

    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["User-Agent"].startswith("WikiVisage/")


def test_requests_omit_authorization_without_token():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]'))
    client = _client(session)

    client.filter_metadata([MetadataItem(page_id=1, title="File:A.jpg")])

    assert "Authorization" not in session.post.call_args.kwargs["headers"]


def test_429_opens_circuit_immediately_and_honours_retry_after():
    session = MagicMock()
    session.post.return_value = _response(status_code=429, payload={}, headers={"Retry-After": "1200"})
    client = _client(session, circuit_failure_threshold=5, circuit_cooldown=600)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is None
    assert client.filter_metadata([item]) is None  # breaker open: no second HTTP call

    assert session.post.call_count == 1
    assert client._consecutive_failures == 1
    assert client._disabled_until - time.monotonic() > 1100  # Retry-After beats the default cooldown
    assert client._rate_limited() is True


def test_parse_retry_after_accepts_seconds_and_http_date():
    import email.utils
    from datetime import UTC, datetime, timedelta

    assert liftwing_client._parse_retry_after("120") == 120.0
    assert liftwing_client._parse_retry_after(None) is None
    assert liftwing_client._parse_retry_after("garbage") is None

    when = datetime.now(UTC) + timedelta(seconds=60)
    parsed = liftwing_client._parse_retry_after(email.utils.format_datetime(when))

    assert parsed is not None
    assert 50 <= parsed <= 61


# ---------------------------------------------------------------------------
# prompt (M1)
# ---------------------------------------------------------------------------


def test_system_prompt_keeps_non_photographic_media():
    prompt = liftwing_client._SYSTEM_PROMPT

    # Faces in paintings, drawings and sculptures must not be thrown away.
    assert "paintings" in prompt
    assert "sculptures" in prompt
    assert "drawings" in prompt
    # And uncertainty must still mean keep.
    assert "unsure" in prompt


def test_system_prompt_names_the_biography_noise_and_drop_reasons():
    prompt = liftwing_client._SYSTEM_PROMPT

    # The negatives that dominate Commons biography categories.
    for negative in ("signatures", "gravestones", "memorial plaques", "landscapes", "coats of arms"):
        assert negative in prompt
    # Fences are banned in the prompt (the parser also strips them), and a
    # reason is only asked for on drops to save output tokens on keeps.
    assert "code fences" in prompt
    assert '"reason"' in prompt


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def test_probe_runs_a_minimal_real_completion():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion("pong"))
    client = _client(session)

    assert client.probe() == (True, None)

    session.get.assert_not_called()
    body = session.post.call_args.kwargs["json"]
    assert body["max_tokens"] == liftwing_client.PROBE_MAX_TOKENS
    assert "pong" in body["messages"][0]["content"]


def test_probe_429_reports_failure_and_opens_the_breaker():
    session = MagicMock()
    session.post.return_value = _response(status_code=429, payload={}, headers={"Retry-After": "120"})
    client = _client(session, circuit_failure_threshold=5, circuit_cooldown=600)

    up, error = client.probe()

    assert up is False
    assert "429" in error
    assert client._consecutive_failures == 1
    assert client._circuit_open() is True
    assert client._rate_limited() is True


def test_probe_honours_the_429_deadline_and_a_success_does_not_clear_it():
    """Regression: a probe used to fire regardless of Retry-After, and a later
    success cleared the deadline, spending quota the server had denied."""
    session = MagicMock()
    session.post.side_effect = [
        _response(status_code=429, payload={}, headers={"Retry-After": "1200"}),
        _response(payload=_completion("pong")),
    ]
    client = _client(session, circuit_failure_threshold=5, circuit_cooldown=600)

    assert client.probe() == (False, "HTTP 429")  # learns the deadline
    assert client._rate_limited() is True
    assert client._rate_limited_until - time.monotonic() > 1100

    calls = session.post.call_count
    up, error = client.probe()
    assert up is False
    assert "rate limited" in error
    assert session.post.call_count == calls  # no request spent against the quota

    # A success elsewhere must not clear the server's deadline early.
    client._record_success()
    assert client._rate_limited() is True

    # Once the deadline expires, probes try the wire again.
    client._rate_limited_until = time.monotonic() - 1
    client._disabled_until = time.monotonic() - 1
    assert client.probe() == (True, None)


def test_probe_reports_transport_error():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("refused")
    client = _client(session)

    up, error = client.probe()

    assert up is False
    assert "refused" in error


def test_probe_reports_http_error():
    session = MagicMock()
    session.post.return_value = _response(status_code=503, payload={})
    client = _client(session)

    up, error = client.probe()

    assert up is False
    assert "503" in error


def test_probe_reports_empty_completion():
    session = MagicMock()
    session.post.return_value = _response(payload=_completion(""))
    client = _client(session)

    up, error = client.probe()

    assert up is False
    assert "empty" in error


def test_probe_never_raises():
    session = MagicMock()
    session.post.side_effect = RuntimeError("boom")
    client = _client(session)

    up, error = client.probe()

    assert up is False
    assert "boom" in error


def test_probe_success_clears_previous_failures():
    session = MagicMock()
    session.post.side_effect = [
        _response(status_code=500, payload={}),
        _response(payload=_completion("pong")),
    ]
    client = _client(session, circuit_failure_threshold=5)

    assert client.probe() == (False, "HTTP 500")
    assert client._consecutive_failures == 1

    # The probe is now a real completion, so its success legitimately closes
    # the filter breaker: completions demonstrably work again.
    assert client.probe() == (True, None)
    assert client._consecutive_failures == 0
    assert client._circuit_open() is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_client_is_a_singleton_and_reset_closes_it(monkeypatch):
    monkeypatch.setenv("WIKIVISAGE_LIFTWING_URL", "https://example.invalid/x")
    liftwing_client.reset_client()
    try:
        first = liftwing_client.get_client()
        assert liftwing_client.get_client() is first
    finally:
        liftwing_client._CLIENT = None

    liftwing_client.reset_client()
    assert liftwing_client._CLIENT is None


# ---------------------------------------------------------------------------
# Politeness gap (request spacing)
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic monotonic clock whose ``sleep`` advances time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture()
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(liftwing_client.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(liftwing_client.time, "sleep", clock.sleep)
    return clock


def test_spacing_waits_between_calls(fake_clock):
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]'))
    client = _client(session, min_call_interval=5.0)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is not None
    assert fake_clock.sleeps == []  # the first call of a burst never waits

    assert client.filter_metadata([item]) is not None
    assert len(fake_clock.sleeps) == 1
    assert 5.0 <= fake_clock.sleeps[0] <= 7.5  # interval + up to half as jitter


def test_spacing_disabled_by_zero_interval(fake_clock):
    session = MagicMock()
    session.post.return_value = _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]'))
    client = _client(session, min_call_interval=0)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    client.filter_metadata([item])
    client.filter_metadata([item])

    assert fake_clock.sleeps == []


def test_spacing_also_applies_to_the_probe(fake_clock):
    session = MagicMock()
    session.post.return_value = _response(payload=_completion("pong"))
    client = _client(session, min_call_interval=5.0)

    assert client.probe() == (True, None)
    assert fake_clock.sleeps == []

    assert client.probe() == (True, None)
    assert len(fake_clock.sleeps) == 1
    assert 5.0 <= fake_clock.sleeps[0] <= 7.5


def test_spacing_applies_after_a_failed_call(fake_clock):
    """A failed batch still arms the gap: retries must not hammer."""
    session = MagicMock()
    session.post.side_effect = [
        requests.exceptions.Timeout("too slow"),
        _response(payload=_completion('[{"page_id": 1, "keep": true, "reason": "ok"}]')),
    ]
    client = _client(session, min_call_interval=5.0)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is None
    assert client.filter_metadata([item]) is not None

    assert len(fake_clock.sleeps) == 1
    assert 5.0 <= fake_clock.sleeps[0] <= 7.5


def test_spacing_skipped_when_the_breaker_is_open(fake_clock):
    session = MagicMock()
    session.post.return_value = _response(status_code=429, payload={}, headers={"Retry-After": "600"})
    client = _client(session, min_call_interval=5.0, circuit_failure_threshold=5)
    item = MetadataItem(page_id=1, title="File:A.jpg")

    assert client.filter_metadata([item]) is None  # the 429 opens the breaker
    fake_clock.sleeps.clear()

    assert client.filter_metadata([item]) is None  # no request, so no wait
    assert fake_clock.sleeps == []
