"""HTTP-поведение провайдера проверяется через httpx.MockTransport."""

import asyncio
import json
import logging

import httpx
import pytest

from openscreen_ai.config import AppConfig
from openscreen_ai.core.request_lifecycle import DIAGNOSTICS
from openscreen_ai.providers.ollama_provider import OllamaError, OllamaProvider


def test_provider_streams_response_and_sends_expected_request() -> None:
    seen_requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            content=(
                b'{"response":"Hello ","done":false}\n'
                b'{"response":"world","done":true}\n'
            ),
            request=request,
        )

    provider = OllamaProvider(
        AppConfig(model="test-model"),
        transport=httpx.MockTransport(handle),
    )

    async def collect_response() -> list[str]:
        return [chunk async for chunk in provider.stream("Explain a list")]

    chunks = asyncio.run(collect_response())

    assert chunks == ["Hello ", "world"]
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert str(request.url) == "http://127.0.0.1:11434/api/generate"
    assert request.content == b'{"model":"test-model","prompt":"Explain a list","stream":true}'


def test_provider_explains_missing_model_finishes_attempt_as_provider_error(caplog) -> None:
    DIAGNOSTICS.reset()
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"}, request=request)

    provider = OllamaProvider(
        AppConfig(model="missing-model"),
        transport=httpx.MockTransport(handle),
    )

    async def collect_response() -> list[str]:
        return [chunk async for chunk in provider.stream("Hello")]

    with caplog.at_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle"):
        with pytest.raises(OllamaError, match="не нашла настроенную модель") as raised:
            asyncio.run(collect_response())
    assert "missing-model" not in str(raised.value)
    starts = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("llm.http_attempt_started")
    ]
    finishes = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("llm.http_attempt_finished")
    ]
    assert len(starts) == len(finishes) == 1
    assert (
        starts[0].split("request_id=")[1].split()[0]
        == finishes[0].split("request_id=")[1].split()[0]
    )
    assert (
        starts[0].split("attempt_id=")[1].split()[0]
        == finishes[0].split("attempt_id=")[1].split()[0]
    )
    assert finishes[0].endswith("state=failed outcome=provider_error")
    assert DIAGNOSTICS.pending_attempt_count == 0


def test_provider_explains_unavailable_server() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect_response() -> list[str]:
        return [chunk async for chunk in provider.stream("Hello")]

    with pytest.raises(OllamaError, match="Не удалось подключиться к Ollama"):
        asyncio.run(collect_response())


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (["private</thi", "nk>answer"], ["answer"]),
        (["ordinary ", "answer"], ["ordinary answer"]),
    ],
)
def test_lfm3_implicit_reasoning_or_plain_response(frames, expected):
    async def handle(request):
        lines = [json.dumps({"response": frame, "done": False}) for frame in frames]
        lines.append(json.dumps({"done": True}))
        return httpx.Response(200, content="\n".join(lines), request=request)

    provider = OllamaProvider(
        AppConfig(model="hf.co/AtomicChat/LFM3-1.7B"), transport=httpx.MockTransport(handle)
    )

    async def collect():
        return [part async for part in provider.stream("test")]

    assert asyncio.run(collect()) == expected
