"""HTTP-поведение провайдера проверяется через httpx.MockTransport."""

import asyncio

import httpx
import pytest

from openscreen_ai.config import AppConfig
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


def test_provider_explains_missing_model() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"}, request=request)

    provider = OllamaProvider(
        AppConfig(model="missing-model"),
        transport=httpx.MockTransport(handle),
    )

    async def collect_response() -> list[str]:
        return [chunk async for chunk in provider.stream("Hello")]

    with pytest.raises(OllamaError, match="missing-model.*не найдена"):
        asyncio.run(collect_response())


def test_provider_explains_unavailable_server() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect_response() -> list[str]:
        return [chunk async for chunk in provider.stream("Hello")]

    with pytest.raises(OllamaError, match="Не удалось подключиться к Ollama"):
        asyncio.run(collect_response())
