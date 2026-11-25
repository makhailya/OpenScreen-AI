"""Fake HTTP coverage for provider failures and request lifecycle."""

from __future__ import annotations

import asyncio
import traceback

import httpx
import pytest

from openscreen_ai.config import AppConfig
from openscreen_ai.providers.ollama_provider import OllamaError, OllamaProvider


def test_timeout_is_reported_without_real_network() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fake timeout", request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("test")]

    with pytest.raises(OllamaError, match="не ответила за отведённое время"):
        asyncio.run(collect())


def test_arbitrary_request_error_text_is_not_exposed() -> None:
    canary = "SYNTHETIC_REQUEST_ERROR_CANARY_3281"

    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError(canary, request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("synthetic prompt")]

    with pytest.raises(OllamaError) as raised:
        asyncio.run(collect())
    assert canary not in str(raised.value)
    assert canary not in "".join(traceback.format_exception(raised.value))


def test_http_error_body_is_not_read_into_public_error() -> None:
    canary = "SYNTHETIC_HTTP_BODY_CANARY_4910"
    body_read = False

    class UnreadBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal body_read
            body_read = True
            yield b'{"error":"' + canary.encode() + b'"}'

        async def aclose(self) -> None:
            return None

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, stream=UnreadBody(), request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("synthetic prompt")]

    with pytest.raises(OllamaError) as raised:
        asyncio.run(collect())
    assert "HTTP 500" in str(raised.value)
    assert canary not in str(raised.value)
    assert canary not in "".join(traceback.format_exception(raised.value))
    assert not body_read


def test_stream_error_and_malformed_line_canaries_are_not_exposed() -> None:
    canary = "SYNTHETIC_STREAM_ERROR_CANARY_5724"

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=f'{{"error":"{canary}"}}\n'.encode(),
            request=request,
        )

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("synthetic prompt")]

    with pytest.raises(OllamaError) as raised:
        asyncio.run(collect())
    assert canary not in str(raised.value)
    assert canary not in "".join(traceback.format_exception(raised.value))

    async def malformed_handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=f'{{"response":"{canary}"\n'.encode(),
            request=request,
        )

    malformed_provider = OllamaProvider(
        AppConfig(), transport=httpx.MockTransport(malformed_handle)
    )
    with pytest.raises(OllamaError) as malformed:
        asyncio.run(collect_with(malformed_provider))
    assert canary not in str(malformed.value)
    assert canary not in "".join(traceback.format_exception(malformed.value))


async def collect_with(provider: OllamaProvider) -> list[str]:
    return [chunk async for chunk in provider.stream("synthetic prompt")]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json\n", "некорректный поток данных"),
        (b"[]\n", "неожиданном формате"),
    ],
)
def test_malformed_stream_lines_are_rejected(body: bytes, message: str) -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("test")]

    with pytest.raises(OllamaError, match=message):
        asyncio.run(collect())


def test_empty_response_and_a_fresh_second_request() -> None:
    prompts: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        prompts.append(request.read().decode())
        response = b'{"response":"second","done":true}\n' if len(prompts) == 2 else b""
        return httpx.Response(200, content=response, request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect_both() -> list[str]:
        with pytest.raises(OllamaError, match="неполный поток ответа"):
            async for _ in provider.stream("first"):
                pass
        return [chunk async for chunk in provider.stream("second")]

    assert asyncio.run(collect_both()) == ["second"]
    assert len(prompts) == 2
    assert '"prompt":"first"' in prompts[0]
    assert '"prompt":"second"' in prompts[1]


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield b'{"response":"partial","done":false}\n'
        self.entered.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


def test_cancellation_closes_active_response_stream() -> None:
    stream = BlockingStream()

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def cancel_active_request() -> tuple[list[str], bool]:
        chunks: list[str] = []

        async def collect() -> None:
            async for chunk in provider.stream("cancel me"):
                chunks.append(chunk)

        task = asyncio.create_task(collect())
        await asyncio.wait_for(stream.entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return chunks, stream.closed.is_set()

    assert asyncio.run(cancel_active_request()) == (["partial"], True)
