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
        (["before <thi", "nk>secret</thi", "nk>after"], "before after"),
        (["<think>private"], ""),
        (["ordinary ", "answer"], "ordinary answer"),
        ([], ""),
    ],
)
def test_explicit_reasoning_frames(frames, expected):
    async def handle(request):
        lines = [json.dumps({"response": frame, "thinking": "secret"}) for frame in frames]
        lines.append(json.dumps({"done": True}))
        return httpx.Response(200, content="\n".join(lines), request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect():
        return "".join([part async for part in provider.stream("test")])

    assert asyncio.run(collect()) == expected


def test_plain_text_arrives_before_terminal_frame():
    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"response":"first","done":false}\n'
            raise AssertionError("terminal frame was read before first chunk")

    async def handle(request):
        return httpx.Response(200, stream=GatedStream(), request=request)

    async def check():
        provider = OllamaProvider(
            AppConfig(think=False), transport=httpx.MockTransport(handle)
        )
        stream = provider.stream("test")
        assert await anext(stream) == "first"
        await stream.aclose()

    asyncio.run(check())


@pytest.mark.parametrize("suffix", ["", "</think>answer"])
def test_reasoning_buffer_limit(suffix):
    async def handle(request):
        line = json.dumps({"response": "<think>" + "x" * 65537 + suffix, "done": True})
        return httpx.Response(200, content=line, request=request)

    async def collect():
        provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))
        return [part async for part in provider.stream("test")]

    with pytest.raises(OllamaError, match="лимит"):
        asyncio.run(collect())


def test_think_false_sent_when_explicitly_configured():
    async def handle(request):
        assert json.loads(request.content)["think"] is False
        return httpx.Response(200, json={"response": "ok", "done": True}, request=request)

    async def collect():
        provider = OllamaProvider(AppConfig(think=False), transport=httpx.MockTransport(handle))
        return [part async for part in provider.stream("test")]

    assert asyncio.run(collect()) == ["ok"]


def test_cancellation_during_reasoning_and_next_request():
    class BlockAfterReasoning(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"response":"<think>secret","done":false}\n'
            await asyncio.Event().wait()

    requests = 0

    async def handle(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(200, stream=BlockAfterReasoning(), request=request)
        return httpx.Response(200, json={"response": "recovered", "done": True}, request=request)

    async def check():
        provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))
        stream = provider.stream("first")
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [part async for part in provider.stream("second")] == ["recovered"]

    asyncio.run(check())
