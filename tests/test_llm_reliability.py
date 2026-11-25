"""Fake-transport regression tests for Ollama stream failure and cleanup paths."""

import asyncio
import logging

import httpx
import pytest

from openscreen_ai.config import AppConfig
from openscreen_ai.core.request_lifecycle import DIAGNOSTICS
from openscreen_ai.providers.ollama_provider import OllamaError, OllamaProvider
from openscreen_ai.ui.main_window import GenerationWorker


def test_unexpected_generation_error_does_not_log_exception(caplog) -> None:
    canary = "SYNTHETIC_UI_CANARY_4f82"

    class FakeAssistant:
        async def stream_response(self, prompt: str, *, request_id: str | None = None):
            raise RuntimeError(canary)
            yield prompt

    worker = GenerationWorker(FakeAssistant(), "synthetic prompt")
    emitted: list[str] = []
    worker.error_occurred.connect(emitted.append)

    with caplog.at_level(logging.ERROR, logger="openscreen_ai.ui.main_window"):
        asyncio.run(worker._consume())

    assert emitted == ["Произошла внутренняя ошибка при генерации ответа."]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "ASSISTANT_GENERATION_UNEXPECTED_ERROR"
    assert record.exc_info is None
    assert canary not in caplog.text


def test_timeout_is_translated_and_attempt_finishes_with_transport_error(caplog) -> None:
    DIAGNOSTICS.reset()
    async def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("synthetic prompt")]

    with caplog.at_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle"):
        with pytest.raises(OllamaError, match="не ответила за отведённое время"):
            asyncio.run(collect())

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
    assert finishes[0].endswith("state=failed outcome=transport_error")
    assert DIAGNOSTICS.snapshot()["http_attempts_total"] == 1
    assert DIAGNOSTICS.pending_attempt_count == 0


@pytest.mark.parametrize("body", [b"not-json\n", b"[]\n"])
def test_malformed_stream_is_reported(body: bytes) -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream("synthetic prompt")]

    with pytest.raises(OllamaError, match="некорректн.*поток|неожиданном формате"):
        asyncio.run(collect())


def test_cancelling_stream_closes_response_and_finishes_attempt(caplog) -> None:
    DIAGNOSTICS.reset()
    started = asyncio.Event()
    release = asyncio.Event()
    closed = False

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal closed
            yield b'{"response":"first","done":false}\n'
            started.set()
            await release.wait()

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BlockingStream(), request=request)

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))

    async def cancel_consumer() -> None:
        async def collect() -> None:
            async for _chunk in provider.stream("synthetic prompt"):
                pass

        task = asyncio.create_task(collect())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle"):
        asyncio.run(cancel_consumer())
    assert closed is True
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
    assert finishes[0].endswith("state=stopping outcome=cancelled")
    assert DIAGNOSTICS.snapshot()["http_attempts_total"] == 1
    assert DIAGNOSTICS.pending_attempt_count == 0
