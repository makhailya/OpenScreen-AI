"""Проверки логики ассистента на поддельном провайдере."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from openscreen_ai.core.assistant import Assistant


class FakeProvider:
    def __init__(self) -> None:
        self.received_prompt: str | None = None

    async def stream(
        self, prompt: str, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        self.received_prompt = prompt
        yield "Первая часть"
        yield " и вторая часть"


def test_assistant_passes_clean_prompt_and_streams_provider_chunks() -> None:
    provider = FakeProvider()
    assistant = Assistant(provider)

    async def collect_response() -> list[str]:
        return [chunk async for chunk in assistant.stream_response("  Как работает Python?  ")]

    chunks = asyncio.run(collect_response())

    assert provider.received_prompt == "Как работает Python?"
    assert chunks == ["Первая часть", " и вторая часть"]


def test_assistant_rejects_empty_prompt_before_calling_provider() -> None:
    provider = FakeProvider()
    assistant = Assistant(provider)

    async def collect_response() -> list[str]:
        return [chunk async for chunk in assistant.stream_response("  ")]

    with pytest.raises(ValueError, match="Введите вопрос"):
        asyncio.run(collect_response())

    assert provider.received_prompt is None
