"""Координация текстового запроса и выбранного AI-провайдера."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class TextProvider(Protocol):
    """Интерфейс провайдера, который возвращает ответ частями."""

    def stream(self, prompt: str, *, request_id: str | None = None) -> AsyncIterator[str]: ...


class Assistant:
    """Передаёт запрос провайдеру и отдаёт UI отдельные фрагменты ответа."""

    def __init__(self, provider: TextProvider) -> None:
        self._provider = provider

    async def stream_response(
        self, prompt: str, request_id: str | None = None
    ) -> AsyncIterator[str]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Введите вопрос перед отправкой.")

        if request_id is None:
            chunks = self._provider.stream(cleaned_prompt)
        else:
            chunks = self._provider.stream(cleaned_prompt, request_id=request_id)
        async for text_chunk in chunks:
            yield text_chunk
