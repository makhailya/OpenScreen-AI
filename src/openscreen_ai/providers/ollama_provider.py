"""Асинхронный потоковый клиент для HTTP API Ollama."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from openscreen_ai.config import AppConfig


class OllamaError(RuntimeError):
    """Ошибка, которую можно безопасно показать пользователю."""


class OllamaProvider:
    """Запрашивает генерацию текста у локальной Ollama без блокировки Qt UI."""

    def __init__(
        self,
        config: AppConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Возвращает текст по мере поступления JSON-строк от Ollama."""
        timeout = httpx.Timeout(
            self._config.timeout_seconds,
            connect=min(10.0, self._config.timeout_seconds),
        )
        url = f"{self._config.ollama_base_url}/api/generate"

        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"model": self._config.model, "prompt": prompt, "stream": True},
                ) as response:
                    if response.is_error:
                        await self._raise_for_response(response)

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        payload = self._decode_line(line)
                        provider_error = payload.get("error")
                        if provider_error:
                            raise OllamaError(f"Ошибка Ollama: {provider_error}")

                        text_chunk = payload.get("response")
                        if isinstance(text_chunk, str) and text_chunk:
                            yield text_chunk
                        if payload.get("done") is True:
                            break
        except httpx.ConnectError as error:
            raise OllamaError(
                "Не удалось подключиться к Ollama. Убедитесь, что она запущена "
                "и доступна по адресу из OLLAMA_BASE_URL."
            ) from error
        except httpx.TimeoutException as error:
            raise OllamaError(
                "Ollama не ответила за отведённое время. Проверьте её работу "
                "или увеличьте OLLAMA_TIMEOUT_SECONDS."
            ) from error
        except httpx.RequestError as error:
            raise OllamaError(f"Ошибка HTTP-соединения с Ollama: {error}") from error

    async def _raise_for_response(self, response: httpx.Response) -> None:
        body = await response.aread()
        detail = self._extract_error(body)
        if response.status_code == 404:
            raise OllamaError(
                f"Модель «{self._config.model}» не найдена в Ollama. "
                f"Проверьте имя в OLLAMA_MODEL и установите модель через Ollama."
            )
        if detail:
            raise OllamaError(f"Ollama вернула ошибку HTTP {response.status_code}: {detail}")
        raise OllamaError(f"Ollama вернула ошибку HTTP {response.status_code}.")

    @staticmethod
    def _decode_line(line: str) -> dict[str, object]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama вернула некорректный поток данных.") from error
        if not isinstance(payload, dict):
            raise OllamaError("Ollama вернула ответ в неожиданном формате.")
        return payload

    @staticmethod
    def _extract_error(body: bytes) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body.decode("utf-8", errors="replace").strip()
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"]
        return ""
