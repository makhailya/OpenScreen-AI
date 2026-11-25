"""Асинхронный потоковый клиент для HTTP API Ollama."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from openscreen_ai.config import AppConfig
from openscreen_ai.core.request_lifecycle import DIAGNOSTICS, new_opaque_id


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

    async def stream(
        self, prompt: str, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        """Возвращает текст по мере поступления JSON-строк от Ollama."""
        if request_id is None or len(request_id) != 32 or not all(
            character in "0123456789abcdef" for character in request_id
        ):
            request_id = new_opaque_id()
        attempt_id = new_opaque_id()
        timeout = httpx.Timeout(
            self._config.timeout_seconds,
            connect=min(10.0, self._config.timeout_seconds),
        )
        url = f"{self._config.ollama_base_url}/api/generate"
        pending_reasoning = [] if "lfm3" in self._config.model.lower() else None
        attempt_started = False
        attempt_outcome = "cancelled"

        async def on_request(_request: httpx.Request) -> None:
            nonlocal attempt_started
            if attempt_started:
                return
            DIAGNOSTICS.record(
                "llm.http_attempt_started",
                request_id=request_id,
                attempt_id=attempt_id,
                state="processing",
                outcome="started",
            )
            attempt_started = True

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                trust_env=False,
                transport=self._transport,
                event_hooks={"request": [on_request]},
            ) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={"model": self._config.model, "prompt": prompt, "stream": True},
                ) as response:
                    if response.is_error:
                        self._raise_for_response(response)

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        payload = self._decode_line(line)
                        provider_error = payload.get("error")
                        if provider_error:
                            raise OllamaError("Ollama сообщила об ошибке генерации.") from None

                        text_chunk = payload.get("response")
                        if isinstance(text_chunk, str) and text_chunk:
                            if pending_reasoning is None:
                                yield text_chunk
                            else:
                                pending_reasoning.append(text_chunk)
                                buffered = "".join(pending_reasoning)
                                if "</think>" in buffered:
                                    _, visible = buffered.split("</think>", 1)
                                    pending_reasoning = None
                                    if visible:
                                        yield visible
                                elif len(buffered) > 65536:
                                    raise OllamaError("Ответ модели превысил лимит рассуждения.")
                        if payload.get("done") is True:
                            if pending_reasoning:
                                yield "".join(pending_reasoning)
                            break
                    else:
                        raise OllamaError("Ollama вернула неполный поток ответа.")
            attempt_outcome = "success"
        except asyncio.CancelledError:
            attempt_outcome = "cancelled"
            raise
        except OllamaError:
            attempt_outcome = "provider_error"
            raise
        except httpx.ConnectError:
            attempt_outcome = "transport_error"
            raise OllamaError(
                "Не удалось подключиться к Ollama. Убедитесь, что она запущена "
                "и доступна по адресу из OLLAMA_BASE_URL."
            ) from None
        except httpx.TimeoutException:
            attempt_outcome = "transport_error"
            raise OllamaError(
                "Ollama не ответила за отведённое время. Проверьте её работу "
                "или увеличьте OLLAMA_TIMEOUT_SECONDS."
            ) from None
        except httpx.RequestError:
            attempt_outcome = "transport_error"
            raise OllamaError("Ошибка HTTP-соединения с Ollama.") from None
        except Exception:
            attempt_outcome = "provider_error"
            raise
        finally:
            if attempt_started:
                DIAGNOSTICS.record(
                    "llm.http_attempt_finished",
                    request_id=request_id,
                    attempt_id=attempt_id,
                    state=(
                        "completed"
                        if attempt_outcome == "success"
                        else "stopping"
                        if attempt_outcome == "cancelled"
                        else "failed"
                    ),
                    outcome=attempt_outcome,
                )

    @staticmethod
    def _raise_for_response(response: httpx.Response) -> None:
        """Raise a bounded category without reading arbitrary server content."""
        if response.status_code == 404:
            raise OllamaError(
                "Ollama не нашла настроенную модель. Проверьте OLLAMA_MODEL."
            ) from None
        raise OllamaError(f"Ollama вернула ошибку HTTP {response.status_code}.") from None

    @staticmethod
    def _decode_line(line: str) -> dict[str, object]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise OllamaError("Ollama вернула некорректный поток данных.") from None
        if not isinstance(payload, dict):
            raise OllamaError("Ollama вернула ответ в неожиданном формате.")
        return payload
