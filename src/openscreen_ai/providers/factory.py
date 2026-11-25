"""Явное создание текстового провайдера без автоматического перехода в облако."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from openscreen_ai.config import AppConfig, ProviderKind
from openscreen_ai.core.assistant import TextProvider
from openscreen_ai.providers.ollama_provider import OllamaProvider


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    streaming: bool = True
    vision: bool = False


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    display_name: str
    capabilities: ModelCapabilities


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    kind: ProviderKind
    ollama: AppConfig = field(repr=False)


class ProviderFactory:
    def __init__(self, ollama_factory: Callable[[AppConfig], TextProvider] = OllamaProvider):
        self._ollama_factory = ollama_factory

    def create(self, config: ProviderConfig) -> TextProvider:
        if config.kind is ProviderKind.OLLAMA:
            return self._ollama_factory(config.ollama)
        raise ValueError("Выбранный провайдер пока недоступен.")
