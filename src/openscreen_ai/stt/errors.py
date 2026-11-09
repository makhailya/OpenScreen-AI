"""Понятные ошибки необязательной подсистемы STT."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class SttError(RuntimeError):
    """Базовая ошибка подготовки и выполнения STT."""


class SttAudioError(ValueError):
    """Аудиоблок не соответствует формату mono float32."""


class SttDecodeError(SttError):
    """Не удалось декодировать завершённую фразу."""


class SttLifecycleError(SttError):
    """Операция движка вызвана в неподходящем состоянии."""


class SttInitializationError(SttError):
    """Не удалось создать распознаватель или VAD."""


class SttCompatibilityError(SttError):
    """Текущий Python не входит в проверенный диапазон для STT."""


class SttConfigurationError(ValueError):
    """Для проверки STT не заданы обязательные значения конфигурации."""


class SttDependencyError(SttError):
    """Не установлены optional-зависимости STT."""

    def __init__(self, missing_distributions: Sequence[str]) -> None:
        self.missing_distributions = tuple(missing_distributions)
        packages = ", ".join(self.missing_distributions)
        super().__init__(
            f"Для STT не найдены зависимости: {packages}. "
            "Установите extra командой: python -m pip install -e '.[stt]'."
        )


class SttFilesError(FileNotFoundError):
    """Не найдены файлы модели Parakeet или Silero VAD."""

    def __init__(
        self,
        *,
        missing_model_files: Sequence[Path] = (),
        missing_vad_model: Path | None = None,
    ) -> None:
        self.missing_model_files = tuple(missing_model_files)
        self.missing_vad_model = missing_vad_model

        details: list[str] = []
        if self.missing_model_files:
            paths = ", ".join(str(path) for path in self.missing_model_files)
            details.append(f"файлы модели Parakeet: {paths}")
        if self.missing_vad_model is not None:
            details.append(f"файл Silero VAD: {self.missing_vad_model}")

        message = "Не найдены " + "; ".join(details) + ". "
        message += (
            "Укажите существующие пути в SttConfig; файлы модели и VAD "
            "не скачиваются автоматически."
        )
        super().__init__(message)
