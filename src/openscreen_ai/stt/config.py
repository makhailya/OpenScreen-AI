"""Типизированная конфигурация Parakeet STT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SttConfig:
    """Настройки модели, VAD, аудиоустройства и распознавания.

    Пути задаются вызывающим кодом. Parakeet и Silero VAD не скачиваются
    и не привязываются к конкретному расположению на компьютере.
    """

    model_dir: Path | str | None = None
    vad_model_path: Path | str | None = None
    audio_device: int | str | None = None
    sample_rate: int = 16_000
    num_threads: int = 4
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    provider: str = "cpu"
    vad_min_silence_duration: float = 0.25
    vad_min_speech_duration: float = 0.15
    vad_max_speech_duration: float = 15.0
    vad_window_size: int = 512
    vad_buffer_size_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Нормализует пути и проверяет параметры до запуска native-кода."""
        for name in ("model_dir", "vad_model_path"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} не может быть пустым.")
            object.__setattr__(self, name, Path(value).expanduser())

        if self.audio_device is not None:
            if isinstance(self.audio_device, bool) or not isinstance(
                self.audio_device, (int, str)
            ):
                raise ValueError("audio_device должен быть индексом, именем устройства или None.")
            if isinstance(self.audio_device, str) and not self.audio_device.strip():
                raise ValueError("Имя audio_device не может быть пустым.")

        if self.sample_rate != 16_000:
            raise ValueError("Для проверенной модели Parakeet требуется sample_rate=16000.")

        for name in ("num_threads", "feature_dim", "vad_window_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} должен быть целым числом больше нуля.")

        durations = (
            self.vad_min_silence_duration,
            self.vad_min_speech_duration,
            self.vad_max_speech_duration,
            self.vad_buffer_size_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in durations
        ):
            raise ValueError("Параметры длительности VAD должны быть конечными числами.")
        if self.vad_min_silence_duration <= 0:
            raise ValueError("vad_min_silence_duration должен быть больше нуля.")
        if self.vad_min_speech_duration <= 0:
            raise ValueError("vad_min_speech_duration должен быть больше нуля.")
        if self.vad_max_speech_duration <= self.vad_min_speech_duration:
            raise ValueError(
                "vad_max_speech_duration должен быть больше vad_min_speech_duration."
            )
        if self.vad_buffer_size_seconds <= 0:
            raise ValueError("vad_buffer_size_seconds должен быть больше нуля.")

        if not self.decoding_method.strip():
            raise ValueError("decoding_method не может быть пустым.")
        if not self.provider.strip():
            raise ValueError("provider не может быть пустым.")
