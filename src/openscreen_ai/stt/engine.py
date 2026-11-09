"""Lazy Parakeet and Silero VAD engine for mono 16 kHz audio blocks."""

from __future__ import annotations

import importlib
import math
import time
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from openscreen_ai.stt.availability import check_stt_available
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.errors import (
    SttAudioError,
    SttDecodeError,
    SttError,
    SttInitializationError,
    SttLifecycleError,
)

RecognizerFactory = Callable[[SttConfig], object]
VadFactory = Callable[[SttConfig], tuple[object, int]]
AvailabilityChecker = Callable[[SttConfig], None]


@dataclass(frozen=True, slots=True)
class SttResult:
    """Распознанная фраза и метрики времени в секундах/миллисекундах."""

    text: str
    endpoint_at: float
    endpoint_audio_at: float
    speech_started_at: float
    # Оценка границы по сегменту VAD, с шагом vad_window_ms.
    last_speech_active_at: float | None
    recognition_ms: float
    cpu_ms: float
    audio_seconds: float
    vad_window_ms: float = 32.0
    decode_completed_at: float | None = None


class SttEngine:
    """Последовательно обрабатывает аудиоблоки и декодирует фразы по VAD.

    Модели создаются только при явном вызове start(). Класс не создаёт
    потоков и рассчитан на последовательные вызовы из будущего worker-потока.
    Для unit-тестов можно передать фабрики распознавателя и VAD.
    """

    def __init__(
        self,
        config: SttConfig,
        *,
        recognizer_factory: RecognizerFactory | None = None,
        vad_factory: VadFactory | None = None,
        availability_checker: AvailabilityChecker = check_stt_available,
    ) -> None:
        self.config = config
        self._recognizer_factory = recognizer_factory
        self._vad_factory = vad_factory
        self._availability_checker = availability_checker

        self._recognizer: Any | None = None
        self._vad: Any | None = None
        self._numpy: Any | None = None
        self._window_size: int | None = None
        self._audio_buffer = array("f")
        self._buffer_audio_at: float | None = None
        self._speech_started_at: float | None = None
        self._vad_samples_processed = 0
        self._started = False
        self._closed = False

    def start(self) -> None:
        """Проверяет готовность и один раз создаёт Parakeet и Silero VAD."""
        if self._closed:
            raise SttLifecycleError("Закрытый STT Engine нельзя запустить повторно.")
        if self._started:
            return

        self._availability_checker(self.config)
        needs_runtime = (
            self._recognizer_factory is None or self._vad_factory is None
        )
        numpy = sherpa_onnx = None
        recognizer = vad = None
        try:
            if needs_runtime:
                numpy = importlib.import_module("numpy")
                sherpa_onnx = importlib.import_module("sherpa_onnx")

            recognizer = (
                self._recognizer_factory(self.config)
                if self._recognizer_factory is not None
                else self._create_recognizer(sherpa_onnx, self.config)
            )
            if self._vad_factory is not None:
                vad, window_size = self._vad_factory(self.config)
            else:
                vad, window_size = self._create_vad(sherpa_onnx, self.config)
            if isinstance(window_size, bool) or not isinstance(window_size, int):
                raise TypeError("Размер окна Silero VAD должен быть целым числом.")
            if window_size <= 0:
                raise ValueError("Размер окна Silero VAD должен быть больше нуля.")
        except Exception as error:
            self._release_resource(vad)
            self._release_resource(recognizer)
            raise SttInitializationError(
                f"Не удалось инициализировать Parakeet STT: {error}"
            ) from error

        self._recognizer = recognizer
        self._vad = vad
        self._numpy = numpy
        self._window_size = window_size
        self._started = True

    @staticmethod
    def _create_recognizer(sherpa_onnx: Any, config: SttConfig) -> object:
        """Создаёт OfflineRecognizer с параметрами проверенного POC."""
        model_dir = config.model_dir
        if model_dir is None:
            raise ValueError("Укажите каталог Parakeet через SttConfig.model_dir.")
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=config.num_threads,
            sample_rate=config.sample_rate,
            feature_dim=config.feature_dim,
            decoding_method=config.decoding_method,
            provider=config.provider,
        )

    @staticmethod
    def _create_vad(sherpa_onnx: Any, config: SttConfig) -> tuple[object, int]:
        """Создаёт Silero VAD и возвращает требуемый размер его окна."""
        if config.vad_model_path is None:
            raise ValueError("Укажите модель Silero через SttConfig.vad_model_path.")

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(config.vad_model_path)
        vad_config.silero_vad.min_silence_duration = (
            config.vad_min_silence_duration
        )
        vad_config.silero_vad.min_speech_duration = config.vad_min_speech_duration
        vad_config.silero_vad.max_speech_duration = config.vad_max_speech_duration
        vad_config.silero_vad.window_size = config.vad_window_size
        vad_config.sample_rate = config.sample_rate
        detector = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=config.vad_buffer_size_seconds,
        )
        return detector, vad_config.silero_vad.window_size

    def process_audio(
        self,
        samples: Iterable[float],
        *,
        captured_at: float | None = None,
    ) -> list[SttResult]:
        """Добавляет mono float32 сэмплы и возвращает завершённые непустые фразы.

        captured_at — монотонное время начала блока; его можно опустить,
        тогда движок возьмёт время поступления блока в этот метод.
        """
        self._ensure_started()
        audio = self._coerce_samples(samples)
        if not audio:
            return []

        if captured_at is not None and (
            isinstance(captured_at, bool)
            or not isinstance(captured_at, (int, float))
            or not math.isfinite(captured_at)
        ):
            raise SttAudioError("captured_at должен быть конечным временем или None.")

        if not self._audio_buffer:
            self._buffer_audio_at = (
                float(captured_at) if captured_at is not None else time.perf_counter()
            )
        self._audio_buffer.extend(audio)

        results: list[SttResult] = []
        assert self._vad is not None
        assert self._window_size is not None
        try:
            while len(self._audio_buffer) >= self._window_size:
                window_audio_at = (
                    self._buffer_audio_at
                    if self._buffer_audio_at is not None
                    else time.perf_counter()
                )
                window = array("f", self._audio_buffer[: self._window_size])
                del self._audio_buffer[: self._window_size]
                self._buffer_audio_at = (
                    window_audio_at + self._window_size / self.config.sample_rate
                    if self._audio_buffer
                    else None
                )

                vad_window = (
                    self._numpy.asarray(window, dtype=self._numpy.float32)
                    if self._numpy is not None
                    else window
                )
                self._vad.accept_waveform(vad_window)
                self._vad_samples_processed += self._window_size
                window_seconds = self._window_size / self.config.sample_rate
                endpoint_audio_at = window_audio_at + window_seconds
                if self._speech_started_at is None and self._vad.is_speech_detected():
                    self._speech_started_at = window_audio_at

                while not self._vad.empty():
                    endpoint_at = time.perf_counter()
                    speech_started_at = (
                        self._speech_started_at
                        if self._speech_started_at is not None
                        else endpoint_audio_at
                    )
                    vad_segment = self._vad.front
                    segment = self._copy_vad_segment(vad_segment.samples)
                    # Sherpa возвращает начало и сэмплы обрезанного сегмента.
                    # Оба смещения отсчитываются по входу VAD после reset().
                    segment_start = getattr(vad_segment, "start", None)
                    last_speech_active_at = None
                    if segment_start is not None:
                        segment_end = segment_start + len(segment)
                        if 0 <= segment_start < segment_end <= self._vad_samples_processed:
                            trailing_samples = self._vad_samples_processed - segment_end
                            last_speech_active_at = (
                                endpoint_audio_at - trailing_samples / self.config.sample_rate
                            )
                    self._vad.pop()
                    self._speech_started_at = None

                    text, recognition_ms, cpu_ms = self._decode(segment)
                    decode_completed_at = time.perf_counter()
                    if text:
                        results.append(
                            SttResult(
                                text=text,
                                endpoint_at=endpoint_at,
                                endpoint_audio_at=endpoint_audio_at,
                                speech_started_at=speech_started_at,
                                last_speech_active_at=last_speech_active_at,
                                recognition_ms=recognition_ms,
                                cpu_ms=cpu_ms,
                                audio_seconds=len(segment) / self.config.sample_rate,
                                vad_window_ms=window_seconds * 1000,
                                decode_completed_at=decode_completed_at,
                            )
                        )
        except SttDecodeError:
            raise
        except Exception as error:
            raise SttError(
                f"Ошибка обработки аудио через Silero VAD: {error}"
            ) from error
        return results

    @staticmethod
    def _coerce_samples(samples: Iterable[float]) -> array:
        """Нормализует одномерный вход в компактный массив float32."""
        dimensions = getattr(samples, "ndim", None)
        shape = getattr(samples, "shape", None)
        if dimensions == 2 and shape is not None and shape[-1] == 1:
            reshape = getattr(samples, "reshape", None)
            if not callable(reshape):
                raise SttAudioError("Ожидается mono-аудиоблок в формате float32.")
            samples = reshape(-1)
        elif dimensions is not None and dimensions != 1:
            raise SttAudioError("Ожидается одномерный mono-аудиоблок float32.")

        try:
            audio = array("f", (float(sample) for sample in samples))
        except (TypeError, ValueError, OverflowError) as error:
            raise SttAudioError(
                "Аудиоблок должен содержать числовые mono float32 сэмплы."
            ) from error
        if any(not math.isfinite(sample) for sample in audio):
            raise SttAudioError("Аудиоблок содержит NaN или бесконечные значения.")
        return audio

    def _copy_vad_segment(self, samples: Any) -> Any:
        copy = getattr(samples, "copy", None)
        segment = copy() if callable(copy) else array("f", samples)
        if self._numpy is not None:
            segment = self._numpy.asarray(segment, dtype=self._numpy.float32).copy()
        return segment

    def _decode(self, segment: Any) -> tuple[str, float, float]:
        assert self._recognizer is not None
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.config.sample_rate, segment)
            cpu_started = time.process_time()
            recognition_started = time.perf_counter()
            self._recognizer.decode_stream(stream)
            recognition_ms = (time.perf_counter() - recognition_started) * 1000
            cpu_ms = (time.process_time() - cpu_started) * 1000
            text = stream.result.text.strip()
        except Exception as error:
            raise SttDecodeError(
                f"Не удалось декодировать завершённую фразу: {error}"
            ) from error
        return text, recognition_ms, cpu_ms

    def reset(self) -> None:
        """Сбрасывает буфер аудио и внутреннее состояние VAD после разрыва."""
        if self._closed:
            raise SttLifecycleError("Нельзя сбросить закрытый STT Engine.")
        self._clear_audio_state()
        if self._started and self._vad is not None:
            try:
                self._vad.reset()
            except Exception as error:
                raise SttError(
                    f"Не удалось сбросить состояние Silero VAD: {error}"
                ) from error

    def close(self) -> None:
        """Сбрасывает VAD и освобождает удерживаемые runtime-ресурсы."""
        if self._closed:
            return

        vad, recognizer = self._vad, self._recognizer
        self._vad = None
        self._recognizer = None
        self._numpy = None
        self._window_size = None
        self._started = False
        self._closed = True
        self._clear_audio_state()

        errors: list[Exception] = []
        if vad is not None:
            reset = getattr(vad, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as error:
                    errors.append(error)
        for resource in (vad, recognizer):
            error = self._release_resource(resource)
            if error is not None:
                errors.append(error)
        if errors:
            raise SttError(
                f"Ошибка при освобождении STT-ресурсов: {errors[0]}"
            ) from errors[0]

    @staticmethod
    def _release_resource(resource: object | None) -> Exception | None:
        if resource is None:
            return None
        close = getattr(resource, "close", None)
        if not callable(close):
            return None
        try:
            close()
        except Exception as error:
            return error
        return None

    def _ensure_started(self) -> None:
        if self._closed:
            raise SttLifecycleError("Закрытый STT Engine нельзя использовать.")
        if not self._started:
            raise SttLifecycleError("Сначала вызовите SttEngine.start().")

    def _clear_audio_state(self) -> None:
        self._audio_buffer = array("f")
        self._buffer_audio_at = None
        self._speech_started_at = None
        self._vad_samples_processed = 0

