"""Независимый источник mono float32 аудио для одного потребителя."""

from __future__ import annotations

import importlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from openscreen_ai.stt.errors import SttDependencyError, SttError, SttLifecycleError

SAMPLE_RATE = 16_000


class AudioCaptureError(SttError):
    """Ошибка захвата, которую можно обработать в потоке потребителя."""


@dataclass(frozen=True, slots=True)
class AudioBlock:
    """Копия блока, начало на шкале perf_counter и признаки потери аудио."""

    samples: Any
    captured_at: float
    sequence: int
    input_overflows: int


class MicrophoneSource:
    """Жизненным циклом и чтением управляет один поток, callback пишет в очередь.

    При заполнении отбрасывается новый блок. Номера начинаются с нуля
    при каждом start(). После перезапуска потребителю нужно сбросить STT.
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        block_ms: int = 40,
        buffer_ms: int = 800,
    ) -> None:
        if type(block_ms) is not int or not 20 <= block_ms <= 100:
            raise ValueError("block_ms должен быть целым числом от 20 до 100.")
        if type(buffer_ms) is not int or not 500 <= buffer_ms <= 1000:
            raise ValueError("buffer_ms должен быть целым числом от 500 до 1000.")
        self._device = device
        self._block_frames = block_ms * SAMPLE_RATE // 1000
        self._blocks: deque[AudioBlock] = deque(maxlen=buffer_ms // block_ms)
        self._stream: Any | None = None
        self._callback_error: Exception | None = None
        self._sequence = 0
        self._dropped_blocks = 0
        self._input_overflows = 0
        self._queue_max_depth = 0

    @property
    def queue_depth(self) -> int:
        """Текущее число блоков в очереди."""
        return len(self._blocks)

    @property
    def queue_max_depth(self) -> int:
        """Максимальное число блоков за запуск."""
        return self._queue_max_depth

    @property
    def dropped_blocks(self) -> int:
        """Число блоков, отброшенных из-за заполнения очереди в этом запуске."""
        return self._dropped_blocks

    @property
    def input_overflows(self) -> int:
        """Число уведомлений PortAudio о потере входных сэмплов в этом запуске."""
        return self._input_overflows

    def start(self) -> None:
        """Открывает выбранный вход и запускает callback."""
        if self._stream is not None:
            self._check_stream()
            return

        try:
            sounddevice = importlib.import_module("sounddevice")
        except ModuleNotFoundError as error:
            if error.name == "sounddevice":
                raise SttDependencyError(("sounddevice",)) from error
            raise AudioCaptureError("Не удалось загрузить sounddevice.") from error
        except (ImportError, OSError) as error:
            raise AudioCaptureError("Не удалось загрузить sounddevice/PortAudio.") from error

        self._blocks.clear()
        self._callback_error = None
        self._sequence = self._dropped_blocks = self._input_overflows = 0
        self._queue_max_depth = 0

        def callback(indata: Any, frames: int, timing: Any, status: Any) -> None:
            try:
                self._enqueue(indata, frames, timing, status)
            except Exception as error:
                self._callback_error = error
                raise sounddevice.CallbackAbort from None

        try:
            self._stream = sounddevice.InputStream(
                device=self._device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self._block_frames,
                callback=callback,
            )
            self._stream.start()
            self._check_stream()
        except Exception as error:
            if self._stream is not None:
                try:
                    self.stop()
                except AudioCaptureError as close_error:
                    error.add_note(str(close_error))
            raise AudioCaptureError(
                f"Не удалось запустить микрофон {self._device!r}: {error}"
            ) from error

    def _enqueue(self, indata: Any, frames: int, timing: Any, status: Any) -> None:
        now = time.perf_counter()
        sequence = self._sequence
        self._sequence += 1
        if status.input_overflow:
            self._input_overflows += 1

        # Только callback добавляет элементы, потребитель может лишь убрать их.
        if len(self._blocks) == self._blocks.maxlen:
            self._dropped_blocks += 1
            return

        # PortAudio использует свои часы. Переносим только разность времён.
        adc_at = timing.inputBufferAdcTime
        current_at = timing.currentTime
        age = current_at - adc_at if 0 < adc_at <= current_at else frames / SAMPLE_RATE
        self._blocks.append(
            AudioBlock(indata.copy().reshape(-1), now - age, sequence, self._input_overflows)
        )
        depth = len(self._blocks)
        if depth > self._queue_max_depth:
            self._queue_max_depth = depth

    def read(self) -> AudioBlock | None:
        """Забирает следующий блок без ожидания или проверяет состояние входа."""
        self._check_stream()
        return self.read_pending()

    def read_pending(self) -> AudioBlock | None:
        """Забирает уже поставленный в очередь блок без проверки устройства."""
        try:
            return self._blocks.popleft()
        except IndexError:
            return None

    def _check_stream(self) -> None:
        if self._stream is None:
            raise SttLifecycleError("Сначала вызовите MicrophoneSource.start().")
        if self._callback_error is not None:
            raise AudioCaptureError("Ошибка в callback микрофона.") from self._callback_error
        try:
            active = self._stream.active
        except Exception as error:
            raise AudioCaptureError("Не удалось проверить состояние микрофона.") from error
        if not active:
            raise AudioCaptureError("Поток микрофона неожиданно остановился.")

    def stop(self) -> None:
        """Закрывает поток и отбрасывает буфер без ожидания его обработки."""
        try:
            self.stop_capture()
        finally:
            self._blocks.clear()

    def stop_capture(self) -> None:
        """Закрывает ввод, сохраняя очередь для штатного drain перед Stop."""
        stream = self._stream
        if stream is None:
            return
        try:
            stream.close(ignore_errors=False)
        except Exception as error:
            raise AudioCaptureError(f"Не удалось закрыть микрофон: {error}") from error
        self._stream = None
        self._callback_error = None
