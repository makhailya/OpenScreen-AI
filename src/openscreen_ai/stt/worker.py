"""Qt-контроллер отдельной сессии распознавания речи."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum
from threading import Event

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from openscreen_ai.stt.audio import MicrophoneSource
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttEngine, SttResult

logger = logging.getLogger(__name__)
_AUDIO_POLL_SECONDS = 0.02


class SttState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class _SttWorker(QThread):
    """Один запуск. Все операции с микрофоном и моделями выполняются в run()."""

    listening = Signal()
    transcription_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        microphone_factory: Callable[[], MicrophoneSource],
        engine_factory: Callable[[], SttEngine],
    ) -> None:
        super().__init__()
        self._microphone_factory = microphone_factory
        self._engine_factory = engine_factory
        self._stop_requested = Event()
        self.failed = False

    def request_stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        microphone = None
        engine = None
        try:
            if self._stop_requested.is_set():
                return
            engine = self._engine_factory()
            engine.start()
            if self._stop_requested.is_set():
                return
            microphone = self._microphone_factory()
            microphone.start()
            if not self._stop_requested.is_set():
                self.listening.emit()
                self._listen(microphone, engine)
        except Exception as error:
            self._report_error(error)
        finally:
            if microphone is not None:
                self._cleanup(microphone.stop)
            if engine is not None:
                self._cleanup(engine.close)

    def _listen(self, microphone: MicrophoneSource, engine: SttEngine) -> None:
        expected_sequence = 0
        accounted_drops = 0
        input_overflows = 0

        while not self._stop_requested.is_set():
            # Снимок до read(): если очередь пуста, учтённые здесь потери
            # произошли до следующего блока, который ещё может прийти.
            dropped_before_read = microphone.dropped_blocks
            block = microphone.read()
            if block is None:
                tail_drops = dropped_before_read - accounted_drops
                if tail_drops > 0:
                    engine.reset()
                    expected_sequence += tail_drops
                    accounted_drops = dropped_before_read
                self._stop_requested.wait(_AUDIO_POLL_SECONDS)
                continue

            gap = block.sequence - expected_sequence
            if gap != 0 or block.input_overflows > input_overflows:
                engine.reset()
            accounted_drops += max(gap, 0)
            expected_sequence = block.sequence + 1
            input_overflows = block.input_overflows

            if self._stop_requested.is_set():
                break
            for result in engine.process_audio(block.samples, captured_at=block.captured_at):
                if self._stop_requested.is_set():
                    break
                self.transcription_ready.emit(result)

    def _cleanup(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as error:
            self._report_error(error)

    def _report_error(self, error: Exception) -> None:
        self.failed = True
        logger.exception("Ошибка в потоке STT")
        self.error_occurred.emit(str(error) or type(error).__name__)


class SttController(QObject):
    """Управляется из своего Qt-потока, обычно главного.

    Перед выходом из приложения вызовите request_stop() и оставьте
    цикл событий активным до сигнала finished.
    """

    transcription_ready = Signal(object)
    state_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    # Удерживаем контроллер и поток, даже если вызывающий код потерял ссылку.
    _active: set[SttController] = set()

    def __init__(
        self,
        config: SttConfig,
        *,
        microphone_factory: Callable[[], MicrophoneSource] | None = None,
        engine_factory: Callable[[], SttEngine] | None = None,
    ) -> None:
        super().__init__()
        self._microphone_factory = microphone_factory or (
            lambda: MicrophoneSource(device=config.audio_device)
        )
        self._engine_factory = engine_factory or (lambda: SttEngine(config))
        self._worker: _SttWorker | None = None
        self._state = SttState.IDLE

    @property
    def state(self) -> SttState:
        return self._state

    @Slot(result=bool)
    def start(self) -> bool:
        """Запускает новую сессию, если предыдущий поток уже завершён."""
        if self._worker is not None:
            return False

        worker = _SttWorker(self._microphone_factory, self._engine_factory)
        self._worker = worker
        self._active.add(self)
        worker.listening.connect(self._on_listening, Qt.ConnectionType.QueuedConnection)
        worker.transcription_ready.connect(
            self._on_transcription, Qt.ConnectionType.QueuedConnection
        )
        worker.error_occurred.connect(self.error_occurred, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._set_state(SttState.STARTING)
        worker.start()
        return True

    @Slot()
    def request_stop(self) -> None:
        """Запрашивает остановку и сразу возвращает управление вызывающему коду."""
        if self._worker is not None and self._state in (SttState.STARTING, SttState.LISTENING):
            self._worker.request_stop()
            self._set_state(SttState.STOPPING)

    def _set_state(self, state: SttState) -> None:
        self._state = state
        self.state_changed.emit(state.value)

    @Slot()
    def _on_listening(self) -> None:
        if self._state == SttState.STARTING:
            self._set_state(SttState.LISTENING)

    @Slot(object)
    def _on_transcription(self, result: SttResult) -> None:
        if self._state == SttState.LISTENING:
            self.transcription_ready.emit(result)

    @Slot()
    def _on_worker_finished(self) -> None:
        worker = self._worker
        assert worker is not None
        # finished может прийти до завершения финальной очистки самого Qt-потока.
        # Нулевой тайм-аут проверяет завершение без блокирующего ожидания.
        if not worker.wait(0):
            QTimer.singleShot(10, self, self._on_worker_finished)
            return
        self._set_state(SttState.ERROR if worker.failed else SttState.STOPPED)
        worker.deleteLater()
        self._worker = None
        self._active.discard(self)
        self.finished.emit()
