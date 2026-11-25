"""Главное окно, которое связывает текстовый и речевой сценарии с LLM."""

from __future__ import annotations

import asyncio
import logging
import threading
from enum import StrEnum
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from openscreen_ai.core.assistant import Assistant
from openscreen_ai.core.request_lifecycle import DIAGNOSTICS, new_opaque_id

if TYPE_CHECKING:
    from openscreen_ai.stt.engine import SttResult
    from openscreen_ai.stt.worker import SttController, SttDiagnostic

logger = logging.getLogger(__name__)


class RuntimeState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    STOPPING = "stopping"
    ERROR = "error"


_ALLOWED_TRANSITIONS: dict[RuntimeState, set[RuntimeState]] = {
    RuntimeState.IDLE: {RuntimeState.STARTING, RuntimeState.PROCESSING, RuntimeState.ERROR},
    RuntimeState.STARTING: {
        RuntimeState.LISTENING,
        RuntimeState.STOPPING,
        RuntimeState.IDLE,
        RuntimeState.ERROR,
    },
    RuntimeState.LISTENING: {RuntimeState.STOPPING, RuntimeState.ERROR},
    RuntimeState.PROCESSING: {
        RuntimeState.IDLE,
        RuntimeState.STOPPING,
        RuntimeState.ERROR,
    },
    RuntimeState.STOPPING: {
        RuntimeState.IDLE,
        RuntimeState.PROCESSING,
        RuntimeState.ERROR,
    },
    RuntimeState.ERROR: {
        RuntimeState.IDLE,
        RuntimeState.STARTING,
        RuntimeState.PROCESSING,
        RuntimeState.STOPPING,
    },
}

_STATE_MESSAGES = {
    RuntimeState.IDLE: "Готово",
    RuntimeState.STARTING: "Запускаю распознавание речи…",
    RuntimeState.LISTENING: "Слушаю…",
    RuntimeState.PROCESSING: "Обрабатываю запрос…",
    RuntimeState.STOPPING: "Завершаю текущую операцию…",
    RuntimeState.ERROR: "Операция не выполнена.",
}
_STT_ERROR_MESSAGE = (
    "Не удалось распознать речь. Проверьте STT-настройки, зависимости "
    "и разрешение на микрофон."
)
_LLM_ERROR_MESSAGE = "Произошла внутренняя ошибка при генерации ответа."
_STT_AGGREGATION_LIMIT_MESSAGE = (
    "Диктовка превысила безопасный лимит. Попробуйте разделить текст на части."
)
_STT_AGGREGATION_INCOMPLETE_MESSAGE = (
    "Не удалось собрать все части диктовки. Повторите запись."
)
_MAX_DICTATION_SEGMENTS = 256
_MAX_DICTATION_CHARACTERS = 32_000


class GenerationWorker(QThread):
    """Выполняет async-запрос в отдельном потоке Qt и поддерживает отмену."""

    text_received = Signal(str)
    error_occurred = Signal(str)
    generation_cancelled = Signal()
    request_text_received = Signal(str, str)
    request_error_occurred = Signal(str, str)
    request_cancelled = Signal(str)
    request_completed = Signal(str, str)

    def __init__(self, assistant: Assistant, prompt: str, request_id: str | None = None) -> None:
        super().__init__()
        self._assistant = assistant
        self._prompt = prompt
        self.request_id = request_id or new_opaque_id()
        self._outcome = "success"
        self._state_lock = threading.Lock()
        self._cancel_requested = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        """Отменяет сетевую coroutine через её собственный asyncio loop."""
        with self._state_lock:
            self._cancel_requested = True
            loop = self._loop
            task = self._task

        if loop is not None and task is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # The worker may finish its loop after is_running() but before
                # this thread schedules cancellation; completion wins that race.
                pass

    def run(self) -> None:
        DIAGNOSTICS.record(
            "llm.request_started",
            request_id=self.request_id,
            state="processing",
            outcome="started",
        )
        try:
            asyncio.run(self._consume())
        finally:
            if self._outcome == "success":
                DIAGNOSTICS.record(
                    "llm.request_completed",
                    request_id=self.request_id,
                    state="completed",
                    outcome="success",
                )
            elif self._outcome == "cancelled":
                DIAGNOSTICS.record(
                    "llm.request_cancelled",
                    request_id=self.request_id,
                    state="stopping",
                    outcome="cancelled",
                )
            else:
                DIAGNOSTICS.record(
                    "llm.request_failed",
                    request_id=self.request_id,
                    state="failed",
                    outcome="failed",
                )
            self.request_completed.emit(self.request_id, self._outcome)

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        with self._state_lock:
            self._loop = loop
            self._task = task
            cancel_before_start = self._cancel_requested

        try:
            if cancel_before_start and task is not None:
                task.cancel()
            async for chunk in self._assistant.stream_response(
                self._prompt, request_id=self.request_id
            ):
                self.text_received.emit(chunk)
                self.request_text_received.emit(self.request_id, chunk)
        except asyncio.CancelledError:
            self._outcome = "cancelled"
            self.generation_cancelled.emit()
            self.request_cancelled.emit(self.request_id)
        except Exception:
            # The provider exception remains available to diagnostics only through
            # its category; its arbitrary text and traceback are not logged or shown.
            logger.error("ASSISTANT_GENERATION_UNEXPECTED_ERROR")
            self._outcome = "error"
            self.error_occurred.emit(_LLM_ERROR_MESSAGE)
            self.request_error_occurred.emit(self.request_id, _LLM_ERROR_MESSAGE)
        finally:
            with self._state_lock:
                self._loop = None
                self._task = None


class MainWindow(QMainWindow):
    """Окно чата с ручным вводом и управляемой STT-диктовкой."""

    def __init__(
        self,
        assistant: Assistant,
        stt_controller: SttController | None = None,
    ) -> None:
        super().__init__()
        self._assistant = assistant
        self._stt_controller = stt_controller
        self._worker: GenerationWorker | None = None
        self._worker_request_id: str | None = None
        self._active_request_id: str | None = None
        self._runtime_state = RuntimeState.IDLE
        self._dictation_session_active = False
        self._dictation_session_id: int | None = None
        self._dictation_cancelled = False
        self._dictation_limit_pending = False
        self._dictation_segments: list[tuple[int | None, str]] = []
        self._dictation_segment_keys: dict[tuple[object, ...], str] = {}
        self._dictation_character_count = 0
        self._dictation_integrity_pending = False
        self._stt_failure_pending = False
        self._closing = False
        self._cancelled = False
        self._error_message: str | None = None

        self.setWindowTitle("OpenScreen AI")
        self.resize(720, 560)
        self._build_ui()
        self._apply_controls()
        self.statusBar().showMessage(_STATE_MESSAGES[self._runtime_state])

        if self._stt_controller is not None:
            self._stt_controller.transcription_ready.connect(self._on_transcription_ready)
            self._stt_controller.diagnostic_ready.connect(self._on_stt_diagnostic)
            self._stt_controller.state_changed.connect(self._on_stt_state_changed)
            self._stt_controller.error_occurred.connect(self._on_stt_error)
            self._stt_controller.finished.connect(self._on_stt_finished)

    @property
    def runtime_state(self) -> RuntimeState:
        return self._runtime_state

    def _build_ui(self) -> None:
        self._answer = QPlainTextEdit()
        self._answer.setReadOnly(True)
        self._answer.setPlaceholderText("Ответ локальной модели появится здесь…")

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Напишите вопрос или начните диктовку…")
        self._prompt.setFixedHeight(110)

        self._listen_button = QPushButton("Начать диктовку")
        self._listen_button.setVisible(self._stt_controller is not None)
        self._listen_button.clicked.connect(self._toggle_listening)

        self._send_button = QPushButton("Отправить")
        self._send_button.clicked.connect(self._send_prompt)
        self._cancel_button = QPushButton("Отменить")
        self._cancel_button.clicked.connect(self._cancel_generation)

        actions = QHBoxLayout()
        actions.addWidget(self._listen_button)
        actions.addStretch()
        actions.addWidget(self._cancel_button)
        actions.addWidget(self._send_button)

        layout = QVBoxLayout()
        layout.addWidget(self._answer, stretch=1)
        layout.addWidget(self._prompt)
        layout.addLayout(actions)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _transition(self, new_state: RuntimeState) -> bool:
        if new_state == self._runtime_state:
            return True
        if new_state not in _ALLOWED_TRANSITIONS[self._runtime_state]:
            # Fixed category only: state details do not contain request or speech data.
            logger.error("UI_STATE_TRANSITION_REJECTED")
            return False
        self._runtime_state = new_state
        self.statusBar().showMessage(_STATE_MESSAGES[new_state])
        self._apply_controls()
        return True

    def _apply_controls(self) -> None:
        can_submit = self._runtime_state in {RuntimeState.IDLE, RuntimeState.ERROR}
        can_toggle_listening = self._runtime_state in {
            RuntimeState.IDLE,
            RuntimeState.ERROR,
            RuntimeState.STARTING,
            RuntimeState.LISTENING,
        }
        is_listening_or_starting = self._runtime_state in {
            RuntimeState.STARTING,
            RuntimeState.LISTENING,
        }
        self._prompt.setReadOnly(not can_submit)
        self._send_button.setEnabled(can_submit and not self._closing)
        self._listen_button.setEnabled(
            self._stt_controller is not None and can_toggle_listening and not self._closing
        )
        self._listen_button.setText(
            "Остановить" if is_listening_or_starting else "Начать диктовку"
        )
        can_cancel_dictation = (
            self._dictation_session_active
            and self._runtime_state
            in {RuntimeState.STARTING, RuntimeState.LISTENING, RuntimeState.STOPPING}
        )
        self._cancel_button.setVisible(
            self._worker is not None or can_cancel_dictation
        )
        self._cancel_button.setText(
            "Отменить диктовку" if can_cancel_dictation else "Отменить"
        )
        self._cancel_button.setEnabled(
            not self._closing
            and (
                can_cancel_dictation
                or (
                    self._worker is not None
                    and self._runtime_state == RuntimeState.PROCESSING
                )
            )
        )

    def _toggle_listening(self) -> None:
        controller = self._stt_controller
        if controller is None or self._closing:
            return

        if self._runtime_state in {RuntimeState.STARTING, RuntimeState.LISTENING}:
            controller.request_stop()
            return
        if self._runtime_state not in {RuntimeState.IDLE, RuntimeState.ERROR}:
            return

        self._clear_dictation_segments()
        self._dictation_cancelled = False
        self._dictation_limit_pending = False
        self._dictation_integrity_pending = False
        self._stt_failure_pending = False
        self._error_message = None
        self._dictation_session_active = True
        self._dictation_session_id = None
        self._answer.clear()
        self._prompt.clear()
        if not controller.start():
            self._dictation_session_active = False
            self._show_stt_error()
        else:
            self._dictation_session_id = getattr(controller, "session_id", None)
            self._apply_controls()

    def _clear_dictation_segments(self) -> None:
        self._dictation_segments.clear()
        self._dictation_segment_keys.clear()
        self._dictation_character_count = 0

    def _finish_dictation_tracking(self) -> None:
        self._clear_dictation_segments()
        self._dictation_session_active = False
        self._dictation_session_id = None

    def _on_stt_state_changed(self, state: str) -> None:
        mapped_state = {
            "starting": RuntimeState.STARTING,
            "listening": RuntimeState.LISTENING,
            "stopping": RuntimeState.STOPPING,
        }.get(state)
        if mapped_state is not None:
            self._transition(mapped_state)
        elif state == "error":
            self._defer_stt_error_until_finished()

    def _on_transcription_ready(self, result: SttResult) -> None:
        if (
            self._closing
            or not self._dictation_session_active
            or self._dictation_cancelled
            or self._dictation_limit_pending
            or self._dictation_integrity_pending
            or self._stt_failure_pending
            or self._runtime_state
            not in {RuntimeState.STARTING, RuntimeState.LISTENING, RuntimeState.STOPPING}
        ):
            return
        if (
            self._dictation_session_id is not None
            and result.session_id is not None
            and result.session_id != self._dictation_session_id
        ):
            return

        if result.session_id is not None and result.segment_index is not None:
            event_key = ("segment", result.session_id, result.segment_index)
        else:
            event_key = ("legacy-result", result)
        if len(result.text) > _MAX_DICTATION_CHARACTERS:
            self._stop_oversized_dictation()
            return
        text = result.text.strip()
        if not text:
            return
        if event_key in self._dictation_segment_keys:
            if self._dictation_segment_keys[event_key] != text:
                self._stop_incomplete_dictation()
            return

        added_characters = len(text) + (1 if self._dictation_segments else 0)
        if (
            len(self._dictation_segments) >= _MAX_DICTATION_SEGMENTS
            or self._dictation_character_count + added_characters
            > _MAX_DICTATION_CHARACTERS
        ):
            self._stop_oversized_dictation()
            return

        self._dictation_segment_keys[event_key] = text
        self._dictation_segments.append((result.segment_index, text))
        self._dictation_character_count += added_characters
        self._prompt.setPlainText(self._joined_dictation_text())

    def _on_stt_diagnostic(self, diagnostic: SttDiagnostic) -> None:
        if (
            not self._dictation_session_active
            or self._closing
            or self._dictation_cancelled
            or self._dictation_limit_pending
            or self._dictation_integrity_pending
        ):
            return
        if (
            (diagnostic.dropped_blocks or 0) > 0
            or (diagnostic.input_overflow_notifications or 0) > 0
            or diagnostic.vad_resets_total > 0
        ):
            # Any audio discontinuity makes the transcript incomplete. Stop and
            # fail closed instead of sending a plausible but partial prompt.
            self._stop_incomplete_dictation()

    def _stop_oversized_dictation(self) -> None:
        self._dictation_limit_pending = True
        self._clear_dictation_segments()
        self._prompt.clear()
        if self._runtime_state in {RuntimeState.STARTING, RuntimeState.LISTENING}:
            self._transition(RuntimeState.STOPPING)
        if self._stt_controller is not None:
            self._stt_controller.request_stop()
        self._apply_controls()

    def _joined_dictation_text(self) -> str:
        segments = self._dictation_segments
        if segments and all(index is not None for index, _ in segments):
            segments = sorted(segments, key=lambda item: item[0])
        return " ".join(text for _, text in segments)

    def _stop_incomplete_dictation(self) -> None:
        self._dictation_integrity_pending = True
        self._clear_dictation_segments()
        self._prompt.clear()
        if self._runtime_state in {RuntimeState.STARTING, RuntimeState.LISTENING}:
            self._transition(RuntimeState.STOPPING)
        if self._stt_controller is not None:
            self._stt_controller.request_stop()
        self._apply_controls()

    def _on_stt_error(self, _message: str) -> None:
        self._defer_stt_error_until_finished()

    def _defer_stt_error_until_finished(self) -> None:
        self._stt_failure_pending = True
        self._clear_dictation_segments()
        self._prompt.clear()
        if self._runtime_state in {RuntimeState.STARTING, RuntimeState.LISTENING}:
            # Controller.error_occurred/state_changed can precede its cleanup and
            # worker release. Keep restart disabled until its finished signal.
            self._transition(RuntimeState.STOPPING)

    def _show_stt_error(self) -> None:
        self._finish_dictation_tracking()
        self._dictation_cancelled = False
        self._dictation_limit_pending = False
        self._dictation_integrity_pending = False
        self._error_message = _STT_ERROR_MESSAGE
        self._answer.setPlainText(f"Ошибка: {_STT_ERROR_MESSAGE}")
        self._transition(RuntimeState.ERROR)

    def _on_stt_finished(self) -> None:
        if self._closing:
            self._finish_dictation_tracking()
            self._maybe_close()
            return
        if not self._dictation_session_active:
            return

        if (
            self._stt_failure_pending
            or (
                self._stt_controller is not None
                and self._stt_controller.state.value == "error"
            )
        ):
            self._stt_failure_pending = False
            self._show_stt_error()
            return

        if self._dictation_cancelled:
            self._finish_dictation_tracking()
            self._dictation_cancelled = False
            self._dictation_limit_pending = False
            self._dictation_integrity_pending = False
            self._prompt.clear()
            if self._runtime_state == RuntimeState.STOPPING:
                self._transition(RuntimeState.IDLE)
            self._apply_controls()
            return

        if self._dictation_limit_pending:
            self._finish_dictation_tracking()
            self._dictation_limit_pending = False
            self._dictation_integrity_pending = False
            self._error_message = _STT_AGGREGATION_LIMIT_MESSAGE
            self._answer.setPlainText(f"Ошибка: {_STT_AGGREGATION_LIMIT_MESSAGE}")
            self._transition(RuntimeState.ERROR)
            self._apply_controls()
            return

        prompt = self._joined_dictation_text()
        indexed = [index for index, _ in self._dictation_segments if index is not None]
        has_unindexed = any(index is None for index, _ in self._dictation_segments)
        has_index_gap = bool(indexed) and (
            has_unindexed or sorted(indexed) != list(range(len(indexed)))
        )
        if self._dictation_integrity_pending or has_index_gap:
            self._finish_dictation_tracking()
            self._dictation_integrity_pending = False
            self._error_message = _STT_AGGREGATION_INCOMPLETE_MESSAGE
            self._answer.setPlainText(f"Ошибка: {_STT_AGGREGATION_INCOMPLETE_MESSAGE}")
            self._transition(RuntimeState.ERROR)
            self._apply_controls()
            return

        prompt = prompt.strip()
        self._finish_dictation_tracking()
        if prompt:
            self._prompt.setPlainText(prompt)
            if self._runtime_state == RuntimeState.STOPPING:
                self._transition(RuntimeState.IDLE)
            self.statusBar().showMessage(
                "Текст распознан. Нажмите «Отправить»."
            )
        elif self._runtime_state == RuntimeState.STOPPING:
            self._transition(RuntimeState.IDLE)
        self._apply_controls()

    def _send_prompt(self) -> None:
        state = "closed" if self._closing else self._runtime_state.value
        if self._closing:
            DIAGNOSTICS.record(
                "llm.send_requested", state=state, outcome="blocked"
            )
            return
        if self._worker is not None or self._runtime_state == RuntimeState.PROCESSING:
            request_id = self._active_request_id or self._worker_request_id
            DIAGNOSTICS.record(
                "llm.send_requested",
                request_id=request_id,
                state=state,
                outcome="blocked",
            )
            DIAGNOSTICS.record(
                "llm.duplicate_send_blocked",
                request_id=request_id,
                state="processing",
                outcome="duplicate",
            )
            return
        if self._runtime_state not in {RuntimeState.IDLE, RuntimeState.ERROR}:
            DIAGNOSTICS.record("llm.send_requested", state=state, outcome="blocked")
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            DIAGNOSTICS.record("llm.send_requested", state=state, outcome="blocked")
            self.statusBar().showMessage("Сначала введите вопрос.")
            return
        DIAGNOSTICS.record("llm.send_requested", state=state, outcome="accepted")
        self._start_generation(prompt)

    def _start_generation(self, prompt: str) -> None:
        cleaned_prompt = prompt.strip()
        if (
            not cleaned_prompt
            or self._closing
            or self._worker is not None
            or self._runtime_state
            not in {RuntimeState.IDLE, RuntimeState.ERROR, RuntimeState.STOPPING}
        ):
            return

        self._answer.clear()
        self._cancelled = False
        self._error_message = None
        request_id = new_opaque_id()
        self._worker_request_id = request_id
        self._active_request_id = request_id
        DIAGNOSTICS.record(
            "llm.request_created",
            request_id=request_id,
            state="processing",
            outcome="accepted",
        )
        self._worker = GenerationWorker(self._assistant, cleaned_prompt, request_id)
        self._worker.request_text_received.connect(self._append_text)
        self._worker.request_error_occurred.connect(self._show_error)
        self._worker.request_cancelled.connect(self._mark_cancelled)
        self._worker.request_completed.connect(self._on_request_completed)
        self._worker.finished.connect(self._on_worker_finished)
        if not self._transition(RuntimeState.PROCESSING):
            DIAGNOSTICS.record(
                "llm.request_failed",
                request_id=request_id,
                state="failed",
                outcome="failed",
            )
            self._active_request_id = None
            self._worker_request_id = None
            self._worker.deleteLater()
            self._worker = None
            return
        self._apply_controls()
        try:
            self._worker.start()
        except Exception:
            worker = self._worker
            DIAGNOSTICS.record(
                "llm.request_failed",
                request_id=request_id,
                state="failed",
                outcome="failed",
            )
            self._error_message = _LLM_ERROR_MESSAGE
            self._answer.setPlainText(f"Ошибка: {self._error_message}")
            self._active_request_id = None
            self._worker_request_id = None
            self._worker = None
            if worker is not None:
                worker.deleteLater()
            self._transition(RuntimeState.ERROR)

    def _cancel_generation(self) -> None:
        if self._closing:
            return
        if self._dictation_session_active:
            self._dictation_cancelled = True
            self._clear_dictation_segments()
            self._prompt.clear()
            if self._runtime_state != RuntimeState.STOPPING:
                self._transition(RuntimeState.STOPPING)
            if self._stt_controller is not None:
                self._stt_controller.cancel()
            self._apply_controls()
            return
        if self._worker is None:
            return
        self._cancelled = True
        if self._active_request_id is not None:
            DIAGNOSTICS.record(
                "llm.request_cancel_requested",
                request_id=self._active_request_id,
                state="stopping",
                outcome="user_cancel",
            )
            self._active_request_id = None
        self._transition(RuntimeState.STOPPING)
        self._worker.cancel()

    def _append_text(self, request_id: str, text: str) -> None:
        if request_id != self._active_request_id or self._cancelled or self._closing:
            DIAGNOSTICS.record(
                "llm.callback_ignored_stale",
                request_id=request_id,
                state="closed" if self._closing else self._runtime_state.value,
                outcome="stale",
            )
            return
        cursor = self._answer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._answer.setTextCursor(cursor)

    def _show_error(self, request_id: str, _message: str) -> None:
        # The callback carries the request ID; only the fixed UI error is retained.
        if request_id != self._active_request_id or self._closing:
            DIAGNOSTICS.record(
                "llm.callback_ignored_stale",
                request_id=request_id,
                state="closed" if self._closing else self._runtime_state.value,
                outcome="stale",
            )
            return
        self._error_message = _LLM_ERROR_MESSAGE

    def _mark_cancelled(self, request_id: str) -> None:
        if request_id in {self._active_request_id, self._worker_request_id}:
            self._cancelled = True
        else:
            DIAGNOSTICS.record(
                "llm.callback_ignored_stale",
                request_id=request_id,
                state="closed" if self._closing else self._runtime_state.value,
                outcome="stale",
            )

    def _on_request_completed(self, request_id: str, outcome: str) -> None:
        if request_id != self._worker_request_id:
            DIAGNOSTICS.record(
                "llm.callback_ignored_stale",
                request_id=request_id,
                state="closed" if self._closing else "stopping",
                outcome="stale",
            )
            return
    def _on_worker_finished(self) -> None:
        worker = self.sender()
        request_id = worker.request_id if isinstance(worker, GenerationWorker) else None
        if request_id is None or request_id != self._worker_request_id:
            return
        is_active = self._active_request_id == request_id
        if is_active and worker._outcome == "error" and self._error_message:
            self._answer.setPlainText(f"Ошибка: {self._error_message}")
            self._transition(RuntimeState.ERROR)
        elif self._runtime_state in {RuntimeState.PROCESSING, RuntimeState.STOPPING}:
            self._transition(RuntimeState.IDLE)
        if is_active:
            self._active_request_id = None
        worker = self._worker
        self._worker = None
        self._worker_request_id = None
        if worker is not None:
            worker.deleteLater()
        self._apply_controls()
        self._maybe_close()

    def _stt_is_active(self) -> bool:
        if self._stt_controller is None:
            return False
        state = self._stt_controller.state
        state_value = getattr(state, "value", state)
        return state_value in {"starting", "listening", "stopping"}

    def _maybe_close(self) -> None:
        if self._closing and self._worker is None and not self._stt_is_active():
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None or self._stt_is_active():
            self._closing = True
            self._cancelled = True
            if self._active_request_id is not None:
                DIAGNOSTICS.record(
                    "llm.request_cancel_requested",
                    request_id=self._active_request_id,
                    state="closed",
                    outcome="window_close",
                )
                self._active_request_id = None
            self._dictation_cancelled = True
            self._clear_dictation_segments()
            if self._runtime_state != RuntimeState.STOPPING:
                self._transition(RuntimeState.STOPPING)
            if self._stt_is_active() and self._stt_controller is not None:
                self._stt_controller.cancel()
            if self._worker is not None:
                self._worker.cancel()
            self._apply_controls()
            event.ignore()
            return
        event.accept()
