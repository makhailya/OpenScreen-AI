"""Главное окно MVP и потоковая передача ответов в интерфейс."""

from __future__ import annotations

import asyncio
import logging
import threading

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
from openscreen_ai.providers.ollama_provider import OllamaError

logger = logging.getLogger(__name__)


class GenerationWorker(QThread):
    """Выполняет async-запрос в отдельном потоке Qt и поддерживает отмену."""

    text_received = Signal(str)
    error_occurred = Signal(str)
    generation_cancelled = Signal()

    def __init__(self, assistant: Assistant, prompt: str) -> None:
        super().__init__()
        self._assistant = assistant
        self._prompt = prompt
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
            loop.call_soon_threadsafe(task.cancel)

    def run(self) -> None:
        asyncio.run(self._consume())

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
            async for chunk in self._assistant.stream_response(self._prompt):
                self.text_received.emit(chunk)
        except asyncio.CancelledError:
            self.generation_cancelled.emit()
        except OllamaError as error:
            self.error_occurred.emit(str(error))
        except ValueError as error:
            self.error_occurred.emit(str(error))
        except Exception:
            logger.exception("Unexpected error during assistant generation")
            self.error_occurred.emit("Произошла внутренняя ошибка при генерации ответа.")
        finally:
            with self._state_lock:
                self._loop = None
                self._task = None


class MainWindow(QMainWindow):
    """Окно чата с локальным AI-провайдером."""

    def __init__(self, assistant: Assistant) -> None:
        super().__init__()
        self._assistant = assistant
        self._worker: GenerationWorker | None = None
        self._closing = False
        self._cancelled = False
        self._error_message: str | None = None

        self.setWindowTitle("OpenScreen AI")
        self.resize(720, 560)
        self._build_ui()
        self._set_busy(False)

    def _build_ui(self) -> None:
        self._answer = QPlainTextEdit()
        self._answer.setReadOnly(True)
        self._answer.setPlaceholderText("Ответ локальной модели появится здесь…")

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Напишите вопрос о Python или разработке…")
        self._prompt.setFixedHeight(110)

        self._send_button = QPushButton("Отправить")
        self._send_button.clicked.connect(self._send_prompt)
        self._cancel_button = QPushButton("Отменить")
        self._cancel_button.clicked.connect(self._cancel_generation)

        actions = QHBoxLayout()
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
        self.statusBar().showMessage("Готово")

    def _send_prompt(self) -> None:
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self.statusBar().showMessage("Сначала введите вопрос.")
            return

        self._answer.clear()
        self._cancelled = False
        self._error_message = None
        self._worker = GenerationWorker(self._assistant, prompt)
        self._worker.text_received.connect(self._append_text)
        self._worker.error_occurred.connect(self._show_error)
        self._worker.generation_cancelled.connect(self._mark_cancelled)
        self._worker.finished.connect(self._on_worker_finished)
        self._set_busy(True)
        self.statusBar().showMessage("Генерирую ответ…")
        self._worker.start()

    def _cancel_generation(self) -> None:
        if self._worker is None:
            return
        self._cancelled = True
        self._cancel_button.setEnabled(False)
        self.statusBar().showMessage("Отменяю генерацию…")
        self._worker.cancel()

    def _append_text(self, text: str) -> None:
        cursor = self._answer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._answer.setTextCursor(cursor)

    def _show_error(self, message: str) -> None:
        self._error_message = message

    def _mark_cancelled(self) -> None:
        self._cancelled = True

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

        self._set_busy(False)
        if self._error_message:
            self._answer.setPlainText(f"Ошибка: {self._error_message}")
            self.statusBar().showMessage("Не удалось получить ответ.")
        elif self._cancelled:
            self.statusBar().showMessage("Генерация отменена.")
        else:
            self.statusBar().showMessage("Готово")

        if self._closing:
            QTimer.singleShot(0, self.close)

    def _set_busy(self, busy: bool) -> None:
        self._prompt.setReadOnly(busy)
        self._send_button.setEnabled(not busy)
        self._cancel_button.setEnabled(busy)
        self._cancel_button.setVisible(busy)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._closing = True
            self._cancel_button.setEnabled(False)
            self.statusBar().showMessage("Завершаю текущий запрос…")
            self._worker.cancel()
            event.ignore()
            return
        event.accept()
