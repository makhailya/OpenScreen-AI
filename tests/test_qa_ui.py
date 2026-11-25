"""Offscreen UI worker lifecycle checks with synthetic assistant responses."""

from __future__ import annotations

import asyncio
import os
from threading import Event, get_ident

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QObject, Qt, QTimer, Slot
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from openscreen_ai.ui.main_window import GenerationWorker, MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def wait_until(predicate, app: QApplication, timeout_ms: int = 3000) -> None:
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTest.qWait(10)
        elapsed += 10
    assert predicate()


class BlockingAssistant:
    def __init__(self) -> None:
        self.entered = Event()
        self.cleaned = Event()
        self.owner_thread: int | None = None

    async def stream_response(self, prompt: str, *, request_id: str | None = None):
        self.owner_thread = get_ident()
        self.entered.set()
        try:
            yield f"synthetic response to {prompt}"
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()


class FailingAssistant:
    async def stream_response(self, _prompt: str, *, request_id: str | None = None):
        raise RuntimeError("synthetic failure")
        yield "unreachable"


class MainThreadReceiver(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[str, int]] = []
        self.cancelled = 0

    @Slot(str)
    def receive_text(self, text: str) -> None:
        self.received.append((text, get_ident()))

    @Slot()
    def receive_cancelled(self) -> None:
        self.cancelled += 1


def test_worker_signals_cross_to_main_thread_and_cancel_responsively(app: QApplication) -> None:
    assistant = BlockingAssistant()
    worker = GenerationWorker(assistant, "synthetic prompt")
    receiver = MainThreadReceiver()
    worker.text_received.connect(receiver.receive_text, Qt.ConnectionType.QueuedConnection)
    worker.generation_cancelled.connect(
        receiver.receive_cancelled, Qt.ConnectionType.QueuedConnection
    )
    finished = QSignalSpy(worker.finished)
    timer_ticks: list[bool] = []
    QTimer.singleShot(0, lambda: timer_ticks.append(True))

    worker.start()
    wait_until(lambda: assistant.entered.is_set() and receiver.received, app)

    assert receiver.received[0][0] == "synthetic response to synthetic prompt"
    assert receiver.received[0][1] == get_ident()
    assert assistant.owner_thread != get_ident()
    assert timer_ticks == [True]

    worker.cancel()
    wait_until(lambda: finished.count() == 1 and receiver.cancelled == 1, app)
    assert assistant.cleaned.is_set()


def test_close_while_request_active_cancels_without_showing_window(app: QApplication) -> None:
    assistant = BlockingAssistant()
    window = MainWindow(assistant)
    window.show()
    wait_until(lambda: window.isVisible(), app)
    window._prompt.setPlainText("synthetic close prompt")
    window._send_prompt()
    wait_until(
        lambda: assistant.entered.is_set()
        and window._answer.toPlainText() == "synthetic response to synthetic close prompt",
        app,
    )
    assert window.isVisible()

    timer_ticks: list[bool] = []
    QTimer.singleShot(0, lambda: timer_ticks.append(True))
    close_accepted = window.close()

    assert close_accepted is False
    assert window._closing is True
    assert window.isVisible()
    wait_until(lambda: timer_ticks, app)
    assert timer_ticks == [True]
    wait_until(
        lambda: assistant.cleaned.is_set() and window._worker is None and not window.isVisible(),
        app,
    )
    assert window._cancelled is True
    assert window._answer.toPlainText() == "synthetic response to synthetic close prompt"


def test_unexpected_failure_is_delivered_as_generic_ui_error(app: QApplication) -> None:
    window = MainWindow(FailingAssistant())
    window._prompt.setPlainText("synthetic error prompt")
    window._send_prompt()

    wait_until(lambda: window._worker is None, app)

    assert window._answer.toPlainText() == (
        "Ошибка: Произошла внутренняя ошибка при генерации ответа."
    )
