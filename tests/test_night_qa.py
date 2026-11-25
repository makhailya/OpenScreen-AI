"""Regression checks for late LLM UI delivery during cancellation and close."""

from __future__ import annotations

import os
from threading import Event

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QApplication

from openscreen_ai.ui.main_window import MainWindow, RuntimeState


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class FakeAssistant:
    async def stream_response(self, _prompt: str, *, request_id: str | None = None):
        if False:
            yield "unreachable"


class FakeWorker(QObject):
    text_received = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


class LateTextEmitter(QThread):
    def __init__(self, worker: FakeWorker) -> None:
        super().__init__()
        self._worker = worker
        self.emitted = Event()

    def run(self) -> None:
        self._worker.text_received.emit("late synthetic answer")
        self.emitted.set()


@pytest.mark.parametrize("action", ["cancel", "close"])
def test_queued_llm_text_is_ignored_after_cancel_or_close(
    app: QApplication, action: str
) -> None:
    window = MainWindow(FakeAssistant())
    worker = FakeWorker()
    window._transition(RuntimeState.PROCESSING)
    window._worker = worker  # type: ignore[assignment]
    worker.text_received.connect(
        window._append_text,
        Qt.ConnectionType.QueuedConnection,
    )
    emitter = LateTextEmitter(worker)

    emitter.start()
    assert emitter.emitted.wait(1), "fake worker did not queue its text signal"

    if action == "cancel":
        window._cancel_generation()
        assert window.runtime_state == RuntimeState.STOPPING
    else:
        assert window.close() is False
        assert window._closing is True
        assert window.runtime_state == RuntimeState.STOPPING

    assert worker.cancel_calls == 1
    app.processEvents()

    assert window._answer.toPlainText() == ""

    assert emitter.wait(1000)
    window._worker = None
    window._closing = False
    window._transition(RuntimeState.IDLE)
    window.close()
