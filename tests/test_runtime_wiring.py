"""Fake-only checks for the application STT-to-LLM flow."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from threading import Event, get_ident

import httpx

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from openscreen_ai import main as app_main
from openscreen_ai.config import AppConfig
from openscreen_ai.core.assistant import Assistant
from openscreen_ai.core.request_lifecycle import DIAGNOSTICS
from openscreen_ai.main import create_window
from openscreen_ai.providers.ollama_provider import OllamaProvider
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttResult
from openscreen_ai.stt.errors import SttFilesError
from openscreen_ai.stt.worker import SttDiagnostic, SttState
from openscreen_ai.ui.main_window import GenerationWorker, MainWindow, RuntimeState


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def wait_until(predicate, timeout_ms: int = 3000) -> None:
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTest.qWait(10)
        elapsed += 10
    assert predicate()


class FakeSttController(QObject):
    transcription_ready = Signal(object)
    diagnostic_ready = Signal(object)
    state_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.state = SttState.IDLE
        self.starts = 0
        self.stop_requests = 0
        self.cancel_requests = 0
        self.session_id = 0

    def start(self) -> bool:
        if self.state not in {SttState.IDLE, SttState.STOPPED, SttState.ERROR}:
            return False
        self.starts += 1
        self.session_id += 1
        self.state = SttState.STARTING
        self.state_changed.emit(self.state.value)
        self.state = SttState.LISTENING
        self.state_changed.emit(self.state.value)
        return True

    def request_stop(self) -> None:
        if self.state in {SttState.STARTING, SttState.LISTENING}:
            self.stop_requests += 1
            self.state = SttState.STOPPING
            self.state_changed.emit(self.state.value)

    def cancel(self) -> None:
        self.cancel_requests += 1
        self.request_stop()

    def finish(self) -> None:
        self.state = SttState.STOPPED
        self.finished.emit()

    def fail(self, message: str) -> None:
        self.report_error(message)
        self.finish_failure()

    def report_error(self, message: str) -> None:
        self.error_occurred.emit(message)

    def finish_failure(self) -> None:
        self.state = SttState.ERROR
        self.state_changed.emit(self.state.value)
        self.finished.emit()


class FakeAssistant:
    def __init__(self, *, failure: str | None = None, block: bool = False) -> None:
        self.prompts: list[str] = []
        self.request_ids: list[str | None] = []
        self.failure = failure
        self.entered = Event()
        self.cleaned = Event()
        self.block = block

    async def stream_response(self, prompt: str, *, request_id: str | None = None):
        self.prompts.append(prompt)
        self.request_ids.append(request_id)
        self.entered.set()
        try:
            if self.failure is not None:
                raise RuntimeError(self.failure)
            yield f"Ответ: {prompt}"
            if self.block:
                await asyncio.Event().wait()
        finally:
            self.cleaned.set()


class SignalEmitter(QThread):
    def __init__(self, signal: Signal, payload: object) -> None:
        super().__init__()
        self._signal = signal
        self._payload = payload
        self.owner_thread: int | None = None

    def run(self) -> None:
        self.owner_thread = get_ident()
        self._signal.emit(self._payload)


class ThreadTrackingWindow(MainWindow):
    def __init__(self, *args, **kwargs) -> None:
        self.transcription_slot_thread: int | None = None
        super().__init__(*args, **kwargs)

    def _on_transcription_ready(self, stt_result: SttResult) -> None:
        self.transcription_slot_thread = get_ident()
        super()._on_transcription_ready(stt_result)


def result(
    text: str, *, session_id: int | None = None, segment_index: int | None = None
) -> SttResult:
    return SttResult(
        text,
        1.0,
        0.9,
        0.2,
        0.8,
        15.0,
        3.0,
        0.7,
        segment_index=segment_index,
        session_id=session_id,
    )


def test_create_window_injects_configured_lazy_stt_controller(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_main, "check_stt_available", lambda _config: None)
    monkeypatch.setenv("OPENSCREEN_STT_MODEL_DIR", "/synthetic/parakeet")
    monkeypatch.setenv("OPENSCREEN_STT_VAD_MODEL_PATH", "/synthetic/vad.onnx")
    captured: list[SttConfig] = []
    controller = FakeSttController()

    def factory(config: SttConfig) -> FakeSttController:
        captured.append(config)
        return controller

    window = create_window(AppConfig(), stt_controller_factory=factory)
    assert captured == [
        SttConfig(
            model_dir=Path("/synthetic/parakeet"),
            vad_model_path=Path("/synthetic/vad.onnx"),
        )
    ]
    assert window._stt_controller is controller
    assert controller.starts == 0
    window.close()


def test_unready_stt_is_disabled_without_blocking_text_chat(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "SYNTHETIC_PRIVATE_STT_PATH_CANARY"
    monkeypatch.delenv("OPENSCREEN_STT_MODEL_DIR", raising=False)
    monkeypatch.delenv("OPENSCREEN_STT_VAD_MODEL_PATH", raising=False)

    def unavailable(_config: SttConfig) -> None:
        raise SttFilesError(missing_vad_model=Path(f"/private/{canary}"))

    monkeypatch.setattr(app_main, "check_stt_available", unavailable)
    factory_calls: list[SttConfig] = []

    def factory(config: SttConfig) -> FakeSttController:
        factory_calls.append(config)
        return FakeSttController()

    window = create_window(AppConfig(), stt_controller_factory=factory)

    assert window._stt_controller is None
    assert window._listen_button.isHidden()
    assert window._send_button.isEnabled()
    assert "doctor" in window.statusBar().currentMessage()
    assert canary not in window.statusBar().currentMessage()
    assert factory_calls == []
    assert "STT_OPTIONAL_NOT_READY" in caplog.text
    assert canary not in caplog.text
    window.close()


def test_manual_send_after_stt_stop_starts_one_llm_request(app: QApplication) -> None:
    DIAGNOSTICS.reset()
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)

    window._toggle_listening()
    assert window.runtime_state == RuntimeState.LISTENING
    controller.transcription_ready.emit(result("  синтетический вопрос  "))
    assert window._prompt.toPlainText() == "синтетический вопрос"
    assert controller.stop_requests == 0
    assert window.runtime_state == RuntimeState.LISTENING

    controller.transcription_ready.emit(result("вторая часть"))
    assert window._prompt.toPlainText() == "синтетический вопрос вторая часть"
    assert assistant.prompts == []
    assert DIAGNOSTICS.snapshot()["logical_requests_total"] == 0
    assert DIAGNOSTICS.snapshot()["http_attempts_total"] == 0
    controller.request_stop()
    assert controller.stop_requests == 1
    assert window.runtime_state == RuntimeState.STOPPING
    controller.finish()

    assert assistant.prompts == []
    assert window._worker is None
    assert window.runtime_state == RuntimeState.IDLE
    assert window._prompt.toPlainText() == "синтетический вопрос вторая часть"
    assert window._send_button.isEnabled()
    assert "Нажмите «Отправить»" in window.statusBar().currentMessage()

    window._send_button.click()
    wait_until(lambda: window._worker is None)

    assert assistant.prompts == ["синтетический вопрос вторая часть"]
    assert DIAGNOSTICS.snapshot()["logical_requests_total"] == 1
    assert window._answer.toPlainText() == "Ответ: синтетический вопрос вторая часть"
    assert window.runtime_state == RuntimeState.IDLE
    window.close()


def test_one_send_has_one_request_id_one_attempt_and_one_answer(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    DIAGNOSTICS.reset()
    prompt = "SYNTHETIC_PROMPT_PRIVACY_CANARY"
    answer = "SYNTHETIC_ANSWER_PRIVACY_CANARY"
    requests: list[httpx.Request] = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            content=(f'{{"response":"{answer}","done":true}}\n').encode(),
            request=request,
        )

    provider = OllamaProvider(AppConfig(), transport=httpx.MockTransport(handle))
    window = MainWindow(Assistant(provider))
    window._prompt.setPlainText(prompt)
    with caplog.at_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle"):
        window._send_prompt()
        wait_until(lambda: window._worker is None)

    counts = DIAGNOSTICS.snapshot()
    assert set(counts) == {
        "send_actions_total",
        "logical_requests_total",
        "http_attempts_total",
        "completed_requests_total",
        "cancelled_requests_total",
        "failed_requests_total",
        "duplicate_send_blocked_total",
        "stale_callbacks_ignored_total",
    }
    assert counts["send_actions_total"] == 1
    assert counts["logical_requests_total"] == 1
    assert counts["http_attempts_total"] == 1
    assert counts["completed_requests_total"] == 1
    assert counts["cancelled_requests_total"] == 0
    assert counts["failed_requests_total"] == 0
    assert counts["duplicate_send_blocked_total"] == 0
    assert counts["stale_callbacks_ignored_total"] == 0
    assert len(requests) == 1
    assert window._answer.toPlainText() == answer
    request_lines = [record.getMessage() for record in caplog.records]
    accepted = next(line for line in request_lines if line.startswith("llm.request_created"))
    attempt = next(line for line in request_lines if line.startswith("llm.http_attempt_started"))
    response = next(line for line in request_lines if line.startswith("llm.http_attempt_finished"))
    assert sum(line.startswith("llm.http_attempt_started") for line in request_lines) == 1
    assert sum(line.startswith("llm.http_attempt_finished") for line in request_lines) == 1
    request_id = accepted.split("request_id=")[1].split()[0]
    attempt_id = attempt.split("attempt_id=")[1].split()[0]
    assert request_id != "none" and attempt_id != "none"
    assert f"request_id={request_id}" in attempt and f"request_id={request_id}" in response
    assert f"attempt_id={attempt_id}" in response
    assert response.endswith("state=completed outcome=success")
    assert "llm.send_requested" in caplog.text
    assert "llm.request_started" in caplog.text
    assert "llm.request_completed" in caplog.text
    assert prompt not in caplog.text and answer not in caplog.text
    assert "127.0.0.1" not in caplog.text
    assert DIAGNOSTICS.pending_timing_count == 0
    assert DIAGNOSTICS.pending_attempt_count == 0
    window.close()


def test_active_duplicate_send_is_counted_and_rejected(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    DIAGNOSTICS.reset()
    caplog.set_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle")
    assistant = FakeAssistant(block=True)
    window = MainWindow(assistant)
    window._prompt.setPlainText("same synthetic prompt")
    window._send_prompt()
    wait_until(lambda: assistant.entered.is_set() and window._answer.toPlainText())
    request_id = window._active_request_id

    window._send_prompt()

    counts = DIAGNOSTICS.snapshot()
    assert counts["send_actions_total"] == 2
    assert counts["logical_requests_total"] == 1
    assert counts["duplicate_send_blocked_total"] == 1
    assert caplog.text.count("llm.send_requested") == 2
    assert "llm.duplicate_send_blocked" in caplog.text
    assert assistant.prompts == ["same synthetic prompt"]
    assert assistant.request_ids == [request_id]
    window._cancel_generation()
    wait_until(lambda: window._worker is None)
    window.close()


def test_same_text_send_after_completion_gets_fresh_request_id(app: QApplication) -> None:
    DIAGNOSTICS.reset()
    assistant = FakeAssistant()
    window = MainWindow(assistant)
    window._prompt.setPlainText("deliberate repeated prompt")

    window._send_prompt()
    wait_until(lambda: window._worker is None)
    first_id = assistant.request_ids[0]
    window._send_prompt()
    wait_until(lambda: window._worker is None and len(assistant.request_ids) == 2)

    assert assistant.prompts == ["deliberate repeated prompt"] * 2
    assert first_id is not None and assistant.request_ids[1] != first_id
    assert DIAGNOSTICS.snapshot()["logical_requests_total"] == 2
    assert DIAGNOSTICS.pending_timing_count == 0
    window.close()


def test_cancel_invalidates_late_callbacks_before_next_send(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    DIAGNOSTICS.reset()
    caplog.set_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle")
    assistant = FakeAssistant(block=True)
    window = MainWindow(assistant)
    window._prompt.setPlainText("first request")
    window._send_prompt()
    wait_until(lambda: assistant.entered.is_set() and bool(window._answer.toPlainText()))
    old_id = window._active_request_id
    old_answer = window._answer.toPlainText()

    window._cancel_generation()
    window._append_text(old_id or "", "stale chunk")
    assert window._answer.toPlainText() == old_answer
    wait_until(lambda: window._worker is None)

    assistant.block = False
    window._prompt.setPlainText("second request")
    window._send_prompt()
    wait_until(lambda: window._worker is None and len(assistant.request_ids) == 2)
    new_id = assistant.request_ids[1]
    current_answer = window._answer.toPlainText()
    window._append_text(old_id or "", "stale chunk")
    window._on_request_completed(old_id or "", "success")

    assert new_id != old_id
    assert window._answer.toPlainText() == current_answer
    counts = DIAGNOSTICS.snapshot()
    assert counts["cancelled_requests_total"] == 1
    assert counts["completed_requests_total"] == 1
    assert counts["stale_callbacks_ignored_total"] == 3
    assert DIAGNOSTICS.pending_timing_count == 0
    assert DIAGNOSTICS.pending_attempt_count == 0
    assert "llm.request_cancel_requested" in caplog.text
    assert "llm.request_cancelled" in caplog.text
    assert "llm.callback_ignored_stale" in caplog.text
    window.close()


def test_failed_request_is_counted_and_cleans_timing(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    DIAGNOSTICS.reset()
    caplog.set_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle")
    assistant = FakeAssistant(failure="synthetic provider failure")
    window = MainWindow(assistant)
    window._prompt.setPlainText("synthetic failing request")
    window._send_prompt()
    wait_until(lambda: window._worker is None)

    counts = DIAGNOSTICS.snapshot()
    assert counts["logical_requests_total"] == 1
    assert counts["failed_requests_total"] == 1
    assert counts["completed_requests_total"] == 0
    assert counts["cancelled_requests_total"] == 0
    assert DIAGNOSTICS.pending_timing_count == 0
    assert DIAGNOSTICS.pending_attempt_count == 0
    assert "llm.request_failed" in caplog.text
    assert window.runtime_state == RuntimeState.ERROR
    window.close()


def test_cancel_then_fast_send_is_blocked_against_current_worker(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    DIAGNOSTICS.reset()
    caplog.set_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle")
    assistant = FakeAssistant(block=True)
    window = MainWindow(assistant)
    window._prompt.setPlainText("cancel then send")
    window._send_prompt()
    wait_until(lambda: assistant.entered.is_set() and bool(window._answer.toPlainText()))
    current_worker_id = window._worker_request_id

    window._cancel_generation()
    assert window._active_request_id is None
    raised: Exception | None = None
    try:
        window._send_prompt()
    except Exception as error:
        raised = error
    finally:
        if window._worker is not None:
            window._cancel_generation()
        wait_until(lambda: window._worker is None)

    counts = DIAGNOSTICS.snapshot()
    blocked_records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("llm.duplicate_send_blocked")
    ]
    blocked = blocked_records[0] if blocked_records else ""
    window.close()
    assert raised is None, f"fast duplicate send raised {type(raised).__name__}"
    assert counts["logical_requests_total"] == 1
    assert counts["duplicate_send_blocked_total"] == 1
    assert assistant.request_ids == [current_worker_id]
    assert f"request_id={current_worker_id}" in blocked
    assert window._worker is None


def test_worker_start_failure_cleans_request_with_generic_ui_error(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    DIAGNOSTICS.reset()
    caplog.set_level(logging.INFO, logger="openscreen_ai.core.request_lifecycle")

    def fail_start(_worker: GenerationWorker) -> None:
        raise RuntimeError("SYNTHETIC_START_FAILURE_CANARY")

    monkeypatch.setattr(GenerationWorker, "start", fail_start)
    window = MainWindow(FakeAssistant())
    window._prompt.setPlainText("synthetic start failure")
    raised: Exception | None = None
    try:
        window._send_prompt()
    except Exception as error:
        raised = error

    stranded_worker = window._worker
    observed = (
        type(raised).__name__ if raised is not None else None,
        window._error_message,
        window.runtime_state,
        window._answer.toPlainText(),
        stranded_worker is not None,
        window._worker_request_id,
        window._active_request_id,
        DIAGNOSTICS.pending_timing_count,
        DIAGNOSTICS.snapshot()["failed_requests_total"],
    )
    if stranded_worker is not None:
        window._worker = None
        window._worker_request_id = None
        window._active_request_id = None
        stranded_worker.deleteLater()
    DIAGNOSTICS.reset()
    window.close()

    assert observed == (
        None,
        "Произошла внутренняя ошибка при генерации ответа.",
        RuntimeState.ERROR,
        "Ошибка: Произошла внутренняя ошибка при генерации ответа.",
        False,
        None,
        None,
        0,
        1,
    )
    assert "SYNTHETIC_START_FAILURE_CANARY" not in caplog.text


@pytest.mark.parametrize(
    "segments",
    [
        ["один сегмент"],
        ["первая часть", "вторая часть"],
        ["первая часть", "вторая часть", "третья часть"],
    ],
    ids=["one-segment", "two-segments", "three-segments"],
)
def test_dictation_stop_aggregates_segments_once_in_order(
    app: QApplication, segments: list[str]
) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)

    window._toggle_listening()
    for segment in segments:
        controller.transcription_ready.emit(result(segment))

    assert controller.stop_requests == 0
    assert assistant.prompts == []
    assert window._prompt.toPlainText() == " ".join(segments)

    window._toggle_listening()
    assert controller.stop_requests == 1
    controller.request_stop()
    controller.finish()

    assert assistant.prompts == []
    assert window._worker is None
    assert window.runtime_state == RuntimeState.IDLE
    assert window._prompt.toPlainText() == " ".join(segments)
    window._send_button.click()
    wait_until(lambda: window._worker is None)

    assert assistant.prompts == [" ".join(segments)]
    assert window.runtime_state == RuntimeState.IDLE
    window.close()


def test_queued_segment_is_kept_after_stop_until_controller_finishes(
    app: QApplication,
) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()

    controller.request_stop()
    controller.request_stop()
    assert window.runtime_state == RuntimeState.STOPPING
    assert controller.stop_requests == 1
    controller.transcription_ready.emit(result("queued before stop"))
    controller.finish()

    assert assistant.prompts == []
    assert window._prompt.toPlainText() == "queued before stop"
    assert window.runtime_state == RuntimeState.IDLE
    window._send_button.click()
    wait_until(lambda: window._worker is None)
    assert assistant.prompts == ["queued before stop"]
    window.close()


def test_cancel_dictation_discards_segments_without_llm_request(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    controller.transcription_ready.emit(result("partial text"))

    assert not window._cancel_button.isHidden()
    assert window._cancel_button.text() == "Отменить диктовку"
    window._cancel_button.click()
    assert controller.stop_requests == 1
    assert controller.cancel_requests == 1
    assert window.runtime_state == RuntimeState.STOPPING
    controller.finish()

    assert assistant.prompts == []
    assert window._prompt.toPlainText() == ""
    assert window.runtime_state == RuntimeState.IDLE
    window.close()


def test_new_dictation_ignores_late_segment_from_old_session(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)

    window._toggle_listening()
    old_session = controller.session_id
    controller.request_stop()
    controller.finish()

    window._toggle_listening()
    current_session = controller.session_id
    assert current_session != old_session
    controller.transcription_ready.emit(
        result("stale phrase", session_id=old_session, segment_index=0)
    )
    controller.transcription_ready.emit(
        result("current phrase", session_id=current_session, segment_index=0)
    )
    assert window._prompt.toPlainText() == "current phrase"

    controller.request_stop()
    controller.finish()
    assert assistant.prompts == []
    assert window._prompt.toPlainText() == "current phrase"
    window._send_button.click()
    wait_until(lambda: window._worker is None)
    assert assistant.prompts == ["current phrase"]
    window.close()


def test_duplicate_segment_event_is_only_aggregated_once(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    duplicate = result("repeat-safe", session_id=controller.session_id, segment_index=0)

    controller.transcription_ready.emit(duplicate)
    controller.transcription_ready.emit(duplicate)
    assert window._prompt.toPlainText() == "repeat-safe"

    controller.request_stop()
    controller.finish()
    assert assistant.prompts == []
    assert window._prompt.toPlainText() == "repeat-safe"
    window._send_button.click()
    wait_until(lambda: window._worker is None)
    assert assistant.prompts == ["repeat-safe"]
    window.close()


def test_indexed_segments_are_presented_in_recognition_order(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    session_id = controller.session_id

    for index, text in ((2, "третья"), (0, "первая"), (1, "вторая")):
        controller.transcription_ready.emit(
            result(text, session_id=session_id, segment_index=index)
        )

    assert window._prompt.toPlainText() == "первая вторая третья"
    window._toggle_listening()
    controller.finish()
    assert assistant.prompts == []
    assert window._prompt.toPlainText() == "первая вторая третья"
    window._send_button.click()
    wait_until(lambda: window._worker is None)
    assert assistant.prompts == ["первая вторая третья"]
    window.close()


def test_missing_segment_index_fails_closed_instead_of_sending_partial_text(
    app: QApplication,
) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    session_id = controller.session_id
    controller.transcription_ready.emit(
        result("первая часть", session_id=session_id, segment_index=0)
    )
    controller.transcription_ready.emit(
        result("третья часть", session_id=session_id, segment_index=2)
    )
    controller.request_stop()
    controller.finish()

    assert assistant.prompts == []
    assert window.runtime_state == RuntimeState.ERROR
    assert "все части" in window._answer.toPlainText()
    window.close()


def test_conflicting_duplicate_segment_identity_fails_closed(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    session_id = controller.session_id
    controller.transcription_ready.emit(
        result("original", session_id=session_id, segment_index=0)
    )
    controller.transcription_ready.emit(
        result("conflicting", session_id=session_id, segment_index=0)
    )
    assert controller.stop_requests == 1
    controller.finish()

    assert assistant.prompts == []
    assert window.runtime_state == RuntimeState.ERROR
    assert "все части" in window._answer.toPlainText()
    window.close()


def test_audio_queue_overflow_discards_partial_dictation_fail_closed(
    app: QApplication,
) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    controller.transcription_ready.emit(result("partial before audio gap"))
    controller.diagnostic_ready.emit(
        SttDiagnostic(
            queue_depth_blocks=20,
            queue_max_depth_blocks=20,
            dropped_blocks=1,
            vad_resets_total=1,
            vad_resets_gap=1,
        )
    )

    assert controller.stop_requests == 1
    controller.finish()

    assert assistant.prompts == []
    assert window.runtime_state == RuntimeState.ERROR
    assert window._prompt.toPlainText() == ""
    window.close()


def test_empty_dictation_finishes_without_empty_llm_request(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    controller.request_stop()
    controller.finish()

    assert assistant.prompts == []
    assert window.runtime_state == RuntimeState.IDLE
    assert window._worker is None
    window.close()


@pytest.mark.parametrize(
    "limit_type",
    ["characters", "segments"],
    ids=["character-limit", "segment-limit"],
)
def test_dictation_aggregation_limits_fail_closed(
    app: QApplication, limit_type: str
) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    if limit_type == "characters":
        controller.transcription_ready.emit(result("x" * 32_001))
    else:
        session_id = controller.session_id
        for index in range(257):
            controller.transcription_ready.emit(
                result(f"segment {index}", session_id=session_id,
                       segment_index=index)
            )
    assert controller.stop_requests == 1
    assert window._prompt.toPlainText() == ""
    controller.finish()

    assert assistant.prompts == []
    assert window.runtime_state == RuntimeState.ERROR
    assert "лимит" in window._answer.toPlainText()
    window.close()


def test_empty_transcription_is_ignored_and_stt_can_restart(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)

    window._toggle_listening()
    controller.transcription_ready.emit(result(" \n "))
    assert controller.stop_requests == 0
    assert assistant.prompts == []

    controller.request_stop()
    controller.finish()
    assert window.runtime_state == RuntimeState.IDLE
    window._toggle_listening()
    assert controller.starts == 2
    assert window.runtime_state == RuntimeState.LISTENING
    controller.request_stop()
    controller.finish()
    window.close()


def test_stop_during_stt_starting_disables_restart_until_finished(app: QApplication) -> None:
    class StartingController(FakeSttController):
        def start(self) -> bool:
            self.starts += 1
            self.state = SttState.STARTING
            self.state_changed.emit(self.state.value)
            if self.starts > 1:
                self.state = SttState.LISTENING
                self.state_changed.emit(self.state.value)
            return True

    controller = StartingController()
    window = MainWindow(FakeAssistant(), controller)

    window._toggle_listening()
    assert window.runtime_state == RuntimeState.STARTING
    assert window._listen_button.isEnabled()
    assert window._listen_button.text() == "Остановить"

    window._toggle_listening()
    assert window.runtime_state == RuntimeState.STOPPING
    assert controller.stop_requests == 1
    assert not window._listen_button.isEnabled()
    window._toggle_listening()
    assert controller.starts == 1

    controller.finish()
    assert window.runtime_state == RuntimeState.IDLE
    window._toggle_listening()
    assert controller.starts == 2
    assert window.runtime_state == RuntimeState.LISTENING
    controller.request_stop()
    controller.finish()
    window.close()


def test_stt_and_llm_errors_are_safe_in_ui_and_logs(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    canary = "SYNTHETIC_PRIVATE_CANARY_8f21"
    controller = FakeSttController()
    window = MainWindow(FakeAssistant(failure=canary), controller)
    window._toggle_listening()
    controller.fail(canary)
    assert window.runtime_state == RuntimeState.ERROR
    assert canary not in window._answer.toPlainText()
    assert canary not in caplog.text

    window._prompt.setPlainText("synthetic prompt")
    window._send_prompt()
    wait_until(lambda: window._worker is None)
    assert canary not in window._answer.toPlainText()
    assert canary not in caplog.text
    assert "ASSISTANT_GENERATION_UNEXPECTED_ERROR" in caplog.text
    window.close()


def test_stt_error_discards_partial_transcript_without_logging_it(
    app: QApplication, caplog: pytest.LogCaptureFixture
) -> None:
    canary = "SYNTHETIC_PARTIAL_TRANSCRIPT_CANARY"
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window._toggle_listening()
    controller.transcription_ready.emit(result(canary))
    assert window._prompt.toPlainText() == canary

    controller.fail("synthetic decoder failure")

    assert assistant.prompts == []
    assert canary not in caplog.text
    assert canary not in window._answer.toPlainText()
    assert window._prompt.toPlainText() == ""
    assert window.runtime_state == RuntimeState.ERROR
    window.close()


def test_stt_error_does_not_allow_restart_until_controller_finished(app: QApplication) -> None:
    controller = FakeSttController()
    window = MainWindow(FakeAssistant(), controller)
    window._toggle_listening()
    controller.report_error("synthetic failure")

    assert window.runtime_state == RuntimeState.STOPPING
    assert not window._listen_button.isEnabled()
    window._toggle_listening()
    assert controller.starts == 1

    controller.finish_failure()
    assert window.runtime_state == RuntimeState.ERROR
    window._toggle_listening()
    assert controller.starts == 2
    assert window.runtime_state == RuntimeState.LISTENING
    controller.request_stop()
    controller.finish()
    window.close()


def test_worker_thread_transcription_is_delivered_to_ui_thread(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = ThreadTrackingWindow(assistant, controller)
    window._toggle_listening()

    emitter = SignalEmitter(
        controller.transcription_ready,
        result("threaded synthetic prompt"),
    )
    emitter.start()
    wait_until(lambda: window.transcription_slot_thread is not None)
    wait_until(emitter.isFinished)

    assert emitter.owner_thread != get_ident()
    assert window.transcription_slot_thread == get_ident()
    assert controller.stop_requests == 0
    controller.request_stop()
    assert controller.stop_requests == 1
    controller.finish()
    assert assistant.prompts == []
    assert window._prompt.toPlainText() == "threaded synthetic prompt"
    window._send_button.click()
    wait_until(lambda: window._worker is None)
    assert assistant.prompts == ["threaded synthetic prompt"]
    window.close()


def test_close_during_stt_defers_close_until_controller_finishes(app: QApplication) -> None:
    assistant = FakeAssistant()
    controller = FakeSttController()
    window = MainWindow(assistant, controller)
    window.show()
    window._toggle_listening()
    controller.transcription_ready.emit(result("partial before close"))

    assert window.close() is False
    assert window.isVisible()
    assert window._closing
    assert controller.stop_requests == 1
    assert controller.cancel_requests == 1
    controller.transcription_ready.emit(result("late after close"))

    controller.finish()
    wait_until(lambda: not window.isVisible())
    assert assistant.prompts == []


def test_cancel_during_llm_keeps_event_loop_responsive(app: QApplication) -> None:
    assistant = FakeAssistant(block=True)
    window = MainWindow(assistant)
    window._prompt.setPlainText("synthetic prompt")
    window._send_prompt()
    wait_until(lambda: assistant.entered.is_set())

    timer_fired: list[bool] = []
    QTimer.singleShot(0, lambda: timer_fired.append(True))
    window._cancel_generation()
    wait_until(lambda: timer_fired and window._worker is None)

    assert timer_fired == [True]
    assert assistant.cleaned.is_set()
    assert window.runtime_state == RuntimeState.IDLE
    window.close()


def test_close_during_llm_defers_close_until_worker_finishes(app: QApplication) -> None:
    assistant = FakeAssistant(block=True)
    window = MainWindow(assistant)
    window.show()
    window._prompt.setPlainText("synthetic close prompt")
    window._send_prompt()
    wait_until(lambda: assistant.entered.is_set())

    assert window.close() is False
    assert window.isVisible()
    assert window._closing
    wait_until(lambda: assistant.cleaned.is_set() and window._worker is None)
    wait_until(lambda: not window.isVisible())
