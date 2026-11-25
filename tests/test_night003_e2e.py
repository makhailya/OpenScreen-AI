"""Night-003 end-to-end tests with fake audio and local mock transports."""

from __future__ import annotations

import asyncio
import json
import os
import time
from array import array
from collections import deque
from threading import Event, get_ident

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import pytest
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from openscreen_ai import main as main_module
from openscreen_ai.config import AppConfig
from openscreen_ai.core.assistant import Assistant
from openscreen_ai.providers.ollama_provider import OllamaProvider
from openscreen_ai.stt.audio import AudioBlock
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttResult
from openscreen_ai.stt.worker import SttController, SttState
from openscreen_ai.ui.main_window import MainWindow, RuntimeState


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def wait_until(app: QApplication, predicate, timeout_ms: int = 5000) -> None:
    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTest.qWait(10)
        elapsed += 10
    assert predicate()


class FakeAudioSource:
    """Fake input yields finite blocks and never imports sounddevice."""

    def __init__(self, blocks: list[AudioBlock]) -> None:
        self._blocks = deque(blocks)
        self.dropped_blocks = 0
        self.input_overflows = 0
        self.queue_max_depth = len(blocks)
        self.owner_thread: int | None = None
        self.started = Event()
        self.capture_stopped = False
        self.closed = Event()

    @property
    def queue_depth(self) -> int:
        return len(self._blocks)

    def start(self) -> None:
        self.owner_thread = get_ident()
        self.started.set()

    def read(self) -> AudioBlock | None:
        assert self.owner_thread == get_ident()
        return self._blocks.popleft() if self._blocks else None

    def stop_capture(self) -> None:
        assert self.owner_thread == get_ident()
        self.capture_stopped = True

    def read_pending(self) -> AudioBlock | None:
        assert self.owner_thread == get_ident()
        assert self.capture_stopped
        return self._blocks.popleft() if self._blocks else None

    def stop(self) -> None:
        assert self.owner_thread == get_ident()
        self._blocks.clear()
        self.closed.set()


class FakeSttEngine:
    """Fake decoder returns controlled segments from the STT worker thread."""

    def __init__(self, segments: tuple[str, ...], tail: tuple[str, ...] = ()) -> None:
        self.owner_thread = get_ident()
        self.segments = segments
        self.tail = tail
        self.started = Event()
        self.processed = Event()
        self.closed = Event()
        self.finish_calls = 0

    def start(self) -> None:
        assert self.owner_thread == get_ident()
        self.started.set()

    def process_audio(self, _samples, *, captured_at: float) -> list[SttResult]:
        assert self.owner_thread == get_ident()
        self.processed.set()
        return [self._result(text, index, captured_at) for index, text in enumerate(self.segments)]

    def finish(self) -> list[SttResult]:
        assert self.owner_thread == get_ident()
        self.finish_calls += 1
        start = len(self.segments)
        return [self._result(text, start + offset, time.perf_counter())
                for offset, text in enumerate(self.tail)]

    @staticmethod
    def _result(text: str, index: int, captured_at: float) -> SttResult:
        return SttResult(
            text=text,
            endpoint_at=time.perf_counter(),
            endpoint_audio_at=captured_at,
            speech_started_at=captured_at,
            last_speech_active_at=captured_at,
            recognition_ms=0.1,
            cpu_ms=0.1,
            audio_seconds=0.04,
            segment_index=index,
        )

    def reset(self) -> None:
        assert self.owner_thread == get_ident()

    def close(self) -> None:
        assert self.owner_thread == get_ident()
        self.closed.set()


class TrackingWindow(MainWindow):
    def __init__(self, assistant, stt_controller=None) -> None:
        self.response_threads: list[int] = []
        super().__init__(assistant, stt_controller)

    def _append_text(self, request_id: str, text: str) -> None:
        self.response_threads.append(get_ident())
        super()._append_text(request_id, text)


def create_wired_window(monkeypatch, segments, tail, response_handler):
    blocks = [AudioBlock(array("f", [0.0]), time.perf_counter(), 0, 0)]
    sources: list[FakeAudioSource] = []
    engines: list[FakeSttEngine] = []
    controllers: list[SttController] = []
    request_payloads: list[dict[str, object]] = []
    request_threads: list[int] = []
    original_provider = OllamaProvider

    async def handle(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        request_threads.append(get_ident())
        return await response_handler(request)

    transport = httpx.MockTransport(handle)

    def provider_factory(config: AppConfig) -> OllamaProvider:
        return original_provider(config, transport=transport)

    def controller_factory(config: SttConfig) -> SttController:
        def microphone_factory() -> FakeAudioSource:
            source = FakeAudioSource(blocks)
            sources.append(source)
            return source

        def engine_factory() -> FakeSttEngine:
            engine = FakeSttEngine(tuple(segments), tuple(tail))
            engines.append(engine)
            return engine

        controller = SttController(
            config,
            microphone_factory=microphone_factory,
            engine_factory=engine_factory,
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(main_module, "OllamaProvider", provider_factory)
    monkeypatch.setattr(main_module, "MainWindow", TrackingWindow)
    # The model files are irrelevant here: the real controller receives fake
    # audio/engine dependencies, while composition, signals and UI stay real.
    monkeypatch.setattr(main_module, "check_stt_available", lambda _config: None)
    window = main_module.create_window(
        AppConfig(model="night003-test-model", timeout_seconds=3),
        stt_config=SttConfig(),
        stt_controller_factory=controller_factory,
    )
    return window, controllers, sources, engines, request_payloads, request_threads


def test_composition_root_stt_segments_reach_mock_ollama_and_ui_once(
    app: QApplication, monkeypatch
) -> None:
    async def success(request: httpx.Request) -> httpx.Response:
        body = (
            b'{"response":"synthetic ","done":false}\n'
            b'{"response":"answer","done":true}\n'
        )
        return httpx.Response(200, content=body, request=request)

    window, controllers, sources, engines, payloads, request_threads = create_wired_window(
        monkeypatch,
        ("first segment", "second segment", "third segment"),
        ("tail segment",),
        success,
    )
    ui_thread = get_ident()
    window.show()
    window._listen_button.click()
    controller = controllers[0]
    expected_before_stop = "first segment second segment third segment"
    wait_until(
        app,
        lambda: controller.state == SttState.LISTENING
        and window._prompt.toPlainText() == expected_before_stop,
    )

    window._listen_button.click()
    expected_prompt = f"{expected_before_stop} tail segment"
    wait_until(
        app,
        lambda: window.runtime_state == RuntimeState.IDLE
        and window._worker is None,
    )

    assert window._prompt.toPlainText() == expected_prompt
    assert window._send_button.isEnabled()
    assert payloads == []
    assert window._answer.toPlainText() == ""
    window._send_button.click()
    wait_until(
        app,
        lambda: window._answer.toPlainText() == "synthetic answer"
        and window._worker is None,
    )

    assert len(payloads) == 1
    assert payloads[0]["prompt"] == expected_prompt
    assert payloads[0]["model"] == "night003-test-model"
    assert request_threads[0] != ui_thread
    assert window.response_threads and set(window.response_threads) == {ui_thread}
    assert controller.state == SttState.STOPPED
    assert controller._worker is None
    assert controllers[0] not in SttController._active
    assert sources[0].closed.is_set()
    assert engines[0].closed.is_set()
    assert engines[0].finish_calls == 1
    assert window.close()


def test_composition_root_provider_failure_is_safe_and_cleans_workers(
    app: QApplication, monkeypatch, caplog
) -> None:
    body_canary = "SYNTHETIC_NIGHT003_HTTP_BODY_CANARY"

    async def failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body_canary.encode(), request=request)

    window, controllers, sources, engines, payloads, _request_threads = create_wired_window(
        monkeypatch, ("synthetic transcript",), (), failure
    )
    window.show()
    window._listen_button.click()
    controller = controllers[0]
    wait_until(app, lambda: window._prompt.toPlainText() == "synthetic transcript")
    window._listen_button.click()

    wait_until(app, lambda: window.runtime_state == RuntimeState.IDLE)
    assert window._prompt.toPlainText() == "synthetic transcript"
    assert payloads == []
    window._send_button.click()
    wait_until(app, lambda: window.runtime_state == RuntimeState.ERROR and window._worker is None)
    assert len(payloads) == 1
    assert "synthetic transcript" in payloads[0]["prompt"]
    assert body_canary not in window._answer.toPlainText()
    assert body_canary not in caplog.text
    assert window._error_message is not None
    assert controller._worker is None
    assert controller not in SttController._active
    assert sources[0].closed.is_set()
    assert engines[0].closed.is_set()
    assert window.close()


class QuietProvider:
    async def stream(self, _prompt: str, *, request_id: str | None = None):
        if False:
            yield ""


def create_repeated_controller(app: QApplication):
    sources: list[FakeAudioSource] = []
    engines: list[FakeSttEngine] = []

    def microphone_factory() -> FakeAudioSource:
        source = FakeAudioSource([])
        sources.append(source)
        return source

    def engine_factory() -> FakeSttEngine:
        engine = FakeSttEngine(())
        engines.append(engine)
        return engine

    return (
        SttController(
            SttConfig(),
            microphone_factory=microphone_factory,
            engine_factory=engine_factory,
        ),
        sources,
        engines,
    )


def test_thirty_fake_stt_start_stop_cycles_release_every_worker(app: QApplication) -> None:
    controller, sources, engines = create_repeated_controller(app)
    finished = QSignalSpy(controller.finished)

    for index in range(30):
        assert controller.start()
        wait_until(app, lambda: controller.state == SttState.LISTENING)
        assert controller.request_stop() is None
        wait_until(
            app,
            lambda index=index: finished.count() == index + 1
            and controller._worker is None,
        )
        assert controller.state == SttState.STOPPED
        assert controller not in SttController._active

    assert len(sources) == len(engines) == 30
    assert all(source.closed.is_set() for source in sources)
    assert all(engine.closed.is_set() for engine in engines)


def test_twenty_fake_stt_cancel_cycles_discard_and_release_every_worker(
    app: QApplication,
) -> None:
    controller, sources, engines = create_repeated_controller(app)
    finished = QSignalSpy(controller.finished)

    for index in range(20):
        assert controller.start()
        wait_until(app, lambda: controller.state == SttState.LISTENING)
        controller.cancel()
        wait_until(
            app,
            lambda index=index: finished.count() == index + 1
            and controller._worker is None,
        )
        assert controller.state == SttState.STOPPED
        assert controller not in SttController._active

    assert len(sources) == len(engines) == 20
    assert all(source.closed.is_set() for source in sources)
    assert all(engine.closed.is_set() for engine in engines)
    assert all(engine.finish_calls == 0 for engine in engines)


def test_ten_offscreen_gui_open_close_cycles_leave_no_visible_windows(
    app: QApplication,
) -> None:
    for _ in range(10):
        window = MainWindow(Assistant(QuietProvider()))
        window.show()
        app.processEvents()
        assert window.isVisible()
        assert window.close()
        app.processEvents()
        assert not window.isVisible()
    assert not [widget for widget in app.topLevelWidgets() if widget.isVisible()]


class RepeatedBlockingProvider:
    def __init__(self) -> None:
        self.started: list[Event] = []
        self.cleaned: list[Event] = []

    async def stream(self, _prompt: str, *, request_id: str | None = None):
        started = Event()
        cleaned = Event()
        self.started.append(started)
        self.cleaned.append(cleaned)
        started.set()
        try:
            yield "synthetic partial"
            await asyncio.Event().wait()
        finally:
            cleaned.set()


def test_ten_llm_cancel_restarts_close_streams_and_leave_no_worker(app: QApplication) -> None:
    provider = RepeatedBlockingProvider()
    window = MainWindow(Assistant(provider))
    window.show()

    for index in range(10):
        window._prompt.setPlainText(f"synthetic request {index}")
        window._send_button.click()
        wait_until(
            app,
            lambda index=index: len(provider.started) == index + 1
            and window._answer.toPlainText() == "synthetic partial",
        )
        worker = window._worker
        assert worker is not None
        finished = QSignalSpy(worker.finished)
        window._cancel_button.click()
        wait_until(
            app,
            lambda finished=finished: finished.count() == 1
            and window._worker is None,
        )
        assert provider.cleaned[index].is_set()
        assert window.runtime_state == RuntimeState.IDLE

    assert window.close()


def test_tail_overflow_without_next_block_blocks_send(app, monkeypatch):
    original_read = FakeAudioSource.read

    def overflow_after_last_block(source):
        item = original_read(source)
        if item is None:
            source.input_overflows = 1
        return item

    monkeypatch.setattr(FakeAudioSource, "read", overflow_after_last_block)

    async def unexpected_request(request):
        raise AssertionError("incomplete dictation must never reach the provider")

    window, controllers, sources, engines, payloads, _ = create_wired_window(
        monkeypatch, ("synthetic incomplete phrase",), (), unexpected_request
    )
    try:
        window._listen_button.click()
        wait_until(app, lambda: sources and sources[0].input_overflows == 1)
        # Loss without a delivered boundary is finalized after capture closes.
        controllers[0].request_stop()
        wait_until(app, lambda: window.runtime_state == RuntimeState.ERROR)
        assert controllers[0]._worker is None
        assert sources[0].closed.is_set()
        assert engines[0].closed.is_set()
        assert window._prompt.toPlainText() == ""
        assert payloads == []
        window._send_button.click()
        assert payloads == []
    finally:
        controllers[0].cancel()
        wait_until(app, lambda: controllers[0]._worker is None)
        window.close()
