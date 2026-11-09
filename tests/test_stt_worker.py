from collections import deque
from threading import Event, get_ident

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtTest import QSignalSpy, QTest

from openscreen_ai.stt.audio import AudioBlock
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttResult
from openscreen_ai.stt.worker import SttController, SttState


class FakeMicrophone:
    def __init__(self, blocks, calls, *, dropped_blocks, fail_read):
        self.blocks = deque(blocks)
        self.calls = calls
        self.dropped_blocks = dropped_blocks
        self.fail_read = fail_read
        self.empty = Event()
        self.owner = get_ident()

    def start(self):
        assert self.owner == get_ident()
        self.calls.append("microphone.start")

    def read(self):
        assert self.owner == get_ident()
        if self.fail_read:
            raise RuntimeError("microphone read failed")
        if self.blocks:
            return self.blocks.popleft()
        self.empty.set()
        return None

    def stop(self):
        assert self.owner == get_ident()
        self.calls.append("microphone.stop")


class FakeEngine:
    def __init__(self, calls, *, load_gate, decode_gate, fail_decode):
        self.calls = calls
        self.load_gate = load_gate
        self.decode_gate = decode_gate
        self.fail_decode = fail_decode
        self.loading = Event()
        self.decoding = Event()
        self.owner = get_ident()

    def start(self):
        assert self.owner == get_ident()
        self.calls.append("engine.start")
        self.loading.set()
        if self.load_gate is not None:
            assert self.load_gate.wait(5)

    def process_audio(self, samples, *, captured_at):
        assert self.owner == get_ident()
        self.calls.append(("decode", samples[0], captured_at))
        self.decoding.set()
        if self.decode_gate is not None:
            assert self.decode_gate.wait(5)
        if self.fail_decode:
            raise RuntimeError("decode failed")
        return [SttResult("тест", 3, 2, 1, None, 1, 1, 0.04)]

    def reset(self):
        assert self.owner == get_ident()
        self.calls.append("reset")

    def close(self):
        assert self.owner == get_ident()
        self.calls.append("engine.close")


@pytest.fixture(scope="module")
def qt_app():
    return QCoreApplication.instance() or QCoreApplication([])


def wait_until(predicate):
    for _ in range(300):
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def session(qt_app):
    controllers = []
    gates = []

    def create(
        blocks=(),
        *,
        dropped_blocks=0,
        fail_read=False,
        fail_decode=False,
        load_gate=None,
        decode_gate=None,
    ):
        calls = []
        microphones = []
        engines = []
        gates.extend(gate for gate in (load_gate, decode_gate) if gate is not None)

        def microphone_factory():
            microphone = FakeMicrophone(
                blocks,
                calls,
                dropped_blocks=dropped_blocks,
                fail_read=fail_read,
            )
            microphones.append(microphone)
            return microphone

        def engine_factory():
            engine = FakeEngine(
                calls,
                load_gate=load_gate,
                decode_gate=decode_gate,
                fail_decode=fail_decode,
            )
            engines.append(engine)
            return engine

        controller = SttController(
            SttConfig(),
            microphone_factory=microphone_factory,
            engine_factory=engine_factory,
        )
        controllers.append(controller)
        return controller, calls, microphones, engines

    yield create

    for controller in controllers:
        controller.request_stop()
    for gate in gates:
        gate.set()
    for controller in controllers:
        wait_until(lambda controller=controller: controller._worker is None)


def block(sequence, overflows=0):
    return AudioBlock([sequence], sequence + 0.5, sequence, overflows)


def test_async_start_results_idle_stop_and_restart(session):
    gate = Event()
    controller, calls, microphones, engines = session([block(0)], load_gate=gate)
    states = QSignalSpy(controller.state_changed)
    results = QSignalSpy(controller.transcription_ready)
    finished = QSignalSpy(controller.finished)
    assert controller.state == SttState.IDLE
    assert controller.start()
    first_worker = controller._worker
    assert not controller.start()
    wait_until(lambda: engines and engines[0].loading.is_set())
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    wait_until(lambda: ticks)
    assert controller.state == SttState.STARTING
    assert engines[0].owner != get_ident()

    gate.set()
    wait_until(lambda: results.count() == 1 and microphones[0].empty.is_set())
    assert results.at(0)[0].text == "тест"
    assert microphones[0].owner == engines[0].owner
    controller.request_stop()
    assert not controller.start()
    wait_until(lambda: finished.count() == 1)
    wait_until(lambda: not shiboken6.isValid(first_worker))
    assert [states.at(i)[0] for i in range(states.count())] == [
        "starting",
        "listening",
        "stopping",
        "stopped",
    ]
    assert calls == [
        "engine.start",
        "microphone.start",
        ("decode", 0, 0.5),
        "microphone.stop",
        "engine.close",
    ]

    assert controller.start()
    second_worker = controller._worker
    wait_until(lambda: results.count() == 2)
    controller.request_stop()
    wait_until(lambda: finished.count() == 2)
    wait_until(lambda: not shiboken6.isValid(second_worker))
    assert len(engines) == len(microphones) == 2


def test_stop_during_decode_waits_for_safe_point(session):
    gate = Event()
    controller, calls, _, engines = session([block(0), block(1)], decode_gate=gate)
    results = QSignalSpy(controller.transcription_ready)
    finished = QSignalSpy(controller.finished)
    assert controller.start()
    wait_until(lambda: engines and engines[0].decoding.is_set())

    controller.request_stop()
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    wait_until(lambda: ticks)
    assert controller.state == SttState.STOPPING
    assert finished.count() == 0
    assert "microphone.stop" not in calls
    assert not controller.start()

    gate.set()
    wait_until(lambda: finished.count() == 1)
    assert controller.state == SttState.STOPPED
    assert results.count() == 0
    assert ("decode", 1, 1.5) not in calls
    assert calls[-2:] == ["microphone.stop", "engine.close"]


def test_stop_during_model_load_does_not_open_microphone(session):
    gate = Event()
    controller, calls, microphones, engines = session(load_gate=gate)
    assert controller.start()
    wait_until(lambda: engines and engines[0].loading.is_set())
    controller.request_stop()
    gate.set()
    wait_until(lambda: controller.state == SttState.STOPPED)
    assert not microphones
    assert calls == ["engine.start", "engine.close"]


@pytest.mark.parametrize(
    ("blocks", "dropped", "expected"),
    [
        ([block(0), block(2)], 1, [("decode", 0, 0.5), "reset", ("decode", 2, 2.5)]),
        ([block(2)], 2, ["reset", ("decode", 2, 2.5)]),
        ([block(0)], 1, [("decode", 0, 0.5), "reset"]),
        (
            [block(0), block(1, 1), block(2, 1)],
            0,
            [("decode", 0, 0.5), "reset", ("decode", 1, 1.5), ("decode", 2, 2.5)],
        ),
    ],
    ids=["gap-before-last", "initial-gap", "tail-drop", "input-overflow"],
)
def test_reset_at_audio_discontinuities(session, blocks, dropped, expected):
    controller, calls, _, _ = session(blocks, dropped_blocks=dropped)
    assert controller.start()
    wait_until(lambda: len(calls) >= len(expected) + 2)
    controller.request_stop()
    wait_until(lambda: controller.state == SttState.STOPPED)
    assert calls[2:-2] == expected


@pytest.mark.parametrize(
    ("failure", "message"),
    [("fail_read", "microphone read failed"), ("fail_decode", "decode failed")],
)
def test_errors_signal_and_release_resources(session, failure, message):
    controller, calls, _, _ = session([block(0)], **{failure: True})
    errors = QSignalSpy(controller.error_occurred)
    finished = QSignalSpy(controller.finished)
    assert controller.start()
    worker = controller._worker
    wait_until(lambda: finished.count() == 1)
    assert controller.state == SttState.ERROR
    assert errors.count() == 1
    assert errors.at(0)[0] == message
    assert calls[-2:] == ["microphone.stop", "engine.close"]
    wait_until(lambda: not shiboken6.isValid(worker))
