from collections import deque
from threading import Event, get_ident

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtTest import QSignalSpy, QTest

from openscreen_ai.stt.audio import AudioBlock
from openscreen_ai.stt.config import SttConfig
from openscreen_ai.stt.engine import SttResult
from openscreen_ai.stt.worker import SttController, SttState, _SttWorker


class FakeMicrophone:
    def __init__(self, blocks, calls, *, dropped_blocks, fail_read):
        self.blocks = deque(blocks)
        self.calls = calls
        self.dropped_blocks = dropped_blocks
        self.input_overflows = max((block.input_overflows for block in blocks), default=0)
        self.queue_max_depth = len(blocks)
        self.fail_read = fail_read
        self.empty = Event()
        self.owner = get_ident()
        self.capture_stopped = False

    @property
    def queue_depth(self):
        return len(self.blocks)

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

    def stop_capture(self):
        assert self.owner == get_ident()
        self.capture_stopped = True

    def read_pending(self):
        assert self.owner == get_ident()
        assert self.capture_stopped
        return self.blocks.popleft() if self.blocks else None

    def stop(self):
        assert self.owner == get_ident()
        self.calls.append("microphone.stop")


class FakeEngine:
    def __init__(self, calls, *, load_gate, decode_gate, fail_decode, result_texts):
        self.calls = calls
        self.load_gate = load_gate
        self.decode_gate = decode_gate
        self.fail_decode = fail_decode
        self.result_texts = result_texts
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
        return [
            SttResult(text, 3, 2, 1, None, 1, 1, 0.04)
            for text in self.result_texts
        ]

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
        result_texts=("тест",),
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
                result_texts=result_texts,
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
    # The in-flight block completes and the already queued block is drained
    # after capture stops; neither may be silently lost on normal Stop.
    assert results.count() == 2
    assert results.at(0)[0].text == "тест"
    assert results.at(1)[0].text == "тест"
    assert ("decode", 1, 1.5) in calls
    assert calls[-2:] == ["microphone.stop", "engine.close"]


def test_stop_during_result_delivery_does_not_drop_other_completed_segments():
    class OneBlockMicrophone:
        dropped_blocks = 0
        input_overflows = 0
        queue_depth = 0
        queue_max_depth = 1

        def __init__(self):
            self._blocks = deque([block(0)])

        def read(self):
            return self._blocks.popleft() if self._blocks else None

    class MultiSegmentEngine:
        def process_audio(self, _samples, *, captured_at):
            return [
                SttResult("первая часть", 3, 2, 1, None, 1, 1, 0.04),
                SttResult("вторая часть", 4, 3, 1, None, 1, 1, 0.04),
            ]

        def reset(self):
            raise AssertionError("unexpected VAD reset")

    microphone = OneBlockMicrophone()
    engine = MultiSegmentEngine()
    worker = _SttWorker(lambda: microphone, lambda: engine)
    received = []

    def stop_after_first(payload):
        received.append(payload[0].text)
        if len(received) == 1:
            worker.request_stop()

    worker.transcription_ready.connect(
        stop_after_first, Qt.ConnectionType.DirectConnection
    )
    worker._listen(microphone, engine)

    assert received == ["первая часть", "вторая часть"]


def test_controller_forwards_all_completed_segments_after_stop(session):
    controller, _calls, _microphones, _engines = session(
        [block(0)], result_texts=("первая часть", "вторая часть")
    )
    received = []
    finished = QSignalSpy(controller.finished)

    def stop_after_first(result):
        received.append(result)
        if len(received) == 1:
            controller.request_stop()

    controller.transcription_ready.connect(
        stop_after_first, Qt.ConnectionType.DirectConnection
    )
    assert controller.start()
    wait_until(lambda: finished.count() == 1)

    assert [result.text for result in received] == ["первая часть", "вторая часть"]
    assert [result.segment_index for result in received] == [None, None]
    assert [result.session_id for result in received] == [controller.session_id] * 2


def test_cancel_after_stop_request_suppresses_final_tail(session):
    controller, _calls, _microphones, _engines = session([])
    assert controller.start()
    wait_until(lambda: _microphones and _microphones[0].empty.is_set())
    worker = controller._worker

    controller.request_stop()
    controller.cancel()

    assert worker is not None
    assert worker._finalize_requested is False
    wait_until(lambda: controller._worker is None)


@pytest.mark.parametrize(
    ("cancel", "expected_texts"),
    [(False, ["completed", "tail"]), (True, ["completed"])],
    ids=["normal-stop-flushes-tail", "cancel-discards-tail"],
)
def test_worker_flushes_tail_only_for_normal_stop(cancel, expected_texts):
    class OneBlockMicrophone:
        dropped_blocks = 0
        input_overflows = 0
        queue_depth = 0
        queue_max_depth = 1

        def read(self):
            return block(0)

    class FlushEngine:
        finish_calls = 0

        def process_audio(self, _samples, *, captured_at):
            return [SttResult("completed", 3, 2, 1, None, 1, 1, 0.04)]

        def finish(self):
            self.finish_calls += 1
            return [SttResult("tail", 4, 3, 1, None, 1, 1, 0.04)]

    microphone = OneBlockMicrophone()
    engine = FlushEngine()
    worker = _SttWorker(lambda: microphone, lambda: engine, session_id=7)
    received = []

    def stop_after_completed(payload):
        received.append(payload[0])
        if len(received) == 1:
            if cancel:
                worker.request_cancel()
            else:
                worker.request_stop()

    worker.transcription_ready.connect(stop_after_completed, Qt.ConnectionType.DirectConnection)
    worker._listen(microphone, engine)

    assert [item.text for item in received] == expected_texts
    assert [item.session_id for item in received] == [7] * len(expected_texts)
    assert engine.finish_calls == (0 if cancel else 1)


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
        ([block(0), block(2, 1)], 1,
         [("decode", 0, 0.5), "reset", ("decode", 2, 2.5)]),
        (
            [block(0), block(1, 1), block(2, 1)],
            0,
            [("decode", 0, 0.5), "reset", ("decode", 1, 1.5), ("decode", 2, 2.5)],
        ),
    ],
    ids=["gap-before-last", "initial-gap", "tail-drop", "gap-and-overflow",
         "input-overflow"],
)
def test_reset_at_audio_discontinuities(session, blocks, dropped, expected):
    controller, calls, microphones, _ = session(blocks, dropped_blocks=dropped)
    diagnostics = QSignalSpy(controller.diagnostic_ready)
    assert controller.start()
    wait_until(lambda: microphones and microphones[0].empty.is_set())
    controller.request_stop()
    wait_until(lambda: controller.state == SttState.STOPPED)
    assert calls[2:-2] == expected
    expected_results = sum(
        item[0] == "decode" for item in expected if isinstance(item, tuple)
    )
    assert diagnostics.count() == expected_results + calls.count("reset")
    for i in range(diagnostics.count()):
        item = diagnostics.at(i)[0]
        assert item.worker_emit_at <= item.controller_received_at
        assert item.vad_resets_total <= calls.count("reset")
    reset_items = [
        diagnostics.at(i)[0] for i in range(diagnostics.count())
        if diagnostics.at(i)[0].endpoint_at is None
    ]
    assert len(reset_items) == calls.count("reset")
    assert reset_items[-1].vad_resets_total == calls.count("reset")
    sequences = [block.sequence for block in blocks]
    assert reset_items[-1].vad_resets_gap == (
        1 if sequences and sequences != list(range(len(sequences))) else 0
    )
    assert reset_items[-1].vad_resets_tail == (1 if expected[-1] == "reset" else 0)
    assert reset_items[-1].vad_resets_overflow == (
        1 if any(block.input_overflows for block in blocks) else 0
    )
    if blocks == [block(0), block(2, 1)]:
        assert reset_items[-1].vad_resets_gap == 1
        assert reset_items[-1].vad_resets_overflow == 1
        assert reset_items[-1].vad_resets_total == 1


def test_diagnostic_matches_transcription_and_forwards_queued_stop_result(session):
    controller, _, _, _ = session([block(0)])
    results = QSignalSpy(controller.transcription_ready)
    diagnostics = QSignalSpy(controller.diagnostic_ready)
    assert controller.start()
    wait_until(lambda: results.count() == diagnostics.count() == 1)
    item = diagnostics.at(0)[0]
    assert item.forwarded is True
    assert item.endpoint_at == results.at(0)[0].endpoint_at
    assert item.recognition_ms == results.at(0)[0].recognition_ms
    assert item.decode_completed_at is None
    controller.request_stop()
    controller._on_transcription((SttResult("late", 3, 2, 1, None, 1, 1, 0.04), item))
    assert results.count() == 2
    assert diagnostics.count() == 2
    assert results.at(1)[0].text == "late"
    assert diagnostics.at(1)[0].forwarded is True
    wait_until(lambda: controller._worker is None)


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


@pytest.mark.parametrize("stop_at_empty", [False, True], ids=["listening", "stop-flush"])
@pytest.mark.parametrize("tail_drops", [0, 1], ids=["overflow-only", "overflow-and-drop"])
def test_tail_input_overflow_resets_before_finish_without_next_block(stop_at_empty, tail_drops):
    calls = []

    class TailOverflowMicrophone:
        dropped_blocks = tail_drops
        input_overflows = 0
        queue_depth = 0
        queue_max_depth = 1

        def read(self):
            # A callback reports loss after the last delivered block. No later
            # AudioBlock exists to carry the updated overflow counter.
            self.input_overflows = 1
            if stop_at_empty:
                worker.request_stop()
            return None

        def stop_capture(self):
            calls.append("stop_capture")

        def read_pending(self):
            return None

    class TailEngine:
        def reset(self):
            calls.append("reset")

        def finish(self):
            calls.append("finish")
            return []

    microphone = TailOverflowMicrophone()
    engine = TailEngine()
    worker = _SttWorker(lambda: microphone, lambda: engine)
    diagnostics = []

    def on_reset(item):
        diagnostics.append(item)
        worker.request_stop()

    worker.reset_diagnostic_ready.connect(on_reset, Qt.ConnectionType.DirectConnection)
    # Bound a broken baseline without starting a Qt worker or real input.
    original_wait = worker._stop_requested.wait

    def bounded_wait(timeout):
        worker.request_stop()
        return original_wait(0)

    worker._stop_requested.wait = bounded_wait
    worker._listen(microphone, engine)

    assert calls.count("reset") == 1
    assert calls.index("reset") < calls.index("finish")
    assert len(diagnostics) == 1
    assert diagnostics[0].input_overflow_notifications == 1
    assert diagnostics[0].vad_resets_total == 1
    assert diagnostics[0].vad_resets_overflow == 1
    assert diagnostics[0].vad_resets_tail == (1 if tail_drops else 0)


def test_300_synthetic_start_stop_cycles_release_workers(session):
    controller, calls, microphones, engines = session([])
    active_before = set(SttController._active)
    for cycle in range(300):
        assert controller.start()
        worker = controller._worker
        wait_until(
            lambda cycle=cycle: len(microphones) == cycle + 1 and microphones[-1].empty.is_set()
        )
        if cycle % 2:
            controller.cancel()
        else:
            controller.request_stop()
        wait_until(lambda: controller._worker is None)
        wait_until(lambda worker=worker: not shiboken6.isValid(worker))
        assert SttController._active == active_before
        assert microphones[-1].queue_depth == 0
        assert microphones[-1].queue_max_depth == 0
        assert controller.state == SttState.STOPPED
    assert len(engines) == len(microphones) == 300
    assert calls.count("engine.close") == calls.count("microphone.stop") == 300


def test_empty_read_overflow_snapshot_ahead_of_queued_blocks_keeps_boundary():
    calls = []

    class InterleavedMicrophone:
        dropped_blocks = 0
        input_overflows = 0
        queue_depth = 0
        queue_max_depth = 2

        def __init__(self):
            self.blocks = deque()
            self.first_read = True

        def read(self):
            if self.first_read:
                self.first_read = False
                # Simulate callbacks after popleft observed an empty queue,
                # before the consumer reads the cumulative overflow counter.
                self.blocks.extend([block(0), block(1, 1)])
                self.input_overflows = 1
                return None
            return self.blocks.popleft() if self.blocks else None

        def stop_capture(self):
            pass

        def read_pending(self):
            return self.blocks.popleft() if self.blocks else None

    class Engine:
        def process_audio(self, samples, *, captured_at):
            calls.append(("block", samples[0]))
            if samples[0] == 1:
                worker.request_stop()
            return []

        def reset(self):
            calls.append("reset")

        def finish(self):
            calls.append("finish")
            return []

    source = InterleavedMicrophone()
    worker = _SttWorker(lambda: source, Engine)
    diagnostics = []
    worker.reset_diagnostic_ready.connect(
        diagnostics.append, Qt.ConnectionType.DirectConnection
    )
    worker._listen(source, Engine())
    assert calls == [("block", 0), "reset", ("block", 1), "finish"]
    assert [item.vad_resets_total for item in diagnostics] == [1]
    assert [item.vad_resets_overflow for item in diagnostics] == [1]
    assert all(item.input_overflow_notifications == 1 for item in diagnostics)
    assert all(item.vad_resets_tail == item.vad_resets_gap == 0 for item in diagnostics)


@pytest.mark.parametrize(
    "injection", ["empty-read", "counter-read", "between-polls", "before-read"]
)
@pytest.mark.parametrize("stop_before_boundary", [False, True])
@pytest.mark.parametrize("notifications", [1, 3])
def test_overflow_boundary_preserves_completed_pre_loss_speech(
    injection, stop_before_boundary, notifications
):
    """Serialized callback interleavings, no sleeps or physical capture."""
    calls = []
    segments = []
    diagnostics = []

    class Source:
        dropped_blocks = 0
        queue_max_depth = 3

        def __init__(self):
            self.blocks = deque([block(0)])
            self.overflows = 0
            self.injected = False

        @property
        def queue_depth(self):
            return len(self.blocks)

        def publish(self):
            self.injected = True
            self.blocks.extend([block(1), block(2, notifications), block(3, notifications + 1)])
            self.overflows = notifications + 1

        @property
        def input_overflows(self):
            if injection == "counter-read" and not self.blocks and not self.injected:
                # Old worker queries this after observing an empty queue.
                self.publish()
            return self.overflows

        def read(self):
            if not self.blocks and not self.injected:
                if injection in {"between-polls", "counter-read"}:
                    return None
                self.publish()
                if injection == "empty-read":
                    return None
            return self.blocks.popleft() if self.blocks else None

        def stop_capture(self):
            calls.append("stop_capture")

        def read_pending(self):
            return self.blocks.popleft() if self.blocks else None

    class Engine:
        def __init__(self):
            self.pending = []

        def process_audio(self, samples, *, captured_at):
            sample = samples[0]
            calls.append(("block", sample))
            self.pending.append(sample)
            if sample == 0 and injection == "before-read":
                source.publish()
            if sample == 1:
                segments.append(tuple(self.pending))
                self.pending.clear()
                if stop_before_boundary:
                    worker.request_stop()
            if sample == 3:
                worker.request_stop()
            return []

        def reset(self):
            calls.append(("reset", tuple(self.pending)))
            self.pending.clear()

        def finish(self):
            calls.append("finish")
            segments.append(tuple(self.pending))
            return []

    source = Source()
    engine = Engine()
    worker = _SttWorker(lambda: source, lambda: engine)
    worker.reset_diagnostic_ready.connect(diagnostics.append, Qt.ConnectionType.DirectConnection)

    def publish_between_polls(timeout):
        assert not diagnostics
        assert engine.pending == [0]
        source.publish()
        return False

    if injection in {"between-polls", "counter-read"}:
        worker._stop_requested.wait = publish_between_polls
    worker._listen(source, engine)
    audio_calls = [item for item in calls if item != "stop_capture"]
    assert audio_calls == [
        ("block", 0), ("block", 1), ("reset", ()),
        ("block", 2), ("reset", (2,)), ("block", 3), "finish",
    ]
    assert segments == [(0, 1), (3,)]
    assert [item.vad_resets_total for item in diagnostics] == [1, 2]
    assert [item.vad_resets_overflow for item in diagnostics] == [1, 2]
    assert all(item.input_overflow_notifications == notifications + 1 for item in diagnostics)
    assert all(item.vad_resets_gap == item.vad_resets_tail == 0 for item in diagnostics)
    assert source.queue_depth == 0
