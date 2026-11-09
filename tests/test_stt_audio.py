"""Тесты аудиоисточника без PortAudio, микрофона и моделей."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openscreen_ai.stt.audio import AudioCaptureError, MicrophoneSource
from openscreen_ai.stt.engine import SttEngine
from openscreen_ai.stt.errors import SttDependencyError, SttLifecycleError


class CallbackAbort(Exception):
    pass


class Samples(list):
    def copy(self):
        return Samples(self)

    def reshape(self, size):
        assert size == -1
        return self


class FakeStream:
    def __init__(self, **settings):
        self.settings = settings
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def close(self, *, ignore_errors):
        assert not ignore_errors
        self.active = False
        self.closed = True

    def emit(self, samples=None, *, overflow=False):
        if samples is None:
            samples = Samples([0.25])
        self.settings["callback"](
            samples,
            self.settings["blocksize"],
            SimpleNamespace(inputBufferAdcTime=5.0, currentTime=5.07),
            SimpleNamespace(input_overflow=overflow),
        )


@pytest.fixture
def device(monkeypatch):
    sounddevice = SimpleNamespace(
        InputStream=Mock(side_effect=FakeStream), CallbackAbort=CallbackAbort
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    return sounddevice


def test_stream_configuration_and_audio_copy(device, monkeypatch):
    monkeypatch.setattr("openscreen_ai.stt.audio.time.perf_counter", lambda: 50.0)
    source = MicrophoneSource(device="USB Microphone", block_ms=20)
    source.start()
    settings = device.InputStream.call_args.kwargs.copy()
    settings.pop("callback")
    assert settings == {
        "device": "USB Microphone",
        "samplerate": 16_000,
        "channels": 1,
        "dtype": "float32",
        "blocksize": 320,
    }
    samples = Samples([0.25])
    source._stream.emit(samples)
    samples[0] = 1.0

    block = source.read()
    assert block.samples == [0.25]
    assert block.captured_at == pytest.approx(49.93)
    assert block.sequence == block.input_overflows == 0
    assert source.read() is None
    source.stop()


def test_overflow_returns_without_consumer_and_exposes_sequence_gap(device):
    source = MicrophoneSource(block_ms=100, buffer_ms=500)
    source.start()
    stream = source._stream
    # Без чтения и без копирования входного блока при полной очереди.
    for _ in range(5):
        stream.emit()
    stream.emit(object(), overflow=True)
    stream.emit(object())
    assert source.dropped_blocks == 2
    assert source.input_overflows == 1
    assert [source.read().sequence for _ in range(5)] == list(range(5))
    assert source.read() is None

    stream.emit()
    block = source.read()
    assert block.sequence == 7
    assert block.input_overflows == 1
    source.stop()


def test_restart_closes_old_stream_and_discards_queued_blocks(device):
    source = MicrophoneSource()
    source.start()
    first = source._stream
    first.emit()
    source.start()
    assert device.InputStream.call_count == 1
    source.stop()
    source.stop()
    assert first.closed
    with pytest.raises(SttLifecycleError):
        source.read()

    source.start()
    second = source._stream
    assert second is not first
    assert source.read() is None
    second.emit()
    assert source.read().sequence == 0
    source.stop()
    assert second.closed


def test_open_error_is_reported_without_device_fallback(device):
    error = OSError("device unavailable")
    device.InputStream.side_effect = error
    source = MicrophoneSource(device=7)
    with pytest.raises(AudioCaptureError, match="device unavailable") as failure:
        source.start()
    assert failure.value.__cause__ is error
    assert device.InputStream.call_count == 1
    assert device.InputStream.call_args.kwargs["device"] == 7
    source.stop()


def test_start_failure_closes_open_stream(device):
    stream = FakeStream()
    stream.start = Mock(side_effect=OSError("disconnected on start"))
    device.InputStream.side_effect = None
    device.InputStream.return_value = stream
    source = MicrophoneSource()
    with pytest.raises(AudioCaptureError, match="disconnected on start"):
        source.start()
    assert stream.closed


def test_unexpected_inactive_stream_is_reported_and_closed(device):
    source = MicrophoneSource(device=7)
    source.start()
    stream = source._stream
    stream.active = False
    with pytest.raises(AudioCaptureError, match="неожиданно остановился"):
        source.read()
    source.stop()
    assert stream.closed
    assert device.InputStream.call_count == 1


def test_callback_failure_aborts_and_reaches_consumer(device):
    source = MicrophoneSource()
    source.start()
    stream = source._stream
    with pytest.raises(CallbackAbort):
        stream.emit(object())
    with pytest.raises(AudioCaptureError, match="callback") as failure:
        source.read()
    assert isinstance(failure.value.__cause__, AttributeError)
    source.stop()
    assert stream.closed
    source.start()
    assert source.read() is None
    source.stop()


def test_callback_never_calls_engine(device, monkeypatch):
    methods = ("start", "process_audio", "reset", "close")
    for method in methods:
        monkeypatch.setattr(SttEngine, method, Mock(side_effect=AssertionError(method)))
    source = MicrophoneSource()
    source.start()
    source._stream.emit()
    assert source.read().sequence == 0
    for method in methods:
        getattr(SttEngine, method).assert_not_called()
    source.stop()


def test_missing_sounddevice_is_reported_only_on_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    source = MicrophoneSource()
    with pytest.raises(SttDependencyError, match="sounddevice"):
        source.start()


def test_portaudio_overflow_marks_discontinuity_without_skipping_sequence(device):
    source = MicrophoneSource()
    source.start()
    stream = source._stream
    stream.emit()
    stream.emit(overflow=True)
    stream.emit()
    blocks = [source.read() for _ in range(3)]
    assert [block.sequence for block in blocks] == [0, 1, 2]
    assert [block.input_overflows for block in blocks] == [0, 1, 1]
    assert source.dropped_blocks == 0
    source.stop()


def test_close_error_reaches_caller(device):
    source = MicrophoneSource()
    source.start()
    source._stream.close = Mock(side_effect=OSError("native close failure"))
    with pytest.raises(AudioCaptureError, match="native close failure"):
        source.stop()
