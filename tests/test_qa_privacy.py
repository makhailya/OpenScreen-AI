"""Synthetic privacy contracts for diagnostics and microphone capture."""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest

from openscreen_ai.stt.audio import MicrophoneSource
from openscreen_ai.stt.worker import SttDiagnostic, _SttWorker


class FakeSamples(list):
    def copy(self) -> FakeSamples:
        return FakeSamples(self)

    def reshape(self, _size: int) -> FakeSamples:
        return self


class FakeInputStream:
    def __init__(self, **settings: object) -> None:
        self.settings = settings
        self.active = False

    def start(self) -> None:
        self.active = True

    def close(self, *, ignore_errors: bool) -> None:
        assert ignore_errors is False
        self.active = False

    def emit_block(self) -> None:
        self.settings["callback"](
            FakeSamples([0.125, -0.125]),
            2,
            SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.01),
            SimpleNamespace(input_overflow=False),
        )


def test_fake_capture_callback_creates_no_recording_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    streams: list[FakeInputStream] = []

    def make_stream(**settings: object) -> FakeInputStream:
        stream = FakeInputStream(**settings)
        streams.append(stream)
        return stream

    fake_device = SimpleNamespace(
        InputStream=make_stream,
        CallbackAbort=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_device)
    monkeypatch.chdir(tmp_path)
    source = MicrophoneSource()

    source.start()
    stream = streams[0]
    stream.emit_block()
    block = source.read()
    source.stop()

    assert block is not None
    assert list(tmp_path.iterdir()) == []


def test_diagnostic_has_only_non_content_measurements() -> None:
    canary = "QA_SYNTHETIC_PRIVATE_CANARY"
    diagnostic = SttDiagnostic(
        queue_depth_blocks=2,
        dropped_blocks=1,
        vad_resets_total=1,
        endpoint_at=123.0,
        recognition_ms=4.0,
    )
    names = {item.name for item in fields(SttDiagnostic)}
    content_field_names = {"prompt", "transcript", "text", "audio", "samples", "raw_audio"}

    assert names.isdisjoint(content_field_names)
    assert canary not in repr(diagnostic)
    assert all(
        value is None or isinstance(value, (int, float, bool))
        for value in asdict(diagnostic).values()
    )


def test_stt_error_diagnostic_log_does_not_include_exception_canary(caplog) -> None:
    canary = "QA_SYNTHETIC_PRIVATE_CANARY"
    worker = _SttWorker(lambda: None, lambda: None)

    with caplog.at_level(logging.ERROR, logger="openscreen_ai.stt.worker"):
        try:
            raise RuntimeError(canary)
        except RuntimeError as error:
            worker._report_error(error)

    assert canary not in caplog.text
