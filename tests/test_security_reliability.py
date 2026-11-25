"""Security regression tests for diagnostic and worker lifecycle behavior."""

import logging

import pytest

from openscreen_ai.stt.worker import _SttWorker


def test_worker_logs_safe_category_and_preserves_error_signal(caplog) -> None:
    canary = "SYNTHETIC_STT_CANARY_7c31"
    worker = _SttWorker(lambda: None, lambda: None)
    emitted: list[str] = []
    worker.error_occurred.connect(emitted.append)

    with caplog.at_level(logging.ERROR, logger="openscreen_ai.stt.worker"):
        worker._report_error(RuntimeError(canary))

    assert worker.failed is True
    assert emitted == [canary]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "STT_WORKER_ERROR"
    assert record.exc_info is None
    assert canary not in caplog.text


@pytest.mark.parametrize("failure_point", ["engine.start", "microphone.start"])
def test_worker_startup_failure_closes_created_resources(failure_point) -> None:
    calls: list[str] = []

    class FakeEngine:
        def start(self) -> None:
            calls.append("engine.start")
            if failure_point == "engine.start":
                raise RuntimeError("synthetic startup error")

        def close(self) -> None:
            calls.append("engine.close")

    class FakeMicrophone:
        def start(self) -> None:
            calls.append("microphone.start")
            if failure_point == "microphone.start":
                raise RuntimeError("synthetic startup error")

        def stop(self) -> None:
            calls.append("microphone.stop")

    worker = _SttWorker(FakeMicrophone, FakeEngine)
    worker.run()

    assert worker.failed is True
    if failure_point == "engine.start":
        assert calls == ["engine.start", "engine.close"]
    else:
        assert calls == ["engine.start", "microphone.start", "microphone.stop", "engine.close"]
