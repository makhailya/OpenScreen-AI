"""Deterministic regression tests for NIGHT-001 runtime lifecycle races."""

from openscreen_ai.ui.main_window import GenerationWorker


def test_cancel_tolerates_loop_closing_after_running_check() -> None:
    class ClosingLoop:
        def is_running(self) -> bool:
            return True

        def call_soon_threadsafe(self, _callback) -> None:
            raise RuntimeError("Event loop is closed")

    class FakeTask:
        def cancel(self) -> None:
            raise AssertionError("closed-loop callback must not run")

    worker = GenerationWorker(object(), "synthetic prompt")  # type: ignore[arg-type]
    worker._loop = ClosingLoop()  # type: ignore[assignment]
    worker._task = FakeTask()  # type: ignore[assignment]

    worker.cancel()

    assert worker._cancel_requested is True
