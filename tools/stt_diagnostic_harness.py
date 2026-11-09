"""Детерминированная проверка очереди без микрофона, Qt и моделей.

Запуск: PYTHONPATH=src python tools/stt_diagnostic_harness.py.
"""

from types import SimpleNamespace

from openscreen_ai.stt.audio import MicrophoneSource


class Samples(list):
    def copy(self):
        return Samples(self)

    def reshape(self, _shape):
        return self


def main() -> None:
    source = MicrophoneSource(block_ms=100, buffer_ms=500)
    source._stream = SimpleNamespace(active=True)  # Только fake: start() не вызываем.
    timing = SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.1)
    for sequence in range(7):
        source._enqueue(
            Samples([sequence]), 1600, timing,
            SimpleNamespace(input_overflow=sequence == 5),
        )
    assert (source.queue_depth, source.queue_max_depth) == (5, 5)
    assert (source.dropped_blocks, source.input_overflows) == (2, 1)
    assert [source.read().sequence for _ in range(5)] == list(range(5))
    assert source.queue_depth == 0
    print("queue: 5/5 blocks; dropped: 2 blocks; overflow: 1 notification")


if __name__ == "__main__":
    main()
