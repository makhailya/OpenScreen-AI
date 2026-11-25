"""Additional fake-only STT edge cases not covered by the baseline suite."""

from array import array

import pytest
from test_stt_engine import make_engine

from openscreen_ai.stt.errors import SttError


def test_short_audio_waits_for_window_then_decodes_at_vad_endpoint() -> None:
    engine, recognizer, _vad, _loads = make_engine(texts=["short phrase"])
    engine.start()

    assert engine.process_audio(array("f", [1]), captured_at=10.0) == []
    assert recognizer.decode_calls == 0
    results = engine.process_audio(array("f", [1, 0, 0]), captured_at=10 + 1 / 16_000)

    # The second call completes one speech window and one silence window. With
    # FakeVad's default silence_windows=1, that is a complete endpoint.
    assert [result.text for result in results] == ["short phrase"]
    assert recognizer.decode_calls == 1


def test_vad_keeps_speech_together_across_a_short_fake_pause() -> None:
    engine, recognizer, vad, _loads = make_engine(texts=["one phrase"])
    vad.silence_windows = 2
    engine.start()

    assert engine.process_audio([1, 1], captured_at=20.0) == []
    assert engine.process_audio([0, 0], captured_at=20 + 2 / 16_000) == []
    assert engine.process_audio([1, 1], captured_at=20 + 4 / 16_000) == []
    assert engine.process_audio([0, 0], captured_at=20 + 6 / 16_000) == []
    results = engine.process_audio([0, 0], captured_at=20 + 8 / 16_000)

    assert [result.text for result in results] == ["one phrase"]
    assert recognizer.decode_calls == 1
    assert results[0].audio_seconds == pytest.approx(4 / 16_000)


def test_vad_failure_is_wrapped_as_stt_error() -> None:
    engine, _recognizer, vad, _loads = make_engine()
    engine.start()
    original = vad.accept_waveform

    def fail(_samples: array) -> None:
        raise RuntimeError("fake VAD failure")

    vad.accept_waveform = fail
    with pytest.raises(SttError, match="Ошибка обработки аудио через Silero VAD") as error:
        engine.process_audio([1, 1])
    assert isinstance(error.value.__cause__, RuntimeError)
    vad.accept_waveform = original
    engine.close()


def test_close_attempts_all_resource_cleanup_after_failure() -> None:
    engine, recognizer, vad, _loads = make_engine()
    engine.start()

    def fail_close() -> None:
        raise RuntimeError("fake native close failure")

    vad.close = fail_close
    with pytest.raises(SttError, match="fake native close failure"):
        engine.close()

    assert recognizer.close_calls == 1
    assert engine._closed is True
    # close is idempotent even when the first cleanup attempt reports an error.
    engine.close()
    assert recognizer.close_calls == 1
