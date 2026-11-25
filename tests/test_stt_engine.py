"""Unit-тесты движка STT на поддельных распознавателе и VAD."""

from __future__ import annotations

from array import array
from collections import deque
from types import SimpleNamespace

import pytest

from openscreen_ai.stt import SttConfig
from openscreen_ai.stt.engine import SttEngine
from openscreen_ai.stt.errors import SttDecodeError, SttLifecycleError


class FakeVad:
    def __init__(self) -> None:
        self._segments: deque[SimpleNamespace] = deque()
        self._speech_samples: list[float] = []
        self._speech_active = False
        self._position = 0
        self._speech_start = 0
        self.silence_windows = 1
        self._silence = 0
        self.reset_calls = 0
        self.close_calls = 0

    def accept_waveform(self, samples: array) -> None:
        values = list(samples)
        position = self._position
        self._position += len(values)
        if any(value != 0 for value in values):
            if not self._speech_samples:
                self._speech_start = position
            self._silence = 0
            self._speech_samples.extend(values)
            self._speech_active = True
        elif self._speech_samples:
            self._silence += 1
            if self._silence < self.silence_windows:
                return
            self._segments.append(
                SimpleNamespace(
                    start=self._speech_start, samples=array("f", self._speech_samples)
                )
            )
            self._speech_samples.clear()
            self._speech_active = False

    def is_speech_detected(self) -> bool:
        return self._speech_active

    def empty(self) -> bool:
        return not self._segments

    @property
    def front(self) -> SimpleNamespace:
        return self._segments[0]

    def pop(self) -> None:
        self._segments.popleft()

    def reset(self) -> None:
        self.reset_calls += 1
        self._position = self._silence = 0
        self._segments.clear()
        self._speech_samples.clear()
        self._speech_active = False

    def flush(self) -> None:
        if self._speech_samples:
            self._segments.append(
                SimpleNamespace(
                    start=self._speech_start, samples=array("f", self._speech_samples)
                )
            )
            self._speech_samples.clear()
            self._speech_active = False

    def close(self) -> None:
        self.close_calls += 1


class FakeStream:
    def __init__(self) -> None:
        self.samples: array | None = None
        self.sample_rate: int | None = None
        self.result = SimpleNamespace(text="")

    def accept_waveform(self, sample_rate: int, samples: array) -> None:
        self.sample_rate = sample_rate
        self.samples = samples


class FakeRecognizer:
    def __init__(
        self,
        texts: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._texts = deque(texts or [])
        self._error = error
        self.decode_calls = 0
        self.close_calls = 0

    def create_stream(self) -> FakeStream:
        return FakeStream()

    def decode_stream(self, stream: FakeStream) -> None:
        self.decode_calls += 1
        if self._error is not None:
            raise self._error
        stream.result.text = self._texts.popleft() if self._texts else ""

    def close(self) -> None:
        self.close_calls += 1


def make_engine(
    *,
    texts: list[str] | None = None,
    error: Exception | None = None,
) -> tuple[SttEngine, FakeRecognizer, FakeVad, dict[str, int]]:
    recognizer = FakeRecognizer(texts, error)
    vad = FakeVad()
    loads = {"recognizer": 0, "vad": 0}

    def create_recognizer(_config: SttConfig) -> FakeRecognizer:
        loads["recognizer"] += 1
        return recognizer

    def create_vad(_config: SttConfig) -> tuple[FakeVad, int]:
        loads["vad"] += 1
        return vad, 2

    engine = SttEngine(
        SttConfig(),
        recognizer_factory=create_recognizer,
        vad_factory=create_vad,
        availability_checker=lambda _config: None,
    )
    return engine, recognizer, vad, loads


def test_start_loads_recognizer_and_vad_only_once() -> None:
    engine, _recognizer, _vad, loads = make_engine()

    engine.start()
    engine.start()

    assert loads == {"recognizer": 1, "vad": 1}


def test_recognizer_loader_uses_existing_sherpa_int8_transducer_contract(
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class FakeOfflineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            captured.update(kwargs)
            return object()

    config = SttConfig(
        model_dir=tmp_path / "ultra-int8",
        num_threads=3,
        provider="cpu",
    )
    recognizer = SttEngine._create_recognizer(
        SimpleNamespace(OfflineRecognizer=FakeOfflineRecognizer), config
    )

    assert recognizer is not None
    assert captured == {
        "encoder": str(tmp_path / "ultra-int8" / "encoder.int8.onnx"),
        "decoder": str(tmp_path / "ultra-int8" / "decoder.int8.onnx"),
        "joiner": str(tmp_path / "ultra-int8" / "joiner.int8.onnx"),
        "tokens": str(tmp_path / "ultra-int8" / "tokens.txt"),
        "model_type": "nemo_transducer",
        "num_threads": 3,
        "sample_rate": 16_000,
        "feature_dim": 80,
        "decoding_method": "greedy_search",
        "provider": "cpu",
    }


def test_recognizes_multiple_phrases_with_one_loaded_model() -> None:
    engine, recognizer, _vad, loads = make_engine(
        texts=["Первая фраза", "Вторая фраза", "Третья фраза"]
    )
    engine.start()

    results = engine.process_audio(
        array("f", [1, 1, 0, 0, 1, 1, 0, 0]),
        captured_at=100.0,
    )

    assert [result.text for result in results] == ["Первая фраза", "Вторая фраза"]
    assert [result.segment_index for result in results] == [0, 1]
    assert recognizer.decode_calls == 2
    assert loads == {"recognizer": 1, "vad": 1}
    assert all(result.audio_seconds == 2 / 16_000 for result in results)
    assert all(result.recognition_ms >= 0 for result in results)

    engine.reset()
    [after_reset] = engine.process_audio(
        array("f", [1, 1, 0, 0]), captured_at=200.0
    )
    assert after_reset.segment_index == 2


def test_decode_completion_is_after_decode_and_recognition_times_only_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, recognizer, _vad, _loads = make_engine(texts=["Фраза"])
    clock = iter((10.0, 11.0, 11.25, 11.5))
    monkeypatch.setattr("openscreen_ai.stt.engine.time.perf_counter", lambda: next(clock))
    engine.start()
    [result] = engine.process_audio(array("f", [1, 1, 0, 0]), captured_at=9.0)
    assert recognizer.decode_calls == 1
    assert result.endpoint_at == 10.0
    assert result.recognition_ms == pytest.approx(250.0)
    assert result.decode_completed_at == 11.5
    assert result.endpoint_at <= result.decode_completed_at


def test_empty_decoding_result_is_ignored() -> None:
    engine, recognizer, _vad, _loads = make_engine(texts=[""])
    engine.start()

    results = engine.process_audio(array("f", [1, 1, 0, 0]))

    assert results == []
    assert recognizer.decode_calls == 1


def test_finish_flushes_unterminated_segment_and_is_idempotent() -> None:
    engine, recognizer, _vad, _loads = make_engine(texts=["Последняя фраза"])
    engine.start()
    assert engine.process_audio(array("f", [1, 1]), captured_at=30.0) == []

    [result] = engine.finish()

    assert result.text == "Последняя фраза"
    assert result.segment_index == 0
    assert recognizer.decode_calls == 1
    assert engine.finish() == []
    with pytest.raises(SttLifecycleError, match="Завершённый"):
        engine.process_audio([0, 0])


def test_finish_pads_only_partial_vad_window() -> None:
    engine, recognizer, _vad, _loads = make_engine(texts=["Частичный блок"])
    engine.start()
    assert engine.process_audio(array("f", [1]), captured_at=40.0) == []

    [result] = engine.finish()

    assert result.text == "Частичный блок"
    assert recognizer.decode_calls == 1


def test_decode_failure_has_clear_exception() -> None:
    engine, _recognizer, _vad, _loads = make_engine(error=RuntimeError("native failure"))
    engine.start()

    with pytest.raises(SttDecodeError, match="Не удалось декодировать.*native failure"):
        engine.process_audio(array("f", [1, 1, 0, 0]))


def test_reset_clears_partial_audio_and_vad_state() -> None:
    engine, recognizer, vad, _loads = make_engine()
    engine.start()
    engine.process_audio(array("f", [1]))

    engine.reset()

    assert vad.reset_calls == 1
    assert engine.process_audio(array("f", [0, 0])) == []
    assert engine.process_audio(array("f", [0, 0])) == []
    assert recognizer.decode_calls == 0


def test_close_releases_resources_and_is_idempotent() -> None:
    engine, recognizer, vad, _loads = make_engine()
    engine.start()

    engine.close()
    engine.close()

    assert vad.reset_calls == 1
    assert vad.close_calls == 1
    assert recognizer.close_calls == 1
    with pytest.raises(SttLifecycleError, match="[Зз]акрытый"):
        engine.start()



def test_metrics_use_segment_boundary_even_while_vad_reports_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, vad, _ = make_engine(texts=["Фраза", "После сброса"])
    vad.silence_windows = 3
    monkeypatch.setattr("openscreen_ai.stt.engine.time.perf_counter", lambda: 200.0)
    engine.start()

    # Блоки пересекают границу окна. VAD держит флаг речи ещё два окна.
    assert engine.process_audio([1], captured_at=100.0) == []
    assert engine.process_audio([1, 0, 0, 0, 0], captured_at=100 + 1 / 16_000) == []
    assert vad.is_speech_detected()
    [result] = engine.process_audio([0, 0], captured_at=100 + 6 / 16_000)

    assert result.last_speech_active_at == pytest.approx(100 + 2 / 16_000, abs=1e-10)
    assert result.endpoint_audio_at == pytest.approx(100 + 8 / 16_000, abs=1e-10)
    assert result.endpoint_at == 200.0
    assert result.vad_window_ms == 0.125

    engine.reset()
    [result] = engine.process_audio([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], captured_at=120)
    assert result.last_speech_active_at == pytest.approx(120 + 4 / 16_000, abs=1e-10)


def test_metrics_leave_unknown_segment_boundary_unset() -> None:
    engine, _, vad, _ = make_engine(texts=["Фраза"])
    engine.start()
    vad._segments.append(SimpleNamespace(samples=array("f", [1, 1])))

    [result] = engine.process_audio([0, 0], captured_at=100)

    assert result.last_speech_active_at is None
