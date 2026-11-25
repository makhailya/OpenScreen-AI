from __future__ import annotations

import csv
import json
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from openscreen_ai.stt.engine import SttResult
from openscreen_ai.stt.worker import SttController

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import stt_benchmark as benchmark  # noqa: E402
from stt_real_adapter import (  # noqa: E402
    MAX_CHUNK_SAMPLES,
    REPLAY_BLOCK_SAMPLES,
    Pcm16Reader,
    RawPcmMetadata,
    ReplayError,
    ReplaySource,
    build_timing_metrics,
    fingerprint_file,
    fingerprint_model_directory,
    sample_index_to_perf,
    validate_labels,
)


def write_wav(path: Path, samples: bytes, *, rate: int = 16_000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples)


def _hanging_spawn_child(_result_pipe: object) -> None:
    time.sleep(30)


def test_stream_valid_wav_in_bounded_float32_chunks(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.wav"
    write_wav(path, struct.pack("<5000h", *range(-2500, 2500)))
    with Pcm16Reader(path) as reader:
        assert reader.metadata.sample_count == 5000
        assert len(reader.metadata.sha256) == 64
        first = reader.read_samples()
        second = reader.read_samples()
        assert len(first) == MAX_CHUNK_SAMPLES
        assert len(second) == 904
        assert first.typecode == "f"
        assert first[0] == pytest.approx(-2500 / 32768)
        assert reader.read_samples() is None


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2)])
def test_reject_wav_rate_and_channel_mismatch(tmp_path: Path, rate: int, channels: int) -> None:
    path = tmp_path / "wrong.wav"
    write_wav(path, b"\0\0" * channels, rate=rate, channels=channels)
    with pytest.raises(ReplayError):
        Pcm16Reader(path)


def test_reject_truncated_wav_and_malformed_data(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    write_wav(path, b"\0\0" * 4)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ReplayError):
        Pcm16Reader(path)


def test_reject_wav_metadata_and_hash_is_repeatable(tmp_path: Path) -> None:
    path = tmp_path / "metadata.wav"
    write_wav(path, b"\0\0" * 16)
    original = path.read_bytes()
    metadata = b"LIST" + struct.pack("<I", 4) + b"INFO"
    malformed = bytearray(original[:36] + metadata + original[36:])
    struct.pack_into("<I", malformed, 4, len(malformed) - 8)
    path.write_bytes(malformed)
    with pytest.raises(ReplayError, match="metadata"):
        Pcm16Reader(path)

    path.write_bytes(original)
    with Pcm16Reader(path) as first, Pcm16Reader(path) as second:
        assert first.metadata.sha256 == second.metadata.sha256


def test_raw_pcm_requires_explicit_matching_metadata(tmp_path: Path) -> None:
    path = tmp_path / "voice.pcm"
    path.write_bytes(b"\x00\x80\xff\x7f")
    with pytest.raises(ReplayError):
        Pcm16Reader(path)
    with Pcm16Reader(path, raw_metadata=RawPcmMetadata("s16le", 16000, 1, 2)) as reader:
        assert len(reader.read_samples()) == 2
    with pytest.raises(ReplayError, match="size"):
        Pcm16Reader(path, raw_metadata=RawPcmMetadata("s16le", 16000, 1, 1))


@pytest.mark.parametrize("start,end,count", [(-1, 0, 4), (2, 1, 4), (0, 5, 4), (True, 1, 4)])
def test_sample_labels_are_fail_closed(start: int, end: int, count: int) -> None:
    with pytest.raises(ReplayError):
        validate_labels(start, end, count)


def test_limits_and_sample_to_perf_mapping(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    write_wav(path, b"\0\0" * 16)
    with pytest.raises(ReplayError, match="duration"):
        Pcm16Reader(path, max_duration_seconds=0.0005)
    with pytest.raises(ReplayError, match="byte limit"):
        Pcm16Reader(path, max_input_bytes=8)
    assert sample_index_to_perf(50.0, 8000, 16000) == 50.5


def test_replay_source_is_bounded_and_counts_drop_new(tmp_path: Path) -> None:
    path = tmp_path / "queue.wav"
    write_wav(path, b"\0\0" * (MAX_CHUNK_SAMPLES * 2))
    reader = Pcm16Reader(path)
    source = ReplaySource(reader, queue_capacity=1, block_samples=MAX_CHUNK_SAMPLES)
    source.start()
    deadline = time.monotonic() + 2
    while not source.finished and time.monotonic() < deadline:
        time.sleep(0.005)
    assert source.finished
    assert source.queue_max_depth == 1
    assert source.dropped_blocks == 1
    block = source.read()
    assert block is not None and block.sequence == 0
    assert block.captured_at == pytest.approx(source.origin_perf)
    source.stop()
    assert reader._handle is None


def test_replay_uses_40ms_blocks_and_timeline_metrics_keep_domains_separate() -> None:
    assert REPLAY_BLOCK_SAMPLES == 640
    metrics = build_timing_metrics(
        replay_origin_perf=100.0,
        sample_rate=16_000,
        truth_end_sample=16_000,
        endpoint_audio_at=101.05,
        endpoint_at=101.08,
        decode_completed_at=101.10,
        worker_emit_at=101.11,
        controller_received_at=101.12,
        result_observed_at=101.13,
        feed_completed_at=101.02,
        audio_sample_count=16_000,
    )
    assert metrics["truth_end_wall_ms"] == pytest.approx(1000)
    assert metrics["endpoint_audio_sample"] == 16_800
    assert metrics["vad_endpoint_audio_lag_ms"] == pytest.approx(50)
    assert metrics["vad_endpoint_wall_ms"] == pytest.approx(80)
    assert metrics["decode_after_endpoint_ms"] == pytest.approx(20)
    assert metrics["result_latency_ms"] == pytest.approx(130)
    assert metrics["feed_lag_ms"] == pytest.approx(20)


def test_timeline_metrics_reject_out_of_order_runtime_events() -> None:
    with pytest.raises(ReplayError, match="out of order"):
        build_timing_metrics(
            replay_origin_perf=10.0,
            sample_rate=16_000,
            truth_end_sample=16_000,
            endpoint_at=11.2,
            decode_completed_at=11.1,
        )


def test_timeline_metrics_keep_wall_values_null_when_replay_never_started() -> None:
    metrics = build_timing_metrics(
        replay_origin_perf=None,
        sample_rate=16_000,
        truth_end_sample=800,
    )
    assert metrics["truth_end_sample"] == 800
    assert metrics["truth_end_wall_ms"] is None
    assert metrics["result_latency_ms"] is None


def test_cleanup_on_producer_conversion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bad-producer.wav"
    write_wav(path, b"\0\0")
    reader = Pcm16Reader(path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(reader, "read_samples", fail)
    source = ReplaySource(reader)
    source.start()
    deadline = time.monotonic() + 2
    while not source.finished and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(ReplayError, match="producer failed"):
        source.read()
    source.stop()
    assert reader._handle is None


def test_stop_timeout_closes_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "blocked-producer.wav"
    write_wav(path, b"\0\0")
    reader = Pcm16Reader(path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_read(*_args: object, **_kwargs: object) -> None:
        entered.set()
        release.wait(2)
        return None

    monkeypatch.setattr(reader, "read_samples", blocked_read)
    source = ReplaySource(reader)
    source.start()
    assert entered.wait(1)
    with pytest.raises(TimeoutError, match="did not stop"):
        source.stop(timeout=0.01)
    assert reader._handle is None
    release.set()
    assert source._thread is not None
    source._thread.join(1)
    assert not source._thread.is_alive()


def test_controller_uses_supplied_factories_and_public_signals(tmp_path: Path) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    path = tmp_path / "controller.wav"
    write_wav(path, b"\0\0" * 640)
    source = ReplaySource(Pcm16Reader(path))
    calls: list[str] = []

    class FakeEngine:
        def start(self) -> None:
            calls.append("engine.start")

        def process_audio(self, samples: object, *, captured_at: float) -> list[SttResult]:
            calls.append("engine.process")
            return [SttResult("synthetic", captured_at, captured_at, captured_at, None, 1, 1, 0.04)]

        def reset(self) -> None:
            calls.append("engine.reset")

        def close(self) -> None:
            calls.append("engine.close")

    controller = SttController(
        benchmark_stt_config(),
        microphone_factory=lambda: source,
        engine_factory=FakeEngine,
    )
    received: list[SttResult] = []
    diagnostics: list[object] = []
    controller.transcription_ready.connect(received.append)
    controller.diagnostic_ready.connect(diagnostics.append)
    assert controller.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not received:
        app.processEvents()
        time.sleep(0.005)
    assert received and received[0].text == "synthetic"
    assert diagnostics and diagnostics[-1].controller_received_at is not None
    controller.request_stop()
    while time.monotonic() < deadline and controller.state.value not in {"stopped", "error"}:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert controller.state.value == "stopped"
    assert calls == ["engine.start", "engine.process", "engine.close"]
    assert source.reader._handle is None


def benchmark_stt_config():
    from openscreen_ai.stt.config import SttConfig

    return SttConfig()


def test_real_gate_does_not_construct_engine_or_touch_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def factory() -> None:
        nonlocal called
        called = True

    with pytest.raises(benchmark.BenchmarkError, match="real-model disabled"):
        benchmark.require_real_opt_in(None, factory)
    assert not called
    assert "sounddevice" not in sys.modules


def test_existing_json_csv_schema_accepts_null_diagnostics(tmp_path: Path) -> None:
    result = benchmark.run_synthetic(
        [benchmark.Utterance("fake", 16_000, 0, 0, 0, outcome="missing_transcript")],
        sampler=benchmark.FakeResourceSampler(),
    )
    json_path, csv_path = benchmark.export_results(result, tmp_path / "result.json")
    assert json.loads(json_path.read_text())["utterances"][0]["result_latency_ms"] is None
    assert "recognition_ms" in csv_path.read_text()


def test_file_backed_runner_exports_fake_controller_results_safely(tmp_path: Path) -> None:
    from openscreen_ai.stt.config import SttConfig

    QCoreApplication.instance() or QCoreApplication([])
    audio_path = tmp_path / "private-user-audio.wav"
    write_wav(audio_path, b"\x01\x02" * 16_000)
    origin: list[float] = []
    calls = 0
    returned_results = 0

    class FakeEngine:
        def start(self) -> None:
            return None

        def process_audio(self, samples: object, *, captured_at: float) -> list[SttResult]:
            nonlocal calls, returned_results
            calls += 1
            time.sleep(0.055)
            if not origin:
                origin.append(captured_at)
            endpoint_audio_at = captured_at + len(samples) / 16_000
            if returned_results >= 2:
                return []
            if returned_results == 0 and endpoint_audio_at < origin[0] + 0.20:
                return []
            if returned_results == 1 and endpoint_audio_at < origin[0] + 0.80:
                return []
            completed_at = time.perf_counter()
            returned_results += 1
            return [
                SttResult(
                    "PRIVATE_SYNTHETIC_TRANSCRIPT",
                    completed_at,
                    endpoint_audio_at,
                    captured_at,
                    None,
                    4.0,
                    3.0,
                    len(samples) / 16_000,
                    decode_completed_at=completed_at,
                )
            ]

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

    sampler = benchmark.FakeResourceSampler(
        samples=[
            {"process_cpu_ms": 0, "rss_current_bytes": None, "rss_peak_bytes": 100},
            {"process_cpu_ms": 3, "rss_current_bytes": None, "rss_peak_bytes": 120},
            {"process_cpu_ms": 12, "rss_current_bytes": None, "rss_peak_bytes": 140},
        ]
    )
    model_dir = tmp_path / "private-model"
    model_dir.mkdir()
    (model_dir / "weights.onnx").write_bytes(b"synthetic model fixture")
    vad_model_path = tmp_path / "private-vad.onnx"
    vad_model_path.write_bytes(b"synthetic VAD fixture")
    model_fingerprint = fingerprint_model_directory(model_dir)
    vad_fingerprint = fingerprint_file(vad_model_path)
    config = SttConfig(
        model_dir=model_dir,
        vad_model_path=vad_model_path,
        audio_device="private-device-name",
    )
    result = benchmark.run_file_backed_clip(
        audio_path,
        benchmark.Utterance("synthetic-utterance", 16_000, 16_000, 0, 3_200),
        dataset_id="synthetic-fixture",
        config=config,
        test_engine_factory=FakeEngine,
        model_sha256=model_fingerprint,
        vad_model_sha256=vad_fingerprint,
        sampler=sampler,
        queue_capacity=1,
        result_timeout_seconds=1,
    )
    row = result["utterances"][0]
    assert row["outcome"] == "recognized"
    assert row["result_count"] == 2
    assert row["accepted_blocks"] == row["processed_blocks"]
    assert row["expected_result_signals"] == row["observed_result_signals"] == 2
    assert row["result_observed_at_ms"] is not None
    assert row["controller_received_at_ms"] is not None
    assert row["dropped_blocks"] > 0
    assert row["input_overflow_notifications"] == 0
    assert row["queue_overflow"] is True
    assert row["vad_resets_gap"] + row["vad_resets_tail"] > 0
    assert row["queue_max_depth_blocks"] == 1
    assert row["process_cpu_ms"] == 12
    assert row["model_startup_process_cpu_ms"] == 3
    assert row["rss_current_bytes"] is None
    with Pcm16Reader(audio_path) as reader:
        assert row["audio_sha256"] == reader.metadata.sha256
    assert row["model_sha256"] == model_fingerprint
    assert row["vad_model_sha256"] == vad_fingerprint
    assert calls > 1

    json_path, csv_path = benchmark.export_results(result, tmp_path / "safe-result.json")
    serialized = json_path.read_text(encoding="utf-8")
    serialized_csv = csv_path.read_text(encoding="utf-8")
    assert str(audio_path.resolve()) not in serialized
    assert str(audio_path.resolve()) not in serialized_csv
    assert str(config.model_dir) not in serialized
    assert str(config.vad_model_path) not in serialized
    assert str(config.audio_device) not in serialized
    assert "PRIVATE_SYNTHETIC_TRANSCRIPT" not in serialized
    assert audio_path.read_bytes().hex() not in serialized
    assert "PRIVATE_SYNTHETIC_TRANSCRIPT" not in serialized_csv
    assert audio_path.read_bytes().hex() not in serialized_csv
    assert "\u0000" not in serialized
    round_trip = json.loads(serialized)
    assert round_trip["utterances"][0]["rss_current_bytes"] is None
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["rss_current_bytes"] == ""
    assert csv_row["audio_sha256"] == row["audio_sha256"]
    assert csv_row["model_sha256"] == model_fingerprint
    assert csv_row["vad_model_sha256"] == vad_fingerprint


def test_controller_waits_for_result_from_late_final_block(tmp_path: Path) -> None:
    from openscreen_ai.stt.config import SttConfig

    audio_path = tmp_path / "late-result.wav"
    write_wav(audio_path, b"\0\0" * (REPLAY_BLOCK_SAMPLES * 5))

    class FakeEngine:
        calls = 0

        def start(self) -> None:
            return None

        def process_audio(self, samples: object, *, captured_at: float) -> list[SttResult]:
            self.calls += 1
            if self.calls == 5:
                # Producer and queue finish before this late result is available.
                time.sleep(0.30)
            if self.calls not in (1, 5):
                return []
            completed_at = time.perf_counter()
            return [
                SttResult(
                    "SYNTHETIC_TRANSCRIPT",
                    completed_at,
                    captured_at + len(samples) / 16_000,
                    captured_at,
                    None,
                    1.0,
                    1.0,
                    len(samples) / 16_000,
                    decode_completed_at=completed_at,
                )
            ]

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

    result = benchmark.run_file_backed_clip(
        audio_path,
        benchmark.Utterance("late-result", 16_000, 3_200, 0, 3_200),
        dataset_id="synthetic-late-result",
        config=SttConfig(),
        test_engine_factory=FakeEngine,
        queue_capacity=5,
        result_timeout_seconds=1,
    )
    row = result["utterances"][0]
    assert row["outcome"] == "recognized"
    assert row["accepted_blocks"] == row["processed_blocks"] == 5
    assert row["expected_result_signals"] == row["observed_result_signals"] == 2
    assert row["last_result_observed_at_ms"] >= row["feed_completed_at_ms"] + 250


def test_real_file_runner_gate_precedes_model_factory_and_file_access(tmp_path: Path) -> None:
    from openscreen_ai.stt.config import SttConfig

    called = False

    def forbidden_factory() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(benchmark.BenchmarkError, match="real-model disabled"):
        benchmark.run_file_backed_clip(
            tmp_path / "does-not-exist.wav",
            benchmark.Utterance("blocked", 16_000, 1, 0, 1),
            dataset_id="synthetic",
            config=SttConfig(),
            engine_factory=forbidden_factory,
            authorization_token="not-the-authorization-token",
        )
    assert not called


def test_real_factory_rejected_in_parent_before_audio_or_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stt_real_adapter

    audio_path = tmp_path / "missing-parent-audio.wav"
    factory_called = False

    def forbidden_factory() -> object:
        nonlocal factory_called
        factory_called = True
        return object()

    def forbidden_reader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parent process accessed the audio path")

    monkeypatch.setattr(stt_real_adapter, "Pcm16Reader", forbidden_reader)
    with pytest.raises(benchmark.BenchmarkError, match="isolated real-mode child"):
        benchmark.run_file_backed_clip(
            audio_path,
            benchmark.Utterance("blocked-parent", 16_000, 1, 0, 1),
            dataset_id="synthetic-parent-check",
            config=object(),
            engine_factory=forbidden_factory,
            authorization_token=benchmark.REAL_MODEL_OPT_IN,
            model_sha256="a" * 64,
            vad_model_sha256="b" * 64,
        )
    assert not audio_path.exists()
    assert not factory_called


def test_real_cli_gate_precedes_manifest_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched = False

    def forbidden_dispatch(_args: object) -> dict[str, object]:
        nonlocal dispatched
        dispatched = True
        return {}

    monkeypatch.setattr(benchmark, "_run_real_manifest", forbidden_dispatch)
    with pytest.raises(SystemExit):
        benchmark.main(
            [
                "--mode", "real-model",
                "--manifest", str(tmp_path / "missing.json"),
                "--output", str(tmp_path / "result.json"),
            ]
        )
    assert not dispatched
    assert not (tmp_path / "result.json").exists()


def test_real_manifest_gate_precedes_child_and_manifest_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_started = False

    def forbidden_child(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal child_started
        child_started = True
        raise AssertionError("unauthorized manifest must not start a child")

    monkeypatch.setattr(benchmark, "run_bounded_child", forbidden_child)
    args = type(
        "Args",
        (),
        {
            "allow_real_model": None,
            "manifest": tmp_path / "manifest-must-not-be-read.json",
            "model_dir": tmp_path / "model-must-not-be-read",
            "vad_model_path": tmp_path / "vad-must-not-be-read.onnx",
        },
    )()
    with pytest.raises(benchmark.BenchmarkError, match="real-model disabled"):
        benchmark._run_real_manifest(args)
    assert not child_started
    assert not args.manifest.exists()


def test_real_child_gate_precedes_asset_fingerprinting(monkeypatch: pytest.MonkeyPatch) -> None:
    import stt_real_adapter

    class CapturePipe:
        messages: list[dict[str, object]]
        closed: bool

        def __init__(self) -> None:
            self.messages = []
            self.closed = False

        def send(self, message: dict[str, object]) -> None:
            self.messages.append(message)

        def close(self) -> None:
            self.closed = True

    def forbidden_fingerprint(_path: object) -> str:
        raise AssertionError("unauthorized child must not inspect model assets")

    monkeypatch.setattr(stt_real_adapter, "fingerprint_file", forbidden_fingerprint)
    monkeypatch.setattr(
        stt_real_adapter, "fingerprint_model_directory", forbidden_fingerprint
    )
    args = type(
        "Args",
        (),
        {
            "allow_real_model": None,
            "model_dir": "missing synthetic model path",
            "vad_model_path": "missing synthetic VAD path",
        },
    )()
    pipe = CapturePipe()
    benchmark._real_mode_child(pipe, args, {})
    assert pipe.closed
    assert pipe.messages == [{"ok": False, "error_type": "BenchmarkError"}]


def test_real_child_deadline_includes_duration_and_per_clip_grace() -> None:
    assert benchmark._real_child_deadline_seconds(
        {"total_duration": 10.0, "total_expanded": 2}, 5.0
    ) == 10.0 + 2 * (benchmark.REAL_STARTUP_GRACE_SECONDS + 5.0) + (
        benchmark.REAL_FINGERPRINT_GRACE_SECONDS
    )


def test_bounded_real_child_terminates_hanging_fake_within_cleanup_grace() -> None:
    deadline_seconds = 1.5
    shutdown_grace_seconds = 0.5
    started = time.monotonic()
    result = benchmark.run_bounded_child(
        _hanging_spawn_child,
        (),
        deadline_seconds=deadline_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    elapsed = time.monotonic() - started
    assert result["timed_out"] is True
    assert result["child_alive"] is False
    assert result["exitcode"] is not None
    assert elapsed < deadline_seconds + shutdown_grace_seconds + 2.0


def test_exported_config_is_a_path_and_secret_free_allowlist() -> None:
    assert benchmark._safe_stt_configuration(
        {
            "provider": "cpu",
            "sample_rate": 16_000,
            "model_dir": "/private/models",
            "api_key": "synthetic-secret-marker",
        }
    ) == {"provider": "cpu", "sample_rate": 16_000}


def test_model_and_vad_fingerprints_are_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "model-a"
    second = tmp_path / "model-b"
    (first / "nested").mkdir(parents=True)
    second.mkdir()
    (first / "z.onnx").write_bytes(b"synthetic model weights")
    (first / "nested" / "tokens.txt").write_bytes(b"synthetic tokens")
    (second / "nested").mkdir()
    (second / "nested" / "tokens.txt").write_bytes(b"synthetic tokens")
    (second / "z.onnx").write_bytes(b"synthetic model weights")
    first_hash = fingerprint_model_directory(first)
    assert first_hash == fingerprint_model_directory(first)
    assert first_hash == fingerprint_model_directory(second)

    vad_a = tmp_path / "vad-a.onnx"
    vad_b = tmp_path / "vad-b.onnx"
    vad_a.write_bytes(b"synthetic vad weights")
    vad_b.write_bytes(b"different synthetic vad weights")
    assert fingerprint_file(vad_a) == fingerprint_file(vad_a)
    assert fingerprint_file(vad_a) != fingerprint_file(vad_b)
    (second / "z.onnx").write_bytes(b"modified synthetic model weights")
    assert first_hash != fingerprint_model_directory(second)


def test_model_fingerprints_fail_closed_for_empty_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty-model"
    empty.mkdir()
    with pytest.raises(ReplayError, match="no files"):
        fingerprint_model_directory(empty)

    target = tmp_path / "target.onnx"
    target.write_bytes(b"synthetic weights")
    linked_file = tmp_path / "linked.onnx"
    linked_dir = tmp_path / "linked-model"
    try:
        linked_file.symlink_to(target)
        linked_dir.mkdir()
        (linked_dir / "weights.onnx").symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable in test temp directory: {error}")
    with pytest.raises(ReplayError, match="symlink"):
        fingerprint_file(linked_file)
    with pytest.raises(ReplayError, match="symlink"):
        fingerprint_model_directory(linked_dir)
