"""Fake-only tests for STT benchmark infrastructure."""

from __future__ import annotations

import copy
import csv
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import stt_benchmark as bench


def item(**overrides: object) -> bench.Utterance:
    data = {
        "utterance_id": "short",
        "sample_rate": 1000,
        "audio_duration_samples": 100,
        "truth_start_sample": 0,
        "truth_end_sample": 80,
        "fixture": "silence_noise",
    }
    data.update(overrides)
    return bench.Utterance(**data)


def run_fake(utterances: list[bench.Utterance], **kwargs: object) -> dict[str, object]:
    return bench.run_synthetic(utterances, clock=bench.FakeClock(), **kwargs)


def test_paced_replay_is_deterministic_and_keeps_clock_domains_separate() -> None:
    clock = bench.FakeClock(10.0)
    result = bench.run_synthetic(
        [item()], clock=clock, engine=bench.FakeEngine(), sampler=bench.FakeResourceSampler()
    )
    row = result["utterances"][0]
    assert clock.sleeps == [0.04, 0.04, 0.02]
    assert row["truth_end_sample"] == 80
    assert row["truth_end_wall_ms"] == pytest.approx(80)
    assert row["endpoint_at_ms"] == pytest.approx(100)
    assert row["endpoint_audio_sample"] == 80
    assert row["endpoint_audio_at_ms"] == pytest.approx(80)
    assert row["vad_endpoint_audio_lag_ms"] == 0
    assert row["vad_endpoint_wall_ms"] == pytest.approx(20)
    assert row["controller_received_at_ms"] < row["result_observed_at_ms"]


def test_wall_endpoint_delay_does_not_change_audio_endpoint_timing() -> None:
    result = run_fake(
        [item()],
        engine=bench.FakeEngine(endpoint_wall_delay_ms=50),
        sampler=bench.FakeResourceSampler(),
    )
    row = result["utterances"][0]
    assert row["endpoint_at_ms"] == pytest.approx(150)
    assert row["vad_endpoint_wall_ms"] == pytest.approx(70)
    assert row["endpoint_audio_sample"] == 80
    assert row["vad_endpoint_audio_lag_ms"] == 0


def test_empty_transcript_keeps_available_result_latency() -> None:
    result = run_fake([item(outcome="empty_transcript")], sampler=bench.FakeResourceSampler())
    row = result["utterances"][0]
    assert row["result_observed_at_ms"] is not None
    assert row["result_latency_ms"] is not None


@pytest.mark.parametrize(
    "values,p,expected",
    [([], 0.95, None), ([1, 2, 3, 4], 0.5, 2), ([1, 2, 3, 4], 0.95, 4), ([1, 2, 3, 4], 0.99, 4)],
)
def test_quantiles_use_nearest_rank_and_null_empty(
    values: list[int], p: float, expected: int | None
) -> None:
    assert bench.nearest_rank_quantile(values, p) == expected


def test_json_csv_round_trip_schema_and_missing_outcome(tmp_path: Path) -> None:
    items = [item(), item(utterance_id="lost", outcome="missing_transcript")]
    result = run_fake(items, sampler=bench.FakeResourceSampler())
    json_path, csv_path = bench.export_results(result, tmp_path / "run.json")
    loaded = json.loads(json_path.read_text())
    assert loaded["schema_version"] == 1
    assert loaded["utterances"][0]["schema_version"] == 1
    assert loaded["utterances"][0]["run_id"] == loaded["utterances"][1]["run_id"]
    assert loaded["utterances"][1]["result_latency_ms"] is None
    assert loaded["utterances"][1]["result_observed_at_ms"] is None
    assert loaded["utterances"][0]["process_cpu_ms"] == 0
    assert loaded["utterances"][0]["rss_current_bytes"] is None
    assert loaded["utterances"][0]["rss_peak_bytes"] == 0
    assert loaded["summary"]["outcomes"]["missing_transcript"] == 1
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[1]["outcome"] == "missing_transcript"
    assert rows[1]["result_latency_ms"] == ""
    assert rows[0]["process_cpu_ms"] == "0"
    assert rows[0]["rss_current_bytes"] == ""
    no_results = run_fake([item(outcome="missing_transcript")], sampler=bench.FakeResourceSampler())
    assert no_results["summary"]["metrics"]["result_latency_ms"]["sample_count"] == 0
    assert no_results["summary"]["metrics"]["result_latency_ms"]["p50"] is None


@pytest.mark.parametrize("existing_suffix", [".json", ".csv"])
def test_export_refuses_to_overwrite_either_output(tmp_path: Path, existing_suffix: str) -> None:
    result = run_fake([item()], sampler=bench.FakeResourceSampler())
    output_path = tmp_path / "existing.json"
    existing_path = output_path if existing_suffix == ".json" else output_path.with_suffix(".csv")
    existing_path.write_text("keep this result")
    with pytest.raises(bench.BenchmarkError, match="refusing to overwrite"):
        bench.export_results(result, output_path)
    assert existing_path.read_text() == "keep this result"
    if existing_suffix == ".csv":
        assert not output_path.exists()
    else:
        assert output_path.read_text() == "keep this result"


def test_manifest_repeat_and_limits_are_validated() -> None:
    manifest = {
        "utterances": [
            {
                "utterance_id": "same",
                "sample_rate": 16000,
                "audio_duration_samples": 1600,
                "truth_start_sample": 0,
                "truth_end_sample": 1200,
                "repeat": 2,
            }
        ]
    }
    expanded = bench.validate_manifest(manifest, max_utterances=2)
    assert [entry.utterance_id for entry in expanded] == ["same-r1", "same-r2"]
    with pytest.raises(bench.BenchmarkError, match="max_utterances"):
        bench.validate_manifest(manifest, max_utterances=1)
    with pytest.raises(bench.BenchmarkError, match="aggregation limit"):
        run_fake(expanded, max_utterances=1)


def test_replay_duration_cap_allows_thirty_minutes_and_rejects_more() -> None:
    thirty_minutes = item(
        sample_rate=16000,
        audio_duration_samples=30 * 60 * 16000,
        truth_end_sample=30 * 60 * 16000,
        fixture="continuous",
    )
    assert bench.replay_duration_seconds([thirty_minutes]) == 1800
    over_cap = item(
        sample_rate=16000,
        audio_duration_samples=30 * 60 * 16000 + 1,
        truth_end_sample=30 * 60 * 16000 + 1,
        fixture="continuous",
    )
    with pytest.raises(bench.BenchmarkError, match="exceeds configured cap"):
        run_fake([over_cap], sampler=bench.FakeResourceSampler())


def test_cli_dry_run_validates_synthetic_manifest_only(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "utterances": [
                    {
                        "utterance_id": "dry-run",
                        "sample_rate": 16000,
                        "audio_duration_samples": 800,
                        "truth_start_sample": 0,
                        "truth_end_sample": 600,
                        "fixture": "short_speech",
                    }
                ]
            }
        )
    )
    output_path = tmp_path / "dry-run.json"
    assert (
        bench.main(["--manifest", str(manifest_path), "--output", str(output_path), "--dry-run"])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "valid": True,
        "mode": "synthetic",
        "utterance_count": 1,
    }
    assert not output_path.exists()


def test_continuous_and_pause_fixtures_use_sample_schedules() -> None:
    utterance = item(
        utterance_id="continuous",
        sample_rate=1000,
        audio_duration_samples=30000,
        truth_end_sample=29000,
        pause_after_samples=250,
        fixture="continuous",
    )
    clock = bench.FakeClock()
    result = bench.run_synthetic([utterance], clock=clock, sampler=bench.FakeResourceSampler())
    assert sum(clock.sleeps) == pytest.approx(30.25)
    assert result["utterances"][0]["pause_after_samples"] == 250


def test_synthetic_noise_is_repeatable_and_generated_in_bounded_chunks() -> None:
    class CapturingSource(bench.FakeSource):
        def __init__(self) -> None:
            super().__init__(capacity=8)
            self.payloads: list[bytes] = []

        def push(self, item: object, *, input_overflow: bool | None = None) -> bool:
            self.payloads.append(item)
            return super().push(item, input_overflow=input_overflow)

    noise = item(
        utterance_id="noise",
        sample_rate=100_000,
        audio_duration_samples=5_000,
        truth_end_sample=5_000,
        fixture="synthetic_noise",
    )
    first_source = CapturingSource()
    second_source = CapturingSource()
    run_fake([noise], source=first_source, block_ms=100, sampler=bench.FakeResourceSampler())
    run_fake([noise], source=second_source, block_ms=100, sampler=bench.FakeResourceSampler())

    assert first_source.payloads == second_source.payloads
    assert [len(payload) for payload in first_source.payloads] == [
        bench.MAX_SYNTHETIC_PCM_SAMPLES * 4,
        (5_000 - bench.MAX_SYNTHETIC_PCM_SAMPLES) * 4,
    ]
    values = struct.unpack("<5f", first_source.payloads[0][:20])
    assert any(values)
    assert max(abs(value) for value in values) <= bench.SYNTHETIC_NOISE_AMPLITUDE
    assert bench.generate_synthetic_pcm("silence_noise", 8) == bytes(8 * 4)
    with pytest.raises(bench.BenchmarkError, match="synthetic PCM block"):
        bench.generate_synthetic_pcm("synthetic_noise", bench.MAX_SYNTHETIC_PCM_SAMPLES + 1)


def test_fake_queue_overflow_drop_and_reset_accounting() -> None:
    source = bench.FakeSource(capacity=2)
    assert source.push(1) and source.push(2)
    assert not source.push(3) and not source.push(4) and not source.push(5)
    assert (source.queue_max_depth, source.queue_depth, source.dropped_blocks) == (2, 2, 3)
    assert (source.input_overflows, source.vad_resets_total) == (0, 0)
    source.finish()
    assert (source.input_overflows, source.vad_resets_total, source.vad_resets_tail) == (0, 1, 1)

    engine = bench.FakeEngine()
    result = run_fake(
        [item(audio_duration_samples=160, truth_end_sample=160, fixture="queue_overload")],
        source=bench.FakeSource(capacity=2),
        engine=engine,
        sampler=bench.FakeResourceSampler(),
    )
    row = result["utterances"][0]
    assert row["dropped_blocks"] > 1
    assert row["input_overflow_notifications"] == 0
    assert row["overflow"] is False
    assert row["queue_overflow"] is True
    assert row["vad_resets_total"] == 1
    assert engine.reset_calls == 1


def test_scripted_device_overflow_is_separate_from_queue_drops() -> None:
    source = bench.FakeSource(capacity=2, scripted_input_overflow_sequences={0})
    engine = bench.FakeEngine()
    result = run_fake([item()], source=source, engine=engine, sampler=bench.FakeResourceSampler())
    row = result["utterances"][0]
    assert row["dropped_blocks"] == 0
    assert row["input_overflow_notifications"] == 1
    assert row["overflow"] is True
    assert row["queue_overflow"] is False
    assert row["vad_resets_total"] == row["vad_resets_overflow"] == 1
    assert engine.reset_calls == 1


def test_internal_multi_block_gap_is_one_consumer_reset() -> None:
    source = bench.FakeSource(capacity=2)
    source.push(bytes(4))
    source.read()
    source.push(bytes(4))
    source.push(bytes(4))
    source.push(bytes(4))
    source.push(bytes(4))
    source.read()
    source.read()
    source.push(bytes(4))
    source.read()
    source.finish()
    assert source.dropped_blocks == 2
    assert source.input_overflows == 0
    assert source.vad_resets_gap == 1
    assert source.vad_resets_total == 1


def test_synthetic_internal_queue_gap_calls_engine_reset_once() -> None:
    engine = bench.FakeEngine()
    result = run_fake(
        [item(audio_duration_samples=200, truth_end_sample=200, fixture="queue_overload")],
        source=bench.FakeSource(capacity=2),
        engine=engine,
        sampler=bench.FakeResourceSampler(),
    )
    row = result["utterances"][0]
    assert row["dropped_blocks"] == 2
    assert row["vad_resets_gap"] == 1
    assert row["vad_resets_tail"] == 0
    assert engine.reset_calls == 1


def test_queue_capacity_is_bounded() -> None:
    with pytest.raises(bench.BenchmarkError, match="queue capacity"):
        bench.FakeSource(capacity=bench.MAX_QUEUE_CAPACITY + 1)


def test_real_model_guard_fails_before_factory_call() -> None:
    called = False

    def factory() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(bench.BenchmarkError, match="real-model disabled"):
        bench.require_real_opt_in(None, factory)
    assert not called
    with pytest.raises(bench.BenchmarkError, match="adapter is unavailable"):
        bench.require_real_opt_in(bench.REAL_MODEL_OPT_IN)
    assert not called


def test_fake_only_module_has_no_audio_device_dependencies() -> None:
    tools_dir = Path(bench.__file__).parent
    code = (
        "import sys; import stt_benchmark; "
        "assert not any(name == 'PySide6' or name.startswith('PySide6.') "
        "or name in {'sounddevice', 'sherpa_onnx'} for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": str(tools_dir)}
    )


def test_absent_engine_diagnostics_are_json_null() -> None:
    class PartialEngine:
        def diagnostics(self, endpoint: float, *, truth_end_sample: int) -> dict[str, float | None]:
            return {
                "endpoint_at": None,
                "endpoint_audio_sample": None,
                "decode_completed_at": None,
                "worker_emit_at": None,
                "controller_received_at": None,
                "recognition_ms": None,
                "cpu_ms": None,
            }

    result = run_fake([item()], engine=PartialEngine(), sampler=bench.FakeResourceSampler())
    row = result["utterances"][0]
    for key in (
        "endpoint_at_ms",
        "decode_completed_at_ms",
        "worker_emit_at_ms",
        "controller_received_at_ms",
        "recognition_ms",
        "cpu_ms",
        "post_vad_ms",
        "vad_endpoint_audio_lag_ms",
        "vad_endpoint_wall_ms",
    ):
        assert row[key] is None
    assert row["result_observed_at_ms"] is not None


def test_resource_sampler_injected_not_system_sampler() -> None:
    sampler = bench.FakeResourceSampler(
        [{"process_cpu_ms": 7, "rss_current_bytes": None, "rss_peak_bytes": 1024}]
    )
    result = run_fake([item()], sampler=sampler)
    assert result["summary"]["process_cpu_ms"]["p50"] == 7
    assert result["summary"]["metrics"]["rss_peak_bytes"]["p50"] == 1024
    assert result["utterances"][0]["process_cpu_ms"] == 7
    assert result["utterances"][0]["process_cpu_percent"] == pytest.approx(7)
    assert result["utterances"][0]["rss_peak_bytes"] == 1024
    assert result["summary"]["metrics"]["queue_depth_blocks"]["sample_count"] == 1
    assert result["utterances"][0]["cpu_ms"] == 1


def test_comparison_helper_reports_paired_deltas_and_preserves_nulls(tmp_path: Path) -> None:
    baseline = run_fake(
        [item(), item(utterance_id="missing", outcome="missing_transcript")],
        sampler=bench.FakeResourceSampler(
            [
                {"process_cpu_ms": 2, "rss_current_bytes": None, "rss_peak_bytes": 1000},
                {"process_cpu_ms": 0, "rss_current_bytes": None, "rss_peak_bytes": 1000},
            ]
        ),
    )
    dev001 = copy.deepcopy(baseline)
    baseline["run"]["git_head"] = "baseline-sha"
    dev001["run"]["git_head"] = "dev001-sha"
    dev001["utterances"][0]["result_latency_ms"] += 5
    dev001["utterances"][0]["recognition_ms"] = 3
    dev001["utterances"][0]["cpu_ms"] = 2
    dev001["utterances"][0]["process_cpu_ms"] = 4
    dev001["utterances"][0]["rss_peak_bytes"] = 1200
    dev001["utterances"][1]["outcome"] = "recognized"
    dev001["utterances"][1]["result_latency_ms"] = 9

    baseline_path = tmp_path / "baseline.json"
    dev001_path = tmp_path / "dev001.json"
    baseline_path.write_text(json.dumps(baseline))
    dev001_path.write_text(json.dumps(dev001))
    comparison = bench.compare_artifacts(baseline_path, dev001_path)
    pair = comparison["pairs"][0]["metrics"]
    assert pair["result_latency_ms"]["absolute_delta"] == 5
    baseline_latency = baseline["utterances"][0]["result_latency_ms"]
    assert pair["result_latency_ms"]["percent_delta"] == pytest.approx(100 * 5 / baseline_latency)
    assert pair["recognition_ms"]["absolute_delta"] == 1
    assert pair["process_cpu_ms"]["absolute_delta"] == 2
    assert pair["rss_peak_bytes"]["absolute_delta"] == 200
    assert pair["rss_peak_bytes"]["percent_delta"] == 20
    assert comparison["pairs"][1]["baseline_outcome"] == "missing_transcript"
    assert comparison["pairs"][1]["dev001_outcome"] == "recognized"
    assert comparison["pairs"][1]["metrics"]["result_latency_ms"] == {
        "baseline": None,
        "dev001": 9,
        "absolute_delta": None,
        "percent_delta": None,
    }
    assert comparison["metrics"]["result_latency_ms"]["paired_sample_count"] == 1
    assert comparison["metrics"]["result_latency_ms"]["missing_pair_count"] == 1


def test_comparison_helper_refuses_mismatched_scenarios() -> None:
    baseline = run_fake([item()], sampler=bench.FakeResourceSampler())
    dev001 = copy.deepcopy(baseline)
    dev001["utterances"][0]["fixture"] = "continuous"
    with pytest.raises(bench.BenchmarkError, match="scenarios differ"):
        bench.compare_artifacts(baseline, dev001)

    dev001 = copy.deepcopy(baseline)
    dev001["run"]["block_ms"] += 20
    with pytest.raises(bench.BenchmarkError, match="configurations/environments differ"):
        bench.compare_artifacts(baseline, dev001)
