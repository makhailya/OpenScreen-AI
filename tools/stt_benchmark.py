"""Deterministic, fake-first STT benchmark primitives and CLI.

Importing this module does not load the application STT stack. Synthetic mode
measures harness behavior only; file-backed execution imports the STT stack
after its explicit authorization gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import multiprocessing.connection
import platform
import queue
import random
import resource
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1
QUANTILE_RULE = "nearest_rank_ceil_n_p"
DEFAULT_MAX_UTTERANCES = 10_000
MAX_CONFIGURED_UTTERANCES = 100_000
DEFAULT_QUEUE_CAPACITY = 20
MAX_QUEUE_CAPACITY = 1_000
MAX_SYNTHETIC_PCM_SAMPLES = 4_096
MAX_REAL_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REAL_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_REAL_CLIPS = 100
MAX_REAL_RESULT_TIMEOUT_SECONDS = 60
REAL_STARTUP_GRACE_SECONDS = 30
REAL_FINGERPRINT_GRACE_SECONDS = 300
REAL_SHUTDOWN_GRACE_SECONDS = 2
_REAL_MODE_CHILD_ACTIVE = False
DEFAULT_MAX_REPLAY_DURATION_SECONDS = 30 * 60
MAX_CONFIGURED_REPLAY_DURATION_SECONDS = 6 * 60 * 60
BASELINE_HEAD = "a63ee21d9f27375b8c7ca8a677f13f792a524302"
REAL_MODEL_OPT_IN = "I_AUTHORIZE_REAL_MODEL_EXECUTION"
SYNTHETIC_NOISE_SEED = 20_261_110
SYNTHETIC_NOISE_AMPLITUDE = 0.02
FIXTURE_CASES = frozenset(
    {
        "speech",
        "short_speech",
        "long_speech",
        "pause",
        "silence_noise",
        "synthetic_noise",
        "continuous",
        "queue_overload",
        "repeated_identical",
    }
)


class BenchmarkError(ValueError):
    """Invalid configuration or bounded benchmark bookkeeping overflow."""


class Clock(Protocol):
    def perf_counter(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass
class PerfClock:
    """Wall-paced monotonic clock for CLI replay; tests use FakeClock."""

    def perf_counter(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ResourceSampler(Protocol):
    def sample(self) -> dict[str, int | None]: ...


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    sample_rate: int
    audio_duration_samples: int
    truth_start_sample: int
    truth_end_sample: int
    outcome: str = "recognized"
    pause_after_samples: int = 0
    fixture: str = "speech"
    repeat: int = 1

    def validate(self) -> None:
        integer_values = (
            self.sample_rate,
            self.audio_duration_samples,
            self.truth_start_sample,
            self.truth_end_sample,
            self.pause_after_samples,
            self.repeat,
        )
        if any(type(value) is not int for value in integer_values):
            raise BenchmarkError("sample labels, rate, duration, pause and repeat must be integers")
        if not self.utterance_id or self.sample_rate <= 0 or self.audio_duration_samples < 0:
            raise BenchmarkError("utterance id, positive sample rate and duration are required")
        if not (
            0 <= self.truth_start_sample <= self.truth_end_sample <= self.audio_duration_samples
        ):
            raise BenchmarkError("truth sample labels must be inside the audio duration")
        if self.pause_after_samples < 0 or self.repeat < 1:
            raise BenchmarkError("pause and repeat must be non-negative/positive")
        if self.fixture not in FIXTURE_CASES:
            raise BenchmarkError(f"unsupported synthetic fixture case: {self.fixture}")
        if self.outcome not in {
            "recognized",
            "missing_transcript",
            "empty_transcript",
            "timeout",
            "decode_error",
            "input_error",
        }:
            raise BenchmarkError(f"unsupported outcome: {self.outcome}")


@dataclass
class FakeClock:
    """A manually advanced monotonic clock for deterministic replay and tests."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise BenchmarkError("sleep duration cannot be negative")
        self.sleeps.append(seconds)
        self.now += seconds


def generate_synthetic_pcm(
    fixture: str,
    sample_count: int,
    *,
    noise_rng: random.Random | None = None,
) -> bytes:
    """Generate one bounded little-endian float32 PCM block without storing audio."""
    if type(sample_count) is not int or not 0 <= sample_count <= MAX_SYNTHETIC_PCM_SAMPLES:
        raise BenchmarkError(
            f"synthetic PCM block must contain 0..{MAX_SYNTHETIC_PCM_SAMPLES} samples"
        )
    if fixture == "synthetic_noise":
        rng = noise_rng or random.Random(SYNTHETIC_NOISE_SEED)
        return struct.pack(
            f"<{sample_count}f",
            *(
                rng.uniform(-SYNTHETIC_NOISE_AMPLITUDE, SYNTHETIC_NOISE_AMPLITUDE)
                for _ in range(sample_count)
            ),
        )
    return bytes(sample_count * 4)


@dataclass
class SystemResourceSampler:
    """Process CPU delta and OS process high-water RSS, sampled per utterance."""

    _last_cpu_ns: int = field(default_factory=time.process_time_ns)

    def sample(self) -> dict[str, int | None]:
        current_cpu_ns = time.process_time_ns()
        cpu_ms = (current_cpu_ns - self._last_cpu_ns) // 1_000_000
        self._last_cpu_ns = current_cpu_ns
        rss_peak_bytes: int | None = None
        try:
            # macOS reports bytes; Linux and the supported BSD fallback report KiB.
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
        except (AttributeError, OSError):
            pass
        return {
            "process_cpu_ms": cpu_ms,
            "rss_current_bytes": None,
            "rss_peak_bytes": rss_peak_bytes,
        }


@dataclass
class FakeResourceSampler:
    samples: list[dict[str, int | None]] = field(default_factory=list)
    index: int = 0

    def sample(self) -> dict[str, int | None]:
        if not self.samples:
            return {
                "process_cpu_ms": 0,
                "rss_current_bytes": None,
                "rss_peak_bytes": 0,
            }
        result = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return dict(result)


class _EngineProgress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processed_blocks = 0
        self._expected_result_signals = 0

    def completed(self, result_count: int) -> None:
        with self._lock:
            self._processed_blocks += 1
            self._expected_result_signals += result_count

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "processed_blocks": self._processed_blocks,
                "expected_result_signals": self._expected_result_signals,
            }


class _ProgressTrackingEngine:
    """Count accepted-block completion without changing the STT worker."""

    def __init__(self, factory: Callable[[], Any], progress: _EngineProgress) -> None:
        self._factory = factory
        self._progress = progress
        self._engine: Any | None = None

    def start(self) -> None:
        self._engine = self._factory()
        self._engine.start()

    def process_audio(self, samples: Any, *, captured_at: float) -> list[Any]:
        if self._engine is None:
            raise BenchmarkError("tracked STT engine was not started")
        results = self._engine.process_audio(samples, captured_at=captured_at)
        result_list = list(results or ())
        self._progress.completed(len(result_list))
        return result_list

    def reset(self) -> None:
        if self._engine is not None:
            self._engine.reset()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()


@dataclass
class FakeAudioBlock:
    sequence: int
    input_overflows: int
    payload: Any


@dataclass
class FakeSource:
    """Bounded synthetic block queue; overflow is explicit and counted."""

    capacity: int = DEFAULT_QUEUE_CAPACITY
    queue: queue.Queue[FakeAudioBlock] = field(init=False)
    scripted_input_overflow_sequences: set[int] = field(default_factory=set)
    dropped_blocks: int = 0
    queue_max_depth: int = 0
    input_overflows: int = 0
    vad_resets_total: int = 0
    vad_resets_gap: int = 0
    vad_resets_overflow: int = 0
    vad_resets_tail: int = 0
    _next_sequence: int = 0
    _expected_sequence: int = 0
    _accounted_drops: int = 0
    _observed_input_overflows: int = 0

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or not 1 <= self.capacity <= MAX_QUEUE_CAPACITY:
            raise BenchmarkError(f"queue capacity must be between 1 and {MAX_QUEUE_CAPACITY}")
        self.queue = queue.Queue(maxsize=self.capacity)

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    def push(self, item: Any, *, input_overflow: bool | None = None) -> bool:
        sequence = self._next_sequence
        self._next_sequence += 1
        status_overflow = (
            sequence in self.scripted_input_overflow_sequences
            if input_overflow is None
            else input_overflow
        )
        if status_overflow:
            self.input_overflows += 1
        block = FakeAudioBlock(sequence, self.input_overflows, item)
        try:
            self.queue.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1
            return False
        self.queue_max_depth = max(self.queue_max_depth, self.queue.qsize())
        return True

    def read(self) -> FakeAudioBlock | None:
        try:
            block = self.queue.get_nowait()
        except queue.Empty:
            return None
        sequence_gap = block.sequence - self._expected_sequence
        saw_overflow = block.input_overflows > self._observed_input_overflows
        if sequence_gap > 0 or saw_overflow:
            # One worker reset is emitted per consumed discontinuity, even when
            # both a sequence gap and a device overflow are present.
            self.vad_resets_total += 1
            if sequence_gap > 0:
                self.vad_resets_gap += 1
                self._accounted_drops += sequence_gap
            if saw_overflow:
                self.vad_resets_overflow += 1
        self._expected_sequence = block.sequence + 1
        self._observed_input_overflows = block.input_overflows
        return block

    def finish(self) -> None:
        """Drain accepted blocks, then account tail drops as one reset."""
        while self.read() is not None:
            pass
        tail_drops = self.dropped_blocks - self._accounted_drops
        if tail_drops > 0:
            self.vad_resets_total += 1
            self.vad_resets_tail += 1
            self._accounted_drops += tail_drops


@dataclass
class FakeEngine:
    """Fake diagnostics have stable values, independent of wall-clock speed."""

    recognition_ms: float = 2.0
    cpu_ms: float = 1.0
    endpoint_wall_delay_ms: float = 0.0
    endpoint_audio_sample: int | None = None
    reset_calls: int = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def diagnostics(self, endpoint: float, *, truth_end_sample: int) -> dict[str, Any]:
        endpoint_at = endpoint + self.endpoint_wall_delay_ms / 1000
        return {
            "endpoint_at": endpoint_at,
            "endpoint_audio_sample": (
                self.endpoint_audio_sample
                if self.endpoint_audio_sample is not None
                else truth_end_sample
            ),
            "recognition_ms": self.recognition_ms,
            "decode_completed_at": endpoint_at + self.recognition_ms / 1000,
            "worker_emit_at": endpoint_at + self.recognition_ms / 1000 + 0.001,
            "controller_received_at": endpoint_at + self.recognition_ms / 1000 + 0.002,
            "cpu_ms": self.cpu_ms,
        }


def nearest_rank_quantile(values: Iterable[float], probability: float) -> float | None:
    """Nearest-rank (ceil(n*p)) quantile, with a 1-based rank."""
    if not 0 <= probability <= 1:
        raise BenchmarkError("quantile probability must be between 0 and 1")
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(len(ordered) * probability))
    return ordered[rank - 1]


def summarize(values: Iterable[float]) -> dict[str, Any]:
    observations = list(values)
    return {
        "sample_count": len(observations),
        "quantile_rule": QUANTILE_RULE,
        "p50": nearest_rank_quantile(observations, 0.50),
        "p95": nearest_rank_quantile(observations, 0.95),
        "p99": nearest_rank_quantile(observations, 0.99),
    }


def replay_duration_seconds(utterances: Iterable[Utterance]) -> float:
    return sum(
        (utterance.audio_duration_samples + utterance.pause_after_samples) / utterance.sample_rate
        for utterance in utterances
    )


def validate_replay_duration(
    utterances: list[Utterance], max_replay_duration_seconds: float
) -> float:
    if (
        isinstance(max_replay_duration_seconds, bool)
        or not isinstance(max_replay_duration_seconds, (int, float))
        or not math.isfinite(max_replay_duration_seconds)
        or not 0 < max_replay_duration_seconds <= MAX_CONFIGURED_REPLAY_DURATION_SECONDS
    ):
        raise BenchmarkError(
            f"max_replay_duration_seconds must be in (0, {MAX_CONFIGURED_REPLAY_DURATION_SECONDS}]."
        )
    for utterance in utterances:
        utterance.validate()
    duration = replay_duration_seconds(utterances)
    if duration > max_replay_duration_seconds:
        raise BenchmarkError(
            f"planned replay duration {duration:g}s exceeds configured cap "
            f"{max_replay_duration_seconds:g}s"
        )
    return duration


def validate_manifest(
    manifest: dict[str, Any], *, max_utterances: int = DEFAULT_MAX_UTTERANCES
) -> list[Utterance]:
    if type(max_utterances) is not int or not 1 <= max_utterances <= MAX_CONFIGURED_UTTERANCES:
        raise BenchmarkError(f"max_utterances must be between 1 and {MAX_CONFIGURED_UTTERANCES}")
    raw = manifest.get("utterances")
    if not isinstance(raw, list) or not raw:
        raise BenchmarkError("manifest must contain a non-empty utterances list")
    expanded: list[Utterance] = []
    for item in raw:
        utterance = Utterance(**item)
        utterance.validate()
        if len(expanded) + utterance.repeat > max_utterances:
            raise BenchmarkError("manifest exceeds max_utterances bookkeeping bound")
        for repeat in range(1, utterance.repeat + 1):
            expanded.append(
                Utterance(
                    **{
                        **asdict(utterance),
                        "repeat": repeat,
                        "utterance_id": f"{utterance.utterance_id}-r{repeat}",
                    }
                )
            )
    return expanded


def run_synthetic(
    utterances: list[Utterance],
    *,
    clock: Clock | None = None,
    source: FakeSource | None = None,
    engine: FakeEngine | None = None,
    sampler: ResourceSampler | None = None,
    block_ms: int = 40,
    timeout_ms: int = 5000,
    max_utterances: int = DEFAULT_MAX_UTTERANCES,
    max_replay_duration_seconds: float = DEFAULT_MAX_REPLAY_DURATION_SECONDS,
) -> dict[str, Any]:
    """Replay metadata/zero PCM blocks on a deterministic paced schedule."""
    if not 20 <= block_ms <= 100 or timeout_ms <= 0:
        raise BenchmarkError("block_ms must be 20..100 and timeout_ms positive")
    if type(max_utterances) is not int or not 1 <= max_utterances <= MAX_CONFIGURED_UTTERANCES:
        raise BenchmarkError(f"max_utterances must be between 1 and {MAX_CONFIGURED_UTTERANCES}")
    if len(utterances) > max_utterances:
        raise BenchmarkError("aggregation limit exceeded; no samples were silently dropped")
    replay_duration = validate_replay_duration(utterances, max_replay_duration_seconds)
    clock = clock or PerfClock()
    source = source or FakeSource()
    engine = engine or FakeEngine()
    origin = clock.perf_counter()
    sampler = sampler or SystemResourceSampler()
    rows: list[dict[str, Any]] = []
    last_resource_sample_at = origin
    run_id = uuid.uuid4().hex
    audio_schedule_offset = 0.0
    observed_reset_total = source.vad_resets_total

    def apply_new_resets() -> None:
        nonlocal observed_reset_total
        new_resets = source.vad_resets_total - observed_reset_total
        reset = getattr(engine, "reset", None)
        if callable(reset):
            for _ in range(max(0, new_resets)):
                reset()
        observed_reset_total = source.vad_resets_total

    for utterance in utterances:
        utterance.validate()
        block_samples = min(
            MAX_SYNTHETIC_PCM_SAMPLES,
            max(1, utterance.sample_rate * block_ms // 1000),
        )
        noise_rng = (
            random.Random(SYNTHETIC_NOISE_SEED) if utterance.fixture == "synthetic_noise" else None
        )
        utterance_start_perf = origin + audio_schedule_offset
        samples_sent = 0
        while samples_sent < utterance.audio_duration_samples:
            chunk_samples = min(block_samples, utterance.audio_duration_samples - samples_sent)
            pcm_chunk = generate_synthetic_pcm(
                utterance.fixture, chunk_samples, noise_rng=noise_rng
            )
            source.push(pcm_chunk)
            if utterance.fixture != "queue_overload":
                source.read()
                apply_new_resets()
            elif source.dropped_blocks >= 2:
                # Let the fake producer continue after a short burst of losses;
                # the next accepted sequence exposes the internal discontinuity.
                while source.read() is not None:
                    apply_new_resets()
            clock.sleep(chunk_samples / utterance.sample_rate)
            samples_sent += chunk_samples
        source.finish()
        apply_new_resets()
        truth_end_wall = utterance_start_perf + utterance.truth_end_sample / utterance.sample_rate
        endpoint = clock.perf_counter()
        diag = engine.diagnostics(endpoint, truth_end_sample=utterance.truth_end_sample)
        endpoint_at = diag.get("endpoint_at")
        endpoint_audio_sample = diag.get("endpoint_audio_sample")
        decode_completed_at = diag.get("decode_completed_at")
        worker_emit_at = diag.get("worker_emit_at")
        controller_at = diag.get("controller_received_at")
        observed_base = clock.perf_counter()
        if controller_at is not None:
            observed_base = max(observed_base, controller_at)
        observed_at = observed_base + 0.002
        # The audio endpoint is an explicit sample-domain diagnostic. It is
        # never inferred from the monotonic endpoint_at wall timestamp.
        endpoint_rel_ms = (endpoint_at - origin) * 1000 if endpoint_at is not None else None
        truth_wall_ms = (truth_end_wall - origin) * 1000
        outcome = utterance.outcome
        latency = (
            (observed_at - truth_end_wall) * 1000
            if outcome in {"recognized", "empty_transcript"}
            else None
        )
        resource_sample_at = clock.perf_counter()
        resource_sample = sampler.sample()
        process_cpu_ms = resource_sample.get("process_cpu_ms")
        rss_current_bytes = resource_sample.get("rss_current_bytes")
        rss_peak_bytes = resource_sample.get("rss_peak_bytes")
        process_cpu_wall_elapsed_ms = max(
            0.0, (resource_sample_at - last_resource_sample_at) * 1000
        )
        process_cpu_percent = (
            100 * process_cpu_ms / process_cpu_wall_elapsed_ms
            if process_cpu_ms is not None and process_cpu_wall_elapsed_ms > 0
            else None
        )
        last_resource_sample_at = resource_sample_at
        row = {
            "utterance_id": utterance.utterance_id,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "repeat": utterance.repeat,
            "pause_after_samples": utterance.pause_after_samples,
            "sample_rate": utterance.sample_rate,
            "audio_duration_samples": utterance.audio_duration_samples,
            "truth_start_sample": utterance.truth_start_sample,
            "truth_end_sample": utterance.truth_end_sample,
            "fixture": utterance.fixture,
            "outcome": outcome,
            "truth_end_wall_ms": truth_wall_ms,
            "endpoint_at_ms": endpoint_rel_ms,
            "decode_completed_at_ms": (
                (decode_completed_at - origin) * 1000 if decode_completed_at is not None else None
            ),
            "worker_emit_at_ms": (
                (worker_emit_at - origin) * 1000 if worker_emit_at is not None else None
            ),
            "controller_received_at_ms": (
                (controller_at - origin) * 1000 if controller_at is not None else None
            ),
            "result_observed_at_ms": (observed_at - origin) * 1000,
            "worker_to_controller_ms": (
                (controller_at - worker_emit_at) * 1000
                if controller_at is not None and worker_emit_at is not None
                else None
            ),
            "controller_to_callback_ms": (
                (observed_at - controller_at) * 1000 if controller_at is not None else None
            ),
            "recognition_ms": diag.get("recognition_ms"),
            "cpu_ms": diag.get("cpu_ms"),
            "process_cpu_ms": process_cpu_ms,
            "process_cpu_percent": process_cpu_percent,
            "process_cpu_interval_wall_ms": process_cpu_wall_elapsed_ms,
            "rss_current_bytes": rss_current_bytes,
            "rss_peak_bytes": rss_peak_bytes,
            "post_vad_ms": (
                (decode_completed_at - endpoint_at) * 1000
                if decode_completed_at is not None and endpoint_at is not None
                else None
            ),
            "endpoint_audio_sample": endpoint_audio_sample,
            "endpoint_audio_at_ms": (
                endpoint_audio_sample * 1000 / utterance.sample_rate
                if endpoint_audio_sample is not None
                else None
            ),
            "vad_endpoint_audio_lag_ms": (
                (endpoint_audio_sample - utterance.truth_end_sample) * 1000 / utterance.sample_rate
                if endpoint_audio_sample is not None
                else None
            ),
            "vad_endpoint_wall_ms": (
                (endpoint_at - truth_end_wall) * 1000 if endpoint_at is not None else None
            ),
            "result_latency_ms": latency,
            "feed_lag_ms": max(
                0.0,
                (
                    endpoint
                    - (
                        utterance_start_perf
                        + utterance.audio_duration_samples / utterance.sample_rate
                    )
                )
                * 1000,
            ),
            "queue_depth_blocks": source.queue_depth,
            "queue_max_depth_blocks": source.queue_max_depth,
            "dropped_blocks": source.dropped_blocks,
            "input_overflow_notifications": source.input_overflows,
            "overflow": source.input_overflows > 0,
            "queue_overflow": source.dropped_blocks > 0,
            "vad_resets_total": source.vad_resets_total,
            "vad_resets_gap": source.vad_resets_gap,
            "vad_resets_overflow": source.vad_resets_overflow,
            "vad_resets_tail": source.vad_resets_tail,
        }
        rows.append(row)
        if utterance.pause_after_samples:
            clock.sleep(utterance.pause_after_samples / utterance.sample_rate)
        audio_schedule_offset += (
            utterance.audio_duration_samples + utterance.pause_after_samples
        ) / utterance.sample_rate
        if outcome in {"timeout", "missing_transcript", "decode_error", "input_error"}:
            # Outcomes remain represented; absent result timestamps/latency are null.
            row["result_observed_at_ms"] = None
            row["controller_received_at_ms"] = None
            row["worker_emit_at_ms"] = None
            row["worker_to_controller_ms"] = None
            row["controller_to_callback_ms"] = None
            if outcome != "decode_error":
                row["decode_completed_at_ms"] = None
                row["recognition_ms"] = None
                row["cpu_ms"] = None
                row["post_vad_ms"] = None
    metric_names = (
        "result_latency_ms",
        "recognition_ms",
        "cpu_ms",
        "post_vad_ms",
        "endpoint_audio_at_ms",
        "vad_endpoint_wall_ms",
        "vad_endpoint_audio_lag_ms",
        "feed_lag_ms",
        "worker_to_controller_ms",
        "controller_to_callback_ms",
        "queue_depth_blocks",
        "queue_max_depth_blocks",
        "dropped_blocks",
        "input_overflow_notifications",
        "vad_resets_total",
        "vad_resets_gap",
        "vad_resets_overflow",
        "vad_resets_tail",
        "process_cpu_ms",
        "process_cpu_percent",
        "process_cpu_interval_wall_ms",
        "rss_current_bytes",
        "rss_peak_bytes",
    )
    metrics = {
        name: summarize(
            row[name]
            for row in rows
            if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
        )
        for name in metric_names
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "mode": "synthetic",
            "run_origin": "monotonic-relative-zero",
            "git_head": _git_head(),
            "baseline_head": BASELINE_HEAD,
            "git_branch": _git_branch(),
            "dirty": _git_dirty(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sample_rate_policy": "per-utterance",
            "block_ms": block_ms,
            "timeout_ms": timeout_ms,
            "max_utterances": max_utterances,
            "max_replay_duration_seconds": max_replay_duration_seconds,
            "planned_replay_duration_seconds": replay_duration,
            "queue_capacity": source.capacity,
            "quantile_rule": QUANTILE_RULE,
            "rss_sampling_policy": (
                "resource.getrusage ru_maxrss once per utterance; process high-water"
            ),
            "rss_limitation": (
                "process-lifetime high-water can predate the run; current RSS is unavailable"
            ),
            "cpu_metric": "process_cpu_ms is a delta between samples; decoder cpu_ms is separate",
            "configuration": {
                "source": "synthetic-only",
                "device_access": False,
                "noise_seed": SYNTHETIC_NOISE_SEED,
                "noise_peak_amplitude": SYNTHETIC_NOISE_AMPLITUDE,
                "pcm_encoding": "float32-little-endian",
                "max_pcm_samples_per_block": MAX_SYNTHETIC_PCM_SAMPLES,
            },
        },
        "summary": {
            "utterance_count": len(rows),
            "recognized_count": sum(r["outcome"] == "recognized" for r in rows),
            "outcomes": {
                outcome: sum(r["outcome"] == outcome for r in rows)
                for outcome in (
                    "recognized",
                    "missing_transcript",
                    "empty_transcript",
                    "timeout",
                    "decode_error",
                    "input_error",
                )
            },
            "metrics": metrics,
            "result_latency_ms": metrics["result_latency_ms"],
            "process_cpu_ms": metrics["process_cpu_ms"],
            "rss_current_bytes": metrics["rss_current_bytes"],
            "rss_peak_bytes": metrics["rss_peak_bytes"],
        },
        "utterances": rows,
    }


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_branch() -> str | None:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


CSV_FIELDS = tuple(Utterance.__dataclass_fields__) + (
    "schema_version",
    "run_id",
    "truth_end_wall_ms",
    "endpoint_at_ms",
    "decode_completed_at_ms",
    "decode_after_endpoint_ms",
    "decode_after_truth_end_ms",
    "worker_emit_at_ms",
    "controller_received_at_ms",
    "result_observed_at_ms",
    "last_result_observed_at_ms",
    "feed_completed_at_ms",
    "worker_to_controller_ms",
    "controller_to_callback_ms",
    "recognition_ms",
    "cpu_ms",
    "process_cpu_ms",
    "process_cpu_percent",
    "process_cpu_interval_wall_ms",
    "rss_current_bytes",
    "rss_peak_bytes",
    "post_vad_ms",
    "endpoint_audio_at_ms",
    "vad_endpoint_audio_lag_ms",
    "vad_endpoint_wall_ms",
    "result_latency_ms",
    "feed_lag_ms",
    "queue_depth_blocks",
    "queue_max_depth_blocks",
    "dropped_blocks",
    "input_overflow_notifications",
    "queue_overflow",
    "vad_resets_total",
    "vad_resets_gap",
    "vad_resets_overflow",
    "vad_resets_tail",
    "endpoint_audio_sample",
    "overflow",
    "dataset_id",
    "audio_sha256",
    "audio_container",
    "audio_encoding",
    "configuration_sha256",
    "model_sha256",
    "vad_model_sha256",
    "result_count",
    "accepted_blocks",
    "processed_blocks",
    "expected_result_signals",
    "observed_result_signals",
    "model_startup_ms",
    "model_startup_process_cpu_ms",
)


def export_results(result: dict[str, Any], output_path: str | Path) -> tuple[Path, Path]:
    """Write JSON and adjacent CSV at explicitly supplied paths."""
    json_path = Path(output_path)
    if not str(output_path):
        raise BenchmarkError("an explicit output path is required")
    csv_path = json_path.with_suffix(".csv")
    if json_path == csv_path:
        raise BenchmarkError("output path must identify the JSON file, not the CSV file")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if json_path.exists() or csv_path.exists():
        raise BenchmarkError("refusing to overwrite an existing JSON or CSV result")

    json_created = False
    csv_created = False
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            json_created = True
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            csv_created = True
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["utterances"])
    except FileExistsError as error:
        if json_created:
            json_path.unlink(missing_ok=True)
        if csv_created:
            csv_path.unlink(missing_ok=True)
        raise BenchmarkError("refusing to overwrite an existing JSON or CSV result") from error
    except Exception:
        if json_created:
            json_path.unlink(missing_ok=True)
        if csv_created:
            csv_path.unlink(missing_ok=True)
        raise
    return json_path, csv_path


def _safe_stt_configuration(config: Any) -> dict[str, Any]:
    """Keep effective scalar settings while omitting paths and device identity."""
    safe_fields = {
        "sample_rate",
        "num_threads",
        "feature_dim",
        "decoding_method",
        "provider",
        "vad_min_silence_duration",
        "vad_min_speech_duration",
        "vad_max_speech_duration",
        "vad_window_size",
        "vad_buffer_size_seconds",
    }
    if hasattr(config, "__dataclass_fields__"):
        values = asdict(config)
    elif isinstance(config, dict):
        values = dict(config)
    else:
        raise BenchmarkError("STT configuration must be a dataclass or mapping")
    safe: dict[str, Any] = {}
    for name, value in values.items():
        if name not in safe_fields:
            continue
        normalized = name.lower()
        if any(word in normalized for word in ("path", "dir", "device", "file")):
            continue
        if isinstance(value, str) and ("/" in value or "\\" in value):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[name] = value
    return safe


def _validate_public_identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise BenchmarkError(
            f"{name} must use letters, digits, dot, underscore or hyphen (max 128 chars)"
        )
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stop_child_process(process: multiprocessing.Process, shutdown_grace_seconds: float) -> bool:
    if not process.is_alive():
        process.join(0)
        return True
    half_grace = shutdown_grace_seconds / 2
    try:
        process.terminate()
    except (OSError, ValueError):
        pass
    process.join(half_grace)
    if process.is_alive():
        try:
            process.kill()
        except (AttributeError, OSError, ValueError):
            pass
        process.join(shutdown_grace_seconds - half_grace)
    return not process.is_alive()


def run_bounded_child(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    deadline_seconds: float,
    shutdown_grace_seconds: float = REAL_SHUTDOWN_GRACE_SECONDS,
) -> dict[str, Any]:
    """Run a child with a spawn context and bounded terminate/kill/join cleanup."""
    if (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(deadline_seconds)
        or not 0 < deadline_seconds <= 12 * 3600
    ):
        raise BenchmarkError("child deadline must be in (0, 43200] seconds")
    if (
        isinstance(shutdown_grace_seconds, bool)
        or not isinstance(shutdown_grace_seconds, (int, float))
        or not math.isfinite(shutdown_grace_seconds)
        or not 0.1 <= shutdown_grace_seconds <= 30
    ):
        raise BenchmarkError("child shutdown grace must be in [0.1, 30] seconds")
    context = multiprocessing.get_context("spawn")
    receive_pipe, send_pipe = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(send_pipe, *args), daemon=False)
    started_at = time.monotonic()
    started = False
    reader_started = False
    reader_done = threading.Event()
    received: dict[str, Any] = {}

    def receive_payload() -> None:
        try:
            received["payload"] = receive_pipe.recv()
        except EOFError:
            received["payload"] = None
        except BaseException as error:
            received["error"] = error
        finally:
            reader_done.set()

    def start_payload_reader() -> None:
        nonlocal reader_started
        if reader_started:
            return
        reader_started = True
        threading.Thread(target=receive_payload, name="stt-child-result", daemon=True).start()

    try:
        process.start()
        started = True
        send_pipe.close()
        deadline = started_at + deadline_seconds
        payload: Any = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stopped = _stop_child_process(process, shutdown_grace_seconds)
                if not stopped:
                    raise BenchmarkError("timed-out real-mode child remained alive after kill")
                return {
                    "timed_out": True,
                    "payload": None,
                    "exitcode": process.exitcode,
                    "pid": process.pid,
                    "child_alive": process.is_alive(),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
            ready = multiprocessing.connection.wait(
                [receive_pipe, process.sentinel], timeout=min(remaining, 0.1)
            )
            if receive_pipe in ready and not reader_started:
                start_payload_reader()
            if process.sentinel in ready and not reader_started:
                if receive_pipe.poll():
                    start_payload_reader()
                else:
                    process.join(0)
                    return {
                        "timed_out": False,
                        "payload": None,
                        "exitcode": process.exitcode,
                        "pid": process.pid,
                        "child_alive": process.is_alive(),
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
            if reader_started and reader_done.is_set():
                payload = received.get("payload")
                cleanup_deadline = time.monotonic() + shutdown_grace_seconds
                process.join(shutdown_grace_seconds)
                if process.is_alive():
                    remaining_cleanup = max(0.0, cleanup_deadline - time.monotonic())
                    if not _stop_child_process(process, remaining_cleanup):
                        raise BenchmarkError("real-mode child did not stop after returning")
                    return {
                        "timed_out": True,
                        "payload": None,
                        "exitcode": process.exitcode,
                        "pid": process.pid,
                        "child_alive": process.is_alive(),
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                return {
                    "timed_out": False,
                    "payload": payload if "error" not in received else None,
                    "exitcode": process.exitcode,
                    "pid": process.pid,
                    "child_alive": process.is_alive(),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
    finally:
        receive_pipe.close()
        try:
            send_pipe.close()
        except OSError:
            pass
        if started and process.is_alive():
            if not _stop_child_process(process, shutdown_grace_seconds):
                raise BenchmarkError("child cleanup failed after terminate and kill")
        if started and not process.is_alive():
            process.close()


def _check_real_execution_authorization(token: str | None) -> None:
    if token != REAL_MODEL_OPT_IN:
        raise BenchmarkError(
            "real-model disabled: pass --allow-real-model "
            "I_AUTHORIZE_REAL_MODEL_EXECUTION after separate explicit authorization"
        )


def run_file_backed_clip(
    audio_path: str | Path,
    utterance: Utterance,
    *,
    dataset_id: str,
    config: Any,
    test_engine_factory: Callable[[], Any] | None = None,
    engine_factory: Callable[[], Any] | None = None,
    authorization_token: str | None = None,
    model_sha256: str | None = None,
    vad_model_sha256: str | None = None,
    raw_metadata: Any | None = None,
    sampler: ResourceSampler | None = None,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    result_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Run one file-backed clip through SttController and build schema-1 output.

    Tests use ``test_engine_factory`` and never cross the model gate. Production
    factories require the explicit token and are never constructed by default.
    """
    dataset_id = _validate_public_identifier(dataset_id, "dataset_id")
    utterance.validate()
    _validate_public_identifier(utterance.utterance_id, "utterance_id")
    if (test_engine_factory is None) == (engine_factory is None):
        raise BenchmarkError("provide exactly one of test_engine_factory or engine_factory")
    if engine_factory is not None:
        _check_real_execution_authorization(authorization_token)
        if (
            multiprocessing.parent_process() is None
            or not _REAL_MODE_CHILD_ACTIVE
        ):
            raise BenchmarkError(
                "real engine factories are available only inside the isolated real-mode child"
            )
        if not _valid_sha256(model_sha256) or not _valid_sha256(vad_model_sha256):
            raise BenchmarkError("authorized real runs require model and VAD SHA-256 fingerprints")
    elif authorization_token is not None:
        raise BenchmarkError("test engine factory cannot receive a real authorization token")
    if (model_sha256 is None) != (vad_model_sha256 is None):
        raise BenchmarkError("model and VAD fingerprints must both be present or both be null")
    if model_sha256 is not None and (
        not _valid_sha256(model_sha256) or not _valid_sha256(vad_model_sha256)
    ):
        raise BenchmarkError("model fingerprints must be lowercase SHA-256 hex strings")
    if type(queue_capacity) is not int or not 1 <= queue_capacity <= MAX_QUEUE_CAPACITY:
        raise BenchmarkError(f"queue capacity must be between 1 and {MAX_QUEUE_CAPACITY}")
    if (
        isinstance(result_timeout_seconds, bool)
        or not isinstance(result_timeout_seconds, (int, float))
        or not math.isfinite(result_timeout_seconds)
        or not 0 < result_timeout_seconds <= MAX_REAL_RESULT_TIMEOUT_SECONDS
    ):
        raise BenchmarkError(
            f"result_timeout_seconds must be in (0, {MAX_REAL_RESULT_TIMEOUT_SECONDS}]")

    from stt_real_adapter import (
        Pcm16Reader,
        ReplayError,
        ReplaySource,
        build_timing_metrics,
        run_controller_replay,
        validate_labels,
    )

    from openscreen_ai.stt.worker import SttController

    try:
        reader = Pcm16Reader(audio_path, raw_metadata=raw_metadata)
    except (OSError, ReplayError) as error:
        raise BenchmarkError(f"invalid file-backed audio: {error}") from error
    metadata = reader.metadata
    try:
        if utterance.sample_rate != metadata.sample_rate:
            raise BenchmarkError("utterance sample rate does not match PCM metadata")
        if utterance.audio_duration_samples not in (0, metadata.sample_count):
            raise BenchmarkError("utterance audio duration does not match PCM sample count")
        validate_labels(
            utterance.truth_start_sample,
            utterance.truth_end_sample,
            metadata.sample_count,
        )
        effective_utterance = Utterance(
            **{
                **asdict(utterance),
                "audio_duration_samples": metadata.sample_count,
            }
        )
        source = ReplaySource(reader, queue_capacity=queue_capacity)
        effective_engine_factory = (
            test_engine_factory if test_engine_factory is not None else engine_factory
        )
        progress = _EngineProgress()

        def tracking_engine_factory() -> _ProgressTrackingEngine:
            return _ProgressTrackingEngine(effective_engine_factory, progress)

        controller = SttController(
            config,
            microphone_factory=lambda: source,
            engine_factory=tracking_engine_factory,
        )
        sampler = sampler or SystemResourceSampler()
        sampler.sample()  # establish process CPU baseline before model startup
        run_started_at = time.perf_counter()
        listening_resource: dict[str, int | None] | None = None

        def sample_when_listening() -> None:
            nonlocal listening_resource
            listening_resource = sampler.sample()

        observation = run_controller_replay(
            controller,
            source,
            progress=progress,
            result_timeout_seconds=result_timeout_seconds,
            on_listening=sample_when_listening,
        )
        reader.close()
        finished_at = time.perf_counter()
        final_resource = sampler.sample()
    except Exception:
        reader.close()
        raise

    diagnostic = observation.get("diagnostic") or {}
    result_diagnostic = observation.get("result_diagnostic") or diagnostic
    result_fields = observation.get("result_fields") or {}
    timeline = build_timing_metrics(
        replay_origin_perf=source.origin_perf,
        sample_rate=metadata.sample_rate,
        truth_end_sample=effective_utterance.truth_end_sample,
        endpoint_at=result_diagnostic.get("endpoint_at") or result_fields.get("endpoint_at"),
        endpoint_audio_at=result_fields.get("endpoint_audio_at"),
        decode_completed_at=(
            result_diagnostic.get("decode_completed_at")
            or result_fields.get("decode_completed_at")
        ),
        worker_emit_at=result_diagnostic.get("worker_emit_at"),
        controller_received_at=result_diagnostic.get("controller_received_at"),
        result_observed_at=observation.get("result_observed_at"),
        feed_completed_at=source.feed_completed_at,
        audio_sample_count=(
            metadata.sample_count if source.feed_completed_at is not None else None
        ),
    )
    run_id = uuid.uuid4().hex
    safe_configuration = _safe_stt_configuration(config)
    configuration_sha256 = _sha256_json(safe_configuration)
    manifest_identity = {
        "dataset_id": dataset_id,
        "utterance": {
            name: getattr(effective_utterance, name)
            for name in (
                "utterance_id", "sample_rate", "audio_duration_samples",
                "truth_start_sample", "truth_end_sample", "repeat",
            )
        },
        "audio_sha256": metadata.sha256,
        "configuration_sha256": configuration_sha256,
        "model_sha256": model_sha256,
        "vad_model_sha256": vad_model_sha256,
    }
    manifest_sha256 = _sha256_json(manifest_identity)
    process_cpu_ms = final_resource.get("process_cpu_ms")
    process_wall_start = observation.get("controller_listening_at") or run_started_at
    process_wall_ms = max(0.0, (finished_at - process_wall_start) * 1000)
    process_cpu_percent = (
        100 * process_cpu_ms / process_wall_ms
        if process_cpu_ms is not None and process_wall_ms > 0
        else None
    )
    endpoint_audio_sample = timeline["endpoint_audio_sample"]
    row: dict[str, Any] = {
        **asdict(effective_utterance),
        "fixture": "file-backed",
        "outcome": observation["result_outcome"],
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        **timeline,
        "recognition_ms": result_fields.get("recognition_ms"),
        "cpu_ms": result_fields.get("cpu_ms"),
        "process_cpu_ms": process_cpu_ms,
        "process_cpu_percent": process_cpu_percent,
        "process_cpu_interval_wall_ms": process_wall_ms,
        "rss_current_bytes": final_resource.get("rss_current_bytes"),
        "rss_peak_bytes": final_resource.get("rss_peak_bytes"),
        "post_vad_ms": timeline["decode_after_endpoint_ms"],
        "endpoint_audio_at_ms": (
            endpoint_audio_sample * 1000 / metadata.sample_rate
            if endpoint_audio_sample is not None
            else None
        ),
        "queue_depth_blocks": observation.get("queue_depth_at_stop"),
        "queue_max_depth_blocks": source.queue_max_depth,
        "dropped_blocks": source.dropped_blocks,
        "input_overflow_notifications": source.input_overflows,
        "queue_overflow": source.dropped_blocks > 0,
        "overflow": source.input_overflows > 0,
        "vad_resets_total": diagnostic.get("vad_resets_total", 0),
        "vad_resets_gap": diagnostic.get("vad_resets_gap", 0),
        "vad_resets_overflow": diagnostic.get("vad_resets_overflow", 0),
        "vad_resets_tail": diagnostic.get("vad_resets_tail", 0),
        "dataset_id": dataset_id,
        "audio_sha256": metadata.sha256,
        "audio_container": metadata.container,
        "audio_encoding": metadata.encoding,
        "configuration_sha256": configuration_sha256,
        "model_sha256": model_sha256,
        "vad_model_sha256": vad_model_sha256,
        "result_count": observation["result_count"],
        "last_result_observed_at_ms": (
            (observation["last_result_observed_at"] - source.origin_perf) * 1000
            if observation.get("last_result_observed_at") is not None
            and source.origin_perf is not None
            else None
        ),
        "accepted_blocks": observation["accepted_blocks"],
        "processed_blocks": observation["processed_blocks"],
        "expected_result_signals": observation["expected_result_signals"],
        "observed_result_signals": observation["observed_result_signals"],
        "worker_to_controller_ms": (
            (result_diagnostic["controller_received_at"] - result_diagnostic["worker_emit_at"])
            * 1000
            if result_diagnostic.get("controller_received_at") is not None
            and result_diagnostic.get("worker_emit_at") is not None
            else None
        ),
        "controller_to_callback_ms": (
            (observation["result_observed_at"] - result_diagnostic["controller_received_at"])
            * 1000
            if observation.get("result_observed_at") is not None
            and result_diagnostic.get("controller_received_at") is not None
            else None
        ),
        "model_startup_ms": (
            (observation["controller_listening_at"] - observation["controller_started_at"])
            * 1000
            if observation.get("controller_listening_at") is not None
            else None
        ),
        "model_startup_process_cpu_ms": (
            listening_resource.get("process_cpu_ms") if listening_resource else None
        ),
    }
    metric_names = (
        "result_latency_ms", "recognition_ms", "cpu_ms", "process_cpu_ms",
        "process_cpu_percent", "process_cpu_interval_wall_ms", "rss_current_bytes",
        "rss_peak_bytes", "post_vad_ms",
        "vad_endpoint_wall_ms", "vad_endpoint_audio_lag_ms", "feed_lag_ms",
        "queue_depth_blocks", "queue_max_depth_blocks", "dropped_blocks",
        "input_overflow_notifications", "vad_resets_total", "vad_resets_gap",
        "vad_resets_overflow", "vad_resets_tail", "model_startup_ms",
        "model_startup_process_cpu_ms",
    )
    metrics = {
        name: summarize(
            value for row_value in [row]
            if (value := row_value.get(name)) is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        for name in metric_names
    }
    outcome = row["outcome"]
    outcomes = {
        name: int(outcome == name)
        for name in (
            "recognized", "missing_transcript", "empty_transcript", "timeout",
            "decode_error", "input_error",
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "mode": "real-file",
            "run_origin": "monotonic-relative-zero",
            "source": "file-backed-replay",
            "device_access": False,
            "git_head": _git_head(),
            "baseline_head": BASELINE_HEAD,
            "git_branch": _git_branch(),
            "dirty": _git_dirty(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dataset_id": dataset_id,
            "manifest_sha256": manifest_sha256,
            "audio_sha256": metadata.sha256,
            "audio_container": metadata.container,
            "audio_encoding": metadata.encoding,
            "sample_rate": metadata.sample_rate,
            "channels": metadata.channels,
            "audio_duration_samples": metadata.sample_count,
            "queue_capacity": queue_capacity,
            "block_samples": source.block_samples,
            "result_timeout_seconds": result_timeout_seconds,
            "configuration": safe_configuration,
            "configuration_sha256": configuration_sha256,
            "model_sha256": model_sha256,
            "vad_model_sha256": vad_model_sha256,
            "rss_sampling_policy": (
                "resource.getrusage ru_maxrss; process high-water may predate the run"
            ),
            "cpu_metric": (
                "process_cpu_ms is a post-start delta; decoder cpu_ms is separate"
            ),
        },
        "summary": {
            "utterance_count": 1,
            "recognized_count": int(outcome == "recognized"),
            "outcomes": outcomes,
            "metrics": metrics,
            "result_latency_ms": metrics["result_latency_ms"],
            "process_cpu_ms": metrics["process_cpu_ms"],
            "rss_current_bytes": metrics["rss_current_bytes"],
            "rss_peak_bytes": metrics["rss_peak_bytes"],
        },
        "utterances": [row],
    }


def _summarize_file_results(results: list[dict[str, Any]], manifest_sha256: str) -> dict[str, Any]:
    if not results:
        raise BenchmarkError("real-file manifest produced no clip runs")
    run_id = uuid.uuid4().hex
    rows = [result["utterances"][0] for result in results]
    for row in rows:
        row["run_id"] = run_id
    run = dict(results[0]["run"])
    run["run_id"] = run_id
    run["manifest_sha256"] = manifest_sha256
    for name in (
        "audio_sha256", "audio_container", "audio_encoding", "sample_rate", "channels",
        "audio_duration_samples", "model_sha256", "vad_model_sha256",
    ):
        if len({row.get(name) for row in rows}) > 1:
            run.pop(name, None)
    run["planned_replay_duration_seconds"] = sum(
        row["audio_duration_samples"] / row["sample_rate"] for row in rows
    )
    metric_names = (
        "result_latency_ms", "recognition_ms", "cpu_ms", "process_cpu_ms",
        "process_cpu_percent", "process_cpu_interval_wall_ms", "rss_current_bytes",
        "rss_peak_bytes", "post_vad_ms",
        "vad_endpoint_wall_ms", "vad_endpoint_audio_lag_ms", "feed_lag_ms",
        "queue_depth_blocks", "queue_max_depth_blocks", "dropped_blocks",
        "input_overflow_notifications", "vad_resets_total", "vad_resets_gap",
        "vad_resets_overflow", "vad_resets_tail", "model_startup_ms",
        "model_startup_process_cpu_ms",
    )
    metrics = {
        name: summarize(
            value for row in rows
            if (value := row.get(name)) is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        for name in metric_names
    }
    outcome_names = (
        "recognized", "missing_transcript", "empty_transcript", "timeout",
        "decode_error", "input_error",
    )
    outcomes = {
        name: sum(row["outcome"] == name for row in rows) for name in outcome_names
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "summary": {
            "utterance_count": len(rows),
            "recognized_count": outcomes["recognized"],
            "outcomes": outcomes,
            "metrics": metrics,
            "result_latency_ms": metrics["result_latency_ms"],
            "process_cpu_ms": metrics["process_cpu_ms"],
            "rss_current_bytes": metrics["rss_current_bytes"],
            "rss_peak_bytes": metrics["rss_peak_bytes"],
        },
        "utterances": rows,
    }


def _prepare_real_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Validate bounded input metadata before starting the isolated model child."""
    _check_real_execution_authorization(args.allow_real_model)
    if args.model_dir is None or args.vad_model_path is None:
        raise BenchmarkError("real-model mode requires --model-dir and --vad-model-path")
    if type(args.queue_capacity) is not int or not 1 <= args.queue_capacity <= MAX_QUEUE_CAPACITY:
        raise BenchmarkError(f"queue capacity must be between 1 and {MAX_QUEUE_CAPACITY}")
    if (
        isinstance(args.result_timeout_seconds, bool)
        or not isinstance(args.result_timeout_seconds, (int, float))
        or not math.isfinite(args.result_timeout_seconds)
        or not 0 < args.result_timeout_seconds <= MAX_REAL_RESULT_TIMEOUT_SECONDS
    ):
        raise BenchmarkError(
            f"result_timeout_seconds must be in (0, {MAX_REAL_RESULT_TIMEOUT_SECONDS}]")
    try:
        if args.manifest.stat().st_size > MAX_REAL_MANIFEST_BYTES:
            raise BenchmarkError("real-file manifest exceeds the 16 MiB input limit")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except BenchmarkError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read real-file manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise BenchmarkError("real-file manifest must be an object")
    dataset_id = manifest.get("dataset_id")
    raw_items = manifest.get("utterances")
    dataset_id = _validate_public_identifier(dataset_id, "dataset_id")
    if not isinstance(raw_items, list) or not raw_items:
        raise BenchmarkError("real-file manifest requires a non-empty utterances list")
    if len(raw_items) > MAX_REAL_CLIPS:
        raise BenchmarkError("real-file manifest exceeds the configured clip limit")

    from stt_real_adapter import Pcm16Reader, RawPcmMetadata, ReplayError, validate_labels

    allowed = {
        "utterance_id", "audio_path", "audio_duration_samples", "sample_rate",
        "truth_start_sample", "truth_end_sample", "repeat", "raw_metadata",
    }
    entries: list[dict[str, Any]] = []
    total_expanded = 0
    total_duration = 0.0
    total_audio_bytes = 0
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise BenchmarkError(f"real-file utterance {index} must be an object")
        unknown = set(raw_item) - allowed
        if unknown:
            raise BenchmarkError(
                f"unsupported real-file manifest fields: {', '.join(sorted(unknown))}"
            )
        path_value = raw_item.get("audio_path")
        utterance_id = raw_item.get("utterance_id")
        if not isinstance(path_value, str) or not path_value.strip():
            raise BenchmarkError(f"real-file utterance {index} requires audio_path")
        utterance_id = _validate_public_identifier(
            utterance_id, f"real-file utterance {index} utterance_id"
        )
        audio_path = Path(path_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = args.manifest.parent / audio_path
        try:
            total_audio_bytes += audio_path.stat().st_size
        except OSError as error:
            raise BenchmarkError(
                f"real-file audio is unavailable: {type(error).__name__}"
            ) from error
        if total_audio_bytes > MAX_REAL_INPUT_BYTES:
            raise BenchmarkError("real-file inputs exceed the aggregate 2 GiB byte limit")
        raw_metadata_value = raw_item.get("raw_metadata")
        try:
            raw_metadata = (
                RawPcmMetadata(**raw_metadata_value)
                if raw_metadata_value is not None
                else None
            )
            with Pcm16Reader(audio_path, raw_metadata=raw_metadata) as reader:
                metadata = reader.metadata
        except (OSError, TypeError, ReplayError) as error:
            raise BenchmarkError(f"invalid audio in real-file manifest: {error}") from error
        sample_rate = raw_item.get("sample_rate", metadata.sample_rate)
        duration = raw_item.get("audio_duration_samples", metadata.sample_count)
        truth_start = raw_item.get("truth_start_sample")
        truth_end = raw_item.get("truth_end_sample")
        repeat = raw_item.get("repeat", 1)
        if sample_rate != metadata.sample_rate or duration != metadata.sample_count:
            raise BenchmarkError("manifest sample rate/duration does not match audio metadata")
        try:
            validate_labels(truth_start, truth_end, metadata.sample_count)
        except ReplayError as error:
            raise BenchmarkError(str(error)) from error
        utterance = Utterance(
            utterance_id=utterance_id,
            sample_rate=sample_rate,
            audio_duration_samples=duration,
            truth_start_sample=truth_start,
            truth_end_sample=truth_end,
            repeat=repeat,
        )
        utterance.validate()
        total_expanded += repeat
        if total_expanded > MAX_REAL_CLIPS:
            raise BenchmarkError("real-file manifest exceeds the expanded clip limit")
        total_duration += duration / sample_rate * repeat
        entries.append(
            {
                "audio_path": audio_path,
                "raw_metadata": raw_metadata,
                "metadata": metadata,
                "utterance": utterance,
            }
        )
    validate_replay_duration(
        [entry["utterance"] for entry in entries],
        max_replay_duration_seconds=args.max_replay_seconds,
    )
    if total_duration > args.max_replay_seconds:
        raise BenchmarkError("expanded real-file replay exceeds the configured duration cap")

    return {
        "dataset_id": dataset_id,
        "entries": entries,
        "total_duration": total_duration,
        "total_expanded": total_expanded,
    }


def _run_real_manifest_child(
    args: argparse.Namespace,
    prepared: dict[str, Any],
    model_sha256: str,
    vad_model_sha256: str,
) -> dict[str, Any]:
    """Construct the real engine and run clips only inside the authorized child."""
    from openscreen_ai.stt.config import SttConfig
    from openscreen_ai.stt.engine import SttEngine

    dataset_id = prepared["dataset_id"]
    entries = prepared["entries"]
    config = SttConfig(
        model_dir=args.model_dir,
        vad_model_path=args.vad_model_path,
        provider=args.provider,
        num_threads=args.num_threads,
    )
    sampler = SystemResourceSampler()
    clip_results: list[dict[str, Any]] = []
    manifest_descriptors: list[dict[str, Any]] = []
    for entry in entries:
        base = entry["utterance"]
        for repeat_index in range(1, base.repeat + 1):
            current = Utterance(
                **{
                    **asdict(base),
                    "utterance_id": (
                        f"{base.utterance_id}-r{repeat_index}" if base.repeat > 1
                        else base.utterance_id
                    ),
                    "repeat": repeat_index,
                }
            )
            result = run_file_backed_clip(
                entry["audio_path"],
                current,
                dataset_id=dataset_id,
                config=config,
                engine_factory=lambda: SttEngine(config),
                authorization_token=args.allow_real_model,
                model_sha256=model_sha256,
                vad_model_sha256=vad_model_sha256,
                raw_metadata=entry["raw_metadata"],
                sampler=sampler,
                queue_capacity=args.queue_capacity,
                result_timeout_seconds=args.result_timeout_seconds,
            )
            clip_results.append(result)
            row = result["utterances"][0]
            manifest_descriptors.append(
                {
                    "utterance_id": row["utterance_id"],
                    "repeat": row["repeat"],
                    "audio_sha256": row["audio_sha256"],
                    "configuration_sha256": row["configuration_sha256"],
                    "model_sha256": row["model_sha256"],
                    "vad_model_sha256": row["vad_model_sha256"],
                    "sample_rate": row["sample_rate"],
                    "audio_duration_samples": row["audio_duration_samples"],
                    "truth_start_sample": row["truth_start_sample"],
                    "truth_end_sample": row["truth_end_sample"],
                }
            )
    return _summarize_file_results(clip_results, _sha256_json(manifest_descriptors))


def _real_mode_child(
    result_pipe: Any,
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> None:
    """Check authorization before model/VAD reads or real engine construction."""
    global _REAL_MODE_CHILD_ACTIVE
    try:
        _check_real_execution_authorization(args.allow_real_model)
        if multiprocessing.parent_process() is None:
            raise BenchmarkError("real-mode execution requires an isolated spawned child")
        _REAL_MODE_CHILD_ACTIVE = True
        from stt_real_adapter import fingerprint_file, fingerprint_model_directory

        model_sha256 = fingerprint_model_directory(args.model_dir)
        vad_model_sha256 = fingerprint_file(args.vad_model_path)
        result = _run_real_manifest_child(
            args, prepared, model_sha256, vad_model_sha256
        )
        result_pipe.send({"ok": True, "result": result})
    except BaseException as error:
        # Keep exceptions, paths, transcripts and model details out of artifacts
        # and IPC diagnostics. The parent receives only the exception class.
        try:
            result_pipe.send({"ok": False, "error_type": type(error).__name__})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        _REAL_MODE_CHILD_ACTIVE = False
        result_pipe.close()


def _real_child_deadline_seconds(prepared: dict[str, Any], result_timeout: float) -> float:
    clip_count = prepared["total_expanded"]
    deadline = (
        prepared["total_duration"]
        + clip_count * (REAL_STARTUP_GRACE_SECONDS + result_timeout)
        + REAL_FINGERPRINT_GRACE_SECONDS
    )
    if not math.isfinite(deadline) or not 0 < deadline <= 12 * 3600:
        raise BenchmarkError("validated real-mode hard deadline is outside the supported bound")
    return deadline


def _run_real_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Run the authorized, prevalidated manifest in a killable child process."""
    _check_real_execution_authorization(args.allow_real_model)
    prepared = _prepare_real_manifest(args)
    child = run_bounded_child(
        _real_mode_child,
        (args, prepared),
        deadline_seconds=_real_child_deadline_seconds(
            prepared, args.result_timeout_seconds
        ),
        shutdown_grace_seconds=REAL_SHUTDOWN_GRACE_SECONDS,
    )
    if child["timed_out"]:
        raise BenchmarkError(
            "real-mode hard deadline expired; child was terminated/killed; no artifact was written"
        )
    payload = child["payload"]
    if child["exitcode"] != 0 or not isinstance(payload, dict):
        raise BenchmarkError("real-mode child exited without a safe result payload")
    if not payload.get("ok"):
        error_type = payload.get("error_type")
        safe_type = error_type if isinstance(error_type, str) else "unknown"
        raise BenchmarkError(
            f"real-mode child failed safely ({safe_type}); no artifact was written"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise BenchmarkError("real-mode child returned an invalid result; no artifact was written")
    return result


COMPARISON_METRICS = (
    "result_latency_ms",
    "recognition_ms",
    "cpu_ms",
    "process_cpu_ms",
    "process_cpu_percent",
    "rss_current_bytes",
    "rss_peak_bytes",
)
COMPARISON_IDENTITY_FIELDS = {
    "run_id",
    "git_head",
    "baseline_head",
    "git_branch",
    "dirty",
}
PAIRED_UTTERANCE_FIELDS = (
    "utterance_id",
    "repeat",
    "sample_rate",
    "audio_duration_samples",
    "truth_start_sample",
    "truth_end_sample",
    "pause_after_samples",
    "fixture",
)


def _paired_value_delta(baseline_value: Any, dev001_value: Any) -> dict[str, Any]:
    values = {"baseline": baseline_value, "dev001": dev001_value}
    if (
        not isinstance(baseline_value, (int, float))
        or isinstance(baseline_value, bool)
        or not isinstance(dev001_value, (int, float))
        or isinstance(dev001_value, bool)
    ):
        return {**values, "absolute_delta": None, "percent_delta": None}
    absolute_delta = dev001_value - baseline_value
    percent_delta = absolute_delta * 100 / baseline_value if baseline_value != 0 else None
    return {**values, "absolute_delta": absolute_delta, "percent_delta": percent_delta}


def compare_artifacts(
    baseline: dict[str, Any] | str | Path,
    dev001: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Compare paired JSON results offline; refuse non-identical run conditions."""
    baseline = _read_result_artifact(baseline)
    dev001 = _read_result_artifact(dev001)
    if (
        baseline.get("schema_version") != SCHEMA_VERSION
        or dev001.get("schema_version") != SCHEMA_VERSION
    ):
        raise BenchmarkError("comparison requires two schema-version-1 result artifacts")
    baseline_run = baseline.get("run")
    dev001_run = dev001.get("run")
    if not isinstance(baseline_run, dict) or not isinstance(dev001_run, dict):
        raise BenchmarkError("both artifacts must include run metadata")
    normalized_baseline = {
        key: value for key, value in baseline_run.items() if key not in COMPARISON_IDENTITY_FIELDS
    }
    normalized_dev001 = {
        key: value for key, value in dev001_run.items() if key not in COMPARISON_IDENTITY_FIELDS
    }
    if normalized_baseline != normalized_dev001:
        raise BenchmarkError("comparison refused: experiment configurations/environments differ")
    baseline_rows = baseline.get("utterances")
    dev001_rows = dev001.get("utterances")
    if not isinstance(baseline_rows, list) or not isinstance(dev001_rows, list):
        raise BenchmarkError("both artifacts must include utterance rows")
    if len(baseline_rows) != len(dev001_rows):
        raise BenchmarkError("comparison refused: utterance counts differ")
    if len(baseline_rows) > MAX_CONFIGURED_UTTERANCES:
        raise BenchmarkError("comparison exceeds the maximum paired utterance bound")

    pairs: list[dict[str, Any]] = []
    values_by_metric: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("baseline", "dev001", "absolute_delta", "percent_delta")}
        for name in COMPARISON_METRICS
    }
    missing_by_metric = dict.fromkeys(COMPARISON_METRICS, 0)
    seen_ids: set[str] = set()
    for base_row, dev_row in zip(baseline_rows, dev001_rows, strict=True):
        if not isinstance(base_row, dict) or not isinstance(dev_row, dict):
            raise BenchmarkError("utterance rows must be JSON objects")
        if any(base_row.get(field) != dev_row.get(field) for field in PAIRED_UTTERANCE_FIELDS):
            raise BenchmarkError("comparison refused: paired utterance scenarios differ")
        utterance_id = base_row.get("utterance_id")
        if not isinstance(utterance_id, str) or utterance_id in seen_ids:
            raise BenchmarkError("comparison refused: utterance IDs must be unique strings")
        seen_ids.add(utterance_id)
        metric_pairs: dict[str, Any] = {}
        for name in COMPARISON_METRICS:
            pair = _paired_value_delta(base_row.get(name), dev_row.get(name))
            metric_pairs[name] = pair
            baseline_value = pair["baseline"]
            dev001_value = pair["dev001"]
            if (
                isinstance(baseline_value, (int, float))
                and not isinstance(baseline_value, bool)
                and isinstance(dev001_value, (int, float))
                and not isinstance(dev001_value, bool)
            ):
                for key in ("baseline", "dev001", "absolute_delta"):
                    values_by_metric[name][key].append(pair[key])
                if pair["percent_delta"] is not None:
                    values_by_metric[name]["percent_delta"].append(pair["percent_delta"])
            else:
                missing_by_metric[name] += 1
        pairs.append(
            {
                "utterance_id": utterance_id,
                "baseline_outcome": base_row.get("outcome"),
                "dev001_outcome": dev_row.get("outcome"),
                "metrics": metric_pairs,
            }
        )

    summary = {
        name: {
            "paired_sample_count": len(values["absolute_delta"]),
            "missing_pair_count": missing_by_metric[name],
            **{key: summarize(samples) for key, samples in values.items()},
        }
        for name, values in values_by_metric.items()
    }
    return {
        "comparison_schema_version": 1,
        "method": "paired_dev001_minus_baseline",
        "baseline_git_head": baseline_run.get("git_head"),
        "dev001_git_head": dev001_run.get("git_head"),
        "pair_count": len(pairs),
        "metrics": summary,
        "pairs": pairs,
    }


def _read_result_artifact(artifact: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(artifact, dict):
        return artifact
    if isinstance(artifact, (str, Path)):
        try:
            result = json.loads(Path(artifact).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"cannot read JSON benchmark artifact: {error}") from error
        if isinstance(result, dict):
            return result
    raise BenchmarkError("benchmark artifact must be a JSON object or path to one")


def require_real_opt_in(token: str | None, factory: Callable[[], Any] | None = None) -> None:
    """Fail before factory invocation unless exact conspicuous opt-in is present."""
    if token != REAL_MODEL_OPT_IN:
        raise BenchmarkError(
            "real-model disabled: pass --allow-real-model "
            "I_AUTHORIZE_REAL_MODEL_EXECUTION; separate user authorization is required"
        )
    if factory is None:
        raise BenchmarkError(
            "real-model adapter is unavailable without a factory; no model was constructed"
        )
    factory()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real-model"), default="synthetic")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate synthetic manifest without replay"
    )
    parser.add_argument(
        "--max-replay-seconds",
        type=float,
        default=DEFAULT_MAX_REPLAY_DURATION_SECONDS,
        help="maximum planned replay duration (default: 1800 seconds)",
    )
    parser.add_argument("--allow-real-model", metavar="EXACT_CONFIRMATION")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--vad-model-path", type=Path)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=DEFAULT_QUEUE_CAPACITY)
    parser.add_argument("--result-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.mode == "real-model":
        try:
            _check_real_execution_authorization(args.allow_real_model)
            if args.dry_run:
                raise BenchmarkError("--dry-run is supported only for synthetic mode")
            if args.output.suffix.lower() != ".json":
                raise BenchmarkError("real-model output path must end in .json")
            csv_output = args.output.with_suffix(".csv")
            if args.output.exists() or csv_output.exists():
                raise BenchmarkError("refusing to overwrite an existing JSON or CSV result")
            result = _run_real_manifest(args)
            export_results(result, args.output)
        except (OSError, ValueError, BenchmarkError) as error:
            parser.error(str(error))
        return 0
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        items = validate_manifest(manifest)
        validate_replay_duration(items, args.max_replay_seconds)
        if args.dry_run:
            print(json.dumps({"valid": True, "mode": args.mode, "utterance_count": len(items)}))
            return 0
        export_results(
            run_synthetic(items, max_replay_duration_seconds=args.max_replay_seconds), args.output
        )
    except (OSError, json.JSONDecodeError, BenchmarkError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
