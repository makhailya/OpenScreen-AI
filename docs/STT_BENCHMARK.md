# STT benchmark infrastructure (schema 1)

## File-backed real STT adapter (DEV-003)

`tools/stt_real_adapter.py` provides a streaming reader for uncompressed
PCM16LE mono 16 kHz WAV and raw `s16le` PCM. Raw PCM must be accompanied by
explicit encoding, sample rate, channel count, and exact sample count metadata.
The reader validates the container and declared lengths before use, hashes the
file in bounded reads, and converts no more than 4,096 samples at a time to the
float32 input expected by `SttEngine`. Replay defaults to 640 samples (40 ms)
per block. It rejects metadata/unknown WAV chunks, other rates/channels, and does
not resample or downmix. Limits are 30 minutes per clip, 2 GiB aggregate audio
input and 16 MiB for the manifest; the replay queue defaults to 20 blocks
(maximum configurable queue capacity: 1,000).

Speech truth boundaries stay as integer sample indices. `sample_index_to_perf`
maps an index to the monotonic replay clock using `replay_origin_perf +
sample_index / sample_rate`; audio indices are never subtracted from runtime
timestamps. `build_timing_metrics` keeps sample-domain endpoint error separate
from wall end-of-speech, decode, feed-lag, and observed-result latency; it
rejects out-of-order runtime events. `ReplaySource` exposes bounded queue depth,
drop-new count, and overflow counters through the source diagnostics used by
`SttController`, with a paced producer and `perf_counter` captured timestamps.
The runner also counts accepted blocks, completed `process_audio` calls, and
expected versus observed controller result signals. It waits for the producer,
queue, all accepted work, and all expected results to finish before its short
settle interval, so an empty queue does not hide a delayed decode result.
The adapter never starts or imports `MicrophoneSource`'s device backend. The
existing controller remains responsible for its engine worker and public
result/diagnostic signals.

Fake-only tests exercise the adapter with temporary synthetic WAV/PCM data and
a fake engine through the existing `SttController` factories and public result
signals. After separate explicit user approval, `--mode real-model` validates
the manifest only after its conspicuous `--allow-real-model` gate, then starts
one spawned child process for model work. The child checks the gate again before
reading model assets or constructing `SttEngine`; the parent enforces a hard
deadline derived from replay duration, per-clip startup/result grace, and
fingerprinting grace, then uses bounded terminate/kill/join cleanup. Every WAV/
PCM entry and sample label is checked before a child starts. Each expanded
utterance uses a new controller/engine; model startup time and CPU are recorded
separately from post-start resource deltas.
The production `engine_factory` is rejected unless called from that authorized
spawned child; direct library calls fail before opening the audio path, keeping
the controller event loop under the parent process deadline.

The schema-1 JSON and CSV contain utterance labels, audio/configuration SHA-256,
timing and queue/reset/resource metrics, and nulls for unavailable values. In
authorized real mode, a model-directory fingerprint and VAD-file fingerprint
are streamed in bounded chunks inside the child process after its authorization
check. Fingerprinting rejects symlinks, special files, empty inputs, and
size/count overflow; artifacts contain only hashes, never asset paths. Fake
tests compute hashes only for synthetic temporary fixtures. Artifacts never
contain transcript text, audio bytes, model paths, audio-device names, or input
file paths. Current RSS is `null`; `ru_maxrss` is a process high-water that can
predate an utterance. Fake tests and synthetic benchmark runs are not model
measurements.

Real manifest entries use the existing `utterances` list with `dataset_id` at
the root, and add `audio_path` plus integer speech sample boundaries to each
utterance. Raw PCM also requires `raw_metadata` with `encoding`, `sample_rate`,
`channels`, and exact `sample_count`; WAV metadata is read from its header.
For example, a WAV clip may declare `utterance_id`, `audio_path`,
`truth_start_sample`, and `truth_end_sample`; `sample_rate` and
`audio_duration_samples` are optional cross-checks against its header. Each
repeat is a separate controller/engine run, and each result row records its
startup interval. The runner caps a manifest at 100 expanded clips, 16 MiB,
2 GiB aggregate audio, 30 minutes by default, and a 60 second result wait.

The existing DEV-002 sampler and schema remain authoritative: process CPU is a
sampled delta, high-water RSS may predate the run, and unavailable current RSS
stays `null`. Do not store audio bytes or transcript text in benchmark output.

Synthetic mode implements deterministic paced replay. Its fixture labels
support short and long durations, pauses, silence/noise controls,
continuous replay, repeated identical inputs and bounded queue overload. Use
`audio_duration_samples` for the duration, `pause_after_samples` for a pause and
`repeat` for repeated input; these definitions exercise the harness and do not
measure model accuracy or real performance.

The `silence_noise` fixture emits zero-valued float32 PCM. The separate
`synthetic_noise` fixture emits deterministic, low-amplitude float32 noise from
fixed seed `20261110`; chunks are capped at 4,096 samples and generated in
memory. This signal is not real speech or a realistic noise recording, and it
does not measure or validate VAD or ASR behavior.

Create a JSON manifest, for example:

```json
{"utterances":[{"utterance_id":"short-001","sample_rate":16000,
 "audio_duration_samples":3200,"truth_start_sample":0,
 "truth_end_sample":2400,"fixture":"short_speech","repeat":1}]}
```

Validate without replay using `python tools/stt_benchmark.py --manifest manifest.json
--output results/run.json --dry-run`. A synthetic replay omits `--dry-run` and
writes the required JSON run plus an adjacent CSV. It produces no audio or
transcript text. The output destination must be explicitly chosen by the caller;
existing JSON or CSV output files are rejected. The planned replay is capped at
30 minutes by default, which includes pauses. `--max-replay-seconds` can select
a lower cap or a higher cap up to six hours.

`real-model` is a separate mode and fails closed without
`--allow-real-model I_AUTHORIZE_REAL_MODEL_EXECUTION`. Even that conspicuous CLI
flag is not user authorization: separate explicit authorization is required.
After that separate approval, the CLI accepts a real manifest plus local
`--model-dir` and `--vad-model-path`; it validates before spawning and creates
the model only in the separately gated child. A timed-out child is terminated
and then killed within the configured shutdown grace; no result artifact is
exported unless the child has exited successfully.
Microphone and system-audio input are not implemented.
Fake queue-full drops increment `dropped_blocks` and can cause a gap or tail
reset; they do not increment `input_overflow_notifications`. Device overflow
notifications are zero unless a test explicitly scripts a fake device status.

All runtime timestamps are relative milliseconds from the replay's monotonic
origin. Truth labels remain sample offsets. `endpoint_audio_sample` and
`endpoint_audio_at_ms` come from the audio timeline; `endpoint_at_ms` and
`vad_endpoint_wall_ms` use monotonic timestamps. They are never derived from
one another. `controller_received_at_ms` and `result_observed_at_ms` represent
distinct delivery points. Decoder `cpu_ms` is separate from sampled process
CPU, and `process_cpu_percent` uses `100 * process_cpu_ms / wall_elapsed_ms`.

RSS is sampled around startup and replay with `resource.getrusage` process high-water
RSS. The exact `ru_maxrss` units are normalized to bytes on macOS and Linux;
current RSS is null. The process-lifetime high-water can predate this run and
does not provide a run-only peak. Synthetic fake diagnostics cannot characterize
real queue timing, VAD, segmentation, CPU, RSS, or recognition quality.

Quantiles use the deterministic nearest-rank rule `ceil(n*p)` (minimum rank 1),
include their sample count, and are `null` for empty observations. Queue and
utterance aggregation limits reject excess input rather than silently dropping
records. Tests inject a fake clock, source, engine, and resource sampler and
write artifacts only to pytest temporary paths.

The library helper `compare_artifacts(baseline, dev001)` compares two already
exported JSON artifacts (paths or parsed objects) offline. It pairs identical
utterance IDs, labels and configurations, rejects mismatched scenarios, and
reports baseline, DEV-001, absolute and percent deltas for latency,
decoder/process CPU and RSS. Different outcomes remain visible in paired rows;
missing metric values stay null. No model or benchmark run is invoked by the
comparison.

Without a controlled telemetry on/off toggle, baseline-vs-DEV-001 measures an
aggregate revision delta. It does not isolate or prove the causal overhead of
DEV-001 telemetry.
