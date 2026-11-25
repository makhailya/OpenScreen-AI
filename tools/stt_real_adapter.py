"""Bounded file-backed PCM replay primitives for the existing STT controller.

Importing this module never imports an audio device or constructs an STT engine.
Actual model execution remains behind the benchmark's separate authorization gate.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import stat
import struct
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openscreen_ai.stt.audio import AudioBlock

SAMPLE_RATE = 16_000
REPLAY_BLOCK_SAMPLES = 640
MAX_CHUNK_SAMPLES = 4_096
MAX_DURATION_SECONDS = 30 * 60
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_FINGERPRINT_FILES = 100_000
MAX_FINGERPRINT_BYTES = 20 * 1024 * 1024 * 1024
FINGERPRINT_CHUNK_BYTES = 1024 * 1024
DEFAULT_QUEUE_CAPACITY = 20
MAX_QUEUE_CAPACITY = 1_000
_MAX_WAV_HEADER_BYTES = 1_048_576


class ReplayError(ValueError):
    """Unsupported, malformed, or over-limit replay input."""


@dataclass(frozen=True, slots=True)
class RawPcmMetadata:
    encoding: str
    sample_rate: int
    channels: int
    sample_count: int

    def validate(self) -> None:
        if self.encoding != "s16le" or self.sample_rate != SAMPLE_RATE or self.channels != 1:
            raise ReplayError("raw PCM requires explicit s16le, mono, 16000 Hz metadata")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ReplayError("raw PCM sample_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    container: str
    encoding: str
    sample_rate: int
    channels: int
    sample_count: int
    sha256: str


def validate_labels(start_sample: int, end_sample: int, sample_count: int) -> None:
    if any(type(value) is not int for value in (start_sample, end_sample, sample_count)):
        raise ReplayError("speech labels and duration must be integer sample indices")
    if sample_count < 0 or not 0 <= start_sample <= end_sample <= sample_count:
        raise ReplayError("speech labels must satisfy 0 <= start <= end <= sample_count")


class Pcm16Reader:
    """Validate once, then read at most 4096 PCM samples per call."""

    def __init__(
        self,
        path: str | Path,
        *,
        raw_metadata: RawPcmMetadata | None = None,
        max_duration_seconds: float = MAX_DURATION_SECONDS,
        max_input_bytes: int = MAX_INPUT_BYTES,
    ) -> None:
        self.path = Path(path)
        self._handle = None
        if not math.isfinite(max_duration_seconds) or not 0 < max_duration_seconds <= 6 * 3600:
            raise ReplayError("duration limit must be in (0, 21600] seconds")
        if type(max_input_bytes) is not int or not 1 <= max_input_bytes <= MAX_INPUT_BYTES:
            raise ReplayError(f"input byte limit must be in 1..{MAX_INPUT_BYTES}")
        size = self.path.stat().st_size
        if size <= 0 or size > max_input_bytes:
            raise ReplayError("input file is empty or exceeds the configured byte limit")
        self._handle = self.path.open("rb")
        try:
            if raw_metadata is None:
                data_offset, data_size, count = self._parse_wav(size)
                container = "wav"
            else:
                raw_metadata.validate()
                if size != raw_metadata.sample_count * 2:
                    raise ReplayError("raw PCM size does not match declared sample_count")
                data_offset, data_size, count = 0, size, raw_metadata.sample_count
                container = "raw-s16le"
            if count <= 0 or count / SAMPLE_RATE > max_duration_seconds:
                raise ReplayError("audio is empty or exceeds the configured duration limit")
            self._data_offset = data_offset
            self._data_size = data_size
            self._samples_read = 0
            self.metadata = AudioMetadata(
                container=container,
                encoding="pcm_s16le",
                sample_rate=SAMPLE_RATE,
                channels=1,
                sample_count=count,
                sha256=self._hash_file(),
            )
            self._handle.seek(self._data_offset)
        except BaseException:
            self.close()
            raise

    def _parse_wav(self, file_size: int) -> tuple[int, int, int]:
        assert self._handle is not None
        header = self._handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ReplayError("expected a valid RIFF/WAVE file")
        declared_end = struct.unpack_from("<I", header, 4)[0] + 8
        if declared_end != file_size:
            raise ReplayError("WAV RIFF size does not match the file (truncated or malformed)")
        cursor = 12
        fmt: tuple[int, int, int, int] | None = None
        data: tuple[int, int] | None = None
        while cursor + 8 <= file_size and cursor < _MAX_WAV_HEADER_BYTES:
            self._handle.seek(cursor)
            chunk_header = self._handle.read(8)
            if len(chunk_header) != 8:
                raise ReplayError("truncated WAV chunk header")
            chunk_id, chunk_size = chunk_header[:4], struct.unpack_from("<I", chunk_header, 4)[0]
            body = cursor + 8
            chunk_end = body + chunk_size
            padded_end = chunk_end + (chunk_size & 1)
            if padded_end > file_size:
                raise ReplayError("truncated WAV chunk")
            if chunk_id == b"fmt ":
                if fmt is not None or chunk_size != 16:
                    raise ReplayError("duplicate or malformed WAV fmt chunk")
                self._handle.seek(body)
                fields = self._handle.read(16)
                if len(fields) != 16:
                    raise ReplayError("truncated WAV fmt data")
                fmt = struct.unpack("<HHIIHH", fields)
            elif chunk_id == b"data":
                if data is not None:
                    raise ReplayError("multiple WAV data chunks are unsupported")
                data = (body, chunk_size)
            else:
                raise ReplayError("WAV metadata and unknown chunks are unsupported")
            cursor = padded_end
        if cursor != file_size:
            raise ReplayError("WAV contains a truncated chunk header")
        if fmt is None or data is None:
            raise ReplayError("WAV must contain fmt and data chunks")
        encoding, channels, rate, byte_rate, block_align, bits = fmt
        if (encoding, channels, rate, byte_rate, block_align, bits) != (
            1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
        ):
            raise ReplayError("WAV must be uncompressed PCM16LE, mono, 16000 Hz")
        data_offset, data_size = data
        if data_size % 2:
            raise ReplayError("PCM16 data chunk has an incomplete sample")
        return data_offset, data_size, data_size // 2

    def _hash_file(self) -> str:
        assert self._handle is not None
        self._handle.seek(0)
        digest = hashlib.sha256()
        while chunk := self._handle.read(MAX_CHUNK_SAMPLES * 2):
            digest.update(chunk)
        return digest.hexdigest()

    def read_samples(self, max_samples: int = MAX_CHUNK_SAMPLES) -> Any | None:
        if type(max_samples) is not int or not 1 <= max_samples <= MAX_CHUNK_SAMPLES:
            raise ReplayError(f"chunk size must be in 1..{MAX_CHUNK_SAMPLES} samples")
        if self._handle is None:
            raise ReplayError("reader is closed")
        remaining = self.metadata.sample_count - self._samples_read
        if remaining == 0:
            return None
        wanted = min(max_samples, remaining) * 2
        raw = self._handle.read(wanted)
        if len(raw) != wanted:
            raise ReplayError("PCM data ended before its declared sample count")
        samples = array(
            "f", (value / 32768.0 for (value,) in struct.iter_unpack("<h", raw))
        )
        self._samples_read += len(samples)
        return samples

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Pcm16Reader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def sample_index_to_perf(origin_perf: float, sample_index: int, sample_rate: int) -> float:
    if type(sample_index) is not int or sample_index < 0 or sample_rate <= 0:
        raise ReplayError("invalid sample index or sample rate")
    return origin_perf + sample_index / sample_rate


def fingerprint_file(path: str | Path) -> str:
    """Hash a regular, non-symlink file in bounded chunks without exposing its path."""
    file_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReplayError("safe no-follow file opens are unavailable on this platform")
    try:
        path_stat = file_path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise ReplayError("fingerprint input is a symlink")
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size <= 0:
            raise ReplayError("fingerprint input must be a non-empty regular file")
        if path_stat.st_size > MAX_FINGERPRINT_BYTES:
            raise ReplayError("fingerprint file exceeds the configured byte limit")
        fd = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ReplayError("fingerprint file is unavailable or is a symlink") from error
    try:
        actual = os.fstat(fd)
        _verify_fingerprint_stat(path_stat, actual)
        digest = hashlib.sha256()
        read_size = 0
        while chunk := os.read(fd, FINGERPRINT_CHUNK_BYTES):
            digest.update(chunk)
            read_size += len(chunk)
        after = os.fstat(fd)
        _verify_fingerprint_stat(actual, after)
        if read_size != path_stat.st_size:
            raise ReplayError("fingerprint file changed while being read")
        return digest.hexdigest()
    finally:
        os.close(fd)


def fingerprint_model_directory(path: str | Path) -> str:
    """Hash sorted relative file names and contents, rejecting links/special files."""
    root = Path(path)
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise ReplayError("safe no-follow directory opens are unavailable on this platform")
    try:
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode):
            raise ReplayError("model fingerprint root is a symlink")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ReplayError("model fingerprint root must be a real directory")
        root_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as error:
        raise ReplayError("model fingerprint directory is unavailable or is a symlink") from error
    digest = hashlib.sha256()
    total_files = 0
    total_bytes = 0

    def walk(directory_fd: int, prefix: str) -> None:
        nonlocal total_files, total_bytes
        directory_stat = os.fstat(directory_fd)
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
        except OSError as error:
            raise ReplayError("model directory could not be scanned safely") from error
        for entry in entries:
            name = entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ReplayError("model directory changed during fingerprinting") from error
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ReplayError("model directory contains a symlink")
            relative_name = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(entry_stat.st_mode):
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise ReplayError("model subdirectory could not be opened safely") from error
                try:
                    _verify_fingerprint_stat(entry_stat, os.fstat(child_fd))
                    walk(child_fd, relative_name)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ReplayError("model directory contains a non-regular file")
            if entry_stat.st_size <= 0:
                raise ReplayError("model directory contains an empty file")
            total_files += 1
            total_bytes += entry_stat.st_size
            if total_files > MAX_FINGERPRINT_FILES or total_bytes > MAX_FINGERPRINT_BYTES:
                raise ReplayError("model directory exceeds fingerprint file/byte limits")
            encoded_name = os.fsencode(relative_name)
            digest.update(struct.pack(">Q", len(encoded_name)))
            digest.update(encoded_name)
            digest.update(struct.pack(">Q", entry_stat.st_size))
            try:
                file_fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
            except OSError as error:
                raise ReplayError("model file could not be opened safely") from error
            try:
                actual_stat = os.fstat(file_fd)
                _verify_fingerprint_stat(entry_stat, actual_stat)
                read_size = 0
                while chunk := os.read(file_fd, FINGERPRINT_CHUNK_BYTES):
                    digest.update(chunk)
                    read_size += len(chunk)
                _verify_fingerprint_stat(actual_stat, os.fstat(file_fd))
                if read_size != entry_stat.st_size:
                    raise ReplayError("model file changed while being fingerprinted")
            finally:
                os.close(file_fd)
        _verify_fingerprint_stat(directory_stat, os.fstat(directory_fd))

    try:
        _verify_fingerprint_stat(root_stat, os.fstat(root_fd))
        walk(root_fd, "")
    finally:
        os.close(root_fd)
    if total_files == 0:
        raise ReplayError("model directory contains no files to fingerprint")
    return digest.hexdigest()


def _verify_fingerprint_stat(expected: os.stat_result, actual: os.stat_result) -> None:
    if (
        expected.st_dev != actual.st_dev
        or expected.st_ino != actual.st_ino
        or expected.st_mode != actual.st_mode
        or expected.st_size != actual.st_size
        or expected.st_mtime_ns != actual.st_mtime_ns
    ):
        raise ReplayError("fingerprint input changed while being read")


def build_timing_metrics(
    *, replay_origin_perf: float | None, sample_rate: int, truth_end_sample: int,
    endpoint_at: float | None = None, endpoint_audio_at: float | None = None,
    decode_completed_at: float | None = None, worker_emit_at: float | None = None,
    controller_received_at: float | None = None, result_observed_at: float | None = None,
    feed_completed_at: float | None = None, audio_sample_count: int | None = None,
) -> dict[str, float | int | None]:
    """Calculate metrics without subtracting sample indices from wall timestamps."""
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ReplayError("invalid replay clock origin or sample rate")
    if type(truth_end_sample) is not int or truth_end_sample < 0:
        raise ReplayError("truth_end_sample must be a non-negative integer")
    if (feed_completed_at is None) != (audio_sample_count is None):
        raise ReplayError(
            "feed completion timestamp and audio sample count must be supplied together"
        )
    if audio_sample_count is not None and (
        type(audio_sample_count) is not int or audio_sample_count < truth_end_sample
    ):
        raise ReplayError("audio_sample_count must cover truth_end_sample")
    events = (endpoint_at, decode_completed_at, worker_emit_at,
              controller_received_at, result_observed_at)
    if any(value is not None and not math.isfinite(value) for value in events):
        raise ReplayError("runtime timestamps must be finite monotonic values")
    if replay_origin_perf is None:
        if any(value is not None for value in (*events, endpoint_audio_at, feed_completed_at)):
            raise ReplayError("timestamps require a replay clock origin")
        return {
            "truth_end_sample": truth_end_sample,
            "truth_end_wall_ms": None,
            "endpoint_audio_sample": None,
            "endpoint_at_ms": None,
            "decode_completed_at_ms": None,
            "worker_emit_at_ms": None,
            "controller_received_at_ms": None,
            "result_observed_at_ms": None,
            "feed_completed_at_ms": None,
            "vad_endpoint_wall_ms": None,
            "vad_endpoint_audio_lag_ms": None,
            "decode_after_endpoint_ms": None,
            "decode_after_truth_end_ms": None,
            "result_latency_ms": None,
            "feed_lag_ms": None,
        }
    if not math.isfinite(replay_origin_perf):
        raise ReplayError("invalid replay clock origin or sample rate")
    if feed_completed_at is not None and (
        not math.isfinite(feed_completed_at) or feed_completed_at < replay_origin_perf
    ):
        raise ReplayError("feed completion timestamp must be finite and after replay origin")
    ordered = [value for value in events if value is not None]
    if any(later < earlier for earlier, later in zip(ordered, ordered[1:], strict=False)):
        raise ReplayError("runtime event timestamps are out of order")
    truth_end_perf = sample_index_to_perf(replay_origin_perf, truth_end_sample, sample_rate)
    endpoint_audio_sample = None
    if endpoint_audio_at is not None:
        if not math.isfinite(endpoint_audio_at) or endpoint_audio_at < replay_origin_perf:
            raise ReplayError("audio endpoint precedes replay clock origin or is invalid")
        endpoint_audio_sample = round((endpoint_audio_at - replay_origin_perf) * sample_rate)

    def relative_ms(value: float | None) -> float | None:
        return (value - replay_origin_perf) * 1000 if value is not None else None

    return {
        "truth_end_sample": truth_end_sample,
        "truth_end_wall_ms": relative_ms(truth_end_perf),
        "endpoint_audio_sample": endpoint_audio_sample,
        "endpoint_at_ms": relative_ms(endpoint_at),
        "decode_completed_at_ms": relative_ms(decode_completed_at),
        "worker_emit_at_ms": relative_ms(worker_emit_at),
        "controller_received_at_ms": relative_ms(controller_received_at),
        "result_observed_at_ms": relative_ms(result_observed_at),
        "feed_completed_at_ms": relative_ms(feed_completed_at),
        "vad_endpoint_wall_ms": (
            (endpoint_at - truth_end_perf) * 1000 if endpoint_at is not None else None
        ),
        "vad_endpoint_audio_lag_ms": (
            (endpoint_audio_sample - truth_end_sample) * 1000 / sample_rate
            if endpoint_audio_sample is not None else None
        ),
        "decode_after_endpoint_ms": (
            (decode_completed_at - endpoint_at) * 1000
            if decode_completed_at is not None and endpoint_at is not None else None
        ),
        "decode_after_truth_end_ms": (
            (decode_completed_at - truth_end_perf) * 1000
            if decode_completed_at is not None
            else None
        ),
        "result_latency_ms": (
            (result_observed_at - truth_end_perf) * 1000
            if result_observed_at is not None
            else None
        ),
        "feed_lag_ms": (
            (feed_completed_at - sample_index_to_perf(
                replay_origin_perf, audio_sample_count, sample_rate
            )) * 1000
            if feed_completed_at is not None and audio_sample_count is not None
            else None
        ),
    }


class ReplaySource:
    """Paced bounded source implementing the interface consumed by SttController."""

    def __init__(
        self,
        reader: Pcm16Reader,
        *,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        block_samples: int = REPLAY_BLOCK_SAMPLES,
        clock: Any = time,
    ) -> None:
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= MAX_QUEUE_CAPACITY:
            raise ReplayError(f"queue capacity must be in 1..{MAX_QUEUE_CAPACITY}")
        if type(block_samples) is not int or not 1 <= block_samples <= MAX_CHUNK_SAMPLES:
            raise ReplayError(f"block size must be in 1..{MAX_CHUNK_SAMPLES} samples")
        self.reader = reader
        self.block_samples = block_samples
        self._queue: queue.Queue[AudioBlock] = queue.Queue(maxsize=queue_capacity)
        self._clock = clock
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.origin_perf: float | None = None
        self._sequence = self._dropped = self._max_depth = self._input_overflows = 0
        self._accepted_blocks = 0
        self._sample_index = 0
        self.feed_completed_at: float | None = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_max_depth(self) -> int:
        return self._max_depth

    @property
    def dropped_blocks(self) -> int:
        return self._dropped

    @property
    def accepted_blocks(self) -> int:
        return self._accepted_blocks

    @property
    def input_overflows(self) -> int:
        return self._input_overflows

    @property
    def producer_error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            raise ReplayError("replay source can only be started once")
        self.origin_perf = self._clock.perf_counter()
        self._thread = threading.Thread(target=self._produce, name="stt-file-replay", daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            while not self._cancel.is_set():
                samples = self.reader.read_samples(self.block_samples)
                if samples is None:
                    # Complete the final block's scheduled audio interval before
                    # declaring the source drained.
                    final_due = sample_index_to_perf(
                        self.origin_perf, self._sample_index, SAMPLE_RATE
                    )
                    delay = final_due - self._clock.perf_counter()
                    if delay > 0 and self._cancel.wait(delay):
                        return
                    self.feed_completed_at = self._clock.perf_counter()
                    return
                index = self._sample_index
                captured_at = sample_index_to_perf(self.origin_perf, index, SAMPLE_RATE)
                delay = captured_at - self._clock.perf_counter()
                if delay > 0 and self._cancel.wait(delay):
                    return
                block = AudioBlock(samples, captured_at, self._sequence, self._input_overflows)
                self._sequence += 1
                try:
                    self._queue.put_nowait(block)
                    self._accepted_blocks += 1
                    self._max_depth = max(self._max_depth, self._queue.qsize())
                except queue.Full:
                    self._dropped += 1
                self._sample_index += len(samples)
        except BaseException as error:
            self._error = error

    def read(self) -> AudioBlock | None:
        if self._error is not None:
            raise ReplayError(f"replay producer failed: {self._error}") from self._error
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def finished(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.reader.close()
                raise TimeoutError("replay producer did not stop before timeout")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.reader.close()


def run_controller_replay(
    controller: Any,
    source: ReplaySource,
    *,
    progress: Any,
    result_timeout_seconds: float = 5.0,
    startup_timeout_seconds: float = 30.0,
    settle_seconds: float = 0.05,
    on_listening: Any | None = None,
) -> dict[str, Any]:
    """Run an already factory-configured SttController and observe public signals.

    No engine is constructed here. Callers must inject a fake engine for tests;
    production callers must pass their engine factory only after authorization.
    Transcript text and exception messages are deliberately discarded.
    """
    for name, value, maximum in (
        ("result_timeout_seconds", result_timeout_seconds, 3600.0),
        ("startup_timeout_seconds", startup_timeout_seconds, 3600.0),
        ("settle_seconds", settle_seconds, 10.0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplayError(f"{name} must be a finite number")
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ReplayError(f"{name} must be in (0, {maximum:g}] seconds")

    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    state: dict[str, Any] = {
        "result_count": 0,
        "transcript_nonempty": False,
        "result_observed_at": None,
        "last_result_observed_at": None,
        "result_fields": None,
        "result_diagnostic": None,
        "diagnostic": None,
        "controller_error": False,
        "finished": False,
        "timed_out": False,
        "controller_started_at": None,
        "controller_listening_at": None,
        "queue_depth_at_stop": None,
        "completion_observed_at": None,
    }
    created_at = time.perf_counter()

    def receive_result(result: Any) -> None:
        state["result_count"] += 1
        observed_at = time.perf_counter()
        state["last_result_observed_at"] = observed_at
        if state["result_observed_at"] is not None:
            return
        state["result_observed_at"] = observed_at
        state["result_diagnostic"] = (
            dict(state["diagnostic"]) if state["diagnostic"] is not None else None
        )
        text = getattr(result, "text", "")
        state["transcript_nonempty"] = bool(text.strip()) if isinstance(text, str) else False
        state["result_fields"] = {
            name: getattr(result, name, None)
            for name in (
                "endpoint_at",
                "endpoint_audio_at",
                "recognition_ms",
                "cpu_ms",
                "decode_completed_at",
            )
        }

    def receive_diagnostic(diagnostic: Any) -> None:
        state["diagnostic"] = {
            name: getattr(diagnostic, name, None)
            for name in (
                "queue_depth_blocks",
                "queue_max_depth_blocks",
                "dropped_blocks",
                "input_overflow_notifications",
                "vad_resets_total",
                "vad_resets_gap",
                "vad_resets_overflow",
                "vad_resets_tail",
                "endpoint_at",
                "recognition_ms",
                "decode_completed_at",
                "worker_emit_at",
                "controller_received_at",
            )
        }

    def receive_state(value: str) -> None:
        if value == "listening" and state["controller_listening_at"] is None:
            state["controller_listening_at"] = time.perf_counter()
            if on_listening is not None:
                on_listening()

    def request_stop(*, timeout: bool = False) -> None:
        if timeout:
            state["timed_out"] = True
        if not state["stop_requested"]:
            state["stop_requested"] = True
            state["queue_depth_at_stop"] = source.queue_depth
            controller.request_stop()

    def poll() -> None:
        now = time.perf_counter()
        if state["finished"]:
            loop.quit()
            return
        if state["controller_error"]:
            request_stop()
            return
        if source.origin_perf is None:
            if now >= created_at + startup_timeout_seconds:
                request_stop(timeout=True)
            return
        run_deadline = (
            source.origin_perf
            + source.reader.metadata.sample_count / SAMPLE_RATE
            + result_timeout_seconds
        )
        feed_at = source.feed_completed_at
        if source.finished and feed_at is None:
            # Producer errors are delivered through the controller; this fallback
            # still bounds shutdown if a custom source ends without a timestamp.
            feed_at = now
        counts = progress.snapshot()
        all_blocks_processed = (
            source.finished
            and source.queue_depth == 0
            and counts["processed_blocks"] == source.accepted_blocks
        )
        all_results_observed = counts["expected_result_signals"] == state["result_count"]
        if feed_at is not None and all_blocks_processed and all_results_observed:
            if state["completion_observed_at"] is None:
                state["completion_observed_at"] = now
            if now >= state["completion_observed_at"] + settle_seconds:
                if state["result_count"] == 0:
                    state["missing_result"] = True
                request_stop()
                return
        if now >= run_deadline:
            request_stop(timeout=True)

    state["stop_requested"] = False
    controller.transcription_ready.connect(receive_result)
    controller.diagnostic_ready.connect(receive_diagnostic)
    controller.state_changed.connect(receive_state)
    controller.error_occurred.connect(lambda _message: state.__setitem__("controller_error", True))
    controller.finished.connect(lambda: (state.__setitem__("finished", True), loop.quit()))
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(poll)
    state["controller_started_at"] = time.perf_counter()
    if not controller.start():
        raise ReplayError("STT controller refused to start")
    timer.start()
    try:
        loop.exec()
    finally:
        timer.stop()
        app.processEvents()
    return {
        **state,
        "accepted_blocks": source.accepted_blocks,
        **progress.snapshot(),
        "observed_result_signals": state["result_count"],
        "result_outcome": (
            "input_error" if source.producer_error is not None
            else "decode_error" if state["controller_error"]
            else "recognized" if state["transcript_nonempty"]
            else "empty_transcript" if state["result_observed_at"] is not None
            else "timeout" if state["timed_out"]
            else "missing_transcript"
        ),
    }
