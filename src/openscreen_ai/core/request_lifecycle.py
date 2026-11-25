"""Safe identifiers and counters for logical LLM requests and HTTP attempts."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from uuid import uuid4

logger = logging.getLogger(__name__)

_EVENT_COUNTERS = {
    "llm.send_requested": "send_actions_total",
    "llm.request_created": "logical_requests_total",
    "llm.http_attempt_started": "http_attempts_total",
    "llm.request_completed": "completed_requests_total",
    "llm.request_cancelled": "cancelled_requests_total",
    "llm.request_failed": "failed_requests_total",
    "llm.duplicate_send_blocked": "duplicate_send_blocked_total",
    "llm.callback_ignored_stale": "stale_callbacks_ignored_total",
}
_EVENTS = {
    "llm.send_requested",
    "llm.request_created",
    "llm.request_started",
    "llm.http_attempt_started",
    "llm.http_attempt_finished",
    "llm.request_cancel_requested",
    "llm.request_cancelled",
    "llm.request_completed",
    "llm.request_failed",
    "llm.callback_ignored_stale",
    "llm.duplicate_send_blocked",
}
_STATES = {
    "idle",
    "starting",
    "listening",
    "processing",
    "stopping",
    "closed",
    "completed",
    "failed",
    "error",
}
_OUTCOMES = {
    "accepted",
    "blocked",
    "duplicate",
    "started",
    "success",
    "provider_error",
    "transport_error",
    "cancelled",
    "user_cancel",
    "window_close",
    "failed",
    "stale",
}
_TERMINAL_EVENTS = {
    "llm.request_completed",
    "llm.request_cancelled",
    "llm.request_failed",
}
_COUNTER_NAMES = tuple(_EVENT_COUNTERS.values())


def new_opaque_id() -> str:
    """Return a random identifier that contains no user or endpoint data."""
    return uuid4().hex


def _safe_id(value: str | None) -> str | None:
    if value is not None and len(value) == 32 and all(
        character in "0123456789abcdef" for character in value
    ):
        return value
    return None


class LifecycleDiagnostics:
    """Thread-safe fixed-category lifecycle events and privacy-bounded counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter({name: 0 for name in _COUNTER_NAMES})
        self._request_started: dict[str, float] = {}
        self._attempt_started: dict[tuple[str, str], float] = {}

    def record(
        self,
        event: str,
        *,
        request_id: str | None = None,
        attempt_id: str | None = None,
        state: str,
        outcome: str,
    ) -> None:
        if event not in _EVENTS or state not in _STATES or outcome not in _OUTCOMES:
            raise ValueError("Unsupported lifecycle diagnostic category.")
        request_id = _safe_id(request_id)
        attempt_id = _safe_id(attempt_id)
        if event != "llm.send_requested" and request_id is None:
            raise ValueError("Request lifecycle events require an opaque request ID.")
        if event in {"llm.http_attempt_started", "llm.http_attempt_finished"} and (
            attempt_id is None
        ):
            raise ValueError("HTTP attempt events require an opaque attempt ID.")

        now = time.monotonic()
        attempt_key = (request_id, attempt_id) if request_id and attempt_id else None
        with self._lock:
            if event == "llm.request_created" and request_id is not None:
                self._request_started[request_id] = now
            if event == "llm.http_attempt_started" and attempt_key is not None:
                if attempt_key in self._attempt_started:
                    raise ValueError("HTTP attempt is already active.")
                self._attempt_started[attempt_key] = now
            if event == "llm.http_attempt_finished" and attempt_key is not None:
                if attempt_key not in self._attempt_started:
                    raise ValueError("HTTP attempt finish has no matching start.")
                del self._attempt_started[attempt_key]

            elapsed = (
                now - self._request_started[request_id]
                if request_id is not None and request_id in self._request_started
                else 0.0
            )
            counter = _EVENT_COUNTERS.get(event)
            if counter is not None:
                self._counts[counter] += 1
            if event in _TERMINAL_EVENTS and request_id is not None:
                self._request_started.pop(request_id, None)

        logger.info(
            "%s request_id=%s attempt_id=%s monotonic=%.6f elapsed=%.6f state=%s outcome=%s",
            event,
            request_id or "none",
            attempt_id or "none",
            now,
            elapsed,
            state,
            outcome,
        )

    def snapshot(self) -> dict[str, int]:
        """Return only the named, stable lifecycle counters."""
        with self._lock:
            return {name: self._counts[name] for name in _COUNTER_NAMES}

    @property
    def pending_timing_count(self) -> int:
        with self._lock:
            return len(self._request_started)

    @property
    def pending_attempt_count(self) -> int:
        with self._lock:
            return len(self._attempt_started)

    def reset(self) -> None:
        """Reset counters and timing state for isolated deterministic tests."""
        with self._lock:
            self._counts.clear()
            self._counts.update({name: 0 for name in _COUNTER_NAMES})
            self._request_started.clear()
            self._attempt_started.clear()


DIAGNOSTICS = LifecycleDiagnostics()
