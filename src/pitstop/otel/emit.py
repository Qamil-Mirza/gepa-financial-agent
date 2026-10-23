"""Verdicts as span events ("emit, don't compete"): each finding PitStop records adds a `pitstop.verdict` event to
a span, so the trace a user already exports shows what PitStop decided and why. Best effort and dependency-free at
import: opentelemetry is imported inside the call, and a missing package, a span that is not recording or any
error is a silent no-op."""
from __future__ import annotations

from typing import Any

EVENT = "pitstop.verdict"
MAX_REASON = 300
_UNSET: Any = object()
_OTEL: Any = _UNSET  # the opentelemetry.trace module once imported, None when it is not installed


def _otel_trace() -> Any:
    """The `opentelemetry.trace` module, or None; a failed import is attempted once per process."""
    global _OTEL
    if _OTEL is _UNSET:
        try:
            from opentelemetry import trace
            _OTEL = trace
        except Exception:
            _OTEL = None
    return _OTEL


def emit_verdict(*, run_id: str | None, drift_type: str, level: Any, action: str, drift_id: int | None,
                 reason: str, span: Any = None) -> None:
    """Add a `pitstop.verdict` event to `span`, or to the current span when it is None. Attributes:
    `pitstop.drift_type`, `pitstop.level`, `pitstop.action`, `pitstop.run_id`, `pitstop.reason` (at most 300
    chars) and `pitstop.drift_id` (omitted when unknown, like run_id). Never raises."""
    try:
        if span is None:
            trace = _otel_trace()
            if trace is None:
                return
            span = trace.get_current_span()
        if not span.is_recording():
            return
        attrs: dict[str, str | int] = {"pitstop.drift_type": str(drift_type),
                                       "pitstop.level": str(getattr(level, "value", level)),
                                       "pitstop.action": str(action), "pitstop.reason": str(reason)[:MAX_REASON]}
        if run_id is not None:
            attrs["pitstop.run_id"] = str(run_id)
        if drift_id is not None:
            attrs["pitstop.drift_id"] = int(drift_id)
        span.add_event(EVENT, attrs)
    except Exception:
        return
