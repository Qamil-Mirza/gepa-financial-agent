"""OpenTelemetry integration (the `pitstop[otel]` extra). Importing this package never imports opentelemetry:
`emit` imports it lazily, inside the call, and `SpanProcessor` (in `processor`) is loaded on first access."""
from __future__ import annotations

from typing import Any

__all__ = ["SpanProcessor"]


def __getattr__(name: str) -> Any:
    if name == "SpanProcessor":
        try:
            from .processor import SpanProcessor
        except ImportError as e:
            raise ImportError(f'pitstop.otel.SpanProcessor needs OpenTelemetry: pip install "pitstop[otel]" ({e})') \
                from e
        return SpanProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
