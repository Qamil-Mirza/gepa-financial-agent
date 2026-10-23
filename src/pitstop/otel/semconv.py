"""Read a GenAI span's attributes into a `SpanView`, whatever convention wrote them: the OpenTelemetry GenAI semantic
conventions (current and older names), OpenInference (Arize, Phoenix) and OpenLLMetry (Traceloop). Pure: no store,
no opentelemetry import. For each field the first attribute present with the right type wins; a wrongly typed value
(a token count that is not an int) is ignored, and strings are cut to MAX_TEXT chars."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MAX_TEXT = 2000

Kind = Literal["llm", "tool", "agent", "other"]

# (attribute, {lowercased value: kind}); a present attribute decides the kind, an unknown value is "other"
_KIND_ATTRS: tuple[tuple[str, dict[str, Kind]], ...] = (
    ("gen_ai.operation.name", {"chat": "llm", "text_completion": "llm", "generate_content": "llm",
                               "execute_tool": "tool", "invoke_agent": "agent"}),
    ("openinference.span.kind", {"llm": "llm", "tool": "tool", "agent": "agent"}),
    ("traceloop.span.kind", {"tool": "tool", "agent": "agent"}),
    ("llm.request.type", {"chat": "llm", "completion": "llm"}),
)
_MODEL = ("gen_ai.response.model", "gen_ai.request.model", "llm.model_name", "llm.response.model_name")
_INPUT_TOKENS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt")
_OUTPUT_TOKENS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.token_count.completion")
_TOOL_NAME = ("gen_ai.tool.name", "tool.name")
_TOOL_ARGS = ("gen_ai.tool.call.arguments",)
_TOOL_ARGS_ON_TOOL_SPANS = ("input.value",)  # OpenInference: any span's input; a tool's arguments on TOOL spans
_AGENT_NAME = ("gen_ai.agent.name",)
_FINISH_REASONS = ("gen_ai.response.finish_reasons", "llm.response.finish_reason")


@dataclass(frozen=True)
class SpanView:
    kind: Kind
    model: str | None
    server: str | None              # server.address
    input_tokens: int | None
    output_tokens: int | None
    finish_reasons: tuple[str, ...]
    tool_name: str | None
    tool_args: str | None
    agent_name: str | None
    error: str | None               # error.type, or "error" when the status is ERROR


def _text(attrs: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for n in names:
        v = attrs.get(n)
        if isinstance(v, str):
            return v[:MAX_TEXT]
    return None


def _count(attrs: Mapping[str, Any], names: Sequence[str]) -> int | None:
    for n in names:
        v = attrs.get(n)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    return None


def _reasons(attrs: Mapping[str, Any]) -> tuple[str, ...]:
    for n in _FINISH_REASONS:
        v = attrs.get(n)
        if isinstance(v, str):
            return (v[:MAX_TEXT],)
        if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray)):
            out = tuple(r[:MAX_TEXT] for r in list(v)[:16] if isinstance(r, str))
            if out:
                return out
    return ()


def _kind(attrs: Mapping[str, Any]) -> Kind:
    for name, kinds in _KIND_ATTRS:
        v = attrs.get(name)
        if isinstance(v, str):
            return kinds.get(v.lower(), "other")
    return "other"


def read_span(name: str, attributes: Mapping[str, Any], status_is_error: bool) -> SpanView:
    """The fields PitStop needs from one span. `name` is the span's name (unused by the readers today; callers fall
    back to it for a tool or agent with no name attribute)."""
    attrs = attributes if attributes is not None else {}
    kind = _kind(attrs)
    args = _text(attrs, _TOOL_ARGS)
    if args is None and kind == "tool":
        args = _text(attrs, _TOOL_ARGS_ON_TOOL_SPANS)
    error = _text(attrs, ("error.type",))
    if error is None and status_is_error:
        error = "error"
    return SpanView(kind=kind, model=_text(attrs, _MODEL), server=_text(attrs, ("server.address",)),
                    input_tokens=_count(attrs, _INPUT_TOKENS), output_tokens=_count(attrs, _OUTPUT_TOKENS),
                    finish_reasons=_reasons(attrs), tool_name=_text(attrs, _TOOL_NAME), tool_args=args,
                    agent_name=_text(attrs, _AGENT_NAME), error=error)
