"""Shapes: what one provider API looks like to the metering engine, and the method tables that route to them.

A `Shape` knows how to estimate a request's prompt and its worst-case output, how to read usage, text and tool
calls from a response or from stream chunks, and which chunk (if any) PitStop's own `prepare` injected and must
hide again. `_Call` (call.py) runs the same reserve -> settle -> record -> detect cycle for every shape.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

DEFAULT_MAX_COMPLETION_TOKENS = 4096


def _g(obj: Any, name: str, default: Any = None) -> Any:
    """None-safe attribute (or dict key) read: provider responses and test fakes vary in shape."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _text_len(v: Any) -> int:
    return len(v) if isinstance(v, str) else 0


@dataclass
class Observed:
    """Everything one response or stream showed."""

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "arguments": raw str}]
    finish_reason: str | None = None
    refused: bool = False  # provider-declared refusal (e.g. Anthropic stop_reason "refusal")
    prompt_tokens: int | None = None  # None = usage not seen -> settle by estimate
    completion_tokens: int | None = None
    cached_tokens: int = 0
    chars: int = 0  # all completion text (content, tool names and arguments), for the estimate
    # stream accumulators, folded into content / tool_calls by seal()
    _text: list[str] = field(default_factory=list, init=False, repr=False)
    _tools: dict[Any, dict] = field(default_factory=dict, init=False, repr=False)

    @property
    def content_chars(self) -> int:
        return len(self.content) if self.content else 0

    def stream_text(self, text: str) -> None:
        """Append streamed content text (the first choice's only)."""
        self._text.append(text)

    def stream_tool(self, key: Any) -> dict:
        """The tool call being streamed under `key` (a provider index), created empty on first sight."""
        return self._tools.setdefault(key, {"name": None, "arguments": ""})

    def seal(self) -> None:
        """Fold the stream accumulators into `content` and `tool_calls`. Idempotent."""
        if self._text:
            self.content = (self.content or "") + "".join(self._text)
            self._text.clear()
        if self._tools:
            try:
                keys = sorted(self._tools)
            except TypeError:  # mixed key types: keep arrival order
                keys = list(self._tools)
            self.tool_calls = self.tool_calls + [self._tools[k] for k in keys]
            self._tools.clear()


class Shape(Protocol):
    name: str  # "chat" | "responses" | "completions" | "anthropic"

    def prepare(self, kwargs: dict) -> tuple[dict, bool]: ...  # (kwargs to send, injected_usage)

    def unprepare(self, kwargs: dict) -> dict: ...  # kwargs without PitStop's injection (retry)

    def estimate_prompt_tokens(self, kwargs: dict) -> int: ...

    def max_output_tokens(self, kwargs: dict) -> int: ...

    def read(self, resp: Any, obs: Observed) -> None: ...  # non-stream response

    def observe(self, chunk: Any, obs: Observed, injected: bool) -> bool: ...  # False = hide this chunk


# -- OpenAI chat completions ------------------------------------------------------------------------------------


class _ChatShape:
    """`chat.completions.create` / `parse`: messages in, choices out; stream usage via stream_options."""

    name = "chat"

    def prepare(self, kwargs: dict) -> tuple[dict, bool]:
        kwargs = dict(kwargs)
        inject = kwargs.get("stream") is True and "stream_options" not in kwargs
        if inject:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs, inject

    def unprepare(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        kwargs.pop("stream_options", None)
        return kwargs

    def estimate_prompt_tokens(self, kwargs: dict) -> int:
        total = 0
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):  # never iterate a caller's one-shot iterable
            for m in messages:
                content = _g(m, "content")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, (list, tuple)):
                    for part in content:
                        total += _text_len(part if isinstance(part, str) else _g(part, "text"))
        tools = kwargs.get("tools")
        if isinstance(tools, (list, tuple)) and tools:
            total += len(json.dumps(tools, default=str))
        return math.ceil(total / 4)

    def max_output_tokens(self, kwargs: dict) -> int:
        for key in ("max_tokens", "max_completion_tokens"):
            v = _int(kwargs.get(key))
            if v is not None and v >= 0:
                return v
        return DEFAULT_MAX_COMPLETION_TOKENS

    @staticmethod
    def _usage(u: Any, obs: Observed) -> None:
        prompt, completion = _int(_g(u, "prompt_tokens")), _int(_g(u, "completion_tokens"))
        if prompt is not None and completion is not None:
            obs.prompt_tokens, obs.completion_tokens = prompt, completion
            obs.cached_tokens = _int(_g(_g(u, "prompt_tokens_details"), "cached_tokens")) or 0

    def read(self, resp: Any, obs: Observed) -> None:
        self._usage(_g(resp, "usage"), obs)
        choices = _g(resp, "choices") or []
        choices = list(choices) if isinstance(choices, (list, tuple)) else []
        for i, ch in enumerate(choices):
            msg = _g(ch, "message")
            text = _g(msg, "content")
            obs.chars += _text_len(text)
            calls = []
            for tc in _g(msg, "tool_calls") or []:
                fn = _g(tc, "function")
                name, args = _g(fn, "name"), _g(fn, "arguments")
                obs.chars += _text_len(name) + _text_len(args)
                calls.append({"name": name, "arguments": args})
            if i == 0:
                obs.content = text if isinstance(text, str) else None
                obs.tool_calls, obs.finish_reason = calls, _g(ch, "finish_reason")

    def observe(self, chunk: Any, obs: Observed, injected: bool) -> bool:
        usage = _g(chunk, "usage")
        if usage is not None:
            self._usage(usage, obs)
        choices = _g(chunk, "choices") or []
        for ch in choices:
            first = (_g(ch, "index", 0) or 0) == 0
            delta = _g(ch, "delta")
            text = _g(delta, "content")
            if isinstance(text, str):
                obs.chars += len(text)
                if first:
                    obs.stream_text(text)
            for tc in _g(delta, "tool_calls") or []:
                fn = _g(tc, "function")
                name, args = _g(fn, "name"), _g(fn, "arguments")
                entry = obs.stream_tool(_g(tc, "index", 0) or 0) if first else None
                if isinstance(name, str) and name and not (entry is not None and entry["name"]):
                    obs.chars += len(name)  # a name arrives once; providers that repeat it are not re-counted
                    if entry is not None:
                        entry["name"] = name
                if isinstance(args, str):
                    obs.chars += len(args)
                    if entry is not None:
                        entry["arguments"] += args
            reason = _g(ch, "finish_reason")
            if first and reason:
                obs.finish_reason = reason
        # The caller did not ask for include_usage; hide the extra chunk so the stream looks exactly like the
        # unwrapped client's (code like chunk.choices[0] must keep working).
        return not (injected and not choices and usage is not None)


CHAT: Shape = _ChatShape()

Kind = Literal["call", "manager"]

# The OpenAI client's metered methods, by attribute path from the client. Every other attribute is delegated
# unchanged. "call": the method returns the response (or a stream). "manager": it returns a stream manager.
METHODS: dict[tuple[str, ...], tuple[Shape, Kind]] = {
    ("chat", "completions", "create"): (CHAT, "call"),
    ("chat", "completions", "parse"): (CHAT, "call"),
}
