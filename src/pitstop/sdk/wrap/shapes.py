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


def _content_chars(content: Any) -> int:
    """Characters of a message's content: a string, or a list of parts (strings or parts with `text`)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, (list, tuple)):  # never iterate a caller's one-shot iterable
        return sum(_text_len(p if isinstance(p, str) else _g(p, "text")) for p in content)
    return 0


def _tools_chars(kwargs: dict) -> int:
    tools = kwargs.get("tools")
    return len(json.dumps(tools, default=str)) if isinstance(tools, (list, tuple)) and tools else 0


def _max_tokens(kwargs: dict, *keys: str) -> int:
    for key in keys:
        v = _int(kwargs.get(key))
        if v is not None and v >= 0:
            return v
    return DEFAULT_MAX_COMPLETION_TOKENS


def _inject_usage(kwargs: dict, stream: bool) -> tuple[dict, bool]:
    """Chat and legacy completions streams: ask for the usage chunk unless the caller set stream_options."""
    kwargs = dict(kwargs)
    inject = stream and "stream_options" not in kwargs
    if inject:
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs, inject


def _drop_usage(kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    kwargs.pop("stream_options", None)
    return kwargs


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
    # The provider keeps generating after the call returns (a Responses background request, or a response still
    # queued / in progress): its output is billed later, so without final usage the call is charged its full
    # reservation.
    pending: bool = False
    # stream accumulators, folded into content / tool_calls by seal()
    _text: list[str] = field(default_factory=list, init=False, repr=False)
    _tools: dict[Any, dict] = field(default_factory=dict, init=False, repr=False)
    # a shape's own stream state (Anthropic: the usage parts and which blocks are tool calls)
    state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

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

    # (kwargs to send, injected_usage). `stream`: the call streams (stream=True, or a stream helper)
    def prepare(self, kwargs: dict, stream: bool) -> tuple[dict, bool]: ...

    def unprepare(self, kwargs: dict) -> dict: ...  # kwargs without PitStop's injection (retry)

    def estimate_prompt_tokens(self, kwargs: dict) -> int: ...

    def max_output_tokens(self, kwargs: dict) -> int: ...

    def read(self, resp: Any, obs: Observed) -> None: ...  # non-stream response

    def observe(self, chunk: Any, obs: Observed, injected: bool) -> bool: ...  # False = hide this chunk

    def pending(self, kwargs: dict) -> bool: ...  # the request keeps generating after the call returns

    # Stream helpers (the "manager" kind). `helper_chunk`: what one helper event feeds to `observe`, or None for an
    # event derived from one already fed (passed through unfed). `hide_injected`: the caller's view of an
    # accumulated helper result (final completion, snapshot) while PitStop's own usage request is in effect.
    def helper_chunk(self, event: Any) -> Any: ...

    def hide_injected(self, result: Any) -> Any: ...


# -- OpenAI chat completions ------------------------------------------------------------------------------------


def _without_usage(result: Any) -> Any:
    """`result` as the caller would see it without include_usage: `usage` None (a copy; never mutated)."""
    if _g(result, "usage") is None:
        return result
    try:
        return result.model_copy(update={"usage": None})
    except Exception:
        return result


def _openai_usage(u: Any, obs: Observed) -> None:
    """`usage` of a chat or legacy completion (or of a stream's include_usage chunk)."""
    prompt, completion = _int(_g(u, "prompt_tokens")), _int(_g(u, "completion_tokens"))
    if prompt is not None and completion is not None:
        obs.prompt_tokens, obs.completion_tokens = prompt, completion
        obs.cached_tokens = _int(_g(_g(u, "prompt_tokens_details"), "cached_tokens")) or 0


class _ChatShape:
    """`chat.completions.create` / `parse` / `stream`: messages in, choices out; stream usage via stream_options."""

    name = "chat"

    def prepare(self, kwargs: dict, stream: bool) -> tuple[dict, bool]:
        return _inject_usage(kwargs, stream)

    def unprepare(self, kwargs: dict) -> dict:
        return _drop_usage(kwargs)

    def estimate_prompt_tokens(self, kwargs: dict) -> int:
        total = 0
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):  # never iterate a caller's one-shot iterable
            for m in messages:
                total += _content_chars(_g(m, "content"))
        return math.ceil((total + _tools_chars(kwargs)) / 4)

    def max_output_tokens(self, kwargs: dict) -> int:
        return _max_tokens(kwargs, "max_tokens", "max_completion_tokens")

    def read(self, resp: Any, obs: Observed) -> None:
        _openai_usage(_g(resp, "usage"), obs)
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
            _openai_usage(usage, obs)
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

    def pending(self, kwargs: dict) -> bool:
        return False

    def helper_chunk(self, event: Any) -> Any:
        # The helper's `chunk` events carry the raw chunk; content.delta, tool_calls.* etc. are derived from it.
        return _g(event, "chunk") if _g(event, "type") == "chunk" else None

    def hide_injected(self, result: Any) -> Any:
        return _without_usage(result)


CHAT: Shape = _ChatShape()


# -- OpenAI Responses API ---------------------------------------------------------------------------------------

_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})
_UNFINISHED = frozenset({"queued", "in_progress"})


class _ResponsesShape:
    """`responses.create` / `parse` / `stream`: instructions + input in, output items out. Streams always end in a
    terminal event (`response.completed`) that carries usage, so nothing is injected."""

    name = "responses"

    def prepare(self, kwargs: dict, stream: bool) -> tuple[dict, bool]:
        return dict(kwargs), False

    def unprepare(self, kwargs: dict) -> dict:
        return dict(kwargs)

    def estimate_prompt_tokens(self, kwargs: dict) -> int:
        total = _text_len(kwargs.get("instructions"))
        items = kwargs.get("input")
        if isinstance(items, str):
            total += len(items)
        elif isinstance(items, (list, tuple)):
            for item in items:
                total += _content_chars(_g(item, "content"))
                # function_call / function_call_output items carry text outside `content`
                total += _text_len(_g(item, "arguments")) + _content_chars(_g(item, "output"))
        return math.ceil((total + _tools_chars(kwargs)) / 4)

    def max_output_tokens(self, kwargs: dict) -> int:
        return _max_tokens(kwargs, "max_output_tokens")

    @staticmethod
    def _usage(u: Any, obs: Observed) -> None:
        prompt, completion = _int(_g(u, "input_tokens")), _int(_g(u, "output_tokens"))
        if prompt is not None and completion is not None:
            obs.prompt_tokens, obs.completion_tokens = prompt, completion
            obs.cached_tokens = _int(_g(_g(u, "input_tokens_details"), "cached_tokens")) or 0

    def read(self, resp: Any, obs: Observed) -> None:
        if _g(resp, "status") in _UNFINISHED:  # a background response: its output is billed after we return
            obs.pending = True
        else:
            self._usage(_g(resp, "usage"), obs)
        output = _g(resp, "output") or []
        texts: list[str] = []
        calls = []
        for item in output if isinstance(output, (list, tuple)) else []:
            kind = _g(item, "type")
            if kind == "function_call":
                name, args = _g(item, "name"), _g(item, "arguments")
                obs.chars += _text_len(name) + _text_len(args)
                calls.append({"name": name, "arguments": args})
            elif kind == "message":
                for part in _g(item, "content") or []:
                    text = _g(part, "text")
                    if _g(part, "type") == "output_text" and isinstance(text, str):
                        obs.chars += len(text)
                        texts.append(text)
        obs.content = "".join(texts) if texts else None
        obs.tool_calls, obs.finish_reason = calls, _g(resp, "status")

    def observe(self, event: Any, obs: Observed, injected: bool) -> bool:
        kind = _g(event, "type")
        if kind == "response.output_text.delta":
            text = _g(event, "delta")
            if isinstance(text, str):
                obs.chars += len(text)
                obs.stream_text(text)
        elif kind == "response.output_item.added":
            item = _g(event, "item")
            if _g(item, "type") == "function_call":
                entry = obs.stream_tool(_g(event, "output_index", 0))
                name, args = _g(item, "name"), _g(item, "arguments")
                if isinstance(name, str) and not entry["name"]:
                    obs.chars += len(name)
                    entry["name"] = name
                if isinstance(args, str) and args:
                    obs.chars += len(args)
                    entry["arguments"] += args
        elif kind == "response.function_call_arguments.delta":
            args = _g(event, "delta")
            if isinstance(args, str):
                obs.chars += len(args)
                obs.stream_tool(_g(event, "output_index", 0))["arguments"] += args
        elif kind in _TERMINAL:
            resp = _g(event, "response")
            self._usage(_g(resp, "usage"), obs)
            obs.finish_reason = _g(resp, "status")
        return True

    def pending(self, kwargs: dict) -> bool:
        return kwargs.get("background") is True

    def helper_chunk(self, event: Any) -> Any:
        return event  # the helper's events are the raw events (some re-built with parsed fields)

    def hide_injected(self, result: Any) -> Any:
        return result  # nothing is injected


RESPONSES: Shape = _ResponsesShape()


# -- OpenAI legacy completions ------------------------------------------------------------------------------------


class _CompletionsShape:
    """`completions.create`: a prompt in, `choices[].text` out; stream usage via stream_options, as for chat."""

    name = "completions"

    def prepare(self, kwargs: dict, stream: bool) -> tuple[dict, bool]:
        return _inject_usage(kwargs, stream)

    def unprepare(self, kwargs: dict) -> dict:
        return _drop_usage(kwargs)

    def estimate_prompt_tokens(self, kwargs: dict) -> int:
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str):
            return math.ceil(len(prompt) / 4)
        if isinstance(prompt, (list, tuple)):
            return math.ceil(sum(_text_len(p) for p in prompt) / 4)
        return 0

    def max_output_tokens(self, kwargs: dict) -> int:
        # The API's own default is 16 tokens; PitStop reserves conservatively.
        return _max_tokens(kwargs, "max_tokens")

    def read(self, resp: Any, obs: Observed) -> None:
        _openai_usage(_g(resp, "usage"), obs)
        choices = _g(resp, "choices") or []
        for i, ch in enumerate(choices if isinstance(choices, (list, tuple)) else []):
            text = _g(ch, "text")
            obs.chars += _text_len(text)
            if i == 0:
                obs.content = text if isinstance(text, str) else None
                obs.finish_reason = _g(ch, "finish_reason")

    def observe(self, chunk: Any, obs: Observed, injected: bool) -> bool:
        usage = _g(chunk, "usage")
        if usage is not None:
            _openai_usage(usage, obs)
        choices = _g(chunk, "choices") or []
        for ch in choices:
            first = (_g(ch, "index", 0) or 0) == 0
            text = _g(ch, "text")
            if isinstance(text, str):
                obs.chars += len(text)
                if first:
                    obs.stream_text(text)
            reason = _g(ch, "finish_reason")
            if first and reason:
                obs.finish_reason = reason
        return not (injected and not choices and usage is not None)  # hide PitStop's own usage chunk

    def pending(self, kwargs: dict) -> bool:
        return False

    def helper_chunk(self, event: Any) -> Any:
        return event  # no stream helper; kept for the protocol

    def hide_injected(self, result: Any) -> Any:
        return _without_usage(result)


COMPLETIONS: Shape = _CompletionsShape()

Kind = Literal["call", "manager"]

# The OpenAI client's metered methods, by attribute path from the client. Every other attribute is delegated
# unchanged. "call": the method returns the response (or a stream). "manager": it returns a stream manager (the
# SDK's stream helpers), which sends the request when it is entered.
METHODS: dict[tuple[str, ...], tuple[Shape, Kind]] = {
    ("chat", "completions", "create"): (CHAT, "call"),
    ("chat", "completions", "parse"): (CHAT, "call"),
    ("chat", "completions", "stream"): (CHAT, "manager"),
    ("responses", "create"): (RESPONSES, "call"),
    ("responses", "parse"): (RESPONSES, "call"),
    ("responses", "stream"): (RESPONSES, "manager"),
    ("completions", "create"): (COMPLETIONS, "call"),
}
