"""The Anthropic Messages API: the `ANTHROPIC` shape and the Anthropic client's method table.

`wrap()` picks `ANTHROPIC_METHODS` for a client that has `messages` and no `chat` (anthropic.Anthropic and
AsyncAnthropic). `messages.create` / `parse` and `beta.messages.create` / `parse` are "call" kind,
`messages.stream` and `beta.messages.stream` are the SDK's stream helpers ("manager" kind), and
`beta.messages.tool_runner` ("runner" kind) runs its model calls through the wrapped client; raw and streaming
responses go through the generic resource (proxy.py).

Usage: Anthropic reports cache writes and cache reads apart from `input_tokens`. The prompt is charged as
`input_tokens + <cache writes> + cache_read_input_tokens`, with the cache reads as cached tokens. Cache writes are
charged at the provider's write premium, never under it: with the `cache_creation` breakdown,
`ceil(1.25 * ephemeral_5m_input_tokens + 2 * ephemeral_1h_input_tokens)`, plus 2x any part of
`cache_creation_input_tokens` the breakdown does not cover; without it, 2x `cache_creation_input_tokens` (the 1-hour
rate, since the TTL is unknown). A stream carries the input usage on `message_start` (the breakdown appears only
there) and the cumulative counts on `message_delta`; a cache-write total raised by `message_delta` is charged the
excess at 2x. Nothing is injected into the request. A stream that ends before its `message_delta` keeps the prompt
`message_start` showed and estimates the completion from the characters seen.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .shapes import Kind, Observed, Shape, _content_chars, _g, _int, _max_tokens, _text_len, _tools_chars

_USAGE_PARTS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
# Events the stream helper (MessageStream) derives from a raw event it also yields: never fed twice.
_DERIVED = frozenset({"text", "input_json", "citation", "thinking", "signature"})


def _json(v: Any) -> str:
    return json.dumps(v, default=str)


def _block_chars(block: Any) -> int:
    """A request content block: its `text`, a tool_result's `content` (a string or text blocks), a tool_use's
    `input` JSON."""
    if isinstance(block, str):
        return len(block)
    total = _text_len(_g(block, "text"))
    if _g(block, "type") == "tool_result":
        total += _content_chars(_g(block, "content"))
    inp = _g(block, "input")
    if inp is not None:
        total += len(_json(inp))
    return total


def _blocks_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, (list, tuple)):  # never iterate a caller's one-shot iterable
        return sum(_block_chars(b) for b in content)
    return 0


def _usage_parts(u: Any, parts: dict, *, output: bool) -> None:
    """Merge the usage counts `u` reports into `parts` (a message_delta's counts are cumulative: they overwrite;
    counts it omits keep message_start's). `output`: take output_tokens too (message_start's is a placeholder)."""
    for name in _USAGE_PARTS:
        if name == "output_tokens" and not output:
            continue
        v = _int(_g(u, name))
        if v is not None:
            parts[name] = v
    breakdown = _g(u, "cache_creation")
    if breakdown is not None:
        parts["cache_5m"] = _int(_g(breakdown, "ephemeral_5m_input_tokens")) or 0
        parts["cache_1h"] = _int(_g(breakdown, "ephemeral_1h_input_tokens")) or 0


def _cache_write_tokens(parts: dict) -> int:
    """Cache-write tokens at their write premium: 1.25x for the 5-minute cache, 2x for the 1-hour cache, and 2x
    for any write the breakdown does not cover (or all of them, when there is no breakdown)."""
    total = parts.get("cache_creation_input_tokens", 0)
    m5, h1 = parts.get("cache_5m"), parts.get("cache_1h")
    if m5 is None or h1 is None:
        return 2 * total
    return math.ceil(1.25 * m5 + 2 * h1) + 2 * max(total - m5 - h1, 0)


def _apply_usage(parts: dict, obs: Observed) -> None:
    inp = parts.get("input_tokens")
    if inp is not None:
        reads = parts.get("cache_read_input_tokens", 0)
        obs.prompt_tokens = inp + _cache_write_tokens(parts) + reads
        obs.cached_tokens = reads
    out = parts.get("output_tokens")
    if out is not None:
        obs.completion_tokens = out


def _stop(reason: Any, obs: Observed) -> None:
    obs.finish_reason = reason
    obs.refused = reason == "refusal"


class _AnthropicShape:
    """`messages.create` / `parse` / `stream`: system + messages in, content blocks out."""

    name = "anthropic"

    def prepare(self, kwargs: dict, stream: bool) -> tuple[dict, bool]:
        return dict(kwargs), False

    def unprepare(self, kwargs: dict) -> dict:
        return dict(kwargs)

    def estimate_prompt_tokens(self, kwargs: dict) -> int:
        total = _blocks_chars(kwargs.get("system"))
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):
            for m in messages:
                total += _blocks_chars(_g(m, "content"))
        return math.ceil((total + _tools_chars(kwargs)) / 4)

    def max_output_tokens(self, kwargs: dict) -> int:
        return _max_tokens(kwargs, "max_tokens")  # required by the API; 4096 if missing

    def read(self, resp: Any, obs: Observed) -> None:
        parts: dict = {}
        _usage_parts(_g(resp, "usage"), parts, output=True)
        if "input_tokens" in parts and "output_tokens" in parts:
            _apply_usage(parts, obs)
        texts: list[str] = []
        calls = []
        content = _g(resp, "content") or []
        for block in content if isinstance(content, (list, tuple)) else []:
            kind = _g(block, "type")
            if kind == "text":
                text = _g(block, "text")
                if isinstance(text, str):
                    obs.chars += len(text)
                    texts.append(text)
            elif kind == "tool_use":
                name, args = _g(block, "name"), _json(_g(block, "input", {}))
                obs.chars += _text_len(name) + len(args)
                calls.append({"name": name, "arguments": args})
            elif kind == "thinking":  # billed as output: counted for the character estimate only
                obs.chars += _text_len(_g(block, "thinking"))
        obs.content = "".join(texts) if texts else None
        obs.tool_calls = calls
        _stop(_g(resp, "stop_reason"), obs)

    def observe(self, event: Any, obs: Observed, injected: bool) -> bool:
        kind = _g(event, "type")
        state = obs.state
        if kind == "message_start":
            parts = state.setdefault("usage", {})
            _usage_parts(_g(_g(event, "message"), "usage"), parts, output=False)
            _apply_usage(parts, obs)
        elif kind == "content_block_start":
            block, index = _g(event, "content_block"), _g(event, "index", 0)
            btype = _g(block, "type")
            if btype == "tool_use":
                state.setdefault("tools", {})[index] = _g(block, "input")  # the input if no delta follows
                entry = obs.stream_tool(index)
                name = _g(block, "name")
                if isinstance(name, str) and not entry["name"]:
                    obs.chars += len(name)
                    entry["name"] = name
            elif btype == "text":
                text = _g(block, "text")
                if isinstance(text, str) and text:
                    obs.chars += len(text)
                    obs.stream_text(text)
        elif kind == "content_block_delta":
            delta, index = _g(event, "delta"), _g(event, "index", 0)
            dtype = _g(delta, "type")
            if dtype == "text_delta":
                text = _g(delta, "text")
                if isinstance(text, str):
                    obs.chars += len(text)
                    obs.stream_text(text)
            elif dtype == "input_json_delta":
                args = _g(delta, "partial_json")
                if isinstance(args, str):
                    obs.chars += len(args)
                    if index in state.get("tools", {}):
                        obs.stream_tool(index)["arguments"] += args
            elif dtype == "thinking_delta":
                obs.chars += _text_len(_g(delta, "thinking"))
        elif kind == "content_block_stop":
            tools = state.get("tools", {})
            index = _g(event, "index", 0)
            if index in tools:
                entry = obs.stream_tool(index)
                if not entry["arguments"]:  # a tool without input sends no input_json_delta
                    entry["arguments"] = _json(tools[index] if tools[index] is not None else {})
        elif kind == "message_delta":
            parts = state.setdefault("usage", {})
            _usage_parts(_g(event, "usage"), parts, output=True)
            _apply_usage(parts, obs)
            _stop(_g(_g(event, "delta"), "stop_reason"), obs)
        return True

    def pending(self, kwargs: dict) -> bool:
        return False

    def helper_chunk(self, event: Any) -> Any:
        # MessageStream yields the raw events (content_block_stop / message_stop re-built with the snapshot) plus
        # events derived from them (text, input_json, ...), which pass through unfed.
        return None if _g(event, "type") in _DERIVED else event

    def hide_injected(self, result: Any) -> Any:
        return result  # nothing is injected


ANTHROPIC: Shape = _AnthropicShape()

# The Anthropic client's metered methods, by attribute path from the client. `beta.messages` has the same wire
# format (its stream helper is BetaMessageStream); `tool_runner` makes its model calls through the wrapped client
# (beta.messages.parse / stream), each metered as its own call. Batches are not metered.
ANTHROPIC_METHODS: dict[tuple[str, ...], tuple[Shape, Kind]] = {
    ("messages", "create"): (ANTHROPIC, "call"),
    ("messages", "parse"): (ANTHROPIC, "call"),
    ("messages", "stream"): (ANTHROPIC, "manager"),
    ("beta", "messages", "create"): (ANTHROPIC, "call"),
    ("beta", "messages", "parse"): (ANTHROPIC, "call"),
    ("beta", "messages", "stream"): (ANTHROPIC, "manager"),
    ("beta", "messages", "tool_runner"): (ANTHROPIC, "runner"),
}
