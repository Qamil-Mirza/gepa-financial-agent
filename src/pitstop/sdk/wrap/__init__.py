"""`pitstop.wrap(client)`: meter an OpenAI-compatible or Anthropic client's model calls against hard budgets.

The wrapper proxies the client (no global monkeypatching). A per-client method table names the metered methods
by attribute path; for OpenAI-compatible clients, `chat.completions.create` / `parse` / `stream`,
`responses.create` / `parse` / `stream` and legacy `completions.create`; for Anthropic clients (a client with
`messages` and no `chat`), `messages.create` / `parse` / `stream` and the same under `beta.messages`
(anthropic.py). Each path routes to a `Shape`
(shapes.py), which knows that API's request and response format. Every other
attribute is delegated unchanged, and `with_options`/`copy` return wrapped clients so derived clients stay
metered. `with_raw_response` and `with_streaming_response`, on the client or on any resource on the way to a
metered method, stay metered too. Each metered call:

  1. raises a pending `Halted` (check_halt) before anything else;
  2. estimates its worst-case cost (ceil(chars/4) prompt tokens + max completion tokens, default 4096)
     and reserves it in the ledger, which raises `BudgetExceeded` before the client is called;
  3. calls the real method, then settles the reservation from `usage` (or from a character estimate
     when usage is missing), records an `llm_call` event (its payload names the ledger meter) and, for
     calls on the "agent" meter only, runs the post-call L0 checks and sets the run's first-call tags.
     Judge, supervisor and other internal calls inside a run never feed detection or the run's model tag.

Streams are returned as proxies that yield every chunk unchanged (and still pass `isinstance` checks against
the SDK's stream class) and settle when the stream ends, fails, is closed, or is dropped after an early
`break`. To read a stream's usage, chat streams are sent with `stream_options={"include_usage": True}` unless
the caller set `stream_options`; the extra usage chunk is hidden from the caller. If an endpoint rejects that
injection with a 400, the call is retried once without it, and once a retry succeeds, later stream calls to
the same base_url skip the injection (they settle by estimate). Legacy completions streams are handled the same
way; Responses streams carry usage on their `response.completed` event and Anthropic streams on `message_start` /
`message_delta`, so nothing is injected into them.

Stream helpers (`chat.completions.stream`, `responses.stream`) reserve when called. Entering the manager sends the
request and yields the SDK's own stream class (proxied: every event passes through unchanged, and
`get_final_completion()` / `get_final_response()` / `get_final_message()` / `get_final_text()` / `until_done()` and
Anthropic's `text_stream` read the stream through the proxy). The call
settles when the manager exits; a manager that is never entered releases its reservation when it is garbage
collected. An error raised mid-stream that carries the billed `completion` (LengthFinishReasonError) settles from
its usage, or at the full reservation when no usage was seen. A Responses call with `background=True` (or a result
still `queued` / `in_progress`) is charged its full reservation unless its final usage is seen: its output is
billed after the call returns.

Raw responses: a non-stream `with_raw_response` call is settled from `result.parse()`. Responses that can't be
read without consuming them for the caller, `with_streaming_response` and `with_raw_response` with
`stream=True`, are charged their full reservation at return (`usage_estimated: "worst_case"`) and feed no
detection: over-counted, never under-counted.

A reservation is always settled or released, including on KeyboardInterrupt and asyncio cancellation. Store or
ledger failures never reach user code: the call proceeds unmetered and its event carries `unmetered: true`. An
invalid config.toml is different: the call fails closed with `Halted("pitstop config invalid: ...")` before the
client is called. The only exceptions raised on purpose are `Halted` and `BudgetExceeded`.
"""
from __future__ import annotations

import contextlib
from typing import Any, TypeVar

from .anthropic import ANTHROPIC, ANTHROPIC_METHODS
from ..context import _meter  # re-exported: the ContextVar lives with the run and agent vars
from .proxy import Table, _Wrapped
from .shapes import CHAT, COMPLETIONS, METHODS, RESPONSES, Observed, Shape

__all__ = ["wrap", "meter", "Observed", "Shape", "CHAT", "RESPONSES", "COMPLETIONS", "METHODS", "ANTHROPIC",
           "ANTHROPIC_METHODS"]

T = TypeVar("T")


@contextlib.contextmanager
def meter(name: str):
    """Tag every metered call made inside the block with ledger meter `name` (default "agent"). Exiting restores
    the meter from before the block, also when the block is exited in another Context than the one it entered."""
    prev = _meter.get()
    token = _meter.set(name)
    try:
        yield
    finally:
        try:
            _meter.reset(token)
        except ValueError:  # the token belongs to another Context (as in _GenRun._pop)
            _meter.set(prev)


def _table_for(client: Any) -> Table:
    """The method table for this client's type: Anthropic's (`messages`, no `chat`) or the OpenAI-compatible one."""
    if hasattr(client, "messages") and not hasattr(client, "chat"):
        return ANTHROPIC_METHODS
    return METHODS


def wrap(client: T) -> T:
    """Return a metered proxy of `client`. Wrapping an already wrapped client returns it unchanged."""
    if type(client) is _Wrapped:
        return client
    return _Wrapped(client, _table_for(client))  # type: ignore[return-value]
