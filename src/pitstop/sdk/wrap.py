"""`pitstop.wrap(client)`: meter an OpenAI-compatible client's chat completions against hard budgets.

The wrapper proxies the client. Only `chat.completions.create` is intercepted; every other attribute is
delegated unchanged through `__getattr__` (no global monkeypatching), and `with_options`/`copy` return
wrapped clients so derived clients stay metered. Each intercepted call:

  1. raises a pending `Halted` (check_halt) before anything else;
  2. estimates its worst-case cost (ceil(chars/4) prompt tokens + max completion tokens, default 4096)
     and reserves it in the ledger, which raises `BudgetExceeded` before the client is called;
  3. calls the real `create`, then settles the reservation from `usage` (or from a character estimate
     when usage is missing), records an `llm_call` event and runs the post-call hook.

Streams are returned as proxies that yield every chunk unchanged and settle when the stream ends, fails,
is closed, or is dropped after an early `break`. A reservation is always settled or released, including
on KeyboardInterrupt and asyncio cancellation. Store or ledger failures never reach user code: the call
proceeds unmetered and its event carries `unmetered: true`. The only exceptions raised on purpose are
`Halted` and `BudgetExceeded`.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..budget.ledger import Reservation
from ..budget.prices import cost_usd, price_for
from .context import RunContext, _safe, check_halt, current_run
from .errors import BudgetExceeded
from .runtime import Runtime, get_runtime

log = logging.getLogger("pitstop")

T = TypeVar("T")

DEFAULT_MAX_COMPLETION_TOKENS = 4096
# Client methods that return a new client; their result is wrapped too so it stays metered.
_REWRAP = frozenset({"with_options", "copy"})


def wrap(client: T) -> T:
    """Return a metered proxy of `client`. Wrapping an already wrapped client returns it unchanged."""
    if type(client) is _Wrapped:
        return client
    return _Wrapped(client)  # type: ignore[return-value]


# -- post-call hooks (L0 detection) --------------------------------------------------------------------------


@dataclass
class ResponseSummary:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": ..., "arguments": <raw string>}]
    finish_reason: str | None = None

    @property
    def content_chars(self) -> int:
        return len(self.content) if self.content else 0


def _apply_findings(ctx: RunContext | None, findings: list, *, run_id: str | None = None) -> None:
    """Record each finding with its level and action. A P0 on a high workflow sets a pending halt that the
    next boundary raises. Outside a workflow: workflow=None, criticality medium, sensitivity normal."""
    from ..detect import drifts
    from ..detect.urgency import Level, action_for, classify

    criticality = ctx.spec.criticality if ctx is not None else "medium"
    workflow = ctx.spec.name if ctx is not None else None
    run_id = ctx.run_id if ctx is not None else run_id
    store = get_runtime().store
    for f in findings:
        level = classify(f.drift_type, criticality, f.side_effects)
        action = action_for(f, level, criticality)
        drift_id = drifts.record_drift(store, run_id=run_id, workflow=workflow, finding=f, level=level,
                                       action=action)
        if ctx is not None and level is Level.P0 and criticality == "high" and ctx.halt_reason is None:
            ctx.halt_reason = f"{f.drift_type} (P0) on high-criticality workflow {workflow!r}"
            ctx.halt_drift_id = drift_id


def _after_llm_call(ctx: RunContext | None, summary: ResponseSummary) -> None:
    """Post-call L0 checks: build the call record, update the run's history, record findings as drifts."""
    from ..detect.l0 import LLMCallRecord, check_llm_call, is_refusal

    failures = 0
    for tc in summary.tool_calls:
        try:
            json.loads(_g(tc, "arguments"))
        except Exception:
            failures += 1
    record = LLMCallRecord(failures, len(summary.tool_calls),
                           is_refusal(summary.content, len(summary.tool_calls)))
    history = ctx.llm_history if ctx is not None else []
    sensitivity = ctx.spec.sensitivity if ctx is not None else "normal"
    findings = check_llm_call(history, record, sensitivity)
    if ctx is not None:
        ctx.llm_history.append(record)
    _apply_findings(ctx, findings)


def _on_budget_exceeded(ctx: RunContext | None, exc: BudgetExceeded) -> None:
    """Record a budget_exhausted drift (P0, halt) before the exception is raised."""
    from ..detect.l0 import Finding

    _apply_findings(ctx, [Finding("budget_exhausted", "degraded", False,
                                  {"scope": exc.scope, "limit_usd": exc.limit_usd, "spent_usd": exc.spent_usd})])


# -- proxies --------------------------------------------------------------------------------------------


class _Wrapped:
    __slots__ = ("_pitstop_client", "_pitstop_chat")

    def __init__(self, client: Any):
        object.__setattr__(self, "_pitstop_client", client)
        object.__setattr__(self, "_pitstop_chat", None)

    @property
    def __class__(self):  # isinstance(wrapped, OpenAI) stays True
        return type(self._pitstop_client)

    @property
    def chat(self) -> _Chat:
        inner = self._pitstop_client.chat
        cached = self._pitstop_chat
        if cached is None or cached._inner is not inner:
            cached = _Chat(self._pitstop_client, inner)
            object.__setattr__(self, "_pitstop_chat", cached)
        return cached

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        attr = getattr(self._pitstop_client, name)
        if name in _REWRAP and callable(attr):
            @functools.wraps(attr)
            def rewrap(*a, **k):
                return wrap(attr(*a, **k))
            return rewrap
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._pitstop_client, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._pitstop_client, name)

    def __dir__(self):
        return dir(self._pitstop_client)

    def __repr__(self) -> str:
        return f"pitstop.wrap({self._pitstop_client!r})"

    def __enter__(self):
        r = self._pitstop_client.__enter__()
        return self if r is self._pitstop_client else r

    def __exit__(self, *exc):
        return self._pitstop_client.__exit__(*exc)

    async def __aenter__(self):
        r = await self._pitstop_client.__aenter__()
        return self if r is self._pitstop_client else r

    async def __aexit__(self, *exc):
        return await self._pitstop_client.__aexit__(*exc)


class _Chat:
    def __init__(self, client: Any, inner: Any):
        self._client, self._inner, self._completions = client, inner, None

    @property
    def completions(self) -> _Completions:
        inner = self._inner.completions
        cached = self._completions
        if cached is None or cached._inner is not inner:
            cached = self._completions = _Completions(self._client, inner)
        return cached

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _is_async(fn: Any) -> bool:
    # openai's AsyncCompletions.create is a sync def (functools.wraps) around an async def, hence unwrap.
    try:
        return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(inspect.unwrap(fn))
    except Exception:
        return False


class _Completions:
    def __init__(self, client: Any, inner: Any):
        self._client, self._inner = client, inner
        real = getattr(inner, "create", None)
        if _is_async(real):
            async def create(*args, **kwargs):
                return await self._acall(args, kwargs)
        else:
            def create(*args, **kwargs):
                return self._call(args, kwargs)
        if real is not None:
            try:
                functools.update_wrapper(create, real)
            except Exception:
                pass
        self.create = create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _call(self, args: tuple, kwargs: dict) -> Any:
        call = _Call.begin(self._client, kwargs)
        handed_off = False
        try:
            result = self._inner.create(*args, **call.kwargs)
            if inspect.isawaitable(result):  # an async create we failed to detect
                handed_off = True
                return _await_then(call, result)
            call.got_result = True
            if call.stream:
                out = _wrap_stream(result, call)
                handed_off = out is not result
                return out
            call.complete(result)
            return result
        finally:
            if not handed_off:
                call.close()

    async def _acall(self, args: tuple, kwargs: dict) -> Any:
        call = _Call.begin(self._client, kwargs)
        handed_off = False
        try:
            result = self._inner.create(*args, **call.kwargs)
            if inspect.isawaitable(result):
                result = await result
            call.got_result = True
            if call.stream:
                out = _wrap_stream(result, call)
                handed_off = out is not result
                return out
            call.complete(result)
            return result
        finally:
            if not handed_off:
                call.close()


async def _await_then(call: _Call, awaitable: Any) -> Any:
    handed_off = False
    try:
        result = await awaitable
        call.got_result = True
        if call.stream:
            out = _wrap_stream(result, call)
            handed_off = out is not result
            return out
        call.complete(result)
        return result
    finally:
        if not handed_off:
            call.close()


def _wrap_stream(result: Any, call: _Call) -> Any:
    if hasattr(result, "__aiter__"):
        return _AsyncStream(result, call)
    if hasattr(result, "__iter__"):
        return _SyncStream(result, call)
    return result  # unknown shape: returned as is, settled by estimate in the caller's finally


class _SyncStream:
    """Yields the provider's chunks unchanged; settles the reservation when the stream ends or is dropped."""

    def __init__(self, inner: Any, call: _Call):
        self._pitstop_inner, self._pitstop_call, self._pitstop_it = inner, call, None

    def __iter__(self):
        return self

    def __next__(self):
        call = self._pitstop_call
        if self._pitstop_it is None:
            self._pitstop_it = iter(self._pitstop_inner)
        while True:
            try:
                chunk = next(self._pitstop_it)
            except BaseException:  # StopIteration, provider errors, KeyboardInterrupt
                call.close()
                raise
            if call.observe(chunk):
                return chunk

    def close(self) -> None:
        try:
            close = getattr(self._pitstop_inner, "close", None)
            if close is not None:
                close()
        finally:
            self._pitstop_call.close()

    def __enter__(self):
        enter = getattr(self._pitstop_inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc):
        try:
            exit_ = getattr(self._pitstop_inner, "__exit__", None)
            return exit_(*exc) if exit_ is not None else None
        finally:
            self._pitstop_call.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return getattr(self._pitstop_inner, name)

    def __del__(self):
        try:
            self._pitstop_call.close()
        except Exception:
            pass


class _AsyncStream:
    """Async counterpart of _SyncStream."""

    def __init__(self, inner: Any, call: _Call):
        self._pitstop_inner, self._pitstop_call, self._pitstop_it = inner, call, None

    def __aiter__(self):
        return self

    async def __anext__(self):
        call = self._pitstop_call
        if self._pitstop_it is None:
            self._pitstop_it = self._pitstop_inner.__aiter__()
        while True:
            try:
                chunk = await self._pitstop_it.__anext__()
            except BaseException:  # StopAsyncIteration, provider errors, CancelledError
                call.close()
                raise
            if call.observe(chunk):
                return chunk

    async def close(self) -> None:
        await self._pitstop_finish_with("close")

    async def aclose(self) -> None:
        await self._pitstop_finish_with("aclose")

    async def _pitstop_finish_with(self, method: str) -> None:
        try:
            fn = getattr(self._pitstop_inner, method, None)
            if fn is not None:
                r = fn()
                if inspect.isawaitable(r):
                    await r
        finally:
            self._pitstop_call.close()

    async def __aenter__(self):
        enter = getattr(self._pitstop_inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc):
        try:
            exit_ = getattr(self._pitstop_inner, "__aexit__", None)
            return await exit_(*exc) if exit_ is not None else None
        finally:
            self._pitstop_call.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_pitstop"):
            raise AttributeError(name)
        return getattr(self._pitstop_inner, name)

    def __del__(self):
        try:
            self._pitstop_call.close()
        except Exception:
            pass


# -- one metered call -------------------------------------------------------------------------------------


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


def _estimate_prompt_tokens(kwargs: dict) -> int:
    total = 0
    messages = kwargs.get("messages")
    if isinstance(messages, (list, tuple)):  # never iterate a caller's one-shot iterable
        for m in messages:
            content = _g(m, "content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, (list, tuple)):
                for part in content:
                    text = part if isinstance(part, str) else _g(part, "text")
                    if isinstance(text, str):
                        total += len(text)
    tools = kwargs.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        total += len(json.dumps(tools, default=str))
    return math.ceil(total / 4)


def _max_completion_tokens(kwargs: dict) -> int:
    for key in ("max_tokens", "max_completion_tokens"):
        v = _int(kwargs.get(key))
        if v is not None and v >= 0:
            return v
    return DEFAULT_MAX_COMPLETION_TOKENS


def _base_url(client: Any) -> str | None:
    try:
        url = getattr(client, "base_url", None)
        return str(url) if url is not None else None
    except Exception:
        return None


class _Call:
    """State of one intercepted call: its reservation, what the response showed, and whether it is closed."""

    def __init__(self, ctx: RunContext | None, kwargs: dict, model: str, base_url: str | None, stream: bool,
                 inject_usage: bool):
        self.ctx = ctx
        self.agent = ctx.agent if ctx is not None else None
        self.kwargs = kwargs
        self.model = model
        self.base_url = base_url
        self.stream = stream
        self.inject_usage = inject_usage
        self.est_prompt = 0
        self.rt: Runtime | None = None
        self.res: Reservation | None = None
        self.unmetered = False
        self.got_result = False
        self.closed = False
        # what the response showed
        self.usage: Any = None
        self.summary: ResponseSummary | None = None
        self.chars = 0  # all completion text (content, tool names and arguments), for the usage estimate
        self._content: list[str] = []
        self._tools: dict[int, dict] = {}
        self._finish: str | None = None

    @classmethod
    def begin(cls, client: Any, kwargs: dict) -> _Call:
        check_halt()
        ctx = current_run()
        kwargs = dict(kwargs)
        stream = kwargs.get("stream") is True
        inject = stream and "stream_options" not in kwargs
        if inject:
            kwargs["stream_options"] = {"include_usage": True}
        model = kwargs.get("model")
        model = model if isinstance(model, str) else "unknown" if model is None else str(model)
        call = cls(ctx, kwargs, model, _base_url(client), stream, inject)
        try:
            call.est_prompt = _estimate_prompt_tokens(kwargs)
            call.rt = get_runtime()
            call.res = call.rt.ledger.reserve(
                model=call.model, base_url=call.base_url, est_prompt_tokens=call.est_prompt,
                max_completion_tokens=_max_completion_tokens(kwargs),
                run_id=ctx.run_id if ctx is not None else None,
                workflow=ctx.spec.name if ctx is not None else None,
                workflow_budget_usd=ctx.spec.budget_usd if ctx is not None else None,
                meter="agent",
            )
        except BudgetExceeded as e:
            _safe(functools.partial(_on_budget_exceeded, ctx, e))
            raise
        except Exception:
            log.warning("pitstop: budget ledger unavailable; this call proceeds unmetered", exc_info=True)
            call.unmetered = True
        return call

    # -- observing the response --

    def complete(self, resp: Any) -> None:
        """Read a non-stream response, then settle and record it."""
        try:
            self.usage = _g(resp, "usage")
            choices = _g(resp, "choices") or []
            choices = list(choices) if isinstance(choices, (list, tuple)) else []
            tool_calls, content, finish = [], None, None
            for i, ch in enumerate(choices):
                msg = _g(ch, "message")
                text = _g(msg, "content")
                if isinstance(text, str):
                    self.chars += len(text)
                calls = []
                for tc in _g(msg, "tool_calls") or []:
                    fn = _g(tc, "function")
                    name, args = _g(fn, "name"), _g(fn, "arguments")
                    self.chars += len(name) if isinstance(name, str) else 0
                    self.chars += len(args) if isinstance(args, str) else 0
                    calls.append({"name": name, "arguments": args})
                if i == 0:
                    content = text if isinstance(text, str) else None
                    tool_calls, finish = calls, _g(ch, "finish_reason")
            self.summary = ResponseSummary(content, tool_calls, finish)
        except Exception:
            log.warning("pitstop: could not read the model response", exc_info=True)
        self.close()

    def observe(self, chunk: Any) -> bool:
        """Accumulate one stream chunk. Returns False for the usage-only chunk PitStop itself asked for."""
        try:
            usage = _g(chunk, "usage")
            if usage is not None:
                self.usage = usage
            choices = _g(chunk, "choices") or []
            for ch in choices:
                first = (_g(ch, "index", 0) or 0) == 0
                delta = _g(ch, "delta")
                text = _g(delta, "content")
                if isinstance(text, str):
                    self.chars += len(text)
                    if first:
                        self._content.append(text)
                for tc in _g(delta, "tool_calls") or []:
                    fn = _g(tc, "function")
                    name, args = _g(fn, "name"), _g(fn, "arguments")
                    entry = self._tools.setdefault(_g(tc, "index", 0) or 0, {"name": None, "arguments": ""}) \
                        if first else None
                    if isinstance(name, str) and name and not (entry is not None and entry["name"]):
                        self.chars += len(name)  # a name arrives once; providers that repeat it are not re-counted
                        if entry is not None:
                            entry["name"] = name
                    if isinstance(args, str):
                        self.chars += len(args)
                        if entry is not None:
                            entry["arguments"] += args
                reason = _g(ch, "finish_reason")
                if first and reason:
                    self._finish = reason
            if self.inject_usage and not choices and usage is not None:
                # The caller did not ask for include_usage; hide the extra chunk so the stream looks exactly
                # like the unwrapped client's (code like chunk.choices[0] must keep working).
                return False
        except Exception:
            log.warning("pitstop: could not read a stream chunk", exc_info=True)
        return True

    # -- closing --

    def close(self) -> None:
        """Settle (the call produced a result) or release (it did not). Idempotent; internal errors are logged."""
        if self.closed:
            return
        self.closed = True
        if not self.got_result:
            if self.res is not None:
                try:
                    self.rt.ledger.release(self.res)
                except Exception:
                    log.warning("pitstop: could not release a reservation", exc_info=True)
            return
        if self.summary is None:
            content = "".join(self._content) if self._content else None
            tools = [self._tools[k] for k in sorted(self._tools)]
            self.summary = ResponseSummary(content, tools, self._finish)
        prompt, completion, cached, estimated = self._tokens()
        cost: float | None = None
        if self.res is not None:
            try:
                cost = self.rt.ledger.settle(self.res, prompt_tokens=prompt, completion_tokens=completion,
                                             cached_tokens=cached)
            except Exception:
                # The reservation stays 'reserved': spend is over-counted, never under-counted.
                log.warning("pitstop: could not settle a reservation", exc_info=True)
                cost = self.res.usd
        else:
            cost = _safe(lambda: self._unmetered_cost(prompt, completion, cached))
        _safe(lambda: self._record(cost, estimated))
        summary = self.summary
        _safe(lambda: _after_llm_call(self.ctx, summary))

    def _tokens(self) -> tuple[int, int, int, bool]:
        u = self.usage
        prompt, completion = _int(_g(u, "prompt_tokens")), _int(_g(u, "completion_tokens"))
        if prompt is None or completion is None:
            return self.est_prompt, math.ceil(self.chars / 4), 0, True
        cached = _int(_g(_g(u, "prompt_tokens_details"), "cached_tokens")) or 0
        return prompt, completion, cached, False

    def _rt(self) -> Runtime:
        return self.rt if self.rt is not None else get_runtime()

    def _price_known(self) -> bool:
        if self.res is not None:
            return self.res.price_known
        return price_for(self.model, self._rt().config.prices)[1]

    def _unmetered_cost(self, prompt: int, completion: int, cached: int) -> float:
        return cost_usd(price_for(self.model, self._rt().config.prices)[0], prompt, completion, cached)

    def _record(self, cost: float | None, estimated: bool) -> None:
        from .. import __version__

        s = self.summary
        payload: dict[str, Any] = {"tool_calls": s.tool_calls, "finish_reason": s.finish_reason,
                                   "content_chars": s.content_chars}
        if estimated:
            payload["usage_estimated"] = True
        if self.unmetered:
            payload["unmetered"] = True
        ctx = self.ctx
        first = False
        if ctx is not None:
            first, ctx.llm_calls = ctx.llm_calls == 0, ctx.llm_calls + 1
        rt = self._rt()
        known = _safe(self._price_known)
        with rt.store.transaction() as c:
            c.execute(
                "INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                (ctx.run_id if ctx else None, time.time(), self.agent, "llm_call", self.model,
                 json.dumps(payload, default=str), cost),
            )
            if ctx is None:
                return
            if first:
                c.execute(
                    "UPDATE runs SET model=?, base_url=?, adapter='wrap', pitstop_version=?, price_known=?"
                    " WHERE id=?",
                    (self.model, self.base_url, __version__, 0 if known is False else 1, ctx.run_id),
                )
            elif known is False:
                c.execute("UPDATE runs SET price_known=0 WHERE id=?", (ctx.run_id,))
