"""One metered model call: reserve -> call -> settle -> record -> detect, the same for every Shape.

See the package docstring (`pitstop.sdk.wrap`) for the behavior contract.
"""
from __future__ import annotations

import functools
import json
import logging
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

from ...userconfig import ConfigInvalid
from ...budget.ledger import Reservation
from ...budget.prices import cost_usd, price_for
from ...store.urls import normalize_base_url
from ..actions import apply_findings
from ..context import RunContext, _meter, _safe, check_halt, current_run
from ..errors import BudgetExceeded
from ..runtime import Runtime, config_halt, get_runtime
from .shapes import Observed, Shape, _g

log = logging.getLogger("pitstop")

# Normalized base_urls whose stream calls rejected PitStop's injected stream_options with a 400 and then
# succeeded without it. Process-level: later stream calls to these endpoints are sent without the injection.
_NO_STREAM_OPTIONS: set[str] = set()

Mode = Literal["normal", "raw", "streaming"]  # "raw": with_raw_response; "streaming": with_streaming_response


class MeteredCell:
    """Set by `metered_cell()` around a framework's model call; `metered` turns True when a wrapped call made inside
    it records its `llm_call` event (any meter). Shared by every context copied from the one that set it."""
    __slots__ = ("metered",)

    def __init__(self) -> None:
        self.metered = False


_metered_cell: ContextVar[MeteredCell | None] = ContextVar("pitstop_metered_cell", default=None)


@contextmanager
def metered_cell() -> Iterator[MeteredCell]:
    """A fresh MeteredCell for the block: an adapter wraps one model call in it and, when the cell stays unmarked,
    knows no `pitstop.wrap`-ed client metered that call (and may record it as observed)."""
    cell = MeteredCell()
    token = _metered_cell.set(cell)
    try:
        yield cell
    finally:
        try:
            _metered_cell.reset(token)
        except ValueError:  # exited from another Context: leave that context's cell alone
            pass


# -- post-call hooks (L0 detection) --------------------------------------------------------------------------


def _refused(obs: Observed) -> bool:
    """Whether the reply is a refusal. A provider-declared refusal (Anthropic's stop_reason "refusal") counts
    whatever the text says."""
    from ...detect.l0 import is_refusal

    return bool(obs.refused or is_refusal(obs.content, len(obs.tool_calls)))


def _after_llm_call(ctx: RunContext | None, obs: Observed, refused: bool) -> None:
    """Post-call L0 checks: build the call record, update the run's history and refusal count, record findings
    as drifts. `refused` is `_refused(obs)`, computed once for the event and for detection."""
    from ...detect.l0 import LLMCallRecord, check_llm_call

    failures = 0
    for tc in obs.tool_calls:
        if _g(tc, "kind") is not None:  # a custom or hosted call: its arguments are not JSON by design
            continue
        try:
            json.loads(_g(tc, "arguments"))
        except Exception:
            failures += 1
    record = LLMCallRecord(failures, len(obs.tool_calls), refused)
    if ctx is None:
        findings = check_llm_call([], record, "normal", 1 if record.refused else 0)
    else:
        with ctx.lock:
            if record.refused:
                ctx.refusals += 1
            findings = check_llm_call(ctx.llm_history, record, ctx.spec.sensitivity, ctx.refusals)
            ctx.llm_history.append(record)
    apply_findings(ctx, findings)


def _after_settle(ctx: RunContext, usd: float) -> None:
    """The spend-rate P0 (`pitstop.detect.spend`): add a settled agent-meter call to the run's spend and, past N x
    the workflow's baseline, apply one `spend_spike` (P0, halt) per run. The halt is pending: the call that
    settled returns normally and the run's next call raises. Runs outside any store transaction."""
    from ...detect import spend
    from ..context import UNSET

    with ctx.lock:
        ctx.spend_usd += usd
        run_usd, baseline = ctx.spend_usd, ctx.spend_baseline
        if ctx.spend_spiked:
            return
    if baseline is UNSET:  # once per run: the first settle reads the store; a failed read is retried next time
        baseline = spend.baseline_run_cost(get_runtime().store, ctx.spec.name, before=ctx.started_at)
        with ctx.lock:
            if ctx.spend_baseline is UNSET:
                ctx.spend_baseline = baseline
            baseline = ctx.spend_baseline
    finding = spend.check_spend(run_usd, baseline, ctx.spec.sensitivity)
    if finding is None:
        return
    with ctx.lock:
        if ctx.spend_spiked:
            return
        ctx.spend_spiked = True
    apply_findings(ctx, [finding])


def _on_budget_exceeded(ctx: RunContext | None, exc: BudgetExceeded) -> None:
    """Record a budget_exhausted drift (P0, halt) before the exception is raised."""
    from ...detect.l0 import Finding

    apply_findings(ctx, [Finding("budget_exhausted", "degraded", False,
                                 {"scope": exc.scope, "limit_usd": exc.limit_usd, "spent_usd": exc.spent_usd})])


# -- one metered call -------------------------------------------------------------------------------------


def _base_url(client: Any) -> str | None:
    try:
        url = getattr(client, "base_url", None)
        return str(url) if url is not None else None
    except Exception:
        return None


def _fallback_from(ctx: RunContext | None, meter: str, kwargs: dict) -> str | None:
    """The requested model when this call goes to the workflow's pinned fallback instead, else None. Only the
    agent's own calls are swapped. A store that cannot be read keeps the requested model (fail open)."""
    if meter != "agent" or ctx is None or not ctx.spec.fallback_model:
        return None
    from .. import fallback

    target = ctx.spec.fallback_model
    requested = kwargs.get("model")
    if requested == target or _safe(lambda: fallback.active(get_runtime().store, ctx.spec.name)) is None:
        return None
    if requested is None:
        return "(unset)"  # no model kwarg (a client-side default, say): never the string "None"
    return requested if isinstance(requested, str) else str(requested)


class _Call:
    """State of one intercepted call: its reservation, what the response showed, and whether it is closed."""

    def __init__(self, ctx: RunContext | None, shape: Shape, kwargs: dict, model: str, base_url: str | None,
                 stream: bool, injected: bool, mode: Mode = "normal"):
        self.ctx = ctx
        self.agent = ctx.agent if ctx is not None else None
        self.shape = shape
        self.kwargs = kwargs  # what is sent to the provider
        self.model = model
        self.base_url = base_url
        self.stream = stream
        self.injected = injected  # kwargs carry PitStop's own usage request (CHAT: stream_options)
        self.mode = mode
        # with_streaming_response, and with_raw_response streams, can't be read: charged the full reservation
        self.worst_case = mode == "streaming" or (mode == "raw" and stream)
        self.meter = "agent"  # the ledger meter, read once at begin
        self.fallback_from: str | None = None  # the requested model, when the workflow's fallback replaced it
        self.est_prompt = 0
        self.max_output = 0
        self.rt: Runtime | None = None
        self.res: Reservation | None = None
        self.unmetered = False
        self.retried = False
        self.got_result = False
        self.closed = False
        self.obs = Observed()
        self.cell = _metered_cell.get()  # the adapter's per-model-call cell, marked when this call is recorded

    @classmethod
    def begin(cls, client: Any, shape: Shape, kwargs: dict, mode: Mode = "normal",
              stream: bool | None = None) -> _Call:
        """`stream`: None reads `stream=True` from kwargs; a stream helper (which streams without it) passes True."""
        check_halt()
        ctx = current_run()
        meter = _meter.get()
        fallback_from = _fallback_from(ctx, meter, kwargs)
        if fallback_from is not None:  # before the estimate and reservation: the pinned model is what is billed
            kwargs = {**kwargs, "model": ctx.spec.fallback_model}
        stream = kwargs.get("stream") is True if stream is None else stream
        base_url = _base_url(client)
        if mode == "normal" and normalize_base_url(base_url) not in _NO_STREAM_OPTIONS:
            send, injected = shape.prepare(kwargs, stream)
        else:  # raw and streaming responses reach the caller as bytes: nothing is injected into them
            send, injected = dict(kwargs), False
        model = kwargs.get("model")
        model = model if isinstance(model, str) else "unknown" if model is None else str(model)
        call = cls(ctx, shape, send, model, base_url, stream, injected, mode)
        call.meter = meter
        call.fallback_from = fallback_from
        call.obs.pending = _safe(lambda: shape.pending(kwargs)) is True
        try:
            call.est_prompt = shape.estimate_prompt_tokens(kwargs)
            call.max_output = shape.max_output_tokens(kwargs)
            call.rt = get_runtime()
            call.res = call.rt.ledger.reserve(
                model=call.model, base_url=call.base_url, est_prompt_tokens=call.est_prompt,
                max_completion_tokens=call.max_output,
                run_id=ctx.run_id if ctx is not None else None,
                workflow=ctx.spec.name if ctx is not None else None,
                workflow_budget_usd=ctx.spec.budget_usd if ctx is not None else None,
                meter=call.meter,
            )
        except BudgetExceeded as e:
            _safe(functools.partial(_on_budget_exceeded, ctx, e))
            raise
        except ConfigInvalid as e:  # fail closed: a broken config must never silently drop the hard budget
            raise config_halt(e) from e
        except Exception:  # transient store trouble (busy/locked DB): fail open, flagged on the event
            log.warning("pitstop: budget ledger unavailable; this call proceeds unmetered", exc_info=True)
            call.unmetered = True
        return call

    # -- the stream_options retry --

    def retryable(self, exc: BaseException) -> bool:
        """A 400 on a call carrying PitStop's injection: the endpoint may reject stream_options. Retried once."""
        return self.injected and not self.retried and isinstance(exc, Exception) and _g(exc, "status_code") == 400

    def drop_injection(self) -> None:
        self.retried = True
        self.kwargs = self.shape.unprepare(self.kwargs)
        self.injected = False

    def retry_succeeded(self) -> None:
        url = normalize_base_url(self.base_url)
        if url is not None:
            _NO_STREAM_OPTIONS.add(url)

    # -- observing the response --

    def complete(self, resp: Any) -> None:
        """Read a non-stream response, then settle and record it."""
        try:
            self.shape.read(resp, self.obs)
        except Exception:
            log.warning("pitstop: could not read the model response", exc_info=True)
        self.close()

    def observe(self, chunk: Any) -> bool:
        """Accumulate one stream chunk. Returns False for a chunk PitStop itself asked for (hidden)."""
        try:
            return self.shape.observe(chunk, self.obs, self.injected) is not False
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
        _safe(self.obs.seal)
        if self.obs.pending and (self.obs.prompt_tokens is None or self.obs.completion_tokens is None):
            self.worst_case = True  # a background response: the output we did not see is billed anyway
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
        # A response charged worst case was never read: it is not known to be a refusal.
        refused = not self.worst_case and _safe(lambda: _refused(self.obs)) is True
        _safe(lambda: self._record(cost, estimated, prompt, completion, refused))
        # Only the agent's own calls feed L0 detection, and only with what the response showed.
        if self.meter == "agent" and not self.worst_case:
            obs = self.obs
            _safe(lambda: _after_llm_call(self.ctx, obs, refused))
        # Every metered agent call counts toward the spend spike, a worst-case charge included.
        if self.meter == "agent" and self.ctx is not None and self.res is not None and cost is not None:
            _safe(lambda: _after_settle(self.ctx, cost))

    def _tokens(self) -> tuple[int, int, int, bool | str]:
        if self.worst_case:
            return self.est_prompt, self.max_output, 0, "worst_case"
        o = self.obs
        if o.prompt_tokens is None or o.completion_tokens is None:
            # Usage not (fully) seen. A part the stream did show is used as is (Anthropic's message_start carries
            # the input usage; its output count arrives only at the end); the rest is estimated.
            if o.prompt_tokens is not None:
                prompt, cached = o.prompt_tokens, o.cached_tokens
            else:
                prompt, cached = self.est_prompt, 0
            completion = o.completion_tokens if o.completion_tokens is not None else math.ceil(o.chars / 4)
            return prompt, completion, cached, True
        return o.prompt_tokens, o.completion_tokens, o.cached_tokens, False

    def _rt(self) -> Runtime:
        return self.rt if self.rt is not None else get_runtime()

    def _price_known(self) -> bool:
        if self.res is not None:
            return self.res.price_known
        return price_for(self.model, self._rt().config.prices)[1]

    def _unmetered_cost(self, prompt: int, completion: int, cached: int) -> float:
        return cost_usd(price_for(self.model, self._rt().config.prices)[0], prompt, completion, cached)

    def _record(self, cost: float | None, estimated: bool | str, prompt: int, completion: int,
                refused: bool) -> None:
        """Write the `llm_call` event (its tokens are the ones settled, or estimated) and the run's tags: the first
        agent call sets model, base_url and adapter (an adapter that tagged the run first keeps its tag), and a
        call redirected to the fallback model records the first model it replaced. Marks the adapter's
        MeteredCell, if one was set when the call began."""
        from ... import __version__

        if self.cell is not None:
            self.cell.metered = True
        o = self.obs
        payload: dict[str, Any] = {"tool_calls": o.tool_calls, "finish_reason": o.finish_reason,
                                   "content_chars": o.content_chars, "meter": self.meter,
                                   "prompt_tokens": prompt, "completion_tokens": completion, "refused": refused}
        if estimated:
            payload["usage_estimated"] = estimated  # True (character estimate) or "worst_case"
        if self.unmetered:
            payload["unmetered"] = True
        if self.fallback_from is not None:
            payload["fallback_from"] = self.fallback_from
        ctx = self.ctx
        first = False
        if ctx is not None and self.meter == "agent":  # judge/supervisor calls never set the run's tags
            with ctx.lock:
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
                # price_known only ever drops: an earlier unknown-priced call (any meter) keeps the run's 0
                c.execute(
                    "UPDATE runs SET model=?, base_url=?, adapter=COALESCE(adapter, 'wrap'), pitstop_version=?,"
                    " price_known=CASE WHEN ? THEN price_known ELSE 0 END WHERE id=?",
                    (self.model, normalize_base_url(self.base_url), __version__, known is not False, ctx.run_id),
                )
            elif known is False:  # any meter: an unknown-priced call makes the run's cost a lower bound
                c.execute("UPDATE runs SET price_known=0 WHERE id=?", (ctx.run_id,))
            if self.fallback_from is not None:  # set only on agent-meter calls (_fallback_from)
                c.execute("UPDATE runs SET fallback_from=COALESCE(fallback_from, ?) WHERE id=?",
                          (self.fallback_from, ctx.run_id))
