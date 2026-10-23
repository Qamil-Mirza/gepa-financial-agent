"""`@pitstop.tool`: guard a tool function with scope and loop checks (block, steer, halt).

Outside a workflow the decorator just calls the function. Inside one, each invocation first raises a pending
`Halted`, runs the L0 tool checks against the current agent's recent calls, records any drift, and then either
blocks the call (side-effect tools), steers (read-only tools, at most 2 per run, then halts) or runs it.
The history check and the steer check-and-increment run under the run's lock, so threads sharing a run cannot
exceed the steer limit.
A side-effect tool on a high or medium workflow that L0 lets through then goes to the L1 pre-action gate
(detect/l1.py): a judge decides whether the call serves the goal. A rejection is an out_of_scope_action drift and
blocks the call; when the judge is missing or cannot decide, the workflow's `on_gate_error` policy blocks or runs it.
Async tools run the gate in a worker thread (asyncio.to_thread copies the run's context), never on the event loop.
Exceptions from the user's function propagate unchanged; PitStop's own failures never break the call."""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
from typing import Any, Callable, NamedTuple

from .context import RunContext, _safe, check_halt, current_run
from .errors import Halted
from .runtime import get_runtime, halt_if_config_invalid

log = logging.getLogger("pitstop")

MAX_STEERS = 2


def tool(fn: Callable | None = None, *, side_effects: bool = False, name: str | None = None):
    """Use as `@tool` or `@tool(side_effects=True, name="...")`; sync and async functions."""
    def deco(f: Callable) -> Callable:
        tname = name or f.__name__
        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*a, **k):
                ctx = current_run()
                if ctx is None:
                    return await f(*a, **k)
                early = _before(ctx, tname, side_effects, a, k)
                if isinstance(early, _NeedsGate):  # the judge call is sync: keep it off the event loop
                    early = await asyncio.to_thread(_gate, ctx, tname, early)
                if early is not _RUN:
                    return early
                try:
                    result = await f(*a, **k)
                except BaseException as e:
                    err = _summ(e)
                    _safe(lambda: _event(ctx, "tool_result", tname, {"error": err}))
                    raise
                _safe(lambda: _event(ctx, "tool_result", tname, {"result": _summ(result)}))
                return result
            return awrapper

        @functools.wraps(f)
        def wrapper(*a, **k):
            ctx = current_run()
            if ctx is None:
                return f(*a, **k)
            early = _before(ctx, tname, side_effects, a, k)
            if isinstance(early, _NeedsGate):
                early = _gate(ctx, tname, early)
            if early is not _RUN:
                return early
            try:
                result = f(*a, **k)
            except BaseException as e:
                err = _summ(e)
                _safe(lambda: _event(ctx, "tool_result", tname, {"error": err}))
                raise
            _safe(lambda: _event(ctx, "tool_result", tname, {"result": _summ(result)}))
            return result
        return wrapper

    return deco(fn) if fn is not None else deco


_RUN = object()
_GATED = ("high", "medium")
_RECENT = 5  # the agent's recent tool calls shown to the gate's judge


class _NeedsGate(NamedTuple):
    """Returned by _before when L0 let a gated call through; the wrapper then runs _gate (inline or in a thread)."""
    args_text: str
    recent: list


def _summ(v: Any, n: int = 500) -> str:
    try:
        return repr(v)[:n]
    except Exception:
        return "<unrepresentable>"


def _one_line(evidence: dict) -> str:
    try:
        text = json.dumps(evidence, sort_keys=True, default=str)
    except Exception:
        text = repr(evidence)
    return " ".join(text.split())[:300]


def _event(ctx: RunContext, kind: str, tname: str, payload: dict) -> None:
    with get_runtime().store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (ctx.run_id, time.time(), ctx.agent, kind, tname, json.dumps(payload, default=str), None))


def _before(ctx: RunContext, tname: str, side_effects: bool, args: tuple, kwargs: dict) -> Any:
    """Everything before the call except the L1 gate. Returns _RUN to run the function, a _NeedsGate when the gate
    must decide, else the value to return instead."""
    check_halt()
    halt_if_config_invalid()
    decision = None
    args_text = ""
    recent: list = []
    gated = side_effects and ctx.spec.criticality in _GATED
    try:
        from ..detect.drifts import ACTION_RANK
        from ..detect.l0 import ToolCallRecord, check_tool_call
        from .actions import apply_findings

        try:
            args_text = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, separators=(",", ":"),
                                   default=repr)
        except Exception:
            args_text = repr((args, kwargs))
        rec = ToolCallRecord(tname, args_text, side_effects)
        with ctx.lock:
            history = ctx.tool_history_for(ctx.agent)
            if gated:
                recent = list(history)[-_RECENT:]
            findings = check_tool_call(history, rec, ctx.spec)
            history.append(rec)
        applied = apply_findings(ctx, findings)
        # the strictest block or steer (first on a tie); other actions do not change how the tool runs
        acting = [a for a in applied if a.action in ("block", "steer")]
        if acting:
            decision = max(acting, key=lambda a: ACTION_RANK[a.action])
    except Exception:
        if not gated:
            log.warning("pitstop: tool check failed; running the tool normally", exc_info=True)
            return _RUN
        log.warning("pitstop: tool check failed; the pre-action gate still decides", exc_info=True)

    if decision is None:
        if gated:
            return _NeedsGate(args_text, recent)
        _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}))
        return _RUN
    finding, _level, action, deciding_id = decision
    text = f"{finding.drift_type}: {_one_line(finding.evidence)}"
    if action == "block":
        _safe(lambda: _event(ctx, "block", tname, {"reason": text, "args": args_text[:500]}))
        return {"blocked_by": "pitstop", "reason": text}
    # steer
    with ctx.lock:
        over = ctx.steers >= MAX_STEERS
        if over:
            if ctx.halt_reason is None:
                ctx.halt_reason = "steer limit exceeded"
            if ctx.halt_drift_id is None:  # the drift whose steer tripped the limit
                ctx.halt_drift_id = deciding_id
        else:
            ctx.steers += 1
        reason, drift_id = ctx.halt_reason, ctx.halt_drift_id
    if over:
        _safe(lambda: _event(ctx, "halt", tname, {"reason": reason, "drift": text, "args": args_text[:500]}))
        raise Halted(reason, drift_id)
    _safe(lambda: _event(ctx, "steer", tname, {"reason": text, "args": args_text[:500]}))
    return (f"PitStop: this step drifts from your goal ({text}). "
            f"Return to: {ctx.agent_goal or ctx.spec.goal or 'your original task'}.")


def _gate(ctx: RunContext, tname: str, need: _NeedsGate) -> Any:
    """The L1 pre-action gate. Returns _RUN or the block dict; Halted and BudgetExceeded from the judge call
    propagate. Any other failure, or no judge at all, is a gate error handled by `on_gate_error`."""
    check_halt()
    args_text = need.args_text
    try:
        from ..detect.l1 import GATE_PROMPT_VERSION, GateVerdict

        rt = get_runtime()
        gate = rt.gate()
        if gate is None:
            verdict = GateVerdict(False, "", error=rt.no_gate_reason())
        else:
            verdict = gate.check(workflow_goal=ctx.spec.goal, agent_goal=ctx.agent_goal, tool=tname,
                                 args_text=args_text, recent=need.recent)
    except Halted:
        raise
    except Exception as e:
        log.warning("pitstop: pre-action gate failed (%s)", type(e).__name__)
        return _gate_error(ctx, tname, type(e).__name__, args_text)
    if verdict.error is not None:
        return _gate_error(ctx, tname, verdict.error, args_text)

    payload = {"allow": verdict.allow, "reason": verdict.reason, "latency_s": round(verdict.latency_s, 4),
               "prompt": GATE_PROMPT_VERSION}
    _safe(lambda: _event(ctx, "gate", tname, payload))
    if verdict.allow:
        _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}))
        return _RUN
    from ..detect.l0 import Finding
    from .actions import apply_findings

    evidence = {"tool": tname, "gate": verdict.reason, "prompt": GATE_PROMPT_VERSION}
    # A side-effect out_of_scope_action is P0 (block); on a high workflow it also sets the pending halt.
    _safe(lambda: apply_findings(ctx, [Finding("out_of_scope_action", "off_task", True, evidence)]))
    text = f"out_of_scope_action: {_one_line(evidence)}"
    _safe(lambda: _event(ctx, "block", tname, {"reason": text, "args": args_text[:500]}))
    return {"blocked_by": "pitstop", "reason": text}


def _gate_error(ctx: RunContext, tname: str, error: str, args_text: str) -> Any:
    """The gate could not decide: record it and apply the workflow's on_gate_error policy."""
    policy = ctx.spec.effective_on_gate_error
    _safe(lambda: _event(ctx, "gate_error", tname, {"error": error, "policy": policy}))
    if policy == "block":
        reason = (f"pre-action check unavailable ({error}); blocked by on_gate_error=block. "
                  "Configure [judge] in config.toml or set on_gate_error=\"allow\".")
        _safe(lambda: _event(ctx, "block", tname, {"reason": reason, "args": args_text[:500]}))
        return {"blocked_by": "pitstop", "reason": reason}
    _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}))
    return _RUN
