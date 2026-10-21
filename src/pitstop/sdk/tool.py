"""`@pitstop.tool`: guard a tool function with scope and loop checks (block, steer, halt).

Outside a workflow the decorator just calls the function. Inside one, each invocation first raises a pending
`Halted`, runs the L0 tool checks against the current agent's recent calls, records any drift, and then either
blocks the call (side-effect tools), steers (read-only tools, at most 2 per run, then halts) or runs it.
Exceptions from the user's function propagate unchanged; PitStop's own failures never break the call."""
from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from typing import Any, Callable

from .context import RunContext, _safe, check_halt, current_run
from .errors import Halted
from .runtime import get_runtime, halt_if_config_invalid

log = logging.getLogger("pitstop")

MAX_STEERS = 2
_RANK = {"record": 0, "steer": 1, "block": 2}


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
                if early is not _RUN:
                    return early
                try:
                    result = await f(*a, **k)
                except BaseException as e:
                    err = repr(e)
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
            if early is not _RUN:
                return early
            try:
                result = f(*a, **k)
            except BaseException as e:
                err = repr(e)
                _safe(lambda: _event(ctx, "tool_result", tname, {"error": err}))
                raise
            _safe(lambda: _event(ctx, "tool_result", tname, {"result": _summ(result)}))
            return result
        return wrapper

    return deco(fn) if fn is not None else deco


_RUN = object()


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
    """Everything before the call. Returns _RUN to run the function, else the value to return instead."""
    check_halt()
    halt_if_config_invalid()
    decision = None
    args_text = ""
    try:
        from ..detect.l0 import ToolCallRecord, check_tool_call
        from .wrap import _apply_findings

        try:
            args_text = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, separators=(",", ":"),
                                   default=repr)
        except Exception:
            args_text = repr((args, kwargs))
        rec = ToolCallRecord(tname, args_text, side_effects)
        history = ctx.tool_history.setdefault(ctx.agent, [])
        findings = check_tool_call(history, rec, ctx.spec)
        history.append(rec)
        applied = _apply_findings(ctx, findings)
        best = None
        for item in applied:
            if best is None or _RANK.get(item[2], 0) > _RANK.get(best[2], 0):
                best = item
        if best is not None and best[2] in ("block", "steer"):
            decision = best
    except Exception:
        log.warning("pitstop: tool check failed; running the tool normally", exc_info=True)
        return _RUN

    if decision is None:
        _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}))
        return _RUN
    finding, _level, action, _drift_id = decision
    text = f"{finding.drift_type}: {_one_line(finding.evidence)}"
    if action == "block":
        _safe(lambda: _event(ctx, "block", tname, {"reason": text, "args": args_text[:500]}))
        return {"blocked_by": "pitstop", "reason": text}
    # steer
    if ctx.steers >= MAX_STEERS:
        if ctx.halt_reason is None:
            ctx.halt_reason = "steer limit exceeded"
        raise Halted(ctx.halt_reason, ctx.halt_drift_id)
    ctx.steers += 1
    _safe(lambda: _event(ctx, "steer", tname, {"reason": text, "args": args_text[:500]}))
    return (f"PitStop: this step drifts from your goal ({text}). "
            f"Return to: {ctx.agent_goal or ctx.spec.goal or 'your original task'}.")
