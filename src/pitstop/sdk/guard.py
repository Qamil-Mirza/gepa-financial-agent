"""The tool guard: scope and loop checks (block, steer, halt) and the L1 pre-action gate for one tool call. Shared by
`@pitstop.tool` and the framework adapters, so every path checks a call the same way.

`guard_call` first raises a pending `Halted`, runs the L0 tool checks against the agent's recent calls, records any
drift, and then either blocks the call (side-effect tools), steers (read-only tools, at most 2 per run, then halts)
or lets it run. The history check and the steer check-and-increment run under the run's lock, so threads sharing a
run cannot exceed the steer limit.
A side-effect tool on a high or medium workflow that L0 lets through then goes to the L1 pre-action gate
(detect/l1.py): a judge decides whether the call serves the goal. A rejection is an out_of_scope_action drift and
blocks the call; when the judge is missing or cannot decide, the workflow's `on_gate_error` policy blocks or runs it.
`aguard_call` runs the gate in a worker thread (asyncio.to_thread copies the run's context), never on the event loop.
`agent=` overrides the run's current agent (the tool-history key and the events' agent column) for adapters that
know the framework's agent name.

Checked once: while a guarded call's function runs, `checked(ctx, tool)` marks (run, tool) in a ContextVar. A
`@pitstop.tool` of the same name called inside it skips its own guard (no L0, no gate, no tool_call event) and only
runs the function and records its result, so a call guarded by an adapter and again by an inner `@pitstop.tool` is
counted once. The decorator sets the mark around its own function; an adapter sets it around the framework's tool
execution after `guard_call` let the call run. One call records one `tool_result` event: `record_tool_result` under
the mark sets the mark's `recorded` flag (the mark object is shared with threads and tasks started inside the block),
so an adapter records the result itself only when no inner `@pitstop.tool` did.
The mark is one-shot: the first nested same-name call consumes it (`Mark.consumed`), so a later same-name call in
the block, such as a tool that retries itself or calls its own name twice, is guarded like any other.
A task created (asyncio.create_task, a thread's copied context) inside a marked call inherits the mark: its first
same-name call skips the guard too, unless the block's own inner call already consumed it. Documented, not
prevented; revisited with M4 (plan decision 14).
PitStop's own failures never break the call; only `Halted` (and `BudgetExceeded` from the judge) propagate."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal, NamedTuple

from .context import RunContext, _safe, check_halt
from .errors import Halted
from .runtime import get_runtime, halt_if_config_invalid

log = logging.getLogger("pitstop")

MAX_STEERS = 2


@dataclass(frozen=True)
class GuardDecision:
    action: Literal["run", "block", "steer"]
    reply: Any = None  # block: the {"blocked_by": "pitstop", "reason": ...} dict; steer: the corrective text


_RUN = GuardDecision("run")
_GATED = ("high", "medium")
_RECENT = 5  # the agent's recent tool calls shown to the gate's judge

class Mark:
    """The check-once mark of one guarded call. `recorded` turns True once its `tool_result` event was recorded;
    `consumed` once the first nested same-name call took the mark (see `is_checked`); `started` is
    `time.monotonic()` at creation (tool latency)."""
    __slots__ = ("recorded", "consumed", "started")

    def __init__(self) -> None:
        self.recorded = False
        self.consumed = False
        self.started = time.monotonic()


# (run_id, tool) -> the Mark of the call already guarded there; see `checked`. Replaced, never mutated.
_checked: ContextVar[dict] = ContextVar("pitstop_tool_checked", default={})


class _NeedsGate(NamedTuple):
    """Returned by _before when L0 let a gated call through; the caller then runs _gate (inline or in a thread)."""
    args_text: str
    recent: list


def tool_args_text(args: tuple, kwargs: dict) -> str:
    """The canonical text of a call's arguments: sorted compact JSON (repr for what JSON cannot encode)."""
    try:
        return json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, separators=(",", ":"), default=repr)
    except Exception:
        pass
    try:
        return repr((args, kwargs))
    except Exception:
        return "<unrepresentable>"


@contextmanager
def checked(ctx: RunContext, tool: str) -> Iterator[Mark]:
    """Mark `tool` as already guarded in this run while the block runs: a `@pitstop.tool` of the same name inside
    it skips its own guard. Use it around the tool's execution once `guard_call` returned "run". Yields the Mark;
    inside an enclosing mark for the same tool it yields that one, so the outermost caller sees `recorded`."""
    key = (ctx.run_id, tool)
    prev = _checked.get()
    if key in prev:
        yield prev[key]
        return
    mark = Mark()
    token = _checked.set({**prev, key: mark})
    try:
        yield mark
    finally:
        try:
            _checked.reset(token)
        except ValueError:  # exited from another Context: drop only this mark, never install the entering
            cur = _checked.get()  # context's other marks there (they would skip L0 and the gate)
            if cur.get(key) is mark:
                _checked.set({k: v for k, v in cur.items() if k != key})


def is_checked(ctx: RunContext, tool: str) -> bool:
    """True for the first same-name call inside a `checked(ctx, tool)` block: that call was already guarded, and it
    consumes the mark. Later same-name calls in the block are guarded normally. The mark itself stays in place until
    the block exits (`record_tool_result` still finds it); only its `consumed` flag is set."""
    mark = _checked.get().get((ctx.run_id, tool))
    if mark is None or mark.consumed:
        return False
    mark.consumed = True
    return True


def guard_call(ctx: RunContext, tool: str, side_effects: bool, args_text: str, *,
               agent: str | None = None) -> GuardDecision:
    """Check one tool call: L0, then the L1 gate inline. Raises `Halted` for a pending halt or the steer limit."""
    who = agent if agent is not None else ctx.agent
    decision = _before(ctx, tool, side_effects, args_text, who)
    if isinstance(decision, _NeedsGate):
        decision = _gate(ctx, tool, decision, who)
    return decision


async def aguard_call(ctx: RunContext, tool: str, side_effects: bool, args_text: str, *,
                      agent: str | None = None) -> GuardDecision:
    """`guard_call` for async callers: the judge call is sync, so the gate runs in a worker thread."""
    who = agent if agent is not None else ctx.agent
    decision = _before(ctx, tool, side_effects, args_text, who)
    if isinstance(decision, _NeedsGate):
        decision = await asyncio.to_thread(_gate, ctx, tool, decision, who)
    return decision


def record_tool_result(ctx: RunContext, tool: str, *, result: Any = None, error: BaseException | None = None,
                       agent: str | None = None) -> None:
    """Record a `tool_result` event: the error's repr when `error` is given, else the result's (capped)."""
    who = agent if agent is not None else ctx.agent
    mark = _checked.get().get((ctx.run_id, tool))
    if mark is not None:
        mark.recorded = True
    payload = {"error": _summ(error)} if error is not None else {"result": _summ(result)}
    _safe(lambda: _event(ctx, "tool_result", tool, payload, who))


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


def _event(ctx: RunContext, kind: str, tname: str, payload: dict, agent: str | None) -> None:
    with get_runtime().store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (ctx.run_id, time.time(), agent, kind, tname, json.dumps(payload, default=str), None))


def _before(ctx: RunContext, tname: str, side_effects: bool, args_text: str, who: str | None) -> Any:
    """Everything before the call except the L1 gate. Returns a GuardDecision, or a _NeedsGate when the gate must
    decide."""
    check_halt()
    halt_if_config_invalid()
    decision = None
    recent: list = []
    gated = side_effects and ctx.spec.criticality in _GATED
    try:
        from ..detect.drifts import ACTION_RANK
        from ..detect.l0 import ToolCallRecord, check_tool_call
        from .actions import apply_findings

        rec = ToolCallRecord(tname, args_text, side_effects)
        with ctx.lock:
            history = ctx.tool_history_for(who)
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
        _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}, who))
        return _RUN
    finding, _level, action, deciding_id = decision
    text = f"{finding.drift_type}: {_one_line(finding.evidence)}"
    if action == "block":
        _safe(lambda: _event(ctx, "block", tname, {"reason": text, "args": args_text[:500]}, who))
        return GuardDecision("block", {"blocked_by": "pitstop", "reason": text})
    # steer
    with ctx.lock:
        over = ctx.steers >= MAX_STEERS
        if over:
            if ctx.halt_reason is None:  # a reason set elsewhere keeps its own drift id (even None)
                ctx.halt_reason = "steer limit exceeded"
                ctx.halt_drift_id = deciding_id  # the drift whose steer tripped the limit
        else:
            ctx.steers += 1
        reason, drift_id = ctx.halt_reason, ctx.halt_drift_id
    if over:
        _safe(lambda: _event(ctx, "halt", tname, {"reason": reason, "drift": text, "args": args_text[:500]}, who))
        raise Halted(reason, drift_id)
    _safe(lambda: _event(ctx, "steer", tname, {"reason": text, "args": args_text[:500]}, who))
    return GuardDecision("steer", f"PitStop: this step drifts from your goal ({text}). "
                                  f"Return to: {ctx.agent_goal or ctx.spec.goal or 'your original task'}.")


def _gate(ctx: RunContext, tname: str, need: _NeedsGate, who: str | None) -> GuardDecision:
    """The L1 pre-action gate. Returns run or block; Halted and BudgetExceeded from the judge call propagate. Any
    other failure, or no judge at all, is a gate error handled by `on_gate_error`."""
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
        return _gate_error(ctx, tname, type(e).__name__, args_text, who)
    if verdict.error is not None:
        return _gate_error(ctx, tname, verdict.error, args_text, who)

    payload = {"allow": verdict.allow, "reason": verdict.reason, "latency_s": round(verdict.latency_s, 4),
               "prompt": GATE_PROMPT_VERSION}
    _safe(lambda: _event(ctx, "gate", tname, payload, who))
    if verdict.allow:
        _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}, who))
        return _RUN
    from ..detect.l0 import Finding
    from .actions import apply_findings

    evidence = {"tool": tname, "gate": verdict.reason, "prompt": GATE_PROMPT_VERSION}
    # A side-effect out_of_scope_action is P0 (block); on a high workflow it also sets the pending halt.
    _safe(lambda: apply_findings(ctx, [Finding("out_of_scope_action", "off_task", True, evidence)]))
    text = f"out_of_scope_action: {_one_line(evidence)}"
    _safe(lambda: _event(ctx, "block", tname, {"reason": text, "args": args_text[:500]}, who))
    return GuardDecision("block", {"blocked_by": "pitstop", "reason": text})


def _gate_error(ctx: RunContext, tname: str, error: str, args_text: str, who: str | None) -> GuardDecision:
    """The gate could not decide: record it and apply the workflow's on_gate_error policy."""
    policy = ctx.spec.effective_on_gate_error
    _safe(lambda: _event(ctx, "gate_error", tname, {"error": error, "policy": policy}, who))
    if policy == "block":
        reason = (f"pre-action check unavailable ({error}); blocked by on_gate_error=block. "
                  "Configure [judge] in config.toml or set on_gate_error=\"allow\".")
        _safe(lambda: _event(ctx, "block", tname, {"reason": reason, "args": args_text[:500]}, who))
        return GuardDecision("block", {"blocked_by": "pitstop", "reason": reason})
    _safe(lambda: _event(ctx, "tool_call", tname, {"args": args_text[:500]}, who))
    return _RUN
