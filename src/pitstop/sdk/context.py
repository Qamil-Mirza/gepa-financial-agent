"""Workflow and agent scopes. The current run and the current agent each live in a ContextVar, so they are
isolated per thread and per asyncio task: parallel sub-agents started with asyncio.gather inside one workflow
share the RunContext but each sees its own agent name and goal.

A scope decorating a generator (or async generator) function opens on the generator's first step and closes
when it finishes, raises or is closed, by the caller or by garbage collection (also inside a reference cycle: an
async generator's inner generator is only ever closed by its wrapper, see `_unhooked`). Each step sets the run (and
agent) in the consumer's own context and resets it when the step yields, so code that consumes the generator
between yields never sees the run."""
from __future__ import annotations

import functools
import inspect
import logging
import sys
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple, TypeVar

from ..userconfig import ConfigInvalid
from .errors import BudgetExceeded, Halted
from .runtime import config_halt, get_runtime, halt_if_config_invalid

log = logging.getLogger("pitstop")

Criticality = Literal["high", "medium", "low"]
Sensitivity = Literal["low", "normal", "high"]

F = TypeVar("F", bound=Callable[..., Any])

_CRITICALITIES = ("high", "medium", "low")
_SENSITIVITIES = ("low", "normal", "high")
_FIX_MODES = ("manual", "auto", None)
_GATE_ERROR_POLICIES = ("block", "allow", None)

HISTORY_MAX = 64  # per-agent tool history and per-run LLM history are capped at this many recent calls
UNSET = object()  # RunContext.spend_baseline before the run's first settle computed it (None: no baseline)


def _history() -> deque:
    return deque(maxlen=HISTORY_MAX)


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    goal: str | None = None
    criticality: Criticality = "medium"
    budget_usd: float | None = None
    sensitivity: Sensitivity = "normal"
    fix_mode: Literal["manual", "auto"] | None = None
    tools: tuple[str, ...] | None = None
    on_gate_error: Literal["block", "allow"] | None = None
    fallback_model: str | None = None  # pinned model a P1 breakage after a model/provider change falls back to

    def __post_init__(self) -> None:
        # Checked when the workflow is defined: a typo (criticality="High", on_gate_error="deny") must never
        # silently turn a check off or fail open.
        for what, value, allowed in (("criticality", self.criticality, _CRITICALITIES),
                                     ("sensitivity", self.sensitivity, _SENSITIVITIES),
                                     ("fix_mode", self.fix_mode, _FIX_MODES),
                                     ("on_gate_error", self.on_gate_error, _GATE_ERROR_POLICIES)):
            if value not in allowed:
                raise ValueError(f"pitstop.workflow {self.name!r}: invalid {what} {value!r} "
                                 f"(expected one of {allowed})")
        if self.fallback_model is not None and not (isinstance(self.fallback_model, str) and self.fallback_model):
            raise ValueError(f"pitstop.workflow {self.name!r}: invalid fallback_model {self.fallback_model!r} "
                             f"(expected a model id or None)")

    @property
    def effective_fix_mode(self) -> str:
        return self.fix_mode or ("manual" if self.criticality == "high" else "auto")

    @property
    def effective_on_gate_error(self) -> str:
        """What a side-effect tool does when the L1 gate cannot decide: high blocks (fail closed), else runs."""
        return self.on_gate_error or ("block" if self.criticality == "high" else "allow")


@dataclass
class RunContext:
    run_id: str
    spec: WorkflowSpec
    steers: int = 0
    halt_reason: str | None = None
    halt_drift_id: int | None = None
    tool_history: dict = field(default_factory=dict)  # agent name (None = no agent) -> deque[ToolCallRecord]
    llm_history: deque = field(default_factory=_history)
    refusals: int = 0  # every refusal in the run; the capped llm_history cannot count them all
    llm_calls: int = 0  # agent-meter LLM calls only
    prompt_versions: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    # The spend-rate P0 (pitstop.detect.spend): the run's settled agent-meter USD, its workflow's baseline
    # (computed once, at the first settle) and whether the spike was already applied (at most once per run).
    spend_usd: float = 0.0
    spend_baseline: Any = UNSET
    spend_spiked: bool = False
    # Guards the counters and histories when several threads share the run (e.g. the steer limit).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def tool_history_for(self, agent: str | None) -> deque:
        hist = self.tool_history.get(agent)
        if hist is None:
            hist = self.tool_history.setdefault(agent, _history())
        return hist

    @property
    def agent(self) -> str | None:
        """The agent name set by the innermost `agent()` scope of the calling task, for this run."""
        cur = _agent_var.get()
        return cur.name if cur is not None and cur.run_id == self.run_id else None

    @property
    def agent_goal(self) -> str | None:
        cur = _agent_var.get()
        return cur.goal if cur is not None and cur.run_id == self.run_id else None


class _AgentFrame(NamedTuple):
    """One agent() scope. Frames link to the enclosing one, so each task's chain lives in its own Context."""
    run_id: str
    name: str
    goal: str | None
    prev: _AgentFrame | None
    owner: object


_current: ContextVar[RunContext | None] = ContextVar("pitstop_run", default=None)
# Innermost agent() scope of the calling task. Scope objects keep no per-instance state, so one agent() or
# workflow() object can be entered concurrently from several tasks or threads.
_agent_var: ContextVar[_AgentFrame | None] = ContextVar("pitstop_agent", default=None)
# The ledger meter `pitstop.meter()` tags calls with. Lives here (re-exported by pitstop.sdk.wrap) so a decorated
# generator scopes it per step like the run and agent: a meter() block spanning a yield never leaks to the consumer.
_meter: ContextVar[str] = ContextVar("pitstop_meter", default="agent")


def current_run() -> RunContext | None:
    return _current.get()


def check_halt() -> None:
    ctx = _current.get()
    if ctx is not None and ctx.halt_reason is not None:
        raise Halted(ctx.halt_reason, ctx.halt_drift_id)


def _safe(fn: Callable[[], Any]) -> Any:
    """Run PitStop-internal work; log and swallow any failure so it never surfaces in user code."""
    try:
        return fn()
    except Exception:
        log.warning("pitstop: internal error ignored", exc_info=True)
        return None


class _Scope:
    """Usable as a decorator (sync or async function, generator or async generator), a context manager and an
    async context manager."""

    def _fresh(self) -> _Scope:
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, et, ev, tb):
        raise NotImplementedError

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, et, ev, tb):
        return self.__exit__(et, ev, tb)

    def __call__(self, fn: F) -> F:
        if inspect.isasyncgenfunction(fn):
            return self._wrap_asyncgen(fn)  # type: ignore[return-value]
        if inspect.isgeneratorfunction(fn):
            return self._wrap_gen(fn)  # type: ignore[return-value]
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **k):
                async with self._fresh():
                    return await fn(*a, **k)
            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with self._fresh():
                return fn(*a, **k)
        return wrapper  # type: ignore[return-value]

    def _wrap_gen(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def gwrapper(*a, **k):
            run = _GenRun(self._fresh())
            run.enter()
            exc_info: tuple = (None, None, None)
            try:
                gen = fn(*a, **k)  # creating the generator runs none of its body
                step, arg = gen.send, None
                while True:
                    try:
                        value = run.step(step, arg)
                    except StopIteration as e:
                        return e.value
                    try:
                        arg = yield value
                        step = gen.send
                    except GeneratorExit:
                        run.step(gen.close)  # the generator's cleanup runs inside the run
                        raise
                    except BaseException as e:  # consumer's throw(): forward it into the generator
                        step, arg = gen.throw, e
            except GeneratorExit:  # closed early (close() or garbage collection): the run ends normally
                raise
            except BaseException as e:
                exc_info = (type(e), e, e.__traceback__)
                raise
            finally:
                run.exit(*exc_info)
        return gwrapper

    def _wrap_asyncgen(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def agwrapper(*a, **k):
            run = _GenRun(self._fresh())
            run.enter()
            exc_info: tuple = (None, None, None)
            try:
                agen = fn(*a, **k)
                # Only this wrapper is registered with the event loop's async-generator hooks; the inner
                # generator is closed by the wrapper, never concurrently by the loop's shutdown or finalizer.
                aw = _unhooked(agen.asend, None)
                while True:
                    try:
                        value = await run.astep(aw)
                    except StopAsyncIteration:
                        return
                    try:
                        arg = yield value
                        aw = agen.asend(arg)
                    except GeneratorExit:
                        if agen.ag_running:  # being stepped elsewhere: closing it now would raise
                            log.warning("pitstop: workflow async generator closed while running; its cleanup "
                                        "was skipped")
                        else:
                            await run.astep(agen.aclose())
                        raise
                    except BaseException as e:
                        aw = agen.athrow(e)
            except GeneratorExit:
                raise
            except BaseException as e:
                exc_info = (type(e), e, e.__traceback__)
                raise
            finally:
                run.exit(*exc_info)
        return agwrapper


def _unhooked(make: Callable, arg: Any) -> Any:
    """Make an async generator's first awaitable with the event loop's hooks replaced, so the loop never closes
    that generator on its own: no firstiter hook (the loop does not track it, or close it at shutdown) and a
    finalizer that does nothing (garbage collection does not schedule its aclose()).

    The finalizer matters when the wrapper and the inner generator are abandoned in a reference cycle: one
    gc.collect() finalizes both. The loop's finalizer would schedule an aclose() for each, and the inner one's
    would race the wrapper's own close of it ("aclose(): asynchronous generator is already running"). With this
    one, only the wrapper's aclose() runs, and it closes the inner generator inside the run. Without a finalizer
    hook at all (no event loop), the inner generator keeps none, as before."""
    firstiter, finalizer = sys.get_asyncgen_hooks()
    sys.set_asyncgen_hooks(firstiter=None, finalizer=_closed_by_wrapper if finalizer is not None else None)
    try:
        return make(arg)
    finally:
        sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=finalizer)


def _closed_by_wrapper(agen: Any) -> None:
    """The inner generator's finalizer: nothing to do, its wrapper closes it."""


class _GenRun:
    """The run, agent and meter ContextVar values a decorated generator sees. Each step sets them in the consumer's
    own context and resets them right after, so the run is visible only while the generator runs, and
    asyncio.timeout, TaskGroup or anyio scopes that span a yield keep working (same task, same context)."""

    _VARS = (_current, _agent_var, _meter)

    def __init__(self, scope: _Scope):
        self.scope = scope
        self.values: tuple | None = None  # the generator's own values, saved after each step

    def _push(self) -> list:
        prev = [v.get() for v in self._VARS]
        values = self.values if self.values is not None else prev
        return [(v, v.set(x), p) for v, x, p in zip(self._VARS, values, prev)]

    def _pop(self, tokens: list) -> None:
        self.values = tuple(v.get() for v in self._VARS)
        for var, token, prev in reversed(tokens):
            try:
                var.reset(token)
            except ValueError:  # stepped from another Context than the one it started in
                var.set(prev)

    def enter(self) -> None:
        tokens = self._push()
        try:
            self.scope.__enter__()
        finally:
            self._pop(tokens)

    def exit(self, et, ev, tb) -> None:
        tokens = self._push()
        try:
            self.scope.__exit__(et, ev, tb)
        finally:
            self._pop(tokens)

    def step(self, fn: Callable, *args: Any) -> Any:
        tokens = self._push()
        try:
            return fn(*args)
        finally:
            self._pop(tokens)

    async def astep(self, aw: Any) -> Any:
        tokens = self._push()
        try:
            return await aw
        finally:
            self._pop(tokens)


class _Workflow(_Scope):
    def __init__(self, spec: WorkflowSpec):
        self.spec = spec

    def _fresh(self) -> _Workflow:
        return _Workflow(self.spec)

    def __enter__(self) -> RunContext:
        if _current.get() is not None:
            raise RuntimeError("pitstop.workflow cannot be nested")
        halt_if_config_invalid()  # an invalid config.toml fails closed before the run starts
        ctx = RunContext(run_id=uuid.uuid4().hex, spec=self.spec)
        refused = self._exhausted()
        if refused is not None:
            _safe(lambda: self._insert(ctx, "refused"))
            _safe(lambda: self._record_refusal(ctx, refused))
            raise refused
        _safe(lambda: self._insert(ctx))
        _current.set(ctx)
        return ctx

    def _exhausted(self) -> BudgetExceeded | None:
        """The used-up budget that refuses this run at its start (spec 3.2), else None: the workflow's own budget,
        a global budget without base_url, or the global budget for the endpoint of the workflow's previous run
        that made a call (a refused run has none); an endpoint the run has not used yet halts at its first call. A store or ledger error is logged
        and the run starts (fail open); an invalid config still fails closed."""
        try:
            rt = get_runtime()
            row = rt.store.connect().execute(
                "SELECT base_url FROM runs WHERE workflow=? AND base_url IS NOT NULL"
                " ORDER BY started_at DESC LIMIT 1", (self.spec.name,)).fetchone()
            return rt.ledger.exhausted(workflow=self.spec.name, workflow_budget_usd=self.spec.budget_usd,
                                       base_url=row["base_url"] if row is not None else None)
        except ConfigInvalid as e:
            raise config_halt(e) from e
        except Exception:
            log.warning("pitstop: could not check the budgets at run start; the run starts", exc_info=True)
            return None

    def _record_refusal(self, ctx: RunContext, exc: BudgetExceeded) -> None:
        from ..detect.l0 import Finding
        from .actions import apply_findings

        ctx.halt_reason = str(exc)  # the run never starts: no pending halt (or its event) to record
        apply_findings(ctx, [Finding("budget_exhausted", "degraded", False,
                                     {"scope": exc.scope, "limit_usd": exc.limit_usd, "spent_usd": exc.spent_usd,
                                      "refused": True})])

    def _insert(self, ctx: RunContext, status: str = "running") -> None:
        from .. import __version__

        s = ctx.spec
        ended = ctx.started_at if status == "refused" else None
        with get_runtime().store.transaction() as c:
            c.execute(
                "INSERT INTO runs(id, workflow, goal, criticality, sensitivity, fix_mode, started_at, ended_at,"
                " status, pitstop_version, budget_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, s.name, s.goal, s.criticality, s.sensitivity, s.effective_fix_mode,
                 ctx.started_at, ended, status, __version__, s.budget_usd),
            )

    def __exit__(self, et, ev, tb):
        # Workflows never nest, so the enclosing value is always None; the run being closed is this
        # Context's current one (not per-instance state, which concurrent tasks would share).
        ctx = _current.get()
        _current.set(None)
        if ctx is None:
            log.warning("pitstop: workflow exit without a current run; ignored")
            return False
        status = "ok" if et is None else "halted" if issubclass(et, Halted) else "error"

        def finish() -> None:
            with get_runtime().store.transaction() as c:
                c.execute("UPDATE runs SET ended_at=?, status=? WHERE id=?", (time.time(), status, ctx.run_id))

        _safe(finish)
        return False


class _Agent(_Scope):
    _warned = False

    def __init__(self, name: str, goal: str | None):
        self.name, self.goal = name, goal

    def _fresh(self) -> _Agent:
        return _Agent(self.name, self.goal)

    def __enter__(self):
        ctx = _current.get()
        if ctx is None:
            if not _Agent._warned:
                _Agent._warned = True
                log.info("pitstop.agent used outside a workflow; ignored")
            return None
        _agent_var.set(_AgentFrame(ctx.run_id, self.name, self.goal, _agent_var.get(), self))
        return ctx

    def __exit__(self, et, ev, tb):
        try:
            cur = _agent_var.get()
            if cur is not None and cur.owner is self:  # no frame was pushed for a no-op (outside-workflow) enter
                _agent_var.set(cur.prev)
        except Exception:
            log.warning("pitstop: internal error ignored", exc_info=True)
        return False


def workflow(name: str, *, goal: str | None = None, criticality: Criticality = "medium",
             budget_usd: float | None = None, sensitivity: Sensitivity = "normal",
             fix_mode: Literal["manual", "auto"] | None = None,
             tools: tuple[str, ...] | list[str] | None = None,
             on_gate_error: Literal["block", "allow"] | None = None,
             fallback_model: str | None = None) -> _Workflow:
    return _Workflow(WorkflowSpec(name, goal, criticality, budget_usd, sensitivity, fix_mode,
                                  tuple(tools) if tools is not None else None, on_gate_error, fallback_model))


def agent(name: str, *, goal: str | None = None) -> _Agent:
    return _Agent(name, goal)
