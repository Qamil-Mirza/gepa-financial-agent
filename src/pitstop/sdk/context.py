"""Workflow and agent scopes. The current run and the current agent each live in a ContextVar, so they are
isolated per thread and per asyncio task: parallel sub-agents started with asyncio.gather inside one workflow
share the RunContext but each sees its own agent name and goal."""
from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple, TypeVar

from .errors import Halted
from .runtime import get_runtime

log = logging.getLogger("pitstop")

Criticality = Literal["high", "medium", "low"]
Sensitivity = Literal["low", "normal", "high"]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    goal: str | None = None
    criticality: Criticality = "medium"
    budget_usd: float | None = None
    sensitivity: Sensitivity = "normal"
    fix_mode: Literal["manual", "auto"] | None = None
    tools: tuple[str, ...] | None = None

    @property
    def effective_fix_mode(self) -> str:
        return self.fix_mode or ("manual" if self.criticality == "high" else "auto")


@dataclass
class RunContext:
    run_id: str
    spec: WorkflowSpec
    steers: int = 0
    halt_reason: str | None = None
    halt_drift_id: int | None = None
    tool_history: dict = field(default_factory=dict)  # agent name (None = no agent) -> [ToolCallRecord]
    llm_history: list = field(default_factory=list)
    llm_calls: int = 0
    prompt_versions: dict[str, int] = field(default_factory=dict)

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
    """Usable as a decorator (sync or async), a context manager and an async context manager."""

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


class _Workflow(_Scope):
    def __init__(self, spec: WorkflowSpec):
        self.spec = spec

    def _fresh(self) -> _Workflow:
        return _Workflow(self.spec)

    def __enter__(self) -> RunContext:
        if _current.get() is not None:
            raise RuntimeError("pitstop.workflow cannot be nested")
        ctx = RunContext(run_id=uuid.uuid4().hex, spec=self.spec)
        _safe(lambda: self._insert(ctx))
        _current.set(ctx)
        return ctx

    def _insert(self, ctx: RunContext) -> None:
        from .. import __version__

        s = ctx.spec
        with get_runtime().store.transaction() as c:
            c.execute(
                "INSERT INTO runs(id, workflow, goal, criticality, sensitivity, fix_mode, started_at, status,"
                " pitstop_version) VALUES (?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, s.name, s.goal, s.criticality, s.sensitivity, s.effective_fix_mode,
                 time.time(), "running", __version__),
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
             tools: tuple[str, ...] | list[str] | None = None) -> _Workflow:
    return _Workflow(WorkflowSpec(name, goal, criticality, budget_usd, sensitivity, fix_mode,
                                  tuple(tools) if tools is not None else None))


def agent(name: str, *, goal: str | None = None) -> _Agent:
    return _Agent(name, goal)
