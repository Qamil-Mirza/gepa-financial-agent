"""OpenAI Agents SDK adapter (`pip install "pitstop[openai-agents]"`): PitStop's tool guard as a native tool input
guardrail, and run hooks that name the agent and tag the run. Metering is the wrapped client's job:

    agent = guard(Agent(name="researcher", tools=[search, send_email]), side_effects={"send_email"})
    set_default_openai_client(pitstop.wrap(AsyncOpenAI()), use_for_tracing=False)
    with pitstop.workflow("research", goal="...", criticality="high"):
        result = await Runner.run(agent, question, hooks=Hooks())

`tool_guardrail(side_effects=...)` runs the shared tool guard (`pitstop.sdk.guard`) on each function-tool call
before the tool runs: a pending halt, the L0 scope and loop checks, then the L1 pre-action gate for side-effect
tools on high and medium workflows (in a worker thread). A call it lets through runs; a blocked or steered call is
rejected with PitStop's reply (a block as compact JSON) as the tool result the model sees. Outside a workflow the
guardrail allows every call.

`Halted` (and `BudgetExceeded`) reach the caller of `Runner.run`. The SDK re-raises its own exceptions from a tool
call unchanged but wraps any other one in a `UserError`, so the adapter re-raises a `Halted` as a copy whose class
derives from both the original class and `agents.exceptions.AgentsException` (`except pitstop.Halted` and
`except pitstop.BudgetExceeded` catch it; it pickles as the original class). The guardrail does this for a pending
halt or the steer limit, the invoker wrapper for a halt raised in, or left pending by, the tool body (the SDK would
otherwise turn it into an error message for the model; a budget hit there therefore arrives as the pending
`Halted`, not as `BudgetExceeded`), and `Hooks.on_llm_start` raises a pending halt before the
next model call, also with a model that is not metered through `pitstop.wrap`.

`guard(agent, side_effects=...)` adds that guardrail to each of the agent's `FunctionTool`s (once; the named tools
are side-effect tools) and wraps the tool's `on_invoke_tool`, so the tool body runs under the guard's check-once
mark: a `@pitstop.tool` of the same name inside it skips its own guard, and one call records one `tool_result`
event (the wrapper records it only when no inner `@pitstop.tool` did). It changes the tools in place and returns
the same agent; call it on each agent of a handoff graph.

`Hooks` names the agent: `on_agent_start` names the running SDK agent for the scope `Runner.run` was called in
(`RunContext.set_framework_agent`, `pitstop.agent(name)` semantics, replacing the agent named before), so the run's
model calls, tool events, drifts' halt events and the gate see it; `on_agent_end` removes the name. The named agent
keeps the goal of the enclosing `pitstop.agent(...)` scope, which the gate judges side-effect calls against (the
workflow goal when there is none). The SDK runs
hooks in child asyncio tasks, so a scope entered there would not reach the run. It also tags the run
`runs.adapter = 'openai-agents'`.

Limits: only `FunctionTool`s are guarded (the SDK runs tool input guardrails for function tools only); hosted
tools, MCP server tools and handoffs are not. Model calls are metered only through a `pitstop.wrap`-ed client. A
guarded tool's `on_invoke_tool` is replaced, so `tool.__wrapped__` no longer reaches the original function. Prefer
`guard` to a bare `tool_guardrail`: around a `@pitstop.tool` of the same name, the bare guardrail checks twice.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from agents import Agent, FunctionTool, RunHooks, ToolGuardrailFunctionOutput, ToolInputGuardrail
from agents.exceptions import AgentsException
from agents.tool_guardrails import ToolInputGuardrailData

from ..context import _safe, check_halt, current_run
from ..errors import Halted
from ..guard import aguard_call, checked, record_tool_result, tool_args_text
from ..runtime import get_runtime

__all__ = ["tool_guardrail", "guard", "Hooks"]

ADAPTER = "openai-agents"


class _PitStopGuardrail(ToolInputGuardrail):
    """The PitStop tool input guardrail; the type lets `guard` find it again."""
    side_effects: bool = False


def tool_guardrail(*, side_effects: bool = False) -> ToolInputGuardrail:
    """A tool input guardrail that runs PitStop's tool guard on each call of the tool it is attached to."""
    async def pitstop_guard(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = current_run()
        if ctx is None:
            return ToolGuardrailFunctionOutput.allow()
        tc = data.context
        try:
            decision = await aguard_call(ctx, tc.tool_name, side_effects, _args_text(tc.tool_arguments),
                                         agent=getattr(data.agent, "name", None))
        except Halted as e:
            raise _escaping(e) from e
        if decision.action == "run":
            return ToolGuardrailFunctionOutput.allow()
        return ToolGuardrailFunctionOutput.reject_content(_reply_text(decision.reply))

    g = _PitStopGuardrail(guardrail_function=pitstop_guard, name="pitstop_side_effects" if side_effects else "pitstop")
    g.side_effects = side_effects
    return g


def guard(agent: Agent, *, side_effects: Iterable[str] = ()) -> Agent:
    """Guard every `FunctionTool` of `agent` (in place; returns `agent`). Tools named in `side_effects` are
    side-effect tools. Idempotent: a tool keeps one PitStop guardrail (upgraded to side-effect when named) and one
    invoker wrapper. Raises ValueError for a side-effect name that is none of the agent's function tools: a typo
    must never leave a side-effect tool ungated."""
    names = {side_effects} if isinstance(side_effects, str) else set(side_effects)
    tools = [t for t in agent.tools if isinstance(t, FunctionTool)]
    unknown = names - {t.name for t in tools}
    if unknown:
        raise ValueError(f"pitstop guard({agent.name!r}): no function tool named {sorted(unknown)} "
                         f"(tools: {sorted(t.name for t in tools)})")
    for t in tools:
        effects = t.name in names
        current = [g for g in t.tool_input_guardrails or [] if isinstance(g, _PitStopGuardrail)]
        if not current:
            t.tool_input_guardrails = [*(t.tool_input_guardrails or []), tool_guardrail(side_effects=effects)]
        elif effects and not current[0].side_effects:
            t.tool_input_guardrails = [tool_guardrail(side_effects=True) if g is current[0] else g
                                       for g in t.tool_input_guardrails]
        if not isinstance(t.on_invoke_tool, _CheckedInvoker):
            t.on_invoke_tool = _CheckedInvoker(t.on_invoke_tool)
    return agent


class Hooks(RunHooks):
    """Run hooks for `Runner.run(..., hooks=Hooks())`: name the running agent for PitStop and tag the run. Subclass
    it (calling super()) to add your own hooks."""

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        ctx = current_run()
        if ctx is None:
            return
        name = getattr(agent, "name", None)
        if name:
            _safe(lambda: ctx.set_framework_agent(name))
        _safe(lambda: _tag_run(ctx.run_id))

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: Any, input_items: Any) -> None:
        _raise_pending()

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        ctx = current_run()
        if ctx is not None and ctx.agent == getattr(agent, "name", None):
            _safe(lambda: ctx.set_framework_agent(None))


class _CheckedInvoker:
    """Wraps a FunctionTool's `on_invoke_tool`: inside a workflow the tool runs under the check-once mark and its
    result (or error) is recorded unless an inner `@pitstop.tool` of the same name recorded it. Attribute lookups
    the SDK makes on the invoker (sync marker, argument preparation) are forwarded to the wrapped one; copying a
    FunctionTool rebinds the wrapped invoker and keeps this wrapper."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def __call__(self, tool_ctx: Any, input: str) -> Any:
        ctx = current_run()
        if ctx is None:
            return await self._inner(tool_ctx, input)
        name = getattr(tool_ctx, "tool_name", None)
        if not name:
            return await self._inner(tool_ctx, input)
        who = getattr(getattr(tool_ctx, "agent", None), "name", None)
        with checked(ctx, name) as mark:
            try:
                result = await self._inner(tool_ctx, input)
            except BaseException as e:
                if not mark.recorded:
                    record_tool_result(ctx, name, error=e, agent=who)
                if isinstance(e, Halted):
                    raise _escaping(e) from e
                raise
            if not mark.recorded:
                record_tool_result(ctx, name, result=result, agent=who)
        _raise_pending()  # a halt the tool body set (e.g. a budget hit the SDK turned into an error message)
        return result

    def __agents_bind_function_tool__(self, function_tool: Any) -> _CheckedInvoker:
        bind = getattr(self._inner, "__agents_bind_function_tool__", None)
        inner = bind(function_tool) if callable(bind) else self._inner
        return self if inner is self._inner else _CheckedInvoker(inner)

    def __getattr__(self, attr: str) -> Any:
        inner = self.__dict__.get("_inner")  # absent while copy/pickle rebuilds the object
        if inner is None:
            raise AttributeError(attr)
        return getattr(inner, attr)


_ESCAPING: dict[type, type] = {}


def _escaping(e: Halted) -> Halted:
    """`e` as an instance of a subclass of its own class and AgentsException, which the SDK re-raises unchanged."""
    if isinstance(e, AgentsException):
        return e
    cls = type(e)
    sub = _ESCAPING.get(cls)
    if sub is None:
        sub = _ESCAPING[cls] = type(cls.__name__, (cls, AgentsException), {
            "__module__": cls.__module__, "__qualname__": cls.__qualname__,
            "__reduce__": lambda self: (cls, cls.__reduce__(self)[1])})
    out = sub.__new__(sub, *e.args)
    out.__dict__.update(e.__dict__)
    out.run_data = None
    return out


def _raise_pending() -> None:
    try:
        check_halt()
    except Halted as e:
        raise _escaping(e) from None


def _args_text(raw: str) -> str:
    """The canonical argument text (as `@pitstop.tool` computes it) of the model's JSON arguments, or the raw text
    when they are not a JSON object."""
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return raw if isinstance(raw, str) else repr(raw)
    return tool_args_text((), parsed) if isinstance(parsed, dict) else raw


def _reply_text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    try:
        return json.dumps(reply, separators=(",", ":"), default=str)
    except Exception:
        return repr(reply)


def _tag_run(run_id: str) -> None:
    """runs.adapter = 'openai-agents', unless a non-wrap adapter tagged the run first (wrap's tag is a default)."""
    with get_runtime().store.transaction() as c:
        c.execute("UPDATE runs SET adapter=? WHERE id=? AND (adapter IS NULL OR adapter='wrap')", (ADAPTER, run_id))
