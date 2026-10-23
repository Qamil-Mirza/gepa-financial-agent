"""LangChain / LangGraph adapter (`pip install "pitstop[langchain]"`): `PitStopMiddleware`, a LangChain v1
`AgentMiddleware` for `create_agent(...)` agents, which run on the LangGraph runtime:

    agent = create_agent(model, tools, middleware=[PitStopMiddleware(side_effects={"send_email"})])
    with pitstop.workflow("research", goal="...", criticality="high"):
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

Tools are guarded. `wrap_tool_call` / `awrap_tool_call` run the shared tool guard (`pitstop.sdk.guard`) on each tool
call before the tool runs: a pending halt, the L0 scope and loop checks, then the L1 pre-action gate for side-effect
tools on high and medium workflows (in a worker thread for `ainvoke`). A call it lets through runs under the guard's
check-once mark (a `@pitstop.tool` of the same name inside it skips its own guard) and one call records one
`tool_result` event. A blocked or steered call does not run: the model gets a `ToolMessage` with PitStop's reply (a
block as compact JSON) and `status="error"`. Outside a workflow the middleware passes every call through.

Halts reach the caller. `before_model` raises a pending halt before each model call, and the guard raises one (or
the steer limit) before each tool call, so `pitstop.Halted` (and `BudgetExceeded` from a wrapped client) propagate
out of `invoke` / `ainvoke`: LangGraph re-raises node exceptions, and LangChain's default tool error handler handles
only invalid tool arguments.

Model calls are metered only through a `pitstop.wrap`-ed client (`ChatOpenAI(client=wrap(OpenAI()).chat.completions,
...)`): that is where budgets are enforced. A model call the wrapped client did not meter (any other chat model) is
observed instead (plan decision 12): `wrap_model_call` records an `llm_call` event with the same payload keys as
`wrap`'s plus `"observed": true`, tokens from the reply's `usage_metadata` and a price-table cost estimate, and runs
the L0 model-call checks on it (invalid tool calls count as parse failures, refusals as refusals). It never writes a
ledger row, so an observed call cannot trip a budget; the first one names the run's model when no wrapped call did.

Agent naming: `agent=` names the agent for PitStop. `before_agent` names it for the scope `invoke` was called in
(`RunContext.set_framework_agent`, `pitstop.agent(name)` semantics: LangGraph runs nodes in copied contexts, so a
scope entered in a node would not reach the run), so the run's wrapped model calls, tool events, halt events and the
gate see it; `after_agent` removes the name (a halted invocation leaves it until the next one replaces it). The named
agent keeps the goal of the enclosing `pitstop.agent(...)` scope, which the gate judges side-effect calls against
(the workflow goal when there is none). Without `agent=` the enclosing scope's agent is used. `before_agent` also
tags the run `runs.adapter = 'langchain'`.

Limits: only tool calls that reach the agent's `ToolNode` are guarded (tools the model calls server side are not).
"Metered by wrap" is judged per run (`RunContext.llm_calls` before and after the model call), so two model calls of
one run in flight at once, one wrapped and one not, can both be taken as metered. Plain LangGraph `StateGraph` graphs
without `create_agent` use `@pitstop.tool` and `pitstop.wrap` directly.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

from ..context import RunContext, _meter, _safe, check_halt, current_run
from ..guard import aguard_call, checked, guard_call, record_tool_result, tool_args_text
from ..runtime import get_runtime

__all__ = ["PitStopMiddleware"]

log = logging.getLogger("pitstop")

ADAPTER = "langchain"
_REFUSAL_REASONS = ("content_filter", "refusal")


class PitStopMiddleware(AgentMiddleware):
    """PitStop for a `create_agent(...)` agent: guards tool calls, raises pending halts before model calls and
    records the model calls `pitstop.wrap` did not meter. Tools named in `side_effects` are side-effect tools (the L1
    pre-action gate judges them on high and medium workflows); `agent` names the agent for PitStop."""

    def __init__(self, *, side_effects: Iterable[str] = (), agent: str | None = None):
        super().__init__()
        self.side_effects = frozenset({side_effects} if isinstance(side_effects, str) else side_effects)
        self.agent = agent
        self._unknown_warned = False

    # -- agent and model nodes ---------------------------------------------------------------------------------

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        ctx = current_run()
        if ctx is not None:
            if self.agent:
                _safe(lambda: ctx.set_framework_agent(self.agent))
            _safe(lambda: _tag_run(ctx.run_id))
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        check_halt()
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        check_halt()
        return None

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        ctx = current_run()
        if ctx is not None and self.agent and ctx.agent == self.agent:
            _safe(lambda: ctx.set_framework_agent(None))
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        ctx = current_run()
        if ctx is None:
            return handler(request)
        _safe(lambda: self._check_tool_names(request))
        before = ctx.llm_calls
        response = handler(request)
        if ctx.llm_calls == before:  # no agent-meter call through a wrapped client: observe it
            _safe(lambda: self._observe(ctx, request, response))
        return response

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        ctx = current_run()
        if ctx is None:
            return await handler(request)
        _safe(lambda: self._check_tool_names(request))
        before = ctx.llm_calls
        response = await handler(request)
        if ctx.llm_calls == before:
            _safe(lambda: self._observe(ctx, request, response))
        return response

    # -- tool node ---------------------------------------------------------------------------------------------

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        ctx = current_run()
        if ctx is None:
            return handler(request)
        name, call_id, args_text = _call_of(request)
        decision = guard_call(ctx, name, name in self.side_effects, args_text, agent=self.agent)
        if decision.action != "run":
            return _refusal(decision.reply, name, call_id)
        with checked(ctx, name) as mark:
            try:
                result = handler(request)
            except BaseException as e:
                if not mark.recorded:
                    record_tool_result(ctx, name, error=e, agent=self.agent)
                raise
            if not mark.recorded:
                _record_result(ctx, name, result, self.agent)
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        ctx = current_run()
        if ctx is None:
            return await handler(request)
        name, call_id, args_text = _call_of(request)
        decision = await aguard_call(ctx, name, name in self.side_effects, args_text, agent=self.agent)
        if decision.action != "run":
            return _refusal(decision.reply, name, call_id)
        with checked(ctx, name) as mark:
            try:
                result = await handler(request)
            except BaseException as e:
                if not mark.recorded:
                    record_tool_result(ctx, name, error=e, agent=self.agent)
                raise
            if not mark.recorded:
                _record_result(ctx, name, result, self.agent)
        return result

    # -- internals ---------------------------------------------------------------------------------------------

    def _check_tool_names(self, request: Any) -> None:
        """Warn once when a side-effect name is none of the agent's tools: a typo would leave that tool ungated."""
        if self._unknown_warned or not self.side_effects:
            return
        names = {getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                 for t in request.tools or []}
        unknown = self.side_effects - names
        if unknown:
            self._unknown_warned = True
            log.warning("pitstop: PitStopMiddleware side_effects names no tool of the agent: %s (tools: %s)",
                        sorted(unknown), sorted(n for n in names if n))

    def _observe(self, ctx: RunContext, request: Any, response: Any) -> None:
        """Record an `llm_call` event for a model call no wrapped client metered, and run the L0 checks on it."""
        from ...budget.prices import cost_usd, price_for
        from ...detect.l0 import LLMCallRecord, check_llm_call, is_refusal
        from ..actions import apply_findings

        msg = _ai_message(response)
        if msg is None:
            return
        meta = msg.response_metadata or {}
        model = meta.get("model_name") or meta.get("model") or _model_name(request)
        calls = [{"name": tc.get("name"), "arguments": tool_args_text((), tc.get("args") or {})}
                 for tc in msg.tool_calls or []]
        invalid = [{"name": tc.get("name"), "arguments": tc.get("args")} for tc in msg.invalid_tool_calls or []]
        text = msg.text
        text = str(text) if text is not None else ""
        finish = meta.get("finish_reason") or meta.get("stop_reason")
        n_calls = len(calls) + len(invalid)
        refused = bool(finish in _REFUSAL_REASONS or (msg.additional_kwargs or {}).get("refusal")
                       or is_refusal(text, n_calls))
        usage = msg.usage_metadata or {}
        prompt, completion = usage.get("input_tokens"), usage.get("output_tokens")
        cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
        meter = _meter.get()
        payload = {"tool_calls": calls + invalid, "finish_reason": finish, "content_chars": len(text),
                   "meter": meter, "prompt_tokens": prompt, "completion_tokens": completion, "refused": refused,
                   "observed": True}
        cost, known = None, None
        if model:
            price, known = price_for(model, get_runtime().config.prices)
            if prompt is not None or completion is not None:
                cost = cost_usd(price, prompt or 0, completion or 0, cached)
        who = self.agent if self.agent is not None else ctx.agent
        with get_runtime().store.transaction() as c:
            c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                      (ctx.run_id, time.time(), who, "llm_call", model, json.dumps(payload, default=str), cost))
            if meter == "agent" and model:  # names the run's model unless a wrapped call already did
                c.execute("UPDATE runs SET model=COALESCE(model, ?) WHERE id=?", (model, ctx.run_id))
            if known is False:  # an unknown-priced call makes the run's cost a lower bound
                c.execute("UPDATE runs SET price_known=0 WHERE id=?", (ctx.run_id,))
        if meter != "agent":  # as with wrap: only the agent's own calls feed L0 detection
            return
        record = LLMCallRecord(len(invalid), n_calls, refused)
        with ctx.lock:
            if record.refused:
                ctx.refusals += 1
            findings = check_llm_call(ctx.llm_history, record, ctx.spec.sensitivity, ctx.refusals)
            ctx.llm_history.append(record)
        apply_findings(ctx, findings)


def _call_of(request: Any) -> tuple[str, Any, str]:
    """(name, id, canonical argument text as `@pitstop.tool` computes it) of a ToolCallRequest's tool call."""
    call = request.tool_call
    args = call.get("args")
    text = tool_args_text((), args) if isinstance(args, dict) else tool_args_text((args,), {})
    return call.get("name") or "", call.get("id"), text


def _refusal(reply: Any, name: str, call_id: Any) -> ToolMessage:
    """The ToolMessage a blocked or steered call returns to the model instead of the tool's result."""
    return ToolMessage(content=_reply_text(reply), tool_call_id=call_id, name=name, status="error")


def _record_result(ctx: RunContext, name: str, result: Any, agent: str | None) -> None:
    """A ToolMessage's content as the result (as an error when the tool node handled a tool error), else the
    result itself (a `Command`)."""
    if isinstance(result, ToolMessage):
        if result.status == "error":
            record_tool_result(ctx, name, error=ToolException(result.content), agent=agent)
        else:
            record_tool_result(ctx, name, result=result.content, agent=agent)
    else:
        record_tool_result(ctx, name, result=result, agent=agent)


def _ai_message(response: Any) -> AIMessage | None:
    """The AIMessage of a model handler's result: a ModelResponse, an ExtendedModelResponse or an AIMessage."""
    if isinstance(response, AIMessage):
        return response
    inner = getattr(response, "model_response", None)  # ExtendedModelResponse
    if inner is not None:
        response = inner
    for m in reversed(getattr(response, "result", None) or []):
        if isinstance(m, AIMessage):
            return m
    return None


def _model_name(request: Any) -> str | None:
    model = getattr(request, "model", None)
    for attr in ("model_name", "model"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _reply_text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    try:
        return json.dumps(reply, separators=(",", ":"), default=str)
    except Exception:
        return repr(reply)


def _tag_run(run_id: str) -> None:
    """runs.adapter = 'langchain', unless a non-wrap adapter tagged the run first (wrap's tag is a default)."""
    with get_runtime().store.transaction() as c:
        c.execute("UPDATE runs SET adapter=? WHERE id=? AND (adapter IS NULL OR adapter='wrap')", (ADAPTER, run_id))
