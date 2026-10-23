"""The OpenAI Agents SDK adapter: PitStop's tool guard as a native tool input guardrail, run hooks that name the
agent and tag the run, and a wrapped AsyncOpenAI client for metering. Offline: ScriptedModel or an httpx
MockTransport; the SDK's own trace export is disabled."""
import json

import pytest

pytest.importorskip("agents")

from agents import Agent, OpenAIResponsesModel, Runner, function_tool, set_tracing_disabled  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402

import pitstop  # noqa: E402
from pitstop.detect.drifts import list_drifts  # noqa: E402
from pitstop.sdk.adapters.openai_agents import Hooks, guard, tool_guardrail  # noqa: E402
from pitstop.sdk.runtime import get_runtime  # noqa: E402
from tests.sdk.fakes import FakeOpenAI, make_response  # noqa: E402
from tests.sdk.http_fakes import openai_client, response_body  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sdk_tracing():
    """The Agents SDK never uploads traces from a test."""
    set_tracing_disabled(True)
    yield


def _init(tmp_path, judge=None):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path, judge_client=judge)


def _verdict(serves: bool, reason: str):
    return make_response(content=json.dumps({"serves_goal": serves, "reason": reason}))


def _db():
    return get_runtime().store.connect()


def _events(kind):
    rows = _db().execute("SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["payload"] = json.loads(r["payload"])
    return out


def _tools(calls):
    @function_tool
    def search(q: str) -> str:
        """Search the web."""
        calls.append(("search", q))
        return f"results for {q}"

    @function_tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        calls.append(("send_email", to))
        return "sent"

    return search, send_email


def _call(name, args, i):
    return [function_call(name, args, call_id=f"call_{i}")]


async def test_off_scope_tool_rejected_with_reason(tmp_path):
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "x@evil.test", "body": "hi"}, 1),
                           [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium", tools=("search",)):
        result = await Runner.run(agent, "go", hooks=Hooks())
    assert result.final_output == "done"
    assert calls == []  # rejected before the tool ran
    second_input = json.dumps(model.calls[1].input, default=str)
    assert "blocked_by" in second_input and "out_of_scope_action" in second_input
    [d] = list_drifts(get_runtime().store, workflow="wf")
    assert d["drift_type"] == "out_of_scope_action" and d["level"] == "P0"
    [b] = _events("block")
    assert b["name"] == "send_email" and b["agent"] == "researcher"


async def test_side_effect_tool_gated(tmp_path):
    judge = FakeOpenAI([_verdict(False, "emails an outsider")])
    _init(tmp_path, judge)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "x@evil.test", "body": "hi"}, 1),
                           [assistant_message("done")]])
    agent = guard(Agent(name="writer", instructions="Write.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    with pitstop.workflow("wf", goal="Answer the ticket", criticality="medium"):
        await Runner.run(agent, "go", hooks=Hooks())
    assert calls == []
    [g] = _events("gate")
    assert g["payload"]["allow"] is False and g["agent"] == "writer"
    assert "emails an outsider" in json.dumps(model.calls[1].input, default=str)
    assert len(judge.calls) == 1


async def test_pending_halt_reaches_caller(tmp_path):
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "x@evil.test", "body": "hi"}, 1),
                           _call("search", {"q": "nvidia"}, 2),
                           [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    with pytest.raises(pitstop.Halted) as ei:
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="high", tools=("search",)) as ctx:
            await Runner.run(agent, "go", hooks=Hooks())
    assert "out_of_scope_action" in str(ei.value)
    assert calls == []
    assert len(model.calls) == 1  # Hooks.on_llm_start raised the pending halt before the next model call
    [row] = _db().execute("SELECT status FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["status"] == "halted"


async def test_guardrail_halt_reaches_caller_without_hooks(tmp_path):
    """Without Hooks the next guarded call raises the pending halt from inside the guardrail; the SDK wraps a
    foreign exception from a tool call in UserError, so the adapter's Halted is also an AgentsException."""
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "x@evil.test", "body": "hi"}, 1),
                           _call("search", {"q": "nvidia"}, 2),
                           [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    with pytest.raises(pitstop.Halted) as ei:
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="high", tools=("search",)):
            await Runner.run(agent, "go")
    assert "out_of_scope_action" in str(ei.value) and ei.value.reason == str(ei.value)
    assert calls == [] and len(model.calls) == 2


async def test_halt_reaches_consumer_of_streamed_run(tmp_path):
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "x@evil.test", "body": "hi"}, 1),
                           _call("search", {"q": "nvidia"}, 2),
                           [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    with pytest.raises(pitstop.Halted, match="out_of_scope_action"):
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="high", tools=("search",)):
            streamed = Runner.run_streamed(agent, "go")
            async for _ in streamed.stream_events():
                pass
    assert calls == []


async def test_halt_left_pending_by_tool_body_reaches_caller(tmp_path):
    """A budget hit inside a tool body: the SDK turns the BudgetExceeded into an error message for the model, and
    the invoker wrapper raises the pending halt it left."""
    _init(tmp_path)
    inner = pitstop.wrap(openai_client(lambda r: __import__("httpx").Response(200, json=response_body("x"))))

    @function_tool
    def summarize(text: str) -> str:
        """Summarize text with a model."""
        return inner.responses.create(model="openai/gpt-oss-120b", input=text).output_text

    model = ScriptedModel([_call("summarize", {"text": "long"}, 1), [assistant_message("done")]])
    agent = guard(Agent(name="lead", instructions="Lead.", model=model, tools=[summarize]))
    with pytest.raises(pitstop.Halted) as ei:
        with pitstop.workflow("wf", criticality="medium", budget_usd=0.000001):
            await Runner.run(agent, "go")
    assert "budget_exhausted" in str(ei.value)
    assert len(model.calls) == 1


async def test_wrapped_model_budget_exceeded_reaches_caller(tmp_path):
    _init(tmp_path)
    seen = []
    client = pitstop.wrap(openai_client(lambda r: seen.append(r) or __import__("httpx").Response(
        200, json=response_body("x")), async_=True))
    agent = Agent(name="analyst", instructions="Analyse.",
                  model=OpenAIResponsesModel("openai/gpt-oss-120b", openai_client=client))
    with pytest.raises(pitstop.BudgetExceeded):
        with pitstop.workflow("wf", criticality="medium", budget_usd=0.000001):
            await Runner.run(agent, "go", hooks=Hooks())
    assert seen == []


def test_escaped_halted_keeps_its_class():
    import pickle

    from agents.exceptions import AgentsException

    from pitstop.sdk.adapters.openai_agents import _escaping

    e = _escaping(pitstop.BudgetExceeded("workflow", 1.0, 1.5))
    assert isinstance(e, pitstop.BudgetExceeded) and isinstance(e, AgentsException)
    assert (e.scope, e.limit_usd, e.spent_usd, str(e)) == ("workflow", 1.0, 1.5, str(pitstop.BudgetExceeded(
        "workflow", 1.0, 1.5)))
    back = pickle.loads(pickle.dumps(e))
    assert type(back) is pitstop.BudgetExceeded and back.spent_usd == 1.5
    h = _escaping(pitstop.Halted("stop", 7))
    assert (h.reason, h.drift_id) == ("stop", 7) and type(pickle.loads(pickle.dumps(h))) is pitstop.Halted


async def test_guardrail_and_inner_tool_check_once(tmp_path):
    _init(tmp_path)
    ran = []

    @pitstop.tool(name="search")
    def search(q: str) -> str:
        """Search the web."""
        ran.append(q)
        return f"results for {q}"

    model = ScriptedModel([_call("search", {"q": "nvidia"}, 1), _call("search", {"q": "nvidia"}, 2),
                           [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=[function_tool(search)]))
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium") as ctx:
        await Runner.run(agent, "go", hooks=Hooks())
        assert len(ctx.tool_history_for("researcher")) == 2  # one entry per call, not two
        assert sum(len(h) for h in ctx.tool_history.values()) == 2
    assert ran == ["nvidia", "nvidia"]
    assert len(_events("tool_call")) == 2
    results = _events("tool_result")
    assert len(results) == 2 and all(r["agent"] == "researcher" for r in results)
    assert list_drifts(get_runtime().store, workflow="wf") == []  # no loop at half the threshold


async def test_wrapper_records_result_of_plain_tool(tmp_path):
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("search", {"q": "nvidia"}, 1), [assistant_message("done")]])
    agent = guard(Agent(name="researcher", instructions="Research.", model=model, tools=list(_tools(calls))))
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium"):
        await Runner.run(agent, "go", hooks=Hooks())
    assert calls == [("search", "nvidia")]
    [c] = _events("tool_call")
    assert c["agent"] == "researcher" and json.loads(c["payload"]["args"]) == {"args": [], "kwargs": {"q": "nvidia"}}
    [r] = _events("tool_result")
    assert r["name"] == "search" and r["agent"] == "researcher" and "results for nvidia" in r["payload"]["result"]


async def test_wrapped_client_metered_by_agents_sdk(tmp_path):
    _init(tmp_path)
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return __import__("httpx").Response(200, json=response_body("all done", input_tokens=40, output_tokens=7))

    client = pitstop.wrap(openai_client(handler, async_=True))
    agent = Agent(name="analyst", instructions="Analyse.",
                  model=OpenAIResponsesModel("openai/gpt-oss-120b", openai_client=client))
    with pitstop.workflow("wf", goal="Analyse", criticality="medium") as ctx:
        result = await Runner.run(agent, "go", hooks=Hooks())
    assert result.final_output == "all done"
    assert seen == ["/v1/responses"]
    [row] = _db().execute("SELECT state, meter FROM ledger").fetchall()
    assert (row["state"], row["meter"]) == ("settled", "agent")
    [e] = _events("llm_call")
    assert e["agent"] == "analyst"
    [run] = _db().execute("SELECT adapter FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert run["adapter"] == "openai-agents"


async def test_hooks_agent_does_not_override_inner_scope(tmp_path):
    """An explicit pitstop.agent() scope opened inside the run (here, in a tool body) still names its own calls."""
    _init(tmp_path)

    @function_tool
    def delegate(task: str) -> str:
        """Delegate a task."""
        with pitstop.agent("sub"):
            assert pitstop.sdk.context.current_run().agent == "sub"
        return pitstop.sdk.context.current_run().agent

    model = ScriptedModel([_call("delegate", {"task": "t"}, 1), [assistant_message("done")]])
    agent = guard(Agent(name="lead", instructions="Lead.", model=model, tools=[delegate]))
    with pitstop.workflow("wf", criticality="medium"):
        await Runner.run(agent, "go", hooks=Hooks())
    [r] = _events("tool_result")
    assert r["payload"]["result"] == repr("lead")


async def test_hooks_named_agent_keeps_enclosing_agent_goal(tmp_path):
    """Hooks name the SDK agent on top of the scope Runner.run was called in; that scope's goal still reaches the
    gate (the agent's directive, not only the broader workflow goal)."""
    judge = FakeOpenAI([_verdict(True, "fine")])
    _init(tmp_path, judge)
    seen = []

    @function_tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        ctx = pitstop.sdk.context.current_run()
        seen.append((ctx.agent, ctx.agent_goal))
        return "sent"

    model = ScriptedModel([_call("send_email", {"to": "ops@example.test", "body": "hi"}, 1),
                           [assistant_message("done")]])
    agent = guard(Agent(name="writer", instructions="Write.", model=model, tools=[send_email]),
                  side_effects={"send_email"})
    with pitstop.workflow("wf", goal="Run the support desk", criticality="medium"):
        with pitstop.agent("x", goal="Email the weekly report to ops"):
            await Runner.run(agent, "go", hooks=Hooks())
    assert seen == [("writer", "Email the weekly report to ops")]
    assert "Email the weekly report to ops" in judge.calls[0]["messages"][-1]["content"]
    [g] = _events("gate")
    assert g["agent"] == "writer" and g["payload"]["allow"] is True


async def test_outside_workflow_allows(tmp_path):
    _init(tmp_path)
    calls = []
    model = ScriptedModel([_call("send_email", {"to": "a@b.test", "body": "hi"}, 1), [assistant_message("done")]])
    agent = guard(Agent(name="writer", instructions="Write.", model=model, tools=list(_tools(calls))),
                  side_effects={"send_email"})
    result = await Runner.run(agent, "go", hooks=Hooks())
    assert result.final_output == "done" and calls == [("send_email", "a@b.test")]
    assert _db().execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_guard_is_idempotent_and_marks_side_effects():
    search, send_email = _tools([])
    agent = Agent(name="a", instructions="x", tools=[search, send_email])
    assert guard(guard(agent, side_effects={"send_email"}), side_effects={"send_email"}) is agent
    for t in (search, send_email):
        assert len(t.tool_input_guardrails) == 1
    guard_names = {t.name: t.tool_input_guardrails[0].get_name() for t in agent.tools}
    assert guard_names == {"search": "pitstop", "send_email": "pitstop_side_effects"}
    guard(agent, side_effects="search")  # a later call upgrades a guarded tool to side-effect, still once
    assert [g.get_name() for g in search.tool_input_guardrails] == ["pitstop_side_effects"]
    with pytest.raises(ValueError, match="send_mail"):  # a typo never leaves a side-effect tool ungated
        guard(agent, side_effects={"send_mail"})


def test_tool_guardrail_is_a_tool_input_guardrail():
    from agents import ToolInputGuardrail

    assert isinstance(tool_guardrail(), ToolInputGuardrail)
    assert isinstance(tool_guardrail(side_effects=True), ToolInputGuardrail)


async def test_nested_agent_end_restores_outer_name(tmp_path):
    """An agent run as a tool (its own Runner.run with Hooks) names its calls, then hands the name back to the
    outer agent; a handoff replaces the name, and the run's end removes it."""
    _init(tmp_path)

    @function_tool
    def whoami() -> str:
        """The agent PitStop names for this call."""
        return pitstop.sdk.context.current_run().agent

    sub = Agent(name="sub", instructions="Answer.", model=ScriptedModel([_call("whoami", {}, 1),
                                                                         [assistant_message("sub answer")]]),
                tools=[whoami])
    lead_model = ScriptedModel([_call("ask_sub", {"input": "q"}, 2), _call("whoami", {}, 3),
                                [assistant_message("done")]])
    lead = guard(Agent(name="lead", instructions="Lead.", model=lead_model,
                       tools=[sub.as_tool("ask_sub", "Ask the sub agent.", hooks=Hooks()), whoami]))
    with pitstop.workflow("wf", criticality="medium") as ctx:
        result = await Runner.run(lead, "go", hooks=Hooks())
        assert result.final_output == "done" and ctx.agent is None
        hooks = Hooks()  # a handoff: on_agent_end only for the agent handed to
        a, b = Agent(name="a"), Agent(name="b")
        await hooks.on_agent_start(None, a)
        await hooks.on_handoff(None, a, b)
        await hooks.on_agent_start(None, b)
        assert ctx.agent == "b"
        await hooks.on_agent_end(None, b, "out")
        assert ctx.agent is None
    named = [r["payload"]["result"] for r in _events("tool_result") if r["name"] == "whoami"]
    assert named == [repr("sub"), repr("lead")]


async def test_tag_run_once_per_run(tmp_path, monkeypatch):
    from contextlib import contextmanager

    _init(tmp_path)
    store = get_runtime().store
    real = store.transaction
    seen = []

    class Spy:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args):
            seen.append(sql)
            return self._conn.execute(sql, *args)

        def __getattr__(self, attr):
            return getattr(self._conn, attr)

    @contextmanager
    def transaction(*a, **k):
        with real(*a, **k) as c:
            yield Spy(c)

    monkeypatch.setattr(store, "transaction", transaction)
    for _ in range(2):
        with pitstop.workflow("wf", criticality="medium") as ctx:
            for _ in range(2):
                agent = Agent(name="a", instructions="x", model=ScriptedModel([[assistant_message("done")]]))
                await Runner.run(agent, "go", hooks=Hooks())
        [run] = _db().execute("SELECT adapter FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
        assert run["adapter"] == "openai-agents"
    assert len([s for s in seen if s.startswith("UPDATE runs SET adapter")]) == 2  # once per run


def test_checked_invoker_keeps_signature():
    import inspect

    search, _ = _tools([])
    original = search.on_invoke_tool
    guard(Agent(name="a", instructions="x", tools=[search]))
    assert search.on_invoke_tool is not original
    assert inspect.signature(search.on_invoke_tool) == inspect.signature(original)
