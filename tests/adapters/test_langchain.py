"""The LangChain / LangGraph adapter: `PitStopMiddleware` on a `create_agent(...)` agent guards tool calls with the
shared tool guard, raises a pending halt before the next model call, and records the model calls `pitstop.wrap` did
not meter. Offline: a scripted fake chat model, or a real `ChatOpenAI` on a wrapped client over an httpx
MockTransport; LangSmith tracing is off."""
import json

import httpx
import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import pitstop  # noqa: E402
from pitstop.detect.drifts import list_drifts  # noqa: E402
from pitstop.sdk.adapters.langchain import PitStopMiddleware  # noqa: E402
from pitstop.sdk.runtime import get_runtime  # noqa: E402
from tests.sdk.fakes import FakeOpenAI, make_response  # noqa: E402
from tests.sdk.http_fakes import MODEL, chat_completion, openai_client  # noqa: E402


@pytest.fixture(autouse=True)
def _no_langsmith(monkeypatch):
    """No trace ever leaves a test for LangSmith."""
    for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(key, "false")
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class ScriptedChat(FakeMessagesListChatModel):
    """A fake chat model that accepts `bind_tools` (none of langchain-core's fakes do) and replays its replies."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _reply(content="", calls=(), *, invalid=(), tokens=(30, 8), i=0):
    tcs = [{"name": n, "args": a, "id": f"call_{i}_{k}", "type": "tool_call"} for k, (n, a) in enumerate(calls)]
    bad = [{"name": n, "args": raw, "id": f"bad_{i}_{k}", "error": "invalid JSON", "type": "invalid_tool_call"}
           for k, (n, raw) in enumerate(invalid)]
    return AIMessage(content=content, tool_calls=tcs, invalid_tool_calls=bad,
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                     "total_tokens": tokens[0] + tokens[1]},
                     response_metadata={"model_name": "fake-1"})


def _model(*replies):
    return ScriptedChat(responses=list(replies))


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
    @tool
    def search(q: str) -> str:
        """Search the web."""
        calls.append(("search", q))
        return f"results for {q}"

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        calls.append(("send_email", to))
        return "sent"

    return [search, send_email]


def _tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_off_scope_tool_blocked_with_tool_message(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=1), _reply("done", i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"},
                                                                             agent="researcher")])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium", tools=("search",)) as ctx:
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert result["messages"][-1].content == "done"
    assert calls == []  # blocked before the tool ran
    [tm] = _tool_messages(result)
    assert tm.status == "error" and tm.name == "send_email" and tm.tool_call_id == "call_1_0"
    assert "blocked_by" in tm.content and "out_of_scope_action" in tm.content
    [d] = list_drifts(get_runtime().store, workflow="wf")
    assert d["drift_type"] == "out_of_scope_action" and d["level"] == "P0"
    [b] = _events("block")
    assert b["name"] == "send_email" and b["agent"] == "researcher"
    [row] = _db().execute("SELECT adapter FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["adapter"] == "langchain"


async def test_blocked_tool_call_without_an_id_returns_a_tool_message(tmp_path):
    from types import SimpleNamespace

    _init(tmp_path)
    mw = PitStopMiddleware(side_effects={"send_email"})
    request = SimpleNamespace(tool_call={"name": "send_email", "args": {"to": "x@evil.test", "body": "hi"}})

    def handler(_):
        raise AssertionError("a blocked call never reaches the tool")

    async def ahandler(_):
        raise AssertionError("a blocked call never reaches the tool")

    with pitstop.workflow("wf", goal="Research NVIDIA", tools=("search",)):
        tm = mw.wrap_tool_call(request, handler)
        atm = await mw.awrap_tool_call(request, ahandler)
    for m in (tm, atm):
        assert isinstance(m, ToolMessage) and m.status == "error" and m.tool_call_id == ""
        assert "out_of_scope_action" in m.content


def test_loop_steers_then_halts(tmp_path):
    _init(tmp_path)
    calls = []
    replies = [_reply(calls=[("search", {"q": "nvidia"})], i=i) for i in range(1, 7)] + [_reply("done", i=7)]
    agent = create_agent(_model(*replies), _tools(calls), middleware=[PitStopMiddleware()])
    with pytest.raises(pitstop.Halted, match="steer limit"):
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium"):
            agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert calls == [("search", "nvidia")] * 2  # the 3rd and 4th repeats are steered, the 5th halts
    steers = _events("steer")
    assert len(steers) == 2 and all(s["name"] == "search" for s in steers)
    [h] = _events("halt")
    assert h["payload"]["reason"] == "steer limit exceeded"


def test_side_effect_tool_gated(tmp_path):
    judge = FakeOpenAI([_verdict(False, "emails an outsider")])
    _init(tmp_path, judge)
    calls = []
    model = _model(_reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=1), _reply("done", i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"},
                                                                             agent="writer")])
    with pitstop.workflow("wf", goal="Answer the ticket", criticality="medium"):
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert calls == []
    [g] = _events("gate")
    assert g["payload"]["allow"] is False and g["agent"] == "writer"
    [tm] = _tool_messages(result)
    assert tm.status == "error" and "emails an outsider" in tm.content
    assert len(judge.calls) == 1


def test_allowed_tool_runs_and_records_result(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("search", {"q": "nvidia"})], i=1), _reply("done", i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(agent="researcher")])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium"):
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert calls == [("search", "nvidia")]
    [c] = _events("tool_call")
    assert c["agent"] == "researcher" and json.loads(c["payload"]["args"]) == {"args": [], "kwargs": {"q": "nvidia"}}
    [r] = _events("tool_result")
    assert r["name"] == "search" and r["agent"] == "researcher" and "results for nvidia" in r["payload"]["result"]


def test_middleware_and_inner_tool_check_once(tmp_path):
    _init(tmp_path)
    ran = []

    @pitstop.tool(name="search")
    def search_impl(q: str) -> str:
        ran.append(q)
        return f"results for {q}"

    @tool
    def search(q: str) -> str:
        """Search the web."""
        return search_impl(q)

    model = _model(_reply(calls=[("search", {"q": "nvidia"})], i=1), _reply(calls=[("search", {"q": "nvidia"})], i=2),
                   _reply("done", i=3))
    agent = create_agent(model, [search], middleware=[PitStopMiddleware()])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium") as ctx:
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        assert sum(len(h) for h in ctx.tool_history.values()) == 2  # one entry per call, not two
    assert ran == ["nvidia", "nvidia"]
    assert len(_events("tool_call")) == 2 and len(_events("tool_result")) == 2
    assert list_drifts(get_runtime().store, workflow="wf") == []  # no loop at half the threshold


def test_unwrapped_model_observed_not_metered(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("search", {"q": "nvidia"})], invalid=[("search", '{"q": nvid')], tokens=(120, 20),
                          i=1),
                   _reply("done", tokens=(150, 4), i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(agent="researcher")])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium") as ctx:
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        assert len(ctx.llm_history) == 2 and ctx.llm_history[0].tool_call_parse_failures == 1
        assert ctx.llm_history[0].tool_calls == 2 and ctx.llm_calls == 0
    events = _events("llm_call")
    assert len(events) == 2
    first, second = events
    assert first["name"] == "fake-1" and first["agent"] == "researcher"
    p = first["payload"]
    assert p["observed"] is True and p["meter"] == "agent" and p["refused"] is False
    assert (p["prompt_tokens"], p["completion_tokens"]) == (120, 20)
    assert [tc["name"] for tc in p["tool_calls"]] == ["search", "search"]
    assert {"tool_calls", "finish_reason", "content_chars", "meter", "prompt_tokens", "completion_tokens",
            "refused"} <= set(p)
    assert first["cost_usd"] is not None and first["cost_usd"] > 0  # a price-table estimate
    assert second["payload"]["content_chars"] == 4 and second["payload"]["tool_calls"] == []
    assert _db().execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 0  # no ledger row: no budget
    [row] = _db().execute("SELECT model FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["model"] == "fake-1"


def test_observed_refusals_feed_l0(tmp_path):
    _init(tmp_path)
    model = _model(_reply("I can't help with that.", i=1))
    agent = create_agent(model, [], middleware=[PitStopMiddleware()])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium", sensitivity="high") as ctx:
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        assert ctx.refusals == 1
    [e] = _events("llm_call")
    assert e["payload"]["refused"] is True
    assert any(d["drift_type"] == "refusal_spike" for d in list_drifts(get_runtime().store, workflow="wf"))


def test_wrapped_chatopenai_metered_once(tmp_path):
    from langchain_openai import ChatOpenAI

    _init(tmp_path)
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=chat_completion("all done", prompt_tokens=40, completion_tokens=7))

    wrapped = pitstop.wrap(openai_client(handler))
    model = ChatOpenAI(model=MODEL, client=wrapped.chat.completions, api_key="x")
    agent = create_agent(model, _tools([]), middleware=[PitStopMiddleware(agent="analyst")])
    with pitstop.workflow("wf", goal="Analyse", criticality="medium") as ctx:
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        assert ctx.agent is None  # after_agent removed the name
    assert result["messages"][-1].content == "all done"
    assert seen == ["/v1/chat/completions"]
    [row] = _db().execute("SELECT state, meter FROM ledger").fetchall()
    assert (row["state"], row["meter"]) == ("settled", "agent")
    [e] = _events("llm_call")
    assert "observed" not in e["payload"] and e["payload"]["prompt_tokens"] == 40
    assert e["agent"] == "analyst"  # the wrapped call inside the model node sees the agent before_agent named
    [run] = _db().execute("SELECT adapter, model FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert (run["adapter"], run["model"]) == ("langchain", MODEL)


def test_pending_halt_raises_from_invoke(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=1),
                   _reply(calls=[("search", {"q": "nvidia"})], i=2), _reply("done", i=3))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"})])
    with pytest.raises(pitstop.Halted) as ei:
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="high", tools=("search",)) as ctx:
            agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert "out_of_scope_action" in str(ei.value)
    assert calls == []
    assert model.i == 1  # before_model raised the pending halt before the second model call
    [row] = _db().execute("SELECT status FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["status"] == "halted"


def test_outside_workflow_passes_through(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=1), _reply("done", i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"})])
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert result["messages"][-1].content == "done"
    assert calls == [("send_email", "x@evil.test")]
    assert _db().execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


async def test_async_agent_ainvoke(tmp_path):
    judge = FakeOpenAI([_verdict(False, "emails an outsider")])
    _init(tmp_path, judge)
    calls = []
    model = _model(_reply(calls=[("search", {"q": "nvidia"})], i=1),
                   _reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=2), _reply("done", i=3))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"},
                                                                             agent="researcher")])
    with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium") as ctx:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert result["messages"][-1].content == "done"
    assert calls == [("search", "nvidia")]
    blocked = [m for m in _tool_messages(result) if m.status == "error"]
    assert len(blocked) == 1 and "emails an outsider" in blocked[0].content
    [g] = _events("gate")
    assert g["payload"]["allow"] is False and g["agent"] == "researcher"
    [r] = _events("tool_result")
    assert r["name"] == "search" and r["agent"] == "researcher"
    observed = [e for e in _events("llm_call") if e["payload"]["meter"] == "agent"]  # not the judge's call
    assert len(observed) == 3 and all(e["payload"]["observed"] is True for e in observed)
    [row] = _db().execute("SELECT adapter FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["adapter"] == "langchain"


async def test_async_pending_halt_raises_from_ainvoke(tmp_path):
    _init(tmp_path)
    calls = []
    model = _model(_reply(calls=[("send_email", {"to": "x@evil.test", "body": "hi"})], i=1), _reply("done", i=2))
    agent = create_agent(model, _tools(calls), middleware=[PitStopMiddleware(side_effects={"send_email"})])
    with pytest.raises(pitstop.Halted, match="out_of_scope_action"):
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="high", tools=("search",)):
            await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert calls == [] and model.i == 1


def test_wrapped_chatopenai_budget_exceeded_reaches_caller(tmp_path):
    from langchain_openai import ChatOpenAI

    _init(tmp_path)
    seen = []
    wrapped = pitstop.wrap(openai_client(lambda r: seen.append(r) or httpx.Response(200, json=chat_completion("x"))))
    model = ChatOpenAI(model=MODEL, client=wrapped.chat.completions, api_key="x")
    agent = create_agent(model, _tools([]), middleware=[PitStopMiddleware()])
    with pytest.raises(pitstop.BudgetExceeded):
        with pitstop.workflow("wf", criticality="medium", budget_usd=0.000001):
            agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert seen == []
    assert _events("llm_call") == []  # nothing observed for a call that never happened


def test_wrapped_chatopenai_under_other_meter_recorded_once(tmp_path):
    from langchain_openai import ChatOpenAI

    from pitstop.sdk.wrap import meter

    _init(tmp_path)
    wrapped = pitstop.wrap(openai_client(lambda r: httpx.Response(200, json=chat_completion("all done"))))
    model = ChatOpenAI(model=MODEL, client=wrapped.chat.completions, api_key="x")
    agent = create_agent(model, _tools([]), middleware=[PitStopMiddleware()])
    with pitstop.workflow("wf", goal="Summarize", criticality="medium"):
        with meter("summarizer"):
            agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    [e] = _events("llm_call")
    assert "observed" not in e["payload"] and e["payload"]["meter"] == "summarizer"


async def test_concurrent_wrapped_and_unwrapped_calls_observed_per_call(tmp_path):
    """The unwrapped model call is still in flight when the wrapped one of the same run is metered: only the
    unwrapped one is observed."""
    import asyncio
    import threading

    from langchain.agents.middleware import AgentMiddleware
    from langchain_openai import ChatOpenAI

    _init(tmp_path)
    metered = threading.Barrier(2)  # the unwrapped call returns only once the wrapped one is metered

    class Held(ScriptedChat):
        def _generate(self, *args, **kwargs):
            metered.wait(10)
            return super()._generate(*args, **kwargs)

    class AfterMetering(AgentMiddleware):
        """Inside PitStopMiddleware: its handler returns after the wrapped client recorded the call."""

        async def awrap_model_call(self, request, handler):
            response = await handler(request)
            await asyncio.to_thread(metered.wait, 10)
            return response

    slow = Held(responses=[_reply("slow done", i=1)])
    sync_w = pitstop.wrap(openai_client(lambda r: httpx.Response(200, json=chat_completion("fast done"))))
    async_w = pitstop.wrap(openai_client(lambda r: httpx.Response(200, json=chat_completion("fast done")),
                                         async_=True))
    fast = ChatOpenAI(model=MODEL, client=sync_w.chat.completions, async_client=async_w.chat.completions,
                      api_key="x")
    a = create_agent(slow, [], middleware=[PitStopMiddleware()])
    b = create_agent(fast, [], middleware=[PitStopMiddleware(), AfterMetering()])
    msg = {"messages": [{"role": "user", "content": "go"}]}
    with pitstop.workflow("wf", goal="Research", criticality="medium"):
        ra, rb = await asyncio.gather(a.ainvoke(msg), b.ainvoke(msg))
    assert (ra["messages"][-1].content, rb["messages"][-1].content) == ("slow done", "fast done")
    events = _events("llm_call")
    assert sorted((e["name"], e["payload"].get("observed", False)) for e in events) == [("fake-1", True),
                                                                                       (MODEL, False)]
    assert events[0]["name"] == MODEL  # the wrapped call finished while the unwrapped one was in flight


async def test_async_loop_halt_raises_from_ainvoke(tmp_path):
    _init(tmp_path)
    calls = []
    replies = [_reply(calls=[("search", {"q": "nvidia"})], i=i) for i in range(1, 7)] + [_reply("done", i=7)]
    agent = create_agent(_model(*replies), _tools(calls), middleware=[PitStopMiddleware()])
    with pytest.raises(pitstop.Halted, match="steer limit"):
        with pitstop.workflow("wf", goal="Research NVIDIA", criticality="medium"):
            await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    assert calls == [("search", "nvidia")] * 2
    assert len(_events("steer")) == 2 and len(_events("halt")) == 1


def test_interrupt_in_tool_passes_through_unrecorded(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt

    _init(tmp_path)

    @tool
    def send_email(to: str) -> str:
        """Send an email after a human approves."""
        interrupt("approve?")
        return "sent"

    model = _model(_reply(calls=[("send_email", {"to": "ops@example.com"})], i=1), _reply("done", i=2))
    agent = create_agent(model, [send_email], middleware=[PitStopMiddleware()], checkpointer=InMemorySaver())
    with pitstop.workflow("wf", goal="Report", criticality="low"):
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                              {"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in result
    assert len(_events("tool_call")) == 1 and _events("tool_result") == []


def test_observed_event_named_by_requested_model(tmp_path):
    """The observed event is named by the model the request named, not the (dated or routed) id the reply echoes."""

    class Named(ScriptedChat):
        model_name: str = "requested-1"

    _init(tmp_path)
    agent = create_agent(Named(responses=[_reply("done", i=1)]), [], middleware=[PitStopMiddleware()])
    with pitstop.workflow("wf", goal="Research", criticality="medium") as ctx:
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    [e] = _events("llm_call")
    assert e["name"] == "requested-1" and e["payload"]["observed"] is True
    [row] = _db().execute("SELECT model FROM runs WHERE id=?", (ctx.run_id,)).fetchall()
    assert row["model"] == "requested-1"
