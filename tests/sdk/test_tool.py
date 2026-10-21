import asyncio
import json

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.sdk.context import agent, workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import FakeOpenAI, make_response

MODEL = "openai/gpt-oss-120b"
MSGS = [{"role": "user", "content": "hi"}]


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _events(kind):
    rows = get_runtime().store.connect().execute(
        "SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    return [dict(r) for r in rows]


def test_tool_runs_normally_and_records_events(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return f"r:{q}"

    with workflow("wf"):
        assert search("a") == "r:a"
    assert len(_events("tool_call")) == 1
    res = _events("tool_result")
    assert len(res) == 1 and res[0]["name"] == "search"


def test_out_of_scope_side_effect_tool_is_blocked(tmp_path):
    _init(tmp_path)
    called = []

    @pitstop.tool(side_effects=True)
    def send_email(to):
        called.append(to)

    with workflow("wf", criticality="medium", tools=["search"]):
        out = send_email("x")
    assert called == []
    assert out["blocked_by"] == "pitstop" and out["reason"].startswith("out_of_scope_action: ")
    assert set(out) == {"blocked_by", "reason"}
    d = list_drifts(get_runtime().store)
    assert [(r["drift_type"], r["level"], r["action"]) for r in d] == [("out_of_scope_action", "P0", "block")]
    assert len(_events("block")) == 1


def test_high_criticality_p0_halts_at_next_boundary(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response(content="ok")]))

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    with pytest.raises(Halted):
        with workflow("wf", criticality="high", tools=["search"]):
            out = send_email("x")
            assert out["blocked_by"] == "pitstop"
            w.chat.completions.create(model=MODEL, messages=MSGS)


def test_read_only_loop_is_steered_twice_then_halts(tmp_path):
    _init(tmp_path)
    calls = []

    @pitstop.tool
    def search(q):
        calls.append(q)
        return "r"

    with pytest.raises(Halted, match="steer limit exceeded"):
        with workflow("wf", goal="wf goal"):
            assert search("a") == "r"
            assert search("a") == "r"
            assert search("a").startswith("PitStop: this step drifts")
            assert search("a").startswith("PitStop: this step drifts")
            search("a")
    assert len(calls) == 2
    assert len(_events("steer")) == 2


def test_steer_message_uses_agent_goal_then_workflow_goal(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return "r"

    with workflow("wf", goal="wf goal"):
        with agent("a", goal="agent goal"):
            for _ in range(3):
                m = search("a")
        assert m.endswith("Return to: agent goal.")
    with workflow("wf2", goal="wf goal"):
        for _ in range(3):
            m = search("a")
        assert m.endswith("Return to: wf goal.")
    with workflow("wf3"):
        for _ in range(3):
            m = search("a")
        assert m.endswith("Return to: your original task.")


def test_parallel_agents_similar_calls_are_not_a_loop(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return "r"

    async def run(name):
        with agent(name):
            await asyncio.sleep(0)
            for _ in range(2):
                assert search("same") == "r"

    async def main():
        async with workflow("wf"):
            await asyncio.gather(run("a"), run("b"))

    asyncio.run(main())
    assert list_drifts(get_runtime().store) == []


def test_async_tool(tmp_path):
    _init(tmp_path)

    @pitstop.tool(side_effects=True, name="mail")
    async def send(to):
        return "sent"

    async def main():
        async with workflow("wf", tools=["mail"]):
            assert await send("x") == "sent"
        async with workflow("wf", tools=["other"]):
            return await send("x")

    out = asyncio.run(main())
    assert out["blocked_by"] == "pitstop"


def test_tool_outside_workflow_just_runs(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def f(x):
        return x + 1

    assert f(1) == 2
    assert _events("tool_call") == []


def test_tool_exception_propagates(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def boom():
        raise ValueError("bad")

    with workflow("wf"):
        with pytest.raises(ValueError, match="bad"):
            boom()
    res = _events("tool_result")
    assert "error" in json.loads(res[0]["payload"])
