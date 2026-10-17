import asyncio
import json

import pytest

from gepa_financial_agent.crusoe_client import MockModelClient
from gepa_financial_agent.policy_store import Policy
from gepa_financial_agent.schemas import EventType, TrajectoryEvent
from gepa_financial_agent.session_manager import SupervisorManager, TriggerConfig

P1 = Policy.from_text("policy one", name="p1")


def ev(agent, i, etype=EventType.AGENT_MESSAGE, tool=None, args=None):
    return TrajectoryEvent(event_id=f"{agent}_e{i}", task_id=f"task_{agent}", agent_id=agent, step=i,
                           event_type=etype, tool_name=tool, tool_arguments=args, content=f"{agent} says {i}")


def ok(status="ON_TRACK", evidence=()):
    return json.dumps({"status": status, "confidence": 0.6, "drift_type": None, "evidence_event_ids": list(evidence),
                       "explanation": "mock", "recommended_action": "NONE"})


def manager(client=None, **kw):
    return SupervisorManager(client or MockModelClient(), policy=P1, **kw)


def test_register_and_duplicate():
    m = manager()
    s = m.register_agent("A", "task_A", "dir A", "swarm")
    assert s.session_id.startswith("supervisor_A") and s.policy.version == P1.version
    with pytest.raises(ValueError):
        m.register_agent("A", "task_A", "dir A", "swarm")


def test_event_count_and_type_triggers():
    m = manager(trigger=TriggerConfig(every_n_events=2))
    m.register_agent("A", "task_A", "d", "s")
    assert asyncio.run(m.observe("A", ev("A", 0))) is False
    assert asyncio.run(m.observe("A", ev("A", 1))) is True
    m.register_agent("B", "task_B", "d", "s")
    assert asyncio.run(m.observe("B", ev("B", 0, EventType.ERROR))) is True


def test_repeated_tool_call_triggers():
    m = manager(trigger=TriggerConfig(every_n_events=100, repeated_tool_call_threshold=3))
    m.register_agent("A", "task_A", "d", "s")
    args = {"query": "same"}
    r = [asyncio.run(m.observe("A", ev("A", i, EventType.TOOL_CALL, "search", args))) for i in range(3)]
    assert r == [False, False, True]


def test_observe_rejects_mismatched_agent():
    m = manager()
    m.register_agent("A", "task_A", "d", "s")
    with pytest.raises(ValueError):
        asyncio.run(m.observe("A", ev("B", 0)))
    with pytest.raises(KeyError):
        asyncio.run(m.observe("Z", ev("Z", 0)))


def test_sessions_are_isolated_and_coalesced():
    seen = []

    def responder(messages):
        seen.append(messages[1]["content"])
        return ok()

    m = manager(MockModelClient(responder), trigger=TriggerConfig(every_n_events=1))
    m.register_agent("A", "task_A", "directive A", "s")
    m.register_agent("B", "task_B", "directive B", "s")

    async def run():
        for i in range(10):
            await m.observe("A", ev("A", i))
        for i in range(3):
            await m.observe("B", ev("B", i))
        return await m.evaluate_pending()

    recs = asyncio.run(run())
    assert len(recs) == 2  # ten pending events on A coalesce into one evaluation
    by_agent = {r.agent_id: r for r in recs}
    assert set(by_agent) == {"A", "B"}
    prompt_a = next(p for p in seen if "directive A" in p)
    prompt_b = next(p for p in seen if "directive B" in p)
    assert "B_e0" not in prompt_a and "A_e0" not in prompt_b and "[A_e9]" in prompt_a
    assert m.latest("A").decision.status.value == "ON_TRACK" and m.decisions("B")[0].agent_id == "B"
    assert asyncio.run(m.evaluate_pending()) == []  # nothing pending afterwards


def test_concurrency_is_bounded():
    client = MockModelClient(latency_s=0.03)
    m = manager(client, max_concurrent_inferences=2, trigger=TriggerConfig(every_n_events=1))

    async def run():
        for a in "ABCDEF":
            m.register_agent(a, f"task_{a}", "d", "s")
            await m.observe(a, ev(a, 0))
        return await m.evaluate_pending()

    recs = asyncio.run(run())
    assert len(recs) == 6 and client.tracker.snapshot()["max_in_flight"] == 2


def test_policy_pinned_per_session():
    m = manager()
    a = m.register_agent("A", "task_A", "d", "s")
    p2 = Policy.from_text("policy two", name="p2")
    m.set_policy(p2)
    b = m.register_agent("B", "task_B", "d", "s")
    assert a.policy.version == P1.version and b.policy.version == p2.version

    async def run():
        await m.observe("A", ev("A", 0, EventType.TASK_COMPLETION))
        await m.observe("B", ev("B", 0, EventType.TASK_COMPLETION))
        return await m.evaluate_pending()

    recs = {r.agent_id: r for r in asyncio.run(run())}
    assert recs["A"].policy_version == P1.version and recs["B"].policy_version == p2.version


def test_close_export_replay():
    m = manager(trigger=TriggerConfig(every_n_events=2))
    m.register_agent("A", "task_A", "d", "s", constraints=["c"], dependencies=["task_E"])

    async def run():
        for i in range(4):
            if await m.observe("A", ev("A", i)):
                await m.evaluate_pending()  # evaluations happen at steps 1 and 3
        exp = m.export_session("A")
        closed = m.close_session("A")
        replayed = await m.replay(exp)
        return exp, closed, replayed

    exp, closed, replayed = asyncio.run(run())
    assert closed.closed and "A" not in m.sessions
    assert len(exp["events"]) == 4 and len(exp["records"]) == 2 and exp["dependencies"] == ["task_E"]
    assert [r.observation_step for r in replayed] == [r["observation_step"] for r in exp["records"]]
    json.dumps(exp)  # JSON-serializable
    stats = m.stats()
    assert stats["usage"]["requests"] == 4 and stats["policy"]["version"] == P1.version
