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
    # event IDs are aliased in prompts, so isolation is checked on event content
    assert "B says" not in prompt_a and "A says" not in prompt_b
    assert "A says 9" in prompt_a and "Trajectory (10 events)" in prompt_a and "Trajectory (3 events)" in prompt_b
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


def test_duplicate_event_is_idempotent_and_conflicting_duplicate_rejected():
    m = manager(MockModelClient(), trigger=TriggerConfig(every_n_events=1))
    m.register_agent("A", "task_A", "d", "s")
    e = ev("A", 0)
    asyncio.run(m.observe("A", e))
    asyncio.run(m.observe("A", e))  # at-least-once redelivery
    assert len(m.sessions["A"].events) == 1
    clash = ev("A", 0).model_copy(update={"content": "different"})
    with pytest.raises(ValueError):
        asyncio.run(m.observe("A", clash))
    recs = asyncio.run(m.evaluate_pending())
    assert len(recs) == 1 and recs[0].error is None


def test_one_failing_session_does_not_stop_others():
    m = manager(MockModelClient(), trigger=TriggerConfig(every_n_events=1))
    for a in "AB":
        m.register_agent(a, f"task_{a}", "d", "s")
        asyncio.run(m.observe(a, ev(a, 0)))

    def boom():
        raise RuntimeError("bad session state")

    m.sessions["A"].build_input = boom
    recs = {r.agent_id: r for r in asyncio.run(m.evaluate_pending())}
    assert set(recs) == {"A", "B"}
    assert recs["A"].decision is None and "bad session state" in recs["A"].error
    assert recs["B"].error is None and recs["B"].decision is not None
    assert m.sessions["A"].evaluating is False


def test_adapter_survives_duplicate_events_for_other_agents():
    from gepa_financial_agent.adapters.base import EventStreamAdapter
    m = manager(MockModelClient(), trigger=TriggerConfig(every_n_events=1))
    for a in "AB":
        m.register_agent(a, f"task_{a}", "d", "s")
    adapter = EventStreamAdapter(m, eval_interval_s=0.01)

    async def run():
        stop = asyncio.Event()
        consumer = asyncio.create_task(adapter.run(stop))
        await adapter.emit("A", ev("A", 0, etype=EventType.ERROR))
        await adapter.emit("A", ev("A", 0, etype=EventType.ERROR))
        await adapter.emit("B", ev("B", 0))
        await adapter.drain()
        stop.set()
        await consumer
        return consumer

    consumer = asyncio.run(run())
    assert consumer.exception() is None
    assert m.latest("B") is not None and m.latest("A") is not None
