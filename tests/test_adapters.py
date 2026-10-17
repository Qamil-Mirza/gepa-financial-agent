import asyncio
import json

from gepa_financial_agent.adapters.base import EventStreamAdapter
from gepa_financial_agent.adapters.mock_agent import ReplayAgent, pick_full_example
from gepa_financial_agent.crusoe_client import MockModelClient
from gepa_financial_agent.dataset import load_jsonl
from gepa_financial_agent.policy_store import Policy
from gepa_financial_agent.session_manager import SupervisorManager, TriggerConfig

FIX = load_jsonl("data/fixtures/dev_examples.jsonl").examples


def responder(messages):
    user = messages[1]["content"]
    drifting = "Ryzen" in user or "GeForce" in user
    return json.dumps({"status": "DRIFTING" if drifting else "ON_TRACK", "confidence": 0.8,
                       "drift_type": "GOAL_DEVIATION" if drifting else None, "evidence_event_ids": [],
                       "explanation": "stub", "recommended_action": "NONE"})


def test_pick_full_example():
    ex = pick_full_example(FIX, "goal_deviation", "AMD")
    assert ex.observation_step == max(e.observation_step for e in FIX if e.scenario_id == ex.scenario_id)
    assert "kind:goal_deviation" in ex.tags and "company:AMD" in ex.tags


def test_three_agents_stream_through_adapter():
    client = MockModelClient(responder)
    manager = SupervisorManager(client, policy=Policy.from_text("p"), max_concurrent_inferences=2,
                                trigger=TriggerConfig(every_n_events=3))
    adapter = EventStreamAdapter(manager, eval_interval_s=0.02)
    seen = []
    adapter.on_decision(seen.append)
    agents = [ReplayAgent.from_example(pick_full_example(FIX, "on_track", "NVDA"), agent_id="A"),
              ReplayAgent.from_example(pick_full_example(FIX, "goal_deviation", "AMD"), agent_id="B"),
              ReplayAgent.from_example(pick_full_example(FIX, "exploratory_legit", "NVDA"), agent_id="C")]

    async def run():
        for a in agents:
            a.register(manager)
        stop = asyncio.Event()
        consumer = asyncio.create_task(adapter.run(stop))
        await asyncio.gather(*(a.run(adapter) for a in agents))
        await adapter.drain()
        stop.set()
        await consumer

    asyncio.run(run())
    assert {r.agent_id for r in seen} == {"A", "B", "C"} and seen == adapter.decisions_seen
    assert manager.latest("B").decision.status.value == "DRIFTING"
    assert manager.latest("A").decision.status.value == "ON_TRACK"
    assert manager.latest("C").decision.status.value == "ON_TRACK"
    assert adapter.queue.empty() and manager.stats()["pending"] == 0
    for a in agents:  # every event reached the right session, none leaked
        assert [e.event_id for e in manager.sessions[a.agent_id].events] == [e.event_id for e in a.events]


def _load_demo():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("demo_supervision", root / "scripts" / "demo_supervision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_shows_comparison_of_current_policy_run_not_latest_dir(tmp_path):
    from gepa_financial_agent.policy_store import PolicyStore
    demo = _load_demo()
    real_run, mock_run = tmp_path / "runs" / "20261017_120000", tmp_path / "runs" / "mock_check"
    for d, label in ((real_run, "REAL"), (mock_run, "MOCK")):
        d.mkdir(parents=True)
        (d / "comparison.txt").write_text(label)
    store = PolicyStore(tmp_path / "policies")
    store.save_version("real policy", "gepa_real", {"mocked": False, "run_dir": str(real_run)})
    store.save_version("mock policy", "gepa_mock", {"mocked": True, "run_dir": str(mock_run)})
    assert demo.current_policy_comparison(store) is None  # nothing promoted yet
    store.promote("gepa_real")
    assert demo.current_policy_comparison(store) == real_run / "comparison.txt"
    store.promote("gepa_mock")
    assert demo.current_policy_comparison(store) is None  # never show a mock run's table
