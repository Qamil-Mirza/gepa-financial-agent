"""A PitStop halt (budget or drift) raised by a metered LLM call must reach the caller, never be swallowed by
the agents' keep-the-swarm-alive handlers (and never trigger a retry sleep)."""
import asyncio
from types import SimpleNamespace

import pytest

from finresearch.agents import synthesizer as synth_mod
from finresearch.agents.synthesizer import ReportSynthesizer
from finresearch.agents.worker import ResearchWorker
from finresearch.collector import TrajectoryCollector
from finresearch.config import PROMPTS
from finresearch.demo.team import plan_team
from finresearch.schemas import AgentSpec
from pitstop.sdk.errors import BudgetExceeded, Halted

WORKER_PROMPTS = PROMPTS / "worker"


def _raise(exc):
    def handler(system, user):
        raise exc
    return handler


def _budget():
    return BudgetExceeded("global:all", 1.0, 1.0)


def _worker(FakeLLMClass, exc):
    return ResearchWorker(AgentSpec(agent_id="a", directive="Research X."), swarm_objective="o",
                          llm=FakeLLMClass(_raise(exc)), research=None, collector=TrajectoryCollector(),
                          prompts_dir=WORKER_PROMPTS, watched=False)


@pytest.mark.parametrize("exc", [_budget(), Halted("steer limit exceeded")])
def test_worker_propose_propagates_halt(FakeLLMClass, exc):
    w = _worker(FakeLLMClass, exc)
    with pytest.raises(type(exc)):
        asyncio.run(w.propose())
    assert w.state.consecutive_plan_failures == 0


def test_worker_extract_propagates_budget(FakeLLMClass):
    w = _worker(FakeLLMClass, _budget())
    res = SimpleNamespace(query="q", text="t", sources=[])
    with pytest.raises(BudgetExceeded):
        asyncio.run(w._extract(res, {}, "e0"))


def test_worker_finish_propagates_budget(FakeLLMClass):
    w = _worker(FakeLLMClass, _budget())
    w.state.findings = [{"event_id": "a-e001", "claim": "c"}]
    with pytest.raises(BudgetExceeded):
        asyncio.run(w.finish())


def test_synthesizer_propagates_budget_without_backoff(FakeLLMClass, monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(synth_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(synth_mod, "select_findings",
                        lambda events, decisions: ([{"agent_id": "a", "claim": "c", "event_id": "a-e001"}], []))
    llm = FakeLLMClass(_raise(_budget()))
    s = ReportSynthesizer(llm, WORKER_PROMPTS, attempts=5, backoff_s=5.0)
    with pytest.raises(BudgetExceeded):
        asyncio.run(s.synthesize(objective="o", agents=[{"agent_id": "a", "directive": "d", "final_report": ""}],
                                 events=[], decisions=[], interventions={}))
    assert slept == [] and len(llm.calls) == 1


def test_plan_team_propagates_budget(FakeLLMClass):
    with pytest.raises(BudgetExceeded):
        asyncio.run(plan_team("objective", FakeLLMClass(_raise(_budget())), WORKER_PROMPTS))
