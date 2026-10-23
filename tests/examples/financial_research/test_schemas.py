import pytest
from pydantic import ValidationError

from finresearch.collector import TrajectoryCollector
from finresearch.schemas import Budget, EventType, ProposedAction, SupervisorInput


def make_input(collector, agent="a1", cutoff=None, **kw):
    cutoff = collector.last_seq(agent) if cutoff is None else cutoff
    return SupervisorInput(
        task_id="t", agent_id=agent, swarm_objective="obj", agent_directive="dir",
        trajectory=collector.history(agent, upto_seq=cutoff),
        remaining_budget=Budget(steps_used=0, steps_max=5, research_calls_used=0, research_calls_max=5),
        evaluation_step=0, observation_cutoff_seq=cutoff, **kw,
    )


def test_collector_ids_and_cutoff(tmp_path):
    c = TrajectoryCollector(tmp_path / "ev.jsonl")
    for i in range(4):
        c.emit("a1", 0, EventType.PROGRESS, f"e{i}")
    assert [e.event_id for e in c.history("a1", upto_seq=1)] == ["a1-e000", "a1-e001"]
    reloaded = TrajectoryCollector.load(tmp_path / "ev.jsonl")
    assert len(reloaded.history("a1")) == 4


def test_input_rejects_events_beyond_cutoff():
    c = TrajectoryCollector()
    for i in range(3):
        c.emit("a1", 0, EventType.PROGRESS, "x")
    with pytest.raises(ValidationError):
        SupervisorInput(
            task_id="t", agent_id="a1", swarm_objective="o", agent_directive="d",
            trajectory=c.history("a1"), observation_cutoff_seq=1, evaluation_step=0,
            remaining_budget=Budget(steps_used=0, steps_max=1, research_calls_used=0, research_calls_max=1),
        )


def test_input_rejects_other_agents_events():
    c = TrajectoryCollector()
    c.emit("a2", 0, EventType.PROGRESS, "x")
    with pytest.raises(ValidationError):
        SupervisorInput(
            task_id="t", agent_id="a1", swarm_objective="o", agent_directive="d",
            trajectory=c.history("a2"), observation_cutoff_seq=0, evaluation_step=0,
            remaining_budget=Budget(steps_used=0, steps_max=1, research_calls_used=0, research_calls_max=1),
        )


def test_proposed_action_must_reference_visible_plan():
    c = TrajectoryCollector()
    c.emit("a1", 0, EventType.PLAN, "plan")
    with pytest.raises(ValidationError):
        make_input(c, proposed_next_action=ProposedAction(action="research", query="q", plan_event_id="a1-e099"))
