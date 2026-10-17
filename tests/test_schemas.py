import pytest
from pydantic import ValidationError

from gepa_financial_agent.schemas import (
    DatasetExample, DriftStatus, DriftType, EventType, GroundTruth,
    SupervisorDecision, SupervisorInput, TrajectoryEvent, split_evidence_ids,
)


def ev(i, step=None, etype=EventType.AGENT_MESSAGE, **kw):
    return TrajectoryEvent(event_id=f"e{i}", task_id="t1", agent_id="a1",
                           step=step if step is not None else i, event_type=etype, **kw)


def make_input(n=3):
    return SupervisorInput(task_id="t1", agent_id="a1", swarm_objective="o", agent_directive="d",
                           trajectory=[ev(i) for i in range(n)], observation_step=n - 1)


def test_event_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TrajectoryEvent(event_id="e", task_id="t", agent_id="a", step=0,
                        event_type="tool_call", bogus=1)


def test_input_rejects_duplicate_event_ids():
    with pytest.raises(ValidationError):
        SupervisorInput(task_id="t1", agent_id="a1", swarm_objective="o", agent_directive="d",
                        trajectory=[ev(0), ev(0, step=1)])


def test_split_evidence_ids():
    inp = make_input()
    valid, invalid = split_evidence_ids(["e1", "e9", "e1", "e2"], inp)
    assert valid == ["e1", "e2"] and invalid == ["e9"]


def test_decision_normalizes_drift_type():
    d = SupervisorDecision(agent_id="a1", task_id="t1", status=DriftStatus.DRIFTING,
                           confidence=0.9, explanation="x", evidence_event_ids=["e1"])
    assert d.drift_type == DriftType.OTHER  # DRIFTING without a type defaults to OTHER
    d2 = SupervisorDecision(agent_id="a1", task_id="t1", status=DriftStatus.ON_TRACK,
                            confidence=0.5, drift_type="GOAL_DEVIATION", explanation="x")
    assert d2.drift_type is None  # non-drifting decisions carry no drift type


def test_decision_confidence_bounds():
    with pytest.raises(ValidationError):
        SupervisorDecision(agent_id="a", task_id="t", status="ON_TRACK", confidence=1.5, explanation="x")


def test_dataset_example_rejects_evidence_not_in_trajectory():
    inp = make_input()
    with pytest.raises(ValidationError):
        DatasetExample(example_id="x", scenario_id="s", observation_step=2, supervisor_input=inp,
                       ground_truth=GroundTruth(status="DRIFTING", drift_type="OTHER",
                                                evidence_event_ids=["nope"]))


def test_dataset_example_rejects_observation_step_before_last_event():
    inp = make_input(3)  # last step is 2
    with pytest.raises(ValidationError):
        DatasetExample(example_id="x", scenario_id="s", observation_step=1, supervisor_input=inp,
                       ground_truth=GroundTruth(status="ON_TRACK"))


def test_dataset_example_roundtrip_json():
    inp = make_input()
    ex = DatasetExample(example_id="x", scenario_id="s", observation_step=2, supervisor_input=inp,
                        ground_truth=GroundTruth(status="ON_TRACK"))
    assert DatasetExample.model_validate_json(ex.model_dump_json()) == ex
