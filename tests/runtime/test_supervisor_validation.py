import json

import pytest

from pitstop.collector import TrajectoryCollector
from pitstop.schemas import EventType, Status, DriftType
from pitstop.supervisor.policy import AlertPolicy
from pitstop.supervisor.signals import compute_signals
from pitstop.supervisor.supervisor import Supervisor, parse_and_validate, render_user_message
from test_schemas import make_input


@pytest.fixture
def inp():
    c = TrajectoryCollector()
    c.emit("a1", 0, EventType.PLAN, "plan", {"proposed_action": "research", "proposed_query": "NVDA DC revenue"})
    c.emit("a1", 0, EventType.TOOL_CALL, "call", {"query": "NVDA data center revenue FY2025"})
    c.emit("a1", 1, EventType.TOOL_CALL, "call", {"query": "NVDA data center revenue FY2025 growth"})
    return make_input(c)


def dec(**kw):
    base = {"agent_id": "a1", "status": "DRIFTING", "confidence": 0.9, "drift_type": "goal_deviation",
            "evidence_event_ids": ["a1-e001"], "explanation": "x"}
    base.update(kw)
    return json.dumps(base)


def test_valid_decision(inp):
    vd = parse_and_validate(dec(), inp)
    assert vd.valid_json and vd.decision.status == Status.DRIFTING and not vd.issues


def test_invented_evidence_dropped(inp):
    vd = parse_and_validate(dec(evidence_event_ids=["a1-e001", "a1-e042"]), inp)
    assert vd.decision.evidence_event_ids == ["a1-e001"]
    assert vd.dropped_evidence_ids == ["a1-e042"]


def test_short_evidence_ids_expanded_only_when_visible(inp):
    vd = parse_and_validate(dec(evidence_event_ids=["e001", "e042", "x001"]), inp)
    assert vd.decision.status == Status.DRIFTING and not vd.downgraded
    assert vd.decision.evidence_event_ids == ["a1-e001"]
    assert vd.dropped_evidence_ids == ["e042", "x001"]
    assert "expanded_short_evidence_ids:['e001']" in vd.issues


def test_drift_without_valid_evidence_is_downgraded(inp):
    vd = parse_and_validate(dec(evidence_event_ids=["made-up"]), inp)
    assert vd.decision.status == Status.UNCERTAIN and vd.downgraded
    assert "drift_without_valid_evidence" in vd.issues


def test_invalid_json(inp):
    vd = parse_and_validate("I think it's drifting.", inp)
    assert not vd.valid_json and vd.decision.status == Status.UNCERTAIN and vd.decision.confidence == 0


def test_reasoning_and_fences_are_tolerated(inp):
    vd = parse_and_validate("<think>hmm {not json}</think>\n```json\n" + dec() + "\n```", inp)
    assert vd.valid_json and vd.decision.status == Status.DRIFTING


def test_normalization(inp):
    vd = parse_and_validate(dec(status="on_track", drift_type="Goal Deviation", confidence=1.7), inp)
    assert vd.decision.status == Status.ON_TRACK
    assert vd.decision.drift_type == DriftType.NONE  # ON_TRACK forces none
    assert vd.decision.confidence == 1.0


def test_signals_detect_near_duplicates(inp):
    s = compute_signals(inp)
    assert s["research_calls_made"] == 2 and s["near_duplicate_query_pairs"] == 1


def test_user_message_contains_contract_and_ids(inp):
    msg = render_user_message(inp)
    assert "a1-e001" in msg and '"evidence_event_ids"' in msg


@pytest.mark.asyncio
async def test_supervisor_retries_on_bad_json(inp, FakeLLMClass, tmp_path):
    replies = iter(["not json", dec()])
    p = tmp_path / "p.txt"; p.write_text("policy")
    sup = Supervisor(FakeLLMClass(lambda s, u: next(replies)), p)
    vd = await sup.evaluate(inp)
    assert vd.valid_json and "needed_json_retry" in vd.issues and vd.prompt_version.startswith("p@")


def test_alert_policy_debounces(inp):
    pol = AlertPolicy()
    moderate = parse_and_validate(dec(confidence=0.6), inp)
    assert pol.update("a1", 0, moderate)[1] != "INTERVENTION POINT"
    assert pol.update("a1", 1, moderate)[1] == "INTERVENTION POINT"
    pol2 = AlertPolicy()
    assert pol2.update("a1", 0, parse_and_validate(dec(confidence=0.9), inp))[1] == "INTERVENTION POINT"
