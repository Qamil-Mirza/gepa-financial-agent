import asyncio
import json

from gepa_financial_agent.crusoe_client import CompletionResult, MockModelClient, ModelUnavailableError
from gepa_financial_agent.policy_store import Policy
from gepa_financial_agent.schemas import DriftStatus, DriftType, EventType, SupervisorInput, TrajectoryEvent
from gepa_financial_agent.supervisor import OUTPUT_CONTRACT, Supervisor, build_messages, extract_json, parse_decision


def ev(i, etype=EventType.AGENT_MESSAGE, content=None):
    return TrajectoryEvent(event_id=f"e{i}", task_id="t", agent_id="a", step=i, event_type=etype,
                           content=content or f"content {i}")


def make_input(n=5):
    return SupervisorInput(task_id="t", agent_id="a", swarm_objective="obj", agent_directive="dir",
                           constraints=["c1"], success_criteria=["s1"], trajectory=[ev(i) for i in range(n)],
                           observation_step=n - 1)


POLICY = Policy.from_text("You are a drift supervisor.", name="test")


def test_extract_json_from_fenced_prose():
    text = 'Sure! Here you go:\n```json\n{"status": "DRIFTING", "confidence": 0.8}\n```\nHope that helps.'
    assert extract_json(text) == {"status": "DRIFTING", "confidence": 0.8}


def test_parse_decision_filters_invalid_evidence_and_ignores_model_ids():
    raw = json.dumps({"agent_id": "WRONG", "task_id": "WRONG", "status": "DRIFTING", "confidence": 0.9,
                      "drift_type": "GOAL_DEVIATION", "evidence_event_ids": ["E2", "ghost", "e4"],
                      "explanation": "x", "recommended_action": "FLAG_FOR_REVIEW"})
    d, err, invalid = parse_decision(raw, make_input(), "v1")
    assert err is None and invalid == ["ghost"]
    assert d.agent_id == "a" and d.task_id == "t" and d.evidence_event_ids == ["e1", "e3"]
    assert d.drift_type == DriftType.GOAL_DEVIATION and d.policy_version == "v1" and d.observation_step == 4


def test_parse_decision_reports_schema_errors():
    d, err, _ = parse_decision('{"status": "MAYBE", "confidence": 0.5}', make_input(), "v1")
    assert d is None and "status" in err


def test_build_messages_bounds_context_and_contains_contract():
    inp = make_input(60)
    msgs = build_messages(POLICY.text, inp, max_events=40)
    assert msgs[0]["role"] == "system" and OUTPUT_CONTRACT.strip() in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "20 earlier events omitted" in user and "[E60]" in user and "[E1]" not in user and "[e59]" not in user
    assert "ground_truth" not in user and "dir" in user and "c1" in user


def test_supervisor_happy_path_with_mock():
    responder = lambda m: '```json\n' + json.dumps({"status": "DRIFTING", "confidence": 0.7, "drift_type": "UNPRODUCTIVE_LOOP",
                                                   "evidence_event_ids": ["e2", "nope"], "explanation": "loop",
                                                   "recommended_action": "FLAG_FOR_REVIEW"}) + '\n```'
    sup = Supervisor(MockModelClient(responder), POLICY)
    rec = sup.evaluate_sync(make_input(), example_id="x1")
    assert rec.decision.status == DriftStatus.DRIFTING and rec.invalid_evidence_ids == ["nope"]
    assert rec.mocked is True and rec.policy_version == POLICY.version and rec.example_id == "x1"
    assert rec.error is None and rec.prompt_tokens > 0


def test_supervisor_truncated_output_is_error_not_on_track():
    class Trunc(MockModelClient):
        def _result(self, messages, t0):
            r = CompletionResult(text="", model="mock", prompt_tokens=1, completion_tokens=1, cached_tokens=0,
                                 latency_ms=1.0, finish_reason="length", mocked=True)
            self.tracker.record(r)
            return r

    rec = Supervisor(Trunc(), POLICY).evaluate_sync(make_input())
    assert rec.decision is None and rec.error.startswith("truncated_output")


def test_supervisor_model_error_is_recorded_not_raised():
    class Boom(MockModelClient):
        def complete(self, *a, **k):
            raise ModelUnavailableError("down")

        async def acomplete(self, *a, **k):
            raise ModelUnavailableError("down")

    sup = Supervisor(Boom(), POLICY)
    assert sup.evaluate_sync(make_input()).error.startswith("model_error")
    assert asyncio.run(sup.evaluate(make_input())).error.startswith("model_error")


def test_prompt_shows_opaque_aliases_not_real_event_ids():
    inp = SupervisorInput(task_id="t", agent_id="a", swarm_objective="o", agent_directive="d",
                          trajectory=[TrajectoryEvent(event_id=f"nvda_goal_deviation_0_e{i}", task_id="t", agent_id="a",
                                                      step=i, event_type=EventType.AGENT_MESSAGE, content="c")
                                      for i in range(3)], observation_step=2)
    user = build_messages("p", inp)[1]["content"]
    assert "goal_deviation" not in user and "nvda_goal" not in user
    assert "[E1]" in user and "[E3]" in user


def test_parse_decision_maps_aliases_back_and_rejects_real_ids():
    raw = json.dumps({"status": "DRIFTING", "confidence": 0.9, "drift_type": "GOAL_DEVIATION",
                      "evidence_event_ids": ["E2", "e2", "E9", "e0"], "explanation": "x"})
    d, err, invalid = parse_decision(raw, make_input(), "v1")
    assert err is None
    assert d.evidence_event_ids == ["e1"]       # E2 is the 2nd event (real id e1); "e2" is the same alias, deduped
    assert invalid == ["E9", "e0"]              # out-of-range alias and a raw real id are both invalid


def test_no_fixture_prompt_contains_a_label_token():
    from pathlib import Path
    from gepa_financial_agent.dataset import load_jsonl
    exs = load_jsonl(Path(__file__).resolve().parents[1] / "data/fixtures/dev_examples.jsonl").examples
    for ex in exs:
        user = build_messages("p", ex.supervisor_input)[1]["content"]
        for tag in ex.tags:
            if tag.startswith("kind:"):
                assert tag[5:] not in user, (ex.example_id, tag)
        assert ex.scenario_id not in user, ex.example_id
        lowered = user.lower()
        for name in [*(x.value for x in DriftStatus), *(x.value for x in DriftType)]:
            assert name.lower() not in lowered, (ex.example_id, name)
