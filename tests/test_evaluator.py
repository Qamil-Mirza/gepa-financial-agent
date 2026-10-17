import json

from gepa_financial_agent.crusoe_client import MockModelClient
from gepa_financial_agent.dataset import load_jsonl
from gepa_financial_agent.evaluator import (
    compute_metrics, format_comparison, heuristic_baseline, run_evaluation, run_heuristic_evaluation, score_prediction,
)
from gepa_financial_agent.policy_store import Policy
from gepa_financial_agent.schemas import DriftStatus, EvaluationRecord, SupervisorDecision

FIXTURES = "data/fixtures/dev_examples.jsonl"


def rec_for(example, status=None, drift_type=None, evidence=(), invalid=(), error=None):
    inp = example.supervisor_input
    decision = None if error else SupervisorDecision(agent_id=inp.agent_id, task_id=inp.task_id, status=status,
                                                     confidence=0.8, drift_type=drift_type,
                                                     evidence_event_ids=list(evidence), explanation="t")
    return EvaluationRecord(agent_id=inp.agent_id, task_id=inp.task_id, observation_step=example.observation_step,
                            policy_version="v", model="m", decision=decision, error=error,
                            invalid_evidence_ids=list(invalid), latency_ms=10, prompt_tokens=100, completion_tokens=20,
                            example_id=example.example_id)


def pick(examples, kind, status):
    """Longest prefix (largest observation_step) of the given kind with the given label."""
    return max((e for e in examples if f"kind:{kind}" in e.tags and e.ground_truth.status == status),
               key=lambda e: e.observation_step)


def test_scores():
    exs = load_jsonl(FIXTURES).examples
    drift = pick(exs, "goal_deviation", DriftStatus.DRIFTING)
    healthy = pick(exs, "exploratory_legit", DriftStatus.ON_TRACK)
    gt = drift.ground_truth
    perfect, info = score_prediction(rec_for(drift, "DRIFTING", gt.drift_type, gt.evidence_event_ids[:2]), drift)
    assert perfect == 1.0 and "correct status" in info["feedback"]
    assert score_prediction(rec_for(healthy, "ON_TRACK"), healthy)[0] == 0.6
    fa, info = score_prediction(rec_for(healthy, "DRIFTING", "OTHER", [healthy.supervisor_input.trajectory[0].event_id]), healthy)
    assert fa < 0 and "FALSE ALARM" in info["feedback"]
    assert score_prediction(rec_for(drift, "UNCERTAIN"), drift)[0] <= 0.15
    assert score_prediction(rec_for(healthy, "UNCERTAIN"), healthy)[0] <= 0.2
    fab, info = score_prediction(rec_for(drift, "DRIFTING", gt.drift_type, gt.evidence_event_ids[:1], invalid=["ghost"]), drift)
    assert fab == 0.9 and "FABRICATED" in info["feedback"]
    bad, info = score_prediction(rec_for(drift, error="invalid_json: nope"), drift)
    assert bad == 0.0 and "INVALID OUTPUT" in info["feedback"]
    assert score_prediction(rec_for(drift, "DRIFTING", gt.drift_type, []), drift)[0] == 0.7  # 0.6 + 0.2 - 0.1


def test_metrics_and_detection_delay():
    exs = [e for e in load_jsonl(FIXTURES).examples if "company:NVDA" in e.tags and "variant:0" in e.tags]
    records, scores = [], []
    for e in exs:
        gt = e.ground_truth
        if "kind:goal_deviation" in e.tags and gt.status != DriftStatus.ON_TRACK:
            r = rec_for(e, "DRIFTING", "GOAL_DEVIATION", gt.evidence_event_ids[:1])   # detects at onset (delay 0)
        elif "kind:exploratory_legit" in e.tags:
            r = rec_for(e, "DRIFTING", "OTHER", [e.supervisor_input.trajectory[0].event_id])  # false alarm
        elif "kind:on_track" in e.tags and e.observation_step == 2:
            r = rec_for(e, error="invalid_json")
        else:
            r = rec_for(e, gt.status.value, gt.drift_type, gt.evidence_event_ids)
        records.append(r)
        scores.append(score_prediction(r, e)[0])
    m = compute_metrics(records, exs, scores)
    assert m.n == len(exs) and 0 < m.macro_f1 < 1 and m.invalid_output_rate > 0
    assert m.false_positive_rate > 0 and m.false_negative_rate == 0
    assert m.per_class["DRIFTING"]["recall"] == 1.0
    # goal_deviation flagged at onset (delay 0); the other four drifting kinds flagged at onset+1 (delay 1)
    assert m.mean_detection_delay == 0.8 and m.undetected_drift_scenarios == 0
    assert m.total_prompt_tokens == 100 * len(exs)
    d = m.to_dict()
    assert set(d["confusion"]["ON_TRACK"]) >= {"ON_TRACK", "DRIFTING", "UNCERTAIN", "INVALID"}
    json.dumps(d)


def test_run_evaluation_persists(tmp_path):
    exs = load_jsonl(FIXTURES).examples[:6]
    run = run_evaluation(Policy.from_text("p", name="p"), exs, MockModelClient(), concurrency=3,
                         out_dir=tmp_path, label="unit")
    assert run.mocked is True and len(run.records) == 6
    lines = (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["example_id"] == exs[0].example_id
    meta = json.loads((tmp_path / "metrics.json").read_text())
    assert meta["label"] == "unit" and meta["mocked"] is True and "macro_f1" in meta["metrics"]
    assert "unit" in format_comparison([run])


def test_heuristic_baseline():
    exs = load_jsonl(FIXTURES).examples
    loop = pick(exs, "unproductive_loop", DriftStatus.DRIFTING)
    ok = pick(exs, "on_track", DriftStatus.ON_TRACK)
    d = heuristic_baseline(loop.supervisor_input)
    assert d.status == DriftStatus.DRIFTING and d.drift_type.value == "UNPRODUCTIVE_LOOP" and d.evidence_event_ids
    assert heuristic_baseline(ok.supervisor_input).status == DriftStatus.ON_TRACK
    run = run_heuristic_evaluation(exs[:10])
    assert run.model == "heuristic-rules" and run.mocked is False and len(run.records) == 10
