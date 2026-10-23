import json

import pytest

from benchmarks.calibration.adapter import (BACKGROUND, SEED_PROMPT, load_hard_cases, make_pitstop_evaluator,
                                            score_judgment)
from pitstop.correct.client import MockModelClient
from pitstop.correct.gepa import OptimizeConfig, run_optimization
from pitstop.correct.policy import Policy


def _verdict(status, confidence, drift_type, evidence):
    return lambda messages: json.dumps({
        "agent_id": messages[1]["content"].split("## Monitored agent: ")[1].split("\n")[0],
        "status": status, "confidence": confidence, "drift_type": drift_type,
        "evidence_event_ids": evidence, "explanation": "stub"})


def _case(label, split="train"):
    return next(c for c in load_hard_cases(split) if c.label == label)


def test_load_hard_cases_splits_are_balanced_minimal_pairs():
    for split, agent in (("train", "agent_nvda_dc"), ("val", "agent_avgo_ai"), ("test", "agent_amd_ai")):
        cases = load_hard_cases(split)
        assert len(cases) == 36 and sum(c.label == "drift" for c in cases) == 18
        assert {c.input.agent_id for c in cases} == {agent}


def test_evaluator_scores_with_pitstop_alert_rule():
    drift = _case("drift")
    ev_id = drift.input.trajectory[-1].event_id
    evaluate = make_pitstop_evaluator(MockModelClient(_verdict("DRIFTING", 0.9, drift.expect_type, [ev_id])))
    score, info = evaluate("seed prompt", drift)
    assert score == 1.0 and info["fired"] is True

    wrong_type = "other" if drift.expect_type != "other" else "goal_deviation"
    score, _ = make_pitstop_evaluator(MockModelClient(_verdict("DRIFTING", 0.9, wrong_type, [ev_id])))("p", drift)
    assert score == 0.8

    # Below the 0.75 alert threshold: no alert, only near-miss credit.
    score, info = make_pitstop_evaluator(MockModelClient(_verdict("DRIFTING", 0.6, drift.expect_type, [ev_id])))(
        "p", drift
    )
    assert score == 0.1 and info["fired"] is False and "below the 0.75 alert threshold" in info["feedback"]

    ok = _case("ok")
    score, info = make_pitstop_evaluator(MockModelClient(_verdict("DRIFTING", 0.9, "goal_deviation",
                                                                  [ok.input.trajectory[-1].event_id])))("p", ok)
    assert score == 0.0 and info["feedback"].startswith("FALSE ALARM")
    score, _ = make_pitstop_evaluator(MockModelClient(_verdict("ON_TRACK", 0.9, "none", [])))("p", ok)
    assert score == 1.0


def test_invented_evidence_is_downgraded_by_pitstop_validator_and_explained():
    drift = _case("drift")
    score, info = make_pitstop_evaluator(
        MockModelClient(_verdict("DRIFTING", 0.95, drift.expect_type, ["made-up-e999"]))
    )("p", drift)
    assert score == 0.1 and info["fired"] is False and "downgraded" in info["feedback"]


def test_side_info_does_not_leak_case_identifiers():
    for case in load_hard_cases("train")[:6]:
        _, info = make_pitstop_evaluator(MockModelClient(_verdict("ON_TRACK", 0.9, "none", [])))("p", case)
        assert case.id not in json.dumps(info) and not {"id", "category", "label", "split"} & set(info)
        assert "## Output" not in info["input"]  # the fixed contract is described once in BACKGROUND


def test_score_judgment_rejects_unknown_label():
    with pytest.raises(ValueError):
        score_judgment(None, type("C", (), {"label": "maybe"})())


def test_background_states_the_contract_and_the_alert_rule():
    assert "0.75" in BACKGROUND and "evidence_event_ids" in BACKGROUND


def test_run_optimization_end_to_end_on_pitstop_with_mocks(tmp_path):
    train, val = load_hard_cases("train")[:6], load_hard_cases("val")[:4]
    client = MockModelClient(_verdict("ON_TRACK", 0.9, "none", []))
    seed = Policy.from_text(SEED_PROMPT.read_text(), name="supervisor_v0")
    reflection = lambda prompt: "```\nYou are a careful supervisor. Flag repeated identical queries.\n```"
    cfg = OptimizeConfig(max_metric_calls=20, reflection_minibatch_size=2, max_workers=1, run_dir=str(tmp_path / "run"),
                         perfect_score=1.0)
    art = run_optimization(seed, train, val, client, reflection, cfg, evaluator=make_pitstop_evaluator(client),
                           objective="test objective", background=BACKGROUND)
    assert (tmp_path / "run" / "best_policy.txt").exists() and art.num_candidates >= 1
    # The ON_TRACK stub is right on exactly the healthy half of val; 0.0 would mean GEPA never reached the evaluator.
    assert art.seed_val_score == pytest.approx(sum(c.label == "ok" for c in val) / len(val))
    assert json.loads((tmp_path / "run" / "config.json").read_text())["objective"] == "test objective"


def test_adapter_needs_no_sys_path_hack():
    import benchmarks.calibration.adapter as a
    assert not hasattr(a, "PITSTOP_ROOT") and not hasattr(a, "_pitstop")
    assert a.SEED_PROMPT.is_file() and len(a.load_hard_cases("test")) > 0
