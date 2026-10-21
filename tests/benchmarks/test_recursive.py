import json

from benchmarks.calibration import recursive
from benchmarks.calibration.adapter import load_hard_cases, make_pitstop_evaluator
from benchmarks.calibration.recursive import (HindsightJudge, JudgeVerdict, LoopState, gate_metrics, mine_run,
                                              parse_verdict, promotion_decision, rows_to_cases, select_replay)
from pitstop.correct.client import MockModelClient
from pitstop.correct.gepa import strip_code_fences


def _case(label="drift"):
    return next(c for c in load_hard_cases("train") if c.label == label)


def _verdict_json(**kw):
    base = {"drifted": False, "onset_step": None, "drift_type": "none", "evidence_event_ids": [],
            "confidence": 0.9, "rationale": "stub"}
    return json.dumps({**base, **kw})


def _record(inp, step, status="ON_TRACK", confidence=0.9, phase="pre_action"):
    return {"agent_id": inp.agent_id, "step": step, "phase": phase, "input": inp.model_dump(mode="json"),
            "result": {"decision": {"status": status, "confidence": confidence}}}


# ---------------- judge ----------------

def test_parse_verdict_validates_evidence_and_onset():
    inp = _case().input
    last = inp.trajectory[-1]
    v = parse_verdict(_verdict_json(drifted=True, onset_step=last.step, drift_type="goal_deviation",
                                    evidence_event_ids=[last.event_id.split("-")[-1]]), inp)  # short id is expanded
    assert v.valid and v.evidence_event_ids == [last.event_id] and v.onset_step == last.step

    v = parse_verdict(_verdict_json(drifted=True, onset_step=1, drift_type="goal_deviation",
                                    evidence_event_ids=["made-up-e999"]), inp)
    assert not v.valid and "drift_without_valid_evidence" in v.issues

    v = parse_verdict(_verdict_json(drifted=True, onset_step=99, drift_type="goal_deviation",
                                    evidence_event_ids=[last.event_id]), inp)
    assert not v.valid and any(i.startswith("onset_outside_run") for i in v.issues)

    # no onset given: the earliest cited event's step; unknown type on a drift verdict becomes "other"
    first = inp.trajectory[0]
    v = parse_verdict(_verdict_json(drifted=True, drift_type="none", evidence_event_ids=[last.event_id, first.event_id]), inp)
    assert v.valid and v.onset_step == first.step and v.drift_type == "other"

    v = parse_verdict(_verdict_json(drifted=False, onset_step=3, drift_type="goal_deviation"), inp)
    assert v.valid and v.onset_step is None and v.drift_type == "none"
    assert parse_verdict("not json", inp).issues == ["invalid_json"]


def test_judge_sees_the_whole_trajectory_but_not_the_supervisor_contract():
    seen = []

    def responder(messages):
        seen.append(messages)
        return _verdict_json()

    inp = _case().input
    v = HindsightJudge(MockModelClient(responder)).judge(inp)
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert v.valid and system == recursive.JUDGE_SYSTEM
    assert all(e.event_id in user for e in inp.trajectory)
    assert "## Output" not in user and "the run has finished" in user


def test_judge_retries_once_on_invalid_json():
    replies = iter(["oops", _verdict_json()])
    client = MockModelClient(lambda m: next(replies))
    assert HindsightJudge(client).judge(_case().input).valid


# ---------------- mining ----------------

def test_mine_run_labels_from_onset_and_keeps_only_disagreements():
    inp = _case().input
    ev = inp.trajectory[-1].event_id
    records = [_record(inp, 0, "DRIFTING", 0.9), _record(inp, 1), _record(inp, 2), _record(inp, 3, "DRIFTING", 0.8),
               _record(inp, 3, "DRIFTING", 0.6, phase="final")]
    verdict = JudgeVerdict(valid=True, drifted=True, onset_step=2, drift_type="goal_deviation",
                           evidence_event_ids=[ev], confidence=0.9, rationale="went off topic")
    rows, summary = mine_run("run1", records, {inp.agent_id: verdict}, judge_model="judge")
    cats = {r["id"]: r["category"] for r in rows}
    assert cats == {f"run1.{inp.agent_id}.s0.pre_action": "false_alarm",
                    f"run1.{inp.agent_id}.s2.pre_action": "late_catch",
                    f"run1.{inp.agent_id}.s3.final": "late_catch"}  # DRIFTING below 0.75 does not alert
    late = next(r for r in rows if r["id"].endswith("s2.pre_action"))
    assert late["label"] == "drift" and late["expect_type"] == "goal_deviation" and "late by 1" in late["why"]
    s = summary[inp.agent_id]
    assert s["first_alert_step"] == 3 and s["agree"] == 2 and s["false_alarm"] == 1


def test_mine_run_missed_drift_healthy_false_alarm_and_skips():
    inp = _case().input
    ev = inp.trajectory[-1].event_id
    drift = JudgeVerdict(valid=True, drifted=True, onset_step=0, drift_type="unproductive_loop",
                         evidence_event_ids=[ev], confidence=0.9)
    rows, _ = mine_run("r", [_record(inp, 0), _record(inp, 1)], {inp.agent_id: drift})
    assert [r["category"] for r in rows] == ["missed_drift", "missed_drift"] and "never alerted" in rows[0]["why"]

    healthy = JudgeVerdict(valid=True, drifted=False, confidence=0.95, rationale="all on directive")
    rows, _ = mine_run("r", [_record(inp, 0, "DRIFTING", 0.9)], {inp.agent_id: healthy})
    assert rows[0]["label"] == "ok" and rows[0]["expect_type"] == "none" and "did not drift" in rows[0]["why"]

    unsure = JudgeVerdict(valid=True, drifted=False, confidence=0.4)
    rows, summary = mine_run("r", [_record(inp, 0, "DRIFTING", 0.9)], {inp.agent_id: unsure})
    assert rows == [] and "confidence" in summary[inp.agent_id]["skipped"]


def test_mined_rows_are_scored_by_the_unchanged_pitstop_evaluator():
    inp = _case().input
    healthy = JudgeVerdict(valid=True, drifted=False, confidence=0.95, rationale="fine")
    rows, _ = mine_run("r", [_record(inp, 0, "DRIFTING", 0.9)], {inp.agent_id: healthy})
    case = rows_to_cases(rows)[0]
    on_track = json.dumps({"agent_id": inp.agent_id, "status": "ON_TRACK", "confidence": 0.9, "drift_type": "none",
                           "evidence_event_ids": [], "explanation": "ok"})
    score, info = make_pitstop_evaluator(MockModelClient(lambda m: on_track))("prompt", case)
    assert score == 1.0 and info["why"] == case.why and "r." not in info["input"][:20]


def test_select_replay_is_balanced_deterministic_and_bounded():
    cases = load_hard_cases("train")
    a, b = select_replay(cases, 12, seed=1), select_replay(cases, 12, seed=1)
    assert [c.id for c in a] == [c.id for c in b] and len(a) == 12
    assert sum(c.label == "ok" for c in a) == 6 and len({c.category for c in a}) >= 5
    assert len(select_replay(cases, 1000)) == len(cases)


# ---------------- gate and state ----------------

def test_gate_metrics_and_promotion_rule():
    cases = load_hard_cases("test")[:4]
    always = lambda fire: (lambda prompt, c: (1.0, {"fired": fire}))
    m_fire, m_quiet = gate_metrics("p", cases, always(True), trials=2), gate_metrics("p", cases, always(False), trials=2)
    n_drift = sum(c.label == "drift" for c in cases) * 2
    assert m_fire["n"] == 8 and m_fire["drift_recall"] == 1.0 and m_fire["false_alarms"] == 8 - n_drift
    assert m_quiet["ok_specificity"] == 1.0 and m_quiet["missed"] == n_drift

    cur = {"accuracy": 0.7, "false_alarms": 3}
    assert promotion_decision(cur, {"accuracy": 0.8, "false_alarms": 3})[0]
    assert not promotion_decision(cur, {"accuracy": 0.7, "false_alarms": 0})[0]   # ties keep the current prompt
    assert not promotion_decision(cur, {"accuracy": 0.9, "false_alarms": 4})[0]   # no trading in false alarms


def test_loop_state_roundtrip_and_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(recursive, "STATE_PATH", tmp_path / "state" / "state.json")
    monkeypatch.setattr(recursive, "PROMOTED_DIR", tmp_path / "state")  # never write into the source tree
    st = LoopState.load()
    assert st.current == recursive.SEED_PROMPT_REL and st.current_path().exists()
    st.record({"round": 0, "promoted": False})
    st.record({"round": 0, "promoted": True})  # re-running a round replaces its entry
    path = st.promote("new prompt", 0)
    st.save()
    again = LoopState.load()
    assert len(again.rounds) == 1 and again.rounds[0]["promoted"] and again.current == path
    assert again.current_path().read_text() == "new prompt\n"


def test_strip_code_fences():
    assert strip_code_fences("```\nbody\n```") == "body"
    assert strip_code_fences("```text\nbody line\n```\n") == "body line"
    assert strip_code_fences("body\n```") == "body"
    assert strip_code_fences("plain") == "plain"


# ---------------- whole round, offline ----------------

def test_round_stages_end_to_end_with_mocks(tmp_path, monkeypatch):
    """judge -> mine -> optimize -> gate on a fabricated logged run, with mock models. Proves the plumbing only."""
    import importlib
    loop = importlib.import_module("benchmarks.calibration.loop")
    monkeypatch.setattr(loop, "LOOP_DIR", tmp_path / "loop")
    monkeypatch.setattr(recursive, "STATE_PATH", tmp_path / "state" / "state.json")
    monkeypatch.setattr(recursive, "PROMOTED_DIR", tmp_path / "state")  # never write into the source tree
    monkeypatch.setenv("CRUSOE_API_KEY", "test-key")

    drift, ok = _case("drift"), _case("ok")
    run_dir = tmp_path / "loop" / "round_00" / "runs" / "fixture-run"
    run_dir.mkdir(parents=True)
    # the live supervisor alerted on the healthy agent and stayed quiet on the drifting one
    recs = [_record(drift.input, drift.input.trajectory[-1].step), _record(ok.input, 0, "DRIFTING", 0.9)]
    recs[1]["agent_id"] = recs[1]["input"]["agent_id"] = "agent_ok"
    for e in recs[1]["input"]["trajectory"]:
        e["agent_id"] = "agent_ok"
    (run_dir / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (run_dir / "meta.json").write_text(json.dumps({"ground_truth": {}}))

    last = drift.input.trajectory[-1]

    def judge_or_supervisor(messages):
        user = messages[-1]["content"]
        if messages[0]["content"] == recursive.JUDGE_SYSTEM:
            if "## Monitored agent: agent_ok" not in user:
                return _verdict_json(drifted=True, onset_step=last.step, drift_type=drift.expect_type,
                                     evidence_event_ids=[last.event_id], rationale="drifted")
            return _verdict_json(rationale="healthy")
        agent = user.split("## Monitored agent: ")[1].split("\n")[0]
        return json.dumps({"agent_id": agent, "status": "ON_TRACK", "confidence": 0.9, "drift_type": "none",
                           "evidence_event_ids": [], "explanation": "stub"})

    mock = MockModelClient(judge_or_supervisor)
    monkeypatch.setattr(loop, "client", lambda s, model, max_concurrent=4: mock)
    monkeypatch.setattr(loop, "CrusoeReflectionLM", lambda c: (lambda prompt: "```\nBe careful. Flag real drift.\n```"))

    common = ["--round", "0", "--workers", "1"]
    assert loop.main(["judge", *common]) == 0
    assert loop.main(["mine", *common]) == 0
    mined = [json.loads(l) for l in (tmp_path / "loop" / "round_00" / "mined.jsonl").read_text().splitlines()]
    assert sorted(r["category"] for r in mined) == ["false_alarm", "missed_drift"]
    assert loop.main(["optimize", *common, "--max-metric-calls", "12", "--replay", "2", "--minibatch", "2"]) == 0
    assert (tmp_path / "loop" / "round_00" / "candidate.txt").exists()
    assert loop.main(["gate", *common, "--trials", "1"]) == 0
    state = json.loads((tmp_path / "state" / "state.json").read_text())
    assert state["rounds"][0]["round"] == 0 and "live_vs_judge" in state["rounds"][0]
