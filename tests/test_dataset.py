import json

from gepa_financial_agent.dataset import (
    expand_prefixes, label_distribution, load_jsonl, split_by_scenario, truncate_example, write_jsonl,
)
from gepa_financial_agent.schemas import (
    DatasetExample, DriftStatus, EventType, GroundTruth, SupervisorInput, TrajectoryEvent,
)


def make_example(scenario, n_events=6, onset=None, status="ON_TRACK", drift_type=None, evidence=()):
    events = [TrajectoryEvent(event_id=f"{scenario}_e{i}", task_id="t", agent_id="a", step=i,
                              event_type=EventType.AGENT_MESSAGE, content=f"{scenario} msg {i}") for i in range(n_events)]
    inp = SupervisorInput(task_id="t", agent_id="a", swarm_objective="o", agent_directive="d", trajectory=events,
                          observation_step=n_events - 1)
    return DatasetExample(example_id=f"{scenario}_full", scenario_id=scenario, observation_step=n_events - 1,
                          supervisor_input=inp,
                          ground_truth=GroundTruth(status=status, drift_type=drift_type,
                                                   evidence_event_ids=list(evidence), drift_onset_step=onset))


def test_load_reports_invalid_lines(tmp_path):
    good = make_example("s1")
    p = tmp_path / "d.jsonl"
    bad_evidence = json.loads(good.model_dump_json())
    bad_evidence["example_id"] = "bad"
    bad_evidence["ground_truth"]["evidence_event_ids"] = ["ghost"]
    p.write_text(good.model_dump_json() + "\n" + "not json\n" + json.dumps(bad_evidence) + "\n\n")
    rep = load_jsonl(p)
    assert [e.example_id for e in rep.examples] == ["s1_full"]
    assert [i.line_number for i in rep.invalid] == [2, 3]
    assert "ghost" in rep.invalid[1].error


def test_write_then_load_roundtrip(tmp_path):
    exs = [make_example("s1"), make_example("s2")]
    p = tmp_path / "o.jsonl"
    write_jsonl(exs, p)
    assert load_jsonl(p).examples == exs


def test_split_keeps_scenarios_together_and_is_deterministic():
    exs = []
    for s in range(10):
        full = make_example(f"s{s}", n_events=6)
        exs.extend(expand_prefixes([full], cutoffs=[2, 4, 5]))
    a = split_by_scenario(exs, seed=1)
    b = split_by_scenario(exs, seed=1)
    assert [e.example_id for e in a.train] == [e.example_id for e in b.train]
    groups = [{e.scenario_id for e in part} for part in (a.train, a.val, a.test)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert len(a.train) + len(a.val) + len(a.test) == len(exs)
    assert all(len(g) > 0 for g in groups)


def test_split_hint_forces_test():
    exs = [make_example(f"s{s}") for s in range(5)]
    exs[0] = exs[0].model_copy(update={"split_hint": "test"})
    sp = split_by_scenario(exs, seed=0)
    assert any(e.scenario_id == "s0" for e in sp.test)


def test_truncate_relabels_prefix_before_onset():
    full = make_example("s1", n_events=8, onset=5, status="DRIFTING", drift_type="GOAL_DEVIATION",
                        evidence=["s1_e5", "s1_e6"])
    before = truncate_example(full, 4)
    at = truncate_example(full, 5)
    after = truncate_example(full, 6)
    assert before.ground_truth.status == DriftStatus.ON_TRACK and before.ground_truth.evidence_event_ids == []
    assert at.ground_truth.status == DriftStatus.UNCERTAIN and at.ground_truth.evidence_event_ids == ["s1_e5"]
    assert after.ground_truth.status == DriftStatus.DRIFTING and after.ground_truth.evidence_event_ids == ["s1_e5", "s1_e6"]
    assert max(e.step for e in after.supervisor_input.trajectory) == 6 and after.observation_step == 6
    assert after.example_id == "s1_full@6" and after.scenario_id == "s1"


def test_label_distribution():
    exs = [make_example("a"), make_example("b", status="DRIFTING", drift_type="OTHER")]
    assert label_distribution(exs) == {"ON_TRACK": 1, "DRIFTING": 1, "UNCERTAIN": 0}


def test_fixture_split_never_puts_the_same_prompt_in_two_splits():
    from pathlib import Path
    from gepa_financial_agent.supervisor import render_input
    exs = load_jsonl(Path(__file__).resolve().parents[1] / "data/fixtures/dev_examples.jsonl").examples
    sp = split_by_scenario(exs, seed=0)
    prompts = [{render_input(e.supervisor_input) for e in part} for part in (sp.train, sp.val, sp.test)]
    assert not (prompts[0] & prompts[1]) and not (prompts[0] & prompts[2]) and not (prompts[1] & prompts[2])
    assert all(prompts)


def test_split_stratifies_by_kind_tag_so_every_kind_is_in_every_split():
    exs = []
    for kind in ("on_track", "goal_deviation", "exploratory_legit"):
        for s in range(5):
            ex = make_example(f"{kind}{s}", n_events=6)
            exs.append(ex.model_copy(update={"tags": [f"kind:{kind}"]}))
    sp = split_by_scenario(exs, seed=0)
    kinds = lambda part: {t for e in part for t in e.tags}
    assert kinds(sp.train) == kinds(sp.val) == kinds(sp.test) == kinds(exs)


def test_split_without_tags_stratifies_by_full_trajectory_label():
    exs = []
    for label, dtype in (("ON_TRACK", None), ("DRIFTING", "GOAL_DEVIATION"), ("DRIFTING", "UNPRODUCTIVE_LOOP")):
        for s in range(5):
            sid = f"{label}_{dtype}_{s}"
            full = make_example(sid, n_events=6, onset=3 if dtype else None, status=label, drift_type=dtype,
                                evidence=[f"{sid}_e4"] if dtype else ())
            exs.extend(expand_prefixes([full], cutoffs=[2, 5]))
    sp = split_by_scenario(exs, seed=0)
    for part in (sp.train, sp.val, sp.test):
        full_labels = {(e.ground_truth.status.value, e.ground_truth.drift_type) for e in part if e.observation_step == 5}
        assert len(full_labels) == 3, full_labels


def test_summary_lists_invalid_lines_with_reasons(tmp_path):
    good = make_example("s1")
    p = tmp_path / "d.jsonl"
    p.write_text(good.model_dump_json() + "\nnot json\n")
    s = load_jsonl(p).summary()
    assert "1 valid, 1 invalid" in s and "line 2:" in s and "json" in s.lower()


def test_scripts_refuse_invalid_dataset_unless_allowed(tmp_path):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("evaluate_supervisor", root / "scripts" / "evaluate_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "d.jsonl"
    good = (root / "data/fixtures/dev_examples.jsonl").read_text()
    p.write_text(good + "not json\n")
    base = ["--mock", "--dataset", str(p), "--split", "val", "--out", str(tmp_path / "out")]
    assert mod.main(base) != 0
    assert mod.main(base + ["--allow-invalid"]) == 0


def test_split_keeps_scenarios_with_identical_content_together():
    exs = []
    for s in range(12):
        full = make_example(f"s{s}", n_events=6)
        if s % 2:  # odd scenarios duplicate the previous one's content under a different id
            full = full.model_copy(update={"scenario_id": f"s{s}", "example_id": f"s{s}_full"})
        exs.append(full)
    # make pairs (0,1), (2,3), ... share content and every pair differ from the others
    for i, ex in enumerate(exs):
        pair = i // 2
        traj = [e.model_copy(update={"content": f"pair {pair} msg {e.step}"}) for e in ex.supervisor_input.trajectory]
        exs[i] = ex.model_copy(update={"supervisor_input": ex.supervisor_input.model_copy(update={"trajectory": traj})})
    sp = split_by_scenario(exs, seed=0)
    where = {e.scenario_id: name for name in ("train", "val", "test") for e in getattr(sp, name)}
    for pair in range(6):
        assert where[f"s{2 * pair}"] == where[f"s{2 * pair + 1}"]
