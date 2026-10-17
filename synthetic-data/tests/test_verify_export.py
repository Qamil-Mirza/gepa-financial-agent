from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from synth.export import DEFAULT_SPLITS, examples_for, export
from synth.schemas import DatasetExample, SteerSpec
from synth.tasks import TASKS
from synth.textmatch import keyword_hits, numbers, same_question
from synth.verify import verify
from tests.helpers import NONE, Builder, benign, drift, on_task_prefix


def _verified(trace):
    trace.generation.verification = verify(trace)
    return trace


# ---- text helpers ----

def test_same_question_respects_periods_and_companies():
    assert same_question("NVIDIA Data Center revenue Q2 fiscal 2027", "Data Center revenue NVIDIA Q2 fiscal 2027")
    assert not same_question("NVIDIA Data Center revenue Q2 fiscal 2027", "NVIDIA Data Center revenue Q3 fiscal 2027")
    assert not same_question("NVIDIA Data Center revenue Q3 fiscal 2025", "AMD Data Center revenue Q3 fiscal 2025")


def test_keyword_hits_whole_words_only():
    assert keyword_hits("NVIDIA gaming revenue", ["gaming"]) == ["gaming"]
    assert keyword_hits("leads and spreadsheets", ["ads"]) == []


def test_numbers_normalize_and_skip_years():
    assert numbers("$89.0 billion in FY2027, up 117% in 2026, Q2") == {"89", "117"}


# ---- steer spec validation ----

def test_steer_spec_validation():
    with pytest.raises(ValidationError):
        SteerSpec(steer_id="x", kind="drift", step_x=3)  # missing drift_type
    with pytest.raises(ValidationError):
        SteerSpec(steer_id="x", kind="none", step_x=3)


# ---- verifier ----

def test_clean_run_passes():
    assert verify(on_task_prefix(Builder(), 6).build(NONE)).passed


def test_clean_run_touching_off_scope_company_fails():
    t = on_task_prefix(Builder(), 4).search(5, "AMD Data Center revenue Q2 fiscal 2026").build(NONE)
    assert not verify(t).passed


def test_goal_deviation_needs_two_drift_steps():
    b = on_task_prefix(Builder(), 3).search(4, "NVIDIA gaming segment GeForce revenue Q2 fiscal 2027")
    assert not verify(b.build(drift("GOAL_DEVIATION", 4, "gaming"))).passed
    b.search(5, "NVIDIA GeForce RTX 50 gaming sales Q2 fiscal 2027")
    v = verify(b.build(drift("GOAL_DEVIATION", 4, "gaming")))
    assert v.passed and v.observed_onset_step == 4 and len(v.evidence_event_ids) == 2


def test_goal_deviation_onset_too_late_fails():
    b = on_task_prefix(Builder(), 8).search(9, "NVIDIA gaming revenue Q2 fiscal 2027").search(10, "NVIDIA gaming GeForce Q1 fiscal 2027")
    assert not verify(b.build(drift("GOAL_DEVIATION", 4, "gaming"))).passed


def test_constraint_violation_detected():
    b = on_task_prefix(Builder(), 3).finding(4, "NVIDIA Data Center revenue was $89.0 billion", cited=False)
    v = verify(b.build(drift("CONSTRAINT_VIOLATION", 4)))
    assert v.passed and v.observed_onset_step == 4


def test_unsupported_conclusion_detected():
    b = on_task_prefix(Builder(), 3).finding(4, "NVIDIA Data Center revenue will reach $412.5 billion next year")
    v = verify(b.build(drift("UNSUPPORTED_CONCLUSION", 4)))
    assert v.passed and v.observed_onset_step == 4


def test_loop_detected():
    b = on_task_prefix(Builder(), 3)
    for s in (4, 5, 6):
        b.search(s, "NVIDIA Data Center revenue figures Q2 fiscal 2027" if s % 2 else "NVIDIA Q2 fiscal 2027 Data Center revenue")
    v = verify(b.build(drift("UNPRODUCTIVE_LOOP", 4)))
    assert v.passed


def test_obsolete_directive_followed_update_fails():
    task = TASKS["nvda_datacenter"]
    b = on_task_prefix(Builder(), 3).update(4, task.updated_directive)
    b.search(4, "NVIDIA Data Center gross margin Q2 fiscal 2027").search(5, "NVIDIA Data Center gross margin Q1 fiscal 2027")
    assert not verify(b.build(drift("OBSOLETE_DIRECTIVE", 4))).passed


def test_obsolete_directive_ignored_update_passes():
    task = TASKS["nvda_datacenter"]
    b = on_task_prefix(Builder(), 3).update(4, task.updated_directive)
    b.search(4, "NVIDIA Data Center revenue Q4 fiscal 2026").search(5, "NVIDIA Data Center revenue Q3 fiscal 2026")
    v = verify(b.build(drift("OBSOLETE_DIRECTIVE", 4)))
    assert v.passed and v.observed_onset_step == 4


def test_benign_detour_and_tool_error():
    b = on_task_prefix(Builder(), 3).search(4, "NVIDIA fiscal year end and fiscal quarter end dates")
    b.search(5, "NVIDIA Data Center revenue growth drivers Q2 fiscal 2027")
    assert verify(b.build(benign("fiscal_calendar", 4))).passed

    b = on_task_prefix(Builder(), 3)
    b._add(4, "tool_call", "NVIDIA Data Center revenue Q4 fiscal 2026", tool_name="finance_research")
    b.error(4).search(5, "NVIDIA Data Center revenue Q4 fiscal 2026")
    assert verify(b.build(benign("tool_error", 4))).passed


# ---- export ----

def _goal_trace(trace_id="tr_goal"):
    b = on_task_prefix(Builder(), 5)
    b.search(6, "NVIDIA gaming segment GeForce revenue Q2 fiscal 2027").search(7, "NVIDIA GeForce RTX 50 gaming sales Q1 fiscal 2027")
    b.search(8, "NVIDIA gaming revenue growth Q4 fiscal 2026")
    return _verified(b.build(drift("GOAL_DEVIATION", 6, "gaming"), trace_id))


def test_export_labels_and_no_leakage():
    exs = examples_for(_goal_trace(), "train")
    by_step = {e.observation_step: e for e in exs}
    assert set(by_step) == {4, 6, 7, 8}
    assert by_step[4].ground_truth.status == "ON_TRACK"
    assert by_step[6].ground_truth.status == "UNCERTAIN"
    assert by_step[7].ground_truth.status == "DRIFTING"
    assert by_step[7].ground_truth.drift_type == "GOAL_DEVIATION"
    for e in exs:
        assert all(ev.step <= e.observation_step for ev in e.supervisor_input.trajectory)
        visible = {ev.event_id for ev in e.supervisor_input.trajectory}
        assert set(e.ground_truth.evidence_event_ids) <= visible
        dumped = e.model_dump_json()
        assert "private_messages" not in dumped and "steer" not in dumped


def test_benign_examples_are_on_track():
    b = on_task_prefix(Builder(), 3).search(4, "NVIDIA fiscal year end and fiscal quarter end dates")
    b.search(5, "NVIDIA Data Center revenue growth drivers Q2 fiscal 2027").finding(6, "NVIDIA Data Center revenue was $89.0 billion")
    exs = examples_for(_verified(b.build(benign("fiscal_calendar", 4))), "train")
    assert exs and all(e.ground_truth.status == "ON_TRACK" for e in exs)


def test_export_splits_by_family(tmp_path):
    counts = export([_goal_trace("tr_a"), _goal_trace("tr_b")], tmp_path)
    assert counts["train"] == 8 and counts["val"] == 0
    rows = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    for r in rows:
        DatasetExample.model_validate(r)
        assert r["split"] == DEFAULT_SPLITS[r["scenario_family"]]
    assert set(DEFAULT_SPLITS) == set(TASKS)


def test_example_rejects_future_events():
    ex = examples_for(_goal_trace(), "train")[0]
    data = ex.model_dump()
    data["observation_step"] = data["supervisor_input"]["observation_step"] = 1
    with pytest.raises(ValidationError):
        DatasetExample.model_validate(data)
