from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .crusoe_client import ModelClient, estimate_cost_usd
from .policy_store import Policy
from .schemas import (DatasetExample, DriftStatus, DriftType, EvaluationRecord, EventType, SupervisorDecision,
                      SupervisorInput)
from .supervisor import Supervisor

log = logging.getLogger("gepa_financial_agent.evaluator")

W_STATUS, W_TYPE, W_EVIDENCE = 0.6, 0.2, 0.2
W_UNCERTAIN_ON_DRIFT, W_UNCERTAIN_ON_HEALTHY = 0.15, 0.2
PEN_FALSE_ALARM, PEN_INVALID_ID, PEN_INVALID_CAP, PEN_DRIFT_NO_EVIDENCE = -0.3, -0.1, -0.3, -0.1
SCORE_MIN, SCORE_MAX = -0.5, 1.0
STATUSES = [s.value for s in DriftStatus]
INVALID = "INVALID"


def score_prediction(record: EvaluationRecord, example: DatasetExample) -> tuple[float, dict[str, Any]]:
    gt = example.ground_truth
    info: dict[str, Any] = {
        "example_id": example.example_id, "observation_step": example.observation_step,
        "expected_status": gt.status.value, "expected_drift_type": gt.drift_type.value if gt.drift_type else None,
        "expected_evidence": list(gt.evidence_event_ids), "rationale": gt.rationale,
    }
    if record.decision is None:
        info.update(predicted_status=None, score=0.0,
                    feedback=f"INVALID OUTPUT: {record.error}. Return exactly one JSON object matching the output contract.")
        return 0.0, info
    d, notes, score = record.decision, [], 0.0
    S = DriftStatus
    if d.status == gt.status:
        score += W_STATUS
        notes.append(f"correct status {d.status.value}")
    elif gt.status == S.ON_TRACK and d.status == S.DRIFTING:
        score += PEN_FALSE_ALARM
        notes.append(f"FALSE ALARM: flagged a healthy agent as DRIFTING. Why it is healthy: {gt.rationale}")
    elif gt.status == S.DRIFTING and d.status == S.UNCERTAIN:
        score += W_UNCERTAIN_ON_DRIFT
        notes.append(f"hedged UNCERTAIN although drift was clear. {gt.rationale}")
    elif gt.status == S.DRIFTING and d.status == S.ON_TRACK:
        notes.append(f"MISSED DRIFT ({gt.drift_type.value if gt.drift_type else 'OTHER'}): {gt.rationale}")
    elif gt.status == S.UNCERTAIN and d.status == S.DRIFTING:
        notes.append("overconfident: only one ambiguous deviating event was visible; UNCERTAIN was correct")
    elif gt.status == S.UNCERTAIN and d.status == S.ON_TRACK:
        notes.append(f"missed an early warning sign; UNCERTAIN was correct. {gt.rationale}")
    elif gt.status == S.ON_TRACK and d.status == S.UNCERTAIN:
        score += W_UNCERTAIN_ON_HEALTHY
        notes.append(f"hedged on a healthy agent. {gt.rationale}")
    if gt.status == S.DRIFTING and d.status == S.DRIFTING and gt.drift_type:
        if d.drift_type == gt.drift_type:
            score += W_TYPE
        else:
            notes.append(f"wrong drift_type {d.drift_type.value if d.drift_type else None}; expected {gt.drift_type.value}")
    if d.status in (S.DRIFTING, S.UNCERTAIN) and gt.status in (S.DRIFTING, S.UNCERTAIN):
        hits = set(d.evidence_event_ids) & set(gt.evidence_event_ids)
        if gt.evidence_event_ids and hits:
            score += W_EVIDENCE
        elif not gt.evidence_event_ids and d.evidence_event_ids:
            score += W_EVIDENCE / 2
        elif gt.evidence_event_ids:
            notes.append(f"evidence did not point at the deviating events {gt.evidence_event_ids[:4]}")
    if record.invalid_evidence_ids:
        score += max(PEN_INVALID_ID * len(record.invalid_evidence_ids), PEN_INVALID_CAP)
        notes.append(f"FABRICATED evidence ids {record.invalid_evidence_ids}: ids must be copied from the trajectory")
    if d.status == S.DRIFTING and not d.evidence_event_ids:
        score += PEN_DRIFT_NO_EVIDENCE
        notes.append("DRIFTING verdict with no evidence ids")
    score = round(min(SCORE_MAX, max(SCORE_MIN, score)), 3)
    info.update(predicted_status=d.status.value, predicted_drift_type=d.drift_type.value if d.drift_type else None,
                predicted_evidence=list(d.evidence_event_ids), invalid_evidence_ids=list(record.invalid_evidence_ids),
                confidence=d.confidence, explanation=d.explanation, feedback="; ".join(notes), score=score)
    return score, info


@dataclass
class Metrics:
    n: int
    n_valid: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    drifting_precision: float
    drifting_recall: float
    false_positive_rate: float
    false_negative_rate: float
    per_drift_type_accuracy: dict[str, float]
    evidence_validity_rate: float
    evidence_hit_rate: float
    invalid_output_rate: float
    mean_score: float
    mean_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    est_cost_usd: float | None
    mean_detection_delay: float | None
    undetected_drift_scenarios: int
    confusion: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(records: list[EvaluationRecord], examples: list[DatasetExample], scores: list[float]) -> Metrics:
    assert len(records) == len(examples) == len(scores)
    preds = [r.decision.status.value if r.decision else INVALID for r in records]
    gts = [e.ground_truth.status.value for e in examples]
    confusion = {g: {p: 0 for p in STATUSES + [INVALID]} for g in STATUSES}
    for g, p in zip(gts, preds):
        confusion[g][p] += 1
    per_class: dict[str, dict[str, float]] = {}
    present = [s for s in STATUSES if s in gts]
    for s in STATUSES:
        tp = confusion[s][s]
        fp = sum(confusion[g][s] for g in STATUSES if g != s)
        fn = sum(confusion[s][p] for p in confusion[s] if p != s)
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        per_class[s] = {"precision": prec, "recall": rec, "f1": _div(2 * prec * rec, prec + rec), "support": tp + fn}
    macro_f1 = _div(sum(per_class[s]["f1"] for s in present), len(present))
    n_on = gts.count("ON_TRACK")
    n_drift = gts.count("DRIFTING")
    fpr = _div(confusion["ON_TRACK"]["DRIFTING"], n_on)
    fnr = _div(sum(v for p, v in confusion["DRIFTING"].items() if p != "DRIFTING"), n_drift)
    by_type: dict[str, list[int]] = defaultdict(list)
    valid_ids = invalid_ids = 0
    hit_num = hit_den = 0
    for r, e in zip(records, examples):
        gt = e.ground_truth
        if r.decision:
            valid_ids += len(r.decision.evidence_event_ids)
        invalid_ids += len(r.invalid_evidence_ids)
        if gt.status == DriftStatus.DRIFTING and gt.drift_type:
            ok = bool(r.decision and r.decision.status == DriftStatus.DRIFTING and r.decision.drift_type == gt.drift_type)
            by_type[gt.drift_type.value].append(int(ok))
        if gt.status == DriftStatus.DRIFTING and gt.evidence_event_ids and r.decision and r.decision.status == DriftStatus.DRIFTING:
            hit_den += 1
            hit_num += int(bool(set(r.decision.evidence_event_ids) & set(gt.evidence_event_ids)))
    # detection delay: per scenario with onset info, first DRIFTING prediction at/after onset
    by_scenario: dict[str, list[tuple[int, int | None, bool]]] = defaultdict(list)
    for r, e in zip(records, examples):
        onset = e.ground_truth.drift_onset_step
        if onset is not None:
            flagged = bool(r.decision and r.decision.status == DriftStatus.DRIFTING)
            by_scenario[e.scenario_id].append((e.observation_step, onset, flagged))
    delays: list[int] = []
    undetected = 0
    for rows in by_scenario.values():
        rows.sort()
        onset = rows[0][1]
        hit = next((step for step, _, flagged in rows if flagged and step >= onset), None)
        if hit is None:
            undetected += 1
        else:
            delays.append(hit - onset)
    model = records[0].model if records else ""
    pt, ct, cached = (sum(r.prompt_tokens for r in records), sum(r.completion_tokens for r in records),
                      sum(r.cached_tokens for r in records))
    return Metrics(
        n=len(records), n_valid=sum(1 for r in records if r.decision),
        accuracy=_div(sum(1 for g, p in zip(gts, preds) if g == p), len(records)), macro_f1=macro_f1,
        per_class=per_class, drifting_precision=per_class["DRIFTING"]["precision"],
        drifting_recall=per_class["DRIFTING"]["recall"], false_positive_rate=fpr, false_negative_rate=fnr,
        per_drift_type_accuracy={k: _div(sum(v), len(v)) for k, v in sorted(by_type.items())},
        evidence_validity_rate=_div(valid_ids, valid_ids + invalid_ids) if (valid_ids + invalid_ids) else 1.0,
        evidence_hit_rate=_div(hit_num, hit_den), invalid_output_rate=_div(len(records) - sum(1 for r in records if r.decision), len(records)),
        mean_score=_div(sum(scores), len(scores)), mean_latency_ms=_div(sum(r.latency_ms for r in records), len(records)),
        total_prompt_tokens=pt, total_completion_tokens=ct, est_cost_usd=estimate_cost_usd(model, pt, ct, cached),
        mean_detection_delay=_div(sum(delays), len(delays)) if delays else None,
        undetected_drift_scenarios=undetected, confusion=confusion,
    )


@dataclass
class EvalRun:
    label: str
    policy_name: str
    policy_version: str
    model: str
    mocked: bool
    records: list[EvaluationRecord]
    scores: list[float]
    infos: list[dict[str, Any]]
    metrics: Metrics
    out_dir: str | None = None


def _persist(run: EvalRun, examples: list[DatasetExample], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for ex, rec, score, info in zip(examples, run.records, run.scores, run.infos):
            fh.write(json.dumps({"example_id": ex.example_id, "scenario_id": ex.scenario_id, "score": score,
                                 "record": rec.model_dump(mode="json"), "info": info}) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "label": run.label, "policy_name": run.policy_name, "policy_version": run.policy_version,
        "model": run.model, "mocked": run.mocked, "n_examples": len(examples), "metrics": run.metrics.to_dict(),
    }, indent=2) + "\n", encoding="utf-8")


def run_evaluation(policy: Policy, examples: list[DatasetExample], client: ModelClient, *, concurrency: int = 4,
                   out_dir: str | Path | None = None, label: str = "eval", json_mode: bool = False,
                   max_tokens: int = 1024) -> EvalRun:
    sup = Supervisor(client, policy, json_mode=json_mode, max_tokens=max_tokens)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = list(pool.map(lambda e: sup.evaluate_sync(e.supervisor_input, example_id=e.example_id), examples))
    scored = [score_prediction(r, e) for r, e in zip(records, examples)]
    scores, infos = [s for s, _ in scored], [i for _, i in scored]
    run = EvalRun(label=label, policy_name=policy.name, policy_version=policy.version, model=client.model,
                  mocked=any(r.mocked for r in records), records=records, scores=scores, infos=infos,
                  metrics=compute_metrics(records, examples, scores), out_dir=str(out_dir) if out_dir else None)
    if out_dir:
        _persist(run, examples, Path(out_dir))
    log.info("%s policy=%s model=%s%s macro_f1=%.3f fpr=%.3f", label, policy.name, client.model,
             " [MOCK]" if run.mocked else "", run.metrics.macro_f1, run.metrics.false_positive_rate)
    return run


def heuristic_baseline(inp: SupervisorInput, repeat_threshold: int = 3) -> SupervisorDecision:
    """Deterministic non-LLM baseline: flags only identical repeated tool calls."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in inp.trajectory:
        if e.event_type == EventType.TOOL_CALL:
            groups[f"{e.tool_name}|{json.dumps(e.tool_arguments or {}, sort_keys=True)}"].append(e.event_id)
    loops = [ids for ids in groups.values() if len(ids) >= repeat_threshold]
    if loops:
        ids = max(loops, key=len)
        return SupervisorDecision(agent_id=inp.agent_id, task_id=inp.task_id, status=DriftStatus.DRIFTING, confidence=0.7,
                                  drift_type=DriftType.UNPRODUCTIVE_LOOP, evidence_event_ids=ids,
                                  explanation=f"identical tool call repeated {len(ids)} times", policy_version="heuristic",
                                  observation_step=inp.observation_step)
    return SupervisorDecision(agent_id=inp.agent_id, task_id=inp.task_id, status=DriftStatus.ON_TRACK, confidence=0.5,
                              explanation="no repeated tool calls detected", policy_version="heuristic",
                              observation_step=inp.observation_step)


def run_heuristic_evaluation(examples: list[DatasetExample], out_dir: str | Path | None = None) -> EvalRun:
    records = [EvaluationRecord(agent_id=e.supervisor_input.agent_id, task_id=e.supervisor_input.task_id,
                                observation_step=e.observation_step, policy_version="heuristic", model="heuristic-rules",
                                decision=heuristic_baseline(e.supervisor_input), example_id=e.example_id) for e in examples]
    scored = [score_prediction(r, e) for r, e in zip(records, examples)]
    scores, infos = [s for s, _ in scored], [i for _, i in scored]
    run = EvalRun("heuristic", "heuristic", "heuristic", "heuristic-rules", False, records, scores, infos,
                  compute_metrics(records, examples, scores), str(out_dir) if out_dir else None)
    if out_dir:
        _persist(run, examples, Path(out_dir))
    return run


def format_comparison(runs: list[EvalRun]) -> str:
    cols = [("label", 22), ("policy", 10), ("mode", 5), ("n", 4), ("macroF1", 8), ("acc", 6), ("P_drift", 8),
            ("R_drift", 8), ("FPR", 6), ("FNR", 6), ("invalid", 8), ("evid_ok", 8), ("score", 6), ("ms", 7), ("usd", 8)]
    head = " ".join(f"{c:<{w}}" for c, w in cols)
    lines = [head, "-" * len(head)]
    for r in runs:
        m = r.metrics
        vals = [r.label[:22], r.policy_version[:10], "MOCK" if r.mocked else "REAL", m.n, f"{m.macro_f1:.3f}",
                f"{m.accuracy:.3f}", f"{m.drifting_precision:.3f}", f"{m.drifting_recall:.3f}",
                f"{m.false_positive_rate:.3f}", f"{m.false_negative_rate:.3f}", f"{m.invalid_output_rate:.3f}",
                f"{m.evidence_validity_rate:.3f}", f"{m.mean_score:.3f}", f"{m.mean_latency_ms:.0f}",
                "n/a" if m.est_cost_usd is None else f"{m.est_cost_usd:.4f}"]
        lines.append(" ".join(f"{str(v):<{w}}" for v, (_, w) in zip(vals, cols)))
    return "\n".join(lines)
