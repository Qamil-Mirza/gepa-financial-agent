"""Score the live supervisor on the hand-labeled hard cases (Crusoe only, no You.com).

  python -m evals.run_hard_cases [--trials 3] [--prompt prompts/supervisor_v0.txt] [--model <crusoe id>]

A case "fires" when the alert policy would intervene on that single verdict
(DRIFTING at confidence >= 0.75). Results go to runs/hard-cases-<timestamp>/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from pitstop.config import ROOT, load_settings
from pitstop.llm import LLMClient, UsageTracker
from pitstop.schemas import Status
from pitstop.supervisor.policy import AlertPolicy
from pitstop.supervisor.supervisor import Supervisor

from .hard_cases import all_cases


def fires(vd) -> bool:
    d = vd.decision
    return d.status == Status.DRIFTING and d.confidence >= AlertPolicy().high


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--prompt", default=str(ROOT / "prompts" / "supervisor_v0.txt"))
    p.add_argument("--model")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()

    s = load_settings()
    model = args.model or s.supervisor_model
    llm = LLMClient(model, api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, temperature=0.0, usage=UsageTracker())
    sup = Supervisor(llm, Path(args.prompt))
    sem = asyncio.Semaphore(args.concurrency)
    cases = all_cases()

    async def one(case, inp, trial):
        async with sem:
            vd = await sup.evaluate(inp)
        d = vd.decision
        return {"id": case["id"], "label": case["label"], "expect_type": case["expect_type"], "trial": trial,
                "status": d.status.value, "confidence": d.confidence, "drift_type": d.drift_type.value,
                "fired": fires(vd), "evidence": d.evidence_event_ids, "dropped_evidence": vd.dropped_evidence_ids,
                "downgraded": vd.downgraded, "valid_json": vd.valid_json, "issues": vd.issues,
                "latency_s": round(vd.latency_s, 1), "explanation": d.explanation}

    t0 = time.time()
    rows = await asyncio.gather(*(one(c, i, t) for c, i in cases for t in range(args.trials)))

    # per-case majority vote across trials
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["id"], []).append(r)
    per_case = []
    for c, _ in cases:
        rs = by_case[c["id"]]
        n_fire = sum(r["fired"] for r in rs)
        maj = n_fire * 2 > len(rs)
        correct = maj == (c["label"] == "drift")
        types = Counter(r["drift_type"] for r in rs if r["fired"])
        per_case.append({"id": c["id"], "label": c["label"], "fired": f"{n_fire}/{len(rs)}", "correct": correct,
                         "consistent": n_fire in (0, len(rs)),
                         "statuses": dict(Counter(f"{r['status']}@{r['confidence']:.2f}" for r in rs)),
                         "type_match": (types.most_common(1)[0][0] == c["expect_type"]) if types else None,
                         "why": c["why"]})

    def rate(pred, lab):
        return sum(1 for r in rows if r["fired"] == pred and (r["label"] == "drift") == lab)

    tp, fn, fp, tn = rate(True, True), rate(False, True), rate(True, False), rate(False, False)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "model": model, "prompt": sup.prompt_version, "trials": args.trials, "cases": len(cases), "judgments": len(rows),
        "per_judgment": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": round((tp + tn) / len(rows), 3),
                         "drift_recall": round(rec, 3), "drift_precision": round(prec, 3),
                         "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                         "false_alarm_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0},
        "per_case_majority_accuracy": f"{sum(c['correct'] for c in per_case)}/{len(per_case)}",
        "inconsistent_cases": [c["id"] for c in per_case if not c["consistent"]],
        "drift_type_match_on_caught": f"{sum(1 for c in per_case if c['type_match'])}/{sum(1 for c in per_case if c['type_match'] is not None)}",
        "judgments_with_dropped_evidence": sum(1 for r in rows if r["dropped_evidence"]),
        "downgraded_for_no_valid_evidence": sum(1 for r in rows if r["downgraded"]),
        "invalid_json": sum(1 for r in rows if not r["valid_json"]),
        "usage": llm.usage.as_dict(), "wall_s": round(time.time() - t0, 1),
    }

    out = s.runs_dir / f"hard-cases-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "summary.json").write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))
    for c in per_case:
        mark = "✓" if c["correct"] else "✗"
        print(f"{mark} {c['id']:<36} label={c['label']:<5} fired={c['fired']}  {c['statuses']}")
    print(json.dumps(summary, indent=2))
    print("results:", out)


if __name__ == "__main__":
    asyncio.run(main())
