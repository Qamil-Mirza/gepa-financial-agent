"""Aggregate repeated evaluate_supervisor.py runs (same split, same policies) into mean/min/max per policy.

Usage: uv run python scripts/summarize_repeats.py --dataset synthetic-data/data/dataset.jsonl --split test \
           --runs 'runs/eval/team_final_r*' --out docs/results/final_test_20261017.json
No model calls: it reads metrics.json and predictions.jsonl written by evaluate_supervisor.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gepa_financial_agent.dataset import load_jsonl, split_by_scenario  # noqa: E402
from gepa_financial_agent.evaluator import compute_metrics  # noqa: E402
from gepa_financial_agent.schemas import EvaluationRecord  # noqa: E402

KEYS = ["macro_f1", "accuracy", "false_positive_rate", "drifting_recall", "mean_detection_delay",
        "undetected_drift_scenarios", "invalid_output_rate", "est_cost_usd"]


def spread(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"mean": statistics.fmean(xs), "min": min(xs), "max": max(xs), "values": xs} if xs else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", required=True, help="glob of run directories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    examples = load_jsonl(a.dataset).examples
    split = examples if a.split == "all" else getattr(split_by_scenario(examples, seed=a.seed), a.split)
    by_id = {e.example_id: e for e in split}
    runs = sorted(Path(p) for p in glob.glob(a.runs))
    per: dict[str, dict] = defaultdict(lambda: {"runs": [], "metrics": defaultdict(list), "infra_failures": [],
                                                 "macro_f1_excl_infra": [], "type_hits": defaultdict(lambda: [0, 0])})
    for r in runs:
        for d in sorted(p for p in r.iterdir() if (p / "metrics.json").exists()):
            pol = d.name
            m = json.load(open(d / "metrics.json"))["metrics"]
            agg = per[pol]
            agg["runs"].append(r.name)
            for k in KEYS:
                agg["metrics"][k].append(m.get(k))
            recs, exs, scores = [], [], []
            fails = 0
            for line in open(d / "predictions.jsonl"):
                p = json.loads(line)
                rec, ex = EvaluationRecord.model_validate(p["record"]), by_id[p["example_id"]]
                gt = ex.ground_truth
                if gt.status.value == "DRIFTING" and gt.drift_type:
                    h = agg["type_hits"][gt.drift_type.value]
                    h[1] += 1
                    h[0] += int(bool(rec.decision and rec.decision.status == gt.status and rec.decision.drift_type == gt.drift_type))
                if rec.decision is None and (rec.error or "").startswith("model_error"):
                    fails += 1  # transport failure (429 / connection) after retries, not a policy output
                    continue
                recs.append(rec), exs.append(ex), scores.append(p["score"])
            agg["infra_failures"].append(fails)
            agg["macro_f1_excl_infra"].append(compute_metrics(recs, exs, scores).macro_f1)

    out = {"dataset": a.dataset, "split": a.split, "split_seed": a.seed, "n": len(split), "runs": [r.name for r in runs],
           "policies": {}}
    for pol, agg in per.items():
        out["policies"][pol] = {
            "repeats": len(agg["runs"]),
            **{k: spread(v) for k, v in agg["metrics"].items()},
            "infra_failures": agg["infra_failures"],
            "macro_f1_excl_infra_failures": spread(agg["macro_f1_excl_infra"]),
            "per_drift_type_correct_pooled": {t: f"{h}/{n}" for t, (h, n) in sorted(agg["type_hits"].items())},
            "per_drift_type_accuracy_mean": {t: h / n for t, (h, n) in sorted(agg["type_hits"].items())},
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"split={a.split} n={len(split)} repeats={len(runs)}")
    for pol, s in out["policies"].items():
        f = s["macro_f1"]
        print(f"{pol:<16} macroF1 mean {f['mean']:.3f} (min {f['min']:.3f}, max {f['max']:.3f})  "
              f"excl-infra mean {s['macro_f1_excl_infra_failures']['mean']:.3f}  infra_failures {s['infra_failures']}  "
              f"undetected {s['undetected_drift_scenarios']['values']}")
        print("   ", s["per_drift_type_correct_pooled"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
