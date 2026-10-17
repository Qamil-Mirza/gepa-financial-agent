"""Summarize PitStop hard-pairs evaluations (evals.run_hard_cases output) into a baseline-vs-GEPA comparison.

  uv run python scripts/summarize_pitstop.py --arm baseline=pitstop/pitstop/runs/compare_nano/eval-supervisor_v0-* \\
      --arm gepa_pitstop=pitstop/pitstop/runs/compare_nano/eval-pitstop_nano_800-* --out docs/results/pitstop_nano.json

The first --arm is the reference for the paired per-case test. Per-trial numbers give the spread; a case counts as
solved by an arm when the majority of its trials are correct. Infra errors are excluded, as in PitStop's harness.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gepa_financial_agent.crusoe_client import estimate_cost_usd  # noqa: E402

V1 = ["verify_loop", "base_year", "honest_gap", "numeric_mismatch", "period_conflation", "primary_sources",
      "obsolete_directive", "peer_suggestion", "detour", "retry_vs_loop", "uncited", "premature_finish"]


def _rates(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["fired"] and r["label"] == "drift" for r in rows)
    tn = sum(not r["fired"] and r["label"] == "ok" for r in rows)
    pos = sum(r["label"] == "drift" for r in rows)
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["id"].rsplit(".", 1)[0], r["trial"])].append(r["correct"])
    return {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "drift_recall": tp / pos if pos else 0.0,
            "ok_specificity": tn / (n - pos) if n - pos else 0.0,
            "pair_accuracy": sum(all(v) for v in pairs.values()) / len(pairs) if pairs else 0.0,
            "false_alarms": n - pos - tn, "missed_drift": pos - tp}


def _spread(values: list[float]) -> dict:
    return {"mean": round(sum(values) / len(values), 3), "min": round(min(values), 3), "max": round(max(values), 3),
            "per_trial": [round(v, 3) for v in values]}


def summarize_arm(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    all_rows = [json.loads(l) for l in (run_dir / "judgments.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in all_rows if not r["infra_error"]]
    trials = sorted({r["trial"] for r in rows})
    per_trial = {t: _rates([r for r in rows if r["trial"] == t]) for t in trials}
    caught = [r for r in rows if r["label"] == "drift" and r["fired"]]
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["id"]].append(r["correct"])
    usage = summary.get("usage", {})
    return {
        "run_dir": str(run_dir), "model": summary["model"], "prompt": summary["prompt"], "trials": len(trials),
        "judgments": len(rows), "infra_errors_excluded": len(all_rows) - len(rows),
        **{k: _spread([per_trial[t][k] for t in trials])
           for k in ("accuracy", "pair_accuracy", "drift_recall", "ok_specificity")},
        "pooled": _rates(rows),
        "pooled_v1_categories": _rates([r for r in rows if r["category"] in V1]),
        "pooled_v2_categories": _rates([r for r in rows if r["category"] not in V1]),
        "by_category": {c: round(_rates([r for r in rows if r["category"] == c])["accuracy"], 3)
                        for c in dict.fromkeys(r["category"] for r in rows)},
        "drift_type_match_when_caught": round(sum(r["drift_type"] == r["expect_type"] for r in caught) / len(caught), 3)
        if caught else None,
        "invalid_json": sum(not r["valid_json"] for r in rows),
        "downgraded_for_no_valid_evidence": sum(r["downgraded"] for r in rows),
        "case_majority_correct": {cid: sum(v) * 2 > len(v) for cid, v in by_case.items()},
        "usage": usage, "est_cost_usd": estimate_cost_usd(summary["model"], usage.get("prompt_tokens", 0),
                                                          usage.get("completion_tokens", 0)),
        "wall_s": summary.get("wall_s"),
    }


def paired(ref: dict, arm: dict) -> dict:
    """Exact two-sided sign test (McNemar) on cases whose majority verdict differs between the two arms."""
    a, b = ref["case_majority_correct"], arm["case_majority_correct"]
    common = sorted(set(a) & set(b))
    fixed = [c for c in common if b[c] and not a[c]]
    broken = [c for c in common if a[c] and not b[c]]
    k, n = min(len(fixed), len(broken)), len(fixed) + len(broken)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"cases": len(common), "fixed": fixed, "broken": broken, "sign_test_p": round(p, 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="label=run_dir (glob ok; newest match is used)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    arms = {}
    for spec in args.arm:
        label, pattern = spec.split("=", 1)
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"no run dir matches {pattern}")
            return 2
        arms[label] = summarize_arm(Path(matches[-1]))
    ref_label = next(iter(arms))
    out = {"reference": ref_label, "arms": arms,
           "paired_vs_reference": {k: paired(arms[ref_label], v) for k, v in arms.items() if k != ref_label}}

    fmt = lambda s: f"{s['mean']:.3f} ({s['min']:.3f}-{s['max']:.3f})"
    print(f"{'arm':<16} {'prompt':<28} {'accuracy':<22} {'pair acc':<22} {'recall':<22} {'specificity':<22} "
          f"FA  miss infra")
    for k, a in arms.items():
        print(f"{k:<16} {a['prompt']:<28} {fmt(a['accuracy']):<22} {fmt(a['pair_accuracy']):<22} "
              f"{fmt(a['drift_recall']):<22} {fmt(a['ok_specificity']):<22} {a['pooled']['false_alarms']:<3} "
              f"{a['pooled']['missed_drift']:<4} {a['infra_errors_excluded']}")
    for k, p in out["paired_vs_reference"].items():
        print(f"{k} vs {ref_label}: fixed {len(p['fixed'])} {p['fixed']}, broken {len(p['broken'])} {p['broken']}, "
              f"sign-test p={p['sign_test_p']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
