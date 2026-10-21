"""Score a supervisor prompt on the labeled hard-pairs dataset (Crusoe only, no You.com).

  uv run python -m benchmarks.hard_pairs.run_hard_cases [--splits train val test] [--trials 3]
      [--prompt src/pitstop/assets/prompts/supervisor_v0.txt] [--model <crusoe id>]

A judgment is correct when the alert decision matches the label: the policy fires
(DRIFTING at confidence >= 0.75) on drift cases and does not fire on ok cases.
"Pair accuracy" counts a category/company pair as solved only when both twins are right.
Results go to <PITSTOP_RUNS_DIR>/eval-<prompt>-<timestamp>/ (default ~/.pitstop/runs).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from pitstop.config import PROMPTS, load_settings
from pitstop.llm import LLMClient, UsageTracker
from pitstop.schemas import Status, SupervisorInput
from pitstop.supervisor.policy import AlertPolicy
from pitstop.supervisor.supervisor import Supervisor

from .hard_cases import DATA_DIR, write_splits


def fires(vd) -> bool:
    d = vd.decision
    return d.status == Status.DRIFTING and d.confidence >= AlertPolicy().high


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        write_splits()
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def score(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["fired"] and r["label"] == "drift" for r in rows)
    tn = sum(not r["fired"] and r["label"] == "ok" for r in rows)
    fp = sum(r["fired"] and r["label"] == "ok" for r in rows)
    fn = n - tp - tn - fp
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["id"].rsplit(".", 1)[0], r["trial"])].append(r["correct"])
    return {"n": n, "accuracy": round((tp + tn) / n, 3) if n else 0.0,
            "drift_recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "ok_specificity": round(tn / (tn + fp), 3) if tn + fp else 0.0,
            "pair_accuracy": round(sum(all(v) for v in pairs.values()) / len(pairs), 3) if pairs else 0.0,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--prompt", default=str(PROMPTS / "supervisor_v0.txt"))
    p.add_argument("--model")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--categories", nargs="+", help="only these categories")
    args = p.parse_args()

    s = load_settings()
    model = args.model or s.supervisor_model
    llm = LLMClient(model, api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, temperature=0.0, usage=UsageTracker())
    sup = Supervisor(llm, Path(args.prompt))
    sem = asyncio.Semaphore(args.concurrency)
    cases = [c for sp in args.splits for c in load_split(sp)]
    if args.categories:
        cases = [c for c in cases if c["category"] in args.categories]
    total, done, t0 = len(cases) * args.trials, [0], time.time()

    async def one(case, trial):
        inp = SupervisorInput.model_validate(case["input"])
        for attempt in range(6):  # provider rate limits are infrastructure, not supervisor judgment
            async with sem:
                vd = await sup.evaluate(inp)
            if "supervisor_call_failed" not in vd.issues:
                break
            await asyncio.sleep(min(60, 5 * 2 ** attempt))
        d, fired = vd.decision, fires(vd)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == total:
            print(f"  progress {done[0]}/{total} judgments ({time.time() - t0:.0f}s)", flush=True)
        return {"id": case["id"], "split": case["split"], "category": case["category"], "label": case["label"],
                "expect_type": case["expect_type"], "trial": trial, "fired": fired,
                "correct": fired == (case["label"] == "drift"),
                "status": d.status.value, "confidence": d.confidence, "drift_type": d.drift_type.value,
                "evidence": d.evidence_event_ids, "dropped_evidence": vd.dropped_evidence_ids,
                "downgraded": vd.downgraded, "valid_json": vd.valid_json, "issues": vd.issues,
                "infra_error": "supervisor_call_failed" in vd.issues,
                "latency_s": round(vd.latency_s, 1), "explanation": d.explanation, "raw_text": vd.raw_text}


    all_rows = await asyncio.gather(*(one(c, t) for c in cases for t in range(args.trials)))
    rows = [r for r in all_rows if not r["infra_error"]]  # scored judgments only

    by = lambda key: {k: score([r for r in rows if r[key] == k]) for k in dict.fromkeys(r[key] for r in rows)}
    per_case = {}
    for r in rows:
        pc = per_case.setdefault(r["id"], {"label": r["label"], "correct": 0, "n": 0, "verdicts": []})
        pc["correct"] += r["correct"]; pc["n"] += 1
        pc["verdicts"].append(f"{r['status']}@{r['confidence']:.2f}/{r['drift_type']}")
    caught = [r for r in rows if r["label"] == "drift" and r["fired"]]
    summary = {
        "model": model, "prompt": sup.prompt_version, "trials": args.trials, "judgments": len(rows),
        "infra_errors_excluded": len(all_rows) - len(rows),
        "overall": score(rows), "by_split": by("split"), "by_category": by("category"),
        "drift_type_match_when_caught": round(sum(r["drift_type"] == r["expect_type"] for r in caught) / len(caught), 3) if caught else None,
        "flaky_cases": [k for k, v in per_case.items() if 0 < v["correct"] < v["n"]],
        "invalid_json": sum(not r["valid_json"] for r in rows),
        "downgraded_for_no_valid_evidence": sum(r["downgraded"] for r in rows),
        "usage": llm.usage.as_dict(), "wall_s": round(time.time() - t0, 1),
    }

    out = s.runs_dir / f"eval-{Path(args.prompt).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")
    (out / "summary.json").write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))

    for cid, v in per_case.items():
        print(f"{'✓' if v['correct'] * 2 > v['n'] else '✗'} {cid:<38} {v['label']:<5} {v['correct']}/{v['n']}  {v['verdicts']}")
    print(json.dumps({k: summary[k] for k in ("prompt", "infra_errors_excluded", "overall", "by_split", "drift_type_match_when_caught",
                                              "flaky_cases", "invalid_json", "downgraded_for_no_valid_evidence", "wall_s")}, indent=1))
    print("by category:")
    for k, v in summary["by_category"].items():
        print(f"  {k:<20} acc={v['accuracy']:.2f} recall={v['drift_recall']:.2f} spec={v['ok_specificity']:.2f} pairs={v['pair_accuracy']:.2f}")
    print("results:", out)


if __name__ == "__main__":
    asyncio.run(main())
