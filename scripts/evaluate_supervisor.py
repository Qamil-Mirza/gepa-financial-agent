"""Evaluate one or more supervisor policies on a dataset split with the same model, decoding, schema, and cutoffs.

Examples:
  uv run python scripts/evaluate_supervisor.py --mock --split val
  uv run python scripts/evaluate_supervisor.py --policy baseline --policy current --split test --heuristic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gepa_financial_agent.config import Settings  # noqa: E402
from gepa_financial_agent.crusoe_client import CrusoeClient, MockModelClient  # noqa: E402
from gepa_financial_agent.dataset import label_distribution, load_jsonl, split_by_scenario  # noqa: E402
from gepa_financial_agent.evaluator import format_comparison, run_evaluation, run_heuristic_evaluation  # noqa: E402
from gepa_financial_agent.policy_store import PolicyStore  # noqa: E402


def load_examples(paths: list[str]):
    out = []
    for p in paths:
        rep = load_jsonl(p)
        print(rep.summary())
        out.extend(rep.examples)
    return out


def select_split(examples, split: str, seed: int):
    if split == "all":
        return examples
    return getattr(split_by_scenario(examples, seed=seed), split)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=None, help="JSONL path(s); default data/fixtures/dev_examples.jsonl")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", action="append", default=None, help="policy name or path (repeatable); default baseline")
    ap.add_argument("--heuristic", action="store_true", help="also run the deterministic rules baseline")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir; default runs/eval/<timestamp>")
    args = ap.parse_args(argv)

    s = Settings.from_env()
    examples = select_split(load_examples(args.dataset or ["data/fixtures/dev_examples.jsonl"]), args.split, args.seed)
    print(f"split={args.split} n={len(examples)} labels={label_distribution(examples)}")
    if args.mock:
        client = MockModelClient()
        print("MODE: MOCK (deterministic stub, not real inference)")
    else:
        if not s.has_key:
            print("CRUSOE_API_KEY not set; use --mock or export the key.")
            return 2
        client = CrusoeClient.from_settings(s, "supervisor", max_concurrent=args.concurrency or s.max_concurrent_inferences)
        print(f"MODE: REAL model={client.model}")
    out = Path(args.out or f"runs/eval/{time.strftime('%Y%m%d_%H%M%S')}")
    store = PolicyStore("policies")
    runs = []
    if args.heuristic:
        runs.append(run_heuristic_evaluation(examples, out / "heuristic"))
    for name in args.policy or ["baseline"]:
        pol = store.load(name)
        runs.append(run_evaluation(pol, examples, client, concurrency=args.concurrency or s.max_concurrent_inferences,
                                   out_dir=out / pol.name, label=f"{pol.name}/{args.split}", json_mode=s.json_mode))
    print(format_comparison(runs))
    (out / "summary.json").write_text(json.dumps({"split": args.split, "seed": args.seed, "n": len(examples),
                                                  "runs": [{"label": r.label, "policy_version": r.policy_version,
                                                            "mocked": r.mocked, "metrics": r.metrics.to_dict()} for r in runs]},
                                                 indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
