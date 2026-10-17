"""Run real GEPA optimization of the supervisor prompt, then compare baseline vs optimized on val and (once) test.

  uv run python scripts/optimize_supervisor.py --mock --max-metric-calls 20          # pipeline check, no inference
  uv run python scripts/optimize_supervisor.py --max-metric-calls 120 --promote       # real run on Crusoe
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
from gepa_financial_agent.evaluator import format_comparison, run_evaluation  # noqa: E402
from gepa_financial_agent.optimizer import CrusoeReflectionLM, OptimizeConfig, run_optimization  # noqa: E402
from gepa_financial_agent.policy_store import PolicyStore  # noqa: E402


def mock_reflection(prompt):
    return ("```\nYou are a careful drift supervisor. [MOCK REFLECTION: this text was produced by a stub, not a model]\n"
            "Flag off-directive topics, repeated identical calls, non-primary sources, unsupported claims, and ignored "
            "directive updates. Do not flag tool retries or brief relevant detours.\n```")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed-policy", default="baseline")
    ap.add_argument("--max-metric-calls", type=int, default=120)
    ap.add_argument("--minibatch", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--gepa-seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="mock supervisor and reflection (pipeline check only)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--promote", action="store_true", help="copy the optimized policy to policies/current.txt")
    ap.add_argument("--final-test", action="store_true",
                    help="also evaluate the held-out test split (do this once, after choosing a policy on val)")
    ap.add_argument("--policies", default="policies", help="policy store root")
    ap.add_argument("--allow-invalid", action="store_true", help="continue even if some dataset lines are invalid")
    args = ap.parse_args(argv)

    s = Settings.from_env()
    examples, n_invalid = [], 0
    for p in args.dataset or ["data/fixtures/dev_examples.jsonl"]:
        rep = load_jsonl(p)
        print(rep.summary())
        examples.extend(rep.examples)
        n_invalid += len(rep.invalid)
    if n_invalid and not args.allow_invalid:
        print(f"{n_invalid} invalid dataset lines (listed above); fix them or pass --allow-invalid")
        return 4
    sp = split_by_scenario(examples, seed=args.split_seed)
    print(f"splits: {sp.summary()} | train labels={label_distribution(sp.train)} val labels={label_distribution(sp.val)}")

    if args.mock:
        print("MODE: MOCK. Supervisor and reflection are stubs. Results prove plumbing only, not GEPA effectiveness.")
        sup_client, reflection = MockModelClient(), mock_reflection
    else:
        if not s.has_key:
            print("CRUSOE_API_KEY not set; export it or use --mock.")
            return 2
        sup_client = CrusoeClient.from_settings(s, "supervisor")
        reflection = CrusoeReflectionLM(CrusoeClient.from_settings(s, "reflection", max_concurrent=1, thinking=None))
        print(f"MODE: REAL. supervisor={sup_client.model} reflection={reflection.client.model}")

    store = PolicyStore(args.policies)
    seed_policy = store.load(args.seed_policy)
    cfg = OptimizeConfig(max_metric_calls=args.max_metric_calls, reflection_minibatch_size=args.minibatch,
                         seed=args.gepa_seed, max_workers=args.max_workers or s.max_concurrent_inferences,
                         json_mode=s.json_mode, **({"run_dir": args.run_dir} if args.run_dir else {}))
    art = run_optimization(seed_policy, sp.train, sp.val, sup_client, reflection, cfg)
    run_dir = Path(art.run_dir)
    print(f"GEPA finished: candidates={art.num_candidates} metric_calls={art.total_metric_calls} "
          f"seed_val={art.seed_val_score:.3f} best_val={art.best_val_score:.3f} run_dir={run_dir}")

    # Comparison: same model, decoding, schema, cutoffs. The test split is only touched with --final-test.
    runs = [run_evaluation(seed_policy, sp.val, sup_client, out_dir=run_dir / "eval_val_baseline", label="baseline/val", json_mode=s.json_mode),
            run_evaluation(art.best_policy, sp.val, sup_client, out_dir=run_dir / "eval_val_gepa", label="gepa/val", json_mode=s.json_mode)]
    if args.final_test:
        runs += [run_evaluation(seed_policy, sp.test, sup_client, out_dir=run_dir / "eval_test_baseline", label="baseline/test", json_mode=s.json_mode),
                 run_evaluation(art.best_policy, sp.test, sup_client, out_dir=run_dir / "eval_test_gepa", label="gepa/test", json_mode=s.json_mode)]
    table = format_comparison(runs)
    print(table)
    comparison = {"mocked": any(r.mocked for r in runs), "supervisor_model": sup_client.model,
                  "runs": {r.label: {"policy_version": r.policy_version, "metrics": r.metrics.to_dict()} for r in runs},
                  "supervisor_usage": sup_client.tracker.snapshot()}
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (run_dir / "comparison.txt").write_text(table + "\n")

    metadata = {"run_dir": str(run_dir), "mocked": comparison["mocked"],
                "val": {r.label: r.metrics.macro_f1 for r in runs if r.label.endswith("/val")},
                "test": {r.label: r.metrics.macro_f1 for r in runs if r.label.endswith("/test")}}
    name = art.best_policy.name if not comparison["mocked"] else "mock_" + art.best_policy.name.removeprefix("gepa_")
    saved = store.save_version(art.best_policy.text, name, metadata=metadata)
    print(f"saved policy version {saved.name} ({saved.version})")
    if args.final_test:
        b, g = next(r for r in runs if r.label == "baseline/test"), next(r for r in runs if r.label == "gepa/test")
        verdict = ("GEPA policy improved held-out macro-F1" if g.metrics.macro_f1 > b.metrics.macro_f1 else
                   "GEPA policy did NOT improve held-out macro-F1")
        print(f"{verdict}: baseline={b.metrics.macro_f1:.3f} gepa={g.metrics.macro_f1:.3f} "
              f"FPR baseline={b.metrics.false_positive_rate:.3f} gepa={g.metrics.false_positive_rate:.3f}"
              + (" [MOCK: not meaningful]" if comparison["mocked"] else ""))
    if args.promote:
        if comparison["mocked"]:
            print("refusing --promote: this run used mock inference")
            return 3
        store.promote(saved.name)
        print("promoted to policies/current.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
