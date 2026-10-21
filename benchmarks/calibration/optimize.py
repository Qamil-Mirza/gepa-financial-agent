"""Run GEPA on PitStop's supervisor prompt, scored by PitStop's own supervisor code.

Reflects on the hard-pairs train split (NVIDIA), selects on val (Broadcom). Never reads test (AMD): compare the saved
prompt against the baseline afterwards with PitStop's own harness, e.g.

  uv run python -m benchmarks.calibration.optimize --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --max-metric-calls 500
  uv run python -m benchmarks.hard_pairs.run_hard_cases --splits test --trials 5 \\
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B --prompt src/pitstop/assets/prompts/gepa/<name>.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitstop.config import PROMPTS
from pitstop.correct.client import CrusoeClient, MockModelClient
from pitstop.correct.gepa import CrusoeReflectionLM, OptimizeConfig, run_optimization
from pitstop.correct.policy import Policy
from pitstop.correct.settings import Settings

from .adapter import BACKGROUND, OBJECTIVE, SEED_PROMPT, load_hard_cases, make_pitstop_evaluator


def mock_reflection(prompt):
    return "```\nYou are a careful drift supervisor. [MOCK REFLECTION: produced by a stub, not a model]\n```"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="supervisor model (default: SUPERVISOR_MODEL / Nemotron Super)")
    ap.add_argument("--max-metric-calls", type=int, default=300)
    ap.add_argument("--minibatch", type=int, default=6)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--val-repeats", type=int, default=2, help="score each val case this many times (less selection noise)")
    ap.add_argument("--gepa-seed", type=int, default=0)
    ap.add_argument("--acceptance", default="strict_improvement", choices=["strict_improvement", "improvement_or_equal"])
    ap.add_argument("--name", default=None, help="run and prompt name (default pitstop_<model>_<budget>)")
    ap.add_argument("--mock", action="store_true", help="mock supervisor and reflection (pipeline check only)")
    args = ap.parse_args(argv)

    s = Settings.from_env()
    model = args.model or s.supervisor_model
    short = "nano" if "nano" in model.lower() else "super" if "super" in model.lower() else model.split("/")[-1].lower()
    name = args.name or f"pitstop_{short}_{args.max_metric_calls}"
    train = load_hard_cases("train")
    val = [c for c in load_hard_cases("val") for _ in range(args.val_repeats)]
    print(f"PitStop hard pairs: train={len(train)} val={len(val)} ({args.val_repeats} repeats); test not loaded")

    if args.mock:
        print("MODE: MOCK. Supervisor and reflection are stubs. Results prove plumbing only.")
        sup_client, reflection = MockModelClient(model="mock-model"), mock_reflection
    else:
        if not s.has_key:
            print("CRUSOE_API_KEY not set; export it or use --mock.")
            return 2
        # thinking=None and no JSON mode: the same request body PitStop's own LLMClient sends.
        sup_client = CrusoeClient(model=model, api_key=s.crusoe_api_key, base_url=s.base_url,
                                  max_concurrent=args.max_workers, max_retries=8, thinking=None)
        reflection = CrusoeReflectionLM(CrusoeClient.from_settings(s, "reflection", max_concurrent=1, thinking=None))
        print(f"MODE: REAL. supervisor={model} reflection={reflection.client.model}")

    seed = Policy.from_text(SEED_PROMPT.read_text().strip(), name="supervisor_v0")  # PitStop strips its prompt file
    cfg = OptimizeConfig(max_metric_calls=args.max_metric_calls, reflection_minibatch_size=args.minibatch,
                         seed=args.gepa_seed, max_workers=args.max_workers, run_dir=f"runs/gepa/{name}",
                         acceptance_criterion=args.acceptance, perfect_score=1.0)
    art = run_optimization(seed, train, val, sup_client, reflection, cfg,
                           evaluator=make_pitstop_evaluator(sup_client), objective=OBJECTIVE, background=BACKGROUND)
    result = json.loads(Path(art.result_path).read_text())
    print(f"GEPA finished: candidates={art.num_candidates} metric_calls={art.total_metric_calls} "
          f"seed_val={art.seed_val_score:.3f} best_val={art.best_val_score:.3f} run_dir={art.run_dir}")
    print("val score per candidate:", [round(x, 3) for x in result.get("val_aggregate_scores", [])])

    meta = {"name": name, "mocked": bool(args.mock), "supervisor_model": model, "run_dir": art.run_dir,
            "seed_prompt": "src/pitstop/assets/prompts/supervisor_v0.txt", "seed_version": seed.version,
            "best_version": art.best_policy.version, "seed_val_score": art.seed_val_score,
            "best_val_score": art.best_val_score, "num_candidates": art.num_candidates,
            "total_metric_calls": art.total_metric_calls, "val_repeats": args.val_repeats,
            "optimizer": {"max_metric_calls": args.max_metric_calls, "minibatch": args.minibatch,
                          "acceptance": args.acceptance, "gepa_seed": args.gepa_seed},
            "supervisor_usage": art.supervisor_usage, "reflection_usage": art.reflection_usage}
    if args.mock:
        print(f"mock run: prompt not written to {PROMPTS / 'gepa'}")
        return 0
    out = PROMPTS / "gepa"  # dev-only write into the source tree; the demo reads from here
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.txt").write_text(art.best_policy.text.strip() + "\n", encoding="utf-8")
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"saved {out / (name + '.txt')} (version {art.best_policy.version}; "
          f"{'unchanged seed' if art.best_policy.version == seed.version else 'new prompt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
