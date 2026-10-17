"""Run real worker rollouts on Crusoe with steers, verify them, and store traces.

Examples:
  python scripts/generate.py --family nvda_datacenter --kind drift --drift-type GOAL_DEVIATION --limit 1
  python scripts/generate.py --repeats 1               # full matrix: every family x steer
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import steers
from synth.crusoe import CallBudgetExceeded, CrusoeClient
from synth.research_bank import ResearchBank
from synth.rollout import run_rollout
from synth.schemas import SteerSpec
from synth.store import TraceStore
from synth.tasks import TASKS
from synth.verify import verify

log = logging.getLogger("generate")


def build_matrix(families: List[str], kinds: List[str], drift_types: List[str], repeats: int,
                 rng: random.Random) -> List[tuple]:
    """Per family: every drift type (each GOAL_DEVIATION distractor), every benign variant, and controls."""
    runs = []
    for _ in range(repeats):
        for fam in families:
            task = TASKS[fam]
            if "drift" in kinds:
                for t in drift_types:
                    for v in steers.variants_for(task, "drift", t):
                        runs.append((fam, SteerSpec(steer_id=steers.steer_id("drift", t, v), kind="drift",
                                                    drift_type=t, variant=v, step_x=rng.randint(3, 6))))
            if "benign" in kinds:
                for v in steers.variants_for(task, "benign"):
                    runs.append((fam, SteerSpec(steer_id=steers.steer_id("benign", None, v), kind="benign",
                                                variant=v, step_x=rng.randint(3, 6))))
            if "none" in kinds:
                for _ in range(2):
                    runs.append((fam, SteerSpec(steer_id="none", kind="none")))
    return runs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", choices=sorted(TASKS))
    ap.add_argument("--kind", action="append", choices=["drift", "benign", "none"])
    ap.add_argument("--drift-type", action="append", choices=steers.DRIFT_TYPES)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, help="run at most this many rollouts")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-crusoe-calls", type=int, default=1500)
    ap.add_argument("--max-youcom-calls", type=int, default=30, help="live You.com calls beyond the cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rng = random.Random(args.seed)
    runs = build_matrix(args.family or sorted(TASKS), args.kind or ["drift", "benign", "none"],
                        args.drift_type or steers.DRIFT_TYPES, args.repeats, rng)
    rng.shuffle(runs)
    if args.limit:
        runs = runs[: args.limit]

    client = CrusoeClient(max_concurrent=args.concurrency, max_total_calls=args.max_crusoe_calls)
    bank = ResearchBank(live_budget=args.max_youcom_calls)
    store = TraceStore()
    log.info("%d rollouts, %d cached research entries, model %s", len(runs), len(bank), client.model)
    stats = {"passed": 0, "failed": 0, "errors": 0}
    rollout_sem = asyncio.Semaphore(args.concurrency)

    async def one(i: int, fam: str, steer: SteerSpec):
        async with rollout_sem:
            try:
                trace = await run_rollout(client, bank, TASKS[fam], steer, max_steps=args.max_steps,
                                          seed=args.seed * 100000 + i)
            except CallBudgetExceeded as exc:
                log.warning("stopping: %s", exc)
                stats["errors"] += 1
                return
            except Exception:
                log.exception("rollout failed: %s %s", fam, steer.steer_id)
                stats["errors"] += 1
                return
            trace.generation.verification = verify(trace)
            store.save(trace)
            v = trace.generation.verification
            stats["passed" if v.passed else "failed"] += 1
            log.info("%s %-22s %-40s x=%s -> %s (%s)", trace.trace_id, fam, steer.steer_id, steer.step_x,
                     "PASS" if v.passed else "FAIL", v.reason)

    await asyncio.gather(*(one(i, f, s) for i, (f, s) in enumerate(runs)))
    log.info("done: %s | crusoe calls %d | you.com live calls %d", stats, client.total_calls, bank.live_calls)


if __name__ == "__main__":
    asyncio.run(main())
