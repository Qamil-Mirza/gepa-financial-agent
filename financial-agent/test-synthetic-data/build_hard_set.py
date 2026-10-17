"""Build the hard held-out test set with the same pipeline as synthetic-data/ (imported, not copied).

  python build_hard_set.py prefetch   # cache You.com answers for on-task, bait and detour queries
  python build_hard_set.py generate   # real Nemotron rollouts on Crusoe, verified, stored in traces/
  python build_hard_set.py export     # re-verify and write data/test.jsonl
  python build_hard_set.py stats

Run with the synthetic-data venv:  ../synthetic-data/.venv/bin/python build_hard_set.py <cmd>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synthetic-data"))
sys.path.insert(0, str(HERE))

from hard_tasks import HARD_TASKS  # noqa: E402
from synth import steers  # noqa: E402
from synth.crusoe import CallBudgetExceeded, CrusoeClient  # noqa: E402
from synth.export import examples_for  # noqa: E402
from synth.research_bank import ResearchBank  # noqa: E402
from synth.rollout import run_rollout  # noqa: E402
from synth.schemas import SteerSpec  # noqa: E402
from synth.store import TraceStore  # noqa: E402
from synth.tasks import TASKS  # noqa: E402
from synth.verify import verify  # noqa: E402

assert not set(HARD_TASKS) & set(TASKS), "hard-set families must not overlap training families"
TASKS.update(HARD_TASKS)  # the verifier looks tasks up by family

CACHE_DIR = HERE / "evidence" / "youcom_cache"
TRACES_DIR = HERE / "traces"
DATA_DIR = HERE / "data"
X_RANGE = (5, 8)        # later onsets than training (3-6): longer clean history before drift
MAX_STEPS = 14
FINISH_FROM_STEP = 11   # finish is hidden until here for every run, so length doesn't reveal the steer
log = logging.getLogger("hard")


def build_matrix(rng: random.Random) -> list:
    """Per family: 7 drift runs, 6 benign (incl. 3 ignored-bait hard negatives), 2 clean."""
    runs = []
    for fam, task in sorted(HARD_TASKS.items()):
        x = lambda: rng.randint(*X_RANGE)  # noqa: E731
        for t in steers.DRIFT_TYPES:
            for v in steers.variants_for(task, "drift", t):
                runs.append((fam, SteerSpec(steer_id=steers.steer_id("drift", t, v), kind="drift",
                                            drift_type=t, variant=v, step_x=x())))
        benign = [d.name for d in task.detours] + ["tool_error"] + \
                 [steers.IGNORED_BAIT + d.name for d in task.distractors]
        for v in benign:
            runs.append((fam, SteerSpec(steer_id=steers.steer_id("benign", None, v), kind="benign",
                                        variant=v, step_x=x())))
        for _ in range(2):
            runs.append((fam, SteerSpec(steer_id="none", kind="none")))
    return runs


async def prefetch(max_calls: int) -> None:
    bank = ResearchBank(cache_dir=CACHE_DIR, live_budget=max_calls, max_concurrent=3)
    queries = [q for t in HARD_TASKS.values()
               for q in t.on_task_queries + [d.bait_query for d in t.distractors] + [d.query for d in t.detours]]
    results = await asyncio.gather(*(bank.prefetch(q) for q in queries), return_exceptions=True)
    failed = [r for r in results if isinstance(r, Exception)]
    print(f"{len(queries)} queries, live calls {bank.live_calls}, failed {len(failed)}, cache {len(ResearchBank(cache_dir=CACHE_DIR))}")


async def generate(args) -> None:
    rng = random.Random(args.seed)
    runs = build_matrix(rng)
    rng.shuffle(runs)
    if args.limit:
        runs = runs[: args.limit]
    client = CrusoeClient(max_concurrent=args.concurrency, max_total_calls=args.max_crusoe_calls)
    bank = ResearchBank(cache_dir=CACHE_DIR, live_budget=args.max_youcom_calls)
    store = TraceStore(TRACES_DIR)
    sem = asyncio.Semaphore(args.concurrency)
    stats = Counter()
    log.info("%d rollouts, %d cached answers, model %s", len(runs), len(bank), client.model)

    async def one(i, fam, steer):
        async with sem:
            try:
                trace = await run_rollout(client, bank, TASKS[fam], steer, max_steps=MAX_STEPS,
                                          seed=args.seed * 100000 + i, finish_from_step=FINISH_FROM_STEP)
            except CallBudgetExceeded as exc:
                log.warning("stopping: %s", exc)
                stats["errors"] += 1
                return
            except Exception:
                log.exception("rollout failed: %s %s", fam, steer.steer_id)
                stats["errors"] += 1
                return
            trace.generation.verification = v = verify(trace)
            store.save(trace)
            stats["passed" if v.passed else "failed"] += 1
            log.info("%s %-20s %-36s x=%s -> %s (%s)", trace.trace_id, fam, steer.steer_id, steer.step_x,
                     "PASS" if v.passed else "FAIL", v.reason)

    await asyncio.gather(*(one(i, f, s) for i, (f, s) in enumerate(runs)))
    log.info("done: %s | crusoe calls %d | you.com live calls %d", dict(stats), client.total_calls, bank.live_calls)


def export() -> None:
    traces = TraceStore(TRACES_DIR).all()
    DATA_DIR.mkdir(exist_ok=True)
    lines = []
    for t in traces:
        t.generation.verification = verify(t)
        if t.generation.verification.passed:
            lines += [e.model_dump_json() for e in examples_for(t, "test")]
    (DATA_DIR / "test.jsonl").write_text("".join(l + "\n" for l in lines))
    print(f"{sum(t.generation.verification.passed for t in traces)}/{len(traces)} traces verified; "
          f"{len(lines)} examples -> {DATA_DIR / 'test.jsonl'}")


def stats() -> None:
    traces = TraceStore(TRACES_DIR).all()
    kinds, fails = Counter(), Counter()
    for t in traces:
        v = verify(t)
        s = t.generation.steer
        k = "benign:ignored_bait" if steers.is_ignored_bait(s) else f"{s.kind}:{s.drift_type or s.variant or '-'}"
        if s.kind == "benign" and not steers.is_ignored_bait(s):
            k = "benign:detour_or_error"
        (kinds if v.passed else fails)[k if v.passed else v.reason.split(":")[0]] += 1
    print(f"traces {len(traces)}, verified {sum(kinds.values())}")
    print("verified by steer:", dict(sorted(kinds.items())))
    print("discard reasons:", fails.most_common(6))
    p = DATA_DIR / "test.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        print("examples by status:", dict(Counter(r["ground_truth"]["status"] for r in rows)))
        print("examples by family:", dict(Counter(r["scenario_family"] for r in rows)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefetch", "generate", "export", "stats"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-crusoe-calls", type=int, default=1500)
    ap.add_argument("--max-youcom-calls", type=int, default=50)
    ap.add_argument("--max-prefetch-calls", type=int, default=45)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cmd == "prefetch":
        asyncio.run(prefetch(args.max_prefetch_calls))
    elif args.cmd == "generate":
        asyncio.run(generate(args))
    elif args.cmd == "export":
        export()
    else:
        stats()


if __name__ == "__main__":
    main()
