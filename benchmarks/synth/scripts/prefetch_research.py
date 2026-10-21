"""Fill the You.com research cache with every task's on-task, bait and detour queries.

Each uncached query costs one live 'deep' request (~$0.11). Already-cached queries are skipped.
Usage: python scripts/prefetch_research.py [--family nvda_datacenter] [--max-calls 45]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import config
from synth.research_bank import ResearchBank
from synth.tasks import all_prefetch_queries
from synth.textmatch import normalize_query


def import_sample(bank_dir: Path) -> None:
    """Reuse the response saved by check_youcom.py instead of paying for it twice."""
    sample = config.EVIDENCE_DIR / "youcom_sample_response.json"
    if not sample.exists():
        return
    d = json.loads(sample.read_text())
    import hashlib
    norm = normalize_query(d["query"])
    path = bank_dir / (hashlib.sha1(norm.encode()).hexdigest()[:16] + ".json")
    if not path.exists():
        path.write_text(json.dumps({"query": d["query"], "normalized": norm, "fetched_at": None,
                                    "elapsed_s": d["elapsed_s"], "research_effort": "deep",
                                    "response": d["response"]}, indent=2))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append")
    ap.add_argument("--max-calls", type=int, default=45)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import_sample(config.YOUCOM_CACHE_DIR)
    bank = ResearchBank(live_budget=args.max_calls, max_concurrent=args.concurrency)
    queries = [q for f, qs in all_prefetch_queries().items() if not args.family or f in args.family for q in qs]
    print(f"{len(queries)} queries, {len(bank)} cached entries before prefetch")

    async def one(q):
        try:
            return await bank.prefetch(q)
        except Exception as exc:  # keep going; report at the end
            logging.error("failed: %s (%s)", q, exc)
            return None

    results = await asyncio.gather(*(one(q) for q in queries))
    print(f"live calls: {sum(r is True for r in results)}, skipped (cached): {sum(r is False for r in results)}, "
          f"failed: {sum(r is None for r in results)}, cache size now {len(ResearchBank())}")


if __name__ == "__main__":
    asyncio.run(main())
