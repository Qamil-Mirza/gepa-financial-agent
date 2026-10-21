"""Summarize the trace DB and exported dataset: balance, discard rate, query breadth, tokens and cost."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import config
from synth.store import TraceStore
from synth.textmatch import ENTITIES, tokens
from synth.verify import verify

# Nemotron 3.5 Lightning list price (USD per 1M tokens) and You.com 'deep' price per request.
PRICE_IN, PRICE_OUT, YOUCOM_PER_CALL = 0.05, 0.20, 0.11


def is_broad(query: str) -> bool:
    """A query with no company named, or fewer than 4 content words, is too broad to localize drift."""
    toks = tokens(query)
    return not (toks & ENTITIES) and len(toks) < 6 or len(toks) < 4


def main() -> None:
    traces = TraceStore().all()
    by = Counter()
    fails = Counter()
    pt = ct = live = queries = broad = 0
    for t in traces:
        v = verify(t)
        key = (t.scenario_family, t.generation.steer.steer_id.split(":")[0] + ":" + (t.generation.steer.drift_type or "-"))
        by[(key, v.passed)] += 1
        if not v.passed:
            fails[v.reason.split(":")[0]] += 1
        u = t.generation.usage
        pt += u.prompt_tokens
        ct += u.completion_tokens
        live += u.youcom_live_calls
        for e in t.events:
            if e.event_type == "tool_call" and e.tool_name == "finance_research":
                queries += 1
                broad += is_broad(str(e.content))

    n_pass = sum(c for (k, p), c in by.items() if p)
    print(f"traces: {len(traces)}  verified: {n_pass}  discard rate: {1 - n_pass / max(1, len(traces)):.0%}")
    kinds = Counter()
    for (k, p), c in by.items():
        if p:
            kinds[k[1]] += c
    print("verified by steer:", dict(sorted(kinds.items())))
    fams = Counter()
    for (k, p), c in by.items():
        if p:
            fams[k[0]] += c
    print("verified by family:", dict(sorted(fams.items())))
    print("top discard reasons:", fails.most_common(6))
    print(f"research queries: {queries}, broad: {broad} ({broad / max(1, queries):.0%})")
    cost = pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT
    print(f"crusoe tokens: {pt:,} in / {ct:,} out (~${cost:.2f}); you.com live calls in rollouts: {live} (~${live * YOUCOM_PER_CALL:.2f})")
    print(f"you.com cache entries: {len(list(config.YOUCOM_CACHE_DIR.glob('*.json')))}")

    ds = config.DATA_DIR / "dataset.jsonl"
    if ds.exists():
        rows = [json.loads(l) for l in ds.read_text().splitlines() if l.strip()]
        lab = Counter((r["split"], r["ground_truth"]["status"]) for r in rows)
        print("examples by split/status:", dict(sorted(lab.items())))
        dt = Counter(r["ground_truth"]["drift_type"] for r in rows if r["ground_truth"]["drift_type"])
        print("DRIFTING examples by type:", dict(sorted(dt.items())))


if __name__ == "__main__":
    main()
