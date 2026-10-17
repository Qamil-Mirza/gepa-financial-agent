# Synthetic drift traces

Labeled agent trajectories for optimizing the drift supervisor with GEPA. Real Nemotron worker
agents (Crusoe Managed Inference) research financial tasks using real You.com Finance Research
answers. At a chosen step the harness privately steers them into drift, a benign detour, or
nothing, so the label and onset step are known by construction. The data format is in
[DATASET_FORMAT.md](DATASET_FORMAT.md).

## What is real and what is synthetic

| Part | Source |
|---|---|
| Worker actions (queries, findings, summaries) | Real `nvidia/Nemotron-3.5-Lightning-30B-A3B` output |
| Research results | Real You.com Finance Research responses, cached in `evidence/youcom_cache/` |
| Drift | Induced by a private steer at step x (the steer itself is never exported) |
| Labels | From the steer plus a deterministic verifier of the agent's actual behavior |

## Setup

```bash
cd financial-agent/synthetic-data
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill CRUSOE_API_KEY and YDC_API_KEY (single-quote values)
```

## Pipeline

```bash
.venv/bin/python scripts/check_crusoe.py            # confirm model + tool calling (1 call)
.venv/bin/python scripts/check_youcom.py            # one real You.com request (~$0.11)
.venv/bin/python scripts/prefetch_research.py       # cache on-task / bait / detour answers (~45 calls)
.venv/bin/python scripts/generate.py --family nvda_datacenter --limit 1   # one rollout
.venv/bin/python scripts/generate.py                # full matrix: 12 runs x 11 families
.venv/bin/python scripts/export_dataset.py          # traces -> data/{train,val,test}.jsonl
.venv/bin/python scripts/stats.py                   # balance, discard rate, cost
.venv/bin/python -m pytest                          # tests (no network)
```

Useful `generate.py` flags: `--kind drift|benign|none`, `--drift-type GOAL_DEVIATION`,
`--repeats N`, `--seed`, `--max-crusoe-calls`, and `--max-youcom-calls` (live You.com calls
beyond the cache; default 30). Generation adds to `traces/`, and export re-verifies everything.

## Costs

- **You.com:** `deep` research is about $0.11 per request. Repeated questions come from the cache. A query is only reused when it asks the same question (same company, period and metric), so narrow queries cause new live calls up to `--max-youcom-calls`. After that, the closest cached answer is returned.
- **Crusoe:** about 10 calls per rollout at Lightning prices, a few cents for the whole matrix.

## Known limitations

- **Ignored steers:** the worker model sometimes ignores steers (most often loops), and those runs fail verification and are dropped. Stats shows the discard rate.
- **No forced tool choice:** Crusoe returns HTTP 500 for a forced `tool_choice` on this model. On the steer turn the worker is offered only the target tool instead.
- **Agents rarely finish:** they often skip `finish` and run to the step limit, which makes later steps more prone to natural drift. Natural drift in healthy runs causes discards, not label noise.
- **Outdated traces:** `traces_archive/` holds traces from earlier generator versions and is not exported.
