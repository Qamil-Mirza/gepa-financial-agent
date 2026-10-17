# gepa-financial-agent

Built at the **Crusoe x Cerebral Valley Recursive Agents Hackathon** (Oct 17, 2026, SF).

A multi-agent financial research swarm on Crusoe Managed Inference, watched by a task-scoped drift supervisor whose system prompt is optimized with [GEPA](https://github.com/gepa-ai/gepa). Three workstreams, defined in `docs/PRD.md`: (1) the agent stack, (2) synthetic labeled drift traces, (3) the supervisor and GEPA pipeline.

## Team
| Name | GitHub | Focus |
|---|---|---|
| Qamil Mirza | @Qamil-Mirza | supervisor + GEPA pipeline (workstream 3) |
| Arjun Mittal | @Arjunmittal16 | agent stack (workstream 1) |
| Gautam | @gautam-bh | synthetic drift dataset (workstream 2) |

## Layout (and who owns what)
```
main.py, agent/, data/sources.py, scripts/check_crusoe.py   agent stack (workstream 1)
synthetic-data/                                             drift-trace generator + dataset (workstream 2); start at synthetic-data/HANDOFF.md
gepa_financial_agent/, policies/, scripts/*.py, tests/      drift supervisor + GEPA pipeline (workstream 3)
data/fixtures/                                              deterministic dev fixtures for workstream 3 (plumbing/tests only)
docs/                                                       PRD, contracts, architecture, progress log
```
Contracts between workstreams: `docs/INTEGRATION_CONTRACT.md` (agent stack -> supervisor) and `docs/DATASET_CONTRACT.md` / `synthetic-data/DATASET_FORMAT.md` (dataset -> evaluator).

---

# Workstream 3: drift supervisor + GEPA pipeline


## Quickstart

```bash
uv sync
cp .env.example .env          # paste CRUSOE_API_KEY in single quotes (keys contain '$')
uv run pytest
uv run python scripts/smoke_test.py --supervise      # one REAL drift judgment (add --mock to run without a key)
```

## Scripts

| script | usage |
|---|---|
| `scripts/smoke_test.py` | `--list-models` checks Crusoe connectivity and supported parameters; `--supervise [EXAMPLE_ID]` runs one judgment |
| `scripts/evaluate_supervisor.py` | `--split val --heuristic [--policy NAME ...] [--dataset PATH ...]` compares policies on a split |
| `scripts/optimize_supervisor.py` | `--max-metric-calls 60 --max-workers 4 [--final-test] [--promote]` runs real GEPA, then compares baseline vs optimized on val (test only with `--final-test`) |
| `scripts/summarize_repeats.py` | `--dataset PATH --split test --runs 'runs/eval/team_final_r*' --out FILE.json` aggregates repeated evaluations into mean/min/max per policy (no model calls) |
| `scripts/demo_supervision.py` | `--delay 0.2 [--mock]` shows three replayed agents supervised concurrently, using baseline and current policies |

Add `--mock` to any script to use the deterministic stub client. Mock output is always labeled `MOCK`.

## Plugging in the teammate dataset

1. Put JSONL files that follow `docs/DATASET_CONTRACT.md` in `data/synthetic/<name>.jsonl`.
2. Validate them:
   ```bash
   uv run python -c "from gepa_financial_agent.dataset import load_jsonl; print(load_jsonl('data/synthetic/<name>.jsonl').summary())"
   ```
3. Pass `--dataset data/synthetic/<name>.jsonl` (repeatable) to `evaluate_supervisor.py` or `optimize_supervisor.py`. The optimizer code doesn't need to change.

## Attaching to the swarm

See `docs/INTEGRATION_CONTRACT.md`. In short: `SupervisorManager.register_agent(...)`, then `observe(agent_id, event)` per event, then `evaluate_pending()`, or push events through `EventStreamAdapter`. The architecture is described in `docs/ARCHITECTURE.md`.

## Results (real Crusoe inference, teammate dataset)

Full writeup: `docs/RESULTS.md`. Raw per-repeat numbers: `docs/results/final_test_20261017.json`.

Setup: supervisor `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`, reflection `deepseek-ai/DeepSeek-V4-Pro`, dataset `synthetic-data/data/dataset.jsonl` (179 train / 54 val / 82 test, task families disjoint across splits). Test is the mean and spread over 5 repeats:

| policy | test macro-F1 mean (min–max) | FPR | undetected drift scenarios | detection delay |
|---|---|---|---|---|
| baseline (hand-written, **current**) | 0.549 (0.519–0.588) | 0.147 | 0.4 (0–2) | 0.40 |
| gepa_team_300 (evaluated, not promoted) | 0.553 (0.532–0.581) | 0.141 | 2.2 (2–3) | 1.04 |

- **GEPA works end to end, but its effect is modest or unproven at this data size.** The ranges overlap, and the GEPA policy misses more drift scenarios, so the promotion rule fails and `policies/current.txt` stays the baseline.
- **It specialized in GOAL_DEVIATION.** On test it scores 70/70 on GOAL_DEVIATION (baseline 62/70) but 1/30 on CONSTRAINT_VIOLATION (baseline 15/30). The other drift types also drop: OBSOLETE_DIRECTIVE 12/30 vs 24/30, UNSUPPORTED_CONCLUSION 4/30 vs 19/30.
- **Val could not separate the policies.** The val gains of the three GEPA runs (0.007 to 0.018 macro-F1) are smaller than the baseline's own val spread across repeats (0.515 to 0.549).

Earlier results on our dev fixtures (a 0.59 → 0.77 test gain) are void: the fixture event IDs leaked the label. See `PROGRESS.md` and the manifest.

## Mock vs real

`MockModelClient` is a deterministic stub. It proves the pipeline is wired correctly, not that GEPA works. Every record carries `mocked`, every script prints `MODE: MOCK` or `MODE: REAL`, and mock policies are never promoted. All numbers above come from real Crusoe inference.

---

# Workstreams 1–2: agent stack and synthetic data

## Agent stack quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Crusoe key into .env
python scripts/check_crusoe.py
python main.py "your goal here"
```

Planner breaks the goal into subtasks -> workers (sub-agents) do each one -> checker grades
each result and sends failures back with feedback -> synthesizer writes the final answer.
Every run is saved to `runs/`, and past checker feedback is fed to the planner next time.

| Role | Model |
|---|---|
| Planner, checker, synthesizer | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Workers | `nvidia/Nemotron-3.5-Lightning-30B-A3B` |

## Synthetic drift dataset
`synthetic-data/` holds the trace generator (real Nemotron workers, real You.com Finance Research answers, private steers, deterministic verifier) and the exported dataset `synthetic-data/data/{train,val,test}.jsonl`. Read `synthetic-data/HANDOFF.md` first; the field contract is `synthetic-data/DATASET_FORMAT.md`.

## Crusoe notes
- Endpoint: `https://api.inference.crusoecloud.com/v1` (OpenAI-compatible)
- Rate limits are per model: 30 RPM without a payment method. Spread agent roles across models.
- See `docs/skills/` for model IDs, reasoning toggles, and troubleshooting. Exact model IDs are case-sensitive; `GET /v1/models` is authoritative.
