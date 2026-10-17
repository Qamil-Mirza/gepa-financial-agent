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

## Results (real Crusoe inference, dev fixtures)

**Headline, test split (n=25), repeated 3 times:**

| policy | test macro-F1, median (min–max) | FPR | undetected drift |
|---|---|---|---|
| baseline (hand-written) | 0.594 (0.594–0.594) | 0.0 in all runs | 0 |
| GEPA (promoted) | 0.731 (0.648–0.810) | 0.0 in all runs | 0 |

Every one of 5 GEPA repeats beat every baseline run. The gain is robust in direction but uncertain in size; the 5 runs are 3 unseeded plus 2 with `seed=42`, and the raw data is in `docs/results/variance_test_20261017.json`.

The trade is detection delay. With the GEPA policy, detection came 0.4 to 2.0 steps after onset, against 0.0 for baseline, with 0 undetected drift scenarios in every run.

Inference is not deterministic even at temperature 0, and Crusoe does not honor `seed`, so the headline table reports the median of 3 unseeded repeats. The GEPA policy's spread is much wider than baseline's.

**Single run, n=25 test:** the run that produced the promoted policy is `runs/gepa/20261017_122148`, with a 60-call budget, seed 0, supervisor `deepseek-ai/Deepseek-V4-Flash` and reflection `deepseek-ai/DeepSeek-V4-Pro`. The table is copied from its `comparison.txt`:

```
label                  policy     mode  n    macroF1  acc    P_drift  R_drift  FPR    FNR    invalid  evid_ok  score  ms      usd
baseline/val           eac5a206b5 REAL  19   0.541    0.789  0.500    1.000    0.083  0.000  0.000    1.000    0.574  1060    0.0017
gepa/val               efded549eb REAL  19   0.689    0.842  0.600    0.750    0.000  0.250  0.000    1.000    0.618  1191    0.0019
baseline/test          eac5a206b5 REAL  25   0.592    0.800  0.692    1.000    0.000  0.000  0.000    1.000    0.656  1142    0.0046
gepa/test              efded549eb REAL  25   0.767    0.840  0.727    0.889    0.000  0.111  0.000    1.000    0.686  1212    0.0031
```

| | mean detection delay (test, this run) | delay across the 3 repeats | undetected drift scenarios (test) |
|---|---|---|---|
| baseline | 0.2 steps | 0.0, 0.0, 0.0 | 0 |
| gepa | 1.2 steps | 2.0, 2.0, 0.4 | 0 |

How to read these numbers:

- **Detection is later, not missed.** The GEPA policy learned to answer UNCERTAIN at drift onset and DRIFTING only once there is more evidence. That explains the higher detection delay and the R_drift drop from 1.000 to 0.889.
- **The sets are small.** These are dev fixtures generated by `scripts/make_dev_fixtures.py`, with 19 val and 25 test examples. One flipped example moves val macro-F1 by about 0.1, so these numbers do not show that the gain generalizes.
- **A longer run did not help.** The 150-call run (`runs/gepa/20261017_122338`) did not beat baseline: test macro-F1 was 0.592 against baseline's 0.594. Our explanation:
  - val noise is about as large as the improvements GEPA was trying to measure;
  - its 11 iterations chased one-example swings on the 19-example val split;
  - it spent $0.095 on reflection, against $0.017 for the 60-call run.

  See `PROGRESS.md`.

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
