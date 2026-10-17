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
pitstop/pitstop/                                            PitStop swarm + supervisor, hard-pairs eval, GEPA demo UI (pitstop/demo/)
data/fixtures/                                              deterministic dev fixtures for workstream 3 (plumbing/tests only)
docs/                                                       PRD, contracts, architecture, progress log
```
Contracts between workstreams: `docs/INTEGRATION_CONTRACT.md` (agent stack -> supervisor) and `docs/DATASET_CONTRACT.md` / `synthetic-data/DATASET_FORMAT.md` (dataset -> evaluator).

## Demo: what GEPA's prompt rewrite does

A click-through page (`pitstop/pitstop/pitstop/demo/`) for a one-minute demo. Two Nemotron Nano supervisors use the same model with different prompts: the original prompt, and the prompt GEPA rewrote. Each one runs and corrects its own three research agents on the same task. Design: `docs/superpowers/specs/2026-10-17-gepa-demo-ui-design.md`.

```bash
cd pitstop/pitstop
uv run --project ../.. --with httpx --with rich python -m pitstop.demo.server --port 8779
# demo mode (replays a committed real recording): http://127.0.0.1:8779
# live mode (real Crusoe + You.com run):          http://127.0.0.1:8779/?mode=live
```

| Screen | Shows |
|---|---|
| 1. Start | Chat box; three agents spawn per side |
| 2. Watch | Original prompt (left) vs GEPA prompt (right): a verdict per step, the injected-drift marker, "caught at step N", ↩ re-plans after a correction, extra alarms, and trace events flowing into each supervisor. Click any cell to see why. |
| 3. Team reports | Each side's final financial report, written only from findings its supervisor didn't flag |
| 4. Prompt evolution | The real GEPA run (`pitstop_nano_800`): best val score 0.800 → 0.854 over 15 loops, with the prompt growing from 1.6k to 7.8k characters |
| 5. Scoreboard | Held-out test accuracy on the PitStop hard pairs (below) |

Keys: → / Space next, ← back, Home / Esc start, R replay, L live. The page is sized for a projector.

**PitStop hard pairs, test split** (AMD cases, never seen by GEPA; 180 judgments = 36 cases × 5 trials per arm; fires = DRIFTING at confidence ≥ 0.75):

| Supervisor | Accuracy | Drift caught | False alarms |
|---|---|---|---|
| Nano, original prompt | 81.7% | 66.7% | 3 |
| **Nano, GEPA prompt** | **87.8%** | **75.6%** | **0** |
| Super (120B), original prompt | 86.1% | 78.9% | 6 |

A 30B model with a rewritten prompt matches the 120B model. The 1.7-point gap over Super is within the noise of 180 judgments.

**Live mode notes:**
- **Keys:** live mode needs `CRUSOE_API_KEY` and `YDC_API_KEY` in `.env`.
- **Time:** a paired run takes about 4–6 minutes; the catch and correction land about 1.5 minutes in.
- **Cost:** up to about $3 of You.com calls with a cold cache. Identical requests from the two sides are de-duplicated, and the agents run at temperature 0, so a rehearsal warms the cache.
- **Rate limits:** stop other jobs on the same Crusoe account first, because concurrent GEPA/eval runs cause 429s.
- **Re-recording demo mode:** run a live pair, then `python -m pitstop.demo.export --swarm-run runs/pair-<id>`.

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

- **On the smaller Nemotron Nano supervisor, the result is the same.** GEPA's +0.075 val gain did not transfer: test macro-F1 is 0.378 (0.356–0.410) against 0.367 (0.360–0.379) for baseline. Its one consistent effect is zero undetected drift, bought with more false alarms. It is not promoted.

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
