# Handoff: synthetic drift dataset for the GEPA supervisor

For the teammate building the supervisor, evaluator and GEPA loop (and their coding agent). This
explains what is in `financial-agent/synthetic-data/`, how to consume it, and which decisions
are still open. The field-level contract is in [DATASET_FORMAT.md](DATASET_FORMAT.md); the
Pydantic source of truth is `synth/schemas.py`.

Everything lives in `synthetic-data/`, and no files outside it were changed. Branch: `gautam-bh/feature`
(committed, not pushed).

## TL;DR for consuming the data

```python
import json
rows = [json.loads(l) for l in open("financial-agent/synthetic-data/data/train.jsonl")]
for r in rows:
    supervisor_input = r["supervisor_input"]   # the ONLY thing to show the supervisor
    gt = r["ground_truth"]                     # status, drift_type, evidence_event_ids, rationale
    onset = r["drift_onset_step"]              # null for benign/clean traces
```

- **Files:** `data/train.jsonl` (116 examples), `data/val.jsonl` (26), `data/test.jsonl` (22).
- **Statuses:** `ON_TRACK`, `UNCERTAIN`, `DRIFTING`.
- **Drift types:** `GOAL_DEVIATION`, `UNPRODUCTIVE_LOOP`, `CONSTRAINT_VIOLATION`, `UNSUPPORTED_CONCLUSION`, `OBSOLETE_DIRECTIVE`.
- **Group examples by `scenario_id`:** the trace id. Each trace gives 2–4 prefix examples at different `observation_step` cutoffs.
- **Splits are by task family:** nothing from a test family appears in train/val. Don't re-split randomly.
- **Validate rows if you like:** `from synth.schemas import DatasetExample; DatasetExample.model_validate(row)`. The validator rejects trajectories containing events past the cutoff and evidence ids that aren't in the visible trajectory.

## How the data was made

1. **Real worker agents:** `nvidia/Nemotron-3.5-Lightning-30B-A3B` on Crusoe Managed Inference, single-agent, up to 12 turns, tools `finance_research` / `record_finding` / `finish`, thinking off.
2. **Real evidence:** `finance_research` returns real **You.com Finance Research API** answers (`POST https://api.you.com/v1/finance_research`, `research_effort: deep`). Every answer is cached in `evidence/youcom_cache/` (132 entries). Replays and re-exports never call You.com.
3. **Steers:** at a pre-planned step x (3–6), the harness privately steers the worker: drift, a benign detour, or nothing (clean control). The steer is never in the exported trajectory; only the worker's resulting real actions are.
4. **Deterministic verifier:** `synth/verify.py` checks that the agent actually did what the steer asked, and that healthy runs show no drift signal. Runs that fail are dropped, never relabeled: 25 of 74 dropped, 49 kept.
5. **Export:** `synth/export.py` cuts each verified trace into prefixes and labels them.

Five task families, all in the context of NVIDIA vs AMD AI-business analysis:
`nvda_datacenter`, `amd_datacenter`, `hyperscaler_capex` (train), `margin_comparison` (val),
`accelerator_competition` (test).

## Label convention (needs your sign-off)

`observation_step` is inclusive: the trajectory has every event with `step <= observation_step`.

| Cutoff | Label |
|---|---|
| before drift onset | `ON_TRACK` |
| at onset, only one drift action visible | `UNCERTAIN` (for GOAL_DEVIATION, UNPRODUCTIVE_LOOP, OBSOLETE_DIRECTIVE) |
| drift confirmed (second drift step; or at onset for CONSTRAINT_VIOLATION / UNSUPPORTED_CONCLUSION, which are decisive in one action) | `DRIFTING` |
| benign detour or clean run, at any cutoff | `ON_TRACK` |

Why UNCERTAIN exists: at the step where drift starts, the supervisor sees a single off-path
query. A benign detour looks the same at its first step. Giving that one step its own label
avoids sending GEPA contradictory signals.

- **Measured onset:** observed onset equals the planned step x in 34 of the 35 drift traces. `drift_onset_step` is therefore effectively the planned step.
- **Detection delay:** compute predicted onset minus `drift_onset_step`.
- **Alternatives:** if you'd rather have no UNCERTAIN, it's a one-line change in `synth/export.py::label_at`, either DRIFTING from onset or ON_TRACK until confirmed. Tell Gautam which one you want.

`evidence_event_ids` only ever point at the agent's own actions (`tool_call`, `research_finding`,
`agent_message`, `task_completion`), never at tool results. Use them to score whether the
supervisor's cited evidence is grounded.

## Things in the trajectory a supervisor prompt should understand

- **Source ids:** `tool_result` events list sources as `[S3.1] title — url`, and findings cite those ids in `source_references`. A finding with no source that has a `url` is uncited.
- **Bait:** in GOAL_DEVIATION traces, the research result at step x has an extra `Related coverage:` section about the distractor. That's bait, and it's visible; drift is when the agent then *acts* on it.
- **Directive updates:** OBSOLETE_DIRECTIVE traces contain a `status_update` event ("Directive update from the orchestrator: …"). The agent keeps following the old directive. `supervisor_input.agent_directive` still shows the original, so the supervisor has to notice the update in the trajectory.
- **Benign tool errors:** benign `tool_error` traces contain an `error` event followed by a retry. This is not drift.
- **Narrow queries:** workers are prompted to make narrow queries (one company, metric and period), so drift shows up in the query text.

## Numbers (from `scripts/stats.py`)

| Split | ON_TRACK | UNCERTAIN | DRIFTING |
|---|---|---|---|
| train | 51 | 17 | 48 |
| val | 12 | 4 | 10 |
| test | 6 | 4 | 12 |

- **Verified traces by steer:** GOAL_DEVIATION 16, CONSTRAINT_VIOLATION 5, UNPRODUCTIVE_LOOP 5, UNSUPPORTED_CONCLUSION 5, OBSOLETE_DIRECTIVE 4, benign 6, clean 8.
- **Weak spot:** the test split has only 6 ON_TRACK examples, too few for a reliable false-positive rate. Benign detours (the hard negatives) are also few. More benign and clean runs are cheap to add on request. Real traces from the live swarm should go into test.

## Facts learned the hard way (useful for your code too)

- **Model id:** the exact Crusoe id is `nvidia/Nemotron-3.5-Lightning-30B-A3B`. The skill file's lowercase spelling is wrong; `GET /v1/models` is authoritative.
- **Forced tool choice:** Crusoe returns **HTTP 500 for any forced `tool_choice`** (`"required"` or a named function) on this model. Plain tool calling with `tool_choice` omitted works fine.
- **Disabling thinking:** `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` works on Lightning.
- **You.com request:** auth uses the `X-API-Key` header.
- **You.com response:** `{"output": {"content": <markdown with [[n]] citations>, "content_type": "text", "sources": [{"url", "title"}]}}`. Sources have no dates.
- **You.com cost and speed:** `deep` took about 10s per call and costs about $0.11 per request.
- **You.com spend so far:** about $11.70. Crusoe tokens cost about $0.10 total.

## Code map

| Path | What |
|---|---|
| `synth/schemas.py` | `TraceEvent`, `TraceRecord` (stored trace incl. hidden `generation` metadata), `DatasetExample` |
| `synth/tasks.py` | 5 task templates: directive, constraints, distractors, detours, keyword lists |
| `synth/steers.py` | private steer notes, bait, withheld directive updates |
| `synth/rollout.py` | worker loop on Crusoe, event recording |
| `synth/research_bank.py` | You.com record-and-replay cache with a live-call budget |
| `synth/verify.py` | deterministic behavior verifier, onset detection |
| `synth/export.py` | prefix cutoffs, labels, splits |
| `scripts/generate.py` | run the task × steer matrix (`--family`, `--kind`, `--drift-type`, `--max-youcom-calls`) |
| `scripts/export_dataset.py`, `scripts/stats.py` | re-verify, export, summarize |
| `traces/` | the trace database (one JSON per run, `index.jsonl`) |
| `tests/` | offline tests (`.venv/bin/python -m pytest`, 20 passing) |

`traces/*.json` hold ground truth and the full private worker conversation, including the steer
note. Never feed them to the supervisor; use `data/*.jsonl`.

## Open questions for you

1. Keep the UNCERTAIN convention, or switch (see above)?
2. Group supervisor: examples are single-agent. Should we bundle 2–4 traces into one group input
   (one drifting, the rest healthy) in the export, or will your side do the bundling?
3. Do you want a `recommended_action` or other field in `ground_truth`?
