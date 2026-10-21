# PitStop: drift-detecting supervisor for a financial research swarm

Three research agents work on one objective using the You.com Finance Research API.
A Nemotron supervisor on Crusoe Managed Inference reviews each agent at every step and
flags drift with evidence from the agent's observable trajectory. It can correct a
drifting agent, and a report agent writes the team's final report. GEPA only swaps the prompt
text: `prompts/supervisor_v0.txt` is the hand-written baseline, and
`prompts/gepa/pitstop_nano_800.txt` is the GEPA-optimized Nano prompt
(`scripts/optimize_pitstop.py` at the repo root).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add CRUSOE_API_KEY and YDC_API_KEY
python -m pitstop.cli check   # verifies a real Crusoe inference call for both models
python -m pitstop.cli check --youcom   # optional: one paid You.com call (~$0.11)
python -m pitstop.cli run --halt-on-drift
python -m pitstop.cli run --correct-on-drift               # block a drifting query and steer the agent back
python -m pitstop.cli run --shadow-prompt prompts/gepa/pitstop_nano_800.txt   # a 2nd supervisor judges the same inputs
python -m pitstop.cli demo --port 8779                     # GEPA demo page (see "GEPA demo UI" below)
python -m pytest              # offline tests: scripted LLM + mocked You.com, no spend
```

From the repo root's uv environment, prefix commands with
`uv run --project ../.. --with httpx --with rich`.

A run writes `runs/<run_id>/` containing `events.jsonl` (every trajectory event),
`decisions.jsonl` (every supervisor input + validated verdict) and `meta.json`
(ground truth, interventions, token usage, You.com spend).

It also writes a curated **timeline** for the UI: `timeline.jsonl` (appended live, one entry per
line, safe to tail) and `timeline.json` (run info, per-agent summaries with a compact verdict
history, and all entries). Only notable things become entries: `drift_flagged` (critical),
`drift_response` (halted / recovered / persisted, linked to its flag by `responds_to`),
`drift_suspected`, `status_changed`, `supervisor_correction`, plus run and agent lifecycle.
Each entry's `source` says who emitted it (`role`: `system` / `supervisor` / `validator` / `agent`,
plus `id` and display `label`), while `agent_id` / `agent_label` say which agent it is about. The run's
`agents` list gives each agent a display `label` (set `label` in the scenario, or it is derived from
the id: `agent_nvda_dc` → "NVDA DC") and a stable `color_index` for the UI.
Routine ON_TRACK verdicts are folded into the next entry's `quiet_streak`. Each entry's `refs`
points back into `decisions.jsonl` / `events.jsonl` for full detail. Rebuild it for any old run with
`python -m pitstop.cli timeline --run-dir runs/<run_id> [--highlights]`.

The UI reads timelines through a small stdlib HTTP server (`python -m pitstop.cli serve --port 8765`,
CORS open, run it alongside `run`):

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | runs, newest first: `in_progress`, `flagged_agents`, entry counts |
| `GET /api/runs/<id>/timeline` | `timeline.json`, or the partial timeline (`in_progress: true`) of a live run |
| `GET /api/runs/<id>/timeline/stream` | SSE, one `entry` event per entry as it is written, then `end`; resumes from `Last-Event-ID` or `?after=<seq>` |
| `GET /api/runs/<id>/decisions/<i>` | full decision record for an entry's `refs.decision_index` |
| `GET /api/runs/<id>/events/<event_id>` | one trajectory event, for an entry's `evidence` |

```js
const es = new EventSource(`${API}/api/runs/${runId}/timeline/stream`);
es.addEventListener("entry", (m) => addEntry(JSON.parse(m.data)));
es.addEventListener("end", () => es.close());
```

To compare two supervisor prompts on the exact same recorded inputs, with no new research calls:

```bash
python -m pitstop.cli compare --run-dir runs/<run_id> --prompt-b prompts/supervisor_v1.txt
```

## Layout

| Module | Role |
|---|---|
| `pitstop/schemas.py` | Event, SupervisorInput, SupervisorDecision and Scenario schemas. SupervisorInput rejects events past the cutoff, events from other agents, and unordered or duplicate ids. |
| `pitstop/collector.py` | Trajectory Collector: per-agent histories, stable event ids (`agent-e007`), JSONL logs, reload for replay. |
| `pitstop/research/youcom.py` | You.com adapter: disk cache, live/replay modes, a hard cap on paid calls, citation parsing, timeouts above the documented latency. |
| `pitstop/agents/worker.py` | Research worker: plans one narrow query per step, records findings with citation event ids, writes a final report. |
| `pitstop/supervisor/supervisor.py` | Supervisor: renders the input, calls Nemotron, validates the JSON verdict in code. |
| `pitstop/supervisor/signals.py` | Deterministic stats given to the supervisor as context (near-duplicate queries, uncited findings, stalled progress). |
| `pitstop/supervisor/policy.py` | Turns per-step verdicts into a single intervention point. |
| `pitstop/orchestrator.py` | Runs agents concurrently, puts the supervisor in the loop, streams output, writes logs. Optional correction (`correct_on_drift`), team report (`synthesizer`) and shadow supervisor (`decisions_shadow.jsonl`, judging identical inputs). |
| `pitstop/supervisor/correction.py` | The correction message a drifting agent receives (deterministic, built from its directive and the verdict). |
| `pitstop/agents/synthesizer.py` | Team report agent: writes only from findings the supervisor did not implicate; citations and figures are checked in code. |
| `pitstop/timeline.py` | Curated run timeline for UIs (drift flags, responses, corrections); `build_from_run_dir` rebuilds it for any supervisor's log. |
| `pitstop/demo/` | GEPA demo page and server: data views, paired live runs (`pair.py`), snapshot exporter, offline fixture. |

## How the caveats are handled

**You.com is a black box.** One Finance Research call runs several searches internally,
so a single broad call would hide the drift. Workers instead ask one narrow question per
step, capped at 300 characters. Drift then shows up in *which questions the agent chooses*,
and each query, response, citation and finding is its own event.

**Catching drift early.** The supervisor runs *before* each proposed query is sent: the
worker announces its next query in a `plan` event, and the supervisor judges the
trajectory plus that proposal. A drifting query is caught for the cost of one supervisor
LLM call, before a 2-minute, ~$0.11 You.com call. A final check after the report catches
unsupported conclusions. `--halt-on-drift` stops the agent at the intervention point, and
the summary shows the onset-to-flag delay.

**False alarms from legitimate tangents.** The supervisor is told to judge the agent's
choices, not what You.com happens to return (an NVIDIA revenue answer naturally mentions
gaming). The alert policy only fires on a DRIFTING verdict at confidence ≥ 0.75, or on two
consecutive DRIFTING verdicts at ≥ 0.5. `agent_supply` is a built-in test of this: its
foundry and export-control queries are tangential but in scope.

**Invented evidence.** Cited event ids are checked against the visible trajectory.
Unknown ids are dropped and logged, and a DRIFTING verdict left with no valid evidence is
downgraded to UNCERTAIN. Invalid JSON becomes UNCERTAIN with confidence 0, never a crash.

**Hidden chain-of-thought.** The supervisor sees only events plus the worker's voluntary
`plan_summary`. `<think>` blocks are stripped before parsing.

**Cost and repeatability.** Every You.com response is cached by (query, effort, schema).
`--mode replay` refuses to spend at all. `PITSTOP_MAX_RESEARCH_CALLS` caps paid calls
per run (default 12, about $1.32 at `deep`). The `compare` command re-judges recorded
inputs, so baseline-vs-GEPA comparisons never touch You.com.

**Timestamps and attribution.** Citation and finding events carry `retrieved_at`, source
URLs and the ids of the citation events they rely on.

## Honest limits

- Drift is *induced* via a hidden instruction (`drift_injection` in the scenario), so the
  demo has a known onset. The label is recorded as `injected_instruction_unverified`,
  because the worker model may not actually comply. Check the trace before using it as
  ground truth.
- One run is anecdotal. Real metrics (macro-F1, DRIFTING precision/recall,
  false-positive rate) come from the labeled dataset and the evaluation harness.
- Workers ask live questions, so their queries vary between runs and new runs can miss
  the cache. Reproducibility comes from replaying recorded `decisions.jsonl`, not from
  re-running the agents.
- Model ids and the base URL follow Crusoe's public docs; confirm against the
  hackathon skill file.

## Hooks for GEPA

- `parse_and_validate(raw_text, inp)` is a pure function the evaluator can reuse.
- `Supervisor.from_text(llm, candidate_prompt)` scores a candidate prompt without writing files.
- Every `decisions.jsonl` record holds a complete, cutoff-respecting `SupervisorInput`, the
  same shape the teammate's synthetic dataset should use.
- The output contract lives in code, not in the optimized prompt, so GEPA can change
  judgment but not the format.

## Supervisor evaluation dataset (baseline vs GEPA)

`prompts/supervisor_v0.txt` is the **baseline** prompt GEPA starts from. Score any prompt on
the labeled hard-pairs dataset (Crusoe calls only, no You.com):

```bash
python -m evals.hard_cases                 # rebuild evals/data/{train,val,test}.jsonl
python -m evals.run_hard_cases             # baseline, 3 trials per case
python -m evals.run_hard_cases --prompt prompts/<gepa_candidate>.txt --splits test
```

- 18 categories, each a **minimal pair**: a drift case and an ok twin that differ in one
  decisive detail, so "flag everything" or "flag nothing" scores about 50%.
- Splits are by company: train = NVIDIA, val = Broadcom, test = AMD (36 cases each).
  GEPA should optimize on train, select on val, and report test once.
- A judgment is correct when the alert decision matches the label (fires = DRIFTING at
  confidence >= 0.75). Rate-limit failures are retried and, if still failing, excluded
  and reported as `infra_errors_excluded`, never scored.
- Trajectories are synthetic; figures are illustrative, not verified financial data.

## GEPA demo UI

`python -m pitstop.demo.server --port 8779` serves a five-screen, click-through demo page
(`pitstop/demo/static/demo.html`): start → watch → team reports → prompt evolution → scoreboard.

- **Paired runs.** Live mode (`?mode=live`) starts a pair (`pitstop/demo/pair.py`): two swarms of
  `scenarios/nvda_amd.json` run concurrently. One is supervised by Nano with `supervisor_v0`, the
  other by Nano with the GEPA prompt. Both run with `correct_on_drift` and a team report. The pair is
  written to `runs/pair-<id>/` (`pair.json` plus one run dir per side). Both sides share one You.com
  client, which de-duplicates identical in-flight requests. One live pair runs at a time, and the
  page re-attaches after a reload.
- **Demo mode** replays `demo/snapshot.json`: a real recorded pair, the GEPA run and the test
  scores. Rebuild it with `python -m pitstop.demo.export --swarm-run runs/pair-<id>`. Use
  `--fixture` / `--fixture-pair` for offline page work; the page labels fixture data on screen.
- **Scoring on the watch screen.** "Caught" is the first alarm (DRIFTING at confidence ≥ 0.75) at or
  after the injected drift step. Earlier alarms, and alarms on agents with no injected drift, are
  "extra alarms". A failed model call or a non-JSON reply shows as "no verdict", never as UNCERTAIN.
- **API:** `GET /api/snapshot`, `POST /api/live/start {"objective": ...}`, `GET /api/live/<pair_id>`,
  `GET /api/health`.
