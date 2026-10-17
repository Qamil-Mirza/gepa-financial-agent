# PitStop: drift-detecting supervisor for a financial research swarm (V1, pre-GEPA)

Three research agents work on one objective using the You.com Finance Research API.
A Nemotron supervisor on Crusoe Managed Inference reviews each agent at every step and
flags drift with evidence from the agent's observable trajectory. GEPA comes next; this
package is built so GEPA only has to swap the text in `prompts/supervisor_v0.txt`.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add CRUSOE_API_KEY and YDC_API_KEY
python -m pitstop.cli check   # verifies a real Crusoe inference call for both models
python -m pitstop.cli check --youcom   # optional: one paid You.com call (~$0.11)
python -m pitstop.cli run --halt-on-drift
python -m pytest              # offline tests: scripted LLM + mocked You.com, no spend
```

A run writes `runs/<run_id>/` containing `events.jsonl` (every trajectory event),
`decisions.jsonl` (every supervisor input + validated verdict) and `meta.json`
(ground truth, interventions, token usage, You.com spend).

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
| `pitstop/orchestrator.py` | Runs agents concurrently, puts the supervisor in the loop, streams output, writes logs. |

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

## Hooks for GEPA (next milestone)

- `parse_and_validate(raw_text, inp)` is a pure function the evaluator can reuse.
- `Supervisor.from_text(llm, candidate_prompt)` scores a candidate prompt without writing files.
- Every `decisions.jsonl` record holds a complete, cutoff-respecting `SupervisorInput`, the
  same shape the teammate's synthetic dataset should use.
- The output contract lives in code, not in the optimized prompt, so GEPA can change
  judgment but not the format.
