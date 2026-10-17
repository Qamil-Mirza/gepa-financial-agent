# Architecture

The `gepa_financial_agent` package is a task-scoped drift supervisor for multi-agent financial research. Its system prompt is optimized with GEPA. This document describes the code as it exists; every name below can be found in the package.

## 1. Task-scoped sessions

There is one `SupervisorSession` per supervised agent (`gepa_financial_agent/session_manager.py`). A session holds its own `session_id`, `agent_id`, `task_id`, `directive`, `swarm_objective`, `constraints`, `success_criteria`, `dependencies`, a pinned `policy`, its `events` and its past `records`.

Every session in a `SupervisorManager` shares two things:

- **Inference client:** one `CrusoeClient` (`crusoe_client.py`), or a `MockModelClient`.
- **Policy:** one immutable `Policy` (`policy_store.py`), a text prompt identified by a 12-character sha256 version.

A clone is a session, not a model deployment. `SupervisorManager.set_policy()` affects only agents registered after the call. Existing sessions keep their pinned policy version, so their judgments are reproducible.

**Isolation:** `SupervisorSession.build_input()` renders only that session's own events. `tests/test_sessions.py::test_sessions_are_isolated_and_coalesced` evaluates two sessions concurrently and asserts that agent A's prompt never contains agent B's event IDs.

## 2. Data flow

```
TrajectoryEvent ─▶ SupervisorManager.observe() ─▶ SupervisorSession.add_event()
                                                     │ TriggerConfig.should_trigger → pending
SupervisorManager.evaluate_pending() ─▶ SupervisorSession.build_input() → SupervisorInput
    ─▶ supervisor.build_messages(policy_text + OUTPUT_CONTRACT, input)
    ─▶ ModelClient.complete / acomplete
    ─▶ supervisor.parse_decision(text, input) → SupervisorDecision (evidence IDs validated)
    ─▶ EvaluationRecord (decision or error, raw_output, invalid_evidence_ids, latency, tokens, mocked)
```

How `parse_decision` handles model output:

- **Recovering JSON:** it pulls the JSON object out of surrounding prose or code fences.
- **Unknown evidence IDs:** IDs that are not in the trajectory are dropped and recorded in `invalid_evidence_ids`; the decision itself is kept.
- **Truncated output:** an empty response with `finish_reason == "length"` becomes an explicit `truncated_output` error, never `ON_TRACK`.
- **Ground truth stays out:** `build_messages` takes a `SupervisorInput`, never a `DatasetExample`.
- **Opaque event aliases:** the prompt never shows real event IDs. `event_aliases` renders them as `E1..En` by trajectory position, and `parse_decision` maps the cited aliases back to real IDs; anything else counts as fabricated evidence. Datasets often encode the scenario, and so the label, in their IDs: an earlier version leaked the label this way (see the Correction in `PROGRESS.md`). `tests/test_supervisor.py::test_no_fixture_prompt_contains_a_label_token` guards it.
- **Robust parsing:** `extract_json` scans every `{` with `json.JSONDecoder.raw_decode` and keeps the last object that has a `status` key, so braces in prose or reasoning and draft objects don't break parsing. Enum values are case-insensitive.

## 3. Bounded concurrency

Three limits apply:

- **Manager semaphore:** `SupervisorManager(max_concurrent_inferences=...)`, default 4. This is an `asyncio.Semaphore` around every evaluation.
- **Client semaphore:** `CrusoeClient(max_concurrent=...)` holds a `threading.BoundedSemaphore` for sync calls and an `asyncio.Semaphore` for async calls. Every path to the model passes through one of them.
- **GEPA workers:** `OptimizeConfig.max_workers`, default 4, set by `--max-workers`. It controls how many evaluator calls GEPA makes in parallel from threads.

Crusoe caps each model at 30 requests/min without a payment method and 600 with one. The client treats 429 and 503 as transient: it backs off starting at 1s, doubles each time up to 30s, and adds jitter. `UsageTracker` records requests, retries, tokens, latency, peak in-flight requests and estimated cost.

## 4. Triggers and coalescing

Not every event triggers an inference. `TriggerConfig` (defaults `every_n_events=4`, `repeated_tool_call_threshold=3`) marks a session as `pending` when any of these happens:

- an event of type `error`, `proposed_next_action` or `task_completion` arrives;
- N events have arrived since the last evaluation;
- the same tool call (tool name plus arguments) has been seen 3 or more times.

`evaluate_pending()` evaluates each pending, non-evaluating session exactly once. Several triggers that fire between two calls are coalesced into a single evaluation.

**Failure isolation:**
- A redelivered event (same `event_id`, same content) is ignored, so at-least-once delivery is safe. The same `event_id` with different content is rejected with `ValueError`, which the adapter logs and drops.
- An exception while evaluating one session becomes an `EvaluationRecord` with a `session_error`; `evaluate_pending` gathers with `return_exceptions=True`. One bad session never stops the others.

## 5. Scoring (GEPA per-example reward)

`evaluator.score_prediction(record, example)` returns a score in [-0.5, 1.0] plus a feedback dict that GEPA's reflection model reads.

| situation | delta |
|---|---|
| status correct | +0.60 |
| gt DRIFTING, pred UNCERTAIN | +0.15 |
| gt ON_TRACK, pred UNCERTAIN | +0.20 |
| gt ON_TRACK, pred DRIFTING (false alarm) | −0.30 |
| gt DRIFTING, pred ON_TRACK (miss) | 0 |
| drift_type correct (both DRIFTING) | +0.20 |
| evidence overlaps ground-truth evidence (pred and gt in {DRIFTING, UNCERTAIN}) | +0.20 (+0.10 if gt has no evidence list but pred cites valid ids) |
| each fabricated evidence id | −0.10, capped at −0.30 |
| DRIFTING with zero valid evidence | −0.10 |
| invalid / unparseable output | score 0, no other terms |

**Why degenerate strategies lose:**

- **Always UNCERTAIN:** it earns 0.20 on healthy agents and 0.15 on clear drift, plus the evidence bonus. On the dev fixtures it averages about 0.33–0.37, below a real detector (about 0.55). This is not a hard cap: UNCERTAIN on an UNCERTAIN example earns the full +0.60, and the evidence bonus checks overlap, not precision (a known weakness, deferred).
- **Always ON_TRACK:** it scores 0.6 only on healthy agents and 0 on every drift.
- **Always DRIFTING:** it is penalized −0.30 on every healthy agent.

Balanced splits (`dataset.label_distribution`) keep all three below a real detector. Macro-F1 and false-positive rate are reported separately on full sets (`compute_metrics`), so a scalar gain cannot hide a worse FPR.

The feedback string names the exact error, for example "overconfident: only one ambiguous deviating event was visible; UNCERTAIN was correct". It also includes the labeler's rationale, which only the reflection model sees, never the supervisor.

## 6. GEPA wiring

- **Installed API:** gepa 0.1.4. `gepa.optimize_anything.optimize_anything(seed_candidate=str, evaluator=fn(candidate, example) -> (score, side_info), dataset=train, valset=val, objective=..., background=..., config=GEPAConfig(engine=EngineConfig(max_metric_calls, run_dir, seed, parallel, max_workers), reflection=ReflectionConfig(reflection_lm=CrusoeReflectionLM, reflection_minibatch_size)))`. The PRD's `OptimizeAnythingConfig`/`max_evals` names do not exist in this version.
- **What gets optimized:** only the policy text. `OUTPUT_CONTRACT` (`supervisor.py`) is a fixed suffix appended to every candidate, so all candidates share one output schema.
- **Split discipline:**
  - `split_by_scenario` keeps every prefix of a scenario in the same split. It stratifies scenarios by their `kind:` tag, or by the full trajectory's label when there are no tags, so every kind, including hard negatives, reaches every split.
  - GEPA sees only train (reflection minibatches) and val (candidate selection).
  - `scripts/optimize_supervisor.py` touches the test split only with `--final-test`. The intended flow is to choose on val, then run one final test evaluation, for example `evaluate_supervisor.py --split test` repeated to measure noise.
  - Reflector feedback (`optimizer._to_prompt_ids`) uses the same aliases the supervisor saw and drops `example_id` and `scenario_id`. Labels and rationales do reach the reflector: that is the training signal, and it only comes from train minibatches.
  - Optimization uses only frozen trajectories; the package never calls You.com.
- **Artifacts in `runs/gepa/<ts>/`:** `config.json`, `seed_policy.txt`, `best_policy.txt`, `result.json`, GEPA's `candidates.json`, `run_log.txt`, `gepa_state.bin`, `comparison.json`/`comparison.txt`, per-split `eval_*/predictions.jsonl` and `metrics.json`, and `artifacts.json` with separate supervisor and reflection usage.
- **Versioning:** each real run saves `policies/versions/gepa_<ts>.txt` and registers it in `policies/versions/manifest.json`. Mock runs are saved as `mock_<ts>`. `--promote` (`PolicyStore.promote`) copies a version to `policies/current.txt`, and refuses (exit 3) when the run was mocked. Old version texts are never changed.

## 7. Supervisor model vs reflection model

The two models are configured separately (`config.Settings`):

- `SUPERVISOR_MODEL`, default `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`: used at high volume. It replaced `deepseek-ai/Deepseek-V4-Flash` on 2026-10-17 (see `PROGRESS.md` for the latency and cost difference).
- `REFLECTION_MODEL`, default `deepseek-ai/DeepSeek-V4-Pro`: the stronger reasoner that proposes new candidates.

Both are served by Crusoe Managed Inference through the OpenAI-compatible SDK at `https://api.inference.crusoecloud.com/v1`. `CrusoeReflectionLM` adapts a `CrusoeClient` to GEPA's `reflection_lm` callable, `(prompt) -> str`, so litellm is not used. Other settings:

- `SUPERVISOR_JSON_MODE` turns on `response_format`; the smoke test confirmed Crusoe supports it for both models.
- `SUPERVISOR_THINKING` pins thinking through `chat_template_kwargs`: the key is `enable_thinking` for Nemotron and Qwen models and `thinking` for DeepSeek (`crusoe_client._thinking_key`). Nemotron ignores the DeepSeek key.

## 8. Mock vs real

`MockModelClient` is a deterministic stub for tests and offline pipeline checks. Every `CompletionResult` and `EvaluationRecord` carries `mocked: bool`, and outputs are labeled as follows:

- every script prints `MODE: MOCK` or `MODE: REAL`;
- comparison tables show `MOCK`/`REAL` per row;
- the policy manifest records `mocked` for each saved version;
- mock reflection output contains `[MOCK REFLECTION ...]`.

Mock runs prove that the pipeline is wired correctly, nothing more. `optimize_supervisor.py` refuses to promote them.

## 9. Optional dependency metadata (DAG failsafe)

`register_agent(..., dependencies=[...])` stores the IDs of downstream tasks on the session. `build_input` passes them as `SupervisorInput.relevant_dependencies`, which the prompt renders as "Downstream tasks depending on this agent". V1 does no graph traversal or impact analysis; those are Milestone F items.

## 10. Known limitations

- **Uncalibrated confidence:** `confidence` is a model estimate, not a calibrated probability.
- **Small data:** results come from the teammate dataset (`synthetic-data/data/dataset.jsonl`, 315 examples: 179 train / 54 val / 82 test). On 54 val examples the baseline's macro-F1 varies by about 0.03 between identical runs, which is as large as the gains GEPA finds. The dev fixtures (`scripts/make_dev_fixtures.py`) are for plumbing and tests only.
- **Render limit:** each event body is clipped at `MAX_EVENT_CHARS` = 3000 characters. 2 of 369 event bodies in the test split exceed it.
- **Dataset validation:** `LoadReport.summary()` lists each invalid line with its line number and reason. The scripts exit with code 4 on invalid lines unless `--allow-invalid` is passed. A missing `supervisor_input.observation_step` is filled in from the top-level one, and a conflicting value is rejected.
- **UNCERTAIN-at-onset is our convention:** `expand_prefixes` labels the onset step UNCERTAIN because our fixture generator was built that way. The thresholds GEPA learns (for example "≥2 identical repeats" for UNPRODUCTIVE_LOOP) may be fitting that generator rather than drift in general.
- **Rate limits:** 30 requests/min per model without a payment method bounds GEPA budgets and evaluation size.
- **Transport failures are scored as INVALID:** `CrusoeClient` retries 429/5xx/timeouts with exponential backoff (`max_retries=5`, about 31 s in total). Orchestrator-level 429s (`max_requests_limit (orchestrator)`) and connection errors sometimes outlast that window. The example then becomes an INVALID prediction (6 rows over the 10 policy-repeats of the final test battery). This is not yet fixed: the retry window should be longer for 429s, and failed rows should be re-queued instead of scored.
- **Reward tolerates the wrong drift type:** with `W_STATUS=0.6, W_TYPE=0.2, W_EVIDENCE=0.2`, DRIFTING with the wrong drift_type and correct evidence still earns 0.8. GEPA exploited this: gepa_team_300 labels most drift as GOAL_DEVIATION (`docs/RESULTS.md`). Future work: raise `W_TYPE`, or make the drift_type a hard requirement for DRIFTING credit.
- **GEPA 0.1.4 acceptance discards ties:** a child must strictly beat its parent on the 3-example reflection minibatch. All 9 rejections across our three runs were exact ties. Each accepted child also costs a full 54-row val eval, so a 300-call budget buys only 10 reflection loops.
- **Observational only:** supervision never blocks or corrects agents.
