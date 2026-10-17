# Progress log

Workstream 3 (supervisor + GEPA pipeline), 2026-10-17. Plan: `docs/superpowers/plans/2026-10-17-gepa-supervisor-pipeline.md`.

| milestone | status | evidence |
|---|---|---|
| A supervisor core | done | `uv run pytest -q` → 53 passed. `smoke_test.py --supervise` was REAL: `nvda_goal_deviation_0_full@4` was expected DRIFTING/GOAL_DEVIATION and came back DRIFTING/GOAL_DEVIATION, conf 0.95, evidence `[e3, e4]` all valid, `error: null` |
| B parallel sessions | done | `tests/test_sessions.py` 8 passed, including the isolation and coalescing test. Real demo: `max_in_flight: 3` across 3 concurrent sessions, 0 errors |
| C evaluation harness | done | `evaluate_supervisor.py --split val --heuristic` (REAL): baseline macroF1 0.541, FPR 0.083, R_drift 1.000; heuristic macroF1 0.400 |
| D GEPA optimization | done, not promoted | Teammate dataset, Nemotron Super supervisor: GEPA runs team_60/150/300 work end to end. 5 test repeats: baseline macro-F1 0.549 (0.519–0.588) vs gepa_team_300 0.553 (0.532–0.581); gepa misses more drift (2.2 vs 0.4 undetected scenarios). `current` stays baseline. See `docs/RESULTS.md`. (The earlier dev-fixture 0.592 → 0.767 result is void: label leak.) |
| E integration | mock side done | Real demo: A ON_TRACK, B DRIFTING/GOAL_DEVIATION, C ON_TRACK under both policies. The teammate adapter is not connected yet |

## Final submission run (teammate dataset, 2026-10-17 afternoon)

- **Test battery:** 5 test repeats of baseline vs gepa_team_300 on `synthetic-data/data/dataset.jsonl`, written to `runs/eval/team_final_r1..r5` and aggregated by `scripts/summarize_repeats.py` into `docs/results/final_test_20261017.json`. Test macro-F1 is 0.549 (0.519–0.588) for baseline and 0.553 (0.532–0.581) for GEPA.
- **Drift types:** gepa_team_300 is better on GOAL_DEVIATION (70/70 vs 62/70) and much worse on CONSTRAINT_VIOLATION (1/30 vs 15/30), OBSOLETE_DIRECTIVE (12/30 vs 24/30) and UNSUPPORTED_CONCLUSION (4/30 vs 19/30).
- **Not promoted:** the promotion rule fails, so `current` stays the baseline. Analysis is in `docs/RESULTS.md`: the GOAL_DEVIATION specialization, train-family examples hard-coded into the prompt, val noise and GEPA loop counts.
- **The dev-fixture sections below are historical.** They used `deepseek-ai/Deepseek-V4-Flash` on dev fixtures, and the "promoted" version there was later voided for a label leak.

## Real inference log (verbatim, with timestamps)

All runs used Crusoe Managed Inference with supervisor `deepseek-ai/Deepseek-V4-Flash`. GEPA reflection used `deepseek-ai/DeepSeek-V4-Pro`.

**Smoke test (~12:15).** `smoke_test.py --list-models` found both models, with `response_format supported: True` for each. The model replied `"text": "pong"`, `"finish_reason": "stop"`, `"reasoning_present": false`, `"mocked": false`, latency 1234.5ms. Based on this, `.env` now sets `SUPERVISOR_JSON_MODE=true` and `SUPERVISOR_THINKING=false`.

**Baseline eval on val (`runs/eval/20261017_122022`):**
```
heuristic              heuristic  REAL  19   0.400    0.684  1.000    0.250    0.000  0.750  0.000    1.000    0.432  0       n/a
baseline/val           eac5a206b5 REAL  19   0.541    0.789  0.500    1.000    0.083  0.000  0.000    1.000    0.574  1313    0.0035
```
The baseline confusion matrix (rows are ground truth) shows UNCERTAIN recall 0: all 3 UNCERTAIN examples were predicted DRIFTING.

**GEPA run, 60 calls (`runs/gepa/20261017_122148`, PROMOTED).**
- **Search:** `GEPA finished: candidates=3 metric_calls=69 seed_val=0.574 best_val=0.618`.
```
baseline/val           eac5a206b5 REAL  19   0.541    0.789  0.500    1.000    0.083  0.000  0.000    1.000    0.574  1060    0.0017
gepa/val               efded549eb REAL  19   0.689    0.842  0.600    0.750    0.000  0.250  0.000    1.000    0.618  1191    0.0019
baseline/test          eac5a206b5 REAL  25   0.592    0.800  0.692    1.000    0.000  0.000  0.000    1.000    0.656  1142    0.0046
gepa/test              efded549eb REAL  25   0.767    0.840  0.727    0.889    0.000  0.111  0.000    1.000    0.686  1212    0.0031
GEPA policy improved held-out macro-F1: baseline=0.592 gepa=0.767 FPR baseline=0.000 gepa=0.000
```
- **Detection on test:** delay 0.2 (baseline) vs 1.2 (gepa); undetected 0 for both.
- **Cost:** supervisor $0.0120 (69 requests), reflection $0.0167 (2 requests). 0 retries.
- **Policy:** grew from 250 to 522 words, adding graduated-evidence rules plus UNCERTAIN-at-onset rules for UNPRODUCTIVE_LOOP and OBSOLETE_DIRECTIVE.

**GEPA run, 150 calls (`runs/gepa/20261017_122338`, not promoted).**
- **Search:** `GEPA finished: candidates=5 metric_calls=161 seed_val=0.574 best_val=0.621`.
```
baseline/val           eac5a206b5 REAL  19   0.541    0.789  0.500    1.000    0.083  0.000  0.000    1.000    0.574  1156    0.0017
gepa/val               b31ca3fe78 REAL  19   0.541    0.789  0.500    1.000    0.083  0.000  0.000    1.000    0.574  1436    0.0028
baseline/test          eac5a206b5 REAL  25   0.594    0.800  0.643    1.000    0.000  0.000  0.000    1.000    0.664  1203    0.0022
gepa/test              b31ca3fe78 REAL  25   0.592    0.800  0.692    1.000    0.000  0.000  0.000    1.000    0.656  1059    0.0035
GEPA policy did NOT improve held-out macro-F1: baseline=0.594 gepa=0.592 FPR baseline=0.000 gepa=0.000
```
- **Cost:** supervisor $0.0305 (161 requests), reflection $0.0954 (11 requests).
- **Why it did not beat baseline:**
  - Val noise is about as large as the improvements GEPA was trying to measure. The candidate scored 0.621 inside GEPA but 0.574 when re-evaluated.
  - Its 11 iterations chased one-example swings on the 19-example val split.
  - It cost 5.6× more in reflection than the 60-call run.

**Selection.** I chose between the two runs on **val** (gepa/val macroF1 0.689 for the 60 run vs 0.541 for the 150 run). The 150 run's test numbers exist in its `comparison.json` but were not used for selection. The 60 run was promoted because its test macroF1 beats baseline and test undetected drift is 0.

**Variance check, test n=25, 5 runs per policy (3 unseeded + 2 with `seed=42`).**
- **Baseline `eac5a206b52d`:** macroF1 0.594 in all 5 runs; FPR 0; delay 0.0; undetected 0.
- **GEPA `efded549ebd7`:** macroF1 0.648 / 0.731 / 0.810 unseeded, and 0.716 / 0.681 seeded.
  - Median of the unseeded runs: 0.731. Median of all 5: 0.716.
  - FPR 0 and undetected 0 in every run; delay 2.0 / 2.0 / 0.4 / 0.4 / 1.2.
- **Cost:** 250 requests, $0.025, 0 errors.
- **Seed:** Crusoe does not honor it. Two `seed=42` runs gave identical raw outputs on only 3/25 (baseline) and 4/25 (GEPA), no better than unseeded runs (4/25 and 6/25). Evaluation and GEPA defaults are unchanged.

**Live demo (`demo_supervision.py --delay 0.2`, REAL).**
- **Both policies:** agent_A ON_TRACK, agent_B DRIFTING/GOAL_DEVIATION (evidence `amd_goal_deviation_0_e3..e6`), agent_C ON_TRACK. All three match the fixture labels.
- **Usage:** 18 requests, 0 errors, 0 retries, `max_in_flight: 3`.

## Known blockers and risks

- **Crusoe key:** available via `.env`, and every real step above has run. Without a payment method each model is capped at 30 requests/min. In the final test battery, orchestrator-level 429s and connection errors sometimes outlasted the retry window, and 6 rows were scored INVALID (`docs/RESULTS.md`, Known issues).
- **Teammate dataset not delivered yet** (`docs/DATASET_CONTRACT.md`). Every number here comes from our own deterministic dev fixtures. Val is 19 examples (12 ON_TRACK / 4 DRIFTING / 3 UNCERTAIN), so val macro-F1 moves by about 0.1 per flipped example. Credible final numbers depend on the teammate dataset. **Blocker for claiming results, not for the pipeline.**
- **UNCERTAIN-at-onset is our convention:** `expand_prefixes` labels the onset step UNCERTAIN, a rule from our fixtures. Teammate labelers should judge onset ambiguity case by case.
- **Possible overfitting to our generator:** the learned thresholds ("UNPRODUCTIVE_LOOP needs ≥2 identical repeats", OBSOLETE_DIRECTIVE "1 post-update event → UNCERTAIN, 2+ → DRIFTING") match our fixture generator's onset rules. GEPA may be fitting the generator rather than drift in general, so it must be validated on teammate data before we claim generalization.
- **Non-deterministic inference:** results vary at temperature 0 and with a seed set. Every single-run number has an error bar; the GEPA policy's test macro-F1 spread is about 0.16.
- **Teammate agent stack:** it doesn't emit events yet. The integration contract (`docs/INTEGRATION_CONTRACT.md`) and `EventStreamAdapter` are ready to receive them. **Blocker for Milestone E (real stack).**

## Next action

1. When the teammate dataset arrives, validate it, then re-run baseline, GEPA and the 3-repeat variance check on it.
2. Connect the teammate's agent events through `EventStreamAdapter` and re-run the demo on real trajectories.
3. Consider averaging 2–3 repeats per val evaluation inside GEPA to reduce selection noise.

## Future work (not to be implemented now)

The variance comes from onset prefixes flipping between UNCERTAIN and DRIFTING at temperature 0. Two mitigations to evaluate later:

- **(a) Self-consistency vote:** take a vote over 3 samples per decision. This triples inference cost.
- **(b) Scoring change:** reward a stable UNCERTAIN on onset prefixes less than the current `W_UNCERTAIN_ON_DRIFT`, so GEPA stops optimizing toward the borderline.

Evidence for the variance analysis: `docs/results/variance_test_20261017.json`. Crusoe ignores the `seed` parameter: two `seed=42` runs gave identical raw outputs on only 3/25 examples (baseline) and 4/25 (GEPA).
