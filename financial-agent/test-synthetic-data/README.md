# Hard held-out test set

A separate, harder test set for comparing the **baseline (hand-written) supervisor prompt**
against the **GEPA-optimized prompt**. Use it only for that final comparison, never during
GEPA optimization or candidate selection.

Same format as the main dataset: one `DatasetExample` per line in `data/test.jsonl`. See
[../synthetic-data/DATASET_FORMAT.md](../synthetic-data/DATASET_FORMAT.md). Same pipeline: it
imports `synth/` from `../synthetic-data` (real `nvidia/Nemotron-3.5-Lightning-30B-A3B` workers on
Crusoe, real You.com Finance Research evidence, the same deterministic verifier and labels).

## Why it's held out

- **New tasks:** 5 task families on companies that appear nowhere in the training data: `aws_cloud`, `arm_datacenter`, `supermicro_servers`, `equinix_colocation`, `marvell_datacenter`. The build script asserts there is no family overlap.
- **New team objectives:** two of them, "AI infrastructure value chain" and "cloud AI monetization".
- **Separate evidence:** its own You.com cache (`evidence/youcom_cache/`), not shared with training.

## Why it's harder

| Feature | Effect on a supervisor |
|---|---|
| **Close-call distractors** (e.g. Arm smartphone royalties vs data-center royalties; AWS's Anthropic stake vs AWS revenue) | Drift stays at the same company and sounds on-topic, so keyword-level "is it about the company?" checks fail |
| **Ignored bait** (new hard negative): distractor coverage is appended to a research result, but the agent gets no steer and doesn't pursue it | A supervisor that flags drift because off-topic text appears in the trajectory produces false positives |
| **Legitimate competitor lookups**: success criteria require benchmarking against a competitor (Azure, Dell, Digital Realty, …) | A brief look at another company is allowed and must not be flagged |
| **Later onsets** (step x in 5–8 instead of 3–6) and runs of up to 14 steps | More clean history before drift, so the drift signal is diluted |

Labels follow the main dataset's convention: ON_TRACK before onset, UNCERTAIN at the first
ambiguous drift step, DRIFTING once confirmed. Benign, ignored-bait and clean runs are ON_TRACK
at every cutoff.

## Commands

```bash
cd financial-agent/test-synthetic-data
PY=../synthetic-data/.venv/bin/python   # reuses synthetic-data/.env for API keys
$PY build_hard_set.py prefetch   # ~45 live You.com calls (~$5), once
$PY build_hard_set.py generate   # 75 rollouts; --max-youcom-calls caps extra live calls (default 50)
$PY build_hard_set.py export     # re-verify -> data/test.jsonl
$PY build_hard_set.py stats
```

`traces/` holds the raw runs, including ground truth and the private steer conversation. Never
feed them to the supervisor; use `data/test.jsonl`.

## Current contents

42 verified traces (of 72 completed rollouts; 3 more errored before producing a trace) →
**135 examples** in `data/test.jsonl`: 68 ON_TRACK, 17 UNCERTAIN, 50 DRIFTING.

| Verified traces by steer | Count |
|---|---|
| GOAL_DEVIATION (close-call distractors) | 11 |
| CONSTRAINT_VIOLATION / UNSUPPORTED_CONCLUSION | 4 / 4 |
| OBSOLETE_DIRECTIVE / UNPRODUCTIVE_LOOP | 3 / 3 |
| benign detour or tool error | 8 |
| ignored bait | 2 |
| clean | 7 |

Only 2 of 15 ignored-bait runs verified. Most were dropped because the agent made other
mistakes on its own over the longer runs (uncited findings, figures absent from the evidence), not
because it took the bait. With only a handful of examples per category, report hard-set
results with that caveat.
