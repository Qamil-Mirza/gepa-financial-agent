# Feedback for the synthetic-data workstream

These answers respond to the open questions in `synthetic-data/HANDOFF.md`. They come from the supervisor workstream, based on running the pipeline on `synthetic-data/data/dataset.jsonl`, which has 164 rows and 49 traces.

## Your three questions

1. **UNCERTAIN convention: keep it.** It matches ours. A cutoff at onset with only one ambiguous drift step is UNCERTAIN, and confirmation rules can differ by type (CONSTRAINT_VIOLATION and UNSUPPORTED_CONCLUSION are decisive at onset). Our scoring and the GEPA reflection text are written to learn the convention from your labels.
2. **Group bundling: we'll do it.** Single-agent examples are correct. The supervisor is task-scoped: one session per agent, sharing one model and one prompt. Any multi-agent bundling for demos happens on our side.
3. **`recommended_action` in `ground_truth`: not needed.** We don't score it. The final classification is kept separate from any intervention.

## What we found

- **Format:** all 164 rows validate after a small automatic mapping (see `docs/DATASET_CONTRACT.md`, "Accepted layouts"). No changes are needed on your side.
- **Leakage:** your IDs are opaque. We found no label tokens in any rendered supervisor prompt.
- **Duplicates:** there are 160 unique prompts among 164 rows, and no prompt appears in more than one split.
- **Long tool results:** 308 of 622 tool results are longer than the 600 characters the supervisor sees per event. Long cited answers are clipped in the prompt.

## Requests, in priority order

1. **More benign and clean traces (hard negatives).** These are the cases where a supervisor raises false alarms: legitimate detours, tool errors with recovery, and strategy changes. Our baseline's false-positive rate on your val split is 0.25.
2. **More ON_TRACK examples in test.** The test split has only 6 ON_TRACK out of 22, so the false-positive rate on test is very coarse (one example moves it by about 0.17).
3. **Optional: shorter tool results.** A shorter `content`, or a `summary` field next to the full text, would fit our prompt budget better.
