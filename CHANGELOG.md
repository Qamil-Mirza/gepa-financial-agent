# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Drop-in SDK: `pitstop.wrap`, `workflow`, `agent`, `tool` (block/steer/halt guard), `prompt` (versioned, developer edits win), `Halted` and `BudgetExceeded`.
- Hard dollar budget ledger with atomic reserve/settle, and per-workflow `budget_usd`; PitStop's own model calls go through it (`docs/config.example.toml`).
- `pitstop status`, `pitstop drifts` and `pitstop budget` commands, and a keyless quickstart example (`examples/quickstart`).
- Open-source repo basics: Apache-2.0 license, Contributor Covenant code of conduct, security policy, issue and pull-request templates, changelog and a CI workflow.
- Ruff linting (pyflakes rules) and offline-by-default test suite.
- `pitstop budget reconcile [--older-than-min 60] [--release]` lists old budget reservations, left open by a process that died mid-call or by a call still running; `--release` frees them and lists what it freed.
- `pitstop.workflow` and `pitstop.agent` decorate generator and async-generator functions; the run spans the generator's steps and never leaks to the consuming code.
- `wrap` meters `chat.completions.parse` and `with_raw_response` / `with_streaming_response` calls (on the client or the resource). Raw responses are settled from their parsed usage; streaming responses and raw streams, which PitStop can't read, are charged their full reservation (`usage_estimated: "worst_case"`).
- An endpoint that rejects PitStop's injected `stream_options` with a 400 is retried once without it, and later stream calls to that base_url skip the injection.

### Changed
- `pitstop` is now an installable `src/` layout package with packaged assets.
- The GEPA engine lives in `pitstop.correct`; the evaluator is supplied by the caller.
- Eval sets and detector calibration tooling are collected under `benchmarks/`.
- Hackathon-era docs moved to `docs/history/`.
- Runs and the You.com cache now default to `~/.pitstop/runs` and `~/.pitstop/cache/youcom` (override with `PITSTOP_HOME`, `PITSTOP_RUNS_DIR`, `PITSTOP_CACHE_DIR`); previously they lived under `pitstop/pitstop/runs` and `pitstop/pitstop/cache` in the checkout.
- `pitstop` is now an installed console script (previously `python -m pitstop.cli` run from `pitstop/pitstop/`).
- Store schema v3: older DBs are upgraded step by step in one transaction, and a DB written by a newer PitStop fails closed with `Halted("pitstop config invalid: ...")`.
- Drift records count every occurrence (`occurrences`, `last_seen`) and keep only the newest 20 evidence entries, each `{"run_id", "ts", "detail"}`; `pitstop drifts` shows the real first and last seen times.
- `runs.base_url` is stored normalized (no trailing `/`), matching the budget ledger.
- Drift detection and the run's model tag use only calls on the `agent` meter; judge, supervisor and GEPA calls no longer trigger L0 drifts. `llm_call` events carry their `meter`.
- Per-agent tool history and per-run LLM history keep the newest 64 calls; refusals are counted over the whole run.
- The steer limit (2 per run) holds when several threads share a run.
- `pitstop.sdk.wrap` is a package; wrapped streams pass `isinstance` checks against the SDK's stream classes.
- A pending P0 halt on a high-criticality workflow is recorded as a `halt` event (`pending: true`) when it is set.

### Removed
- The superseded supervisor stack, dead scaffold and committed generated data.
