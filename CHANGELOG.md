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
- `pitstop budget reconcile [--older-than-min 60] [--release]` lists, and with `--release` frees, budget reservations left open by a process that died mid-call.

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

### Removed
- The superseded supervisor stack, dead scaffold and committed generated data.
