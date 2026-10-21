# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source repo basics: Apache-2.0 license, Contributor Covenant code of conduct, security policy, issue and pull-request templates, changelog and a CI workflow.
- Ruff linting (pyflakes rules) and offline-by-default test suite.

### Changed
- `pitstop` is now an installable `src/` layout package with packaged assets.
- The GEPA engine lives in `pitstop.correct`; the evaluator is supplied by the caller.
- Eval sets and detector calibration tooling are collected under `benchmarks/`.
- Hackathon-era docs moved to `docs/history/`.

### Removed
- The superseded supervisor stack, dead scaffold and committed generated data.
