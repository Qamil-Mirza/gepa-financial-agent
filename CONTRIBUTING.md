# Contributing to PitStop

Thanks for helping out. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```sh
uv sync
cp .env.example .env   # then fill in the keys you need
```

Never commit `.env` or any API key. If you do, revoke the key immediately.

## Repo map

- `src/pitstop`: the installable package (runtime, CLI, demo server and `pitstop.correct`, the GEPA engine).
- `examples`: runnable demos (`quickstart` shows the SDK with no keys).
- `benchmarks`: eval sets and detector calibration tooling (`hard_pairs`, `calibration`, `held_out`, `synth`).
- `tests`: the offline test suite (`runtime`, `correct`, `benchmarks`, repo hygiene).
- `docs`: design specs and plans under `docs/superpowers/`; hackathon-era documents under `docs/history/`.

## Tests and lint

```sh
uv run pytest      # offline by design
uv run ruff check
```

The test suite is offline: sockets are blocked via `pytest-socket`. Loopback (127.0.0.1/localhost) is allowed, so tests that start a local server are fine. Tests are meant to be run via `uv run pytest` (some need the project environment and `uv`). Tests must not call paid APIs; use fakes or recorded fixtures.

### Before deleting a tracked directory

Run `git clean -ndX <dir>` to list ignored local files inside it (runs, caches, `.env`) and move them out first.

## Branches and pull requests

1. Branch from `main`: `git checkout -b <name>/<thing>`.
2. Commit small and often, using conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
3. Make sure tests and lint pass, then open a PR into `main` and fill in the template.

## Design

The current design is in [docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md](docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md).
