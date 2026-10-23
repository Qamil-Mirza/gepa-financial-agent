# Contributing to PitStop

Thanks for helping out. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```sh
uv sync
cp .env.example .env   # then fill in the keys you need
```

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): the root project is the `pitstop` package, and `examples/financial-research` (package `finresearch`, distribution `pitstop-financial-research`) is a workspace member that depends on it. `uv sync` installs both, editable, with the `pitstop` and `finresearch` console scripts. The `pitstop` wheel ships only the SDK; code for the example goes in `examples/financial-research`, and `pitstop` must never import `finresearch`. After changing a dependency in either `pyproject.toml`, run `uv lock` and commit `uv.lock` (CI runs `uv sync --locked`).

Never commit `.env` or any API key. If you do, revoke the key immediately.

## Repo map

- `src/pitstop`: the installable package: the SDK, the `pitstop` CLI and `pitstop.correct` (the GEPA engine).
- `examples`: runnable demos (`quickstart` shows the SDK with no keys; `financial-research` is the research swarm and the GEPA demo page, a workspace member with the `finresearch` CLI).
- `ui`: the local dashboard (React, built into `src/pitstop/server/static/` and served by `pitstop ui`).
- `benchmarks`: eval sets and detector calibration tooling (`hard_pairs`, `calibration`, `held_out`, `synth`).
- `tests`: the offline test suite (`sdk`, `detect`, `budget`, `alerts`, `store`, `correct`, `benchmarks`, `examples/financial_research`, repo hygiene).
- `docs`: design specs and plans under `docs/superpowers/`; hackathon-era documents under `docs/history/`.

## Tests and lint

```sh
uv run pytest      # offline by design
uv run ruff check
```

The test suite is offline: sockets are blocked via `pytest-socket`. Loopback (127.0.0.1/localhost) is allowed, so tests that start a local server are fine. Tests are meant to be run via `uv run pytest` (some need the project environment and `uv`). Tests must not call paid APIs; use fakes or recorded fixtures.

### Before deleting a tracked directory

Run `git clean -ndX <dir>` to list ignored local files inside it (runs, caches, `.env`) and move them out first.

## The dashboard (`ui/`)

The dashboard is a React app in `ui/` (Node 22, npm). `npm --prefix ui run build` writes it into `src/pitstop/server/static/` (git-ignored), which `pitstop ui` serves and the wheel bundles. Dependencies are pinned to exact versions in `ui/package-lock.json`; install them with `npm ci`, and pin any new one exactly (`ui/.npmrc` sets `save-exact`).

```sh
npm --prefix ui ci                  # install the locked dependencies
npm --prefix ui run dev             # Vite on http://localhost:5173, proxying /api to a running `pitstop ui` on 8787
npm --prefix ui run typecheck
npm --prefix ui test -- --run       # Vitest with jsdom, on JSON fixtures; never a real server
npm --prefix ui run build           # into src/pitstop/server/static/
```

For live data while developing, run `uv run pitstop ui` in a second terminal, or the offline demo: `uv run python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo`, then `PITSTOP_HOME=/tmp/pitstop-demo uv run pitstop ui`.

The UI's API types come from the server's models. After changing `src/pitstop/server/models.py` or an endpoint, regenerate the schema and the types, and commit both `ui/openapi.json` and `ui/src/api/schema.d.ts` (`tests/server/test_openapi_snapshot.py` fails until you do):

```sh
uv run python -c "import json; from pitstop.server.app import export_openapi; open('ui/openapi.json', 'w').write(json.dumps(export_openapi(), indent=2, sort_keys=True) + '\n')"
npm --prefix ui run gen:api
```

The Vitest fixtures in `ui/src/test/fixtures/` are real API responses on the demo data. Regenerate them with `uv run python ui/scripts/gen_fixtures.py` (it seeds its own temp home and never touches `~/.pitstop`).

## Branches and pull requests

1. Branch from `main`: `git checkout -b <name>/<thing>`.
2. Commit small and often, using conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
3. Make sure tests and lint pass, then open a PR into `main` and fill in the template.

## Design

The current design is in [docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md](docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md).
