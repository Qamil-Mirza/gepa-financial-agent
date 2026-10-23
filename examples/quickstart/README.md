# PitStop quickstart

A small tool-using agent (`search` plus a side-effecting `send_report`) wrapped with PitStop.

```sh
uv run python examples/quickstart/agent.py --offline --inject-loop
```

`--offline` replaces the model (and the judge of the pre-action gate on `send_report`) with scripted fakes, so no keys or network are needed. `--inject-loop` makes the agent repeat the same search: PitStop steers it twice, then halts the run with `Halted: steer limit exceeded`. Then look at what it recorded:

```sh
uv run pitstop status
uv run pitstop drifts
```

With a real model on Crusoe (paid calls, capped by the workflow's `budget_usd=1.0`):

```sh
export CRUSOE_API_KEY=...        # CRUSOE_BASE_URL and PITSTOP_WORKER_MODEL are optional
uv run python examples/quickstart/agent.py
```

State is stored under `~/.pitstop` (set `PITSTOP_HOME` to use another directory).

## Dashboard demo data

To see the local dashboard with something on it, fill an empty directory with offline demo data, then start the dashboard on it:

```sh
uv run python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo --runs 60
PITSTOP_HOME=/tmp/pitstop-demo uv run pitstop ui
```

The script runs four workflows through the real SDK. The model and the pre-action gate's judge are real OpenAI SDK clients whose HTTP transport is answered in-process with scripted replies, so it needs no keys, makes no network request and costs nothing. All text in it is synthetic.

- `earnings-research` (high criticality, `budget_usd=2.0` per day): `--runs` runs on model `m-a` (default 60, at least 35), then 35 on `m-b`, which uses twice the tokens per call. The scan's change detection alarms on tokens per call and cost, and attribution names the model change (P1). In the last run the writer tries to publish to a public channel; the gate's judge rejects it (P0 `out_of_scope_action`) and the run halts. Its tile is red.
- `support-triage` (medium): one run repeats the same knowledge-base search, and PitStop steers it twice (P1 `unproductive_loop`). Its tile is yellow.
- `nightly-digest` and `kb-sync` (low): healthy runs only. Their tiles are green.

It writes a `config.toml` with prices for its made-up models into the directory. It refuses a directory that already has a `pitstop.db` or a `config.toml`, so it can't touch an existing PitStop home.
