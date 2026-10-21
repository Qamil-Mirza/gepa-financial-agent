# PitStop quickstart

A small tool-using agent (`search` plus a side-effecting `send_report`) wrapped with PitStop.

```sh
uv run python examples/quickstart/agent.py --offline --inject-loop
```

`--offline` replaces the model with a scripted fake, so no keys or network are needed. `--inject-loop` makes the agent repeat the same search: PitStop steers it twice, then halts the run with `Halted: steer limit exceeded`. Then look at what it recorded:

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
