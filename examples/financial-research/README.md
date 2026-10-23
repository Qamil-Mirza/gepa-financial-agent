# PitStop example: financial research swarm

The multi-agent research swarm PitStop started as at the hackathon, kept as an example built on the `pitstop` SDK. A scenario gives three research agents a shared objective (the default compares NVIDIA and AMD); they plan queries, call the You.com Finance Research API and record cited findings. Before each query runs, a per-step supervisor (a Nemotron model plus code-computed trajectory signals) judges whether the agent has drifted from its task, and can block the query and send a correction or halt the agent. A report agent then writes the final report from the findings the supervisor did not flag. Scenarios can inject drift into an agent's directive, so a run shows the supervisor catching it.

It is a uv workspace member of the PitStop repo: the package is `finresearch` (distribution `pitstop-financial-research`) and its CLI is `finresearch`. A `uv sync` at the repo root installs both `pitstop` and `finresearch`.

## Offline (no keys, no cost)

```sh
uv sync
uv run finresearch demo            # http://127.0.0.1:8770, use --port if 8770 is taken
```

The demo page replays a bundled snapshot of a real paired run: two Nemotron Nano supervisors (the original prompt and the GEPA-rewritten one) each watch and correct their own three research agents. These commands also work offline on a run directory you already have:

```sh
uv run finresearch timeline --run-dir <run dir>   # rebuild and print a run's curated drift timeline
uv run finresearch serve                          # JSON + server-sent-events API over your runs, port 8765
```

## Live (paid calls)

Live runs cost money: every step makes Crusoe inference calls, and each uncached research query is a paid You.com call (about $0.11 each at the default `deep` effort, up to `PITSTOP_MAX_RESEARCH_CALLS`, default 12, per run). Cached answers are reused for free.

```sh
cp .env.example .env                        # at the repo root; fill in CRUSOE_API_KEY and YDC_API_KEY
uv run finresearch check                    # one Crusoe call per configured model (--youcom: one paid You.com call)
uv run finresearch run --correct-on-drift   # the default scenario; --scenario PATH for another
```

| Command | What it does |
|---|---|
| `finresearch run [--scenario PATH] [--halt-on-drift] [--correct-on-drift] [--no-report]` | Run the swarm (`--mode replay` uses only the research cache) |
| `finresearch run --shadow-prompt <prompt.txt>` | Also judge every input with a second supervisor prompt, logged but never acting |
| `finresearch compare --run-dir <run dir> --prompt-b <prompt.txt>` | Re-judge a recorded run's supervisor inputs with two prompts; no new research calls |

Runs are written to `~/.pitstop/runs/<run_id>/` and the You.com cache to `~/.pitstop/cache/youcom/` (`PITSTOP_HOME`, `PITSTOP_RUNS_DIR` and `PITSTOP_CACHE_DIR` move them). The models are set with `PITSTOP_SUPERVISOR_MODEL`, `PITSTOP_WORKER_MODEL` and `PITSTOP_REPORT_MODEL` (see `.env.example`). An invalid `~/.pitstop/config.toml` makes `finresearch run` exit 2 before any paid call.

## Layout

```
finresearch/
  cli.py            the `finresearch` console script
  config.py         settings from env vars, and the packaged asset paths
  orchestrator.py   the swarm: agents, supervisor checkpoints, corrections, the report
  agents/           research worker agents and the report synthesizer
  supervisor/       drift supervisor: signals, LLM verdict + evidence validation, AlertPolicy, corrections
  research/         You.com Finance Research client with an on-disk cache
  timeline.py, server.py   the curated drift timeline and its API
  demo/             the GEPA comparison demo server, its static page and the bundled snapshot
  ui/               a prototype local run viewer (python -m finresearch.ui.server)
  assets/           prompts (baseline supervisor, GEPA, worker) and scenarios
```

The offline tests are in `tests/examples/financial_research/` at the repo root.
