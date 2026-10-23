# PitStop example: financial research swarm

The multi-agent research swarm PitStop started as at the hackathon, kept as an example built on the `pitstop` SDK. A scenario gives three research agents a shared objective (the default compares NVIDIA and AMD); they plan queries, call the You.com Finance Research API and record cited findings. A report agent then writes the final report. Scenarios can inject drift into an agent's directive, so a run shows it being caught.

The swarm is watched by the PitStop SDK with a few lines in `orchestrator.py` and `agents/worker.py`:

```python
async with pitstop.workflow("financial-research", goal=scenario.swarm_objective, criticality="medium",
                            tools=("youcom_research",)):
    ...  # each worker's loop runs inside: async with pitstop.agent(agent_id, goal=directive)

research = pitstop.tool(side_effects=True, name="youcom_research")(self.research.research)
res = await research(query)   # {"blocked_by": "pitstop", "reason": ...} when PitStop blocks the call
```

The You.com call costs money outside the process, so it is a side-effect tool. On this medium workflow, PitStop's L1 pre-action gate asks a judge model whether each research call serves the agent's directive before it runs. A rejected call is a P0 `out_of_scope_action` drift: the call does not run, the worker gets an ERROR event naming the reason and re-plans its next step. A `pitstop.Halted` (a budget hit, for example) ends the swarm, which prints the reason and records the run as halted.

The hackathon's per-step supervisor (a Nemotron model plus code-computed trajectory signals, which can block a query and send a correction or halt the agent) is still here, but opt-in: `finresearch run --legacy-supervisor` (still watched by PitStop). The demo page and the benchmarks use it with `Swarm(watch=False)`, which skips the PitStop wiring entirely, so they measure the supervisor alone with no gate and no judge calls.

It is a uv workspace member of the PitStop repo: the package is `finresearch` (distribution `pitstop-financial-research`) and its CLI is `finresearch`. A `uv sync` at the repo root installs both `pitstop` and `finresearch`.

## Offline (no keys, no cost)

```sh
uv sync
uv run finresearch run --offline   # the swarm watched by PitStop; the injected drift is blocked
uv run pitstop drifts              # lists the P0 out_of_scope_action drift (action block)
uv run finresearch demo            # http://127.0.0.1:8770, use --port if 8770 is taken
```

`finresearch run --offline` (`finresearch/offline.py`) uses a scripted worker model, canned research answers marked OFFLINE and a scripted judge for the gate (`pitstop.init(judge_client=...)`) that rejects calls about consumer gaming. In the default scenario the NVIDIA data-center agent is told from step 2 to research GeForce gaming instead; its gaming research calls are blocked and never run. `--no-inject-drift` removes the injection, and the run records no drift.

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
uv run finresearch run                      # the default scenario; --scenario PATH for another
```

Live, the gate's judge comes from `[judge]` in `~/.pitstop/config.toml`: by default `openai/gpt-oss-120b` on Crusoe, using `CRUSOE_API_KEY`, so each research call also costs one small judge call (on PitStop's `watch` meter, counted against your budgets). If the judge is missing or cannot decide, the medium workflow lets the call run (`on_gate_error` defaults to allow) and records a `gate_error` event.

| Command | What it does |
|---|---|
| `finresearch run [--scenario PATH] [--no-inject-drift] [--no-report]` | Run the swarm watched by PitStop (`--mode replay` uses only the research cache) |
| `finresearch run --legacy-supervisor [PROMPT] [--halt-on-drift] [--correct-on-drift]` | Also run the hackathon per-step supervisor (default prompt `supervisor_v0.txt`) |
| `finresearch run --legacy-supervisor --shadow-prompt <prompt.txt>` | Also judge every input with a second supervisor prompt, logged but never acting |
| `finresearch compare --run-dir <run dir> --prompt-b <prompt.txt>` | Re-judge a recorded run's supervisor inputs with two prompts; no new research calls |

Runs are written to `~/.pitstop/runs/<run_id>/` and the You.com cache to `~/.pitstop/cache/youcom/` (`PITSTOP_HOME`, `PITSTOP_RUNS_DIR` and `PITSTOP_CACHE_DIR` move them). The models are set with `PITSTOP_SUPERVISOR_MODEL`, `PITSTOP_WORKER_MODEL` and `PITSTOP_REPORT_MODEL` (see `.env.example`). An invalid `~/.pitstop/config.toml` makes `finresearch run` exit 2 before any paid call.

## Layout

```
finresearch/
  cli.py            the `finresearch` console script
  config.py         settings from env vars, and the packaged asset paths
  orchestrator.py   the swarm under pitstop.workflow; legacy supervisor checkpoints, corrections, the report
  offline.py        `run --offline`: scripted worker model, canned research, scripted gate judge
  agents/           research worker agents and the report synthesizer
  supervisor/       drift supervisor: signals, LLM verdict + evidence validation, AlertPolicy, corrections
  research/         You.com Finance Research client with an on-disk cache
  timeline.py, server.py   the curated drift timeline and its API
  demo/             the GEPA comparison demo server, its static page and the bundled snapshot
  ui/               a prototype local run viewer (python -m finresearch.ui.server)
  assets/           prompts (baseline supervisor, GEPA, worker) and scenarios
```

The offline tests are in `tests/examples/financial_research/` at the repo root.
