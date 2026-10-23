# PitStop Milestone 2c: Observe (OTel), Framework Adapters, L2/L5 Background Detection and the Example Move

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish milestone 2.
- PitStop can observe agents through OpenTelemetry, guard tools inside the OpenAI Agents SDK and LangChain/LangGraph, and write its verdicts back as span events.
- A background `scan` pass finds sudden or slow degradation with change-point statistics (L2) and says what changed (L5).
- The research swarm moves to `examples/financial-research/` and catches its injected drift through the SDK.
- Every item in the "Carried to milestone 2c" list of the M2b plan is closed or explicitly moved on.

**Architecture:**
- **One guard for every tool path.** The L0 + L1 logic now in `@pitstop.tool` moves to `pitstop.sdk.guard`. The decorator, the OpenAI Agents SDK tool guardrail and the LangChain middleware all call it.
- **Observe-only sources.** `pitstop.otel.SpanProcessor` writes runs and events from GenAI spans. Its findings are recorded with action `record`, because PitStop did nothing to the agent. It never writes the ledger and never raises.
- **Background detection.** `pitstop scan` (and later the milestone 3 server) reads finished runs once each:
  - computes per-run metrics from `events` (`detect/metrics.py`);
  - feeds one CUSUM detector per (workflow, metric), with state kept in `l2_state` (`detect/l2.py`);
  - on an alarm, compares runs before and after the change point by run tags (`detect/attribution.py`) and records a drift.
- **The example.** The swarm becomes its own uv workspace member (`pitstop-financial-research`, import `finresearch`). It is not shipped in the `pitstop` wheel.

**Tech Stack:** Python ≥3.11, uv (workspace), openai 3.x, anthropic 2.x, new optional extras: opentelemetry-api/-sdk 1.45, openai-agents ≥0.24, langchain ≥1.4 / langchain-core ≥1.7 (langchain-openai ≥1.7 as a dev dependency only), pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md`. This plan covers:
- §3: the OTel span processor, the two framework adapters, "emit, don't compete";
- §3.2: refusing new runs once a budget is exhausted;
- §5: the spend-rate P0 trigger;
- §6.2: L2 and L5;
- §10 milestone 2: the example wired in with a few lines and catching injected drift;
- §12: `examples/financial-research/`;
- the "Carried to milestone 2c" list at the end of `docs/superpowers/plans/2026-10-21-pitstop-m2b-hardening-gate-alerts.md`.

API facts for OTel, the Agents SDK and LangChain were checked against the released packages on 2026-10-22. They are summarized in each task's Interfaces block.

## Decisions this plan makes (design calls; the owner can revisit)

1. **One run per trace.** The OTel processor makes one PitStop run per trace, with id `otel-<32-hex trace id>`. The workflow name comes from, in order:
   - a `pitstop.workflow` attribute on the trace's local root span;
   - the `workflow=` argument of `SpanProcessor(...)`;
   - the root span's name.
2. **No double counting.** A span that starts or ends while a PitStop workflow run is current in that context is skipped: `wrap` and `@tool` already record it. OTel cost is an estimate on the event (price table), never a ledger row, so OTel can't trip a budget.
3. **Verdicts as span events.** Every recorded finding adds a `pitstop.verdict` event, best effort:
   - SDK path: on the current span;
   - OTel path: on the trace's still-open root span.

   Attributes: `pitstop.drift_type`, `pitstop.level`, `pitstop.action`, `pitstop.run_id`, `pitstop.reason` (≤300 chars), and `pitstop.drift_id` (when known). This is a no-op when opentelemetry is not installed.
4. **L2 is a two-sided tabular CUSUM** on standardized per-run metrics.
   - One stream per (workflow, metric), not per tag value: a model change must show up as a change, and L5 then explains it by tag.
   - The baseline (mean, std) is learned from the first 20 runs and then frozen. After an alarm the detector resets and relearns.
   - Slack `k = 0.5`; threshold `h` = 8 / 5 / 4 for sensitivity low / normal / high.
5. **L5 escalates.** An L2 drift whose attribution names a `model`, `base_url` or `prompt_versions` change is raised to P1 on high/medium workflows and to P2 on low ones. It is never lowered (spec §5: "sudden quality step change right after a model/provider/prompt change" is P1).
6. **The spend-rate P0 is inline and per run.** A run whose agent spend passes N× its workflow's baseline gets a P0 `spend_spike` drift with action `halt` on any criticality, like a budget hit.
   - The baseline is the median agent spend of the last 20 `ok` runs; at least 10 are needed, and it is floored at $0.01.
   - N is 10 / 5 / 3 for sensitivity low / normal / high.
7. **Refusing runs at the start.** Entering a workflow raises `BudgetExceeded` when one of these is already used up:
   - the workflow's own budget;
   - a global budget without `base_url`;
   - the global budget for the endpoint of the workflow's previous run.

   The run is recorded with status `refused`. An endpoint budget for an endpoint the run has not used yet still halts at its first call, as today.
8. **Fallback engages only on a model change** (clarifies M2b decision 3). A `base_url`-only change still records the P1 and alerts, but sending the pinned model id to a new provider is not a safe fix. M3's L4 canaries cover provider swaps.
9. **The example is a workspace member.** `examples/financial-research/` gets its own `pyproject.toml` (`pitstop-financial-research`, package `finresearch`, console script `finresearch`). The root project lists it as a uv workspace member and a dev dependency, so `uv run finresearch ...` and the benchmarks import it normally, with no `sys.path` hacks.
   - The `pitstop` CLI keeps only SDK commands; `pitstop run/check/compare/timeline/serve/demo` become `finresearch ...`.
   - `pitstop.llm` stays in the core package (spec §12 layout).
10. **The example's paid research call is a gated tool.** The You.com call is a `@pitstop.tool(side_effects=True, name="youcom_research")`: it spends money outside the process. On the medium example workflow this means the L1 gate judges it against the agent's directive, which is what catches the injected topic drift. The hackathon per-step supervisor stays, but only as an opt-in mode (`--legacy-supervisor`), for the demo page and the benchmarks.
11. **GEPA carry items move to milestone 4**, which owns the correction loop: the `stop_callbacks` stopper, `BaseException` handling and `max_workers > 1`.
12. **Observed (unmetered) LLM calls.** The LangChain middleware records a model call it saw but `wrap` did not meter. This happens when the chat model's client is not a wrapped one. The record is an `llm_call` event with `"observed": true` and an estimated cost; L0 runs on it, but no budget is enforced. The README says to pass a wrapped client for hard budgets.

## Global Constraints

- **Python ≥3.11. No new *required* runtime dependency.** New optional extras:
  - `pitstop[otel]` = `opentelemetry-api>=1.45`, `opentelemetry-sdk>=1.45`;
  - `pitstop[openai-agents]` = `openai-agents>=0.24`;
  - `pitstop[langchain]` = `langchain>=1.4`, `langchain-core>=1.7`.

  All of them, plus `langchain-openai>=1.7`, join the dev group. If resolving them upgrades `openai`, the full suite must still pass; record the resolved versions in the task report.
- **Optional imports.** Code that needs an optional extra imports it inside the adapter or processor module only. `import pitstop` must work without any extra installed (pinned by a subprocess test in Task 8).
- **No network in tests.** pytest-socket stays on. Use these instead:
  - HTTP fakes (`tests/sdk/http_fakes.py`, real SDK clients with `httpx.MockTransport`);
  - `agents.testing.ScriptedModel`;
  - a `FakeMessagesListChatModel` subclass whose `bind_tools` returns `self`;
  - an `InMemorySpanExporter`.
- **Never a real key, never real spend.** Subprocess tests run with `cwd=tmp_path`, and every key env var is scrubbed (the root `conftest.py` list).
- **Never delete untracked or git-ignored files.** Never run `examples/quickstart/agent.py` or `finresearch run` without `--offline`. Never use `git clean`, `git add -A` or `git add .`; add explicit paths.
- **Errors the SDK may raise.** It raises only `pitstop.Halted` / `pitstop.BudgetExceeded` on purpose:
  - PitStop's own failures are logged and never surface in user code (fail open);
  - an invalid config or a too-new store fails closed.

  Observe-only code (the OTel processor, verdict emission, `scan` inside the server later) never raises into its caller.
- **Secrets stay out.** A secret value never reaches the DB, events, span events, alert rows or log messages.
- **Commits.** No attribution trailers (no `Co-Authored-By`, no "Generated with"). Use conventional prefixes (`feat(otel):`, `fix(alerts):`, `refactor:`, `docs:`).
- **Style.** ruff `F` clean and line length 120. Match the surrounding docstring and comment density.
- **Docs ship with the feature.** Each user-facing task updates its README section, `docs/config.example.toml` (when config changes) and `CHANGELOG.md` (Unreleased) in the same task. The README describes only what works (spec §9.2).
- **Every task ends green.** `uv run pytest -q` and `uv run ruff check` pass at the end of every task.

## Review Focus

1. **A malformed span or a store error never breaks the traced app.** A span with missing, wrongly typed or huge attributes, or a store that is locked, leaves the traced app running with no exception and at most one warning. Pinned in Task 8: `test_processor_survives_bad_attributes_and_store_errors`.
2. **The same tool call guarded twice is counted once.** A `@pitstop.tool` function used inside an OpenAI Agents SDK tool that also has the PitStop guardrail would run L0 twice and count a loop at half the threshold. The guard marks the call it already checked: the inner `@tool` sees the mark and only runs the function. Pinned in Task 9: `test_guardrail_and_inner_tool_check_once`.
3. **A workflow whose metrics are flat zero (no tools, no refusals) never alarms on the first non-zero value.** The std floor holds. Pinned in Task 11: `test_floor_prevents_zero_std_alarm`.
4. **Two `pitstop scan` processes at once scan each run exactly once and keep one detector state.** Pinned in Task 13: `test_concurrent_scans_process_each_run_once`.
5. **Moving the swarm keeps every documented command working.** The README's run-the-example commands work offline, and the wheel no longer ships the swarm. Pinned in Task 2 (`test_wheel_ships_only_the_sdk`) and Task 3 (`test_finresearch_offline_run_blocks_injected_drift`, run as a subprocess through the console script).

---

### Task 1: Store v5, richer run and call records, SDK runtime hygiene

**Files:**
- Modify:
  - `src/pitstop/store/__init__.py` and `schema.sql`;
  - `src/pitstop/sdk/wrap/call.py`;
  - `src/pitstop/sdk/wrap/__init__.py` (`meter`);
  - `src/pitstop/sdk/tool.py`;
  - `src/pitstop/sdk/context.py`;
  - `src/pitstop/alerts/dispatch.py`.
- Test: `tests/store/test_store.py` (or the existing store test file), `tests/sdk/test_wrap.py`, `tests/sdk/test_tool.py`, `tests/sdk/test_context.py`, `tests/alerts/test_dispatch.py`

**Interfaces:**
- Produces:
  - **Schema v5** (migration step `_v5_observe` plus `schema.sql`, which must match a migrated DB):
    - `runs.scanned_at REAL`, `runs.fallback_from TEXT`;
    - `events.span_id TEXT`, `events.parent_span_id TEXT`;
    - table `l2_state(workflow TEXT, metric TEXT, state TEXT, updated_at REAL, PRIMARY KEY(workflow, metric))`;
    - indexes `idx_runs_wf_started ON runs(workflow, started_at)`, `idx_runs_scan ON runs(scanned_at, ended_at)` and `idx_ledger_run ON ledger(run_id)`.
  - **`llm_call` event payload** gains `prompt_tokens: int`, `completion_tokens: int` and `refused: bool`, with the same token values that were settled or estimated.
  - **Fallback runs are tagged.** When an agent-meter call is redirected to the pinned fallback model: `UPDATE runs SET fallback_from = COALESCE(fallback_from, <requested model>)`.
  - **The run's adapter tag.** The run's first agent call sets `adapter = COALESCE(adapter, 'wrap')`, so an adapter that tagged the run first keeps its tag.
  - `Store.close_local() -> None` becomes public (the current `_close_local`). The store tracks the connections it opened: `Store.open_connections() -> int`, for tests.

- [ ] **Step 1: Write the failing tests.**
  - `test_v4_db_migrates_to_v5`: open a DB built with the v4 `schema.sql` text kept as a test fixture string. Expected: version 5, new columns and table present.
  - `test_fresh_and_migrated_schemas_match`: compare `PRAGMA table_info` and the index list.
  - `test_llm_call_payload_has_tokens_and_refusal`: a refusal reply makes `payload["refused"] is True` and the tokens equal the ledger row.
  - `test_fallback_run_tagged`: an engaged fallback sets `runs.fallback_from` to the requested model.
  - `test_adapter_tag_not_overwritten`: pre-set `adapter='langchain'`; it stays after the first wrapped call.
  - `test_steer_limit_halt_carries_drift_id`: the third steer's `Halted.drift_id` is the drift that tripped it, and `ctx.halt_drift_id` matches.
  - `test_meter_exits_in_other_context`: enter `meter("watch")` in one `contextvars.Context`, exit in another. Expected: no ValueError, and the meter is back to `"agent"` there.
  - `test_alert_worker_closes_its_connection`: after `notify()` finishes, `store.open_connections()` is back to its value before.
  - `test_async_gen_workflow_in_reference_cycle`: an async-generator workflow abandoned inside a reference cycle and collected by `gc.collect()` while the loop runs. Expected: the run ends exactly once, and no "already running" error is logged.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/store tests/sdk tests/alerts -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - Add the v5 migration and bump `SCHEMA_VERSION = 5`.
  - The alerts pool's job wrapper calls `store.close_local()` in a `finally`. A background thread must never keep a connection after its job.
  - `meter()` reset falls back to `_meter.set(prev)` on ValueError, like `_GenRun._pop`.
  - The steer-limit halt sets `ctx.halt_drift_id` from the deciding drift when it is still None.
  - The async-generator race: if it can't be closed without patching the event loop, replace the test with one that pins the current behavior, document the limitation in the `context.py` module docstring and the README "Known limitations", and say so in the report.
- [ ] **Step 4: Fix the two M2b test-hygiene items.**
  - The fork test gets a `waitpid` with a 10 s deadline (`os.WNOHANG` polling), killing the child on timeout.
  - `test_chat_stream_helper_caller_stream_options_keeps_usage` exits its manager.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(store): schema v5 for scan state, span links and run tags; runtime hygiene"` (explicit paths).

---

### Task 2: Move the research swarm to `examples/financial-research/`

**Files:**
- Create:
  - `examples/financial-research/pyproject.toml`;
  - `examples/financial-research/README.md`;
  - `examples/financial-research/finresearch/__init__.py`, `config.py`, `cli.py`;
  - `src/pitstop/detect/text.py`.
- Move with `git mv` into `examples/financial-research/finresearch/`:
  - from `src/pitstop/`: `orchestrator.py`, `collector.py`, `schemas.py`, `timeline.py`, `server.py`, `agents/`, `research/`, `demo/`, `ui/`, `supervisor/` (all four modules);
  - assets: `assets/scenarios/`, `assets/prompts/worker/`, `assets/prompts/supervisor_v0.txt`, `assets/prompts/gepa/` → `finresearch/assets/...`.
- Modify:
  - `src/pitstop/config.py`: keep `PACKAGE_DIR`, `ASSETS`, `PROMPTS`, `pitstop_home` and the dotenv load; `Settings`, `load_settings`, `SCENARIOS` and `SNAPSHOT` move to `finresearch/config.py`;
  - `src/pitstop/cli.py`: drop `check`/`run`/`compare`/`timeline`/`serve`/`demo`; they move to `finresearch/cli.py`;
  - `src/pitstop/detect/l0.py`: `jaccard` now from `.text`;
  - the root `pyproject.toml`: workspace, dev dependency, `[tool.uv.sources]`, description "Drift detection, urgency labels and prompt correction for LLM agent workflows";
  - `benchmarks/**` imports;
  - `tests/test_repo_hygiene.py`, `tests/runtime/test_packaging.py`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `uv.lock`.
- Move tests: `tests/runtime/*` → `tests/examples/financial_research/`; `tests/test_cli_run_config.py` → same folder, testing `finresearch run`.

**Interfaces:**
- Produces:
  - `pitstop.detect.text.jaccard(a: str, b: str) -> float` (and the private `_tokens`). `finresearch.supervisor.signals` imports it from there.
  - Console script `finresearch` with the six moved commands and their flags unchanged.
  - `finresearch.config.load_settings()` and the `Settings` dataclass, unchanged.
- The `pitstop` wheel contains only `pitstop/` (`[tool.hatch.build.targets.wheel] packages = ["src/pitstop"]` stays). Its `assets/prompts` keeps the `l1_gate_*` files only.

- [ ] **Step 1: Write the failing tests.**
  - `test_wheel_ships_only_the_sdk`: the wheel has `pitstop/assets/prompts/l1_gate_v1.txt` and has no `pitstop/demo/`, `pitstop/ui/`, `pitstop/orchestrator.py` or `pitstop/assets/scenarios/`.
  - `test_finresearch_assets_resolve_from_any_cwd`: moved from `test_assets_resolve_from_any_cwd`, now against `finresearch.config`.
  - `test_core_does_not_import_example`: in a subprocess, `import pitstop, pitstop.cli` must not put any `finresearch` module in `sys.modules`.
  - `test_superseded_paths_are_gone`: extended with the old `src/pitstop/...` paths.
- [ ] **Step 2: Do the move.** Use `git mv` so history follows, then fix imports: relative imports inside `finresearch` and `pitstop.*` for SDK pieces.
- [ ] **Step 3: Set up the workspace.**
  - Root `pyproject.toml`: `[tool.uv.workspace] members = ["examples/financial-research"]`; add `"pitstop-financial-research"` to the dev group; `[tool.uv.sources] pitstop-financial-research = { workspace = true }`.
  - The example's `pyproject.toml` depends on `pitstop` (workspace source), uses hatchling, and has `packages = ["finresearch"]` and `[project.scripts] finresearch = "finresearch.cli:main"`.
  - Run `uv lock`.
- [ ] **Step 4: Update the docs.**
  - README: run-the-example commands (`uv run finresearch run ...`), repo map, the "How it works today" swarm section and the benchmarks commands.
  - CONTRIBUTING: the workspace note.
  - `examples/financial-research/README.md`: what the example is, its commands, offline vs live, and that live runs cost money.
  - CHANGELOG "Changed".
- [ ] **Step 5: Verify.** Run `uv sync && uv run finresearch --help && uv run pitstop --help && uv run pytest -q && uv run ruff check`. Expected: both CLIs list their commands, and everything passes.
- [ ] **Step 6: Commit.** Run `git commit -m "refactor: move the research swarm to examples/financial-research (finresearch)"`.

---

### Task 3: Wire the example onto the SDK; catch the injected drift offline

**Files:**
- Modify: `finresearch/orchestrator.py`, `finresearch/agents/worker.py` (only where the research call is made), `finresearch/cli.py`
- Create: `finresearch/offline.py`
- Modify docs: `examples/financial-research/README.md`, `README.md` Quickstart, CHANGELOG
- Test: `tests/examples/financial_research/test_sdk_wiring.py`

**Interfaces:**
- Consumes: `pitstop.workflow`, `pitstop.agent`, `pitstop.tool`, `pitstop.init(judge_client=...)`, `Halted`.
- Produces:
  - `Swarm(..., supervisor: Supervisor | None = None, ...)`: None (the default) means the SDK alone watches the run.
  - `Swarm.run_all()` runs inside `pitstop.workflow("financial-research", goal=scenario.swarm_objective, criticality="medium", tools=("youcom_research",))`. Each worker's loop runs inside `pitstop.agent(agent_id, goal=spec.directive)`.
  - The You.com call goes through `@pitstop.tool(side_effects=True, name="youcom_research")`. A block reply (`{"blocked_by": "pitstop", ...}`) reaches the worker as an `ERROR` trajectory event naming the reason, and the worker re-plans its next step. `Halted` ends the swarm, which prints the reason and records the run as halted.
  - `finresearch run --offline [--inject-drift/--no-inject-drift]` (inject defaults on) runs on `finresearch.offline`:
    - a scripted worker LLM that follows the injection: from the injection step it proposes GeForce gaming queries;
    - a canned research client;
    - a scripted judge for `pitstop.init(judge_client=...)` that rejects any proposed call whose args mention "gaming" or "GeForce", with reason "about consumer gaming, not the directive".
  - `finresearch run --legacy-supervisor [PROMPT]` turns the hackathon per-step supervisor back on (default prompt `supervisor_v0.txt`).

- [ ] **Step 1: Write the failing tests.**
  - `test_injected_drift_blocked_and_recorded`: in-process offline run. Expected:
    - one open `out_of_scope_action` drift at P0 with action `block`;
    - `block` events whose `agent` is `agent_nvda_dc`;
    - no block events for the other two agents;
    - run status `ok` (medium: block, not halt).
  - `test_finresearch_offline_run_blocks_injected_drift`: subprocess `uv run finresearch run --offline` with `PITSTOP_HOME=tmp`, `cwd=tmp`, keys scrubbed. Expected: exit 0, stdout names the block, and `pitstop drifts` in the same home lists it.
  - `test_clean_offline_run_records_no_drift`: `--no-inject-drift` records zero drifts.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/examples -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the wiring (a few lines in the orchestrator, as in spec §3) and `offline.py`. Live runs keep needing `CRUSOE_API_KEY` and `YDC_API_KEY`.
- [ ] **Step 4: Update the docs.**
  - README Quickstart "Run the example": the offline command, its output, then `uv run pitstop drifts`.
  - Example README: how live mode is gated by the judge.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(example): financial-research swarm watched by the SDK; offline injected-drift demo"`.

---

### Task 4: Refuse runs on an exhausted budget; the spend-spike P0

**Files:**
- Create: `src/pitstop/detect/spend.py`
- Modify:
  - `src/pitstop/budget/ledger.py`;
  - `src/pitstop/sdk/context.py` (`_Workflow.__enter__`, `RunContext`);
  - `src/pitstop/sdk/wrap/call.py`;
  - `src/pitstop/detect/urgency.py`;
  - `src/pitstop/sdk/actions.py`;
  - README budget section, CHANGELOG.
- Test: `tests/budget/test_ledger.py`, new `tests/detect/test_spend.py`, `tests/sdk/test_context.py`

**Interfaces:**
- Produces:
  - `Ledger.exhausted(*, workflow: str, workflow_budget_usd: float | None, base_url: str | None) -> BudgetExceeded | None`. It checks, in this order, and returns the first scope whose spent ≥ limit − `_EPS`:
    - the workflow scope (when a budget is set);
    - every global budget without `base_url`;
    - global budgets matching `base_url` (skipped when `base_url` is None).
  - `_Workflow.__enter__`:
    - reads the workflow's latest run's `base_url` (via `idx_runs_wf_started`);
    - if `exhausted(...)` returns an exception: inserts the run with status `refused`, records a `budget_exhausted` drift through `apply_findings`, and raises it;
    - a store or ledger error here is logged and the run starts (fail open); `ConfigInvalid` still fails closed.
  - `detect/spend.py`:

```python
SPEND_MULTIPLIER = {"low": 10.0, "normal": 5.0, "high": 3.0}
BASELINE_RUNS = 20
MIN_BASELINE_RUNS = 10
MIN_BASELINE_USD = 0.01
def baseline_run_cost(store, workflow: str, *, before: float) -> float | None: ...
def check_spend(run_usd: float, baseline: float | None, sensitivity: Sensitivity) -> Finding | None: ...
```

  - **`baseline_run_cost`.** The median of settled agent-meter ledger USD per run, over the last `BASELINE_RUNS` runs of the workflow with status `ok` that started before `before`. None when fewer than `MIN_BASELINE_RUNS` runs qualify.
  - **`check_spend`.** Returns `Finding("spend_spike", "degraded", False, {"run_usd", "baseline_usd", "multiplier"})` when `run_usd > multiplier * max(baseline, MIN_BASELINE_USD)`.
  - **New `RunContext` fields:**
    - `spend_usd: float = 0.0`, the run's settled agent spend;
    - `spend_baseline`: the baseline, computed once at the first settle and cached;
    - `spend_spiked: bool = False`, so the finding is applied at most once per run.
  - **Urgency.** `classify` treats `("spend_spike", None)` as P0 on every criticality. `action_for` returns `"halt"` for `budget_exhausted` and `spend_spike`.
  - **Pending halt.** `apply_findings` now sets the pending halt when `action == "halt"` **or** (P0 and high), so the next wrapped call raises `Halted`.

- [ ] **Step 1: Write the failing tests.**
  - `test_exhausted_workflow_budget_refuses_run`: a $1 budget with $1 settled today. Entering raises `BudgetExceeded` (scope `workflow:<name>`), a run row has status `refused`, and a P0 `budget_exhausted` drift exists.
  - `test_endpoint_budget_checked_from_previous_run`: a global budget for URL A is exhausted and the previous run used A. Expected: refused. When the previous run used B: the run starts.
  - `test_refusal_check_fails_open`: the ledger raises `sqlite3.OperationalError`. Expected: the run starts and a warning is logged.
  - `test_spend_spike_halts_next_call`: 10 ok runs at $0.02 each; a run that settles $0.11 (normal, 5×) records a P0 `spend_spike` with action `halt`, and its next wrapped call raises `Halted`.
  - `test_no_spike_below_min_baseline_runs`: 9 runs give no finding.
  - `test_spike_applied_once_per_run`.
  - `test_spike_on_low_criticality_still_halts`.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/budget tests/detect tests/sdk -q`. Expected: FAIL. Existing tests that expected the first call of a run to raise on an already-spent workflow budget now see entry raise; update them, since the exception type and scope are unchanged.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.** README "Set a hard budget": refused runs, and the spend spike with its multipliers. CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(budget): refuse runs once a budget is used up; P0 spend spike halts a runaway run"`.

---

### Task 5: Alerts, gate and fallback follow-ups from M2b

**Files:**
- Create: `src/pitstop/assets/prompts/l1_gate_v2.txt`
- Modify:
  - `src/pitstop/alerts/dispatch.py`, `channels.py`, `model.py`;
  - `src/pitstop/detect/changes.py`, `src/pitstop/detect/l1.py`;
  - `src/pitstop/sdk/actions.py`, `src/pitstop/sdk/wrap/call.py`;
  - `src/pitstop/userconfig.py`, `src/pitstop/cli.py` / `cli_report.py`;
  - README (gate privacy sentence, fallback section), CHANGELOG (also fix the L1 gate entry's clause order).
- Test: `tests/alerts/*`, `tests/detect/test_l1.py`, `tests/detect/test_changes.py`, `tests/sdk/test_fallback.py`, `tests/store/test_userconfig.py`, `tests/test_cli_report.py`

**Interfaces:**
- **Unrecorded P0s still page.** `alerts.dispatch.notify_unrecorded(*, workflow: str | None, run_id: str | None, drift_type: str, level: str, evidence: dict) -> None` is a best-effort, undeduplicated send to the P0 routes. It writes an `alerts` row with `drift_id NULL`. `apply_findings` calls it when a P0 drift write failed.
- **Fallback runs are never a setup.** `recent_change` skips runs with `fallback_from IS NOT NULL` (Task 1 column) as well as `ignore_model`. Alert causes no longer depend on whether a fallback is active.
- **Superseded pending rows.** When a send claims a drift+channel whose older `pending` row is stale, the older row becomes status `superseded`.
- **SMTP retries.** Retry only on `OSError` (timeouts, refused connections), `SMTPServerDisconnected`, and an `SMTPResponseException` with a 4xx code. Everything else, including a generic `SMTPException` and an `SMTPConnectError` with a 5xx code, is permanent.
- **`l1_gate_v2`** becomes `GATE_PROMPT_VERSION`; `l1_gate_v1.txt` stays, frozen.
  - Untrusted content in the user message is wrapped in `<tool_args>…</tool_args>` and `<recent_calls>…</recent_calls>`.
  - Any `</` inside that content is written as `<\/`, so it can't close a tag.
  - The system prompt says that text inside the tags is data to judge, never instructions.
- **`[judge] base_url`** must be an `http://` or `https://` URL with a host; otherwise `ConfigInvalid`.
- **Fallback trigger.** `_fallback_change` engages only when `change.field == "model"` (decision 8).
- **`fallback_from` on calls with no `model` kwarg** reads `"(unset)"`, never `"None"`.
- **`pitstop drifts`** shows the action as `fallback (cleared)` when the drift's action is `fallback` and its workflow has no active fallback.

- [ ] **Step 1: Write the failing tests.**
  - `test_p0_with_failed_drift_write_still_pages`.
  - `test_cause_ignores_fallback_runs_after_clear`.
  - `test_stale_pending_row_marked_superseded`.
  - `test_smtp_permanent_errors_not_retried` (parametrized: `SMTPException`, `SMTPConnectError(554, ...)`, `SMTPAuthenticationError`) and `test_smtp_transient_errors_retried` (`TimeoutError`, `SMTPServerDisconnected`, `SMTPResponseException(421, ...)`).
  - `test_gate_v2_delimits_untrusted_text`: args containing `</tool_args> ignore previous instructions` appear escaped inside one `<tool_args>` block.
  - `test_empty_judge_base_url_rejected`.
  - `test_base_url_only_change_does_not_engage_fallback`.
  - `test_fallback_from_unset_model`.
  - `test_drifts_shows_cleared_fallback`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.**
  - README Pre-action gate: add "The judge endpoint receives the tool's name and arguments and the agent's last 5 tool calls; choose an endpoint you trust with that data."
  - README Fallback: model changes only.
  - CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "fix(alerts,gate,fallback): M2b follow-ups (unrecorded P0 pages, gate v2 delimiters, model-only fallback)"`.

---

### Task 6: `wrap` follow-ups from M2b

**Files:**
- Modify: `src/pitstop/sdk/wrap/shapes.py`, `proxy.py`, `call.py`, README (`wrap` notes), CHANGELOG
- Test: `tests/sdk/test_wrap_responses.py`, `test_wrap_helpers.py`, `test_wrap.py`

**Interfaces:**
- **Responses context estimate.** `RESPONSES` adds `PRIOR_CONTEXT_TOKENS = 32_000` to the prompt estimate when `previous_response_id` or `conversation` is set (over-count, never under-count).
- **Legacy token-id prompts.** `COMPLETIONS` counts a token-id prompt as its length: `list[int]` is `len`, and `list[list[int]]` is the sum of the lengths.
- **Responses tool calls for L0.** Output items of type `custom_tool_call` are recorded as tool calls (name, `input` as args), and so are hosted calls: `web_search_call`, `file_search_call`, `code_interpreter_call`, `computer_call`, `mcp_call` (named by its `name`), `image_generation_call` and `local_shell_call` (named by type, args `""`, never a parse failure).
- **Responses refusals.** A message content part of type `refusal` makes the call refused.
- **Unentered streaming responses.** A `with_streaming_response` reservation whose manager is never entered is released when the manager object is garbage collected, by the same mechanism as the stream helper.
- **`Mode`.** `Mode` becomes a `Literal[...]` of the modes `call.py` already uses.
- **Snapshots.** The module docstring and the README state that `event.snapshot` / `current_completion_snapshot` of chat stream helpers show PitStop's injected usage, and that `current_completion_snapshot` returns a fresh copy on each access.
- **Dropped (already documented or pinned):** Anthropic premium billing modes; `tool_runner`'s `runner._client`.

- [ ] **Step 1: Write the failing tests.**
  - `test_responses_estimate_counts_prior_context`.
  - `test_completions_token_id_prompt_estimate`.
  - `test_responses_hosted_and_custom_tool_calls_recorded`.
  - `test_responses_refusal_part_counts`.
  - `test_unentered_streaming_response_releases_reservation`.
  - `test_async_enter_error_releases` (async twins of the existing sync enter-error tests).
- [ ] **Step 2: Run them.** Expected: FAIL (the async twin may pass; keep it).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(wrap): M2b follow-ups (prior-context estimate, hosted tool calls, refusal parts, unentered streams)"`.

---

### Task 7: One tool guard for every path; observe-only findings; verdict span events

**Files:**
- Create: `src/pitstop/sdk/guard.py`, `src/pitstop/otel/__init__.py`, `src/pitstop/otel/emit.py`
- Modify: `src/pitstop/sdk/tool.py` (becomes a thin decorator over `guard`), `src/pitstop/sdk/actions.py`, `pyproject.toml` (the `otel` extra and dev dependencies)
- Test: new `tests/sdk/test_guard.py`, new `tests/otel/test_emit.py`; existing `tests/sdk/test_tool*.py` pass unchanged

**Interfaces:**
- Produces in `sdk/guard.py`:

```python
@dataclass(frozen=True)
class GuardDecision:
    action: Literal["run", "block", "steer"]
    reply: Any = None   # block: the {"blocked_by": "pitstop", "reason": ...} dict; steer: the corrective text
def tool_args_text(args: tuple, kwargs: dict) -> str: ...        # the canonical form @tool uses today
def guard_call(ctx: RunContext, tool: str, side_effects: bool, args_text: str, *,
               agent: str | None = None) -> GuardDecision: ...    # sync: L0, then the L1 gate inline
async def aguard_call(ctx: RunContext, tool: str, side_effects: bool, args_text: str, *,
                      agent: str | None = None) -> GuardDecision: ...  # the gate runs in a worker thread
def record_tool_result(ctx: RunContext, tool: str, *, result: Any = None, error: BaseException | None = None,
                       agent: str | None = None) -> None: ...
```

  - **Behavior.** Exactly today's `_before`/`_gate`/`_gate_error`: the same events, steer limit, and `Halted` raised.
  - **`agent=`** overrides `ctx.agent`, both as the tool-history key and in the events' `agent` column, for adapters that know the framework's agent name.
  - **Checked once.** While a guarded call's function runs, a ContextVar marks `(run_id, tool)` as already checked. A nested `@pitstop.tool` with the same name inside it skips its own guard and only records the result (Review Focus 2).
- **`apply_findings(ctx, findings, *, run_id=None, spec: WorkflowSpec | None = None, observe_only: bool = False)`:**
  - `spec` supplies the workflow name, criticality and sensitivity when `ctx` is None;
  - `observe_only=True` keeps the level but forces action `"record"`: no pending halt, no fallback, no steer, no block;
  - after recording, each applied finding calls `pitstop.otel.emit.emit_verdict(...)`.
- **`otel/emit.py`:** `emit_verdict(*, run_id, drift_type, level, action, drift_id, reason, span=None) -> None`.
  - It imports `opentelemetry.trace` lazily and uses `span` or the current span.
  - It adds event `pitstop.verdict` with the decision-3 attributes when the span `is_recording()`.
  - It swallows every error.
- **Extras.** `pyproject.toml`: `[project.optional-dependencies] otel = ["opentelemetry-api>=1.45", "opentelemetry-sdk>=1.45"]`, added to dev.

- [ ] **Step 1: Write the failing tests.**
  - `test_guard_call_block_steer_run`, matching the `@tool` behavior on the same inputs.
  - `test_guard_agent_override_keys_history`.
  - `test_nested_tool_checked_once`: a `@tool("t")` called inside `guard_call(..., "t")` → run; history grows by 1.
  - `test_observe_only_records_without_acting`: a P0 off-task finding observed gives a drift with action `record`, `halt_reason` stays None and no `block` event.
  - `test_verdict_event_on_current_span`: use `TracerProvider` + `SimpleSpanProcessor(InMemorySpanExporter())`; a blocked tool call inside an open span leaves a `pitstop.verdict` event with `pitstop.action == "block"`.
  - `test_emit_without_otel_installed_is_noop`: `monkeypatch.setitem(sys.modules, "opentelemetry", None)`.
- [ ] **Step 2: Run them.** Expected: FAIL. The existing tool tests still PASS before and after.
- [ ] **Step 3: Implement.** Move the code, don't rewrite it. `tool.py` keeps its public behavior and its module docstring, pointing to `guard`.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "refactor(sdk): one tool guard for decorator and adapters; observe-only findings; verdicts as span events"`.

---

### Task 8: OpenTelemetry span processor

**Files:**
- Create: `src/pitstop/otel/semconv.py`, `src/pitstop/otel/processor.py`; export `SpanProcessor` from `pitstop/otel/__init__.py`
- Modify: README (new "Already tracing with OpenTelemetry?" section), CHANGELOG
- Test: new `tests/otel/test_semconv.py`, `tests/otel/test_processor.py`

**Interfaces:**
- **Consumes:**
  - `apply_findings(..., spec=..., observe_only=True)` and `emit_verdict(span=...)` (Task 7);
  - `check_tool_call` and `check_llm_call` (L0);
  - `price_for` / `cost_usd`;
  - the Task 1 columns.
- **OTel facts (opentelemetry-sdk 1.45.1):**
  - Subclass `opentelemetry.sdk.trace.SpanProcessor`: `on_start(span, parent_context=None)`, `on_end(span: ReadableSpan)`, `shutdown()`, `force_flush(timeout_millis=30000) -> bool`.
  - A `ReadableSpan` cannot take events. Events go only on a live span, which is why the processor keeps a reference to each trace's local root from `on_start`.
- **`semconv.py`:**

```python
@dataclass(frozen=True)
class SpanView:
    kind: Literal["llm", "tool", "agent", "other"]
    model: str | None
    server: str | None              # server.address
    input_tokens: int | None
    output_tokens: int | None
    finish_reasons: tuple[str, ...]
    tool_name: str | None
    tool_args: str | None
    agent_name: str | None
    error: str | None               # error.type, or "error" when the status is ERROR
def read_span(name: str, attributes: Mapping[str, Any], status_is_error: bool) -> SpanView: ...
```

  The reader tolerates both naming generations. The first present name wins:

  | Field | Attribute names, in order |
  |---|---|
  | kind | `gen_ai.operation.name` (`chat` / `text_completion` / `generate_content` → llm, `execute_tool` → tool, `invoke_agent` → agent); `openinference.span.kind` (LLM / TOOL / AGENT); `traceloop.span.kind` (tool / agent); `llm.request.type` (chat / completion → llm) |
  | model | `gen_ai.response.model`, `gen_ai.request.model`, `llm.model_name`, `llm.response.model_name` |
  | input tokens | `gen_ai.usage.input_tokens`, `gen_ai.usage.prompt_tokens`, `llm.token_count.prompt` |
  | output tokens | `gen_ai.usage.output_tokens`, `gen_ai.usage.completion_tokens`, `llm.token_count.completion` |
  | tool name | `gen_ai.tool.name`, `tool.name` |
  | tool args | `gen_ai.tool.call.arguments`, `input.value` (TOOL spans) |
  | agent name | `gen_ai.agent.name` |
  | finish reasons | `gen_ai.response.finish_reasons`, `llm.response.finish_reason` |

  Wrong types are ignored (a token count that isn't an int gives None). Strings are truncated to 2000 chars.
- **`processor.py`:** `SpanProcessor(workflow: str | None = None, *, goal: str | None = None, criticality="medium", sensitivity="normal", tools: Sequence[str] | None = None)`. It builds a `WorkflowSpec` per resolved workflow name.
  - **`on_start`:**
    - skipped when `pitstop.sdk.context.current_run()` is not None;
    - a local root (no parent, or a remote parent) creates the run row (id `otel-<trace hex>`, `adapter='otel'`, status `running`, `started_at` = span start) and keeps the root span;
    - every span records `span_id → (parent_span_id, agent_name)` for its trace.
  - **`on_end`**, for spans of known traces:
    - **llm spans** → an `llm_call` event: payload `tool_calls` (1 when `finish_reasons` contains `tool_calls`, else 0: span attributes carry no count), `finish_reason`, `prompt_tokens`, `completion_tokens`, `refused` (finish reason `content_filter` or `refusal`), `meter: "agent"`, `source: "otel"`. `cost_usd` comes from the price table. The first one sets `runs.model` and `runs.base_url` (from `server`). L0 `check_llm_call` runs.
    - **tool spans** → `tool_call` and `tool_result` events (`{"error": ...}` when the span errored). L0 `check_tool_call` runs with the trace's per-agent history.
    - **agent spans** → an `agent` event named after the agent.
    - Every event gets `span_id` / `parent_span_id` in hex, and `agent` = the nearest agent ancestor's name.
    - Findings go through `apply_findings(None, findings, run_id=..., spec=..., observe_only=True)` and `emit_verdict(span=<root>)`.
    - When the root ends, the run gets `ended_at` and status `ok` or `error`, and the trace's state is dropped.
  - **Bounds.** At most `MAX_OPEN_TRACES = 1000` traces are kept; the oldest is dropped, with one warning per processor. At most `MAX_SPANS_PER_TRACE = 10_000` spans are mapped.
  - **Never fails.** Every entry point is wrapped so the processor never raises (a logged warning at most).
  - `shutdown()` drops all state; `force_flush()` returns True.

- [ ] **Step 1: Write the failing tests** (a real `TracerProvider`, the processor plus an `InMemorySpanExporter`; spans built with the tracer and attributes as each convention writes them).
  - `test_semconv_reads_current_genai_names`, `test_semconv_reads_old_genai_names`, `test_semconv_reads_openinference`, `test_semconv_reads_openllmetry`.
  - `test_trace_becomes_run_with_events`: an `invoke_agent` root with a chat child and an `execute_tool` child gives:
    - one run with the right workflow, model and base_url;
    - three events linked by span ids;
    - `agent` filled on the children;
    - status `ok`.
  - `test_workflow_name_resolution`: the attribute, then the argument, then the root name.
  - `test_tool_loop_in_trace_records_drift_and_root_event`: three identical `execute_tool` spans under a normal workflow give an `unproductive_loop` drift with action `record`, and the exported root span carries a `pitstop.verdict` event.
  - `test_spans_inside_pitstop_run_skipped`.
  - `test_processor_never_writes_ledger`.
  - `test_processor_survives_bad_attributes_and_store_errors` (Review Focus 1): token counts as strings, a 1 MB args string, and the store's `transaction` raising `sqlite3.OperationalError`.
  - `test_open_traces_bounded`.
  - `test_import_pitstop_without_extras`: in a subprocess with `opentelemetry`, `agents` and `langchain` blocked via a `sys.meta_path` finder, `import pitstop` and `pitstop.wrap` work.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.**
  - README section: `pip install "pitstop[otel]"`, the three lines (`from pitstop.otel import SpanProcessor`; `provider.add_span_processor(SpanProcessor(workflow="..."))`).
  - What is observed and what still needs `wrap`/`@tool` (budgets, blocking).
  - The `pitstop.verdict` event and its attributes.
  - CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(otel): span processor reads GenAI spans into runs, L0 on traces, verdicts on the root span"`.

---

### Task 9: OpenAI Agents SDK adapter

**Files:**
- Create: `src/pitstop/sdk/adapters/__init__.py` (empty docstring), `src/pitstop/sdk/adapters/openai_agents.py`
- Modify: `pyproject.toml` (the `openai-agents` extra and dev), README (new "OpenAI Agents SDK" section), CHANGELOG
- Test: new `tests/adapters/test_openai_agents.py`

**Interfaces:**
- **Agents SDK facts (openai-agents 0.24.0):**
  - **Guardrails.** `@tool_input_guardrail` functions take `ToolInputGuardrailData(context: ToolContext, agent)`. `context.tool_name`, `context.tool_arguments` (raw JSON string) and `context.tool_call_id` are available.
  - **Their results.** They return `ToolGuardrailFunctionOutput.allow()`, `.reject_content(message)` (the tool is not run, and the message is the tool result) or `.raise_exception()`.
  - **Scope and order.** They apply to `FunctionTool`s only, through `FunctionTool.tool_input_guardrails`. A rejected call fires no `on_tool_start`/`on_tool_end`.
  - **Hooks.** `RunHooks.on_agent_start(context, agent)` is async, passed as `Runner.run(..., hooks=...)`.
  - **Offline model.** `agents.testing.ScriptedModel`, `function_call(name, args, call_id=...)` and `assistant_message(text)`.
- **Produces:**

```python
def tool_guardrail(*, side_effects: bool = False) -> ToolInputGuardrail: ...
def guard(agent: Agent, *, side_effects: Iterable[str] = ()) -> Agent: ...   # adds the guardrail to every FunctionTool
class Hooks(RunHooks):                                                       # names the current agent; tags the run
    async def on_agent_start(self, context, agent) -> None: ...
```

  - **The guardrail** (async):
    - with no current PitStop run, it returns `allow()`;
    - otherwise it calls `aguard_call(ctx, context.tool_name, side_effects, args_text, agent=agent.name)`, where `args_text` is `tool_args_text((), parsed_kwargs)` (raw text when the JSON doesn't parse);
    - `run` → `allow()`; `block` / `steer` → `reject_content(<reply as text>)`, a block dict as compact JSON.
    - `Halted` must reach the caller of `Runner.run` as `pitstop.Halted`. If the SDK wraps exceptions from guardrails, unwrap it in the adapter (document how) and keep the test.
  - **`guard(agent, side_effects=...)`** marks the named tools `side_effects=True`. It is idempotent: it never adds the guardrail twice.
  - **`Hooks.on_agent_start`** sets the agent name for later model calls in this task (`pitstop.agent(agent.name)` semantics, replacing the previous adapter-set agent) and sets `runs.adapter = 'openai-agents'`.
  - **Metering.** Users pass a wrapped client: `OpenAIResponsesModel(model, openai_client=pitstop.wrap(AsyncOpenAI()))` or `set_default_openai_client(pitstop.wrap(AsyncOpenAI()), use_for_tracing=False)`.

- [ ] **Step 1: Write the failing tests** (`pytest.importorskip("agents")`).
  - `test_off_scope_tool_rejected_with_reason`: a ScriptedModel calls a tool outside `tools=`; the model's next input contains the PitStop block text, and a drift is recorded with `agent` = the Agent's name.
  - `test_side_effect_tool_gated`: a scripted judge rejects; the tool body never runs.
  - `test_pending_halt_reaches_caller`: a high workflow with an off-task side-effect call raises `pitstop.Halted` from `Runner.run`.
  - `test_guardrail_and_inner_tool_check_once` (Review Focus 2): the function tool's body is a `@pitstop.tool` of the same name; two identical calls give one history entry each, not two.
  - `test_wrapped_client_metered_by_agents_sdk`: a real `AsyncOpenAI` with `httpx.MockTransport` (Responses API body from `tests/sdk/http_fakes.py`), wrapped, inside `OpenAIResponsesModel`, gives a settled ledger row and an `llm_call` event with `agent` set by `Hooks`. If the SDK rejects the proxy type, fix `wrap` so the proxy passes the SDK's checks (no monkeypatching of the SDK), and say how in the report.
  - `test_outside_workflow_allows`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.** README section: install the extra, the few lines (`agent = guard(Agent(...), side_effects={"send_email"})`, `Runner.run(agent, q, hooks=Hooks())` inside `pitstop.workflow`, the wrapped client), and the limits (function tools only). CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(adapters): OpenAI Agents SDK tool guardrail and run hooks"`.

---

### Task 10: LangChain / LangGraph adapter

**Files:**
- Create: `src/pitstop/sdk/adapters/langchain.py`
- Modify: `pyproject.toml` (the `langchain` extra; dev also gets `langchain-openai>=1.7`), README (new "LangChain / LangGraph" section), CHANGELOG
- Test: new `tests/adapters/test_langchain.py`

**Interfaces:**
- **LangChain facts (langchain 1.4.7, langchain-core 1.7.0):**
  - **Middleware.** `langchain.agents.middleware.AgentMiddleware` hooks: `before_agent`, `before_model`, `wrap_model_call(request, handler)` and `wrap_tool_call(request, handler)`, each with an `a`-prefixed async twin. They are attached with `create_agent(model, tools, middleware=[...])`.
  - **Tool call requests.** `ToolCallRequest.tool_call` is a dict with `name`, `args` (dict) and `id`.
  - **Blocking.** Return `ToolMessage(content=..., tool_call_id=..., name=..., status="error")` without calling the handler.
  - **Model replies.** `AIMessage.usage_metadata` gives `input_tokens`/`output_tokens`; `response_metadata.get("model_name")` gives the model; `invalid_tool_calls` lists unparseable tool calls.
  - **Wrapped clients.** `ChatOpenAI(client=wrapped.chat.completions, async_client=wrapped_async.chat.completions)` accepts a wrapped client.
- **Produces:**

```python
class PitStopMiddleware(AgentMiddleware):
    def __init__(self, *, side_effects: Iterable[str] = (), agent: str | None = None): ...
```

  - **`before_agent`** sets `runs.adapter = 'langchain'`.
  - **`before_model`** calls `check_halt()`, so a pending halt raises `pitstop.Halted` out of `invoke`.
  - **`wrap_tool_call` / `awrap_tool_call`:**
    - with no current run, it calls `handler(request)`;
    - otherwise it calls `guard_call` / `aguard_call` with `agent=self.agent`;
    - `run` → the handler's result, plus `record_tool_result`;
    - `block` / `steer` → a `ToolMessage` with the reply text and `status="error"`.
  - **`wrap_model_call` / `awrap_model_call`** compare `ctx.llm_calls` before and after `handler`. When `wrap` did not meter the call, they record an observed `llm_call` event:
    - payload: `observed: true`, tokens from `usage_metadata`, cost from the price table, `tool_calls` (tool calls plus invalid ones), and `refused`;
    - L0 `check_llm_call` runs with `len(invalid_tool_calls)` as parse failures (decision 12).

- [ ] **Step 1: Write the failing tests** (`pytest.importorskip("langchain")`, a `FakeMessagesListChatModel` subclass with `bind_tools` returning `self`, and `AIMessage`s with `tool_calls`, `usage_metadata` and `response_metadata={"model_name": "fake-1"}`).
  - `test_off_scope_tool_blocked_with_tool_message`.
  - `test_loop_steers_then_halts`.
  - `test_side_effect_tool_gated`.
  - `test_unwrapped_model_observed_not_metered`: an observed `llm_call` event and no ledger row.
  - `test_wrapped_chatopenai_metered_once`: `ChatOpenAI(client=wrap(OpenAI(http_client=mock)).chat.completions, api_key="x")` gives one ledger row and one `llm_call` event without `observed`.
  - `test_pending_halt_raises_from_invoke`.
  - `test_async_agent_ainvoke`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.**
  - README section: `create_agent(model, tools, middleware=[PitStopMiddleware(side_effects={"send_email"})])` inside `pitstop.workflow`.
  - For hard budgets, a `ChatOpenAI` on a wrapped client.
  - Plain `StateGraph` graphs use `@pitstop.tool` and `wrap` directly.
  - CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(adapters): LangChain/LangGraph middleware guards tools and observes model calls"`.

---

### Task 11: L2 change-point statistics and per-run metrics

**Files:**
- Create: `src/pitstop/detect/metrics.py`, `src/pitstop/detect/l2.py`, `tests/detect/run_factory.py`
- Test: new `tests/detect/test_metrics.py`, `tests/detect/test_l2.py`

**Interfaces:**
- **`detect/metrics.py`:**

```python
@dataclass(frozen=True)
class MetricSpec:
    drift_type: str
    directions: tuple[Literal["up", "down"], ...]
    floor: float               # minimum std used for z-scores
    relative: bool             # floor is a fraction of |baseline mean| (else absolute)
METRICS: dict[str, MetricSpec]
def run_metrics(store, run_id: str) -> dict[str, float]: ...
```

  | metric | drift_type | directions | floor |
  |---|---|---|---|
  | `tool_error_rate` | `tool_call_breakage` | up | 0.05 absolute |
  | `refusal_rate` | `refusal_spike` | up | 0.05 absolute |
  | `tokens_per_call` | `cost_or_latency_creep` | up | 5% relative |
  | `llm_calls` | `cost_or_latency_creep` | up | 5% relative |
  | `cost_usd` | `cost_or_latency_creep` | up | 5% relative |
  | `duration_s` | `cost_or_latency_creep` | up | 5% relative |
  | `output_chars` | `quality_drop` | up, down | 5% relative |

  - **Where the values come from.** `run_metrics` reads only that run's `events` plus `runs.started_at` / `ended_at`, over agent-meter `llm_call` events (`payload.meter == "agent"`, which includes OTel and observed calls).
  - **Missing data.** A metric is left out when the run has no data for it: no tool results → no `tool_error_rate`; no LLM calls → no per-call metrics; not ended → no `duration_s`.
- **`detect/l2.py`:**

```python
WARMUP = 20
PARAMS: dict[Sensitivity, tuple[float, float]] = {"low": (0.5, 8.0), "normal": (0.5, 5.0), "high": (0.5, 4.0)}  # (k, h)
@dataclass(frozen=True)
class Alarm:
    direction: Literal["up", "down"]
    at: float                  # ts of the first run of the shift: when the alarming sum last left 0
    baseline_mean: float
    baseline_std: float
    value: float
@dataclass
class Cusum:
    n: int = 0; mean: float = 0.0; m2: float = 0.0
    s_hi: float = 0.0; s_lo: float = 0.0; hi_since: float | None = None; lo_since: float | None = None
    def update(self, x: float, ts: float, spec: MetricSpec, sensitivity: Sensitivity) -> Alarm | None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, text: str) -> Cusum: ...
```

  - **Warmup.** For the first `WARMUP` values, update Welford only. After that the baseline is frozen.
  - **The update.** `std = max(sqrt(m2/(n-1)), floor)`, where the floor is `spec.floor * abs(mean)` when relative (and at least `1e-9`). With `z = (x - mean)/std`:
    - `s_hi = max(0, s_hi + z - k)`, with `hi_since` = the ts when `s_hi` went from 0 to positive;
    - `s_lo` the same with `-z`.
  - **The alarm.** An alarm fires in an allowed direction when its sum is greater than `h`. Then the detector resets to `Cusum()`.
- **`tests/detect/run_factory.py`:** `make_run(store, workflow, *, started_at, ended_at, model="m1", base_url="https://a/v1", prompt_versions=None, llm_calls=((100, 20, 400),), tool_results=(), status="ok", criticality="medium", sensitivity="normal", adapter="wrap") -> str`, where `llm_calls` are `(prompt_tokens, completion_tokens, content_chars)` and `tool_results` are booleans (True = error). It writes the run row and its events in the Task 1 payload shape, and is reused by Tasks 12 and 13.

- [ ] **Step 1: Write the failing tests.**
  - `test_run_metrics_from_events`: values match the factory inputs exactly.
  - `test_missing_metrics_omitted`.
  - `test_no_alarm_on_stationary_noise`: `random.Random(7)`, 300 draws `gauss(10, 1)`, normal → None every time.
  - `test_step_up_alarms_quickly`: 20 at `gauss(10, 1)`, then `gauss(13, 1)` → an alarm within 8 updates, direction up, `at` within 2 runs of the first shifted ts.
  - `test_down_ignored_when_only_up`.
  - `test_floor_prevents_zero_std_alarm` (Review Focus 3): 20 × 0.0 `tool_error_rate`, then 0.02 → no alarm; then three runs of 0.5 → an alarm.
  - `test_state_roundtrip`.
  - `test_reset_after_alarm`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(detect): L2 per-run metrics and two-sided CUSUM change-point detector"`.

---

### Task 12: L5 attribution and `pitstop explain`

**Files:**
- Create: `src/pitstop/detect/attribution.py`
- Modify: `src/pitstop/detect/urgency.py`, `src/pitstop/cli.py` (with `cli_report.py` if the table helper lives there), README (new "What changed?" section), CHANGELOG
- Test: new `tests/detect/test_attribution.py`, `tests/test_cli_explain.py`

**Interfaces:**
- **`detect/attribution.py`:**

```python
TAGS = ("model", "base_url", "prompt_versions", "pitstop_version", "adapter")
WINDOW = 50
NEW_SHARE_AFTER = 0.5
MAX_SHARE_BEFORE = 0.1
@dataclass(frozen=True)
class TagChange:
    tag: str; before: str | None; after: str; share_before: float; share_after: float
@dataclass(frozen=True)
class Attribution:
    workflow: str; at: float; changes: tuple[TagChange, ...]; n_before: int; n_after: int
    metric: str | None; before_mean: float | None; after_mean: float | None; summary: str
def attribute(store, workflow: str, at: float, *, metric: str | None = None, window: int = WINDOW) -> Attribution: ...
def explains_change(a: Attribution) -> bool: ...   # a change in model, base_url or prompt_versions
```

  - **The runs compared.** Before: up to `window` runs of the workflow started before `at`, newest first. After: up to `window` runs started at or after `at`, oldest first.
  - **A tag change.** A tag value with share ≥ `NEW_SHARE_AFTER` after and ≤ `MAX_SHARE_BEFORE` before is a change; `before` is the most common value before.
  - **Means.** `before_mean` / `after_mean` are the means of `run_metrics(...)[metric]` when a metric is given.
  - **Summary strings**, with times in UTC ISO 8601 to the minute:
    - with changes: `"<tag> changed from <before> to <after> (<pct>% of runs since <iso>)"`, several joined with `"; "`;
    - no changes: `"no run tag changed around <iso>; likely an input or upstream shift"`;
    - no earlier runs: `"no runs before <iso> to compare"`.
- **`urgency.escalate_for_change(level: Level, criticality: Criticality) -> Level`:** the more urgent of `level` and P1 (high/medium) or P2 (low). It never lowers the level.
- **CLI `pitstop explain <drift-id>`:**
  - `at` is the newest evidence entry's `detail.change_at` when present (L2 drifts), else the drift's `ts`; `metric` is `detail.metric` when present;
  - it prints the summary, a table of tag changes, and the before/after mean with counts;
  - it exits 1 with "no drift <id>" for an unknown id.

- [ ] **Step 1: Write the failing tests** (`run_factory`).
  - `test_model_change_attributed`: 30 runs on m1, then 20 on m2 → one change, model m1 → m2, share_after 1.0, and the summary text matches exactly.
  - `test_no_change_summary`.
  - `test_minority_value_not_a_change`: 20% of runs after on m2 → no change.
  - `test_prompt_version_change_explains`.
  - `test_metric_means`.
  - `test_escalate_for_change` (parametrized over levels and criticalities).
  - `test_explain_cli_prints_summary` and `test_explain_unknown_drift_exits_1` (subprocess, `PITSTOP_HOME=tmp`).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.** README section and CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(detect): L5 attribution compares runs before and after a change; pitstop explain"`.

---

### Task 13: The background `scan` pass

**Files:**
- Create: `src/pitstop/detect/scan.py`
- Modify: `src/pitstop/cli.py`, `src/pitstop/alerts/model.py` (the cause line), README (new "Background checks: `pitstop scan`" section and the roadmap line), CHANGELOG
- Test: new `tests/detect/test_scan.py`, `tests/alerts/test_dispatch.py` (cause from attribution)

**Interfaces:**
- **Consumes:** `run_metrics`, `METRICS`, `Cusum` (Task 11); `attribute`, `explains_change`, `escalate_for_change` (Task 12); `classify`; `record_drift`; `alerts.dispatch.notify`.
- **Produces:**

```python
@dataclass(frozen=True)
class ScanAlarm:
    workflow: str; metric: str; drift_type: str; level: str; drift_id: int | None; summary: str
@dataclass(frozen=True)
class ScanReport:
    runs: int; alarms: tuple[ScanAlarm, ...]
def scan(store, *, limit: int = 500) -> ScanReport: ...
```

  - **Which runs.** Finished runs (`ended_at IS NOT NULL`) with `scanned_at IS NULL`, ordered by `ended_at, rowid`, up to `limit`.
  - **One transaction per run** (`BEGIN IMMEDIATE`):
    1. claim the run: `UPDATE runs SET scanned_at=? WHERE id=? AND scanned_at IS NULL`, and skip it if no row changed;
    2. compute `run_metrics`;
    3. for each metric present, load `l2_state(workflow, metric)`, `update` it with the run's criticality and sensitivity, and save it.
  - **Alarms are handled after the run's transaction commits:**
    - `attribute(store, workflow, alarm.at, metric=metric)`;
    - the finding is `Finding(spec.drift_type, "degraded", False, {"layer": "L2", "metric", "direction", "value", "baseline_mean", "baseline_std", "change_at": alarm.at, "attribution": {"summary", "changes": [asdict(...)]}})`;
    - the level is `classify(...)`, escalated by `escalate_for_change` when `explains_change`;
    - `record_drift(..., action="record")`; `notify(drift_id)` for P0/P1.
  - **Runs with no metrics** are still marked scanned.
- **Alert cause.** `alerts/model.py` uses the newest evidence entry's `detail.attribution.summary` when present, else today's `recent_change` text.
- **CLI.** `pitstop scan [--every SECONDS]` prints `scanned <n> runs, <m> alarms` and one line per alarm (`<level> <workflow> <drift_type> (<metric>): <summary>`). With `--every` it loops until Ctrl-C and exits 0. It exits 2 on an invalid config, like the other commands.

- [ ] **Step 1: Write the failing tests** (`run_factory`).
  - `test_stationary_workflow_no_alarms`: 60 runs → `ScanReport(runs=60, alarms=())`.
  - `test_token_jump_after_model_change_is_p1`: 30 runs on m1 at ~120 tokens per call (seeded noise), then 15 on m2 at ~300. Expected:
    - exactly one open `cost_or_latency_creep` drift;
    - level P1 on medium (escalated from P2);
    - evidence attribution summary starts with `"model changed from m1 to m2"`.
  - `test_low_criticality_escalates_only_to_p2`.
  - `test_rescan_is_noop`.
  - `test_unfinished_runs_wait`.
  - `test_concurrent_scans_process_each_run_once` (Review Focus 4): two threads each call `scan` with their own `Store` on the same file. The sum of `runs` equals the number of runs, and `l2_state` `n` equals the run count for an untriggered metric.
  - `test_scan_state_survives_restart`: scan 25 runs, then a new `Store` scans 10 more; no re-warmup (n = 35 minus resets).
  - `test_alert_cause_uses_attribution`.
  - `test_scan_cli_once` (subprocess).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the docs.** README section: what L2 watches (the metrics table), when to run `pitstop scan` (cron or `--every`; the M3 server will run it), and how it reads in `pitstop drifts` / `pitstop explain`. CHANGELOG.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(detect): pitstop scan runs L2 over finished runs and records attributed drifts"`.

---

### Task 14: Milestone wrap-up

**Files:**
- Modify:
  - `README.md` (roadmap: milestone 2 done, 3 next; repo map; table of contents; "How it works today" now covers OTel, adapters and scan);
  - `CHANGELOG.md`;
  - `docs/config.example.toml` (only if earlier tasks missed something);
  - this plan (append "Carried to milestone 3" at the end of execution).
- Test: the full suite

**Interfaces:** none new.

- [ ] **Step 1: Check the offline demos.**
  - Run `uv run python examples/quickstart/agent.py --offline --inject-loop` with `PITSTOP_HOME` set to a temp dir. Expected: two steers then a halt, as in M2b.
  - In a fresh temp home, run `uv run finresearch run --offline`. Expected: a block for `agent_nvda_dc`.
  - Then run `uv run pitstop drifts --all`, `uv run pitstop explain <id>` and `uv run pitstop scan`. Expected: each prints sensibly.
- [ ] **Step 2: Run the checks.** Run `uv run pytest -q && uv run ruff check && uv build --wheel --out-dir <tmp>`. Expected: all PASS, and the wheel has no `finresearch`.
- [ ] **Step 3: Check the README.** Every new section is linked from the table of contents, and nothing unbuilt is described outside the roadmap.
- [ ] **Step 4: Commit.** Run `git commit -m "docs: milestone 2 complete; roadmap points to milestone 3"`.

---

## M2b carry list: where each item went

| M2b carry item | Here |
|---|---|
| Refuse new runs once a budget is exhausted (§3.2) | Task 4 |
| Spend-rate P0 trigger (§5) | Task 4 (inline, decision 6) |
| P0 with failed drift write pages nobody | Task 5 |
| Alert cause after a fallback clear | Tasks 1 + 5 (`runs.fallback_from`) |
| Superseded stale `pending` rows | Task 5 |
| SMTP permanent errors retried | Task 5 |
| `l1_gate_v2` delimiters | Task 5 |
| `[judge] base_url = ""` accepted | Task 5 |
| README privacy sentence | Task 5 |
| Fallback on a `base_url` change | Task 5 (decision 8) |
| Drift keeps action `fallback` after a clear | Task 5 (`fallback (cleared)` in `pitstop drifts`) |
| `fallback_from` reads `"None"` | Task 5 |
| Responses estimate with `previous_response_id` / token-id prompts | Task 6 |
| Non-function Responses tool calls, refusal parts | Task 6 |
| Anthropic premium billing modes | Dropped: documented in M2b; users add a price override |
| `tool_runner` relies on `runner._client` | Dropped: pinned by a test, with a warning fallback |
| Snapshot doc | Task 6 |
| Streaming-response worst case when never entered | Task 6 |
| Async enter-error tests | Task 6 |
| `Mode` Literal | Task 6 |
| Async generator in a reference cycle | Task 1 |
| Steer-limit halt without `halt_drift_id` | Task 1 |
| `meter()` reset across Contexts | Task 1 |
| GEPA `stop_callbacks`, `BaseException`, `max_workers > 1` | Milestone 4 (decision 11) |
| `runs(workflow, started_at)` index | Task 1 |
| Close background pool connections | Task 1 |
| Fork test `waitpid` timeout; unexited manager in a test | Task 1 |
| CHANGELOG L1 sentence | Task 5 |
| `budget reconcile` docstring | No action needed (fixed in M2b) |

---

## Carried to milestone 3 (recorded at the end of execution)

These came out of the task reviews and the final whole-branch review. The M3 plan must cover or explicitly drop them. GEPA items stay with milestone 4 (decision 11).

- **Open from the final review (residual):** the OTel filter added to `recent_change` also hides other OTel runs when building the alert cause for an OTel run's own drift, so an OTel-only workflow's alert can say "No model or provider change detected" when the model did change. Compare like with like (exclude OTel history only when the current run is not OTel) and add an alert-cause test.
- **Run sources sharing a workflow name:** an OTel run's response model (dated snapshot) vs an SDK run's requested model can read as a model change in alert causes; L2 streams and L5 windows mix OTel and SDK runs of one workflow by design (decision 1) — revisit when the dashboard shows them side by side.
- **Server-owned scan:** `notify(drift_id)` resolves against the SDK runtime's store, not the `store` passed to `scan` (add `store=`); an unrecorded P0 page waits the 5 s busy timeout per channel when the store is locked (send first, record after).
- **OTel processor:** agent identity is read from start-time attributes only; a GenAI child ending concurrently with its root can lazy-insert a run left `running`; no multi-threaded span-end test; an opt-in `pitstop.side_effects` span attribute; unrecorded-P0 logging outside the once-per-key gate.
- **Adapters:** Agents SDK — `on_agent_end` clears rather than restores the framework agent name (nested `as_tool` runs, concurrent runs in one scope share a name); `_CheckedInvoker` should forward `__signature__`; `_tag_run` writes at every agent start. LangChain — observed events are named by the response's model name; one observed event per model-node turn; a handler mixing wrapped and unwrapped calls hides the unwrapped one; `langgraph` is imported directly but only arrives through `langchain`; the concurrent observed test relies on a 0.3 s sleep. Both — async hooks write SQLite on the event loop.
- **Guard:** a nested same-name tool skips the gate for inner calls with different args, and `create_task` inside a marked call inherits the mark; `emit_verdict` retries the failing import when OTel is missing.
- **L2/L5/scan:** a failed attribution prints an alarm line with an empty summary; the L5 summary with no runs after the change point reads like "no tag changed"; `run_metrics` is called per run in attribution; negative `duration_s` on clock skew; `pitstop budget reconcile --older-than-min inf` is accepted.
- **Budgets:** the refused-run alert cause reads "model unknown at base_url unknown" (name the budget instead); refused rows count as runs in `pitstop status` (documented).
- **Alerts and gate:** no test for the store-refused branch of `notify_unrecorded`; `SMTPRecipientsRefused` with all-4xx codes is treated as permanent and `SSLCertVerificationError` is retried; near-miss closers (`< /tool_args>`, full-width) are not escaped by gate v2.
- **wrap:** a mixed `[int, str]` completions prompt counts bare ints as 0.
- **SDK runtime and tests:** the steer-limit halt can attach the steer's drift id to a halt whose reason came from elsewhere; `_unhooked` docstring should say a skipped-cleanup inner async generator is never finalized; `open_connections()` counts connections closed directly; a dispatch test half never asserts its failing job ran; a context test replaces the loop exception handler without restoring it; an explain test for exit 2 and workflow-less drifts; a static grep for `finresearch` in `src/pitstop`; a few lines over 120 characters.
- **Example (finresearch):** the worker sees PitStop's raw block text; `--offline` silently ignores `--no-report`/`--mode`/`--cache-dir`/`--max-research-calls`; `--prompt` has no alias; under `watch=False` a halted benchmark run no longer fails loudly; the research tool decorator is applied per call; `--max-corrections` help; the example's pyproject omits `python-dotenv`.
- **Docs:** the README TOC omits the "Where it's going" subsections; one long CHANGELOG swarm bullet.
