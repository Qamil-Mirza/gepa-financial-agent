# PitStop Milestone 3a: Local Server and Dashboard on Real Data

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pitstop ui` opens a local dashboard on the data the SDK already writes.
- The dashboard has the pit wall, a workflow page (Run graph, What changed, Runs), run detail, read-only settings and an empty state that teaches the setup.
- A background worker in the same process owns the `scan` pass and the daily digest.
- Every item in the "Carried to milestone 3" list of the M2c plan is closed or explicitly moved on.

**Architecture:**
- **One local process.** `pitstop ui` starts a FastAPI app with uvicorn on `127.0.0.1` and a background worker thread. The app is a JSON API over the same SQLite store the SDK writes. The SDK never depends on it (spec §4).
- **Read models in Python, rendering in React.** All aggregation lives in `pitstop.server.views`, `pitstop.server.graph` and `pitstop.server.changes`: the pit-wall flags, the run graph, the time series and change markers. These are pure functions of the store, tested with pytest. The React app only fetches and draws.
- **One typed contract.** Pydantic response models produce the OpenAPI schema. It is committed as `ui/openapi.json`, and the TypeScript types are generated from it. A Python test fails when the committed schema is stale.
- **Bundled UI.** `ui/` (React + TypeScript + Vite) builds into `src/pitstop/server/static/`. That folder is git-ignored and included in the wheel as a hatch artifact, so users never need Node (spec §9).

**Tech Stack:**
- Python ≥3.11, uv, pytest.
- New optional extra `pitstop[ui]`: `fastapi>=0.115`, `uvicorn>=0.32`.
- Node 22 LTS and npm, for the `ui/` build only: React 19, TypeScript 5 (strict), Vite 7, React Router 7, `@xyflow/react` 12 and `elkjs` for the run graph, Vitest 3 with Testing Library, `openapi-typescript` 7.
- Fonts are self-hosted through `@fontsource`.

**Spec:** `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md`. This plan covers:
- §4: the server process and its background worker (L2/L5 scan and P1–P3 routing; L3/L4 are 3b);
- §9: pit wall, workflow page (Run graph, What changed, Runs, right column), settings (read-only), empty state; the Tech line;
- §9.2: the Quickstart update for the dashboard;
- the "Carried to milestone 3" list at the end of `docs/superpowers/plans/2026-10-22-pitstop-m2c-observe-adapters-background.md`.

**Milestone 3 is split like milestone 2:**
- **3a (this plan):** the server and the dashboard.
- **3b:** L3 run judge, L4 endpoint canaries, watch budget (§6.4) and the Endpoints page.
- **3c:** the landing site (`site/`, §9.1), README screenshots and GIF, and retiring the two old example UIs (§12).

The Prompt versions tab and the right column's Review diff / Canary actions belong to milestone 4 (correction) and 3b (canaries). In 3a the right column shows open drifts with Explain and Dismiss.

Design context: `.impeccable.md` and the mockups in `docs/superpowers/specs/2026-10-20-pitstop-mockups/`: `health-board-directions.html` (direction B, "Pit wall") and `workflow-detail.html`. Every UI task's implementer reads both, plus the impeccable `frontend-design` skill.

## Decisions this plan makes (design calls; the owner can revisit)

1. **The server is an optional extra.** `pip install "pitstop[ui]"` adds FastAPI and uvicorn. The SDK stays lean, so it can be embedded in services that pin their own FastAPI.
   - `pitstop ui` without the extra exits 2 with: `The dashboard needs the ui extra: pip install "pitstop[ui]"`.
   - The dev group includes the extra.
2. **Local only, no login.**
   - It binds `127.0.0.1` by default. `--host` with a non-loopback address prints a warning that the dashboard shows prompts and tool arguments and has no authentication.
   - A Host-header check (DNS-rebinding guard) only accepts `localhost`, `127.0.0.1`, `[::1]` and the `--host` value. Anything else gets 400.
   - Every write endpoint requires the header `X-PitStop: 1` (403 without it). No CORS headers are ever sent, so a cross-site page cannot send that header.
3. **The only write in 3a is Dismiss.** `POST /api/drifts/{id}/dismiss` sets status `dismissed` on an open drift. A later occurrence of the same (workflow, drift_type) opens a new drift (existing dedup is over `status='open'`). Settings stay read-only (spec §9).
4. **The worker runs `scan` every 60 s and the digest check every hour,** in one daemon thread. `pitstop ui --no-worker` turns it off, and `--scan-every SECONDS` changes the interval.
   - Each pass is wrapped: an exception is logged once per type and the loop continues. `/api/health` reports the last pass time and error type.
   - `pitstop scan` from cron keeps working next to it (scan claims each run once).
5. **Workflow names travel as query parameters,** never as path segments. Names may contain `/`, `?`, `#`, spaces or emoji. Examples: API `/api/workflow?name=…`, UI route `/w?name=…&tab=graph`.
6. **Pit-wall flag.**
   - `red`: an open P0 drift, or the latest run's status is `halted` or `refused`.
   - `yellow`: any other open drift.
   - `green`: no open drifts.

   Red and yellow tiles are large and green tiles are small (direction B). Tiles are sorted by flag, then by most urgent open level, then by latest activity. Every flag also carries a text label ("Halted", "Degraded", "Healthy"), so urgency is never shown by color alone.
7. **Spend on a tile** is today's agent-meter spend against the workflow's daily budget (`runs.budget_usd` of the latest run; ledger scope `workflow:<name>`, period `day`). When no workflow budget is set, the tile shows spend only. Global budgets appear in the pit-wall header.
8. **Run graph structure** (SDK runs have no spans):
   - **Nodes.**
     - `workflow` is the root.
     - `agent:<path>` is each agent. `path` is the agent's chain of names joined with ` › `, so two agents with one name under different parents stay apart.
     - Leaves: `llm:<agent path>|<model>` and `tool:<agent path>|<tool name>`.
   - **SDK runs.** Agent parentage comes from new `agent` events written when an `agent()` scope is entered (payload `{"parent": <enclosing agent name or null>}`) or when a framework hook names an agent.
   - **OTel runs.** Agent parentage comes from `parent_span_id`, walking up to the nearest `agent` event.
   - **Node statistics.** Over the runs in scope: calls, runs touched, cost, p50 latency, time per run and verdict counts. Verdicts are the run's `block`, `steer`, `halt`, `fallback`, `gate` (with `allow` false), `gate_error` and errored `tool_result` events on that node.
   - **The slowest path** starts at the root and repeatedly takes the child with the largest `time_per_run_s`. An agent's time is the sum of its children's.
   - **Scope.** Either one run, or the newest runs started in a window (`1h`, `24h` default, `7d`, `30d`), capped at 200 runs.
9. **Latency is recorded at the source.**
   - `llm_call` events gain `latency_s`: wall time from the call's start to its record; for streams, to the end of the stream.
   - `tool_result` events gain `latency_s` from the guard's check-once mark (`Mark.started`).
   - OTel `llm_call` and `tool_result` events gain `latency_s` from span start and end.
   - Older events without it count as unknown, never as 0.
10. **What changed.**
    - The series are the L2 per-run metrics (`run_metrics`) of the newest 200 runs that have metrics.
    - The markers are:
      - tag changes between consecutive runs **of the same source** (OTel runs are compared only with OTel runs, SDK runs only with SDK runs) for `model`, `base_url` and `prompt_versions`;
      - every L2 drift's `change_at`.
    - A quality series (judge pass rate) arrives with L3 in 3b.
11. **Live updates by polling.** The pit wall polls `/api/board` every 10 s and `/api/feed` every 5 s, and the workflow page polls every 10 s. Polling pauses while the tab is hidden. No websockets in 3a.
12. **Bundling.**
    - `npm --prefix ui run build` writes `src/pitstop/server/static/`, which is git-ignored and listed under `[tool.hatch.build.targets.wheel] artifacts`.
    - When the folder has no `index.html`, the server serves a one-page "Dashboard not built" message with the two build commands, and the API still works.
    - CI gains a `ui` job: typecheck, tests and build. It then builds the wheel with `uv build` and checks that it contains `pitstop/server/static/index.html`.
13. **Fonts and assets are local.** Instrument Sans and Newsreader (the mockups' fonts) are installed with `@fontsource` packages and bundled. The dashboard makes no request to any other host, because it runs offline and on private data.
14. **Carry-list items explicitly not done in 3a** (each with its reason; Task 16 copies them to the 3b carry list):
    - the opt-in `pitstop.side_effects` span attribute: a feature, deferred to "Later";
    - async adapter hooks writing SQLite on the event loop: WAL writes are sub-millisecond, so this is revisited only on a report;
    - L2 streams and L5 windows mixing OTel and SDK runs of one workflow: by design (M2c decision 1). The dashboard now separates sources in its change markers.
    - LangChain's "one observed event per model-node turn" and "a handler mixing wrapped and unwrapped calls hides the unwrapped one": documented as known limitations in the README.
    - `create_task` inside a marked call inheriting the guard mark: documented in `guard.py`, revisited with M4.
    - attribution calling `run_metrics` per run: at most 100 small queries.
    - refused runs counting in `pitstop status`: documented. The dashboard labels them "refused".

## Global Constraints

- **Python ≥3.11.**
  - No new *required* runtime dependency. The only new extra is `pitstop[ui]` = `fastapi>=0.115`, `uvicorn>=0.32`. The dev group adds both.
  - `import pitstop` and every CLI command except `ui`/`serve` must work without the extra (pinned by a subprocess test in Task 6).
- **Node 22 for `ui/` only.**
  - Exact versions are locked in `ui/package-lock.json`, which is committed. Use `npm ci` in CI.
  - Every dependency is a direct npm package. The built app makes no request to any non-local host (no CDN, no Google Fonts, no analytics).
- **No network in tests.**
  - pytest-socket stays on. FastAPI tests use `fastapi.testclient.TestClient`, which runs in-process.
  - Vitest tests use fixtures and never `fetch` a real server.
  - Tests never start uvicorn on a real port, except one subprocess smoke test bound to `127.0.0.1:0`. That test is allowed by the existing `--allow-hosts`.
- **Never a real key, never real spend.** Subprocess tests run with `cwd=tmp_path`, with every key env var scrubbed (the root `conftest.py` list) and `PITSTOP_HOME=tmp_path`.
- **Never delete untracked or git-ignored files.**
  - Never run `examples/quickstart/agent.py` or `finresearch run` without `--offline`.
  - Never use `git clean`, `git add -A` or `git add .`; add explicit paths.
  - The repo-root `pitstop.db*`, `runs/` and `dist-data/` belong to the owner.
- **Errors the SDK may raise.** It raises only `pitstop.Halted` / `pitstop.BudgetExceeded` on purpose: PitStop's own failures fail open, and an invalid config or a too-new store fails closed.
  - The server is not the SDK. It fails fast at startup, exiting 2 with one line, on an invalid config or a too-new store.
  - After startup, an endpoint never returns a stack trace: malformed rows are skipped or shown as unknown, and a locked store returns 503 `{"error": "store busy, retry"}`.
- **Secrets stay out.** API responses, logs and the UI never contain a secret value. For an env var, they show the variable's *name* and whether it is set (`key_set: true`), never its value.
- **Commits.**
  - No attribution trailers (no `Co-Authored-By`, no "Generated with").
  - Use conventional prefixes (`feat(server):`, `feat(ui):`, `fix(alerts):`, `docs:`).
- **Style.**
  - Python: ruff `F` and `E501` clean at line length 120 (Task 2 turns on `E501`). Match the surrounding docstring density.
  - TypeScript: `strict: true`, with no `any` in `ui/src` except in generated files.
  - CSS: tokens on `:root`, as in the mockups (oklch, hue 250, P-level colors). Dark mode follows the system, with a manual toggle stored in `localStorage` (wrapped in try/catch).
- **Design.**
  - Follow `.impeccable.md`: quiet until it matters, urgency never by color alone, money first-class, calm motion with `prefers-reduced-motion` respected.
  - Avoid the anti-references: no neon-on-dark, no gradients, no glass cards, no hero-metric tiles, no decorative sparklines.
  - All layouts work at 1280 px and at 390 px wide. The run graph may scroll inside its own panel.
- **Docs ship with the feature.** Each user-facing task updates its README section and `CHANGELOG.md` (Unreleased) in the same task. The README describes only what works (spec §9.2).
- **Every task ends green.**
  - Always: `uv run pytest -q` and `uv run ruff check`.
  - For tasks touching `ui/`, also `npm --prefix ui run typecheck`, `npm --prefix ui test -- --run` and `npm --prefix ui run build`.

## Review Focus

1. **Malformed rows never break a page.** A run with a NULL workflow, NULL timestamps or an unknown status, an event whose `payload` is not JSON, or a drift whose `evidence` is not a list: every endpoint answers 200 and skips the bad value or shows it as unknown. Pinned in Task 7: `test_endpoints_survive_malformed_rows`.
2. **A big store stays bounded.** With 5,000 runs and 100,000 events, every list endpoint returns at most its documented limit:
   - board: every workflow, with drift lists capped at 50;
   - feed: 100;
   - runs: 200 per page;
   - graph: 200 runs in scope;
   - changes: 200 points.

   Each endpoint uses an index-backed query. Pinned in Task 9: `test_large_store_responses_are_capped` (asserts the caps and uses `EXPLAIN QUERY PLAN` to assert no full scan of `events`).
3. **Odd workflow names round-trip.** These all work in every endpoint and UI route: `a/b`, `x?y#z`, `spaced name`, `résumé 🏁`. Pinned in Task 7 (`test_workflow_names_round_trip`) and Task 11 (Vitest `encodes workflow names in links`).
4. **Cross-site and rebinding requests are refused.**
   - A request with `Host: evil.example` gets 400.
   - A `POST` without `X-PitStop: 1` gets 403.
   - A response never carries `Access-Control-Allow-Origin`.

   Pinned in Task 6: `test_host_check_and_write_header`.
5. **A locked store degrades cleanly.**
   - Reads still answer, because WAL readers are not blocked.
   - Dismiss while another connection holds `BEGIN IMMEDIATE` answers 503 within the busy timeout plus 1 s and changes nothing.
   - The worker logs one warning and tries again next pass.

   Pinned in Task 7 (`test_dismiss_on_locked_store_returns_503`) and Task 6 (`test_worker_survives_locked_store`).

---

### Task 1: Alerts and budgets carry-overs

**Files:**
- Modify: `src/pitstop/alerts/model.py`, `src/pitstop/alerts/dispatch.py`, `src/pitstop/alerts/channels.py`, `src/pitstop/detect/changes.py`, `src/pitstop/detect/scan.py`
- Test: `tests/alerts/test_model.py` (or the existing alert-model test file), `tests/alerts/test_dispatch.py`, `tests/alerts/test_channels.py` (or existing), `tests/detect/test_changes.py`, `tests/detect/test_scan.py`

**Interfaces:**
- Produces:
  - `recent_change(store, workflow, run_id, *, ignore_model=None, now=None) -> Change | None` compares like with like. When the current run's adapter is `otel`, the walk considers only earlier `otel` runs. Otherwise it considers only non-`otel` runs, as today.
  - `notify(drift_id: int, *, store=None) -> None` and `notify_unrecorded(..., store=None)`: with `store` given, the alert is built from and recorded in that store instead of the runtime's. `scan(store, ...)` passes its own `store`.
  - The alert cause for a `budget_exhausted` drift on a `refused` run names the budget: `Budget <scope> is used up (<spent> of <limit> USD).` The scope, spent and limit come from the drift's evidence, which already carries them; read the existing evidence keys in `sdk/context.py` `_record_refusal`.

- [ ] **Step 1: Write the failing tests.**
  - `test_otel_only_workflow_alert_names_model_change`: two OTel runs of workflow W, on `m1` and then `m2`. `build_alert` for a drift on the second run has a cause containing `changed from m1 to m2`.
  - `test_sdk_run_cause_ignores_otel_history`: an earlier OTel run on a different model does not count as a change for an SDK run.
  - `test_refused_run_cause_names_budget`.
  - `test_scan_alerts_use_scan_store`: `scan(store_b)` with the runtime on store A still sends P0/P1 alerts for drifts in B, and records the alert rows in B.
  - `test_unrecorded_p0_sends_before_recording`: with the alerts table locked (a held `BEGIN IMMEDIATE`), a P0 page goes out within 1 s. The fake channel's send timestamp minus the start is under 1 s, and the row is recorded once the lock is released or is logged as unrecorded.
  - `test_notify_unrecorded_store_refused_branch`: covers the branch the M2c review found untested.
  - `test_recipients_refused_all_4xx_is_retryable`: `SMTPRecipientsRefused` where every code is 4xx is retried; a mix with any 5xx is not.
  - `test_ssl_cert_verification_error_not_retried`.
  - Fix the dispatch test half that never asserts its failing job ran: assert the failing job's channel was called.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/alerts tests/detect -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - **Like-with-like filter** in `changes.py`. Update the docstring.
  - **`store=` threading** through `notify`, `notify_unrecorded`, `_deliver` and `_deliver_unrecorded`.
  - **Unrecorded P0:** send first, then record the `alerts` row, whose failure is logged once.
  - **SMTP classification:** `SMTPRecipientsRefused` is retryable only when every recipient code is 4xx. `ssl.SSLCertVerificationError` is permanent; check it before the generic `OSError` branch.
  - **Refused-run cause:** a `budget_exhausted` drift's cause comes from its evidence.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(alerts): like-with-like alert causes, scan's own store, budget-named causes, SMTP retry classes"` with explicit paths.

### Task 2: Detection, gate, guard, wrap and SDK runtime carry-overs

**Files:**
- Modify:
  - `src/pitstop/detect/attribution.py`, `src/pitstop/detect/metrics.py`, `src/pitstop/detect/scan.py`, `src/pitstop/cli.py`;
  - `src/pitstop/detect/l1.py` (the gate's data escaping);
  - `src/pitstop/sdk/wrap/shapes.py` (or wherever the legacy prompt is counted);
  - `src/pitstop/otel/emit.py`, `src/pitstop/sdk/guard.py`, `src/pitstop/sdk/actions.py`, `src/pitstop/sdk/context.py`, `src/pitstop/store/__init__.py`;
  - `pyproject.toml` (ruff `select = ["F", "E501"]`) and every file over 120 characters per line.
- Test:
  - `tests/detect/test_attribution.py`, `tests/detect/test_metrics.py`, `tests/detect/test_scan.py`, `tests/detect/test_l1.py`;
  - `tests/sdk/test_wrap.py`, `tests/otel/test_emit.py`, `tests/sdk/test_guard.py`, `tests/sdk/test_context.py`;
  - `tests/test_cli_explain.py`, `tests/test_cli_report.py`, `tests/test_core_packaging.py`.

**Interfaces:**
- Produces:
  - `Attribution.summary` with no runs after the change point reads `No runs after <time> to compare yet.`
  - A failed attribution in `scan` gives `ScanAlarm.summary == "attribution failed (<ExceptionType>)"`.
  - `run_metrics` omits `duration_s` when `ended_at < started_at` (clock skew), so it is never negative.
  - **Guard marks are one-shot.** `is_checked(ctx, tool)` is true only for the first nested same-name call inside a `checked()` block: the inner call consumes the mark. Later same-name calls in the block are guarded normally. `Mark` gains `started: float` (`time.monotonic()` at creation), which Task 5 uses.
  - **Gate v2 data escaping** also covers near-miss closers. In `<tool_args>`/`<recent_calls>` text, any `<` or `＜` (U+FF1C), followed by optional whitespace and then `/` or `／` (U+FF0F), is written `<\/`. The prompt file is unchanged.

- [ ] **Step 1: Write the failing tests.**
  - `test_summary_with_no_runs_after`.
  - `test_scan_alarm_with_failed_attribution` (patch `attribute` to raise).
  - `test_duration_omitted_on_clock_skew`.
  - `test_reconcile_rejects_inf` (`--older-than-min inf` exits 2).
  - `test_gate_escapes_near_miss_closers`, parametrized over `< /tool_args>`, `<\t/recent_calls>`, `＜/tool_args>` and `<／tool_args>`.
  - `test_mixed_int_str_completions_prompt_counts_both`: `[1, 2, "hello world"]` estimates more than 2 tokens, with each str counted like a string prompt.
  - `test_emit_verdict_caches_missing_otel`: with the import patched to fail, two calls attempt the import once.
  - `test_steer_limit_halt_keeps_foreign_reason_drift`: when a halt reason set elsewhere is already pending, the steer limit does not overwrite `halt_drift_id`.
  - `test_nested_same_name_second_call_is_guarded`: inside one checked block, the first inner `@pitstop.tool` call skips the guard and the second one is checked (a `tool_call` event is recorded for it).
  - `test_explain_exit_2_on_invalid_config` and `test_explain_workflowless_drift_exits_1`.
  - `test_core_has_no_finresearch_imports`: grep `src/pitstop/**/*.py` for `finresearch` and expect zero hits.
  - Fix `tests/sdk/test_context.py`: the test that replaces the loop exception handler restores it in a `finally`.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/detect tests/sdk tests/otel tests/test_cli_explain.py tests/test_core_packaging.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the produced behaviors above.
  - **`cli._minutes`** rejects non-finite values, like `_seconds`.
  - **Docstrings:**
    - `_unhooked`: an inner async generator whose cleanup was skipped is never finalized.
    - `Store.open_connections`: a connection closed directly, not through `close_local`, still counts until garbage collected.
  - **`guard.py` module docstring:** a task created inside a marked call inherits the mark (decision 14).
- [ ] **Step 4: Turn on `E501`** and wrap every line over 120 characters in `src/`, `tests/`, `benchmarks/` and `examples/`, without changing behavior.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 6: Commit.** Run `git commit -m "fix(detect): carry-overs from milestone 2c reviews; ruff line length"` with explicit paths.

### Task 3: OpenTelemetry processor and framework adapter carry-overs

**Files:**
- Modify: `src/pitstop/otel/processor.py`, `src/pitstop/sdk/adapters/openai_agents.py`, `src/pitstop/sdk/adapters/langchain.py`, `pyproject.toml` (the `langchain` extra adds `langgraph>=1.0`), `uv.lock`, `README.md` (known limitations)
- Test: `tests/otel/test_processor.py`, `tests/adapters/test_openai_agents.py`, `tests/adapters/test_langchain.py`

**Interfaces:**
- Produces:
  - **OTel agent identity:** an agent name set on a span after it started (in its end-time attributes) is used, through `on_end` reading `span.attributes` again.
  - **No lingering runs:** a GenAI child span that ends after its root has ended never leaves a run `running`. The late child's events are written, and the run keeps its final status. Never insert a fresh run row after the root ended.
  - **Unrecorded-P0 logging** goes through the processor's once-per-key warning gate.
  - **Agents SDK `Hooks`:**
    - `on_agent_end` restores the framework agent name that was current before the matching `on_agent_start`; keep a per-scope stack.
    - `_tag_run` runs once per run; remember tagged run ids in a bounded set.
    - `_CheckedInvoker` forwards `__signature__` of the wrapped invoker.
  - **LangChain middleware:**
    - An observed `llm_call` event is named by the model the request named: the chat model's `model_name` or `model` attribute when it is a string, else the response's model.
    - `langgraph` is a declared dependency of the extra.

- [ ] **Step 1: Write the failing tests.**
  - `test_agent_name_from_end_attributes`.
  - `test_child_ending_after_root_keeps_status`.
  - `test_concurrent_span_ends_from_threads`: 8 threads each end 50 GenAI spans of 4 traces. Every trace becomes exactly one run with 100 or more events and status `ok`, with no exception.
  - `test_unrecorded_p0_warning_once`.
  - `test_nested_agent_end_restores_outer_name`.
  - `test_tag_run_once_per_run` (count UPDATE statements through a store spy).
  - `test_checked_invoker_keeps_signature`.
  - `test_observed_event_named_by_requested_model`.
  - Replace the 0.3 s sleep in the concurrent observed test with a `threading.Barrier`.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/otel tests/adapters -q`. Expected: the new tests FAIL, except the barrier rewrite.
- [ ] **Step 3: Implement.** Then add the two LangChain known limitations (decision 14) to the README's LangChain section.
- [ ] **Step 4: Run the checks.** Run `uv sync && uv run pytest -q && uv run ruff check`. Expected: all PASS. Record the resolved `langgraph` version in the report.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(otel,adapters): carry-overs from milestone 2c reviews"` with explicit paths.

### Task 4: Example (finresearch) carry-overs

**Files:**
- Modify: `examples/financial-research/finresearch/cli.py`, `examples/financial-research/finresearch/orchestrator.py`, `examples/financial-research/finresearch/agents/worker.py`, `examples/financial-research/pyproject.toml`, `benchmarks/` (the scripts that run the swarm with `watch=False`), `uv.lock`, `examples/financial-research/README.md`
- Test: `tests/examples/financial_research/test_cli_run_config.py`, `tests/examples/financial_research/test_sdk_wiring.py`, `tests/examples/financial_research/test_halted_propagates.py`

**Interfaces:**
- Produces:
  - **Clear block text for the worker.** The worker's next plan sees `PitStop blocked this research call: <reason>`, not the raw `{"blocked_by": ...}` dict.
  - **`--offline` rejects ignored flags.** Combined with any of `--no-report`, `--mode replay`, `--cache-dir` or `--max-research-calls`, it exits 2 with `--offline does not support <flag>`.
  - **`--prompt PROMPT` is a deprecated alias.** It exits 2 with `--prompt was replaced by --legacy-supervisor PROMPT`.
  - **Halted benchmark runs fail loudly.** A benchmark whose swarm run ends `halted` exits nonzero and names the run.
  - **The gated tool is built once.** The research tool is decorated once per `Swarm`, in `__init__`, not per call.
  - `--max-corrections` help text says what it limits and its default.
  - `python-dotenv>=1.0` is in the example's dependencies.

- [ ] **Step 1: Write the failing tests:**
  - `test_worker_sees_block_reason_text`;
  - `test_offline_rejects_unsupported_flags` (parametrized);
  - `test_prompt_alias_points_to_legacy_supervisor`;
  - `test_benchmark_fails_on_halted_run`;
  - `test_research_tool_decorated_once` (patch `pitstop.tool` and count calls across two research calls).
- [ ] **Step 2: Run them.** Run `uv run pytest tests/examples -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the checks.** Run `uv sync && uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(example): finresearch carry-overs from milestone 2c reviews"` with explicit paths.

### Task 5: SDK records what the dashboard draws

**Files:**
- Modify: `src/pitstop/sdk/wrap/call.py`, `src/pitstop/sdk/guard.py`, `src/pitstop/sdk/context.py`, `src/pitstop/otel/processor.py`, `src/pitstop/detect/drifts.py`
- Test: `tests/sdk/test_wrap.py`, `tests/sdk/test_guard.py`, `tests/sdk/test_context.py`, `tests/otel/test_processor.py`, `tests/detect/test_drifts.py` (create if missing)

**Interfaces:**
- Consumes: `Mark.started` (Task 2).
- Produces:
  - **`llm_call` latency.** The payload gains `latency_s: float`, rounded to 4 places (decision 9): from the call object's creation (`time.monotonic()`) to `_record`.
  - **`tool_result` latency.** The payload gains `latency_s` when a check-once mark is open for the call: `time.monotonic() - mark.started`.
  - **OTel latency.** `llm_call` and `tool_result` events gain `latency_s = (end_time - start_time) / 1e9`.
  - **SDK `agent` events.** `_Agent.__enter__` inside a run writes `kind='agent', name=<agent name>, agent=<agent name>, payload={"parent": <enclosing frame's name or null>, "source": "sdk"}`. `RunContext.set_framework_agent(name)` writes the same event with `parent` = the base frame's name, once per (run, parent, name), using a set on the context. Writes fail open (`_safe`).
  - **`dismiss_drift(store, drift_id: int) -> str`** in `detect/drifts.py`. It returns `"dismissed"`, `"not_found"` or `"not_open"`. It is one `BEGIN IMMEDIATE` transaction: `UPDATE drifts SET status='dismissed' WHERE id=? AND status='open'`. A later occurrence opens a new drift (existing behavior).

- [ ] **Step 1: Write the failing tests:**
  - `test_llm_call_records_latency` (sync, async and stream; latency ≥ 0);
  - `test_tool_result_records_latency` (a tool that sleeps 0.05 s records ≥ 0.05);
  - `test_agent_scope_writes_agent_event_with_parent` (nested `agent("a")` → `agent("b")` gives parent `a`);
  - `test_framework_agent_event_once`;
  - `test_otel_events_have_latency`;
  - `test_dismiss_drift_states` (dismissed, then not_open, then not_found; the next same-type finding opens a new drift id).
- [ ] **Step 2: Run them.** Run `uv run pytest tests/sdk tests/otel tests/detect -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** No schema change.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(sdk): record call latency and agent parentage; dismiss_drift"` with explicit paths.

### Task 6: The server process, worker and `pitstop ui`

**Files:**
- Create:
  - `src/pitstop/server/__init__.py`;
  - `src/pitstop/server/app.py` (app factory, middleware, static serving);
  - `src/pitstop/server/worker.py`;
  - `src/pitstop/server/models.py` (Pydantic response models, filled in by Tasks 7–9);
  - `tests/server/__init__.py`, `tests/server/conftest.py`, `tests/server/test_app.py`, `tests/server/test_worker.py`, `tests/server/test_cli_ui.py`.
- Modify: `src/pitstop/cli.py`, `pyproject.toml` (extra `ui`, dev group, hatch `artifacts = ["src/pitstop/server/static/**"]`), `.gitignore` (`src/pitstop/server/static/`), `uv.lock`.

**Interfaces:**
- Consumes: `pitstop.sdk.runtime.init/get_runtime`, `pitstop.detect.scan.scan(store)`, `pitstop.alerts.dispatch.run_digest()`.
- Produces:
  - **`create_app(rt: Runtime, *, static_dir: Path | None = None, allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]"), worker: Worker | None = None) -> FastAPI`.**
    - The API lives under `/api`. Any other GET path serves `static_dir/index.html` (SPA fallback), and `/assets/*` files are served from the static dir.
    - Without an `index.html`, every non-API GET returns the "Dashboard not built" HTML page (status 200). It names `npm --prefix ui ci` and `npm --prefix ui run build`.
  - **Middleware.**
    - The Host check is described in decision 2. Compare the host without its port; a bracketed IPv6 address keeps its brackets.
    - A write guard: POST, PUT, PATCH or DELETE without `X-PitStop: 1` gets 403 `{"error": "missing X-PitStop header"}`.
    - `sqlite3.OperationalError` containing "locked" or "busy" becomes 503 `{"error": "store busy, retry"}`.
    - Any other exception becomes 500 `{"error": "<ExceptionType>"}`, logged with its traceback on the server.
  - **`GET /api/health`** returns `Health`: `{ok: true, version, schema_version, home, worker: {enabled, running, last_scan_at, last_scan_runs, last_error, scan_every_s}}`.
  - **`class Worker(store, *, scan_every_s: float = 60.0, digest_every_s: float = 3600.0)`** with `start()`, `stop(timeout=5.0)` and `status() -> dict`.
    - It is a daemon thread with an Event-based sleep, so `stop` wakes it up.
    - Each scan pass calls `scan(store)` and stores the report's run count. The digest check calls `run_digest()` (a no-op without channels).
    - Errors are logged once per exception type and the loop continues. It calls `store.close_local()` after each pass.
  - **CLI.**
    - `pitstop ui [--host 127.0.0.1] [--port 8787] [--no-open] [--no-worker] [--scan-every SECONDS]` starts uvicorn programmatically with `log_level="warning"`, prints `PitStop dashboard: http://<host>:<port>` and opens the browser (`webbrowser.open`) unless `--no-open`.
    - `pitstop serve` is the same as `ui --no-open`.
    - A port in use exits 2 with `port <n> is in use; pass --port`.
    - A missing extra exits 2 (decision 1).
    - An invalid config or a too-new store exits 2 with the existing `Invalid PitStop config:` line.
  - **`export_openapi() -> dict`** in `server/app.py`: `create_app(...)` on a temporary in-memory runtime, then `.openapi()`. Task 10 commits it as `ui/openapi.json`.

- [ ] **Step 1: Write the failing tests.**
  - `test_health_reports_worker_and_schema`.
  - `test_host_check_and_write_header` (Review Focus 4):
    - `Host: evil.example` gets 400 and `Host: localhost:8787` gets 200;
    - POST `/api/drifts/1/dismiss` without the header gets 403;
    - no response has `access-control-allow-origin`.
  - `test_spa_fallback_and_not_built_page`: with a tmp static dir holding `index.html` and `assets/app.js`, `/w?name=x` returns index and `/assets/app.js` returns the file. With an empty dir, `/` returns the not-built page containing `npm --prefix ui run build`.
  - `test_api_404_is_json_not_index`: `/api/nope` returns 404 JSON.
  - `test_worker_scans_and_stops`: a store with one finished run; within 2 s, `status()["last_scan_runs"] == 1`; `stop()` returns within 1 s.
  - `test_worker_survives_locked_store` (Review Focus 5): hold `BEGIN IMMEDIATE` on a second connection; the worker logs one warning, keeps running and scans after release. Use `scan_every_s=0.1`.
  - `test_ui_without_extra_exits_2`: a subprocess with `-c` that blocks `import fastapi` through a `sys.modules` stub, then calls `pitstop.cli.main(["ui"])`.
  - `test_ui_refuses_newer_store`.
  - `test_import_pitstop_without_fastapi`.
  - `test_serve_smoke_subprocess`: pick a free port by binding a socket to `127.0.0.1:0` and closing it, start `pitstop serve --port <p> --no-worker`, poll `/api/health` with httpx for up to 10 s, then terminate the process.
- [ ] **Step 2: Run them.** Run `uv sync && uv run pytest tests/server -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the app factory, middleware, worker and CLI. `main()` accepts an optional `argv` list (needed by the test).
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(server): local dashboard server with background scan worker; pitstop ui / serve"` with explicit paths. Do not add `src/pitstop/server/static/` files.

### Task 7: Read models and endpoints: board, feed, workflow, drifts, runs, settings

**Files:**
- Create: `src/pitstop/server/views.py`, `src/pitstop/server/routes.py`, `tests/server/factory.py` (seeds rows with SQL), `tests/server/test_views.py`, `tests/server/test_routes.py`
- Modify: `src/pitstop/server/models.py`, `src/pitstop/server/app.py`

**Interfaces:**
- Consumes:
  - `cli_report.workflow_rows`'s logic: reuse it or factor a shared helper.
  - `Ledger.spent`, `alerts.model.build_alert(store, drift_id)` for each drift's title, cause and action text, `detect.attribution.attribute`, `cli_report.explain_target`, `detect.drifts.dismiss_drift`, `sdk.fallback.active`.
- Produces: pure functions in `views.py`, each taking `(rt, ...)` and returning the Pydantic model. Each endpoint is a thin wrapper.

  | Endpoint | Model | Notes |
  |---|---|---|
  | `GET /api/board` | `Board{workflows: [Tile], budgets: [BudgetRow], now}` | `Tile{name, criticality, flag: "red"\|"yellow"\|"green", flag_label, open: {P0,P1,P2,P3}, headline: str\|None, headline_level, last_run: RunRef\|None, runs_today, spend_today_usd, budget_usd, adapters: [str], fallback_active: bool}`. Flag and sort follow decision 6. `headline` is the most urgent open drift's `build_alert` title. |
  | `GET /api/feed?limit=50` (max 100) | `[FeedItem{id, ts, kind, workflow, level, text, run_id, drift_id}]` | Newest first. Kinds:<br>• `drift`: drifts opened, by `ts`, P0–P2 only;<br>• `halt`: `halt` events;<br>• `fallback`: `fallback` events;<br>• `refused`: refused runs;<br>• `model_change`: a run whose (model, base_url) differs from the previous run of the same workflow and source.<br>Ids are `"<kind>:<row id>"`. |
  | `GET /api/workflow?name=` | `WorkflowDetail{tile: Tile, goal, sensitivity, fix_mode, drifts: [DriftCard]}` | `DriftCard{id, level, drift_type, action, occurrences, first_seen, last_seen, title, cause, doing}`, from `build_alert`. Open drifts only, in `drift_rows` order, capped at 50. 404 for an unknown workflow. |
  | `POST /api/drifts/{id}/dismiss` | `{status}` | 200 `dismissed`, 404 `not_found`, 409 `not_open`. |
  | `GET /api/drifts/{id}/explain` | `Explain{summary, changes: [TagChange], metric, before_mean, after_mean, n_before, n_after, at}` | 404 for an unknown id, 422 for a drift without a workflow. |
  | `GET /api/runs?workflow=&limit=50&before=` (max 200; `before` is a `started_at` cursor) | `RunPage{runs: [RunRow], next_before}` | `RunRow{id, status, started_at, ended_at, duration_s, model, base_url, adapter, cost_usd, llm_calls, tool_calls, verdicts}`. `cost_usd` is the sum of agent-meter `llm_call` cost. `verdicts` counts `block`/`steer`/`halt`/`fallback` events. |
  | `GET /api/runs/{id}` | `RunDetail{run: RunRow, workflow, goal, events: [EventRow], truncated: bool}` | `EventRow{id, ts, agent, kind, name, cost_usd, latency_s, summary}`. `summary` is a one-line text of at most 300 characters built from the payload: the tool args, block reason, model and token counts, or the error. Raw payloads are never returned. Capped at 2,000 events; `truncated` is set when capped. |
  | `GET /api/settings` | `Settings{home, workflows: [WorkflowSettings], budgets: [BudgetRow], judge: {model, base_url, api_key_env, key_set}, alerts: {channels: [{name, configured, env_var, env_set}], routes}, worker}` | `WorkflowSettings` comes from each workflow's latest run: criticality, sensitivity, fix_mode, budget_usd. |

- [ ] **Step 1: Write the failing tests.** Build fixtures with `tests/server/factory.py`, which inserts runs, events, drifts and ledger rows directly.
  - `test_board_flags_and_sort`: three workflows: one with a P0 open (red), one with a P2 open (yellow) and one with none (green). Expect the order red, yellow, green, the labels "Halted", "Degraded" and "Healthy", and the headline equal to the P0 drift's alert title.
  - `test_board_halted_last_run_is_red_without_drift`.
  - `test_tile_spend_against_workflow_budget` (ledger rows today vs yesterday).
  - `test_feed_kinds_and_order`; `test_feed_model_change_same_source_only`.
  - `test_workflow_detail_drift_cards_use_alert_text`.
  - `test_dismiss_endpoint_states` (200/409/404).
  - `test_dismiss_on_locked_store_returns_503` (Review Focus 5).
  - `test_explain_endpoint`.
  - `test_runs_pagination_cursor` (75 runs: 50 runs plus `next_before`, then 25 runs and `next_before` null).
  - `test_run_detail_summaries_never_include_raw_payload` (a payload with a 10 kB result gives a summary of at most 300 characters).
  - `test_settings_never_returns_env_values`: set `CRUSOE_API_KEY=sk-test-123` and a Slack env var with `monkeypatch`, then assert that neither value appears anywhere in the response text and that `key_set` is true.
  - `test_endpoints_survive_malformed_rows` (Review Focus 1).
  - `test_workflow_names_round_trip` (Review Focus 3), through every endpoint that takes a workflow.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/server -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `views.py` and `routes.py`, and register the routes in `create_app`.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(server): board, feed, workflow, drift, run and settings endpoints"` with explicit paths.

### Task 8: Run graph

**Files:**
- Create: `src/pitstop/server/graph.py`, `tests/server/test_graph.py`
- Modify: `src/pitstop/server/models.py`, `src/pitstop/server/routes.py`

**Interfaces:**
- Consumes: the `agent` events and `latency_s` fields (Task 5); `events.span_id` and `events.parent_span_id` (OTel).
- Produces:
  - **`build_graph(store, workflow: str, *, run_id: str | None = None, window: str = "24h", now: float | None = None, max_runs: int = 200) -> Graph`** (decision 8).
  - **The graph model.**
    - `Graph{nodes: [GraphNode], edges: [{source, target}], runs_in_scope, window, run_id, slowest_path: [node id]}`.
    - `GraphNode{id, kind: "workflow"|"agent"|"llm"|"tool", label, agent_path, calls, runs, cost_usd, latency_p50_s: float|None, time_per_run_s: float|None, verdicts: {block, steer, halt, fallback, gate_reject, gate_error, tool_error}, on_slowest_path: bool}`.
  - **Building it.**
    - Events with no agent hang directly under the workflow root.
    - An `llm_call` on a meter other than `agent` is left out: the judge's calls are PitStop's own, not the agent's.
    - `window` must be one of `1h`, `24h`, `7d` and `30d`; anything else is a 422 at the endpoint.
  - **Endpoint:** `GET /api/graph?workflow=&run_id=&window=24h`. 404 for an unknown workflow or a run of another workflow.

- [ ] **Step 1: Write the failing tests:**
  - `test_sdk_run_graph_shape`: workflow root, then agent `planner`, then nested agent `researcher` (from `agent` events), with an llm leaf and a tool leaf under each. Check the edge set exactly.
  - `test_same_agent_name_under_two_parents_stays_apart`.
  - `test_otel_run_graph_uses_span_links`.
  - `test_node_stats`: calls, runs, cost, p50 over known latencies, with unknown latency ignored.
  - `test_verdict_counts_on_tool_node`: block, steer and a gate with `allow: false`.
  - `test_slowest_path_follows_largest_child`.
  - `test_window_and_run_scope`, including the 200-run cap.
  - `test_judge_meter_calls_excluded`.
  - `test_graph_endpoint_errors` (422 bad window, 404 run of another workflow).
- [ ] **Step 2: Run them.** Run `uv run pytest tests/server/test_graph.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Use one query per scope: `events WHERE run_id IN (<the scoped run ids>)`. Compute the p50 with `statistics.median`.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(server): run graph built from agent events and span links"` with explicit paths.

### Task 9: What changed: series and change markers

**Files:**
- Create: `src/pitstop/server/changes.py`, `tests/server/test_changes.py`, `tests/server/test_scale.py`
- Modify: `src/pitstop/server/models.py`, `src/pitstop/server/routes.py`

**Interfaces:**
- Consumes: `detect.metrics.run_metrics`, `detect.metrics.METRICS`, `drifts.evidence[-1].detail.change_at` and `.metric`.
- Produces:
  - **`change_series(store, workflow, *, limit: int = 200) -> Changes`.**
  - **The model.**
    - `Changes{points: [Point{run_id, started_at, status, source: "sdk"|"otel", metrics: dict[str, float]}], markers: [Marker], metrics: [MetricInfo{name, label, unit}]}`.
    - `Marker{at, kind: "model"|"base_url"|"prompt_versions"|"l2_alarm", before, after, run_id, drift_id, metric, source}`.
  - **Content.**
    - Points are oldest first. Runs without metrics (refused, dropped) are left out.
    - Markers follow decision 10.
    - Labels: `tool_error_rate` "Tool errors" (%), `refusal_rate` "Refusals" (%), `cost_usd` "Cost per run" ($), `tokens_per_call` "Tokens per call", `llm_calls` "LLM calls", `duration_s` "Duration" (s), `output_chars` "Output length" (chars).
  - **Endpoint:** `GET /api/changes?workflow=&limit=200` (max 200).

- [ ] **Step 1: Write the failing tests.**
  - `test_points_oldest_first_without_refused`.
  - `test_model_marker_same_source_only`: alternating OTel and SDK runs on different models give no marker. A real model change within the SDK runs gives exactly one, with `before` and `after`.
  - `test_prompt_versions_marker`.
  - `test_l2_alarm_marker_from_drift_evidence`.
  - `test_large_store_responses_are_capped` (Review Focus 2):
    - Seed 5,000 runs and 100,000 events across 3 workflows with `executemany`.
    - Assert the caps on `/api/board`, `/api/feed`, `/api/runs`, `/api/graph` and `/api/changes`.
    - Assert that `EXPLAIN QUERY PLAN` for the graph and changes event queries uses `idx_events_run`.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/server -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(server): what-changed series and change markers"` with explicit paths.

### Task 10: Offline dashboard demo data

**Files:**
- Create: `examples/quickstart/dashboard_demo.py`, `tests/server/test_dashboard_demo.py`
- Modify: `examples/quickstart/README.md` (create it if missing)

**Interfaces:**
- Consumes:
  - the public SDK (`pitstop.init(home=...)`, `wrap`, `workflow`, `agent`, `tool`);
  - the scripted HTTP fakes pattern from `tests/sdk/http_fakes.py`: copy what the script needs, because examples must not import `tests`;
  - `pitstop.detect.scan.scan`.
- Produces: `python examples/quickstart/dashboard_demo.py --home DIR [--runs 60]`. It fills a fresh PitStop home through the real SDK, with no keys and no network:
  - `earnings-research` (high criticality): 60 runs on model `m-a`, then 35 on `m-b` with double the tokens per call, so the L2 scan alarms and L5 names the model change. Its last run is halted by an off-task side-effect call blocked by the scripted gate (P0).
  - `support-triage` (medium): a run with a repeated tool loop that is steered twice (a P1/P2 drift).
  - `nightly-digest` and `kb-sync` (low): healthy runs only.
  - A workflow budget of $2/day on `earnings-research`.
  - At the end it runs `scan(store)` and prints `Seeded <n> runs in <home>. Run: PITSTOP_HOME=<home> pitstop ui`.
  - It refuses a `--home` that already has a `pitstop.db` (exit 2, `pick an empty directory`), so it can never touch the owner's real store.

- [ ] **Step 1: Write the failing test.** `test_dashboard_demo_seeds_expected_board`:
  - run it as a subprocess with `--home tmp_path/home --runs 40`, with keys scrubbed and `cwd=tmp_path`;
  - then, through `create_app` and `TestClient`, check `/api/board`: `earnings-research` is red, `support-triage` is yellow, the other two are green, and `/api/changes` for `earnings-research` has a `model` marker and an `l2_alarm` marker.

  Also `test_dashboard_demo_refuses_existing_db`.
- [ ] **Step 2: Run them.** Run `uv run pytest tests/server/test_dashboard_demo.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Tune the run counts so the L2 alarm fires with the M2c parameters (warm-up 30, normal sensitivity). Use the default `--runs` in the README.
- [ ] **Step 4: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(example): offline dashboard demo data"` with explicit paths.

### Task 11: UI scaffold, design tokens, API client and the empty state

**Files:**
- Create:
  - `ui/package.json`, `ui/package-lock.json`, `ui/tsconfig.json`, `ui/vite.config.ts`, `ui/index.html`;
  - `ui/openapi.json` and `ui/src/api/schema.d.ts` (generated);
  - `ui/src/api/client.ts`, `ui/src/api/poll.ts`;
  - `ui/src/main.tsx`, `ui/src/App.tsx`;
  - `ui/src/styles/tokens.css`, `ui/src/styles/base.css`;
  - `ui/src/components/{Shell,ThemeToggle,Level,Flag,Money,EmptyState,ErrorState}.tsx`;
  - `ui/src/routes.ts`;
  - `ui/src/test/setup.ts`, `ui/src/test/fixtures/*.json`;
  - `ui/src/**/*.test.tsx`;
  - `tests/server/test_openapi_snapshot.py`.
- Modify: `.github/workflows/ci.yml` (add the `ui` job and the wheel check), `CONTRIBUTING.md` (UI dev setup).

**Interfaces:**
- Consumes: `export_openapi()` (Task 6) and the endpoints from Tasks 7–9.
- Produces:
  - **npm scripts:**
    - `dev`: Vite on 5173, proxying `/api` to `http://127.0.0.1:8787`;
    - `build`: `tsc -b && vite build` with `build.outDir = "../src/pitstop/server/static"` and `emptyOutDir: true`;
    - `typecheck`, `test` (Vitest with jsdom);
    - `gen:api`: `openapi-typescript openapi.json -o src/api/schema.d.ts`.
  - **The API client.** `client.ts` exports typed functions:
    - `getBoard()`, `getFeed(limit)`, `getWorkflow(name)`, `dismissDrift(id)` (sends `X-PitStop: 1`), `explainDrift(id)`;
    - `getRuns(workflow, before?)`, `getRun(id)`, `getGraph(workflow, {runId?, window})`, `getChanges(workflow)`, `getSettings()`, `getHealth()`.

    Each uses the types from `schema.d.ts`, builds query strings with `URLSearchParams`, and throws `ApiError{status, message}`.
  - **Polling.** `usePoll<T>(fn, intervalMs)` in `poll.ts` returns `{data, error, loading, refresh}`. It pauses on `document.hidden` and aborts in-flight requests on unmount.
  - **Routes** (React Router 7, `createBrowserRouter`):
    - `/` is the pit wall;
    - `/w?name=&tab=graph|changes|runs` is the workflow page;
    - `/run?id=` is run detail;
    - `/settings` is settings.

    `routes.ts` exports `workflowHref(name, tab?)` and `runHref(id)`.
  - **Tokens and theme.**
    - `tokens.css` carries the mockups' token set from `workflow-detail.html`: `--bg`, `--surface`, `--sunken`, `--line`, `--line-soft`, `--ink`, `--ink-2`, `--ink-3`, `--p0`…`--p3` with their `-bg` variants, `--ok`, `--accent` and `--change`.
    - Dark values sit under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`.
    - `ThemeToggle` cycles system, light and dark and stores the choice in `localStorage` (in try/catch).
  - **Shell.** The PitStop wordmark, the nav (Pit wall, Settings), the theme toggle and a worker status dot with a text label. It reads `/api/health` every 30 s.
  - **Building blocks.**
    - `Level` is a P-level chip: always the text `P0`…`P3`, plus color.
    - `Flag` is a flag glyph plus text label.
    - `Money` formats USD: 2 decimals at $1 or more, 4 decimals below, and `—` for null.
  - **`EmptyState`** appears when the board has no workflows. It teaches the few-lines setup:
    - `pip install "pitstop[ui]"`;
    - a 6-line snippet with `pitstop.wrap(OpenAI())` and `with pitstop.workflow("my-agent", criticality="medium"):`;
    - `pitstop ui`;
    - a pointer to `python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo`.
  - **The schema snapshot.** `tests/server/test_openapi_snapshot.py::test_committed_openapi_is_current` asserts that `json.load(ui/openapi.json) == export_openapi()`. On failure it says to run `uv run python -c "…"` and `npm --prefix ui run gen:api`. Put that one-line regen command in `CONTRIBUTING.md`.

- [ ] **Step 1: Scaffold `ui/`** with the pinned majors (Tech Stack), then `npm install` to produce the lockfile. Add `@fontsource/instrument-sans` and `@fontsource/newsreader`. Export `ui/openapi.json` and run `gen:api`.
- [ ] **Step 2: Write the failing tests.**
  - Vitest:
    - `client builds query strings and sends X-PitStop on writes` (with a stubbed `fetch`);
    - `usePoll pauses when hidden`;
    - `Level always renders its text`;
    - `Money formats`;
    - `EmptyState shows the setup snippet`;
    - `encodes workflow names in links` (Review Focus 3: `workflowHref("a/b ?#🏁")` round-trips through `URLSearchParams`).
  - pytest: `test_committed_openapi_is_current`.
- [ ] **Step 3: Implement** the shell, tokens, components, client and routes. The page components are placeholders filled by Tasks 12–15.
- [ ] **Step 4: Add the CI jobs.**
  - The `ui` job runs on ubuntu-latest with `actions/setup-node@v4` (node 22, npm cache): `npm --prefix ui ci`, `npm --prefix ui run typecheck`, `npm --prefix ui test -- --run` and `npm --prefix ui run build`.
  - It then runs `uv build --wheel -o "$RUNNER_TEMP/wheel"` and `python -c` to open the wheel with `zipfile` and assert that `pitstop/server/static/index.html` is in it.
- [ ] **Step 5: Run the checks.** Run `uv run pytest -q && uv run ruff check && npm --prefix ui run typecheck && npm --prefix ui test -- --run && npm --prefix ui run build`, then `uv build --wheel -o <a new temp dir outside the repo>` and the zip check. Expected: all PASS. Never build into the repo's `dist/`.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(ui): dashboard scaffold, tokens, typed API client, empty state; CI ui job"` with explicit paths, including `ui/package-lock.json` and `ui/openapi.json`. Never commit `src/pitstop/server/static/` or `ui/node_modules/`; add `ui/node_modules/` to `.gitignore`.

### Task 12: Pit wall and live feed

**Files:**
- Create: `ui/src/pages/PitWall.tsx`, `ui/src/components/{Tile,TileSmall,Feed,BudgetBar}.tsx`, `ui/src/pages/PitWall.test.tsx`

**Interfaces:**
- Consumes: `getBoard`, `getFeed`, `usePoll` (Task 11), and the fixtures written from the Task 10 demo.
- Produces: the home page, which follows mockup B in `health-board-directions.html`.
  - **Header.** "Pit wall", with a count line ("2 workflows need you") and the global budgets as spent / limit with a bar. The bar is not the only cue: it also shows text.
  - **Large tiles** (red and yellow flags):
    - flag with label, workflow name, criticality;
    - P-level chips with counts, for non-zero levels only;
    - the headline (alert title) in plain language;
    - last run status and time ago;
    - today's spend against the daily budget;
    - "Fallback active" when set.
    - The whole tile links to the workflow page.
  - **Small tiles** (green): name, criticality, runs today, spend. They are compact, in a wrapping grid.
  - **Live feed** in the right column (stacked below the tiles at narrow widths): newest first, with kind, workflow (linked), level chip, text and time ago. New items fade in: `opacity` only, 150 ms, and none under reduced motion.
  - **States:** loading skeletons (no spinners), an error banner with retry, and the `EmptyState` when there are no workflows.

- [ ] **Step 1: Write the failing tests** with fixtures:
  - `renders red and yellow as large tiles and green as small ones`;
  - `tile shows text flag label and level chips`;
  - `feed lists newest first and links workflows`;
  - `empty board shows EmptyState`;
  - `error shows retry`.
- [ ] **Step 2: Run them.** Run `npm --prefix ui test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Before marking it done, follow the impeccable `frontend-design` skill's checklist. Run the demo and check the page in a browser at 1280 px and 390 px, in light and dark mode:
  - `python examples/quickstart/dashboard_demo.py --home <tmp>`;
  - `PITSTOP_HOME=<tmp> uv run pitstop serve`;
  - `npm --prefix ui run dev`.

  Report what you checked.
- [ ] **Step 4: Run the checks.** Run the full check set (Global Constraints). Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(ui): pit wall with live feed"` with explicit paths.

### Task 13: Workflow page, right column and the Run graph tab

**Files:**
- Create: `ui/src/pages/Workflow.tsx`, `ui/src/components/{DriftCard,ExplainPanel,RunGraph,NodeDrawer,WindowPicker}.tsx`, `ui/src/graph/layout.ts` (ELK), `ui/src/pages/Workflow.test.tsx`, `ui/src/graph/layout.test.ts`

**Interfaces:**
- Consumes: `getWorkflow`, `getGraph`, `dismissDrift`, `explainDrift`, `getRuns` (filtered to the node's runs client-side from the scope's runs).
- Produces: the workflow page, following `workflow-detail.html`.
  - **Header:** the tile facts (flag, criticality, spend against budget, last run).
  - **Tabs:** Run graph (default), What changed, Runs. The tab is in the URL (`tab=`).
  - **Right column**, always visible: open drifts as `DriftCard`s (level chip, title, cause, what PitStop is doing, occurrences, first and last seen), with **Explain** and **Dismiss**.
    - Explain opens the `ExplainPanel` inline, below the card. It is not a modal.
    - Dismiss uses an optimistic update and restores the card with an error line on failure. It confirms with an inline "Dismiss? Yes / No", not a browser dialog.
  - **Run graph.**
    - `@xyflow/react` nodes laid out by `elkjs` (`elk.algorithm: layered`, `elk.direction: RIGHT`), computed in `layout.ts`. The layout is memoized per graph.
    - Each node shows its label, calls, cost, p50 latency and verdict badges. The badges are text with color, e.g. "2 blocked".
    - The slowest path's edges and nodes are emphasized with weight and an "on slowest path" label, not color alone.
    - A `WindowPicker` (1h / 24h / 7d / 30d) and a run selector (latest 20 runs) set the scope.
    - Clicking a node opens the `NodeDrawer` (a side panel) with the node's stats, its verdicts and links to its runs.
    - Under reduced motion there is no animated fit-view.

- [ ] **Step 1: Write the failing tests:**
  - `layout places root left of agents and leaves`: run `layout.ts` on the Task 8 fixture graph and assert x ordering;
  - `slowest path is labelled`;
  - `dismiss is optimistic and restores on failure`;
  - `explain opens inline`;
  - `tab is reflected in the URL`;
  - `unknown workflow shows not-found state`.
- [ ] **Step 2: Run them.** Run `npm --prefix ui test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Do the same browser check as Task 12 against the demo data, including `earnings-research`'s graph, and report it.
- [ ] **Step 4: Run the checks.** Run the full check set. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(ui): workflow page with run graph, open drifts, explain and dismiss"` with explicit paths.

### Task 14: What changed tab, Runs tab and run detail

**Files:**
- Create: `ui/src/components/{SeriesChart,MarkerLegend,RunsTable,EventTimeline}.tsx`, `ui/src/pages/RunDetail.tsx`, their `*.test.tsx`
- Modify: `ui/src/pages/Workflow.tsx`

**Interfaces:**
- Consumes: `getChanges`, `getRuns`, `getRun`, `getGraph(workflow, {runId})`.
- Produces:
  - **What changed.** One small-multiple `SeriesChart` per metric that has data, as a hand-written SVG with no chart library.
    - x is time and y is the value, with axis labels and units.
    - Points are colored by run status; failed runs are hollow.
    - Markers are vertical lines in `--change` with a text label ("model m-a → m-b", "L2 alarm: tokens per call").
    - Hovering a point shows run id, value and status; a click goes to run detail.
    - A `MarkerLegend` lists the markers in order, each with a link to Explain for an L2 alarm.
    - A note says "Quality score arrives with the run judge (milestone 3b)". **Copy:** "Quality score: coming with the run judge."
  - **Runs.** `RunsTable` with status, started, duration, model, adapter, cost, LLM calls, tool calls and verdicts. It has a "Load more" cursor, and each row links to run detail.
  - **Run detail** at `/run?id=`:
    - the run facts;
    - the run's own graph (`getGraph` with `runId`);
    - an `EventTimeline` of `EventRow`s grouped by agent, with kind icons plus text, the summary, latency and cost;
    - the line "Showing the first 2,000 events" when `truncated`.

- [ ] **Step 1: Write the failing tests:**
  - `chart draws one path per metric and labels markers`;
  - `point click navigates to run`;
  - `runs table loads more with cursor`;
  - `run detail shows truncated notice`.
- [ ] **Step 2: Run them.** Run `npm --prefix ui test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.** Do the browser check against the demo data: the `earnings-research` What-changed tab shows the model marker where tokens per call steps up. Report it.
- [ ] **Step 4: Run the checks.** Run the full check set. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(ui): what-changed charts, runs table and run detail"` with explicit paths.

### Task 15: Settings page

**Files:**
- Create: `ui/src/pages/Settings.tsx`, `ui/src/pages/Settings.test.tsx`

**Interfaces:**
- Consumes: `getSettings`, `getHealth`.
- Produces: a read-only settings page with these sections:
  - **Workflows:** a table of criticality, sensitivity, fix mode and daily budget, with a note "Set in code: `pitstop.workflow(..., criticality=…)`".
  - **Budgets:** global budgets with spent and limit.
  - **Pre-action judge:** model, base URL, the key env var name and "set" / "not set".
  - **Alert channels:** name, configured, env var name and set / not set, plus the routes per P-level.
  - **Background worker:** enabled, interval, last scan and last error.
  - **Where data lives:** the `PITSTOP_HOME` path.

  Every section that comes from config says "Edit `~/.pitstop/config.toml`, see docs/config.example.toml". The page never shows an env var's value (the API doesn't have it).

- [ ] **Step 1: Write the failing tests:** `renders each section from fixture`; `shows not set for missing keys`.
- [ ] **Step 2: Run them.** Run `npm --prefix ui test -- --run`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the checks.** Run the full check set. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(ui): read-only settings page"` with explicit paths.

### Task 16: Milestone wrap-up

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md` (milestone status line only), this plan (the "Carried to milestone 3b" section)

- [ ] **Step 1: Update the README.**
  - Add a **Dashboard** section after "Add PitStop to your agent". It covers:
    - `pip install "pitstop[ui]"` (from the repo: `uv sync`, then `uv run pitstop ui`);
    - what each page shows;
    - the offline demo command;
    - `--host` safety;
    - that the worker replaces a cron `pitstop scan`.
  - Update the status blurb, the TOC (including the "Where it's going" subsections, a carry item), the Roadmap (3a done; 3b and 3c next) and the repo map (`src/pitstop/server/`, `ui/`).
  - The README must not claim screenshots yet (3c).
- [ ] **Step 2: Update `CHANGELOG.md` (Unreleased).**
  - Add the dashboard, server, worker, endpoints, latency fields, `agent` events, `dismiss` and the carry fixes.
  - Split the long swarm bullet (a carry item) into two.
- [ ] **Step 3: Write the "Carried to milestone 3b" section** at the end of this plan: decision 14's items, plus anything the reviews park.
- [ ] **Step 4: Run the checks.** Run the full check set. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "docs: milestone 3a complete; dashboard quickstart"` with explicit paths.

## M2c carry list: where each item goes

| Carried item | Task |
|---|---|
| OTel alert cause residual | 1 |
| Run sources sharing a workflow name | 1 (alert causes), 9 (markers), decision 14 (L2/L5 mixing) |
| `notify` store; unrecorded-P0 latency | 1 |
| OTel processor: end-time agent identity, late child, threaded test, logging gate | 3 |
| OTel `pitstop.side_effects` attribute | decision 14 (Later) |
| Agents SDK: end restores name, `__signature__`, `_tag_run` once | 3 |
| LangChain: requested model name, `langgraph` dependency, sleep in test | 3 |
| LangChain: one event per model-node turn; mixed handler | decision 14 (documented, Task 3) |
| Adapters: async hooks write on the loop | decision 14 |
| Guard: nested same-name calls | 2 (one-shot mark) |
| Guard: `create_task` inherits mark | decision 14 (documented, Task 2) |
| `emit_verdict` import retry | 2 |
| L2/L5/scan: failed attribution, empty-after summary, negative duration, reconcile `inf` | 2 |
| L2/L5/scan: `run_metrics` per run | decision 14 |
| Budgets: refused cause | 1 |
| Budgets: refused rows in `status` | decision 14 |
| Alerts: `notify_unrecorded` store-refused test, SMTP/SSL classes | 1 |
| Gate: near-miss closers | 2 |
| wrap: mixed `[int, str]` prompt | 2 |
| SDK runtime and tests (steer drift id, docstrings, dispatch and context tests, explain tests, finresearch grep, long lines) | 1, 2 |
| Example (finresearch) items | 4 |
| Docs: README TOC, long CHANGELOG bullet | 16 |

## Carried to milestone 3b

Items not done in 3a, each with its reason.

- The opt-in `pitstop.side_effects` span attribute: a feature, deferred to "Later".
- Async adapter hooks writing SQLite on the event loop: WAL writes are sub-millisecond, so this is revisited only on a report.
- L2 streams and L5 windows mixing OTel and SDK runs of one workflow: by design (M2c decision 1). The dashboard separates sources in its change markers.
- LangChain's "one observed event per model-node turn" and "a handler mixing wrapped and unwrapped calls hides the unwrapped one": documented as known limitations in the README.
- `create_task` inside a marked call inheriting the guard mark: documented in `guard.py`, revisited with M4.
- Attribution calling `run_metrics` per run: at most 100 small queries.
- Refused runs counting in `pitstop status`: documented. The dashboard labels them "refused".
- Upgrade Vitest to a major without the npm-audit advisories (`@vitest/mocker` moderate, `tinypool` critical; dev-only).
- A quality series (judge pass rate) for What changed arrives with L3.
- README screenshots and the hero GIF are 3c.

### From the milestone 3a reviews

(added after the final review)
