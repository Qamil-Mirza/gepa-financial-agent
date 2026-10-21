# PitStop Milestone 2a: SDK Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A developer adds a few lines to an existing agent and gets hard dollar budgets, cheap inline drift checks, P0–P3 urgency labels, steer/block/halt actions, managed prompts, and a CLI that shows it all. PitStop's own model calls go through the same budget, so a global $50 Crusoe cap applies to everything.

**Architecture:**
- **Shared state.** Everything lives in one local SQLite file (`$PITSTOP_HOME/pitstop.db`, WAL mode), so separate processes share budgets and drift records.
- **Model calls.** `pitstop.wrap(client)` proxies an OpenAI-compatible client's `chat.completions.create`, sync and async, streaming and non-streaming. Before each call it reserves an estimated cost against every budget that applies. After the call it settles the actual usage, records the call, and runs L0 checks on the response.
- **Tools.** `@pitstop.tool` guards tool execution, which is where the block and steer actions happen.
- **Scoping.** `pitstop.workflow` and `pitstop.agent` set the run's scope through `contextvars`, so concurrent asyncio tasks stay separate.
- **Not in this milestone.** No server, no LLM judge, no alerts (those are milestone 2b).

**Tech Stack:** Python ≥3.11 (stdlib `sqlite3`, `tomllib`, `contextvars`), the `openai` SDK (an existing dependency), rich for the CLI, pytest with fake clients (no network).

**Spec:** `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md`. This plan covers §3 (minus OTel and framework adapters), §3.1 (steer, block and halt; fallback is in 2b), §3.2, §4 (store and inline side only), §5 and §6.1–6.2 (L0 only). It also covers the "$50 hard budget from day one" item from §10.

**Milestone split (ruling, recorded for the owner):** Spec milestone 2 is split in two.
- **2a (this plan):** the SDK core.
- **2b (next plan):** the OTel span processor, the OpenAI Agents SDK and LangGraph adapters, the Anthropic SDK in `wrap`, L1/L2/L5 (including the spend-rate-vs-baseline and steps/tokens-vs-p95 checks, which need run history), alert routing including P0 alerts, the fallback action, moving the research swarm to `examples/financial-research/`, and the deferred M1 minors.

## Global Constraints

- Python ≥3.11. No new runtime dependencies beyond what `pyproject.toml` already lists (`tomllib`, `sqlite3` and `contextvars` are stdlib).
- **Data location.** All state lives under `pitstop.config.pitstop_home()` (`PITSTOP_HOME`, default `~/.pitstop`): the DB is `pitstop.db` and user config is `config.toml`. Never write into the source tree or the cwd.
- **No network in tests.** Tests use fake clients (the pytest-socket guard is already on). No test may spend money.
- **Never break the user's agent.**
  - `wrap` must behave like the wrapped client for everything it does not intercept: `__getattr__` delegation, no global monkeypatching.
  - PitStop's own internal errors (DB locked, a bug in a check) are logged and never raised into user code. The only exceptions PitStop raises on purpose are `pitstop.Halted` and its subclass `pitstop.BudgetExceeded`.
- **Stream boundaries.** Actions take effect at the next call boundary. A stream is never cut mid-flight (spec §3.1).
- **Budget limits are hard stops** (spec §3.2): estimate and reserve before the call, settle after.
- **Commits carry no attribution trailers.** No `Co-Authored-By`, no "Generated with" lines.
- Follow the existing code style: dataclasses and pydantic as already used, `from __future__ import annotations`, a ruff `F` clean pass, and module docstrings at the density of `src/pitstop/supervisor/*.py`.

## Review Focus

1. **Two processes or threads spend against the same $50 budget at once.** They must never jointly overshoot by more than one in-flight estimate. Reservations happen inside `BEGIN IMMEDIATE`. Pinned in Task 2 (`test_concurrent_reservations_never_exceed_limit`).
2. **A call raises, or a stream ends without usage.** The reservation is always settled or released, never left hanging. A missing stream usage is settled from character counts. Pinned in Task 4 (`test_failed_call_releases_reservation`, `test_stream_without_usage_settles_by_estimate`).
3. **`wrap` used outside any workflow.** It still enforces global budgets, records the call under workflow `None`, and never crashes. Pinned in Task 4 (`test_wrap_outside_workflow_counts_global_budget`).
4. **Two workflows running concurrently in `asyncio.gather`.** Runs, agents and steer counts stay isolated per task. Pinned in Task 3 (`test_concurrent_async_workflows_are_isolated`).
5. **The user's own exception inside a workflow.** It propagates unchanged and the run is recorded as `error`. `Halted` propagates and the run is recorded as `halted`. A PitStop DB failure never surfaces as an exception in user code. Pinned in Task 3 (`test_user_exception_propagates_and_run_marked_error`) and Task 4 (`test_store_failure_does_not_break_user_call`).

---

## File structure (new unless noted)

```
src/pitstop/
  __init__.py            MODIFY: public API re-exports (init, wrap, workflow, agent, tool, prompt, Halted, BudgetExceeded)
  userconfig.py          load $PITSTOP_HOME/config.toml → UserConfig (budgets, prices, timezone)
  store/__init__.py      Store: connection, schema, migrations, transactions
  store/schema.sql       DDL (package data)
  budget/__init__.py
  budget/prices.py       Price, PRICES_PER_M (moved from correct/client.py), CONSERVATIVE_PRICE, price_for, estimate_cost_usd
  budget/ledger.py       Ledger: reserve/settle/release, period windows, BudgetExceeded
  sdk/__init__.py
  sdk/errors.py          Halted, BudgetExceeded
  sdk/runtime.py         init(), get_runtime() (store + config + ledger singleton per PITSTOP_HOME)
  sdk/context.py         WorkflowSpec, RunContext, workflow(), agent(), current_run()
  sdk/wrap.py            wrap(): sync/async/stream proxies of chat.completions.create
  sdk/tool.py            @tool decorator: scope check, loop check, block/steer/halt
  sdk/prompt.py          prompt(): managed prompt registry
  detect/__init__.py
  detect/l0.py           pure L0 checks → Finding
  detect/urgency.py      Level, classify(), action_for()
  detect/drifts.py       record_drift() with dedup
  cli.py                 MODIFY: add `status`, `drifts`, `budget` subcommands
  llm.py                 MODIFY: build its AsyncOpenAI through wrap()
  correct/client.py      MODIFY: build its OpenAI/AsyncOpenAI through wrap(); import prices from budget.prices
examples/quickstart/agent.py   ~60-line tool-using agent showing the few lines; --inject-loop flag
examples/quickstart/README.md
tests/sdk/fakes.py             FakeOpenAI / FakeAsyncOpenAI with scripted responses and streams
tests/sdk/test_*.py, tests/store/, tests/budget/, tests/detect/
```

---

### Task 1: Store and user config

**Files:**
- Create: `src/pitstop/store/__init__.py`, `src/pitstop/store/schema.sql`, `src/pitstop/userconfig.py`
- Test: `tests/store/test_store.py`, `tests/store/test_userconfig.py`

**Interfaces:**
- Produces `pitstop.store`:
  - `class Store`:
    - `__init__(self, path: Path)` creates parent dirs, sets `journal_mode=WAL` and `busy_timeout=5000`, and applies the schema.
    - `connect() -> sqlite3.Connection` returns a per-thread connection with row factory `sqlite3.Row`.
    - `transaction(immediate: bool = False) -> ContextManager[sqlite3.Connection]` opens with `BEGIN IMMEDIATE` when `immediate`, commits on success and rolls back on exception.
    - `schema_version() -> int`.
  - `SCHEMA_VERSION = 1`
- Schema (`schema.sql`, version 1):
  - `meta(key TEXT PRIMARY KEY, value TEXT)`
  - `runs(id TEXT PRIMARY KEY, workflow TEXT, goal TEXT, criticality TEXT, sensitivity TEXT, fix_mode TEXT, started_at REAL, ended_at REAL, status TEXT, model TEXT, base_url TEXT, prompt_versions TEXT, pitstop_version TEXT, adapter TEXT, price_known INTEGER DEFAULT 1)`
  - `events(id INTEGER PRIMARY KEY, run_id TEXT, ts REAL, agent TEXT, kind TEXT, name TEXT, payload TEXT, cost_usd REAL, parent_id INTEGER)`
  - `drifts(id INTEGER PRIMARY KEY, run_id TEXT, workflow TEXT, ts REAL, level TEXT, kind TEXT, drift_type TEXT, evidence TEXT, action TEXT, status TEXT DEFAULT 'open')`
  - `ledger(id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, workflow TEXT, meter TEXT, model TEXT, base_url TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, usd REAL, state TEXT)`. `state` is `reserved | settled | released`.
  - `prompts(name TEXT, version INTEGER, text TEXT, sha TEXT, source TEXT, created_at REAL, active INTEGER, PRIMARY KEY(name, version))`
  - Indexes on `events(run_id)`, `drifts(workflow, status)`, `ledger(ts)`.
- Produces `pitstop.userconfig`:
  - `@dataclass(frozen=True) BudgetSpec(name: str, limit_usd: float, period: Literal["total","day","month"], base_url: str | None)`
  - `@dataclass(frozen=True) PriceOverride(input: float, output: float, cached_input: float)`
  - `@dataclass(frozen=True) UserConfig(budgets: tuple[BudgetSpec, ...], prices: dict[str, PriceOverride], timezone: str | None)`
  - `load_user_config(path: Path | None = None) -> UserConfig`. The default path is `pitstop_home() / "config.toml"`. A missing file returns an empty config. An invalid period or a negative limit raises `ValueError` naming the budget.
  - TOML format:
    ```toml
    timezone = "America/Los_Angeles"   # optional; default system local
    [budgets.crusoe]
    base_url = "https://api.inference.crusoecloud.com/v1"   # optional; omitted = applies to every call
    limit_usd = 50
    period = "total"
    [prices."nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"]
    input = 0.30
    output = 2.40
    cached_input = 0.15
    ```

- [ ] **Step 1: Write the failing tests.**
  - `test_store_creates_db_with_wal_and_schema(tmp_path)` checks that `pitstop.db` exists, `PRAGMA journal_mode` is `wal`, `schema_version() == 1`, and all six tables exist.
  - `test_transaction_rolls_back_on_error(tmp_path)` checks that an insert followed by a raise leaves 0 rows.
  - `test_store_is_reopenable_and_idempotent(tmp_path)` checks that a second `Store(path)` on the same file keeps the rows and the version.
  - `test_missing_config_is_empty(tmp_path)` checks that `load_user_config(tmp_path/"none.toml")` returns `budgets == ()` and `prices == {}`.
  - `test_config_parses_budgets_and_prices(tmp_path)` uses the TOML above and checks that `budgets[0] == BudgetSpec("crusoe", 50.0, "total", "https://api.inference.crusoecloud.com/v1")` and that the price override is parsed.
  - `test_invalid_period_raises(tmp_path)` checks that `period = "weekly"` raises `ValueError` matching `crusoe`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/store -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement** `Store` and `load_user_config`.
  - Ship `schema.sql` as package data. Hatch includes it, and the existing wheel test pattern covers it, so add `"pitstop/store/schema.sql"` to that test's required list in `tests/runtime/test_packaging.py`.
  - Connections are per-thread via `threading.local`.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/store tests/runtime/test_packaging.py -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(store): local SQLite store and user config loader"`

---

### Task 2: Prices and budget ledger

**Files:**
- Create: `src/pitstop/budget/__init__.py`, `src/pitstop/budget/prices.py`, `src/pitstop/budget/ledger.py`, `src/pitstop/sdk/__init__.py` (empty for now), `src/pitstop/sdk/errors.py`
- Modify: `src/pitstop/correct/client.py`. Delete `PRICES_PER_M` and `estimate_cost_usd` there and import them from `pitstop.budget.prices`. Existing callers (`benchmarks/calibration/summarize.py`, tests) keep working through that import.
- Test: `tests/budget/test_prices.py`, `tests/budget/test_ledger.py`

**Interfaces:**
- Consumes: `Store` and `UserConfig` from Task 1.
- Produces in `pitstop.budget.prices`:
  - `@dataclass(frozen=True) Price(input: float, output: float, cached_input: float)` (USD per 1M tokens)
  - `PRICES_PER_M: dict[str, Price]`, with the same entries and values as today's table in `correct/client.py`
  - `CONSERVATIVE_PRICE = Price(5.00, 15.00, 5.00)`
  - `price_for(model: str, overrides: dict[str, PriceOverride]) -> tuple[Price, bool]` returns `(price, known)`. Overrides win, then the table, then `(CONSERVATIVE_PRICE, False)`.
  - `cost_usd(price: Price, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float`
  - `estimate_cost_usd(model, prompt_tokens, completion_tokens, cached_tokens=0) -> float | None` keeps its existing signature and behavior (None when unknown).
- Produces in `pitstop.budget.ledger`:
  - `@dataclass Reservation(id: int, usd: float, scopes: tuple[str, ...], model: str, base_url: str | None, price: Price, price_known: bool)`
  - `class Ledger`:
    - `__init__(self, store: Store, config: UserConfig, now: Callable[[], float] = time.time)`
    - `reserve(self, *, model: str, base_url: str | None, est_prompt_tokens: int, max_completion_tokens: int, run_id: str | None, workflow: str | None, workflow_budget_usd: float | None, meter: str = "agent") -> Reservation`. It raises `BudgetExceeded`.
    - `settle(self, res: Reservation, *, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float` returns the actual USD.
    - `release(self, res: Reservation) -> None`
    - `spent(self, scope: str) -> float` counts settled plus reserved spend in the current period window for that scope.
  - Scope keys: `"global:<budget name>"` for each `BudgetSpec` whose `base_url` is None or equal to the call's `base_url` after trailing-`/` normalization; `"workflow:<name>"` when `workflow_budget_usd` is set. A workflow budget's period is always `day`.
  - **Period windows:** `total` is all time. `day` and `month` are calendar periods in `UserConfig.timezone`, or the system local timezone when that is None.
  - **Workflow budgets** count only `meter == "agent"` rows for that workflow. **Global budgets** count all meters (spec §3.2).
- Produces in `pitstop.sdk.errors` (create the file in this task):
  - `class Halted(Exception)` with `__init__(self, reason: str, drift_id: int | None = None)` and attributes `reason` and `drift_id`
  - `class BudgetExceeded(Halted)` with attributes `scope: str`, `limit_usd: float`, `spent_usd: float`. Its message is `"PitStop budget exceeded: <scope> spent $X of $Y"`.

- [ ] **Step 1: Write the failing tests.**
  - `test_price_override_wins_then_table_then_conservative`
  - `test_cost_usd_matches_old_estimate` checks that `estimate_cost_usd("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B", 1_000_000, 1_000_000) == 0.30 + 2.40`.
  - `test_reserve_then_settle_records_actual` checks that after reserve and settle, `spent("global:crusoe")` equals the settled cost and no `reserved` rows remain.
  - `test_reserve_over_limit_raises_budget_exceeded` uses a limit of 0.01 and an estimate of 0.02, and checks that it raises `BudgetExceeded` with `scope == "global:crusoe"`.
  - `test_release_frees_reservation`
  - `test_base_url_scoping` checks that a call to another base_url does not count toward a budget with a `base_url`, and that a budget without a `base_url` counts everything.
  - `test_day_period_resets_at_local_midnight` uses an injected `now` and `timezone="UTC"`. Spend at 23:59 counts that day, and at 00:01 the next day the spend is 0.
  - `test_workflow_budget_counts_agent_meter_only`
  - `test_concurrent_reservations_never_exceed_limit` (Review Focus 1): 8 threads, each with its own `Ledger` on the same DB file, each try 20 reservations of $0.10 against a $5 limit. At the end, the sum of reserved and settled is ≤ $5.00 and the number of `BudgetExceeded` raised is ≥ 1.
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/budget -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** `reserve` sums `spent(scope)` for every scope and inserts the `reserved` row inside one `store.transaction(immediate=True)`, so the check and the insert are atomic across processes.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/budget tests/correct tests/benchmarks -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(budget): price table and atomic reserve/settle budget ledger"`

---

### Task 3: Runtime, workflow and agent context

**Files:**
- Create: `src/pitstop/sdk/runtime.py`, `src/pitstop/sdk/context.py`
- Modify: `src/pitstop/__init__.py` (re-export `init`, `workflow`, `agent`, `Halted`, `BudgetExceeded`; later tasks add more)
- Test: `tests/sdk/test_context.py`

**Interfaces:**
- Consumes: `Store`, `load_user_config`, `Ledger`, and the errors (Tasks 1–2).
- Produces in `pitstop.sdk.runtime`:
  - `@dataclass Runtime(store: Store, config: UserConfig, ledger: Ledger, home: Path)`
  - `init(config_path: Path | None = None, home: Path | None = None) -> Runtime` is idempotent per home and returns the existing runtime on repeated calls.
  - `get_runtime() -> Runtime` calls `init()` on first use.
  - `reset_runtime() -> None`, for tests.
- Produces in `pitstop.sdk.context`:
  - `Criticality = Literal["high","medium","low"]`; `Sensitivity = Literal["low","normal","high"]`
  - `@dataclass(frozen=True) WorkflowSpec(name: str, goal: str | None = None, criticality: Criticality = "medium", budget_usd: float | None = None, sensitivity: Sensitivity = "normal", fix_mode: Literal["manual","auto"] | None = None, tools: tuple[str, ...] | None = None)`
    - `fix_mode` None means the default by criticality: `high` is `manual`, otherwise `auto`. It is stored now and used in milestone 4.
    - `tools` None means no allowlist.
  - `@dataclass RunContext`, with fields:
    - `run_id: str` (uuid4 hex)
    - `spec: WorkflowSpec`
    - `agent: str | None`
    - `agent_goal: str | None`
    - `steers: int = 0`
    - `halt_reason: str | None = None`, set when a P0 asks to halt at the next boundary
    - `halt_drift_id: int | None = None`
    - `tool_history: list` (used by Task 6)
    - `llm_history: list` (used by Task 5)
  - `current_run() -> RunContext | None`
  - `workflow(name: str, *, goal=None, criticality="medium", budget_usd=None, sensitivity="normal", fix_mode=None, tools=None)` works both as a decorator (sync or async functions, preserving `functools.wraps`) and as a context manager (`with` and `async with`).
    - On entry it inserts a `runs` row with `status='running'`.
    - On exit it sets `ended_at` and the status: `ok`, `halted` (a `Halted` was raised) or `error` (any other exception). It always re-raises.
    - Nested `workflow()` inside a running workflow raises `RuntimeError("pitstop.workflow cannot be nested")`.
  - `agent(name: str, *, goal: str | None = None)` works as a decorator or context manager. It sets `RunContext.agent` and `agent_goal` for its scope and restores the previous values on exit. Outside a workflow it is a no-op that logs once.
  - `check_halt() -> None` raises `Halted(ctx.halt_reason, ctx.halt_drift_id)` if the current run has a pending halt. Tasks 4 and 6 call it at every boundary.
  - Run status writes go through a `_safe(fn)` helper that logs and swallows store errors (Global Constraints).

- [ ] **Step 1: Write the failing tests.**
  - `test_decorator_records_run_ok(tmp_path)`
  - `test_context_manager_sync_and_async(tmp_path)`
  - `test_user_exception_propagates_and_run_marked_error(tmp_path)` (Review Focus 5) checks that the same exception object propagates and `status == "error"`.
  - `test_halted_marks_run_halted`
  - `test_agent_sets_and_restores_name_and_goal`
  - `test_nested_workflow_raises`
  - `test_concurrent_async_workflows_are_isolated` (Review Focus 4): two `async with workflow(...)` blocks in `asyncio.gather` each see their own `current_run().run_id` and `agent` across `await`s.
  - `test_fix_mode_default_by_criticality`

  Each test uses `pitstop.init(home=tmp_path)` and `reset_runtime()` in an autouse fixture in `tests/sdk/conftest.py`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/sdk/test_context.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** with a module-level `ContextVar[RunContext | None]`.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/sdk -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(sdk): runtime init and workflow/agent run context"`

---

### Task 4: `pitstop.wrap`

**Files:**
- Create: `src/pitstop/sdk/wrap.py`, `tests/sdk/fakes.py`
- Modify: `src/pitstop/__init__.py` (export `wrap`)
- Test: `tests/sdk/test_wrap.py`

**Interfaces:**
- Consumes: `get_runtime`, `current_run`, `check_halt`, `Ledger.reserve/settle/release` (Tasks 2–3).
- Produces `wrap(client: T) -> T`:
  - The result proxies `client`. `wrapped.chat.completions.create(...)` is intercepted. Every other attribute and method is delegated unchanged through `__getattr__`.
  - Detect async by `inspect.iscoroutinefunction(client.chat.completions.create)`.
  - `wrap(wrap(c))` returns the same wrapper, so a client is never double-counted.
- Per intercepted call, in this order:
  1. Run `check_halt()`.
  2. Estimate the prompt tokens as `ceil(total_chars / 4)` over every message `content` (strings, or the text parts of list content) plus `json.dumps(tools)`. The max completion tokens come from `max_tokens` or `max_completion_tokens`, or `4096` when neither is set.
  3. Reserve with `meter="agent"`, the base URL from `str(client.base_url)` (or None if absent), and the run's workflow and budget. On `BudgetExceeded`, record a `budget_exhausted` drift via Task 5's hook (a no-op until Task 5), set nothing else, and raise it.
  4. Call the real `create`. If it raises, release the reservation and re-raise.
  5. **Non-stream:** settle from `response.usage` (`prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens` if present), record an `events` row (`kind='llm_call'`, `name=model`, payload `{"tool_calls": [...names and raw argument strings...], "finish_reason", "content_chars"}`, `cost_usd`), and call `_after_llm_call(ctx, response_summary)`. That hook is a no-op stub that Task 5 fills in.
  6. **Stream:** if the caller did not pass `stream_options`, set `stream_options={"include_usage": True}`. Return an iterator (or async iterator) that yields every chunk unchanged, accumulates content and tool-call deltas, and in `finally` settles using the usage chunk if one arrived, otherwise `ceil(chars/4)` for completion and the estimate for the prompt. It records the event the same way.
  7. On the first call in a run, update `runs.model`, `runs.base_url`, `runs.adapter='wrap'`, `runs.pitstop_version` and `runs.price_known`.
- Store and ledger internal errors other than `BudgetExceeded` are logged and the call proceeds unmetered. The ledger records nothing for it, and the event has `payload.unmetered=true`.
- `tests/sdk/fakes.py` provides `FakeOpenAI(responses: list)` and `FakeAsyncOpenAI(responses: list)`:
  - `.base_url` and `.chat.completions.create(**kw)` pop the next scripted item. An item is a response object with `.choices[0].message` (`content`, `tool_calls`) plus `.usage`, an `Exception` to raise, or a list of chunks for `stream=True`.
  - `.calls` records the kwargs of each call.
  - `make_response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5)` and `make_tool_call(name, arguments: str)` are helpers.

- [ ] **Step 1: Write the failing tests.**
  - `test_wrap_delegates_unknown_attributes` checks that `wrap(FakeOpenAI([])).models` is the fake's object.
  - `test_sync_call_settles_usage_and_records_event`
  - `test_async_call_settles_usage`
  - `test_stream_yields_all_chunks_and_settles_from_usage_chunk`
  - `test_stream_without_usage_settles_by_estimate` (Review Focus 2) checks that the ledger has 1 `settled` row, 0 `reserved` rows, and a cost > 0.
  - `test_failed_call_releases_reservation` (Review Focus 2) checks that the fake raises `RuntimeError`, the same error propagates, and the ledger has 0 `reserved` rows.
  - `test_budget_exceeded_raises_before_calling_client` checks that with a `limit_usd=0.000001` global budget, `BudgetExceeded` is raised and `fake.calls == []`.
  - `test_wrap_outside_workflow_counts_global_budget` (Review Focus 3)
  - `test_double_wrap_is_idempotent`
  - `test_store_failure_does_not_break_user_call` (Review Focus 5): monkeypatch `Ledger.reserve` to raise `sqlite3.OperationalError`. The call still returns the fake response.
  - `test_unknown_model_uses_conservative_price_and_flags_run`
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/sdk/test_wrap.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `wrap` and the fakes.
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/sdk -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(sdk): wrap() meters OpenAI-compatible calls against hard budgets"`

---

### Task 5: L0 checks, urgency levels and drift records

**Files:**
- Create: `src/pitstop/detect/__init__.py`, `src/pitstop/detect/l0.py`, `src/pitstop/detect/urgency.py`, `src/pitstop/detect/drifts.py`
- Modify: `src/pitstop/sdk/wrap.py` (fill in `_after_llm_call` and the budget-exhausted drift hook)
- Test: `tests/detect/test_l0.py`, `tests/detect/test_urgency.py`, `tests/detect/test_drifts.py`, `tests/sdk/test_wrap_detect.py`

**Interfaces:**
- Consumes: `RunContext` and `WorkflowSpec` (Task 3), and the wrap hook (Task 4).
- Produces in `pitstop.detect.l0`:
  - `@dataclass(frozen=True) ToolCallRecord(name: str, args: str, side_effects: bool)`. Here `args` is canonical JSON, or the raw string if it doesn't parse.
  - `@dataclass(frozen=True) LLMCallRecord(tool_call_parse_failures: int, tool_calls: int, refused: bool)`
  - `@dataclass(frozen=True) Finding(drift_type: str, kind: Literal["off_task","degraded"], side_effects: bool, evidence: dict)`
  - `THRESHOLDS: dict[Sensitivity, dict]`, with exact values:

    | sensitivity | loop_window | loop_repeats | near_dup_jaccard | parse_fail_rate | refusals_per_run | min_calls |
    |---|---|---|---|---|---|---|
    | low | 6 | 4 | 0.95 | 0.20 | 3 | 5 |
    | normal | 6 | 3 | 0.90 | 0.10 | 2 | 5 |
    | high | 8 | 2 | 0.85 | 0.05 | 1 | 5 |

  - `REFUSAL_RE = re.compile(r"^\s*(I'm sorry|I am sorry|I can(?:not|'t) (?:help|assist|comply)|As an AI\b)", re.I)`. It counts only when the content is under 400 chars and there are no tool calls.
  - `check_tool_call(history: list[ToolCallRecord], call: ToolCallRecord, spec: WorkflowSpec) -> list[Finding]`:
    - **Out of scope.** If `spec.tools` is set and `call.name` is not in it, return `out_of_scope_action` (off_task, with `side_effects` from the call).
    - **Loop.** Count the calls among the last `loop_window` of `history` (plus `call`) that have the same `name` and either an identical `args` or a jaccard ≥ `near_dup_jaccard` on args tokens (reuse `pitstop.supervisor.signals.jaccard`). If the count is ≥ `loop_repeats`, return `unproductive_loop` (off_task).
  - `check_llm_call(history: list[LLMCallRecord], call: LLMCallRecord, sensitivity) -> list[Finding]`:
    - **Breakage.** Over the last 20 calls including this one with `tool_calls > 0`, if there are ≥ `min_calls` of them and failures/calls ≥ `parse_fail_rate`, return `tool_call_breakage` (degraded).
    - **Refusals.** If the run's refusals so far, including this call, are ≥ `refusals_per_run`, return `refusal_spike` (degraded).
- Produces in `pitstop.detect.urgency`:
  - `class Level(str, Enum)`: `P0`, `P1`, `P2`, `P3`
  - `classify(drift_type: str, criticality: Criticality, side_effects: bool) -> Level`, with this exact table:

    | drift_type | side_effects | high | medium | low |
    |---|---|---|---|---|
    | budget_exhausted | any | P0 | P0 | P0 |
    | out_of_scope_action | true | P0 | P0 | P0 |
    | out_of_scope_action | false | P1 | P2 | P3 |
    | unproductive_loop | true | P0 | P0 | P0 |
    | unproductive_loop | false | P0 | P1 | P2 |
    | tool_call_breakage | any | P1 | P1 | P2 |
    | refusal_spike | any | P1 | P1 | P2 |
    | any other | any | P2 | P2 | P3 |

    (Ruling: a read-only loop on a `high` workflow is P0 because a loop is spend without progress. The spec's P0 row lists "loop on a high-criticality workflow".)
  - `action_for(finding: Finding, level: Level, criticality: Criticality) -> Literal["block","steer","halt","record"]`, applied in this order:
    1. `budget_exhausted` → `halt`
    2. off-task with `side_effects` → `block`
    3. off-task without side effects → `steer`
    4. degraded → `record`

    Separately, any P0 on a `high` workflow also sets a pending halt (Task 6 and wrap apply it at the next boundary).
- Produces in `pitstop.detect.drifts`:
  - `record_drift(store, *, run_id, workflow, finding, level, action) -> int` inserts a drift with `status='open'`, unless an open drift with the same `(workflow, drift_type)` exists. In that case it appends the new evidence to that row's `evidence` list and returns the existing id. A workflow of `None` dedups on `(NULL, drift_type)`.
  - `list_drifts(store, *, workflow=None, open_only=True) -> list[dict]`
- Wrap integration:
  - `_after_llm_call` builds an `LLMCallRecord`. A tool call's `arguments` that fail `json.loads` count as a parse failure.
  - It appends the record to `ctx.llm_history`, runs `check_llm_call`, records each finding with its level and action, and, for a P0 on a high workflow, sets `ctx.halt_reason` and `ctx.halt_drift_id`.
  - Outside a workflow, findings are recorded with `workflow=None` and `criticality="medium"`.
  - `BudgetExceeded` records a `budget_exhausted` drift before raising.

- [ ] **Step 1: Write the failing tests.** Table-driven tests:
  - `test_classify_table` covers every row above.
  - `test_action_for_order`
  - `test_loop_detected_at_threshold_per_sensitivity` (the normal sensitivity flags the 3rd identical call; low does not)
  - `test_near_duplicate_args_count_as_loop`
  - `test_out_of_scope_tool`
  - `test_parse_failure_rate_needs_min_calls`
  - `test_refusal_regex_and_limit`
  - `test_record_drift_dedups_open_drift`
  - `test_wrap_records_breakage_drift_from_bad_tool_args`, using the fakes with tool-call `arguments='{"q": '` five times
  - `test_budget_exhausted_records_p0_drift`
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/detect tests/sdk/test_wrap_detect.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/detect tests/sdk -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(detect): L0 checks, P0-P3 urgency table and drift records"`

---

### Task 6: `@pitstop.tool`: block, steer and halt

**Files:**
- Create: `src/pitstop/sdk/tool.py`
- Modify: `src/pitstop/__init__.py` (export `tool`)
- Test: `tests/sdk/test_tool.py`

**Interfaces:**
- Consumes: `current_run`, `check_halt` (Task 3), and `check_tool_call`, `classify`, `action_for`, `record_drift` (Task 5).
- Produces `tool(fn=None, *, side_effects: bool = False, name: str | None = None)`. It works as `@tool` and as `@tool(side_effects=True)`, on sync and async functions. On each invocation:
  1. Run `check_halt()`.
  2. Build `ToolCallRecord(name or fn.__name__, canonical_json({"args": args, "kwargs": kwargs}, default=repr), side_effects)`.
  3. Run `check_tool_call(ctx.tool_history, rec, ctx.spec)`. For each finding, classify it, record the drift, and decide the action. The strictest action wins: `block` beats `steer` beats `record`.
  4. Apply the action:
     - **block:** do not call `fn`. Record an `events` row `kind='block'` and return `{"blocked_by": "pitstop", "reason": "<drift_type>: <one-line evidence>"}` exactly.
     - **steer:** if `ctx.steers < 2`, do not call `fn`. Increment `steers`, record `kind='steer'`, and return the string `f"PitStop: this step drifts from your goal ({drift_type}: {evidence}). Return to: {ctx.agent_goal or ctx.spec.goal or 'your original task'}."`. Otherwise set the pending halt with reason `"steer limit exceeded"` and raise `Halted` immediately.
     - **record or none:** call `fn` and record `kind='tool_call'` (args summary truncated to 500 chars), then `kind='tool_result'`.
  5. If the drift's level is P0 and criticality is `high`, set the pending halt so the next boundary raises. The current call still returns its block result.
  6. Append `rec` to `ctx.tool_history` in every case.
- Outside a workflow, `@tool` just calls `fn`. Scope and loop checks need a run.
- Exceptions from `fn` propagate unchanged. Record `kind='tool_result'` with `payload.error`.

- [ ] **Step 1: Write the failing tests.**
  - `test_tool_runs_normally_and_records_events`
  - `test_out_of_scope_side_effect_tool_is_blocked` checks the returned dict exactly, that `fn` was not called, that a P0 drift exists, and the action is `block`.
  - `test_high_criticality_p0_halts_at_next_boundary` checks that after the block, the next wrapped LLM call raises `Halted`.
  - `test_read_only_loop_is_steered_twice_then_halts` (normal sensitivity): calls 1–2 run, call 3 steers, call 4 steers, and call 5 raises `Halted("steer limit exceeded")`.
  - `test_steer_message_uses_agent_goal_then_workflow_goal`
  - `test_async_tool`
  - `test_tool_outside_workflow_just_runs`
  - `test_tool_exception_propagates`
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/sdk/test_tool.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/sdk tests/detect -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(sdk): tool() guard with block, steer and halt actions"`

---

### Task 7: Managed prompts

**Files:**
- Create: `src/pitstop/sdk/prompt.py`
- Modify: `src/pitstop/__init__.py` (export `prompt`), `src/pitstop/sdk/wrap.py` (on the first call in a run, write `runs.prompt_versions` as `{name: version}` for prompts read in that run)
- Test: `tests/sdk/test_prompt.py`

**Interfaces:**
- Produces `prompt(name: str, default: str) -> ManagedPrompt`, where `ManagedPrompt` is a frozen dataclass with `name`, `text`, `version: int`, `source: Literal["default","dev_edit","gepa"]` and `sha` (sha256 hex of the text).
  - On first sight of `name`, insert version 1, `source='default'`, `active=1`.
  - If `default`'s sha differs from every stored version's sha, the developer edited the code default. Insert a new version with `source='dev_edit'`, make it active and deactivate the others. This is spec §7.1: the developer's edit wins.
  - If `default` matches an existing non-active version and a `gepa` version is active, keep serving the active version. The developer didn't change anything.
  - Return the active version.
  - Reading inside a run adds `{name: version}` to a per-run dict on `RunContext` (add the field `prompt_versions: dict[str, int]`).
  - Store failure: log it and return a `ManagedPrompt` built from `default` with `version=0`.
- `set_active_version(name, version)` is internal and is used by tests now and by milestone 4 promotion and rollback.

- [ ] **Step 1: Write the failing tests.**
  - `test_first_use_creates_v1_default`
  - `test_same_default_is_stable`
  - `test_edited_default_becomes_dev_edit_and_wins` covers this sequence: default A gives v1; set a gepa v2 active; then default B gives v3 `dev_edit` as the active version.
  - `test_unchanged_default_keeps_active_gepa_version`
  - `test_prompt_versions_recorded_on_run`
  - `test_store_failure_returns_default_v0`
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/sdk/test_prompt.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run: `uv run pytest tests/sdk -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(sdk): managed prompts with versioning; developer edits win"`

---

### Task 8: PitStop's own calls go through the ledger, plus the $50 Crusoe budget

**Files:**
- Modify:
  - `src/pitstop/llm.py:128` so `self._client = wrap(AsyncOpenAI(...))`
  - `src/pitstop/correct/client.py:173-178` so both the sync client and the async factory's product are passed through `wrap()`
  - `benchmarks/synth/synth/crusoe.py:31` so the generator's `AsyncOpenAI` is wrapped too
- Create: `docs/config.example.toml` (the config from Task 1 with the `crusoe` $50 `total` budget and a comment on where to copy it: `~/.pitstop/config.toml`)
- Test: `tests/budget/test_internal_clients_metered.py`

**Interfaces:**
- Consumes: `wrap` (Task 4).
- Internal calls made outside a user workflow get `workflow=None`, so only global budgets apply. Calls made through `pitstop.correct` set the meter to `correction` via a context var `pitstop.sdk.wrap.meter(name)` context manager, which this task adds. Calls through `pitstop.llm` inside `Supervisor` use the `watch` meter. Wrap the supervisor's `complete` call site in `with meter("watch")`.

- [ ] **Step 1: Write the failing tests.**
  - `test_runtime_llm_client_is_metered(tmp_path)`: construct `pitstop.llm.LLMClient` with an injected fake async client (add an optional `client=` parameter to `LLMClient.__init__` for injection; production behavior is unchanged), make one `complete`, and check that the ledger has one settled row.
  - `test_correct_client_hits_global_budget(tmp_path)`: a $0.000001 global budget makes `CrusoeClient(sync_client=FakeOpenAI(...)).complete(...)` raise `BudgetExceeded`.
  - `test_supervisor_calls_use_watch_meter`
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/budget/test_internal_clients_metered.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Check that `CrusoeClient`'s retry loop does not swallow `BudgetExceeded`: it must propagate immediately and never be retried. Add it to `_is_retryable`'s non-retryable path.
- [ ] **Step 4: Run the full suite.** Run: `uv run pytest -q`. Expected: everything passes, with the existing runtime, correct and benchmark tests unchanged.
- [ ] **Step 5: Commit.** `git commit -m "feat(budget): route PitStop's own model calls through the ledger; example \$50 Crusoe budget"`

---

### Task 9: CLI, quickstart example and docs

**Files:**
- Modify: `src/pitstop/cli.py` (add `status`, `drifts`, `budget`; these handlers are sync, so the dispatch must allow both sync and async handlers)
- Create: `src/pitstop/cli_report.py` (pure functions returning rows; the CLI renders them with rich), `examples/quickstart/agent.py`, `examples/quickstart/README.md`
- Modify: `README.md` (Quickstart gains "Add PitStop to your agent", Roadmap marks M2a done), `CHANGELOG.md`, `CONTRIBUTING.md` (repo map adds `examples/`)
- Test: `tests/test_cli_report.py`, `tests/test_quickstart_example.py`

**Interfaces:**
- Consumes everything above.
- Produces in `pitstop.cli_report`:
  - `workflow_rows(store, now) -> list[dict]` with keys `workflow`, `criticality` (latest run's value), `runs_today`, `open_p0`, `open_p1`, `open_p2`, `open_p3`, `spend_today_usd`, `budget_usd`, `last_status`
  - `budget_rows(ledger, config) -> list[dict]` with keys `scope`, `period`, `spent_usd`, `limit_usd`, `remaining_usd`
  - `drift_rows(store, workflow=None, open_only=True) -> list[dict]` with keys `id`, `level`, `workflow`, `drift_type`, `action`, `count` (evidence entries), `first_seen`, `last_seen`
- CLI commands:
  - `pitstop status` prints a workflow table and a budget table.
  - `pitstop drifts [--workflow W] [--all]` lists drifts, sorted by level then last_seen descending.
  - `pitstop budget` prints the budget table only.
  - All three exit 0 with a friendly "No runs yet. See examples/quickstart." when the DB is empty.
- `examples/quickstart/agent.py` is an OpenAI-compatible tool-using agent of about 60 lines:
  - It has one read-only tool `search(query)` (a stub that returns canned text) and one `side_effects=True` tool `send_report(to, body)` (it prints).
  - It is wrapped in `pitstop.workflow("quickstart", goal=..., criticality="medium", budget_usd=1.0, tools=("search","send_report"))`. It uses medium rather than high so the demo shows two steers before the halt; on a high workflow a read-only loop is P0 and halts at the next boundary..
  - `--inject-loop` makes the system prompt tell the model to keep re-running the same search. PitStop should steer and then halt.
  - It reads `CRUSOE_API_KEY` and the model from env, and exits with a clear message if the key is missing.
  - `--offline` swaps in `tests/sdk/fakes.py`-style scripted responses that are inline in the example, so the README demo runs without keys.

- [ ] **Step 1: Write the failing tests.**
  - `test_workflow_rows_counts_open_drifts_by_level(tmp_path)`
  - `test_budget_rows_show_remaining(tmp_path)`
  - `test_cli_status_empty_db_is_friendly(tmp_path)`, a subprocess with `PITSTOP_HOME=tmp_path` that checks exit code 0 and the text "No runs yet".
  - `test_quickstart_offline_inject_loop_steers_then_halts(tmp_path)`: run `examples/quickstart/agent.py --offline --inject-loop` as a subprocess with `PITSTOP_HOME=tmp_path`. Check exit code 0 (the example catches `Halted` and prints it). Then `pitstop drifts --all` shows an `unproductive_loop` at `P1` (a read-only loop on a medium workflow, per the Task 5 table) with action `steer`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run: `uv run pytest tests/test_cli_report.py tests/test_quickstart_example.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the report functions, the CLI subcommands and the example.
  - **README.** Add a short "Add PitStop to your agent" section with the real few-lines snippet (`wrap`, `workflow`, `tool`, `prompt`) and the `pitstop status` / `pitstop drifts` output. Add a "Set a hard budget" section that points to `docs/config.example.toml`.
  - **Roadmap.** M2a is done; M2b is next.
  - **CHANGELOG.** Add an `[Unreleased]` entry.
- [ ] **Step 4: Run every check.** Run: `uv run ruff check && uv run pytest -q`. Expected: all pass. Run `uv run python examples/quickstart/agent.py --offline --inject-loop` by hand. Expected: it prints two steers and then `Halted: steer limit exceeded`.
- [ ] **Step 5: Commit.** `git commit -m "feat(cli): status, drifts and budget commands; quickstart example and docs"`

---

## Carried to milestone 2b (recorded at the end of execution)

These came out of the task and whole-branch reviews. The M2b plan must cover them.

- **Config validation gaps.** A config with any of these problems still runs unmetered (fail-open):
  - a non-string `budgets.*.base_url`;
  - negative or NaN price overrides;
  - an unreadable `config.toml` (the OSError is not wrapped in `ConfigInvalid`).

  Validate all three in `userconfig.py`. Do this first.
- **`wrap` metering holes:**
  - `with_raw_response` / `with_streaming_response`;
  - `client.responses.create`;
  - legacy `completions`;
  - `chat.completions.parse()` and `.stream()`.
- **GEPA.** `raise_on_exception=False` swallows `Halted`. Add an abort path so a halted optimization does not report "optimized" results.
- **`stream_options` injection** may get a 400 from some OpenAI-compatible backends. Retry once without it.
- **Crashed processes leave `reserved` ledger rows** that count forever toward `total` budgets. Add an explicit `pitstop budget reconcile` command, not auto-expiry.
- **Drift evidence lists are unbounded,** with an O(n²) rewrite. Cap them and add an occurrences count, plus run_id and ts per entry.
- **`pitstop run` with an invalid config** ends in a Halted traceback. Give it a clean exit 2 like status/drifts/budget.
- **Minor, from the reviews:**
  - store: schema version is not checked on open, and nested `transaction()` is unsupported;
  - `workflow()` on generator functions;
  - `isinstance(openai.Stream)` is False for wrapped streams;
  - `runs` and the ledger normalize `base_url` differently;
  - `llm_history` and `tool_history` are unbounded;
  - `steers += 1` is not atomic across threads;
  - a P0 pending halt is not recorded as a `halt` event;
  - drifts `first_seen == last_seen`.
