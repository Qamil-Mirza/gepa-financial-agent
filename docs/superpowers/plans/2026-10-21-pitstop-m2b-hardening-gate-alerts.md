# PitStop Milestone 2b: Hardening, Full Client Coverage, L1 Gate, Fallback and Alerts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap carried over from milestone 2a. Then:
- `wrap` meters every model-call path of the OpenAI and Anthropic SDKs.
- Side-effecting tools on high and medium workflows pass an LLM pre-action gate.
- A broken model can fall back to a pinned one.
- Urgent drifts reach the developer by page, Slack, email or webhook.

**Architecture:**
- **`wrap`.** It becomes a small package. A generic resource proxy routes intercepted methods to a per-API `Shape`, which knows how to:
  - estimate the prompt;
  - read usage, text and tool calls from a response or a stream;
  - hide any chunk PitStop injected.

  One `_Call` class keeps the reserve → settle → record → detect cycle for every shape.
- **New inline parts in the user's process:**
  - the L1 gate (`detect/l1.py`, called from `@tool`);
  - fallback, a sticky per-workflow switch stored in `meta`;
  - alert dispatch: a small thread pool that never blocks the agent, so alerts still go out when the process exits.
- **No server yet.** Everything runs in the SDK process or from the CLI. The milestone 3 server will call the same functions on a schedule.

**Tech Stack:** Python ≥3.11, openai 3.x (installed: 3.33.0), anthropic 2.x (new, optional), httpx (an existing dependency; `httpx.MockTransport` in tests), stdlib `smtplib`/`email`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-pitstop-runtime-design.md`. This plan covers:
- §3 client coverage (Anthropic, Responses API and helpers);
- §3.1 Fallback;
- §6.2 L1;
- §8 Alerts;
- §14.1 gate failure policy;
- the "Carried to milestone 2b" list at the end of `docs/superpowers/plans/2026-10-20-pitstop-m2a-sdk-core.md`.

**Milestone split (ruling, recorded for the owner).** What the 2a plan called "2b" is split in two again.
- **This plan (2b)** is hardening plus the inline features.
- **2c (next plan)** covers the observe-and-background side:
  - the OTel span processor, plus verdicts emitted as span events;
  - the OpenAI Agents SDK and LangGraph adapters;
  - L2 change-point stats;
  - L5 attribution;
  - a background `scan` pass;
  - moving the research swarm to `examples/financial-research/`, where it is wired in with a few lines.

  The swarm move touches `benchmarks/` and most of `src/pitstop/`, so it gets its own review surface.

## Decisions this plan makes (design calls; the owner can revisit)

1. **P1 alerts in the SDK.** The SDK sends P0 **and** P1 alerts directly. The spec puts P1 routing in the server, but there is no server until milestone 3. Without this, P1 alerts would wait for a manual `pitstop alerts flush`. P2/P3 go out as a digest from `pitstop alerts digest`, which the milestone 3 server will run daily.
2. **Secrets in alert config.** Alert secrets go only in environment variables. `config.toml` names the variable (`routing_key_env = "PAGERDUTY_ROUTING_KEY"`). A secret written inline (`routing_key = "..."`) is rejected with `ConfigInvalid`, so `config.toml` stays safe to share or commit.
3. **What fallback swaps.** Fallback swaps the **model id on the same client**. It does not switch providers. It turns on when all of these hold:
   - a P1 `tool_call_breakage` or `refusal_spike` is recorded;
   - the run's model or base_url differs from the workflow's previous run;
   - the workflow declares `fallback_model`.

   It stays on until `pitstop fallback clear <workflow>`. L4 canaries (milestone 3) will later catch silent provider swaps too.
4. **The judge.** The L1 judge is any OpenAI-compatible endpoint from `[judge]` in config.toml:
   - defaults: Crusoe base URL, `api_key_env = "CRUSOE_API_KEY"`, model `openai/gpt-oss-120b`;
   - or a client passed to `pitstop.init(judge_client=...)`.

   Its calls use the `watch` meter. One vote per gated call (latency matters inline).
5. **Helpers that can't be read.** Raw-response helpers that can't be read for usage are charged their worst-case reservation (over-count, never under-count):
   - `with_streaming_response`;
   - `with_raw_response` with `stream=True`.
6. **Unknown Claude prices.** Unknown `claude-*` models get the conservative price `Price(15, 75, 15)`, not the generic `Price(5, 15, 5)`. That stops an un-priced Opus-class model from under-counting. No built-in Anthropic prices are shipped; users add `[prices."<model>"]` overrides.
7. **Store too new.** A `pitstop.db` written by a newer PitStop fails closed (as `ConfigInvalid`), like an invalid config. A budget must never silently vanish.
8. **Generator workflows.** `@pitstop.workflow` on generator functions runs each step inside a private `contextvars.Context`, so the run never leaks into the consumer's code between yields.

## Global Constraints

- **Python ≥3.11.** No new *required* runtime dependency.
  - `anthropic>=2.0` becomes an optional extra (`pitstop[anthropic]`) and a dev dependency.
  - httpx is already required.
- **No network in tests.** pytest-socket stays on. Use these instead:
  - HTTP fakes: real SDK clients with `httpx.MockTransport`, plus the existing `tests/sdk/fakes.py`;
  - a loopback server, only where a subprocess must make the request.
- **Never a real key, never real spend.** Subprocess tests run with `cwd=tmp_path`, and every key env var is scrubbed: `CRUSOE_API_KEY`, `YDC_API_KEY`, plus the alert vars added in Task 10.
- **Never delete untracked or git-ignored files.** Never run `examples/quickstart/agent.py` without `--offline`.
- **Errors the SDK may raise.** It raises only `pitstop.Halted` / `pitstop.BudgetExceeded` on purpose:
  - PitStop's own failures are logged and never surface in user code (fail open);
  - an invalid config or a too-new store fails closed with `Halted("pitstop config invalid: ...")`.
- **Secrets stay out.** A secret value (API key, routing key, webhook URL, SMTP password) never reaches the DB, events, alert rows or log messages.
- **Stream boundaries.** Inline actions take effect at the next call boundary. A stream is never cut mid-flight.
- **Commits.** No attribution trailers (no `Co-Authored-By`, no "Generated with"). Use conventional prefixes (`feat(sdk):`, `fix(store):`, `docs:`).
- **Style.** ruff `F` clean and line length 120. Match the surrounding module docstring and comment density.
- **Docs ship with the feature.** Each task that adds a user-facing feature updates its README section, `docs/config.example.toml` and `CHANGELOG.md` (Unreleased) in the same task.

## Review Focus

1. **A slow or down alert channel never slows or breaks the agent.** Recording a P0 returns while the webhook is still hanging, and the failure is recorded. Pinned in Task 10: `test_hanging_channel_does_not_block_agent`, `test_failed_send_recorded`.
2. **Secrets never leak.** A failing Slack send whose URL contains the secret leaves no trace of it in `alerts` rows, events or captured logs. Pinned in Task 10: `test_secret_never_stored_or_logged`.
3. **The process exits right after a P0 halt.** The alert is still delivered before the interpreter exits. Pinned in Task 10 with a subprocess and a loopback webhook: `test_p0_alert_delivered_when_process_exits`.
4. **Helper streams behave exactly like the unwrapped SDK.** This covers `with client.chat.completions.stream(...) as s` and `client.messages.stream(...)`: events, `get_final_completion()` / `get_final_message()`, and `isinstance` checks. They are still metered. Pinned in Task 5 (`test_chat_stream_helper_matches_unwrapped`) and Task 6 (`test_messages_stream_helper_matches_unwrapped`).
5. **A high-criticality workflow with no judge configured.** Side-effect calls are blocked with a reason that says how to fix it, and a `gate_error` event is recorded. Medium workflows go ahead. Pinned in Task 8: `test_high_without_judge_blocks_with_fix_hint`, `test_medium_without_judge_allows_and_records`.

---

### Task 1: Config validation gaps and `pitstop run` exit code

**Files:**
- Modify: `src/pitstop/userconfig.py`, `src/pitstop/cli.py` (`cmd_run`)
- Test: `tests/store/test_userconfig.py`, `tests/test_cli_report.py` (or a new `tests/test_cli_run_config.py`)

**Interfaces:**
- Produces: `load_user_config` raises `ConfigInvalid` for all three of these:
  - a non-string `budgets.*.base_url`;
  - a negative, NaN or infinite price (`input`, `output` or `cached_input`) or a non-number price;
  - an unreadable file (`OSError`: a directory, permission denied, invalid UTF-8).
- Produces: a private helper `_finite_nonneg(value, what: str) -> float` in `userconfig.py`. Later tasks reuse it to validate their own numeric fields.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("body", [
    '[budgets.x]\nlimit_usd = 1\nbase_url = 5',
    '[prices."m"]\ninput = -1\noutput = 1',
    '[prices."m"]\ninput = nan\noutput = 1',
    '[prices."m"]\ninput = 1\noutput = inf',
    '[prices."m"]\ninput = "cheap"\noutput = 1',
    '[prices."m"]\ninput = 1\noutput = 1\ncached_input = -0.5',
])
def test_bad_values_raise_config_invalid(tmp_path, body):
    p = tmp_path / "config.toml"; p.write_text(body)
    with pytest.raises(ConfigInvalid):
        load_user_config(p)

def test_unreadable_config_raises_config_invalid(tmp_path):
    p = tmp_path / "config.toml"; p.mkdir()          # reading a directory raises IsADirectoryError (OSError)
    with pytest.raises(ConfigInvalid, match="cannot read"):
        load_user_config(p)

def test_non_utf8_config_raises_config_invalid(tmp_path):
    p = tmp_path / "config.toml"; p.write_bytes(b"\xff\xfe[budgets]")
    with pytest.raises(ConfigInvalid):
        load_user_config(p)

def test_pitstop_run_with_invalid_config_exits_2(tmp_path, monkeypatch):
    # subprocess: cwd=tmp_path, PITSTOP_HOME=tmp_path/home with a broken config.toml, keys scrubbed
    r = subprocess.run([sys.executable, "-m", "pitstop.cli", "run", "--mode", "replay"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "Invalid PitStop config" in r.stdout and "Traceback" not in r.stderr
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/store/test_userconfig.py tests/test_cli_run_config.py -v`. Expected: the new cases FAIL.
- [ ] **Step 3: Implement.**
  - In `load_user_config`, catch `OSError` and `UnicodeDecodeError` from `read_text()` → `ConfigInvalid(f"{path}: cannot read config: {type(e).__name__}")`.
  - In `_parse`:
    - require `base_url` to be `str | None`;
    - build each `PriceOverride` through `_finite_nonneg` (rejects bool, NaN, ±inf and negative values).
  - In `cmd_run`, call `_runtime()` first. It already exits 2 with "Invalid PitStop config".
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest tests/store tests/test_cli_run_config.py -v`, then the full suite with `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(config): reject non-string base_url, bad prices and unreadable config; pitstop run exits 2 on invalid config"`.

---

### Task 2: Store v3 (stepwise migrations, version check, nested transactions), drift hygiene, budget reconcile

**Files:**
- Modify: `src/pitstop/store/__init__.py`, `src/pitstop/store/schema.sql`, `src/pitstop/detect/drifts.py`, `src/pitstop/budget/ledger.py`, `src/pitstop/cli_report.py`, `src/pitstop/cli.py`, `src/pitstop/sdk/runtime.py`, `src/pitstop/sdk/wrap.py` (`_record` only: normalized base_url)
- Create: `src/pitstop/store/urls.py`
- Test: `tests/store/test_store.py`, `tests/detect/test_drifts.py`, `tests/budget/test_ledger.py`, `tests/test_cli_report.py`

**Interfaces:**
- Produces:
  - `pitstop.store.SCHEMA_VERSION = 3`.
  - `pitstop.store.MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]]`, keyed by target version. `_migrate` applies every step above the DB's version in one `BEGIN IMMEDIATE`. Later tasks add a step by appending an entry, bumping `SCHEMA_VERSION` and updating `schema.sql`.
  - `class StoreVersionError(RuntimeError)`, raised by `Store(...)` when the DB's schema_version > `SCHEMA_VERSION`. `runtime.init` re-raises it as `ConfigInvalid` (decision 7).
  - `Store.transaction()` nests. An inner call uses `SAVEPOINT pitstop_<n>` / `RELEASE` / `ROLLBACK TO`. An inner `immediate=True` inside a deferred outer transaction is allowed: it does not upgrade the lock, which the docstring says.
  - `pitstop.store.urls.normalize_base_url(url: str | None) -> str | None`, which strips trailing `/`. The ledger and the `runs` table both use it.
  - v3 schema: `drifts.last_seen REAL`, `drifts.occurrences INTEGER NOT NULL DEFAULT 1`. The migration backfills `last_seen = ts`.
  - `detect.drifts.MAX_EVIDENCE = 20`. Each evidence entry is `{"run_id": str | None, "ts": float, "detail": dict}`. On dedup: `occurrences += 1`, `last_seen = now`, and the entry is appended with only the newest 20 kept.
  - `Ledger.stale_reservations(older_than_s: float) -> list[dict]` (id, ts, run_id, workflow, meter, model, usd) and `Ledger.release_stale(older_than_s: float) -> int`.
  - CLI: `pitstop budget reconcile [--older-than-min 60] [--release]`.
    - Without `--release`, it lists the stale rows and says "re-run with --release to free them".
    - With it, it releases them and prints the count.
- `cli_report.drift_rows` now reports `count = occurrences`, `first_seen = ts` and `last_seen = last_seen`.

- [ ] **Step 1: Write the failing tests.** Key assertions:

```python
def test_fresh_db_matches_migrated_db(tmp_path):            # schema.sql and MIGRATIONS agree
    old = _make_v1_db(tmp_path / "old.db")                 # helper builds the v1 runs/drifts tables + schema_version 1
    assert _columns(Store(old)) == _columns(Store(tmp_path / "new.db"))
    assert Store(old).schema_version() == SCHEMA_VERSION == 3

def test_newer_db_raises_store_version_error(tmp_path): ...  # meta schema_version = 99 -> StoreVersionError
def test_runtime_fails_closed_on_newer_db(tmp_path):         # wrapped call -> Halted("pitstop config invalid: ...")
def test_nested_transaction_inner_rollback_keeps_outer(tmp_path):
    with s.transaction() as c:
        c.execute("INSERT INTO meta VALUES ('a','1')")
        with pytest.raises(RuntimeError):
            with s.transaction() as c2:
                c2.execute("INSERT INTO meta VALUES ('b','1')"); raise RuntimeError
    assert _meta(s, "a") == "1" and _meta(s, "b") is None
def test_dedup_caps_evidence_and_counts_occurrences(store):  # 25 records -> occurrences 25, 20 entries, newest kept,
                                                             # entries carry run_id/ts/detail, last_seen > ts
def test_runs_and_ledger_store_same_base_url(tmp_path):      # wrapped call with base_url ".../v1/" -> both "…/v1"
def test_stale_reservations_listed_and_released(tmp_path):   # old reserved row listed; release -> state released,
                                                             # spent() drops; fresh reserved + settled rows untouched
def test_budget_reconcile_cli_dry_run_then_release(tmp_path, capsys): ...
def test_drift_rows_use_occurrences_and_last_seen(store): ...
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/store tests/detect/test_drifts.py tests/budget tests/test_cli_report.py -v`. Expected: the new cases FAIL.
- [ ] **Step 3: Implement as the Interfaces block describes.**
  - Turn the existing v2 `budget_usd` step into `MIGRATIONS[2]`.
  - Make the `Store` constructor check the version before running migrations.
  - Replace the ledger's private `_norm` with `normalize_base_url`.
  - Make `_Call._record` write `normalize_base_url(self.base_url)`.
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "feat(store): schema v3 with stepwise migrations, version check and nested transactions; cap drift evidence; budget reconcile"`.

---

### Task 3: SDK runtime hygiene

**Files:**
- Create: `src/pitstop/sdk/actions.py`
- Modify:
  - `src/pitstop/sdk/wrap.py`: move `_apply_findings` out to `actions.py`; meter-aware close;
  - `src/pitstop/sdk/tool.py`, `src/pitstop/sdk/context.py`, `src/pitstop/detect/l0.py`;
  - `tests/sdk/test_wrap_detect.py`, which uses `wrapmod._apply_findings`.
- Test: `tests/sdk/test_context.py`, `tests/sdk/test_tool.py`, `tests/sdk/test_wrap_detect.py`, `tests/detect/test_l0.py`

**Interfaces:**
- Produces:
  - `pitstop.sdk.actions.Applied` is a `NamedTuple(finding, level, action, drift_id)`.
  - `apply_findings(ctx, findings, *, run_id=None) -> list[Applied]` has the behavior of today's `_apply_findings`, plus one change. When it sets a pending P0 halt, it inserts a `halt` event: `name = drift_type`, `payload = {"reason", "drift_id", "pending": true}`.
  - The `tool.py` steer decision reads `best.action`.
- **History caps.** `RunContext.tool_history` values and `RunContext.llm_history` become `collections.deque(maxlen=HISTORY_MAX)`, with `HISTORY_MAX = 64` in `context.py`.
- **Refusal counter.** New `RunContext.refusals: int` counts every refusal in the run, because the history is now capped.
- **L0 signature change.** `check_llm_call(history, call, sensitivity, refusals: int)`, where `refusals` is the run total including `call`. The refusal-spike rule uses it instead of scanning the history.
- **Steer limit under threads.** New `RunContext.lock: threading.Lock` (`field(default_factory=threading.Lock, repr=False, compare=False)`). The steer check-and-increment in `tool.py` runs under it.
- **Only agent calls run detection.** `_Call` stores `self.meter = _meter.get()` at begin. Only when `meter == "agent"`:
  - `close` runs `_after_llm_call`;
  - `_record` sets the run's first-call tags (`model`, `base_url`, `adapter`, `price_known`).

  Every `llm_call` event payload gains `"meter"`. Judge, supervisor and GEPA calls inside a run no longer pollute L0 or the run's model tag. Task 8 and Task 9 rely on this.
- **Generator workflows.** `workflow()` decorating a generator or async-generator function opens the run on the first step and closes it when the generator finishes, raises or is closed. Each step runs inside the run's private Context (decision 8):

```python
# sync: one private Context for the whole run; every step of the generator runs inside it.
run_cv = contextvars.copy_context()
scope = self._fresh(); run_cv.run(scope.__enter__)
gen = run_cv.run(fn, *a, **k)
# loop: value = run_cv.run(gen.send, sent) / run_cv.run(gen.throw, exc); on StopIteration / exception / close
#       -> run_cv.run(scope.__exit__, et, ev, tb) exactly once
# async: same, but each step is awaited as asyncio.create_task(agen.asend(v), context=run_cv)
```

- [ ] **Step 1: Write the failing tests.**

```python
def test_pending_p0_halt_records_halt_event(tmp_path): ...  # high wf, side-effect out-of-scope -> 'halt' event, pending
def test_histories_are_bounded(tmp_path): ...               # 200 tool calls -> len(tool_history[agent]) == 64
def test_refusal_spike_counts_whole_run_past_cap(tmp_path): # 64 normal + refusals still triggers on run total
def test_steer_limit_is_atomic_across_threads(tmp_path):    # 8 threads x looping read-only tool -> steers == 2, then Halted
def test_judge_meter_calls_skip_detection_and_run_tags(tmp_path):
    # inside a workflow: a 'watch'-meter call returning refusals -> no refusal drift; runs.model stays the agent call's
def test_llm_event_payload_has_meter(tmp_path): ...
def test_workflow_on_generator_records_run_and_does_not_leak(tmp_path):
    @pitstop.workflow("g")
    def gen():
        yield current_run().run_id
        yield current_run().run_id
    it = gen(); a = next(it)
    assert current_run() is None                            # consumer code between yields sees no run
    b = next(it); assert a == b; list(it)
    assert _run_status(a) == "ok"
async def test_workflow_on_async_generator(tmp_path): ...   # same, plus an exception inside -> status 'error'
def test_generator_closed_early_finishes_run(tmp_path): ... # it.close() -> status 'ok', ended_at set
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/sdk tests/detect -v`. Expected: the new cases FAIL.
- [ ] **Step 3: Implement as the Interfaces block describes.** Update the existing L0 tests to the new `check_llm_call` signature.
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(sdk): bounded histories, atomic steer limit, pending P0 halt event, detection only on agent-meter calls, generator workflows"`.

---

### Task 4: `wrap` engine refactor (Shapes) and chat helpers: `parse`, raw responses, `stream_options` retry

**Files:**
- Delete `src/pitstop/sdk/wrap.py` and create the package `src/pitstop/sdk/wrap/`:
  - `__init__.py`: `wrap`, `meter`, re-exports;
  - `proxy.py`: client/resource proxies, stream proxies;
  - `call.py`: `_Call`;
  - `shapes.py`: `Observed`, `Shape`, `CHAT`, the OpenAI method table.
- Modify: `src/pitstop/sdk/tool.py` imports, if needed.
- Create: `tests/sdk/http_fakes.py`
- Test: `tests/sdk/test_wrap.py`, `tests/sdk/test_wrap_detect.py` (both must pass unchanged, except for imports of moved private names), new `tests/sdk/test_wrap_helpers.py`

**Interfaces:**
- Consumes: `_Call.meter` (Task 3), `normalize_base_url` (Task 2), `actions.apply_findings` (Task 3).
- Produces:

```python
@dataclass
class Observed:                       # everything one response or stream showed
    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)   # [{"name", "arguments": raw str}]
    finish_reason: str | None = None
    refused: bool = False             # provider-declared refusal (e.g. Anthropic stop_reason "refusal")
    prompt_tokens: int | None = None  # None = usage not seen -> settle by estimate
    completion_tokens: int | None = None
    cached_tokens: int = 0
    chars: int = 0                    # all completion text, for the estimate

class Shape(Protocol):
    name: str                                                    # "chat" | "responses" | "completions" | "anthropic"
    def prepare(self, kwargs: dict) -> tuple[dict, bool]: ...    # (kwargs to send, injected_usage)
    def unprepare(self, kwargs: dict) -> dict: ...               # kwargs without PitStop's injection (retry)
    def estimate_prompt_tokens(self, kwargs: dict) -> int: ...
    def max_output_tokens(self, kwargs: dict) -> int: ...
    def read(self, resp: Any, obs: Observed) -> None: ...        # non-stream response
    def observe(self, chunk: Any, obs: Observed, injected: bool) -> bool: ...  # False = hide this chunk
```

- **Method table.** Maps an attribute path to a shape and a kind:

  ```python
  METHODS: dict[tuple[str, ...], tuple[Shape, Literal["call", "manager"]]]
  # this task's two entries:
  ("chat", "completions", "create"): (CHAT, "call")
  ("chat", "completions", "parse"):  (CHAT, "call")
  ```

  `wrap()` picks the table by client type. Task 6 adds the Anthropic table.
- **Resource proxies.** `_Resource(client, inner, path, mode)` proxies any attribute that is a prefix of a table path. The names `with_raw_response` and `with_streaming_response` (at client or resource level) return the same path with `mode = "raw"` or `"streaming"`. Anything else is delegated unchanged.
- **Raw mode.** A non-stream call reads usage via `result.parse()` (awaited for async). For `stream=True` the call is settled at the worst-case reservation, with payload `usage_estimated: "worst_case"`.
- **Streaming mode.** Settled at the worst-case reservation at return (decision 5).
- **`stream_options` retry.** When an injected `stream_options` call raises an error with `status_code == 400`, retry it once with `shape.unprepare(kwargs)`. If the retry succeeds, add the normalized base_url to the process-level `_NO_STREAM_OPTIONS: set[str]`, so later calls to that endpoint skip the injection. Whatever error the retry raises propagates.
- **`isinstance` on streams.** Wrapped streams report the inner stream's class via a `__class__` property, so `isinstance(s, openai.Stream)` stays True.
- **Test helpers** (`tests/sdk/http_fakes.py`):
  - `openai_client(handler, *, async_=False)`: a real `OpenAI`/`AsyncOpenAI` with `base_url=BASE_URL`, `api_key="test"`, `max_retries=0` and `http_client` on `httpx.MockTransport(handler)`;
  - `json_response(body, status=200)`;
  - `sse_response(events, *, named=False)`: OpenAI style `data: …\n\n` + `data: [DONE]`, or Anthropic style `event:`/`data:`;
  - `chat_completion(...)` / `chat_chunk(...)` dict builders.

- [ ] **Step 1: Write the failing tests** in `tests/sdk/test_wrap_helpers.py`:

```python
def test_parse_is_metered(tmp_path): ...                    # real client; ledger row settled from usage; llm_call event
def test_raw_response_create_settles_from_usage(tmp_path):
    raw = wrap(client).chat.completions.with_raw_response.create(model=MODEL, messages=MSGS)
    assert raw.parse().choices[0].message.content == "hi"   # caller still gets a working raw response
    assert _ledger_rows()[0]["state"] == "settled" and _ledger_rows()[0]["prompt_tokens"] == 11
def test_client_level_with_raw_response_is_metered(tmp_path): ...   # client.with_raw_response.chat.completions.create
def test_streaming_response_charged_worst_case(tmp_path): ...       # usd == reservation; payload usage_estimated
async def test_async_raw_response_is_metered(tmp_path): ...
def test_stream_options_400_retries_once_without_injection(tmp_path):
    # handler: 400 if "stream_options" in body else SSE; -> 2 requests, chunks delivered, settled by estimate;
    # a second stream call to the same base_url sends no stream_options (1 request)
def test_stream_options_400_not_retried_when_caller_set_it(tmp_path): ...  # caller-provided -> error propagates, 1 request
def test_wrapped_stream_isinstance_openai_stream(tmp_path): ...
def test_unlisted_methods_delegate_unchanged(tmp_path): ...         # models.list, embeddings.create pass through, no ledger row
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/sdk -v`. Expected: the new tests FAIL; the existing tests PASS.
- [ ] **Step 3: Implement the package.**
  - Move today's chat logic into `CHAT` (shapes) and `_Call` (call), keeping the module docstring's behavior contract. Extend it with: the method table, raw/streaming modes, the retry, and "detection only for agent meter".
  - `_Call.begin(client, shape, kwargs)` replaces `_Call.begin(client, kwargs)`.
  - Keep `pitstop.sdk.wrap.wrap` and `pitstop.sdk.wrap.meter` importable from the same paths.
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS, including every pre-existing wrap test.
- [ ] **Step 5: Commit.** Run `git commit -m "refactor(sdk): wrap as a shape-based package; meter chat parse and raw responses; retry without stream_options on 400"`.

---

### Task 5: Chat stream helper, Responses API and legacy completions

**Files:**
- Modify: `src/pitstop/sdk/wrap/shapes.py` (add `RESPONSES` and `COMPLETIONS`, plus table entries), `src/pitstop/sdk/wrap/proxy.py` (stream-manager proxies)
- Test: `tests/sdk/test_wrap_helpers.py`, new `tests/sdk/test_wrap_responses.py`

**Interfaces:**
- Consumes: `Shape`, `Observed`, `METHODS`, `http_fakes` (Task 4).
- Produces these table entries:

  ```python
  ("chat", "completions", "stream"): (CHAT, "manager")
  ("responses", "create"):           (RESPONSES, "call")
  ("responses", "parse"):            (RESPONSES, "call")
  ("responses", "stream"):           (RESPONSES, "manager")
  ("completions", "create"):         (COMPLETIONS, "call")
  ```

- **Manager kind.** The call reserves when the helper is called, after `prepare` (CHAT injects `stream_options` here too).
  - It returns a manager proxy whose `__enter__`/`__aenter__` yields a stream proxy.
  - The stream proxy passes every event through unchanged and feeds `event.chunk` (chat) or the raw event (responses) to `shape.observe`.
  - It delegates everything else, including `get_final_completion()` / `get_final_response()` / `until_done()`.
  - It settles on manager exit. A manager that is never entered releases its reservation when it is garbage collected.
- **`RESPONSES` shape.**
  - Prompt estimate: `instructions` + `input` (a string, or items whose `content` is a string or parts with `text`) + `tools` JSON.
  - Max output: `max_output_tokens`, else 4096.
  - Read: `usage.input_tokens` / `output_tokens` / `input_tokens_details.cached_tokens`; `output` items of type `function_call` → tool calls (`name`, `arguments`); text from `output_text` items; `finish_reason = status`.
  - Stream: `response.output_text.delta`; `response.output_item.added` with a `function_call` item; `response.function_call_arguments.delta`; usage from `response.completed`.
  - No injection.
- **`COMPLETIONS` shape.**
  - Prompt: `prompt` as a string or a list of strings.
  - Max: `max_tokens`, else 4096. The API default is 16, but PitStop reserves conservatively.
  - Read: `choices[].text`.
  - Stream: injects `stream_options` like chat.

- [ ] **Step 1: Write the failing tests.**

```python
def test_chat_stream_helper_matches_unwrapped(tmp_path):        # Review Focus 4
    for c in (raw_client, wrap(raw_client)):                     # same SSE script for both
        with c.chat.completions.stream(model=MODEL, messages=MSGS) as s:
            kinds = [e.type for e in s]
            final = s.get_final_completion()
        results.append((kinds, final.choices[0].message.content))
    assert results[0] == results[1]
    assert _ledger_rows()[-1]["state"] == "settled"
async def test_async_chat_stream_helper_metered(tmp_path): ...
def test_stream_helper_never_entered_releases(tmp_path): ...    # del manager; gc.collect() -> state 'released'
def test_responses_create_settles_usage_and_tool_calls(tmp_path): ...   # llm_call payload has the function_call
def test_responses_stream_settles_from_completed_event(tmp_path): ...
def test_responses_stream_helper_metered(tmp_path): ...
def test_responses_budget_exceeded_before_request(tmp_path): ...       # 0 requests reach the transport
def test_legacy_completions_metered_and_stream_usage_hidden(tmp_path): ...
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/sdk/test_wrap_helpers.py tests/sdk/test_wrap_responses.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the shapes, table entries and manager proxies (sync and async).**
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Docs.**
  - README "Add PitStop to your agent": list the metered methods.
  - Name what is not metered yet: Assistants/Realtime/Batch, and `client.beta.*` of either SDK.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(sdk): meter chat stream helper, Responses API and legacy completions"`.

---

### Task 6: Anthropic SDK support

**Files:**
- Create: `src/pitstop/sdk/wrap/anthropic.py` (`ANTHROPIC` shape, `ANTHROPIC_METHODS`)
- Modify:
  - `src/pitstop/sdk/wrap/__init__.py` (table selection);
  - `src/pitstop/budget/prices.py` (prefix fallback);
  - `src/pitstop/sdk/wrap/call.py` (`_after_llm_call` uses `Observed.refused`);
  - `pyproject.toml`;
  - README, CHANGELOG, `docs/config.example.toml`.
- Test: new `tests/sdk/test_wrap_anthropic.py`, `tests/budget/test_prices.py`

**Interfaces:**
- **Client detection.** `wrap(client)` uses the Anthropic table when `hasattr(client, "messages") and not hasattr(client, "chat")`.
- **Table:**

  ```python
  ("messages", "create"): (ANTHROPIC, "call")
  ("messages", "parse"):  (ANTHROPIC, "call")
  ("messages", "stream"): (ANTHROPIC, "manager")
  ```

  Raw and streaming modes work through Task 4's generic resource.
- **`ANTHROPIC` shape.**
  - Prompt estimate: `system` (a string or text blocks) + `messages` content (a string, or blocks with `text`, `tool_result` content or `input`) + `tools` JSON.
  - Max: `max_tokens`, which is required by the API; fall back to 4096 if it is missing.
  - Read the usage:
    - `prompt_tokens = input_tokens + ceil(1.25 * cache_creation_input_tokens) + cache_read_input_tokens`;
    - `cached_tokens = cache_read_input_tokens`;
    - `completion_tokens = output_tokens`.

    The 1.25 factor charges cache writes at the provider's write premium, so it is a deliberate over-count.
  - Read the content:
    - `content` blocks: `text` → content; `tool_use` → `{"name", "arguments": json.dumps(input)}`;
    - `finish_reason = stop_reason`, and `refused = stop_reason == "refusal"`.
  - Stream events:
    - `message_start` gives `message.usage`;
    - `content_block_start` with a `tool_use` block gives the tool name;
    - `content_block_delta` with `text_delta` / `input_json_delta` gives the text and tool arguments;
    - `message_delta` gives `usage.output_tokens` (cumulative) and `delta.stop_reason`.
  - No injection.
- **`refused` feeds L0.** `_after_llm_call` treats `obs.refused or is_refusal(content, n_tool_calls)` as a refusal.
- **Prices.** `prices._CONSERVATIVE_BY_PREFIX = {"claude-": Price(15.00, 75.00, 15.00)}`. `price_for` checks the user overrides, then the exact table, then the longest matching prefix (`known=False`), then `CONSERVATIVE_PRICE`.
- **`pyproject.toml`.** Add `[project.optional-dependencies] anthropic = ["anthropic>=2.0"]` and add `"anthropic>=2.0"` to the dev group.

- [ ] **Step 1: Write the failing tests** (a real `anthropic.Anthropic` / `AsyncAnthropic` over `httpx.MockTransport`; `max_retries=0`):

```python
def test_wrapped_anthropic_keeps_type(tmp_path):
    assert isinstance(wrap(anthropic.Anthropic(api_key="test")), anthropic.Anthropic)
def test_messages_create_settles_with_cache_tokens(tmp_path):
    # usage input 100, cache_creation 40, cache_read 60, output 20 -> prompt 210, cached 60, completion 20
def test_messages_stream_true_settles_from_events(tmp_path): ...
def test_messages_stream_helper_matches_unwrapped(tmp_path):    # Review Focus 4: text_stream + get_final_message()
async def test_async_messages_create_metered(tmp_path): ...
def test_refusal_stop_reason_counts_as_refusal(tmp_path): ...    # 2 refusals on normal sensitivity -> refusal_spike drift
def test_tool_use_blocks_feed_l0_loop_check(tmp_path): ...       # repeated tool_use -> recorded in llm_call payload
def test_unknown_claude_model_uses_family_conservative_price():
    assert price_for("claude-mystery-9", {}) == (Price(15.0, 75.0, 15.0), False)
def test_global_budget_by_anthropic_base_url(tmp_path): ...      # [budgets.anthropic] base_url = "https://api.anthropic.com"
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv sync && uv run pytest tests/sdk/test_wrap_anthropic.py tests/budget/test_prices.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Docs.**
  - README: show `pitstop.wrap(Anthropic())`.
  - `docs/config.example.toml`: a commented `[budgets.anthropic]` and a `[prices."claude-…"]` override example that tells users to copy current prices from the provider's pricing page.
  - CHANGELOG: a line for Anthropic support.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(sdk): wrap the Anthropic SDK (create, parse, stream, raw); conservative claude-* fallback price"`.

---

### Task 7: GEPA abort path

**Files:**
- Modify: `src/pitstop/correct/gepa.py` (`run_optimization`)
- Test: `tests/correct/test_gepa.py`

**Interfaces:**
- **Problem.** `run_optimization` currently swallows `Halted`: GEPA runs with `raise_on_exception=False`.
- **Guard the callables.** `run_optimization` wraps `evaluator` and `reflection_lm` in guards that share one `halted: list[Halted]`:
  - the first `Halted` (including `BudgetExceeded`) raised by either is stored and re-raised;
  - once one is stored, every later guarded call raises it immediately, without calling through.
- **After `optimize_anything` returns or raises with a stored `Halted`:**
  - write `run_dir/aborted.json` as `{"reason", "drift_id", "type"}`;
  - write no `result.json`, `best_policy.txt` or `artifacts.json`;
  - re-raise the stored `Halted`, keeping the same object and type.
- Produces: `OptimizationArtifacts` is returned only for runs that were not halted.

- [ ] **Step 1: Write the failing tests.**

```python
def test_budget_exceeded_aborts_optimization(tmp_path):
    calls = []
    def evaluator(candidate, example):
        calls.append(1)
        if len(calls) == 3:
            raise BudgetExceeded("global:crusoe", 50.0, 50.0)
        return 0.5, {}
    with pytest.raises(BudgetExceeded):
        run_optimization(..., evaluator=evaluator, cfg=OptimizeConfig(run_dir=str(tmp_path / "r"), max_workers=1, ...))
    assert len(calls) == 3                                   # nothing called through after the halt
    assert json.loads((tmp_path / "r/aborted.json").read_text())["type"] == "BudgetExceeded"
    assert not (tmp_path / "r/best_policy.txt").exists()
def test_halt_in_reflection_lm_aborts(tmp_path): ...
def test_unhalted_run_unchanged(tmp_path): ...               # existing happy path still writes artifacts, no aborted.json
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/correct/test_gepa.py -v`. Expected: the new cases FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit.** Run `git commit -m "fix(correct): a Halted or BudgetExceeded inside GEPA aborts the optimization instead of reporting results"`.

---

### Task 8: L1 pre-action gate

**Files:**
- Create: `src/pitstop/detect/l1.py`, `src/pitstop/assets/prompts/l1_gate_v1.txt`
- Modify:
  - `src/pitstop/userconfig.py` (`JudgeSpec`);
  - `src/pitstop/sdk/runtime.py` (`init(..., judge_client=None, judge_model=None)`, `Runtime.gate()`);
  - `src/pitstop/sdk/context.py` (`WorkflowSpec.on_gate_error`);
  - `src/pitstop/sdk/tool.py`;
  - README, `docs/config.example.toml`, CHANGELOG.
- Test: new `tests/detect/test_l1.py`, `tests/sdk/test_tool_gate.py`, `tests/store/test_userconfig.py`

**Interfaces:**
- Consumes:
  - `actions.apply_findings` (Task 3);
  - the "detection only for agent meter" rule (Task 3);
  - `meter("watch")` and `wrap`;
  - `extract_json_object` from `pitstop.llm`.
- **Judge config.** `userconfig.JudgeSpec(model: str, base_url: str, api_key_env: str, timeout_s: float)` comes from `[judge]`, with defaults:
  - `model = "openai/gpt-oss-120b"`;
  - `base_url = "https://api.inference.crusoecloud.com/v1"`;
  - `api_key_env = "CRUSOE_API_KEY"`;
  - `timeout_s = 10.0`.

  `UserConfig.judge: JudgeSpec` is always present (defaults when the table is absent). Validation: strings must be strings, and `timeout_s` must satisfy `_finite_nonneg` and be > 0.
- **`detect/l1.py`:**

```python
GATE_PROMPT_VERSION = "l1_gate_v1"
@dataclass(frozen=True)
class GateVerdict:
    allow: bool
    reason: str
    error: str | None = None          # set when the judge could not decide; caller applies on_gate_error
    latency_s: float = 0.0
class Gate:
    def __init__(self, client, model: str, *, max_tokens: int = 512): ...
    def check(self, *, workflow_goal: str | None, agent_goal: str | None, tool: str, args_text: str,
              recent: list[ToolCallRecord]) -> GateVerdict: ...
```

  - **The call.** `check` makes one sync chat call under `meter("watch")` with `temperature=0`. The user message lists:
    - the goals;
    - the proposed tool and its args (truncated to 2000 chars);
    - the agent's last 5 tool calls (each truncated to 300 chars).
  - **The verdict.** The prompt file asks for JSON `{"serves_goal": bool, "reason": str}`.
  - **Errors.** `Halted`/`BudgetExceeded` propagate. Any other exception, or an unparseable or missing `serves_goal`, returns `GateVerdict(False, "", error="<ExceptionType or 'unparseable verdict'>")`. The error text never contains the request, the key or the URL.
- **`Runtime.gate() -> Gate | None`** is built once and cached:
  - an `init(judge_client=...)` client is wrapped and used with `judge_model or config.judge.model`;
  - otherwise, if `os.environ[config.judge.api_key_env]` is set, `wrap(OpenAI(api_key=…, base_url=…, timeout=timeout_s, max_retries=0))` is used;
  - otherwise it returns `None`.
- **Gate failure policy.** New `WorkflowSpec.on_gate_error: Literal["block", "allow"] | None = None`. Its property `effective_on_gate_error` gives `"block"` for high and `"allow"` otherwise (spec §14.1). `workflow()` gains the keyword.
- **The gate in `@tool`.** It runs when all of these hold:
  - the tool has `side_effects=True`;
  - criticality is `high` or `medium`;
  - L0 produced no block, steer or halt.

  Sync tools call the gate inline. Async tools use `await asyncio.to_thread(...)`; the run context is copied into the thread so `meter` and the run's ledger tags apply.
- **Outcomes:**
  - **Every decision:** a `gate` event `{"allow", "reason", "latency_s", "prompt": GATE_PROMPT_VERSION}`.
  - **Reject:** `Finding("out_of_scope_action", "off_task", True, {"tool", "gate": reason, "prompt": GATE_PROMPT_VERSION})` goes through `apply_findings`, and the tool returns the standard block dict.
  - **Error or no gate:** a `gate_error` event `{"error", "policy"}`. If the policy is `block`, the tool returns `{"blocked_by": "pitstop", "reason": "pre-action check unavailable (<error>); blocked by on_gate_error=block. Configure [judge] in config.toml or set on_gate_error=\"allow\"."}`. With no gate, `<error>` is `"no judge configured: set CRUSOE_API_KEY or [judge] in config.toml"`.

- [ ] **Step 1: Write the failing tests** (the judge is the `FakeOpenAI` from `tests/sdk/fakes.py`, scripted with JSON content):

```python
def test_gate_allows_and_tool_runs(tmp_path): ...            # gate event allow=True; tool body ran; watch-meter ledger row
def test_gate_rejects_blocks_and_records_p0(tmp_path):        # medium wf -> block dict; drift P0 out_of_scope_action, gate evidence
def test_gate_reject_on_high_sets_pending_halt(tmp_path): ... # next wrapped call raises Halted
def test_high_without_judge_blocks_with_fix_hint(tmp_path):   # Review Focus 5
    r = send_email("a@b", "hi")
    assert r["blocked_by"] == "pitstop" and "[judge]" in r["reason"]
    assert _events("gate_error")[0]["payload"]["policy"] == "block"
def test_medium_without_judge_allows_and_records(tmp_path): ...
@pytest.mark.parametrize("crit,override,expect_run", [("high","allow",True), ("medium","block",False)])
def test_on_gate_error_override(tmp_path, crit, override, expect_run): ...
def test_unparseable_and_exception_verdicts_are_errors(tmp_path): ...
def test_judge_budget_exceeded_halts(tmp_path): ...           # global budget hit on judge call -> Halted from the tool call
def test_low_criticality_and_read_only_tools_not_gated(tmp_path): ...   # zero judge calls
def test_l0_block_skips_gate(tmp_path): ...                   # tools allowlist violation -> no judge call
async def test_async_tool_gate_runs_off_loop(tmp_path): ...
def test_judge_calls_do_not_run_l0_or_tag_run(tmp_path): ...  # judge refusals -> no refusal drift; runs.model is agent's
def test_judge_config_validation(tmp_path): ...               # timeout_s = 0 / "x" -> ConfigInvalid
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/detect/test_l1.py tests/sdk/test_tool_gate.py tests/store/test_userconfig.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Write `l1_gate_v1.txt`. It is short and frozen. It frames the judge as a reviewer of one proposed action against the stated goal and asks for the JSON object only. It follows the LlamaFirewall AlignmentCheck pattern from spec §6.2.
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Docs.**
  - README: a "Pre-action gate" subsection covering which tools are gated, the failure policy, cost and latency (~0.5–1.5 s per gated call), and `[judge]`.
  - `docs/config.example.toml`: `[judge]`.
  - CHANGELOG.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(detect): L1 pre-action gate for side-effect tools with per-workflow failure policy"`.

---

### Task 9: Fallback action

**Files:**
- Create: `src/pitstop/detect/changes.py`, `src/pitstop/sdk/fallback.py`
- Modify:
  - `src/pitstop/sdk/context.py` (`WorkflowSpec.fallback_model`);
  - `src/pitstop/sdk/actions.py` (activation);
  - `src/pitstop/sdk/wrap/call.py` (model swap at begin);
  - `src/pitstop/detect/drifts.py` (`_ACTION_RANK`);
  - `src/pitstop/detect/urgency.py`;
  - `src/pitstop/cli.py`;
  - README, CHANGELOG.
- Test: new `tests/detect/test_changes.py`, `tests/sdk/test_fallback.py`

**Interfaces:**
- **Change detection** (`detect/changes.py`):

  ```python
  @dataclass(frozen=True)
  class Change:
      field: Literal["model", "base_url"]
      before: str | None
      after: str | None
      prev_run_id: str
      at: float
  def recent_change(store, workflow: str, run_id: str) -> Change | None
  ```

  `recent_change` compares this run's `model`/`base_url` (from the `runs` row) with the most recent *other* run of the workflow that has a non-null model. The model is checked first. Task 10's alerts use it too.
- **Fallback state** (`sdk/fallback.py`):
  - `FALLBACK_TRIGGERS = frozenset({"tool_call_breakage", "refusal_spike"})`.
  - `active(store, workflow) -> dict | None` reads meta key `fallback:<workflow>`. The value is JSON `{"model", "since", "drift_id", "change": asdict(Change)}`.
  - `activate(store, workflow, *, model, drift_id, change) -> bool` returns False if a fallback is already active.
  - `clear(store, workflow) -> bool`.
  - `list_active(store) -> list[dict]`.
- **Workflow option.** New `WorkflowSpec.fallback_model: str | None = None`, plus the `workflow(..., fallback_model=None)` keyword.
- **Activation** (in `apply_findings`). It needs all of these:
  - `ctx` is set and the finding is in `FALLBACK_TRIGGERS`;
  - the level is P1;
  - `spec.fallback_model` is set and differs from the run's model;
  - `recent_change(...)` is not None.

  Then: `activate`, the drift's action becomes `"fallback"`, and a `fallback` event `{"from", "to", "change"}` is recorded. `_ACTION_RANK` becomes record 0, fallback 1, steer 2, block 3, halt 4. `urgency.action_for` is unchanged; the swap happens after it.
- **The model swap** (in `_Call.begin`). It needs all of these:
  - the meter is `agent`;
  - `ctx` is set and `ctx.spec.fallback_model` is set;
  - `fallback.active(store, ctx.spec.name)` is true.

  Then it sets `kwargs["model"] = fallback_model` before the estimate. The `llm_call` payload gets `"fallback_from": <requested model>`. If the store read fails, the call keeps its model (fail open).
- **CLI:**
  - `pitstop fallback status` prints a table: workflow, model, since, change;
  - `pitstop fallback clear <workflow>` prints "cleared" or "no active fallback" (exit 0 either way).

- [ ] **Step 1: Write the failing tests.**

```python
def test_recent_change_detects_model_and_base_url(store): ...
def test_recent_change_none_without_history_or_change(store): ...
def test_breakage_after_model_change_activates_fallback(tmp_path):
    # run 1 on model A (ok); run 2 on model B with parse failures -> P1 tool_call_breakage, action 'fallback',
    # 'fallback' event; next call in run 2 and calls in run 3 send model == fallback_model
def test_no_change_no_fallback(tmp_path): ...                  # same model both runs -> action 'record'
def test_no_fallback_model_declared_no_fallback(tmp_path): ...
def test_low_criticality_breakage_is_p2_no_fallback(tmp_path): ...
def test_watch_meter_calls_not_swapped(tmp_path): ...
def test_clear_restores_requested_model(tmp_path): ...
def test_fallback_cli_status_and_clear(tmp_path, capsys): ...
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/detect/test_changes.py tests/sdk/test_fallback.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Docs.**
  - README: a "Fallback" subsection covering when it engages, that it is sticky, and how to clear it.
  - CHANGELOG.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(sdk): fallback to a pinned model on P1 breakage after a model or provider change"`.

---

### Task 10: Alerts

**Files:**
- Create the package `src/pitstop/alerts/`:
  - `__init__.py`;
  - `model.py`: `Alert`, `build_alert`;
  - `channels.py`: PagerDuty, Opsgenie, Slack, Email, Webhook;
  - `dispatch.py`: routing, dedup, the thread pool, digest.
- Modify:
  - `src/pitstop/userconfig.py` (`AlertsSpec`);
  - `src/pitstop/store/__init__.py` + `schema.sql` (v4: `alerts` table);
  - `src/pitstop/sdk/actions.py` (hook);
  - `src/pitstop/cli.py`;
  - `tests/conftest.py` (scrub the alert env vars);
  - README, `docs/config.example.toml`, CHANGELOG.
- Test: new `tests/alerts/test_channels.py`, `tests/alerts/test_dispatch.py`, `tests/alerts/test_alerts_cli.py`, `tests/store/test_userconfig.py`

**Interfaces:**
- Consumes:
  - `recent_change` (Task 9);
  - `MIGRATIONS` (Task 2);
  - `apply_findings` (Task 3);
  - `_finite_nonneg` (Task 1).
- **Config.** `userconfig.AlertsSpec` is `UserConfig.alerts`; it is empty when absent. It parses `[alerts.<channel>]` for these channels:

  | Channel | Fields (defaults) |
  |---|---|
  | pagerduty | `routing_key_env` (`"PAGERDUTY_ROUTING_KEY"`) |
  | opsgenie | `api_key_env` (`"OPSGENIE_API_KEY"`), `region` (`"us"` \| `"eu"`) |
  | slack | `webhook_url_env` (`"SLACK_WEBHOOK_URL"`) |
  | email | `smtp_host`, `smtp_port` (587), `starttls` (true), `username`, `password_env` (`"SMTP_PASSWORD"`), `from_addr`, `to` (list) |
  | webhook | `url_env` (`"PITSTOP_WEBHOOK_URL"`) |

  - **Inline secrets.** The inline keys `routing_key`, `api_key`, `webhook_url`, `password` and `url` raise `ConfigInvalid("…put secrets in environment variables; use <key>_env")` (decision 2).
  - **Routes.** `[alerts.routes]` takes the keys `P0`, `P1` and `digest`, each a list of channel names, and unknown names are rejected. The defaults are:
    - P0: `["pagerduty", "opsgenie", "slack", "webhook"]`;
    - P1: `["slack", "email", "webhook"]`;
    - digest: `["email", "slack", "webhook"]`.

    Only channels that are configured and whose env var is set are used.
- **Store v4:**

  ```sql
  alerts(id INTEGER PRIMARY KEY, drift_id INTEGER, level TEXT, channel TEXT, kind TEXT, ts REAL, status TEXT,
         error TEXT)
  ```

  - `kind` is `immediate` | `digest` | `test`.
  - `status` is `pending` | `sent` | `failed`.
- **The `Alert` model** (`alerts/model.py`):

  ```python
  @dataclass(frozen=True)
  class Alert:
      drift_id: int | None
      workflow: str | None
      level: str
      drift_type: str
      what: str       # what drifted
      cause: str      # what likely changed
      action: str     # what is being done
      run_id: str | None
      ts: float
      @property
      def title(self) -> str: ...      # f"[{level}] {workflow}: {drift_type}"
      @property
      def body(self) -> str: ...       # "What drifted: …\nWhat likely changed: …\nWhat PitStop did: …"
      @property
      def dedup_key(self) -> str: ...  # f"pitstop:{workflow}:{drift_type}"
  def build_alert(store, drift_id: int) -> Alert: ...
  ```

  `build_alert` fills `cause` from `recent_change`. Without one, the text is "No model or provider change detected for this run (model X at base_url Y)."
- **Channels** (`channels.py`). Each implements `send(alert: Alert) -> None` and raises `ChannelError(kind: str)` on failure. The `kind` is only an exception type name or `f"HTTP {status}"`: never `str(e)`, a URL or a body.
  - HTTP goes through the module-level `_http_client() -> httpx.Client(timeout=5.0)`, which tests monkeypatch.
  - SMTP goes through `_smtp(host, port) -> smtplib.SMTP` with `timeout=10`.

  | Channel | Request | OK status |
  |---|---|---|
  | PagerDuty | `POST https://events.pagerduty.com/v2/enqueue` with `{"routing_key", "event_action": "trigger", "dedup_key", "payload": {"summary": title[:1024], "source": "pitstop", "severity": {"P0": "critical", "P1": "error", "P2": "warning", "P3": "info"}[level], "custom_details": {...}}}` | 202 |
  | Opsgenie | `POST https://api.opsgenie.com/v2/alerts` (eu: `https://api.eu.opsgenie.com/v2/alerts`), header `Authorization: GenieKey <key>`, body `{"message": title[:130], "alias": dedup_key, "description": body[:15000], "priority": {"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4"}[level], "source": "pitstop"}` | 202 |
  | Slack | `POST <webhook>` with `{"text": f"*{title}*\n{body}"}` | 200 |
  | Email | `EmailMessage` (Subject = title), STARTTLS if set, login if `username` is set | — |
  | Webhook | `POST <url>` with `asdict(alert) \| {"title", "body"}` | any 2xx |
- **Dispatch** (`dispatch.py`):
  - `notify(drift_id: int) -> None` sends P0/P1 drifts only. It is called by `apply_findings` (via `_safe`) after each record of a P0/P1 drift. It submits the work to the module's `ThreadPoolExecutor(max_workers=2, thread_name_prefix="pitstop-alerts")` and returns at once. Its worker threads are joined at interpreter exit, which delivers Review Focus 3.
  - Per routed channel, under `BEGIN IMMEDIATE`, it skips the channel if an `immediate` row exists for this drift and channel with `status IN ('pending', 'sent')` and a level at least as urgent. Otherwise it inserts a `pending` row, sends, then updates the row to `sent` or `failed` + `error`. A drift escalated from P1 to P0 alerts again.
  - `digest(now: float | None = None, *, force: bool = False) -> int` collects open P2/P3 drifts that have no `digest` row. It sends one message per digest channel, records a row per drift and channel, and returns the number of drifts. Unless `force` is set, it does nothing if the last digest (meta `last_digest_ts`) was under 24 h ago.
  - `send_test(channel: str | None = None) -> dict[str, str]` returns `channel -> "sent" | "failed: <kind>" | "not configured"`.
  - `wait_idle(timeout: float = 10.0) -> bool` is for tests and the CLI.
- **CLI:** `pitstop alerts test [--channel NAME]`, `pitstop alerts digest [--force]`.
- **Test env scrub.** `tests/conftest.py` also deletes `PAGERDUTY_ROUTING_KEY`, `OPSGENIE_API_KEY`, `SLACK_WEBHOOK_URL`, `PITSTOP_WEBHOOK_URL` and `SMTP_PASSWORD`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_pagerduty_payload(monkeypatch): ...                  # exact URL, routing_key, dedup_key, severity mapping
def test_opsgenie_payload_and_eu_region(monkeypatch): ...
def test_slack_and_webhook_payloads(monkeypatch): ...
def test_email_uses_starttls_and_login(monkeypatch): ...      # fake SMTP class records calls
def test_inline_secret_rejected(tmp_path): ...                # [alerts.slack] webhook_url = "https://…" -> ConfigInvalid
def test_unknown_route_channel_rejected(tmp_path): ...
def test_p0_drift_alerts_routed_channels_once(tmp_path): ...  # 3 identical P0 findings -> 1 send per channel
def test_escalation_realerts(tmp_path): ...                   # P1 then P0 on same drift -> second send at P0
def test_p2_not_sent_immediately_but_in_digest(tmp_path): ...
def test_digest_respects_24h_unless_forced(tmp_path): ...
def test_failed_send_recorded(tmp_path): ...                  # 500 -> row failed, error "HTTP 500"
def test_hanging_channel_does_not_block_agent(tmp_path):      # Review Focus 1
    release = threading.Event()
    # MockTransport handler waits on `release`; apply_findings(P0) must return in < 0.5 s; then release.set()
def test_secret_never_stored_or_logged(tmp_path, caplog):     # Review Focus 2
    # SLACK_WEBHOOK_URL=https://hooks.slack.test/SECRET123, transport raises httpx.ConnectError(url in message)
    # -> "SECRET123" absent from every alerts/events row and from caplog.text
def test_p0_alert_delivered_when_process_exits(tmp_path):     # Review Focus 3
    # loopback http.server on 127.0.0.1:0 records POSTs; subprocess (cwd=tmp_path, keys scrubbed,
    # PITSTOP_WEBHOOK_URL=http://127.0.0.1:<port>/hook) runs a high wf whose side-effect tool is out of scope,
    # catches nothing, exits via Halted -> server saw exactly 1 POST with level "P0"
def test_alert_body_names_model_change(tmp_path): ...         # cause mentions "model changed from A to B"
def test_alerts_test_cli_reports_per_channel(tmp_path, capsys): ...
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/alerts tests/store/test_userconfig.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add `MIGRATIONS[4]` and bump `SCHEMA_VERSION = 4`. Task 2's `test_fresh_db_matches_migrated_db` must still pass.
- [ ] **Step 4: Run them and confirm they pass.** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Docs.**
  - README: an "Alerts" section with the routing table from spec §5, setup per channel (env vars only), `pitstop alerts test`, and the digest command.
  - `docs/config.example.toml`: `[alerts.*]` and `[alerts.routes]`, commented out.
  - CHANGELOG.
- [ ] **Step 6: Commit.** Run `git commit -m "feat(alerts): PagerDuty, Opsgenie, Slack, email and webhook alerts routed by urgency, deduplicated per open drift"`.

---

### Task 11: Milestone wrap-up

**Files:**
- Modify: `README.md` (roadmap: 2b done, 2c next), `CHANGELOG.md`, and this plan (append "Carried to milestone 2c" at the end of execution)
- Test: `tests/test_quickstart_example.py` (existing), the full suite

**Interfaces:** none new.

- [ ] **Step 1: Check the offline quickstart.** Run `uv run python examples/quickstart/agent.py --offline --inject-loop` with `PITSTOP_HOME` set to a temp dir. Expected: two steers then a halt, exactly as in 2a. Run `uv run pitstop status` and `uv run pitstop drifts --all`. Expected: occurrences and last_seen are shown.
- [ ] **Step 2: Run the checks.** Run `uv run pytest -q && uv run ruff check`. Expected: all PASS, and "All checks passed!".
- [ ] **Step 3: Check the README.** Cover the roadmap update and confirm every new section is linked from the table of contents.
- [ ] **Step 4: Commit.** Run `git commit -m "docs: milestone 2b complete; roadmap points to 2c"`.

---

## Carried to milestone 2c (recorded at the end of execution)

These came out of the task reviews and the final whole-branch review. The M2c plan must cover or explicitly drop them.

- **From the spec, not yet built:**
  - New runs are not refused at the start once a budget is exhausted (spec §3.2); today the first call of the run halts instead.
  - The "spend rate above N× baseline" P0 trigger (§5) needs L2.
- **Alerts:**
  - A P0 whose drift write failed pages nobody (`notify` needs a drift id); add a best-effort, undeduplicated send.
  - An alert's cause ignores fallback-tagged runs only while a fallback is active, so within 24 h of a clear it can read "model changed from <pinned>".
  - Superseded stale `pending` rows stay `pending` in the table (cosmetic).
  - Generic `SMTPException` and an `SMTPConnectError` carrying a permanent 5xx are still retried (bounded).
- **Gate (L1):**
  - The frozen prompt does not delimit args and history: ship `l1_gate_v2` with delimiters (prompt-injection hardening).
  - `[judge] base_url = ""` is accepted and fails only at call time.
  - Add a README privacy sentence: tool arguments are sent to the judge endpoint.
- **Fallback:**
  - Decide whether a fallback triggered by a `base_url` change is useful without a pinned provider (it sends the pinned model id to the new provider). This is a plan decision 3 clarification.
  - The drift keeps action `fallback` after a clear.
  - `fallback_from` reads `"None"` when a call has no `model` kwarg.
- **`wrap`:**
  - The Responses estimate is small with `previous_response_id`/`conversation`, and legacy token-id prompts estimate 0. Both settle from real usage, but one large call can overshoot the hard stop.
  - Responses tool calls other than `function_call` (`custom_tool_call`, hosted tools) are not recorded for L0, and refusal parts are not counted.
  - Anthropic premium billing modes (fast speed, long context, server-tool fees) are not modelled. This is documented; users need a price override.
  - `tool_runner` metering relies on the SDK's private `runner._client`. It is pinned by a test and falls back to unmetered with a warning.
  - Earlier chat-helper events' `event.snapshot` still show injected usage, and `current_completion_snapshot` returns a fresh copy per access. Document this.
  - Streaming-response mode is charged worst case even if its manager is never entered (plan decision 5).
  - Async tests are missing for enter-error release.
  - `Mode = str` could be a `Literal`.
- **SDK runtime:**
  - An async-generator workflow collected inside a reference cycle can still race the inner generator's finalizer.
  - The steer-limit halt sets `halt_reason` without `halt_drift_id`.
  - `meter()` stepped from a different Context than the one that entered it raises ValueError on reset.
- **GEPA:**
  - GEPA keeps iterating (fast-fail) after a halt; a `stop_callbacks` stopper would end it at once.
  - `except BaseException` turns KeyboardInterrupt into the stored `Halted`.
  - `max_workers > 1` is untested.
- **Store:**
  - Add a `runs(workflow, started_at)` index with the next schema bump.
  - Close per-thread connections of background pools at shutdown before 2c adds more background threads.
- **Tests and docs:**
  - The fork test has no `waitpid` timeout.
  - `test_chat_stream_helper_caller_stream_options_keeps_usage` leaves a manager entered.
  - The CHANGELOG L1 gate entry's clause order reads ambiguously.
  - The `pitstop budget reconcile` CLI docstring wording is now fixed; no action needed.
