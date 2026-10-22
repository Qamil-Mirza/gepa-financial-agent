# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Drop-in SDK: `pitstop.wrap`, `workflow`, `agent`, `tool` (block/steer/halt guard), `prompt` (versioned, developer edits win), `Halted` and `BudgetExceeded`.
- Hard dollar budget ledger with atomic reserve/settle, and per-workflow `budget_usd`; PitStop's own model calls go through it (`docs/config.example.toml`).
- `pitstop status`, `pitstop drifts` and `pitstop budget` commands, and a keyless quickstart example (`examples/quickstart`).
- Open-source repo basics: Apache-2.0 license, Contributor Covenant code of conduct, security policy, issue and pull-request templates, changelog and a CI workflow.
- Ruff linting (pyflakes rules) and offline-by-default test suite.
- `pitstop budget reconcile [--older-than-min 60] [--release]` lists old budget reservations, left open by a process that died mid-call or by a call still running; `--release` frees them and lists what it freed.
- `pitstop.workflow` and `pitstop.agent` decorate generator and async-generator functions; the run spans the generator's steps and never leaks to the consuming code.
- `wrap` meters `chat.completions.parse` and `with_raw_response` / `with_streaming_response` calls (on the client or the resource). Raw responses are settled from their parsed usage; streaming responses and raw streams, which PitStop can't read, are charged their full reservation (`usage_estimated: "worst_case"`).
- An endpoint that rejects PitStop's injected `stream_options` with a 400 is retried once without it, and later stream calls to that base_url skip the injection.
- `wrap` meters the chat stream helper (`chat.completions.stream`), the Responses API (`responses.create` / `parse` / `stream`) and legacy `completions.create`, sync and async. A stream helper reserves when called, settles when its manager exits, and releases its reservation if it is never entered. A background Responses call (`background=True`, or a result still `queued` / `in_progress`) is charged its full reservation. Assistants, Realtime, Batch and `client.beta.*` are not metered yet (Anthropic's `beta.messages` is, see below).
- `wrap` supports the Anthropic SDK (`Anthropic` / `AsyncAnthropic`, install `pitstop[anthropic]`): `messages.create` (and `stream=True`), `messages.parse`, `messages.stream` (with `text_stream`, `get_final_message()` and `get_final_text()`), the same under `beta.messages`, `beta.messages.tool_runner` (each of the runner's model calls is metered), and raw / streaming responses; `messages.batches` is not metered. Cache writes are charged at their write premium (1.25x input for 5-minute writes, 2x for 1-hour writes and for writes without a breakdown), a `stop_reason` of `"refusal"` counts as a refusal for L0, and an unpriced `claude-*` model (or Bedrock-style `…anthropic.claude-*` id) falls back to a conservative $15 / $75 per million tokens (`known=False`).
- L1 pre-action gate: a `side_effects=True` tool on a `high` or `medium` workflow is checked by a judge model (prompt `l1_gate_v1`, `watch` meter) before it runs; a rejection is a P0 `out_of_scope_action` drift and blocks the call. Configure the judge in `[judge]` (`model`, `base_url`, `api_key_env`, `timeout_s`; defaults to `openai/gpt-oss-120b` on Crusoe via `CRUSOE_API_KEY`) or pass `pitstop.init(judge_client=..., judge_model=...)`. When the judge is missing or cannot decide, `pitstop.workflow(on_gate_error=...)` decides: `high` blocks, `medium` runs by default. Verdicts are `gate` events, failures `gate_error` events.
- `pitstop.workflow(...)` rejects an invalid `criticality`, `sensitivity`, `fix_mode` or `on_gate_error` with a `ValueError` when the workflow is defined.
- Fallback action: `pitstop.workflow(..., fallback_model="...")` pins a model. A P1 `tool_call_breakage` or `refusal_spike` in a run whose model or `base_url` changed within the last 24 hours (runs on the pinned model don't count as a change) engages it: later agent-meter calls of the workflow go to the pinned model (`llm_call` events carry `fallback_from`), the drift's action is `fallback` and a `fallback` event is recorded. It stays on until `pitstop fallback clear <workflow>`; `pitstop fallback status` lists active fallbacks.
- Alerts: P0 and P1 drifts are sent right away to PagerDuty (Events v2), Opsgenie, Slack (incoming webhook), SMTP email or a generic webhook, routed by urgency (`[alerts.routes]`; defaults P0 -> PagerDuty, Opsgenie, Slack, webhook and P1 -> Slack, email, webhook) from a background thread that never blocks the agent and finishes before the process exits. Each open drift alerts once per channel, again when it escalates from P1 to P0. Failed P0/P1 sends are retried twice with a short backoff, and a send left pending by a killed process stops blocking after 2 minutes. P2/P3 drifts go in a daily digest (`pitstop alerts digest [--force]`). Every alert says what drifted, what likely changed and what PitStop did. `pitstop alerts test [--channel NAME]` checks each channel. Secrets come only from env vars named in `[alerts.<channel>]`; inline secrets are rejected.

### Changed
- `pitstop` is now an installable `src/` layout package with packaged assets.
- The GEPA engine lives in `pitstop.correct`; the evaluator is supplied by the caller.
- Eval sets and detector calibration tooling are collected under `benchmarks/`.
- Hackathon-era docs moved to `docs/history/`.
- Runs and the You.com cache now default to `~/.pitstop/runs` and `~/.pitstop/cache/youcom` (override with `PITSTOP_HOME`, `PITSTOP_RUNS_DIR`, `PITSTOP_CACHE_DIR`); previously they lived under `pitstop/pitstop/runs` and `pitstop/pitstop/cache` in the checkout.
- `pitstop` is now an installed console script (previously `python -m pitstop.cli` run from `pitstop/pitstop/`).
- Store schema v4 (v4 adds the `alerts` table): older DBs are upgraded step by step in one transaction, and a DB written by a newer PitStop fails closed with `Halted("pitstop config invalid: ...")`.
- Drift records count every occurrence (`occurrences`, `last_seen`) and keep only the newest 20 evidence entries, each `{"run_id", "ts", "detail"}`; `pitstop drifts` shows the real first and last seen times.
- `runs.base_url` is stored normalized (no trailing `/`), matching the budget ledger.
- Drift detection and the run's model tag use only calls on the `agent` meter; judge, supervisor and GEPA calls no longer trigger L0 drifts. `llm_call` events carry their `meter`.
- Per-agent tool history and per-run LLM history keep the newest 64 calls; refusals are counted over the whole run.
- The steer limit (2 per run) holds when several threads share a run.
- `pitstop.sdk.wrap` is a package; wrapped streams pass `isinstance` checks against the SDK's stream classes.
- A pending P0 halt on a high-criticality workflow is recorded as a `halt` event (`pending: true`) when it is set.
- Drift actions escalate on dedup in the order record < fallback < steer < block < halt.
- With no judge configured (no `CRUSOE_API_KEY`, no `[judge]` key, no `judge_client`), `side_effects=True` tools on `high` workflows are now blocked by the pre-action gate's failure policy; set `on_gate_error="allow"` on the workflow to run them anyway. The offline quickstart uses a scripted judge.

### Removed
- The superseded supervisor stack, dead scaffold and committed generated data.
