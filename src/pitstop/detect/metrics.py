"""L2 per-run metrics (spec 6.2): the numbers one finished run contributes to its workflow's change detectors.

`run_metrics` reads only that run's `events` and its `runs.started_at` / `ended_at`. "Agent calls" are the run's
`llm_call` events with `payload.meter == "agent"`, which includes OTel-observed and observed (unmetered) calls.

- `tool_error_rate`: min(errored `tool_result` events, `tool_call` events) / `tool_call` events. A result is errored
  when its payload has an `"error"` key (every source writes one; an OTel success is `{"source": "otel"}`). The cap
  covers a nested `@pitstop.tool`, which can record two results for one call.
- `refusal_rate`: agent calls with `refused` true / agent calls.
- `tokens_per_call`: mean of `prompt_tokens + completion_tokens` over agent calls where both are set.
- `output_chars`: mean of `content_chars` over agent calls where it is set (OTel writes null).
- `llm_calls`: the number of agent calls (0 included).
- `cost_usd`: the sum of `events.cost_usd` over agent calls (OTel and observed estimates included; null counts 0).
- `duration_s`: `ended_at - started_at`.

A metric is left out when the run has no data for it: no `tool_call` events, no agent calls (the rates and
`cost_usd`), no non-null value (the means), or no end time. A `refused` run (it never started) or a `dropped` one
(a partial OTel trace) has no metrics at all.

`METRICS` gives each metric its drift type, its alarming directions and its std floor for L2: 0.05 absolute for
the rates; 5% of the baseline mean, but at least `min_floor` (1 call / 1 token / 1 char / $0.0001 / 0.1 s), for
the rest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

_NO_METRICS_STATUSES = ("refused", "dropped")


@dataclass(frozen=True)
class MetricSpec:
    drift_type: str
    directions: tuple[Literal["up", "down"], ...]
    floor: float  # minimum std used for z-scores
    relative: bool  # floor is a fraction of |baseline mean| (else absolute)
    min_floor: float = 0.0  # absolute minimum std, so a relative floor holds on a zero baseline


METRICS: dict[str, MetricSpec] = {
    "tool_error_rate": MetricSpec("tool_call_breakage", ("up",), 0.05, False),
    "refusal_rate": MetricSpec("refusal_spike", ("up",), 0.05, False),
    "tokens_per_call": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 1.0),
    "llm_calls": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 1.0),
    "cost_usd": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 0.0001),
    "duration_s": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 0.1),
    "output_chars": MetricSpec("quality_drop", ("up", "down"), 0.05, True, 1.0),
}


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _payload(text) -> dict:
    try:
        p = json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {}
    return p if isinstance(p, dict) else {}


def run_metrics(store, run_id: str) -> dict[str, float]:
    """The run's L2 metrics (see the module docstring); `{}` for an unknown, refused or dropped run."""
    conn = store.connect()
    run = conn.execute("SELECT started_at, ended_at, status FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] in _NO_METRICS_STATUSES:
        return {}
    rows = conn.execute("SELECT kind, payload, cost_usd FROM events WHERE run_id = ?"
                        " AND kind IN ('llm_call', 'tool_call', 'tool_result')", (run_id,)).fetchall()
    calls = refusals = tool_calls = tool_errors = 0
    cost = 0.0
    tokens: list[float] = []
    chars: list[float] = []
    for r in rows:
        p = _payload(r["payload"])
        if r["kind"] == "tool_call":
            tool_calls += 1
        elif r["kind"] == "tool_result":
            tool_errors += "error" in p
        elif p.get("meter") == "agent":
            calls += 1
            refusals += p.get("refused") is True
            cost += _num(r["cost_usd"]) or 0.0
            prompt, completion = _num(p.get("prompt_tokens")), _num(p.get("completion_tokens"))
            if prompt is not None and completion is not None:
                tokens.append(prompt + completion)
            if (c := _num(p.get("content_chars"))) is not None:
                chars.append(c)

    out: dict[str, float] = {"llm_calls": float(calls)}
    if tool_calls:
        out["tool_error_rate"] = min(tool_errors, tool_calls) / tool_calls
    if calls:
        out["refusal_rate"] = refusals / calls
        out["cost_usd"] = cost
    if tokens:
        out["tokens_per_call"] = sum(tokens) / len(tokens)
    if chars:
        out["output_chars"] = sum(chars) / len(chars)
    started, ended = _num(run["started_at"]), _num(run["ended_at"])
    if started is not None and ended is not None:
        out["duration_s"] = ended - started
    return out
