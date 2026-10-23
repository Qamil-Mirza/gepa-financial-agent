"""The spend-rate P0 (spec 5: "spend rate above N x baseline"): a run whose settled agent spend passes N times its
workflow's typical run cost is a `spend_spike`, P0 with action `halt` on any criticality, like a budget hit.

The baseline is the median settled agent-meter spend of the workflow's last `BASELINE_RUNS` runs that finished
`ok`; with fewer than `MIN_BASELINE_RUNS` of them there is no baseline and no check. It is floored at
`MIN_BASELINE_USD`, so a near-free workflow does not spike on its first real cent. The check runs inline after
each agent-meter settle (`pitstop.sdk.wrap.call`); the run halts at its next call boundary."""
from __future__ import annotations

import statistics

from ..sdk.context import Sensitivity
from .l0 import Finding

SPEND_MULTIPLIER: dict[Sensitivity, float] = {"low": 10.0, "normal": 5.0, "high": 3.0}
BASELINE_RUNS = 20
MIN_BASELINE_RUNS = 10
MIN_BASELINE_USD = 0.01


def baseline_run_cost(store, workflow: str, *, before: float) -> float | None:
    """The median settled agent-meter USD per run over the last BASELINE_RUNS `ok` runs of `workflow` that started
    before `before`; None when fewer than MIN_BASELINE_RUNS qualify. Runs observed through OpenTelemetry (adapter
    `otel`) never count: their spend is never in the ledger, so they would read as $0 runs."""
    rows = store.connect().execute(
        "SELECT (SELECT COALESCE(SUM(l.usd), 0) FROM ledger l"
        "        WHERE l.run_id = r.id AND l.meter = 'agent' AND l.state = 'settled') AS usd"
        " FROM runs r WHERE r.workflow = ? AND r.status = 'ok' AND r.started_at < ?"
        " AND (r.adapter IS NULL OR r.adapter != 'otel')"
        " ORDER BY r.started_at DESC LIMIT ?", (workflow, before, BASELINE_RUNS)).fetchall()
    if len(rows) < MIN_BASELINE_RUNS:
        return None
    return float(statistics.median(float(r["usd"]) for r in rows))


def check_spend(run_usd: float, baseline: float | None, sensitivity: Sensitivity) -> Finding | None:
    """A `spend_spike` finding when `run_usd` is above the sensitivity's multiple of the (floored) baseline."""
    if baseline is None:
        return None
    multiplier = SPEND_MULTIPLIER[sensitivity]
    if run_usd > multiplier * max(baseline, MIN_BASELINE_USD):
        return Finding("spend_spike", "degraded", False,
                       {"run_usd": round(run_usd, 6), "baseline_usd": round(baseline, 6), "multiplier": multiplier})
    return None
