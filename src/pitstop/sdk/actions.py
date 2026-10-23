"""Turn L0 findings into actions: classify each one (level, action), record it as a drift, and set the pending
halt that a "halt" action (a budget hit or spend spike) or a P0 on a high-criticality workflow calls for. Shared
by `@pitstop.tool` and the wrapped clients."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, NamedTuple

from .context import RunContext, _safe
from .runtime import get_runtime

log = logging.getLogger("pitstop")


class Applied(NamedTuple):
    finding: Any  # pitstop.detect.l0.Finding
    level: Any  # pitstop.detect.urgency.Level
    action: str
    drift_id: int | None


def apply_findings(ctx: RunContext | None, findings: list, *, run_id: str | None = None) -> list[Applied]:
    """Classify each finding (level, action), then record it. A "halt" action (a budget hit or spend spike, any
    criticality), or a P0 on a high workflow, sets a pending halt that the next boundary raises, and records a
    `halt` event marked `pending`. A P1 breakage or refusal spike on a workflow that pins a `fallback_model`, in a
    run whose model or base_url changed within the last 24 h (see `pitstop.detect.changes.recent_change`), engages
    the fallback (sticky until cleared): the drift's action is "fallback" and a `fallback` event is recorded.
    Outside a workflow: workflow=None, criticality medium, sensitivity normal.

    The block/steer/halt decision never depends on the store: a failed drift or event write is logged, leaves
    drift_id None, and the action still applies. The fallback does need the store (the run's model, the previous
    run, the shared state); when it cannot be read the call keeps its model (fail open).

    Each recorded P0/P1 occurrence is handed to `pitstop.alerts.dispatch.notify`, which queues the alert and
    returns at once (the alert is sent once per drift and channel, again on escalation to P0)."""
    from ..detect import drifts
    from ..detect.urgency import Level, action_for, classify

    criticality = ctx.spec.criticality if ctx is not None else "medium"
    workflow = ctx.spec.name if ctx is not None else None
    run_id = ctx.run_id if ctx is not None else run_id
    applied: list[Applied] = []
    for f in findings:
        level = classify(f.drift_type, criticality, f.side_effects)
        action = action_for(f, level, criticality)
        engage = _safe(lambda: _fallback_change(ctx, f.drift_type, level)) if ctx is not None else None
        if engage is not None:
            action = "fallback"
        try:
            drift_id = drifts.record_drift(get_runtime().store, run_id=run_id, workflow=workflow, finding=f,
                                           level=level, action=action)
        except Exception:
            log.warning("pitstop: could not record a %s drift; its action still applies", f.drift_type,
                        exc_info=True)
            drift_id = None
        if engage is not None:
            _safe(lambda: _engage_fallback(ctx, f.drift_type, drift_id, *engage))
        if drift_id is not None and level in (Level.P0, Level.P1):
            _safe(lambda: _notify(drift_id))
        if ctx is not None and (action == "halt" or (level is Level.P0 and criticality == "high")):
            reason = None
            with ctx.lock:  # one check-and-set per run, shared with the tool's steer-limit halt
                if ctx.halt_reason is None:
                    reason = ctx.halt_reason = \
                        f"{f.drift_type} ({level.value}) on {criticality}-criticality workflow {workflow!r}"
                    ctx.halt_drift_id = drift_id
            if reason is not None:
                payload = {"reason": reason, "drift_id": drift_id, "pending": True}
                _safe(lambda: _halt_event(ctx, f.drift_type, payload))
        applied.append(Applied(f, level, action, drift_id))
    return applied


def _notify(drift_id: int) -> None:
    from ..alerts.dispatch import notify

    notify(drift_id)  # queues the alert on the alerts thread pool; never waits for a channel


def _fallback_change(ctx: RunContext, drift_type: str, level: Any) -> tuple[str, Any] | None:
    """(the run's model, the Change) when this finding engages the workflow's fallback, else None."""
    from ..detect.changes import recent_change
    from ..detect.urgency import Level
    from .fallback import FALLBACK_TRIGGERS

    target = ctx.spec.fallback_model
    if level is not Level.P1 or drift_type not in FALLBACK_TRIGGERS or not target:
        return None
    store = get_runtime().store
    row = store.connect().execute("SELECT model FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    if row is None or row["model"] is None or row["model"] == target:  # already on the pinned model
        return None
    change = recent_change(store, ctx.spec.name, ctx.run_id, ignore_model=target)  # pinned runs are no setup
    return (row["model"], change) if change is not None else None


def _engage_fallback(ctx: RunContext, name: str, drift_id: int | None, model: str, change: Any) -> None:
    from dataclasses import asdict

    from . import fallback

    store, target = get_runtime().store, ctx.spec.fallback_model
    if not fallback.activate(store, ctx.spec.name, model=target, drift_id=drift_id, change=change):
        return  # already engaged (an earlier finding, or another process)
    log.warning("pitstop: workflow %r falls back to %r after %s on %r (%s changed)", ctx.spec.name, target, name,
                model, change.field)
    payload = {"from": model, "to": target, "change": asdict(change)}
    with store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (ctx.run_id, time.time(), ctx.agent, "fallback", name, json.dumps(payload, default=str), None))


def _halt_event(ctx: RunContext, name: str, payload: dict) -> None:
    # Its own short transaction: never wraps a ledger call (the ledger refuses to run inside one).
    with get_runtime().store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (ctx.run_id, time.time(), ctx.agent, "halt", name, json.dumps(payload, default=str), None))
