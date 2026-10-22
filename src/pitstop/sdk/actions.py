"""Turn L0 findings into actions: classify each one (level, action), record it as a drift, and set the pending
halt that a P0 on a high-criticality workflow calls for. Shared by `@pitstop.tool` and the wrapped clients."""
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
    """Classify each finding (level, action), then record it. A P0 on a high workflow sets a pending halt that
    the next boundary raises, and records a `halt` event marked `pending`. Outside a workflow: workflow=None,
    criticality medium, sensitivity normal.

    The decision never depends on the store: a failed drift or event write is logged, leaves drift_id None,
    and the block/steer/halt still applies."""
    from ..detect import drifts
    from ..detect.urgency import Level, action_for, classify

    criticality = ctx.spec.criticality if ctx is not None else "medium"
    workflow = ctx.spec.name if ctx is not None else None
    run_id = ctx.run_id if ctx is not None else run_id
    applied: list[Applied] = []
    for f in findings:
        level = classify(f.drift_type, criticality, f.side_effects)
        action = action_for(f, level, criticality)
        try:
            drift_id = drifts.record_drift(get_runtime().store, run_id=run_id, workflow=workflow, finding=f,
                                           level=level, action=action)
        except Exception:
            log.warning("pitstop: could not record a %s drift; its action still applies", f.drift_type,
                        exc_info=True)
            drift_id = None
        if ctx is not None and level is Level.P0 and criticality == "high":
            reason = None
            with ctx.lock:  # one check-and-set per run, shared with the tool's steer-limit halt
                if ctx.halt_reason is None:
                    reason = ctx.halt_reason = f"{f.drift_type} (P0) on high-criticality workflow {workflow!r}"
                    ctx.halt_drift_id = drift_id
            if reason is not None:
                payload = {"reason": reason, "drift_id": drift_id, "pending": True}
                _safe(lambda: _halt_event(ctx, f.drift_type, payload))
        applied.append(Applied(f, level, action, drift_id))
    return applied


def _halt_event(ctx: RunContext, name: str, payload: dict) -> None:
    # Its own short transaction: never wraps a ledger call (the ledger refuses to run inside one).
    with get_runtime().store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (ctx.run_id, time.time(), ctx.agent, "halt", name, json.dumps(payload, default=str), None))
