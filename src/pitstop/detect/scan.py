"""The background scan pass (spec 6.2): L2 change detection over finished runs, L5 attribution on each alarm.

`scan` reads finished runs (`ended_at` set) that no scan has read yet (`scanned_at` NULL), oldest end first. Each
run is one `BEGIN IMMEDIATE` transaction: it is claimed (`scanned_at` set, skipped when another scanner got there
first), its `run_metrics` are computed, and each metric feeds its workflow's `Cusum` from `l2_state`, saved back
in the same transaction. So two scanners at once (two processes, or the M3 server and a cron job) feed each run
into one detector state exactly once. A run with no metrics (`refused`, `dropped`, no workflow) is still marked
scanned.

Each alarm is handled after its run's transaction commits: `attribute` names the run tags that changed around the
alarm's change point, the drift is the metric's drift type at `classify`'s level, raised by `escalate_for_change`
when a model, base_url or prompt_versions change explains it, and is recorded with action `record`. P0/P1 drifts
are sent through `pitstop.alerts.dispatch.notify` with this pass's own `store`, so the alert is built from and
recorded in the store that holds the drift (channels and routes still come from the SDK runtime's config).

Observe only: `scan` never raises. A corrupt `l2_state` row restarts that detector (one warning). A run that
fails is logged; a store error (locked, I/O) ends the pass and leaves the run for the next one, any other error
marks it scanned so it cannot block later runs."""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, fields

from .attribution import attribute, explains_change
from .drifts import record_drift
from .l0 import Finding
from .l2 import Alarm, Cusum
from .metrics import METRICS, MetricSpec, run_metrics
from .urgency import Level, classify, escalate_for_change

log = logging.getLogger("pitstop")

DEFAULT_LIMIT = 500
_CRITICALITIES = ("high", "medium", "low")
_SENSITIVITIES = ("low", "normal", "high")


@dataclass(frozen=True)
class ScanAlarm:
    workflow: str
    metric: str
    drift_type: str
    level: str
    drift_id: int | None  # None when the drift could not be recorded
    summary: str  # the L5 attribution summary


@dataclass(frozen=True)
class ScanReport:
    runs: int  # runs this pass claimed (marked scanned)
    alarms: tuple[ScanAlarm, ...]


@dataclass(frozen=True)
class _Pending:
    run_id: str
    workflow: str
    criticality: str
    metric: str
    spec: MetricSpec
    alarm: Alarm


def scan(store, *, limit: int = DEFAULT_LIMIT) -> ScanReport:
    """One pass over up to `limit` finished, unscanned runs (see the module docstring)."""
    runs, alarms = 0, []
    try:
        rows = store.connect().execute(
            "SELECT id FROM runs WHERE ended_at IS NOT NULL AND scanned_at IS NULL ORDER BY ended_at, rowid LIMIT ?",
            (max(limit, 0),)).fetchall()
    except Exception as e:
        log.warning("pitstop: scan could not read the runs to scan (%s)", type(e).__name__)
        return ScanReport(0, ())
    for row in rows:
        try:
            pending = _scan_run(store, row["id"])
        except sqlite3.OperationalError as e:  # locked or unwritable: the run stays unscanned for the next pass
            log.warning("pitstop: scan stopped at run %s (%s)", row["id"], type(e).__name__)
            break
        except Exception as e:
            log.warning("pitstop: scan skipped run %s (%s)", row["id"], type(e).__name__)
            if _mark_scanned(store, row["id"]):
                runs += 1
            continue
        if pending is None:
            continue  # another scanner claimed it
        runs += 1
        alarms.extend(_record(store, p) for p in pending)
    return ScanReport(runs, tuple(alarms))


def _scan_run(store, run_id: str) -> list[_Pending] | None:
    """Claim the run and feed its metrics to its detectors in one transaction; None when already claimed."""
    pending = []
    with store.transaction(immediate=True) as c:
        if c.execute("UPDATE runs SET scanned_at=? WHERE id=? AND scanned_at IS NULL",
                     (time.time(), run_id)).rowcount == 0:
            return None
        run = c.execute("SELECT workflow, criticality, sensitivity, started_at, ended_at FROM runs WHERE id=?",
                        (run_id,)).fetchone()
        workflow = run["workflow"]
        if workflow is None:
            return pending
        criticality = run["criticality"] if run["criticality"] in _CRITICALITIES else "medium"
        sensitivity = run["sensitivity"] if run["sensitivity"] in _SENSITIVITIES else "normal"
        ts = run["started_at"] if run["started_at"] is not None else run["ended_at"]
        for metric, value in run_metrics(store, run_id).items():
            spec = METRICS.get(metric)
            if spec is None:
                continue
            detector = _load(c, workflow, metric)
            alarm = detector.update(value, ts, spec, sensitivity)
            c.execute("INSERT OR REPLACE INTO l2_state(workflow, metric, state, updated_at) VALUES (?,?,?,?)",
                      (workflow, metric, detector.to_json(), time.time()))
            if alarm is not None:
                pending.append(_Pending(run_id, workflow, criticality, metric, spec, alarm))
    return pending


def _load(c, workflow: str, metric: str) -> Cusum:
    """The saved detector, or a fresh one; a row that does not read back as a Cusum is logged and restarted."""
    row = c.execute("SELECT state FROM l2_state WHERE workflow=? AND metric=?", (workflow, metric)).fetchone()
    if row is None:
        return Cusum()
    try:
        detector = Cusum.from_json(row["state"])
        _check(detector)
        return detector
    except Exception as e:
        log.warning("pitstop: unreadable l2_state for %r / %s, restarting its detector (%s)", workflow, metric,
                    type(e).__name__)
        return Cusum()


def _check(d: Cusum) -> None:
    for f in fields(d):
        v = getattr(d, f.name)
        if v is None and f.name.endswith("_since"):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise TypeError(f"{f.name} is not a finite number")
    if not isinstance(d.n, int) or d.n < 0:
        raise ValueError("n is not a count")


def _mark_scanned(store, run_id: str) -> bool:
    try:
        with store.transaction(immediate=True) as c:
            return c.execute("UPDATE runs SET scanned_at=? WHERE id=? AND scanned_at IS NULL",
                             (time.time(), run_id)).rowcount > 0
    except Exception as e:
        log.warning("pitstop: scan could not mark run %s scanned (%s)", run_id, type(e).__name__)
        return False


def _record(store, p: _Pending) -> ScanAlarm:
    """Attribute the alarm, record the drift and queue its alert; a failure is logged (drift_id None, and P2 if
    the level itself could not be classified)."""
    a, summary, drift_id, level = p.alarm, "", None, Level.P2
    try:
        level = classify(p.spec.drift_type, p.criticality, False)
        attribution = attribute(store, p.workflow, a.at, metric=p.metric)
        summary = attribution.summary
        if explains_change(attribution):
            level = escalate_for_change(level, p.criticality)
        finding = Finding(p.spec.drift_type, "degraded", False, {
            "layer": "L2", "metric": p.metric, "direction": a.direction, "value": a.value,
            "baseline_mean": a.baseline_mean, "baseline_std": a.baseline_std, "change_at": a.at,
            "attribution": {"summary": summary, "changes": [asdict(c) for c in attribution.changes]}})
        drift_id = record_drift(store, run_id=p.run_id, workflow=p.workflow, finding=finding, level=level,
                                action="record")
    except Exception as e:
        log.warning("pitstop: scan could not record the %s alarm for %r (%s)", p.metric, p.workflow,
                    type(e).__name__)
    if drift_id is not None and level in (Level.P0, Level.P1):
        try:
            from ..alerts.dispatch import notify

            notify(drift_id, store=store)
        except Exception as e:
            log.warning("pitstop: scan could not queue the alert for drift %s (%s)", drift_id, type(e).__name__)
    return ScanAlarm(p.workflow, p.metric, p.spec.drift_type, level.value, drift_id, summary)

