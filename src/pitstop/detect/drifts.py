"""Drift records in the local store."""
from __future__ import annotations

import json
import time

from .l0 import Finding
from .urgency import Level


def record_drift(store, *, run_id: str | None, workflow: str | None, finding: Finding, level: Level | str,
                 action: str) -> int:
    """Insert an open drift, or append evidence to the open drift with the same (workflow, drift_type)."""
    level_s = level.value if isinstance(level, Level) else str(level)
    with store.transaction(immediate=True) as c:
        row = c.execute(
            "SELECT id, evidence FROM drifts WHERE workflow IS ? AND drift_type=? AND status='open'"
            " ORDER BY id LIMIT 1", (workflow, finding.drift_type)).fetchone()
        if row is not None:
            try:
                ev = json.loads(row["evidence"])
                ev = ev if isinstance(ev, list) else [ev]
            except Exception:
                ev = []
            ev.append(finding.evidence)
            c.execute("UPDATE drifts SET evidence=? WHERE id=?", (json.dumps(ev, default=str), row["id"]))
            return row["id"]
        cur = c.execute(
            "INSERT INTO drifts(run_id, workflow, ts, level, kind, drift_type, evidence, action, status)"
            " VALUES (?,?,?,?,?,?,?,?,'open')",
            (run_id, workflow, time.time(), level_s, finding.kind, finding.drift_type,
             json.dumps([finding.evidence], default=str), action))
        return cur.lastrowid


def list_drifts(store, *, workflow: str | None = None, open_only: bool = True) -> list[dict]:
    sql, params = "SELECT * FROM drifts WHERE 1=1", []
    if workflow is not None:
        sql += " AND workflow=?"
        params.append(workflow)
    if open_only:
        sql += " AND status='open'"
    rows = store.connect().execute(sql + " ORDER BY id", params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["evidence"] = json.loads(d["evidence"]) if d["evidence"] else []
        except Exception:
            pass
        out.append(d)
    return out
