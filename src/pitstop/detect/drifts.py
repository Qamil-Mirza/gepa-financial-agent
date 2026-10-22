"""Drift records in the local store."""
from __future__ import annotations

import json
import time

from .l0 import Finding
from .urgency import Level

MAX_EVIDENCE = 20  # evidence entries kept per drift (newest); `occurrences` keeps the full count
_LEVEL_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}  # lower is more urgent
_ACTION_RANK = {"record": 0, "fallback": 1, "steer": 2, "block": 3, "halt": 4}  # higher is stricter


def _more_urgent(a: str, b: str) -> str:
    return a if _LEVEL_RANK.get(a, len(_LEVEL_RANK)) <= _LEVEL_RANK.get(b, len(_LEVEL_RANK)) else b


def _stricter(a: str, b: str) -> str:
    return a if _ACTION_RANK.get(a, -1) >= _ACTION_RANK.get(b, -1) else b


def record_drift(store, *, run_id: str | None, workflow: str | None, finding: Finding, level: Level | str,
                 action: str) -> int:
    """Insert an open drift, or add an occurrence to the open drift with the same (workflow, drift_type).

    Each occurrence appends an evidence entry `{"run_id", "ts", "detail"}` (detail is the finding's evidence);
    only the newest MAX_EVIDENCE entries are kept, while `occurrences` and `last_seen` track every one.
    On dedup the open drift escalates, never downgrades: it keeps the more urgent level (P0 < P1 < P2 < P3)
    and the stricter action (halt > block > steer > fallback > record)."""
    level_s = level.value if isinstance(level, Level) else str(level)
    now = time.time()
    entry = {"run_id": run_id, "ts": now, "detail": finding.evidence}
    with store.transaction(immediate=True) as c:
        row = c.execute(
            "SELECT id, evidence, level, action FROM drifts WHERE workflow IS ? AND drift_type=? AND status='open'"
            " ORDER BY id LIMIT 1", (workflow, finding.drift_type)).fetchone()
        if row is not None:
            try:
                ev = json.loads(row["evidence"])
                ev = ev if isinstance(ev, list) else [ev]
            except Exception:
                ev = []
            ev = (ev + [entry])[-MAX_EVIDENCE:]
            c.execute("UPDATE drifts SET evidence=?, level=?, action=?, occurrences=occurrences+1, last_seen=?"
                      " WHERE id=?",
                      (json.dumps(ev, default=str), _more_urgent(row["level"], level_s),
                       _stricter(row["action"], action), now, row["id"]))
            return row["id"]
        cur = c.execute(
            "INSERT INTO drifts(run_id, workflow, ts, level, kind, drift_type, evidence, action, status, last_seen,"
            " occurrences) VALUES (?,?,?,?,?,?,?,?,'open',?,1)",
            (run_id, workflow, now, level_s, finding.kind, finding.drift_type,
             json.dumps([entry], default=str), action, now))
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
