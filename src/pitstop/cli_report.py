"""Pure row builders for `pitstop status`, `pitstop drifts` and `pitstop budget` (the CLI renders them)."""
from __future__ import annotations

from datetime import datetime

from .detect.drifts import list_drifts

_LEVELS = ("P0", "P1", "P2", "P3")
_LIVE = ("reserved", "settled")


def _day_start(now: float) -> float:
    return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def workflow_rows(store, now: float) -> list[dict]:
    conn = store.connect()
    start = _day_start(now)
    names = [r[0] for r in conn.execute(
        "SELECT workflow FROM runs WHERE workflow IS NOT NULL GROUP BY workflow ORDER BY MAX(started_at) DESC")]
    rows = []
    for wf in names:
        latest = conn.execute("SELECT criticality, status FROM runs WHERE workflow=? ORDER BY started_at DESC, rowid DESC"
                              " LIMIT 1", (wf,)).fetchone()
        runs_today = conn.execute("SELECT COUNT(*) FROM runs WHERE workflow=? AND started_at>=?", (wf, start)).fetchone()[0]
        spend = conn.execute(
            "SELECT COALESCE(SUM(usd),0) FROM ledger WHERE workflow=? AND meter='agent' AND state IN (?,?) AND ts>=?",
            (wf, *_LIVE, start)).fetchone()[0]
        counts = {lv: 0 for lv in _LEVELS}
        for d in list_drifts(store, workflow=wf, open_only=True):
            if d["level"] in counts:
                counts[d["level"]] += 1
        rows.append({"workflow": wf, "criticality": latest["criticality"], "runs_today": runs_today,
                     "open_p0": counts["P0"], "open_p1": counts["P1"], "open_p2": counts["P2"],
                     "open_p3": counts["P3"], "spend_today_usd": float(spend),
                     "budget_usd": None,  # per-workflow budget_usd is not persisted with the run
                     "last_status": latest["status"]})
    return rows


def budget_rows(ledger, config) -> list[dict]:
    rows = []
    for b in config.budgets:
        spent = ledger.spent(f"global:{b.name}")
        rows.append({"scope": f"global:{b.name}", "period": b.period, "spent_usd": spent,
                     "limit_usd": b.limit_usd, "remaining_usd": max(b.limit_usd - spent, 0.0)})
    return rows


def drift_rows(store, workflow: str | None = None, open_only: bool = True) -> list[dict]:
    drifts = list_drifts(store, workflow=workflow, open_only=open_only)
    drifts.sort(key=lambda d: -d["ts"])  # stable: newest first within a level
    drifts.sort(key=lambda d: _LEVELS.index(d["level"]) if d["level"] in _LEVELS else len(_LEVELS))
    return [{"id": d["id"], "level": d["level"], "workflow": d["workflow"], "drift_type": d["drift_type"],
             "action": d["action"], "count": len(d["evidence"]) if isinstance(d["evidence"], list) else 1,
             "first_seen": d["ts"], "last_seen": d["ts"]} for d in drifts]
