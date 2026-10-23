"""Pure row builders for `pitstop status`, `pitstop drifts`, `pitstop budget` and `pitstop explain` (the CLI
renders them)."""
from __future__ import annotations

import json
from datetime import datetime, tzinfo

from .detect.drifts import list_drifts

_LEVELS = ("P0", "P1", "P2", "P3")
_LIVE = ("reserved", "settled")


def _day_start(now: float, tz: tzinfo | None = None) -> float:
    """Start of `now`'s day in `tz` (the config's timezone, as the ledger uses); system local time when None."""
    return datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def workflow_rows(store, now: float, tz: tzinfo | None = None) -> list[dict]:
    start = _day_start(now, tz)
    return [row for wf in workflow_names(store) if (row := workflow_row(store, wf, start)) is not None]


def workflow_names(store) -> list[str]:
    """Every workflow that has a run, most recently started first."""
    return [r[0] for r in store.connect().execute(
        "SELECT workflow FROM runs WHERE workflow IS NOT NULL GROUP BY workflow ORDER BY MAX(started_at) DESC")]


def workflow_row(store, wf: str, day_start: float) -> dict | None:
    """One `pitstop status` row (also the dashboard tile's counts); None when `wf` has no run. `day_start` is the
    start of today (`_day_start`): runs today and today's agent-meter spend count from it."""
    conn = store.connect()
    latest = conn.execute(
        "SELECT criticality, status, budget_usd FROM runs WHERE workflow=? ORDER BY started_at DESC, rowid DESC"
        " LIMIT 1",
        (wf,),
    ).fetchone()
    if latest is None:
        return None
    runs_today = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE workflow=? AND started_at>=?", (wf, day_start)
    ).fetchone()[0]
    spend = conn.execute(
        "SELECT COALESCE(SUM(usd),0) FROM ledger WHERE workflow=? AND meter='agent' AND state IN (?,?) AND ts>=?",
        (wf, *_LIVE, day_start)).fetchone()[0]
    counts = {lv: 0 for lv in _LEVELS}
    for d in list_drifts(store, workflow=wf, open_only=True):
        if d["level"] in counts:
            counts[d["level"]] += 1
    return {"workflow": wf, "criticality": latest["criticality"], "runs_today": runs_today,
            "open_p0": counts["P0"], "open_p1": counts["P1"], "open_p2": counts["P2"],
            "open_p3": counts["P3"], "spend_today_usd": float(spend),
            "budget_usd": latest["budget_usd"],
            "last_status": latest["status"]}


def budget_rows(ledger, config) -> list[dict]:
    rows = []
    for b in config.budgets:
        spent = ledger.spent(f"global:{b.name}")
        rows.append({"scope": f"global:{b.name}", "period": b.period, "spent_usd": spent,
                     "limit_usd": b.limit_usd, "remaining_usd": max(b.limit_usd - spent, 0.0)})
    return rows


def drift_rows(store, workflow: str | None = None, open_only: bool = True) -> list[dict]:
    """The action reads "fallback (cleared)" when it was a fallback and the workflow has no active one now."""
    drifts = list_drifts(store, workflow=workflow, open_only=open_only)
    drifts.sort(key=lambda d: -_sort_ts(_last_seen(d)))  # stable: most recently seen first within a level
    drifts.sort(key=lambda d: _LEVELS.index(d["level"]) if d["level"] in _LEVELS else len(_LEVELS))
    return [{"id": d["id"], "level": d["level"], "workflow": d["workflow"], "drift_type": d["drift_type"],
             "action": _action(store, d), "count": d["occurrences"],
             "first_seen": d["ts"], "last_seen": _last_seen(d)} for d in drifts]


def _action(store, d: dict) -> str:
    from .sdk import fallback

    if d["action"] == "fallback" and (d["workflow"] is None or fallback.active(store, d["workflow"]) is None):
        return "fallback (cleared)"
    return d["action"]


def _last_seen(d: dict) -> float:
    return d["last_seen"] if d.get("last_seen") is not None else d["ts"]


def _sort_ts(v: object) -> float:
    """A time for sorting: a malformed (missing or non-numeric) time sorts as the oldest."""
    return float(v) if _is_number(v) else float("-inf")


def explain_target(store, drift_id: int) -> dict | None:
    """What `pitstop explain` attributes for a drift: `{"workflow", "at", "metric"}`, None for an unknown id.

    `at` is the newest evidence entry's `detail.change_at` when it is a number (L2 drifts), else the drift's `ts`
    (None when that is not a number either: a malformed row); `metric` is that entry's `detail.metric` when it is a
    string."""
    d = store.connect().execute("SELECT workflow, ts, evidence FROM drifts WHERE id = ?", (drift_id,)).fetchone()
    if d is None:
        return None
    try:
        evidence = json.loads(d["evidence"]) if d["evidence"] else []
    except ValueError:
        evidence = []
    evidence = evidence if isinstance(evidence, list) else []
    detail = evidence[-1].get("detail") if evidence and isinstance(evidence[-1], dict) else None
    detail = detail if isinstance(detail, dict) else {}
    change_at, metric = detail.get("change_at"), detail.get("metric")
    at = change_at if _is_number(change_at) else d["ts"] if _is_number(d["ts"]) else None
    return {"workflow": d["workflow"], "at": float(at) if at is not None else None,
            "metric": metric if isinstance(metric, str) else None}


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
