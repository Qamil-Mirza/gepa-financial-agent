"""Model and provider changes between runs of a workflow, read from the `runs` table (each run's model and
normalized base_url come from its first agent-meter call). Used by the fallback action and by alerts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Change:
    field: Literal["model", "base_url"]
    before: str | None
    after: str | None
    prev_run_id: str
    at: float  # when the changed run started


def recent_change(store, workflow: str, run_id: str) -> Change | None:
    """How run `run_id` differs from the workflow's previous run: the most recent other run, started no later
    than this one, that has a model. The model is compared first, then the base_url. None when this run has no
    model yet, there is no earlier run, or nothing changed."""
    conn = store.connect()
    cur = conn.execute("SELECT rowid, model, base_url, started_at FROM runs WHERE id=? AND workflow=?",
                       (run_id, workflow)).fetchone()
    if cur is None or cur["model"] is None or cur["started_at"] is None:
        return None
    started = cur["started_at"]
    prev = conn.execute(
        "SELECT id, model, base_url FROM runs WHERE workflow=? AND id<>? AND model IS NOT NULL"
        " AND (started_at < ? OR (started_at = ? AND rowid < ?)) ORDER BY started_at DESC, rowid DESC LIMIT 1",
        (workflow, run_id, started, started, cur["rowid"])).fetchone()
    if prev is None:
        return None
    for field in ("model", "base_url"):
        if prev[field] != cur[field]:
            return Change(field, prev[field], cur[field], prev["id"], started)
    return None
