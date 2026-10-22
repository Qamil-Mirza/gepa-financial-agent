"""Model and provider changes between runs of a workflow, read from the `runs` table (each run's model and
normalized base_url come from its first agent-meter call). Used by the fallback action and by alerts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

CHANGE_WINDOW_S = 86400.0  # a change counts as recent for this long after the first run on the new setup


@dataclass(frozen=True)
class Change:
    field: Literal["model", "base_url"]
    before: str | None
    after: str | None
    prev_run_id: str
    at: float  # started_at of the first run on the current (model, base_url) after the previous setup


def recent_change(store, workflow: str, run_id: str, *, ignore_model: str | None = None,
                  now: float | None = None) -> Change | None:
    """The change of (model, base_url) that run `run_id` is on, if it happened within CHANGE_WINDOW_S.

    Earlier runs of the workflow (started no later than this one) are walked newest first, skipping runs with no
    model and runs on `ignore_model` (the workflow's pinned fallback: its runs are not a setup of their own). The
    first run on a different (model, base_url) is the previous setup; the model is compared first, then the
    base_url. `at` is when the current setup started: the earliest run on it after the previous one. None when
    this run has no model yet, no earlier setup differs, or the change is older than the window."""
    now = time.time() if now is None else now
    conn = store.connect()
    cur = conn.execute("SELECT rowid, model, base_url, started_at FROM runs WHERE id=? AND workflow=?",
                       (run_id, workflow)).fetchone()
    if cur is None or cur["model"] is None or cur["started_at"] is None:
        return None
    at = cur["started_at"]
    earlier = conn.execute(
        "SELECT id, model, base_url, started_at FROM runs WHERE workflow=? AND id<>? AND model IS NOT NULL"
        " AND (started_at < ? OR (started_at = ? AND rowid < ?)) ORDER BY started_at DESC, rowid DESC",
        (workflow, run_id, at, at, cur["rowid"]))
    for prev in earlier:  # a cursor: stops reading at the first decisive row
        if now - at > CHANGE_WINDOW_S:
            return None  # `at` only moves back in time from here
        if ignore_model is not None and prev["model"] == ignore_model:
            continue
        if (prev["model"], prev["base_url"]) == (cur["model"], cur["base_url"]):
            at = prev["started_at"]  # an earlier run on the current setup: it started no later than this
            continue
        if now - at > CHANGE_WINDOW_S:
            return None
        field = "model" if prev["model"] != cur["model"] else "base_url"
        return Change(field, prev[field], cur[field], prev["id"], at)
    return None
