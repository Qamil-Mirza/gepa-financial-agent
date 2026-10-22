"""Per-workflow fallback state. A P1 breakage or refusal spike right after a model or provider change engages
the workflow's pinned `fallback_model`; from then on its agent-meter calls go to that model until the state is
cleared (`pitstop fallback clear <workflow>`). The state is sticky across runs and processes: it lives in the
store's `meta` table under `fallback:<workflow>` as JSON `{"model", "since", "drift_id", "change"}`."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from ..detect.changes import Change

FALLBACK_TRIGGERS = frozenset({"tool_call_breakage", "refusal_spike"})
_PREFIX = "fallback:"


def _key(workflow: str) -> str:
    return _PREFIX + workflow


def _parse(value: str | None) -> dict | None:
    try:
        state = json.loads(value) if value else None
    except Exception:
        return None
    return state if isinstance(state, dict) and isinstance(state.get("model"), str) else None


def active(store, workflow: str) -> dict | None:
    """The workflow's active fallback state, or None."""
    row = store.connect().execute("SELECT value FROM meta WHERE key=?", (_key(workflow),)).fetchone()
    return _parse(row[0]) if row is not None else None


def activate(store, workflow: str, *, model: str, drift_id: int | None, change: Change) -> bool:
    """Engage the fallback. False (and the existing state kept) when one is already active; an unreadable state
    is replaced."""
    value = json.dumps({"model": model, "since": time.time(), "drift_id": drift_id, "change": asdict(change)})
    with store.transaction(immediate=True) as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (_key(workflow),)).fetchone()
        if row is not None and _parse(row[0]) is not None:
            return False
        c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (_key(workflow), value))
        return True


def clear(store, workflow: str) -> bool:
    """Disengage the fallback. False when none was active."""
    with store.transaction(immediate=True) as c:
        return c.execute("DELETE FROM meta WHERE key=?", (_key(workflow),)).rowcount == 1


def list_active(store) -> list[dict]:
    """Every fallback state, by workflow name: its state plus `"workflow"`. A state that cannot be read (no swap
    happens for it) is listed with `"invalid": True` and the other fields None, so it can be seen and cleared."""
    rows = store.connect().execute("SELECT key, value FROM meta WHERE substr(key, 1, ?)=? ORDER BY key",
                                   (len(_PREFIX), _PREFIX)).fetchall()
    out = []
    for key, value in rows:
        state = _parse(value)
        if state is None:
            state = {"model": None, "since": None, "drift_id": None, "change": None, "invalid": True}
        out.append({"workflow": key[len(_PREFIX):], **state})
    return out
