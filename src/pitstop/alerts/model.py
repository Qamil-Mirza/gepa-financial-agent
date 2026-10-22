"""The alert message: built from a drift row, the run it was last seen in and the recent model/provider change."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

_ACTIONS = {
    "halt": "Halted the run at its next call boundary.",
    "block": "Blocked the side-effect tool call; the agent got a blocked result instead.",
    "steer": "Skipped the step and steered the agent back to its goal.",
    "fallback": "Switched the workflow's agent calls to its pinned fallback model (`pitstop fallback status`).",
    "record": "Recorded the drift; no in-the-moment action.",
}


@dataclass(frozen=True)
class Alert:
    drift_id: int | None
    workflow: str | None
    level: str
    drift_type: str
    what: str  # what drifted
    cause: str  # what likely changed
    action: str  # what is being done
    run_id: str | None
    ts: float

    @property
    def title(self) -> str:
        return f"[{self.level}] {self.workflow}: {self.drift_type}"

    @property
    def body(self) -> str:
        return f"What drifted: {self.what}\nWhat likely changed: {self.cause}\nWhat PitStop did: {self.action}"

    @property
    def dedup_key(self) -> str:
        return f"pitstop:{self.workflow}:{self.drift_type}"


def build_alert(store, drift_id: int) -> Alert:
    """The alert for drift `drift_id` at its current (possibly escalated) level. The run is the drift's latest
    occurrence; the cause is that run's model or provider change (`recent_change`) or, without one, the run's
    model and base_url. KeyError when the drift does not exist."""
    conn = store.connect()
    d = conn.execute("SELECT * FROM drifts WHERE id=?", (drift_id,)).fetchone()
    if d is None:
        raise KeyError(f"drift {drift_id} not found")
    latest = _latest_evidence(d["evidence"])
    run_id = latest.get("run_id") or d["run_id"]
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone() if run_id else None
    what = f"{d['drift_type']} ({d['kind']}), seen {d['occurrences']} time(s)"
    if latest.get("detail"):
        what += f"; latest evidence: {_one_line(latest['detail'])}"
    action = _ACTIONS.get(d["action"], f"Action: {d['action']}.")
    if d["level"] == "P0" and run is not None and run["criticality"] == "high" and d["action"] != "halt":
        action += " The run halts at its next call boundary (high-criticality workflow)."
    return Alert(drift_id=d["id"], workflow=d["workflow"], level=d["level"], drift_type=d["drift_type"], what=what,
                 cause=_cause(store, d["workflow"], run), action=action, run_id=run_id,
                 ts=d["last_seen"] if d["last_seen"] is not None else d["ts"])


def _cause(store, workflow: str | None, run) -> str:
    from ..detect.changes import recent_change
    from ..sdk import fallback

    if run is None:
        return "Unknown: the drift was recorded outside a run."
    if workflow is not None:
        # Runs during an active fallback record the pinned model; they are not a setup of their own.
        state = fallback.active(store, workflow)
        change = recent_change(store, workflow, run["id"], ignore_model=state["model"] if state else None)
        if change is not None:
            at = datetime.fromtimestamp(change.at, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            what = "model" if change.field == "model" else "provider base_url"
            return f"The {what} changed from {change.before} to {change.after} (first run on it at {at})."
    return (f"No model or provider change detected for this run (model {run['model'] or 'unknown'} at base_url "
            f"{run['base_url'] or 'unknown'}).")


def _latest_evidence(raw: str | None) -> dict:
    try:
        ev = json.loads(raw) if raw else []
    except Exception:
        return {}
    ev = ev if isinstance(ev, list) else [ev]
    last = ev[-1] if ev else None
    return last if isinstance(last, dict) else {}


def _one_line(value: object) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        text = repr(value)
    return " ".join(text.split())[:300]
