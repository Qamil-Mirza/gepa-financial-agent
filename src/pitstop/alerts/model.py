"""The alert message: built from a drift row, the run it was last seen in and the recent model/provider change (or
the drift's L5 attribution)."""
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
        workflow = self.workflow if self.workflow is not None else "(no workflow)"
        return f"[{self.level}] {workflow}: {self.drift_type}"

    @property
    def body(self) -> str:
        return f"What drifted: {self.what}\nWhat likely changed: {self.cause}\nWhat PitStop did: {self.action}"

    @property
    def dedup_key(self) -> str:
        workflow = self.workflow if self.workflow is not None else "-"  # a drift recorded outside any workflow
        return f"pitstop:{workflow}:{self.drift_type}"


def build_alert(store, drift_id: int) -> Alert:
    """The alert for drift `drift_id` at its current (possibly escalated) level. The run is the drift's latest
    occurrence. The cause is the newest evidence entry's L5 attribution summary (`detail.attribution.summary`, an
    L2 drift recorded by `pitstop scan`) when it has one, else that run's model or provider change
    (`recent_change`) or, without one, the run's model and base_url. KeyError when the drift does not exist."""
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
    if run is not None and run["status"] == "refused":  # a budget used up before the run began
        action = "Refused the run at its start (no model call was made)."
    if d["level"] == "P0" and run is not None and run["criticality"] == "high" and d["action"] != "halt":
        action += " The run halts at its next call boundary (high-criticality workflow)."
    return Alert(drift_id=d["id"], workflow=d["workflow"], level=d["level"], drift_type=d["drift_type"], what=what,
                 cause=_attributed_cause(latest) or _cause(store, d["workflow"], run), action=action, run_id=run_id,
                 ts=d["last_seen"] if d["last_seen"] is not None else d["ts"])


def build_unrecorded_alert(store, *, workflow: str | None, run_id: str | None, drift_type: str, level: str,
                           evidence: dict, ts: float) -> Alert:
    """The alert for a drift whose write failed (`dispatch.notify_unrecorded`): no drift row to read, so the text
    comes from the finding. The cause is read from the run when the store allows, else it says so."""
    what = f"{drift_type}; the drift could not be recorded in PitStop's store (not deduplicated)"
    if evidence:
        what += f"; evidence: {_one_line(evidence)}"
    try:
        run = store.connect().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone() if run_id else None
        cause = _cause(store, workflow, run) if run is not None else "Unknown: the drift was seen outside a run."
    except Exception as e:
        cause = f"Unknown: the store could not be read ({type(e).__name__})."
    action = ("Applied the drift's in-the-moment action (block, steer or halt) as usual; the drift is missing "
              "from `pitstop drifts` (see the agent's log).")
    return Alert(drift_id=None, workflow=workflow, level=level, drift_type=drift_type, what=what, cause=cause,
                 action=action, run_id=run_id, ts=ts)


def _attributed_cause(entry: dict) -> str | None:
    """The L5 attribution summary of an evidence entry as a sentence, or None when it has none."""
    detail = entry.get("detail")
    attribution = detail.get("attribution") if isinstance(detail, dict) else None
    summary = attribution.get("summary") if isinstance(attribution, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        return None
    summary = " ".join(summary.split())[:300]
    return summary[0].upper() + summary[1:] + ("" if summary.endswith(".") else ".")


def _cause(store, workflow: str | None, run) -> str:
    from ..detect.changes import recent_change

    if run is None:
        return "Unknown: the drift was recorded outside a run."
    if workflow is not None:
        # Runs redirected to a pinned fallback are skipped by recent_change itself (runs.fallback_from), so the
        # cause is the same whether or not a fallback is active now.
        change = recent_change(store, workflow, run["id"])
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
