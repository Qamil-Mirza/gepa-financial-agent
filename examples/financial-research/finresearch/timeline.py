"""Run Timeline: the curated, UI-facing log of what the supervisor did.

events.jsonl and decisions.jsonl record everything. The timeline records only
what a human should look at: drift flags, what happened in response, verdict
changes and validator corrections. Routine ON_TRACK verdicts are not logged one
by one; they are folded into a `quiet_streak` count on the agent's next entry
and kept in compact form in the per-agent verdict history of `timeline.json`.
Every entry carries `refs` back into the raw logs, so the UI can load full
detail (the complete SupervisorInput, raw model output) on demand.

The same builder runs live inside the Swarm and offline over a finished run
directory (`build_from_run_dir`), so old runs can be re-published.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import Status, utc_now

TIMELINE_VERSION = "1.1"
Severity = Literal["critical", "warning", "info"]
SourceRole = Literal["system", "supervisor", "validator", "agent"]
MAX_DETAIL_CHARS = 600


class TimelineKind(str, Enum):
    RUN_STARTED = "run_started"
    DRIFT_FLAGGED = "drift_flagged"            # alert policy fired: the intervention point
    DRIFT_SUSPECTED = "drift_suspected"        # DRIFTING verdict that did not (or no longer) fires
    DRIFT_RESPONSE = "drift_response"          # what happened after a flag: halted / recovered / persisted
    STATUS_CHANGED = "status_changed"          # verdict changed, e.g. ON_TRACK -> UNCERTAIN
    SUPERVISOR_CORRECTION = "supervisor_correction"  # validator dropped evidence or downgraded a verdict
    AGENT_CORRECTED = "agent_corrected"        # supervisor blocked a drifting query and told the agent to return
    REPORT_WRITTEN = "report_written"          # the report agent wrote the team's final report
    AGENT_FINISHED = "agent_finished"
    RUN_FINISHED = "run_finished"


class EvidenceRef(BaseModel):
    event_id: str
    type: Optional[str] = None
    step: Optional[int] = None
    summary: str = ""


class Source(BaseModel):
    """Who emitted the entry. `agent_id` on the entry is who it is *about*."""
    role: SourceRole
    id: str
    label: str


class TimelineEntry(BaseModel):
    entry_id: str
    seq: int
    kind: TimelineKind
    severity: Severity
    timestamp: str
    title: str
    source: Source
    detail: str = ""
    agent_id: Optional[str] = None
    agent_label: Optional[str] = None
    step: Optional[int] = None
    phase: Optional[str] = None
    status: Optional[Status] = None
    confidence: Optional[float] = None
    drift_type: Optional[str] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    responds_to: Optional[str] = None   # entry_id of the drift_flagged entry this answers
    quiet_streak: int = 0               # uneventful ON_TRACK verdicts folded in since this agent's last entry
    refs: dict[str, Any] = Field(default_factory=dict)  # decision_index / event_ids into the raw logs
    data: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _AgentTrack:
    last_status: Optional[Status] = None
    quiet: int = 0
    open_flag: Optional[str] = None     # drift_flagged entry awaiting its response
    flags: int = 0
    suspected: int = 0
    halted: bool = False
    verdicts: list[dict] = field(default_factory=list)


class TimelineBuilder:
    def __init__(self, log_path: Optional[Path] = None):
        self._log_path = log_path
        self._listeners: list[Callable[[TimelineEntry], None]] = []
        self._agents: dict[str, _AgentTrack] = {}
        self._labels: dict[str, str] = {}
        self.entries: list[TimelineEntry] = []
        self.run: dict[str, Any] = {}
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, fn: Callable[[TimelineEntry], None]) -> None:
        self._listeners.append(fn)

    def _track(self, agent_id: str) -> _AgentTrack:
        return self._agents.setdefault(agent_id, _AgentTrack())

    def label(self, agent_id: str) -> str:
        return self._labels.get(agent_id) or agent_label(agent_id)

    def _source(self, role: SourceRole, agent_id: Optional[str] = None) -> Source:
        if role == "agent" and agent_id:
            return Source(role=role, id=agent_id, label=self.label(agent_id))
        if role == "supervisor":
            return Source(role=role, id=self.run.get("supervisor", {}).get("model") or "supervisor", label="Supervisor")
        if role == "validator":
            return Source(role=role, id="validator", label="Validator")
        return Source(role="system", id="orchestrator", label="Orchestrator")

    def _emit(self, kind: TimelineKind, severity: Severity, title: str, *, source: SourceRole,
              timestamp: Optional[str] = None, **kw: Any) -> TimelineEntry:
        agent_id = kw.get("agent_id")
        if agent_id:
            tr = self._track(agent_id)
            kw["quiet_streak"], tr.quiet = tr.quiet, 0
            kw["agent_label"] = self.label(agent_id)
        kw["source"] = self._source(source, agent_id)
        seq = len(self.entries)
        entry = TimelineEntry(entry_id=f"t{seq:04d}", seq=seq, kind=kind, severity=severity,
                              timestamp=timestamp or utc_now(), title=title, **kw)
        self.entries.append(entry)
        if self._log_path:
            with self._log_path.open("a") as f:
                f.write(entry.model_dump_json() + "\n")
        for fn in self._listeners:
            fn(entry)
        return entry

    # ---------------- lifecycle ----------------
    def run_started(self, run_id: str, scenario: dict, *, supervisor: Optional[dict] = None,
                    halt_on_drift: bool = False, timestamp: Optional[str] = None) -> TimelineEntry:
        self.run = {"run_id": run_id, "task_id": scenario.get("task_id"),
                    "objective": scenario.get("swarm_objective"),
                    "agents": [{"agent_id": a["agent_id"], "label": a.get("label") or agent_label(a["agent_id"]),
                                "color_index": i, "directive": a["directive"]}
                               for i, a in enumerate(scenario.get("agents", []))],
                    "supervisor": supervisor or {}, "halt_on_drift": halt_on_drift}
        for a in self.run["agents"]:
            self._track(a["agent_id"])
            self._labels[a["agent_id"]] = a["label"]
        return self._emit(TimelineKind.RUN_STARTED, "info", f"Run {run_id} started", source="system",
                          detail=self.run["objective"] or "", timestamp=timestamp,
                          data={"agents": [a["agent_id"] for a in self.run["agents"]],
                                "halt_on_drift": halt_on_drift})

    def on_decision(self, record: dict, decision_index: int) -> list[TimelineEntry]:
        """Feed one decisions.jsonl record. Returns the entries it produced (often none)."""
        agent_id, step, phase = record["agent_id"], record["step"], record["phase"]
        inp, res = record["input"], record["result"]
        d = res["decision"]
        status = Status(d["status"])
        tr = self._track(agent_id)
        ts = inp.get("timestamp")
        refs = {"decision_index": decision_index}
        base = dict(agent_id=agent_id, step=step, phase=phase, status=status, confidence=d["confidence"],
                    drift_type=d.get("drift_type"), timestamp=ts, refs=refs)
        tr.verdicts.append({"step": step, "phase": phase, "status": status.value,
                            "confidence": d["confidence"], "alert_level": record.get("alert_level"),
                            "decision_index": decision_index})
        out: list[TimelineEntry] = []

        if res.get("dropped_evidence_ids") or res.get("downgraded"):
            what = []
            if res.get("dropped_evidence_ids"):
                what.append(f"dropped unknown evidence ids {res['dropped_evidence_ids']}")
            if res.get("downgraded"):
                what.append("downgraded DRIFTING to UNCERTAIN (no valid evidence left)")
            out.append(self._emit(TimelineKind.SUPERVISOR_CORRECTION, "info",
                                  f"Validator corrected the supervisor on {self.label(agent_id)}", source="validator",
                                  detail="; ".join(what) + ".",
                                  data={"issues": res.get("issues", []),
                                        "dropped_evidence_ids": res.get("dropped_evidence_ids", []),
                                        "downgraded": bool(res.get("downgraded"))}, **base))

        # A verdict after an open flag is the agent's response to it.
        if tr.open_flag and record.get("alert_note") != "INTERVENTION POINT":
            outcome = {Status.ON_TRACK: "recovered", Status.DRIFTING: "persisted",
                       Status.UNCERTAIN: "unclear"}[status]
            sev: Severity = "warning" if status == Status.DRIFTING else "info"
            out.append(self._emit(TimelineKind.DRIFT_RESPONSE, sev,
                                  f"{self.label(agent_id)} {outcome} after drift flag", source="supervisor",
                                  detail=_clip(d.get("explanation", "")), responds_to=tr.open_flag,
                                  evidence=_evidence(d, inp), data={"outcome": outcome},
                                  **base))
            tr.open_flag = None

        if record.get("alert_note") == "INTERVENTION POINT":
            tr.flags += 1
            proposal = inp.get("proposed_next_action") or {}
            entry = self._emit(TimelineKind.DRIFT_FLAGGED, "critical",
                               f"Drift flagged on {self.label(agent_id)}: {d.get('drift_type', 'other')}",
                               source="supervisor",
                               detail=_clip(d.get("explanation", "")), evidence=_evidence(d, inp),
                               data={"blocked_query": proposal.get("query"),
                                     "proposed_rationale": proposal.get("rationale"),
                                     "drift_injection_active": record.get("drift_injection_active")},
                               **{**base, "refs": {**refs, "event_ids": d.get("evidence_event_ids", [])}})
            tr.open_flag = entry.entry_id
            out.append(entry)
        elif status == Status.DRIFTING:
            tr.suspected += 1
            already = record.get("alert_note") == "already flagged"
            out.append(self._emit(TimelineKind.DRIFT_SUSPECTED, "warning",
                                  f"{self.label(agent_id)} judged DRIFTING" + (" (already flagged)" if already else
                                                                   " (below alert threshold)"),
                                  detail=_clip(d.get("explanation", "")), evidence=_evidence(d, inp),
                                  source="supervisor",
                                  data={"alert_level": record.get("alert_level"),
                                        "alert_note": record.get("alert_note")},
                                  **{**base, "refs": {**refs, "event_ids": d.get("evidence_event_ids", [])}}))
        elif status != (tr.last_status or Status.ON_TRACK) and not any(
                e.kind == TimelineKind.DRIFT_RESPONSE for e in out):
            prev = (tr.last_status or Status.ON_TRACK).value
            out.append(self._emit(TimelineKind.STATUS_CHANGED, "info",
                                  f"{self.label(agent_id)}: {prev} → {status.value}", source="supervisor",
                                  detail=_clip(d.get("explanation", "")), evidence=_evidence(d, inp),
                                  data={"from": prev, "to": status.value}, **base))
        elif not out:
            tr.quiet += 1

        tr.last_status = status
        return out

    def agent_halted(self, agent_id: str, step: int, blocked_query: Optional[str] = None,
                     timestamp: Optional[str] = None) -> TimelineEntry:
        tr = self._track(agent_id)
        tr.halted = True
        flag, tr.open_flag = tr.open_flag, None
        return self._emit(TimelineKind.DRIFT_RESPONSE, "critical",
                          f"{self.label(agent_id)} halted before executing its next query", source="system",
                          agent_id=agent_id, step=step, responds_to=flag, timestamp=timestamp,
                          detail=f"Blocked query: {blocked_query!r}" if blocked_query else "",
                          data={"outcome": "halted", "blocked_query": blocked_query})

    def agent_corrected(self, agent_id: str, step: int, message: str, attempt: int,
                        blocked_query: Optional[str] = None, timestamp: Optional[str] = None) -> TimelineEntry:
        """The open flag stays open: the agent's next verdict is its response (recovered / persisted)."""
        tr = self._track(agent_id)
        return self._emit(TimelineKind.AGENT_CORRECTED, "warning",
                          f"Supervisor corrected {self.label(agent_id)} (#{attempt})", source="supervisor",
                          agent_id=agent_id, step=step, responds_to=tr.open_flag, timestamp=timestamp,
                          detail=_clip(message), data={"attempt": attempt, "blocked_query": blocked_query})

    def report_written(self, report: dict, timestamp: Optional[str] = None) -> TimelineEntry:
        problems = bool(report.get("error") or report.get("invalid_citations") or report.get("unverified_figures"))
        return self._emit(
            TimelineKind.REPORT_WRITTEN,
            "warning" if problems else "info",
            f"Report written: {report.get('title')}",
            source="agent",
            timestamp=timestamp,
            detail=_clip(report.get("error") or f"{len(report.get('cited_finding_ids', []))} findings cited"),
            data=report,
        )

    def agent_finished(self, agent_id: str, step: int, report: Optional[str] = None,
                       timestamp: Optional[str] = None) -> TimelineEntry:
        tr = self._track(agent_id)
        sev: Severity = "warning" if tr.flags else "info"
        return self._emit(TimelineKind.AGENT_FINISHED, sev, f"{self.label(agent_id)} finished", source="agent",
                          agent_id=agent_id, step=step, detail=_clip(report or ""), timestamp=timestamp,
                          status=tr.last_status, data={"flags": tr.flags, "suspected": tr.suspected})

    def run_finished(self, *, interventions: Optional[dict] = None, ground_truth: Optional[dict] = None,
                     timestamp: Optional[str] = None, out_path: Optional[Path] = None) -> dict:
        flagged = [a for a, t in self._agents.items() if t.flags]
        self._emit(TimelineKind.RUN_FINISHED, "warning" if flagged else "info",
                   f"Run finished: {len(flagged)} agent(s) flagged for drift", source="system",
                   timestamp=timestamp, data={"flagged_agents": flagged})
        doc = self.document(interventions=interventions, ground_truth=ground_truth)
        if out_path:
            out_path.write_text(json.dumps(doc, indent=2))
        return doc

    # ---------------- export ----------------
    def document(self, *, interventions: Optional[dict] = None, ground_truth: Optional[dict] = None) -> dict:
        """The whole timeline as one JSON document for the UI."""
        agents = {}
        for agent_id, t in self._agents.items():
            agents[agent_id] = {
                "label": self.label(agent_id),
                "final_status": t.last_status.value if t.last_status else None,
                "flags": t.flags, "suspected": t.suspected, "halted": t.halted,
                "intervention": (interventions or {}).get(agent_id),
                "ground_truth": (ground_truth or {}).get(agent_id),
                "verdicts": t.verdicts,
            }
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return {"timeline_version": TIMELINE_VERSION, "run": self.run, "agents": agents,
                "counts": counts, "entries": [e.model_dump(mode="json") for e in self.entries]}


def agent_label(agent_id: str) -> str:
    """agent_nvda_dc -> 'NVDA DC', agent_supply -> 'Supply'."""
    words = [w for w in agent_id.removeprefix("agent_").replace("-", "_").split("_") if w] or [agent_id]
    return " ".join(w.upper() if len(w) <= 4 else w.capitalize() for w in words)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_DETAIL_CHARS else text[:MAX_DETAIL_CHARS - 1] + "…"


def _evidence(decision: dict, inp: dict) -> list[EvidenceRef]:
    """Resolve cited event ids against the trajectory the supervisor saw."""
    by_id = {e["event_id"]: e for e in inp.get("trajectory", [])}
    refs = []
    for eid in decision.get("evidence_event_ids", []):
        ev = by_id.get(eid)
        refs.append(EvidenceRef(event_id=eid, type=ev["type"], step=ev["step"], summary=ev["summary"])
                    if ev else EvidenceRef(event_id=eid))
    return refs


def build_from_run_dir(run_dir: Path, *, write: bool = True, decisions_name: str = "decisions.jsonl",
                       out_stem: str = "timeline", supervisor_key: str = "supervisor") -> dict:
    """Rebuild the timeline of a finished run from its raw logs. The shadow supervisor's timeline is
    decisions_name="decisions_shadow.jsonl", out_stem="timeline_shadow", supervisor_key="shadow_supervisor"."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    dec_path = run_dir / decisions_name
    records = [json.loads(l) for l in dec_path.read_text().splitlines() if l.strip()] if dec_path.exists() else []
    log_path = run_dir / f"{out_stem}.jsonl" if write else None
    if log_path and log_path.exists():
        log_path.unlink()
    tb = TimelineBuilder(log_path)
    scenario = meta.get("scenario") or {"task_id": None, "swarm_objective": None, "agents": []}
    sup = meta.get(supervisor_key) or {}
    tb.run_started(meta.get("run_id", run_dir.name), scenario,
                   supervisor={"model": sup.get("model"), "prompt_version": sup.get("prompt_version")},
                   halt_on_drift=bool(meta.get("halt_on_drift")),
                   timestamp=records[0]["input"].get("timestamp") if records else None)
    events_path = run_dir / "events.jsonl"
    corrections = {}  # (agent_id, step) -> supervisor_message events, in order
    for line in (events_path.read_text().splitlines() if events_path.exists() else []):
        ev = json.loads(line) if line.strip() else {}
        if ev.get("type") == "supervisor_message":
            corrections.setdefault((ev["agent_id"], ev["step"]), []).append(ev)
    for i, r in enumerate(records):
        tb.on_decision(r, i)
        ts = r["input"].get("timestamp")
        iv = (meta.get("interventions") or {}).get(r["agent_id"]) or {}
        pending = corrections.get((r["agent_id"], r["step"])) or []
        if r.get("alert_note") == "INTERVENTION POINT" and pending:
            ev = pending.pop(0)
            tb.agent_corrected(r["agent_id"], r["step"], ev["payload"].get("message", ""),
                               ev["payload"].get("attempt", 1), ev["payload"].get("blocked_query"),
                               timestamp=ev.get("timestamp"))
        elif r.get("alert_note") == "INTERVENTION POINT" and meta.get("halt_on_drift") and (
                "halted_step" not in iv or iv["halted_step"] == r["step"]):
            tb.agent_halted(r["agent_id"], r["step"], (r["input"].get("proposed_next_action") or {}).get("query"),
                            timestamp=ts)
        elif r["phase"] == "final":
            tb.agent_finished(r["agent_id"], r["step"], timestamp=ts)
    if meta.get("report"):
        tb.report_written(meta["report"], timestamp=meta.get("finished_at"))
    return tb.run_finished(interventions=meta.get("interventions"), ground_truth=meta.get("ground_truth"),
                           timestamp=meta.get("finished_at"),
                           out_path=run_dir / f"{out_stem}.json" if write else None)
