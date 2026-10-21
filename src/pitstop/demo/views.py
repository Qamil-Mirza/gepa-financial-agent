"""JSON views for the GEPA demo page. Pure functions over files on disk.

  build_swarm_view(run_dir)      screen 2: one swarm, two supervisors judging identical inputs ("mode": "single")
  build_pair_view(pair_dir)      screen 2: two swarms of one scenario, each run by its own supervisor ("pair")
  build_gepa_view(gepa_run_dir)  screen 3: candidates, lineage and val score per GEPA iteration
  build_scores_view(arms)        screen 4: held-out test metrics per arm

Both screen-2 views have the same per-pane keys ("a" = original prompt, "b" = GEPA): events, decisions,
timeline, agent_reports, team_report. In a single run both panes share one event stream.

Every number comes from a file. A missing source gives a pending arm or a null, never a made-up value.
Logs are read tolerantly (a trailing partial line is skipped), so a run can be viewed while it is written.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from ..timeline import TimelineBuilder, TimelineKind, agent_label
from ..ui.server import read_jsonl_complete

SUMMARY_CHARS = 200
EXPLANATION_CHARS = 600
REPORT_CHARS = 12_000
PAIR_FILE = "pair.json"  # written by demo/pair.py; marks a dir holding two swarm runs
PathLike = Union[str, Path]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _trunc(text: Optional[str], n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n - 1] + "…"


def _display_path(path: Path) -> str:
    """Relative to the current directory when possible (so a snapshot exported from the repo root carries no
    home-directory paths), the absolute path otherwise."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


# ---------------- screen 2: swarm ----------------

def _slim_event(e: dict) -> dict:
    p = e.get("payload") or {}
    query = p.get("proposed_query") if e["type"] == "plan" else p.get("query") if e["type"] == "tool_call" else None
    out = {"event_id": e["event_id"], "agent_id": e["agent_id"], "step": e["step"], "seq": e["seq"],
           "type": e["type"], "summary": _trunc(e.get("summary"), SUMMARY_CHARS), "query": query or None}
    if e["type"] == "supervisor_message":  # a correction: the panel shows what the agent was told
        out.update(message=_trunc(p.get("message"), 900), attempt=p.get("attempt"),
                   blocked_query=p.get("blocked_query"))
    return out


def _failure(res: dict) -> Optional[str]:
    """A check with no real verdict: the call failed (rate limit / transport) or the reply was not JSON.
    The supervisor records these as UNCERTAIN at 0.0; the page must not present them as a judgment."""
    issues = [str(i) for i in res.get("issues", [])]
    if any("supervisor_call_failed" in i for i in issues):
        return "call_failed"
    return "invalid_json" if "invalid_json" in issues else None


def _slim_decision(r: dict, index: int) -> dict:
    d = r["result"]["decision"]
    return {"index": index, "failed": _failure(r["result"]), "agent_id": r["agent_id"], "step": r["step"], "phase": r["phase"], "status": d["status"],
            "confidence": d["confidence"], "drift_type": d.get("drift_type"),
            "alert_level": r.get("alert_level"), "alert_note": r.get("alert_note"),
            "explanation": _trunc(d.get("explanation"), EXPLANATION_CHARS),
            "evidence_event_ids": d.get("evidence_event_ids", []),
            "cutoff_seq": r["input"].get("observation_cutoff_seq")}


# Entries a "why did it say that?" panel can show; lifecycle entries are left out.
_PANEL_KINDS = {TimelineKind.DRIFT_FLAGGED, TimelineKind.DRIFT_SUSPECTED, TimelineKind.DRIFT_RESPONSE,
                TimelineKind.STATUS_CHANGED, TimelineKind.SUPERVISOR_CORRECTION}


def _timeline(run_id: str, scen: dict, records: list[dict], supervisor: Optional[dict]) -> list[dict]:
    """The curated timeline (pitstop.timeline) of one supervisor's decisions, built in memory so it works
    for the shadow supervisor and for a run still being written."""
    tb = TimelineBuilder()
    tb.run_started(run_id, {"swarm_objective": scen.get("swarm_objective"),
                            "agents": [{"agent_id": a["agent_id"], "label": a.get("label"),
                                        "directive": a.get("directive", "")} for a in scen.get("agents", [])]},
                   supervisor=supervisor)
    for i, r in enumerate(records):
        tb.on_decision(r, i)
    return [{"entry_id": e.entry_id, "kind": e.kind.value, "severity": e.severity, "title": e.title,
             "detail": e.detail, "agent_id": e.agent_id, "step": e.step, "phase": e.phase,
             "decision_index": e.refs.get("decision_index"), "responds_to": e.responds_to,
             "evidence": [ev.model_dump() for ev in e.evidence],
             "outcome": e.data.get("outcome"), "blocked_query": e.data.get("blocked_query")}
            for e in tb.entries if e.kind in _PANEL_KINDS]


def _supervisor_of(meta_entry: Optional[dict], given: Optional[dict], records: list[dict]) -> Optional[dict]:
    if meta_entry:
        return {"model": meta_entry.get("model", ""), "prompt_version": meta_entry.get("prompt_version", "")}
    if given:
        return {"model": given.get("model", ""), "prompt_version": given.get("prompt_version", "")}
    if records:  # in-progress run without meta.json: every record's result names its judge
        res = records[0]["result"]
        return {"model": res.get("model", ""), "prompt_version": res.get("prompt_version", "")}
    return None


def _scenario_from_logs(timeline: list[dict], decisions: list[dict]) -> dict:
    """Best effort for a run still being written (meta.json lands at the end). Drift injections are
    hidden from the supervisor, so they cannot be recovered here: pass `scenario` to know them."""
    started = next((e for e in timeline if e["kind"] == "run_started"), {})
    first = {}
    for r in decisions:
        first.setdefault(r["agent_id"], r["input"])
    ids = (started.get("data") or {}).get("agents") or list(first)
    objective = started.get("detail") or next((i.get("swarm_objective") for i in first.values()), "")
    return {"swarm_objective": objective,
            "agents": [{"agent_id": a, "directive": first.get(a, {}).get("agent_directive", ""),
                        "max_steps": first.get(a, {}).get("remaining_budget", {}).get("steps_max", 0)}
                       for a in ids]}


def _final_reports(events: list[dict]) -> dict[str, Optional[str]]:
    return {e["agent_id"]: (e.get("payload") or {}).get("report") or None for e in events
            if e["type"] == "agent_message" and (e.get("payload") or {}).get("kind") == "final_report"}


def _agents(scen: dict) -> list[dict]:
    out = []
    for a in scen.get("agents", []):
        inj = a.get("drift_injection")
        out.append({"agent_id": a["agent_id"], "label": a.get("label") or agent_label(a["agent_id"]),
                    "directive": a.get("directive", ""), "max_steps": a.get("max_steps", 0),
                    "drift_at_step": inj["at_step"] if inj else None})
    return out


def _report_body(md: str, title: str) -> str:
    """report.md minus what the page shows anyway: the "# title" line and the "*Objective:*" line.
    The body, verification notes and Sources stay."""
    lines = md.split("\n")
    if lines and title and lines[0].strip() == f"# {title}".strip():
        lines = lines[1:]
    if lines and lines[0].startswith("*Objective:*"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _team_report(run_dir: Path, meta: dict) -> Optional[dict]:
    """The team report of one swarm (agents/synthesizer.py), or None before its report agent has written it.
    report.md and report.json land together after the agents finish; meta.json's summary of it at the very end."""
    full = _read_json(run_dir / "report.json")
    full = full if isinstance(full, dict) else {}
    rec = meta.get("report") or {}
    if not full and not rec:
        return None
    title = str(full.get("title") or rec.get("title") or "")
    try:
        md = (run_dir / "report.md").read_text()
    except OSError:
        md = str(full.get("markdown") or "")
    count = lambda k, n: len(full[k]) if isinstance(full.get(k), list) else rec.get(n, 0)
    pick = lambda k: full.get(k) if k in full else rec.get(k)
    return {"title": title, "markdown": _trunc(_report_body(md, title), REPORT_CHARS),
            "error": pick("error") or None,
            "findings_included": count("included", "findings_included"),
            "findings_excluded": count("excluded", "findings_excluded"),
            "cited_finding_ids": pick("cited_finding_ids") or [], "unverified_figures": pick("unverified_figures") or [],
            "invalid_citations": pick("invalid_citations") or [], "model": str(pick("model") or "")}


LAYOUT_FILE = "demo.json"  # {"left": "primary"|"shadow", "right": ...}; written by the demo's earlier live runs


def build_swarm_view(run_dir: Path, *, scenario: Optional[dict] = None,
                     supervisors: Optional[dict] = None) -> dict:
    """One run as replayable lanes. Pane "a" (left) is the original prompt, "b" (right) the GEPA prompt.
    Which log feeds which pane comes from demo.json; without it the primary supervisor is on the left.
    `in_charge` names the pane whose supervisor actually ran the swarm (and sent any corrections).
    Both panes share the one event stream and agent reports; the team report sits under the primary's pane.
    `scenario` / `supervisors` ({"primary": {...}, "shadow": {...}}) fill in what an in-progress run has
    not written yet; meta.json wins once it exists."""
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "meta.json") or {}
    layout = _read_json(run_dir / LAYOUT_FILE) or {"left": "primary", "right": "shadow"}
    events = read_jsonl_complete(run_dir / "events.jsonl")
    logs = {"primary": read_jsonl_complete(run_dir / "decisions.jsonl"),
            "shadow": read_jsonl_complete(run_dir / "decisions_shadow.jsonl")}
    scen = meta.get("scenario") or scenario or _scenario_from_logs(
        read_jsonl_complete(run_dir / "timeline.jsonl"), logs["primary"])
    given = supervisors or {}
    sups = {"primary": _supervisor_of(meta.get("supervisor"), given.get("primary"), logs["primary"])
                       or {"model": "", "prompt_version": ""},
            "shadow": _supervisor_of(meta.get("shadow_supervisor"), given.get("shadow"), logs["shadow"])}
    pane = {"a": layout["left"], "b": layout["right"]}
    run_id = meta.get("run_id") or run_dir.name
    reports = _final_reports(events)
    agents = [{**a, "final_report": reports.get(a["agent_id"])} for a in _agents(scen)]
    slim = [_slim_event(e) for e in events]
    team = _team_report(run_dir, meta)
    return {
        "source": "recorded",
        "mode": "single",
        "run_id": run_id,
        "in_progress": not meta.get("finished_at"),
        "objective": scen.get("swarm_objective", ""),
        # only runs made for the demo (with a layout file) have a supervisor visibly "in charge"
        "in_charge": next(k for k, v in pane.items() if v == "primary") if (run_dir / LAYOUT_FILE).exists() else None,
        "supervisors": {k: sups[v] for k, v in pane.items()},
        "agents": agents,
        "events": {k: slim for k in pane},
        "decisions": {k: [_slim_decision(r, i) for i, r in enumerate(logs[v])] for k, v in pane.items()},
        "timeline": {k: _timeline(run_id, scen, logs[v], sups[v]) for k, v in pane.items()},
        "agent_reports": {k: {a["agent_id"]: a["final_report"] for a in agents} for k in pane},
        "team_report": {k: team if v == "primary" else None for k, v in pane.items()},
    }


def build_pair_view(pair_dir: Path, *, live: Optional[Any] = None) -> dict:
    """A pair (demo/pair.py): two swarms of one scenario, each run by its own supervisor, which corrects drift.
    Pane "a" is the left swarm (original prompt), "b" the right one (GEPA); nothing is shared between panes
    but the scenario. `live` (the running demo.pair.Pair) fills in the scenario and judges before a swarm's
    meta.json exists. in_progress until pair.json has finished_at."""
    pair_dir = Path(pair_dir)
    doc = _read_json(pair_dir / PAIR_FILE) or {}
    swarms = {"a": live.left, "b": live.right} if live is not None else {}
    panes = {}
    for k, side in (("a", "left"), ("b", "right")):
        sw = swarms.get(k)
        rid = sw.run.run_id if sw else doc.get(side)
        run_dir = pair_dir / rid if rid else None
        meta = (_read_json(run_dir / "meta.json") or {}) if run_dir else {}
        events = read_jsonl_complete(run_dir / "events.jsonl") if run_dir else []
        records = read_jsonl_complete(run_dir / "decisions.jsonl") if run_dir else []
        scen = meta.get("scenario") or (sw.scenario.model_dump(mode="json") if sw else None)
        given = {"model": sw.supervisor.llm.model, "prompt_version": sw.supervisor.prompt_version} if sw else None
        panes[k] = {"run_id": rid or "", "run_dir": run_dir, "meta": meta, "events": events, "records": records,
                    "scen": scen, "reports": _final_reports(events),
                    "sup": _supervisor_of(meta.get("supervisor"), given, records) or {"model": "", "prompt_version": ""}}
    known = next((p["scen"] for p in panes.values() if p["scen"]), None)  # the same scenario on both sides
    for p in panes.values():
        p["scen"] = p["scen"] or known or _scenario_from_logs(
            read_jsonl_complete(p["run_dir"] / "timeline.jsonl") if p["run_dir"] else [], p["records"])
    scen = known or panes["a"]["scen"]
    agents = _agents(scen)
    ids = [a["agent_id"] for a in agents]
    return {
        "source": "recorded",
        "mode": "pair",
        "run_id": doc.get("pair_id") or (live.pair_id if live is not None else pair_dir.name),
        "in_progress": not doc.get("finished_at"),
        "objective": doc.get("objective") or scen.get("swarm_objective", ""),
        "in_charge": "both",
        "agents": agents,
        "supervisors": {k: p["sup"] for k, p in panes.items()},
        "events": {k: [_slim_event(e) for e in p["events"]] for k, p in panes.items()},
        "decisions": {k: [_slim_decision(r, i) for i, r in enumerate(p["records"])] for k, p in panes.items()},
        "timeline": {k: _timeline(p["run_id"], p["scen"], p["records"], p["sup"]) for k, p in panes.items()},
        "agent_reports": {k: {a: p["reports"].get(a) for a in ids} for k, p in panes.items()},
        "team_report": {k: _team_report(p["run_dir"], p["meta"]) if p["run_dir"] else None for k, p in panes.items()},
    }


# ---------------- screen 3: GEPA ----------------

_ITER = re.compile(r"^Iteration (\d+): (.*)$")
_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_REJECTED = ("is not better than", "Skipping", "did not propose a new candidate")


def _tree_nodes(path: Path) -> dict[int, dict]:
    """candidate_tree.html embeds `const NODES = [...]` (one JSON array, possibly spanning lines)."""
    try:
        html = path.read_text()
        start = html.index("const NODES = ") + len("const NODES = ")
        nodes, _ = json.JSONDecoder().raw_decode(html, start)
    except (OSError, ValueError):
        return {}
    out = {}
    for n in nodes:
        par = n.get("parents")
        if isinstance(par, list):
            par = par[0] if par else None
        par = str(par).split(",")[0].strip() if par is not None else ""
        out[int(n["idx"])] = {"score": n.get("score"), "parent": int(par) if par.isdigit() else None}
    return out


def _parse_gepa_log(path: Path) -> dict:
    """run_log.txt -> per-iteration parent / acceptance / new candidate index / val score."""
    iters: dict[int, dict] = {}
    seed_score, text = None, ""
    try:
        text = path.read_text()
    except OSError:
        pass
    for line in text.splitlines():
        m = _ITER.match(line)
        if not m:
            continue
        it, msg = int(m.group(1)), m.group(2)
        rec = iters.setdefault(it, {})
        if (sm := re.match(r"Base program full valset score: " + _FLOAT, msg)):
            seed_score = float(sm.group(1))
        elif (sm := re.match(r"Selected program (\d+)", msg)):
            rec["parent"] = int(sm.group(1))
        elif msg.startswith("Accepted candidate"):
            rec["accepted"] = True
        elif (sm := re.match(r"Valset score for new program: " + _FLOAT, msg)):
            rec["score"] = float(sm.group(1))
        elif (sm := re.match(r"New program candidate index: (\d+)", msg)):
            rec["idx"] = int(sm.group(1))
        elif any(k in msg for k in _REJECTED):
            rec["rejected"] = True
    return {"iters": iters, "seed_score": seed_score, "finished": "GEPA finished" in text}


def build_gepa_view(gepa_run_dir: Path) -> dict:
    d = Path(gepa_run_dir)
    cfg = _read_json(d / "config.json") or {}
    texts = [c.get("current_candidate", "") if isinstance(c, dict) else str(c)
             for c in (_read_json(d / "candidates.json") or [])]
    nodes = _tree_nodes(d / "candidate_tree.html")
    log = _parse_gepa_log(d / "run_log.txt")
    iters = log["iters"]
    iterations_run = max(iters, default=0)
    accepted = sorted(i for i, r in iters.items() if r.get("accepted"))
    idx_iter = {r["idx"]: i for i, r in iters.items() if "idx" in r}
    for n, it in enumerate(accepted, start=1):  # older logs without "New program candidate index"
        idx_iter.setdefault(n, it)

    candidates = []
    for idx, text in enumerate(texts):
        it = 0 if idx == 0 else idx_iter.get(idx)
        rec = iters.get(it, {}) if it else {}
        node = nodes.get(idx, {})
        score = node.get("score")
        if score is None:
            score = log["seed_score"] if idx == 0 else rec.get("score")
        if score is None or it is None:
            continue  # candidate written but not yet scored (or not traceable to an iteration): don't guess
        parent = None if idx == 0 else node.get("parent", rec.get("parent"))
        candidates.append({"idx": idx, "parent": parent, "iteration": it, "val_score": float(score),
                           "chars": len(text)})

    by_idx = {c["idx"]: c for c in candidates}
    for c in candidates:  # added = line absent from the parent's prompt (seed: nothing is added)
        par = c["parent"] if c["parent"] in by_idx else None
        seen = {l.strip() for l in texts[par].split("\n")} if par is not None else None
        c["lines"] = [{"text": l, "added": bool(seen is not None and l.strip() and l.strip() not in seen)}
                      for l in texts[c["idx"]].split("\n")]

    finished = log["finished"] or (d / "result.json").exists() or \
        "GEPA finished" in ((d.parent / f"{d.name}.log").read_text(errors="replace")
                            if (d.parent / f"{d.name}.log").exists() else "")
    accepted_set = set(idx_iter.values())
    rejected = sorted(i for i in range(1, iterations_run + 1) if i not in accepted_set and (
        iters.get(i, {}).get("rejected") or (finished and not iters.get(i, {}).get("accepted"))))
    best_so_far = []
    for it in range(iterations_run + 1):
        scores = [c["val_score"] for c in candidates if c["iteration"] <= it]
        if scores:
            best_so_far.append({"iteration": it, "score": max(scores)})
    seed_score = by_idx[0]["val_score"] if 0 in by_idx else log["seed_score"]
    return {
        "run": d.name, "model": cfg.get("supervisor_model", ""), "reflection_model": cfg.get("reflection_model", ""),
        "budget": (cfg.get("optimizer") or {}).get("max_metric_calls"), "metric": "val score",
        "status": "finished" if finished else "in_progress", "iterations_run": iterations_run,
        "seed_score": seed_score, "candidates": candidates, "rejected_iterations": rejected,
        "best_so_far": best_so_far,
    }


# ---------------- screen 4: scores ----------------

def _resolve_summary(path: Union[PathLike, Sequence[PathLike]]) -> Optional[tuple[Path, dict]]:
    """First match among `path` (or its fallbacks): an eval output dir, a glob of them (newest wins),
    or a summary JSON file. Relative paths are relative to the current directory."""
    for cand in [path] if isinstance(path, (str, Path)) else list(path):
        p = Path(cand)
        hits = sorted(glob.glob(str(p)), reverse=True) if any(c in str(p) for c in "*?[") else [str(p)]
        for h in hits:
            f = Path(h) / "summary.json" if Path(h).is_dir() else Path(h)
            doc = _read_json(f) if f.is_file() else None
            if isinstance(doc, dict):
                s = doc.get("summary", doc)
                if isinstance(s, dict) and (s.get("by_split") or s.get("overall")):
                    return f, s
    return None


def build_scores_view(arms: list[tuple[str, str, str, Union[PathLike, Sequence[PathLike]]]]) -> dict:
    """`path` may be one source or a list of fallbacks. `split` on each arm says which numbers were used."""
    out = []
    for key, label, sublabel, path in arms:
        found = _resolve_summary(path)
        if not found:
            src = ", ".join(map(str, [path] if isinstance(path, (str, Path)) else path))
            out.append({"key": key, "label": label, "sublabel": sublabel, "pending": True, "source": src})
            continue
        f, s = found
        split = "test" if (s.get("by_split") or {}).get("test") else "overall"
        m = s["by_split"]["test"] if split == "test" else s["overall"]
        out.append({"key": key, "label": label, "sublabel": sublabel, "model": s.get("model", ""),
                    "prompt": s.get("prompt", ""), "accuracy": m.get("accuracy"),
                    "drift_recall": m.get("drift_recall"), "false_alarms": m.get("fp"), "n": m.get("n"),
                    "trials": s.get("trials"), "split": split, "source": _display_path(f)})
    return {"split": "test", "metric": "accuracy", "arms": out}
