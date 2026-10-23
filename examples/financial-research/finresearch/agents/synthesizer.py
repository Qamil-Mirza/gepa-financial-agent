"""Report synthesizer: the final agent that turns the swarm's findings into one financial report.

It writes only from findings the supervisor did not implicate. A finding is excluded when the supervisor cited it,
or the plan/query/response it came from, as evidence in a DRIFTING verdict that raised an alert, or when it has no
citation. The model writes the prose and cites findings by short tags (F1, F2, ...), which code maps back to event
ids: models shorten long ids (agent_x-e007 -> e007), and short ids collide across agents. Code validates the tags,
builds the Sources section from the findings' own URLs (so sources cannot be invented), and lists any figure in the
report that no finding contains.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

from pitstop.llm import LLMClient
from pitstop.sdk.errors import Halted

from ..schemas import EventType, TrajectoryEvent

_GROUP = re.compile(r"\[([^\[\]]{1,200})\]")
_TAG = re.compile(r"F\d+")
_EVENT_ID = re.compile(r"[A-Za-z0-9_\-]+-e\d{3}|e\d{3}")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def citation_groups(markdown: str) -> list[tuple[str, list[str]]]:
    """Bracket groups that are citations: every comma/semicolon-separated token is a tag or an event id."""
    out = []
    for m in _GROUP.finditer(markdown):
        toks = [t.strip() for t in re.split(r"[,;]", m.group(1)) if t.strip()]
        if toks and all(_TAG.fullmatch(t) or _EVENT_ID.fullmatch(t) for t in toks):
            out.append((m.group(0), toks))
    return out


def _ancestors(ev: TrajectoryEvent, by_id: dict[str, TrajectoryEvent]) -> set[str]:
    out, cur = set(), ev
    while cur.parent_event_id and cur.parent_event_id in by_id and cur.parent_event_id not in out:
        out.add(cur.parent_event_id)
        cur = by_id[cur.parent_event_id]
    return out


def flagged_evidence(decisions: list[dict]) -> dict[str, set[str]]:
    """Per agent, every event id cited by a DRIFTING verdict at alert level."""
    out: dict[str, set[str]] = {}
    for r in decisions:
        d = r["result"]["decision"]
        if d["status"] == "DRIFTING" and r.get("alert_level") == "alert":
            out.setdefault(r["agent_id"], set()).update(d.get("evidence_event_ids", []))
    return out


def select_findings(events: list[TrajectoryEvent], decisions: list[dict]) -> tuple[list[dict], list[dict]]:
    """(included, excluded) findings, each {"event_id", "agent_id", "claim", ..., "reason" for excluded}."""
    by_id = {e.event_id: e for e in events}
    flagged = flagged_evidence(decisions)
    included, excluded = [], []
    for ev in events:
        if ev.type != EventType.FINDING:
            continue
        f = {"event_id": ev.event_id, "agent_id": ev.agent_id, "step": ev.step, **ev.payload}
        hit = ({ev.event_id} | _ancestors(ev, by_id)) & flagged.get(ev.agent_id, set())
        if hit:
            excluded.append({**f, "reason": f"supervisor flagged it as drift (evidence {sorted(hit)})"})
        elif not ev.payload.get("citation_event_ids"):
            excluded.append({**f, "reason": "no citation"})
        else:
            included.append(f)
    return included, excluded


def unverified_figures(markdown: str, findings: list[dict]) -> list[str]:
    """Numbers in the report that appear in no included finding (claim, value or period)."""
    corpus = " ".join(f"{f.get('claim', '')} {f.get('value') or ''} {f.get('period') or ''}" for f in findings)
    known = {n.replace(",", "") for n in _NUM.findall(corpus)}
    body = markdown
    for group, _ in citation_groups(markdown):
        body = body.replace(group, "")
    return sorted({n for n in _NUM.findall(body) if len(n.replace(",", "").replace(".", "")) > 1
                   and n.replace(",", "") not in known})


class ReportSynthesizer:
    def __init__(self, llm: LLMClient, prompts_dir: Path, *, attempts: int = 5, backoff_s: float = 5.0):
        self.llm = llm
        self.prompt = (Path(prompts_dir) / "synthesizer.txt").read_text()
        self.attempts, self.backoff_s = attempts, backoff_s

    async def synthesize(self, *, objective: str, agents: list[dict], events: list[TrajectoryEvent],
                         decisions: list[dict], interventions: dict[str, dict]) -> dict[str, Any]:
        """agents: [{"agent_id", "directive", "final_report"}]. Returns the report record (also see write())."""
        included, excluded = select_findings(events, decisions)
        if not included:
            return {"title": "No report", "markdown": "", "error": "no usable findings", "included": [],
                    "excluded": excluded, "cited_finding_ids": [], "invalid_citations": [],
                    "unverified_figures": [], "model": self.llm.model}
        for i, f in enumerate(included, 1):
            f["tag"] = f"F{i}"
        listing = []
        for a in agents:
            fs = [f for f in included if f["agent_id"] == a["agent_id"]]
            status = " (was corrected by the supervisor)" if interventions.get(a["agent_id"]) else ""
            listing.append(f"### Agent {a['agent_id']}{status}\nDirective: {a['directive']}\n" + (
                "\n".join(f"[{f['tag']}] {f['claim']} (period: {f.get('period') or 'n/a'})" for f in fs)
                or "(no usable findings)"))
        user = f"Team objective: {objective}\n\nFindings by agent:\n\n" + "\n\n".join(listing)
        by_tag = {f["tag"]: f for f in included}
        by_id = {f["event_id"]: f for f in included}
        for rewrite in range(2):  # one rewrite when the model ignores the citation rule
            obj = None
            for attempt in range(self.attempts):  # provider rate limits are common at the end of a run
                try:
                    obj, _ = await self.llm.complete_json(self.prompt, user)
                    break
                except Halted:  # a budget/drift halt is final: never retried, never turned into a failed report
                    raise
                except Exception as e:  # the swarm's research is still logged; the failure is recorded, not raised
                    if attempt == self.attempts - 1:
                        return {"title": "Report failed", "markdown": "", "error": f"{type(e).__name__}: {e}"[:300],
                                "included": included, "excluded": excluded, "cited_finding_ids": [],
                                "invalid_citations": [], "unverified_figures": [], "model": self.llm.model}
                    await asyncio.sleep(min(60.0, self.backoff_s * 2 ** attempt))
            body = str(obj.get("report_markdown", "")).strip()
            cited, invalid = [], []
            for _, toks in citation_groups(body):
                for t in toks:
                    f = by_tag.get(t) or by_id.get(t)
                    (cited if f else invalid).append(f["event_id"] if f else t)
            if cited:
                break
            user += ("\n\nYour previous report cited no findings. Rewrite it so that every figure and claim is "
                     "followed by its finding tag in square brackets, e.g. [F3] or [F3, F7].")
        return {"title": str(obj.get("title") or "Financial research report")[:200], "markdown": body,
                "gaps": str(obj.get("gaps", ""))[:1000], "cited_finding_ids": list(dict.fromkeys(cited)),
                "invalid_citations": list(dict.fromkeys(invalid)), "unverified_figures": unverified_figures(body, included),
                "included": included, "excluded": excluded, "model": self.llm.model}

    @staticmethod
    def render(report: dict[str, Any], objective: str) -> str:
        """Final markdown: title, model-written body, then code-built caveats and Sources."""
        out = [f"# {report['title']}", f"*Objective:* {objective}", "", report["markdown"] or f"_{report.get('error')}_"]
        notes = []
        if report["invalid_citations"]:
            notes.append(f"- Citations that match no usable finding: {', '.join(report['invalid_citations'])}.")
        if report["unverified_figures"]:
            notes.append(f"- Figures not found in any cited finding (check before use): "
                         f"{', '.join(report['unverified_figures'])}.")
        if report["excluded"]:
            notes.append(f"- {len(report['excluded'])} finding(s) excluded: " + "; ".join(
                f"{f['event_id']} ({f['reason']})" for f in report["excluded"][:10]) + ".")
        if notes:
            out += ["", "## Verification notes", *notes]
        by_id = {f["event_id"]: f for f in report["included"]}
        if report["cited_finding_ids"]:
            out += ["", "## Sources"]
            for fid in report["cited_finding_ids"]:
                f = by_id[fid]
                urls = ", ".join(f.get("source_urls") or []) or "(source URL not recorded)"
                out.append(f"- [{f.get('tag', fid)}] {f.get('claim', '')} — {urls} (finding {fid}, retrieved "
                           f"{f.get('retrieved_at') or 'n/a'})")
        return "\n".join(out) + "\n"

    def write(self, report: dict[str, Any], objective: str, run_dir: Path) -> Optional[Path]:
        path = Path(run_dir) / "report.md"
        path.write_text(self.render(report, objective))
        (Path(run_dir) / "report.json").write_text(json.dumps(report, indent=2))
        return path
