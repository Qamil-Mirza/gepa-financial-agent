"""Research worker: plans one narrow You.com query at a time.

The worker never sees supervisor verdicts. When the swarm runs with corrections
on, it does receive the supervisor's correction messages (supervisor/correction.py),
which go last in its prompts. Its optional drift injection is hidden from the
supervisor and recorded only as ground truth in the run metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..collector import TrajectoryCollector
from ..llm import JSONParseError, LLMClient
from ..research.youcom import FinanceResearchClient, YouComError
from ..schemas import AgentSpec, EventType, ProposedAction

MAX_QUERY_CHARS = 300
MAX_ANSWER_CHARS_FOR_EXTRACTION = 6000


@dataclass
class WorkerState:
    step: int = 0
    research_calls: int = 0
    queries: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)  # {"event_id", "claim", ...}
    current_plan: Optional[str] = None
    consecutive_plan_failures: int = 0
    finished: bool = False
    corrections: list[str] = field(default_factory=list)  # supervisor correction messages, oldest first
    final_report: Optional[str] = None


class ResearchWorker:
    def __init__(
        self,
        spec: AgentSpec,
        *,
        swarm_objective: str,
        llm: LLMClient,
        research: FinanceResearchClient,
        collector: TrajectoryCollector,
        prompts_dir: Path,
    ):
        self.spec = spec
        self.id = spec.agent_id
        self.objective = swarm_objective
        self.llm = llm
        self.research = research
        self.collector = collector
        self.planner_prompt = (prompts_dir / "planner.txt").read_text()
        self.extractor_prompt = (prompts_dir / "extractor.txt").read_text()
        self.reporter_prompt = (prompts_dir / "reporter.txt").read_text()
        self.state = WorkerState()

    def drift_active(self) -> bool:
        inj = self.spec.drift_injection
        return bool(inj and self.state.step >= inj.at_step)

    def _context(self) -> str:
        s = self.state
        findings = "\n".join(f"- {f['claim']} ({f.get('period') or 'n/a'})" for f in s.findings) or "- none yet"
        queries = "\n".join(f"- {q}" for q in s.queries) or "- none yet"
        return (
            f"Team objective: {self.objective}\n"
            f"Your directive: {self.spec.directive}\n"
            "Constraints:\n" + "\n".join(f"- {c}" for c in self.spec.constraints) + "\n"
            "Success criteria:\n" + "\n".join(f"- {c}" for c in self.spec.success_criteria) + "\n"
            f"Questions already asked:\n{queries}\n"
            f"Findings so far:\n{findings}\n"
            f"Turns remaining including this one: {self.spec.max_steps - s.step}"
        )

    # ---------------- planning ----------------
    async def propose(self) -> ProposedAction:
        s = self.state
        user = self._context()
        if self.drift_active():
            user += f"\n\nPRIVATE PRIORITY UPDATE: {self.spec.drift_injection.instruction}"
        if s.corrections:
            user += f"\n\n{s.corrections[-1]}"
        if s.step >= self.spec.max_steps - 1 and s.research_calls > 0:
            user += "\n\nThis is your last turn: you must choose finish."
        try:
            obj, _ = await self.llm.complete_json(self.planner_prompt, user)
        except (JSONParseError, Exception) as e:  # keep the swarm alive; the failure is observable
            s.consecutive_plan_failures += 1
            self.collector.emit(self.id, s.step, EventType.ERROR, f"planning failed: {type(e).__name__}",
                                {"error": str(e)[:300]})
            action = "finish" if s.consecutive_plan_failures >= 2 else "research"
            return ProposedAction(action=action, query=None, rationale="planning failure")
        s.consecutive_plan_failures = 0

        action = "finish" if str(obj.get("action", "")).lower() == "finish" else "research"
        query = (obj.get("query") or "").strip()[:MAX_QUERY_CHARS] if action == "research" else None
        if action == "research" and not query:
            action = "finish"
        if s.step >= self.spec.max_steps - 1 and s.research_calls > 0:
            action, query = "finish", None
        s.current_plan = str(obj.get("plan_summary", ""))[:500]
        ev = self.collector.emit(
            self.id, s.step, EventType.PLAN, s.current_plan or "(no plan summary)",
            {"proposed_action": action, "proposed_query": query, "rationale": str(obj.get("rationale", ""))[:400]},
        )
        return ProposedAction(action=action, query=query, rationale=str(obj.get("rationale", ""))[:400],
                              plan_event_id=ev.event_id)

    # ---------------- acting ----------------
    async def execute(self, proposal: ProposedAction) -> None:
        s = self.state
        if proposal.action != "research" or not proposal.query:
            return
        call = self.collector.emit(
            self.id, s.step, EventType.TOOL_CALL, "finance_research call",
            {"tool": "you.com/finance_research", "query": proposal.query, "effort": self.research.effort},
            parent_event_id=proposal.plan_event_id,
        )
        s.queries.append(proposal.query)
        try:
            res = await self.research.research(proposal.query)
        except YouComError as e:
            self.collector.emit(self.id, s.step, EventType.ERROR, f"research failed: {type(e).__name__}",
                                {"error": str(e)[:300]}, parent_event_id=call.event_id)
            return
        s.research_calls += 1
        resp = self.collector.emit(
            self.id, s.step, EventType.TOOL_RESPONSE,
            f"answer with {len(res.sources)} sources ({len(res.text)} chars){' [cached]' if res.from_cache else ''}"
            f"{' [FIXTURE]' if res.fixture else ''}",
            {"digest": res.text[:1500], "cache_key": res.cache_key, "retrieved_at": res.retrieved_at,
             "from_cache": res.from_cache, "fixture": res.fixture, "latency_s": round(res.latency_s, 1),
             "cited_indices": res.cited_indices},
            parent_event_id=call.event_id,
        )
        cite_ids: dict[int, str] = {}
        for src in res.sources:
            ev = self.collector.emit(
                self.id, s.step, EventType.CITATION, src.title[:120] or src.url,
                {"index": src.index, "url": src.url, "title": src.title, "retrieved_at": res.retrieved_at,
                 "cited_in_answer": src.index in res.cited_indices},
                parent_event_id=resp.event_id,
            )
            cite_ids[src.index] = ev.event_id
        await self._extract(res, cite_ids, resp.event_id)

    async def _extract(self, res, cite_ids: dict[int, str], response_event_id: str) -> None:
        s = self.state
        sources = "\n".join(f"[{src.index}] {src.title} <{src.url}>" for src in res.sources)
        user = (
            f"Your directive: {self.spec.directive}\n"
            + (f"PRIVATE PRIORITY UPDATE: {self.spec.drift_injection.instruction}\n" if self.drift_active() else "")
            + (f"{s.corrections[-1]}\n" if s.corrections else "")
            + f"Question asked: {res.query}\n\nResearch answer:\n{res.text[:MAX_ANSWER_CHARS_FOR_EXTRACTION]}\n\nSources:\n{sources}"
        )
        try:
            obj, _ = await self.llm.complete_json(self.extractor_prompt, user)
        except Exception as e:
            self.collector.emit(self.id, s.step, EventType.ERROR, "finding extraction failed",
                                {"error": str(e)[:300]}, parent_event_id=response_event_id)
            return
        for f in (obj.get("findings") or [])[:5]:
            if not isinstance(f, dict) or not f.get("claim"):
                continue
            idx = [i for i in (f.get("source_indices") or []) if isinstance(i, int)]
            cids = [cite_ids[i] for i in idx if i in cite_ids]
            payload = {
                "claim": str(f["claim"])[:400], "metric": f.get("metric"), "value": f.get("value"),
                "period": f.get("period"), "citation_event_ids": cids,
                "source_urls": [ev_url for ev_url in (res.sources[i - 1].url for i in idx if 0 < i <= len(res.sources))],
                "retrieved_at": res.retrieved_at,
            }
            ev = self.collector.emit(self.id, s.step, EventType.FINDING, payload["claim"], payload,
                                     parent_event_id=response_event_id)
            s.findings.append({"event_id": ev.event_id, **payload})

    async def finish(self) -> None:
        s = self.state
        s.finished = True
        if not s.findings:
            self.collector.emit(self.id, s.step, EventType.AGENT_MESSAGE, "final report: no findings recorded",
                                {"kind": "final_report", "report": "", "cited_finding_ids": []})
            return
        listing = "\n".join(f"[{f['event_id']}] {f['claim']}" for f in s.findings)
        try:
            obj, _ = await self.llm.complete_json(self.reporter_prompt,
                                                  f"Your directive: {self.spec.directive}\n\nFindings:\n{listing}")
        except Exception as e:
            self.collector.emit(self.id, s.step, EventType.ERROR, "report failed", {"error": str(e)[:300]})
            return
        valid = {f["event_id"] for f in s.findings}
        cited = [i for i in (obj.get("cited_finding_ids") or []) if i in valid]
        s.final_report = str(obj.get("report", ""))[:2000]
        self.collector.emit(
            self.id, s.step, EventType.AGENT_MESSAGE, "final report: " + s.final_report[:200],
            {"kind": "final_report", "report": s.final_report, "cited_finding_ids": cited,
             "gaps": str(obj.get("gaps", ""))[:400]},
        )

    def receive_correction(self, message: str, *, drift_type: str, evidence_event_ids: list[str],
                           blocked_query: Optional[str]):
        """Record a supervisor correction. It is a visible event, so later supervisor checks know it was sent."""
        s = self.state
        s.corrections.append(message)
        return self.collector.emit(
            self.id, s.step, EventType.SUPERVISOR_MESSAGE, f"supervisor correction #{len(s.corrections)}",
            {"message": message, "attempt": len(s.corrections), "drift_type": drift_type,
             "evidence_event_ids": evidence_event_ids, "blocked_query": blocked_query},
        )

    def progress(self) -> None:
        s = self.state
        self.collector.emit(self.id, s.step, EventType.PROGRESS,
                            f"step {s.step} done: {s.research_calls} calls, {len(s.findings)} findings",
                            {"research_calls": s.research_calls, "findings": len(s.findings)})
