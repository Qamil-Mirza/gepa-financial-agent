"""Swarm orchestrator: runs research workers concurrently, watched by the PitStop SDK.

The few lines that hook the swarm into PitStop: `run_all` runs inside `pitstop.workflow("financial-research",
...)`, each worker's loop inside `pitstop.agent(agent_id, goal=directive)`, and the worker's paid You.com call is a
`@pitstop.tool(side_effects=True, name="youcom_research")`. On this medium workflow PitStop's L1 pre-action gate
judges each research call against the agent's directive and blocks an off-task one (the worker gets an ERROR event
naming the reason and re-plans); a `pitstop.Halted` (budget, or a P0 on a high workflow) ends the swarm.

`Swarm(watch=False)` skips that wiring entirely (research is called directly): the benchmarks and the demo pair use it
to measure the legacy supervisor alone, with no gate and no judge calls.

The hackathon per-step supervisor below is opt-in (`supervisor=...`, `finresearch run --legacy-supervisor`); it
still drives the demo page and the benchmarks. Its early-detection design:
  1. Workers ask one narrow question per step, so drift shows up in *which*
     questions they choose (You.com's internal research is a black box).
  2. Every step, the supervisor reviews the trajectory PLUS the proposed next
     query *before* it is sent. Catching drift here costs one LLM call instead
     of a slow, paid You.com call.
  3. An alert policy debounces single tangents to limit false alarms.
  4. A final check runs after the agent's report (catches unsupported conclusions).
  5. Optional correction: at an intervention point the drifting query is blocked and the agent gets a
     firm correction (supervisor/correction.py), then re-plans the same step under supervision.
  6. Optional synthesis: after all agents finish, a report agent writes the team's financial report
     from the findings the supervisor did not implicate (agents/synthesizer.py).
  7. An optional shadow supervisor judges the SAME inputs in parallel (decisions_shadow.jsonl)
     so two prompts can be compared on identical agent behavior. It never affects the run.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape

import pitstop

from .agents.synthesizer import ReportSynthesizer
from .agents.worker import ResearchWorker
from .collector import TrajectoryCollector
from .research.youcom import EST_COST_USD, FinanceResearchClient
from .schemas import (
    Budget,
    EventType,
    ProposedAction,
    Scenario,
    Status,
    SupervisorInput,
    SwarmMessage,
    TrajectoryEvent,
    ValidatedDecision,
    utc_now,
)
from .supervisor.correction import build_correction
from .supervisor.policy import AlertPolicy
from .supervisor.supervisor import Supervisor
from .timeline import TimelineBuilder, build_from_run_dir

STATUS_STYLE = {Status.ON_TRACK: "green", Status.DRIFTING: "bold red", Status.UNCERTAIN: "yellow"}
EVENT_STYLE = {
    EventType.PLAN: "cyan", EventType.TOOL_CALL: "blue", EventType.TOOL_RESPONSE: "dim",
    EventType.CITATION: "dim", EventType.FINDING: "white", EventType.AGENT_MESSAGE: "magenta",
    EventType.ERROR: "red", EventType.PROGRESS: "dim", EventType.SUPERVISOR_MESSAGE: "bold yellow",
}


@dataclass
class SwarmRun:
    run_id: str
    run_dir: Path
    interventions: dict[str, dict] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    report: Optional[dict] = None
    pitstop_run_id: Optional[str] = None  # the run's id in PitStop's store (`pitstop drifts`, `pitstop status`)
    halted: Optional[str] = None  # why PitStop halted the run, if it did


WORKFLOW = "financial-research"
RESEARCH_TOOL = "youcom_research"


class Swarm:
    def __init__(
        self,
        scenario: Scenario,
        *,
        worker_llm,
        supervisor: Optional[Supervisor] = None,
        research: FinanceResearchClient,
        prompts_dir: Path,
        runs_dir: Path,
        halt_on_drift: bool = False,
        policy: Optional[AlertPolicy] = None,
        console: Optional[Console] = None,
        verbose_events: bool = True,
        correct_on_drift: bool = False,
        max_corrections: int = 2,
        synthesizer: Optional[ReportSynthesizer] = None,
        shadow_supervisor: Optional[Supervisor] = None,
        watch: bool = True,
    ):
        self.scenario = scenario
        self.watch = watch  # False: no PitStop workflow, agents or gated tool (measure the legacy supervisor alone)
        self.supervisor = supervisor  # None: the PitStop SDK alone watches the run
        # judges every input the primary judges; own policy, own log, never halts.
        # Its timeline (timeline_shadow.json) is rebuilt from that log when the run ends.
        self.shadow_supervisor = shadow_supervisor
        self.shadow_policy = AlertPolicy()
        self.research = research
        self.halt_on_drift = halt_on_drift
        self.correct_on_drift = correct_on_drift
        self.max_corrections = max_corrections
        self.synthesizer = synthesizer
        self._last_verdict: Optional[ValidatedDecision] = None
        self.policy = policy or AlertPolicy()
        self.console = console or Console()
        self.run = SwarmRun(run_id=f"{scenario.task_id}-{uuid.uuid4().hex[:6]}", run_dir=Path())
        self.run.run_dir = Path(runs_dir) / self.run.run_id
        self.run.run_dir.mkdir(parents=True, exist_ok=True)
        self.collector = TrajectoryCollector(self.run.run_dir / "events.jsonl")
        # curated, UI-facing view: drift flags, responses and verdict changes only
        self.timeline = TimelineBuilder(self.run.run_dir / "timeline.jsonl")
        if verbose_events:
            self.collector.subscribe(self._print_event)
        self.board: list[SwarmMessage] = []
        # the gated research tool is decorated once per swarm, not per call
        self._research_tool = (pitstop.tool(side_effects=True, name=RESEARCH_TOOL)(research.research)
                               if watch else None)
        self.workers = [
            ResearchWorker(spec, swarm_objective=scenario.swarm_objective, llm=worker_llm,
                           research=research, collector=self.collector, prompts_dir=prompts_dir, watched=watch,
                           research_call=self._research_tool)
            for spec in scenario.agents
        ]

    # ---------------- display ----------------
    def _print_event(self, ev: TrajectoryEvent) -> None:
        if ev.type in (EventType.CITATION, EventType.PROGRESS):
            return
        extra = f"  q={ev.payload.get('query')!r}" if ev.type == EventType.TOOL_CALL else ""
        if ev.type == EventType.PLAN and ev.payload.get("proposed_query"):
            extra = f"\n      → proposes: {ev.payload['proposed_query']!r}"
        self.console.print(
            f"[{EVENT_STYLE[ev.type]}]  {ev.event_id:<22} {ev.type.value:<13} {escape(ev.summary[:110] + extra)}[/]"
        )

    def _print_decision(self, agent_id: str, phase: str, vd: ValidatedDecision, level: str, note: str) -> None:
        d = vd.decision
        style = STATUS_STYLE[d.status]
        tag = f" [bold white on red] {note} [/]" if note == "INTERVENTION POINT" else ""
        self.console.print(
            f"[{style}]  ◆ SUPERVISOR {agent_id} ({phase}): {d.status.value} conf={d.confidence:.2f} "
            f"type={d.drift_type.value} evidence={escape(str(d.evidence_event_ids))}[/]{tag}"
        )
        self.console.print(f"[{style}]      {escape(d.explanation[:300])}[/]")
        if vd.issues:
            self.console.print(f"[yellow]      validator: {escape(str(vd.issues))}[/]")

    # ---------------- supervisor input ----------------
    def build_input(self, worker: ResearchWorker, proposal: Optional[ProposedAction]) -> SupervisorInput:
        cutoff = self.collector.last_seq(worker.id)
        s = worker.state
        return SupervisorInput(
            task_id=self.scenario.task_id,
            agent_id=worker.id,
            swarm_objective=self.scenario.swarm_objective,
            agent_directive=worker.spec.directive,
            constraints=worker.spec.constraints,
            success_criteria=worker.spec.success_criteria,
            trajectory=self.collector.history(worker.id, upto_seq=cutoff),
            current_plan=s.current_plan,
            proposed_next_action=proposal,
            relevant_swarm_messages=[m for m in self.board if m.sender != worker.id][-5:],
            remaining_budget=Budget(
                steps_used=s.step, steps_max=worker.spec.max_steps,
                research_calls_used=s.research_calls, research_calls_max=worker.spec.max_steps,
                est_cost_usd=round(s.research_calls * EST_COST_USD[self.research.effort], 2),
            ),
            evaluation_step=s.step,
            observation_cutoff_seq=cutoff,
        )

    @staticmethod
    def _record(worker: ResearchWorker, phase: str, inp: SupervisorInput, vd: ValidatedDecision,
                level: str, note: str) -> dict:
        return {
            "agent_id": worker.id, "step": worker.state.step, "phase": phase,
            "alert_level": level, "alert_note": note,
            # ground-truth helper for later scoring; NEVER shown to the supervisor
            "drift_injection_active": worker.drift_active(),
            "input": inp.model_dump(mode="json"), "result": vd.model_dump(mode="json"),
        }

    async def _supervise(
        self, worker: ResearchWorker, proposal: Optional[ProposedAction], phase: str
    ) -> tuple[str, str]:
        inp = self.build_input(worker, proposal)
        if self.shadow_supervisor:
            vd, shadow_vd = await asyncio.gather(self.supervisor.evaluate(inp), self.shadow_supervisor.evaluate(inp))
            s_level, s_note = self.shadow_policy.update(worker.id, worker.state.step, shadow_vd)
            with (self.run.run_dir / "decisions_shadow.jsonl").open("a") as f:
                f.write(json.dumps(self._record(worker, phase, inp, shadow_vd, s_level, s_note)) + "\n")
        else:
            vd = await self.supervisor.evaluate(inp)
        level, note = self.policy.update(worker.id, worker.state.step, vd)
        record = self._record(worker, phase, inp, vd, level, note)
        self.run.decisions.append(record)
        with (self.run.run_dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.timeline.on_decision(record, len(self.run.decisions) - 1)
        self._print_decision(worker.id, phase, vd, level, note)
        if note == "INTERVENTION POINT" and worker.id not in self.run.interventions:  # keep the first flag
            self.run.interventions[worker.id] = {
                "step": worker.state.step, "phase": phase,
                "evidence_event_ids": vd.decision.evidence_event_ids,
                "drift_type": vd.decision.drift_type.value,
                "research_calls_so_far": worker.state.research_calls,
                "blocked_query": proposal.query if proposal else None,
                "corrections": [],
            }
        self._last_verdict = vd
        return level, note

    def _correct(self, worker: ResearchWorker, proposal: Optional[ProposedAction]) -> None:
        """Block the flagged proposal and tell the agent to get back on track."""
        d = self._last_verdict.decision
        msg = build_correction(worker.spec, d, proposal, len(worker.state.corrections) + 1)
        ev = worker.receive_correction(msg, drift_type=d.drift_type.value, evidence_event_ids=d.evidence_event_ids,
                                       blocked_query=proposal.query if proposal else None)
        self.run.interventions[worker.id]["corrections"].append(
            {"step": worker.state.step, "event_id": ev.event_id, "drift_type": d.drift_type.value,
             "blocked_query": proposal.query if proposal else None})
        self.policy.state.pop(worker.id, None)  # a fresh alert state, so persisting drift can be flagged again
        self.timeline.agent_corrected(worker.id, worker.state.step, msg, len(worker.state.corrections),
                                      blocked_query=proposal.query if proposal else None)
        self.console.print(f"[bold yellow]  ↩ {worker.id} corrected (#{len(worker.state.corrections)}); "
                           f"blocked {escape(repr(proposal.query if proposal else None))}[/]")

    # ---------------- main loop ----------------
    async def _run_worker(self, w: ResearchWorker) -> None:
        async with pitstop.agent(w.id, goal=w.spec.directive) if self.watch else contextlib.nullcontext():
            await self._worker_loop(w)

    async def _worker_loop(self, w: ResearchWorker) -> None:
        for step in range(w.spec.max_steps):
            w.state.step = step
            proposal = await w.propose()
            note = ""
            if self.supervisor:
                _, note = await self._supervise(w, proposal, "pre_action")
            while (note == "INTERVENTION POINT" and self.correct_on_drift
                   and len(w.state.corrections) < self.max_corrections):
                self._correct(w, proposal)
                proposal = await w.propose()
                _, note = await self._supervise(w, proposal, "post_correction")
            if note == "INTERVENTION POINT" and self.halt_on_drift:
                self.console.print(
                    f"[bold red]  ■ {w.id} halted at step {step} before executing {escape(repr(proposal.query))}[/]"
                )
                self.timeline.agent_halted(w.id, step, proposal.query)
                self.run.interventions[w.id]["halted_step"] = step
                return
            if proposal.action == "finish":
                break
            await w.execute(proposal)
            w.progress()
        await w.finish()
        if w.state.final_report:
            self.board.append(SwarmMessage(sender=w.id, timestamp=utc_now(), text=w.state.final_report[:400]))
        if self.supervisor:
            await self._supervise(w, None, "final")
        self.timeline.agent_finished(w.id, w.state.step, w.state.final_report)

    def _supervisor_info(self) -> Optional[dict]:
        sup = self.supervisor
        return {"model": sup.llm.model, "prompt_version": sup.prompt_version} if sup else None

    async def run_all(self) -> SwarmRun:
        self._print_header()
        self.timeline.run_started(self.run.run_id, self.scenario.model_dump(mode="json"),
                                  halt_on_drift=self.halt_on_drift, supervisor=self._supervisor_info())
        scope = pitstop.workflow(WORKFLOW, goal=self.scenario.swarm_objective, criticality="medium",
                                 tools=(RESEARCH_TOOL,)) if self.watch else contextlib.nullcontext()
        try:
            async with scope as ctx:
                if ctx is not None:
                    self.run.pitstop_run_id = ctx.run_id
                await self._gather_workers()
                if self.synthesizer:
                    await self._synthesize()
        except pitstop.Halted as e:  # the workflow scope has recorded the run as halted (or refused, at entry)
            self.run.halted = str(e)
            self.console.print(f"[bold red]  ■ PitStop halted the run: {escape(str(e))}[/]")
        self._write_meta()
        self._print_summary()
        return self.run

    async def _gather_workers(self) -> None:
        """Run every worker concurrently; when one raises (e.g. `pitstop.Halted`), cancel the others first."""
        tasks = [asyncio.ensure_future(self._run_worker(w)) for w in self.workers]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _synthesize(self) -> None:
        events = [e for w in self.workers for e in self.collector.history(w.id)]
        agents = [{"agent_id": w.id, "directive": w.spec.directive, "final_report": w.state.final_report}
                  for w in self.workers]
        rep = await self.synthesizer.synthesize(objective=self.scenario.swarm_objective, agents=agents, events=events,
                                                decisions=self.run.decisions, interventions=self.run.interventions)
        path = self.synthesizer.write(rep, self.scenario.swarm_objective, self.run.run_dir)
        self.run.report = {k: rep[k] for k in ("title", "cited_finding_ids", "invalid_citations", "unverified_figures")}
        self.run.report.update(path=str(path), model=rep["model"], error=rep.get("error"),
                               findings_included=len(rep["included"]), findings_excluded=len(rep["excluded"]))
        self.timeline.report_written(self.run.report)

    # ---------------- reporting ----------------
    def _print_header(self) -> None:
        c = self.console
        c.rule(f"[bold]PitStop run {self.run.run_id}")
        c.print(f"[bold]Objective:[/] {self.scenario.swarm_objective}")
        for a in self.scenario.agents:
            c.print(f"[bold]{a.agent_id}[/]: {a.directive}")
        watched = (f"watched by PitStop: workflow {WORKFLOW!r} (medium), {RESEARCH_TOOL} gated against each "
                   f"agent's directive" if self.watch else "not watched by PitStop (watch=False)")
        c.print(f"{watched} | You.com mode: {self.research.mode}")
        if self.supervisor:
            c.print(f"legacy supervisor prompt: {self.supervisor.prompt_version} | model: {self.supervisor.llm.model}"
                    f" | halt_on_drift: {self.halt_on_drift}")
        if self.shadow_supervisor:
            c.print(f"shadow supervisor prompt: {self.shadow_supervisor.prompt_version} | "
                    f"model: {self.shadow_supervisor.llm.model} (logged to decisions_shadow.jsonl only)")
        c.rule()

    def _ground_truth(self) -> dict:
        gt = {}
        for a in self.scenario.agents:
            inj = a.drift_injection
            gt[a.agent_id] = {
                "drift_injected": bool(inj),
                "onset_step": inj.at_step if inj else None,
                "expected_drift_type": inj.expected_drift_type.value if inj else None,
                "label_source": "injected_instruction_unverified" if inj else "no_injection",
            }
        return gt

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run.run_id, "scenario": self.scenario.model_dump(mode="json"),
            "ground_truth": self._ground_truth(), "interventions": self.run.interventions,
            "supervisor": ({**self._supervisor_info(), "usage": self.supervisor.llm.usage.as_dict()}
                           if self.supervisor else None),
            "pitstop": ({"workflow": WORKFLOW, "run_id": self.run.pitstop_run_id, "halted": self.run.halted}
                        if self.watch else None),
            "worker_usage": self.workers[0].llm.usage.as_dict() if self.workers else {},
            "research": self.research.stats(), "halt_on_drift": self.halt_on_drift,
            "correct_on_drift": self.correct_on_drift, "report": self.run.report,
            "finished_at": utc_now(),
        }
        if self.shadow_supervisor:
            sh = self.shadow_supervisor
            meta["shadow_supervisor"] = {"model": sh.llm.model, "prompt_version": sh.prompt_version,
                                         "usage": sh.llm.usage.as_dict()}
        (self.run.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self.timeline.run_finished(interventions=self.run.interventions, ground_truth=meta["ground_truth"],
                                   timestamp=meta["finished_at"], out_path=self.run.run_dir / "timeline.json")
        if self.shadow_supervisor:
            build_from_run_dir(self.run.run_dir, decisions_name="decisions_shadow.jsonl",
                               out_stem="timeline_shadow", supervisor_key="shadow_supervisor")

    def _pitstop_blocks(self, agent_id: str) -> list[TrajectoryEvent]:
        """The agent's research calls PitStop blocked (the worker records each as an ERROR event)."""
        return [e for e in self.collector.history(agent_id)
                if e.type == EventType.ERROR and e.payload.get("blocked_by") == "pitstop"]

    def _print_summary(self) -> None:
        c = self.console
        c.rule("[bold]Run summary (single run: anecdotal, not a benchmark)")
        gt = self._ground_truth()
        for a in self.scenario.agents if self.watch else ():
            g, blocks = gt[a.agent_id], self._pitstop_blocks(a.agent_id)
            injected = f"drift injected at step {g['onset_step']}" if g["drift_injected"] else "no injection"
            if blocks:
                note = "" if g["drift_injected"] else " (possible false alarm, or genuine uninjected drift)"
                c.print(f"{'[red]' if note else ''}{a.agent_id}: {injected} → PitStop blocked {len(blocks)} research "
                        f"call(s), first at step {blocks[0].step}{note}: {escape(blocks[0].payload['reason'])}"
                        f"{'[/]' if note else ''}")
            else:
                c.print(f"[{'red' if g['drift_injected'] else 'green'}]{a.agent_id}: {injected} → "
                        f"not blocked by PitStop[/]")
        if self.supervisor:
            self._print_supervisor_summary(gt)
        if self.run.report:
            r = self.run.report
            c.print(f"report: {r['path']} ({len(r['cited_finding_ids'])} findings cited, "
                    f"{r['findings_excluded']} excluded, unverified figures {r['unverified_figures'] or 'none'})")
        if self.supervisor:
            c.print(f"supervisor usage: {self.supervisor.llm.usage.as_dict()}")
        c.print(f"You.com: {self.research.stats()}")
        status = f"halted: {escape(self.run.halted)}" if self.run.halted is not None else "ok"
        if self.watch:
            c.print(f"PitStop run {self.run.pitstop_run_id or '(not started)'} ({status}); "
                    f"`pitstop drifts` lists what PitStop caught")
        elif self.run.halted is not None:
            c.print(f"run {status}")
        c.print(f"logs: {self.run.run_dir}")

    def _print_supervisor_summary(self, gt: dict) -> None:
        c = self.console
        for a in self.scenario.agents:
            g, iv = gt[a.agent_id], self.run.interventions.get(a.agent_id)
            if g["drift_injected"]:
                if iv:
                    delay = iv["step"] - g["onset_step"]
                    c.print(f"{a.agent_id}: legacy supervisor flagged at step {iv['step']} "
                            f"({iv['phase']}, delay {delay} steps) evidence={iv['evidence_event_ids']}")
                else:
                    c.print(f"[red]{a.agent_id}: legacy supervisor did NOT flag the injected drift[/]")
            elif iv:
                c.print(f"[red]{a.agent_id}: legacy supervisor flagged at step {iv['step']} (possible false alarm, "
                        f"or genuine uninjected drift: inspect the trace)[/]")
        for a in self.scenario.agents:
            iv = self.run.interventions.get(a.agent_id) or {}
            if iv.get("corrections"):
                last = next((r for r in reversed(self.run.decisions) if r["agent_id"] == a.agent_id), None)
                after = last["result"]["decision"]["status"] if last else "n/a"
                c.print(f"{a.agent_id}: {len(iv['corrections'])} correction(s); last verdict {after}")
