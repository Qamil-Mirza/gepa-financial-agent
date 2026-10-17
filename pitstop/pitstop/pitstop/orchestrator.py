"""Swarm orchestrator: runs research workers concurrently with a supervisor in the loop.

Early-detection design:
  1. Workers ask one narrow question per step, so drift shows up in *which*
     questions they choose (You.com's internal research is a black box).
  2. Every step, the supervisor reviews the trajectory PLUS the proposed next
     query *before* it is sent. Catching drift here costs one LLM call instead
     of a slow, paid You.com call.
  3. An alert policy debounces single tangents to limit false alarms.
  4. A final check runs after the agent's report (catches unsupported conclusions).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape

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
from .supervisor.policy import AlertPolicy
from .supervisor.supervisor import Supervisor

STATUS_STYLE = {Status.ON_TRACK: "green", Status.DRIFTING: "bold red", Status.UNCERTAIN: "yellow"}
EVENT_STYLE = {
    EventType.PLAN: "cyan", EventType.TOOL_CALL: "blue", EventType.TOOL_RESPONSE: "dim",
    EventType.CITATION: "dim", EventType.FINDING: "white", EventType.AGENT_MESSAGE: "magenta",
    EventType.ERROR: "red", EventType.PROGRESS: "dim",
}


@dataclass
class SwarmRun:
    run_id: str
    run_dir: Path
    interventions: dict[str, dict] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)


class Swarm:
    def __init__(
        self,
        scenario: Scenario,
        *,
        worker_llm,
        supervisor: Supervisor,
        research: FinanceResearchClient,
        prompts_dir: Path,
        runs_dir: Path,
        halt_on_drift: bool = False,
        policy: Optional[AlertPolicy] = None,
        console: Optional[Console] = None,
        verbose_events: bool = True,
    ):
        self.scenario = scenario
        self.supervisor = supervisor
        self.research = research
        self.halt_on_drift = halt_on_drift
        self.policy = policy or AlertPolicy()
        self.console = console or Console()
        self.run = SwarmRun(run_id=f"{scenario.task_id}-{uuid.uuid4().hex[:6]}", run_dir=Path())
        self.run.run_dir = Path(runs_dir) / self.run.run_id
        self.run.run_dir.mkdir(parents=True, exist_ok=True)
        self.collector = TrajectoryCollector(self.run.run_dir / "events.jsonl")
        if verbose_events:
            self.collector.subscribe(self._print_event)
        self.board: list[SwarmMessage] = []
        self.workers = [
            ResearchWorker(spec, swarm_objective=scenario.swarm_objective, llm=worker_llm,
                           research=research, collector=self.collector, prompts_dir=prompts_dir)
            for spec in scenario.agents
        ]

    # ---------------- display ----------------
    def _print_event(self, ev: TrajectoryEvent) -> None:
        if ev.type in (EventType.CITATION, EventType.PROGRESS):
            return
        extra = f"  q={ev.payload.get('query')!r}" if ev.type == EventType.TOOL_CALL else ""
        if ev.type == EventType.PLAN and ev.payload.get("proposed_query"):
            extra = f"\n      → proposes: {ev.payload['proposed_query']!r}"
        self.console.print(f"[{EVENT_STYLE[ev.type]}]  {ev.event_id:<22} {ev.type.value:<13} {escape(ev.summary[:110] + extra)}[/]")

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

    async def _supervise(self, worker: ResearchWorker, proposal: Optional[ProposedAction], phase: str) -> tuple[str, str]:
        inp = self.build_input(worker, proposal)
        vd = await self.supervisor.evaluate(inp)
        level, note = self.policy.update(worker.id, worker.state.step, vd)
        record = {
            "agent_id": worker.id, "step": worker.state.step, "phase": phase,
            "alert_level": level, "alert_note": note,
            # ground-truth helper for later scoring; NEVER shown to the supervisor
            "drift_injection_active": worker.drift_active(),
            "input": inp.model_dump(mode="json"), "result": vd.model_dump(mode="json"),
        }
        self.run.decisions.append(record)
        with (self.run.run_dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._print_decision(worker.id, phase, vd, level, note)
        if note == "INTERVENTION POINT":
            self.run.interventions[worker.id] = {
                "step": worker.state.step, "phase": phase,
                "evidence_event_ids": vd.decision.evidence_event_ids,
                "drift_type": vd.decision.drift_type.value,
                "research_calls_so_far": worker.state.research_calls,
                "blocked_query": proposal.query if proposal else None,
            }
        return level, note

    # ---------------- main loop ----------------
    async def _run_worker(self, w: ResearchWorker) -> None:
        for step in range(w.spec.max_steps):
            w.state.step = step
            proposal = await w.propose()
            _, note = await self._supervise(w, proposal, "pre_action")
            if note == "INTERVENTION POINT" and self.halt_on_drift:
                self.console.print(f"[bold red]  ■ {w.id} halted at step {step} before executing {escape(repr(proposal.query))}[/]")
                return
            if proposal.action == "finish":
                break
            await w.execute(proposal)
            w.progress()
        await w.finish()
        if w.state.final_report:
            self.board.append(SwarmMessage(sender=w.id, timestamp=utc_now(), text=w.state.final_report[:400]))
        await self._supervise(w, None, "final")

    async def run_all(self) -> SwarmRun:
        self._print_header()
        await asyncio.gather(*(self._run_worker(w) for w in self.workers))
        self._write_meta()
        self._print_summary()
        return self.run

    # ---------------- reporting ----------------
    def _print_header(self) -> None:
        c = self.console
        c.rule(f"[bold]PitStop run {self.run.run_id}")
        c.print(f"[bold]Objective:[/] {self.scenario.swarm_objective}")
        for a in self.scenario.agents:
            c.print(f"[bold]{a.agent_id}[/]: {a.directive}")
        c.print(f"supervisor prompt: {self.supervisor.prompt_version} | model: {self.supervisor.llm.model} | "
                f"You.com mode: {self.research.mode} | halt_on_drift: {self.halt_on_drift}")
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
            "supervisor": {"model": self.supervisor.llm.model, "prompt_version": self.supervisor.prompt_version,
                           "usage": self.supervisor.llm.usage.as_dict()},
            "worker_usage": self.workers[0].llm.usage.as_dict() if self.workers else {},
            "research": self.research.stats(), "halt_on_drift": self.halt_on_drift,
            "finished_at": utc_now(),
        }
        (self.run.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _print_summary(self) -> None:
        c = self.console
        c.rule("[bold]Run summary (single run: anecdotal, not a benchmark)")
        gt = self._ground_truth()
        for a in self.scenario.agents:
            g, iv = gt[a.agent_id], self.run.interventions.get(a.agent_id)
            if g["drift_injected"]:
                if iv:
                    delay = iv["step"] - g["onset_step"]
                    c.print(f"{a.agent_id}: drift injected at step {g['onset_step']} → flagged at step {iv['step']} "
                            f"({iv['phase']}, delay {delay} steps) evidence={iv['evidence_event_ids']}")
                else:
                    c.print(f"[red]{a.agent_id}: drift injected at step {g['onset_step']} → NOT flagged[/]")
            else:
                c.print(f"{a.agent_id}: no injection → " + (f"[red]flagged at step {iv['step']} (possible false alarm, "
                        f"or genuine uninjected drift: inspect the trace)[/]" if iv else "[green]not flagged[/]"))
        c.print(f"supervisor usage: {self.supervisor.llm.usage.as_dict()}")
        c.print(f"You.com: {self.research.stats()}")
        c.print(f"logs: {self.run.run_dir}")
