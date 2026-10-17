from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .crusoe_client import ModelClient
from .policy_store import Policy
from .schemas import EvaluationRecord, EventType, SupervisorInput, TrajectoryEvent
from .supervisor import Supervisor

log = logging.getLogger("gepa_financial_agent.session_manager")


@dataclass(frozen=True)
class TriggerConfig:
    every_n_events: int = 4
    on_event_types: frozenset[EventType] = frozenset({EventType.ERROR, EventType.PROPOSED_NEXT_ACTION,
                                                      EventType.TASK_COMPLETION})
    repeated_tool_call_threshold: int = 3

    def should_trigger(self, session: "SupervisorSession", event: TrajectoryEvent) -> bool:
        if event.event_type in self.on_event_types:
            return True
        if session.events_since_eval >= self.every_n_events:
            return True
        if event.event_type == EventType.TOOL_CALL and self.repeated_tool_call_threshold > 0:
            key = _tool_key(event)
            n = sum(1 for e in session.events if e.event_type == EventType.TOOL_CALL and _tool_key(e) == key)
            if n >= self.repeated_tool_call_threshold:
                return True
        return False


def _tool_key(e: TrajectoryEvent) -> str:
    return f"{e.tool_name}|{json.dumps(e.tool_arguments or {}, sort_keys=True)}"


@dataclass
class SupervisorSession:
    """Isolated supervision context for one agent/task. Holds only this agent's events."""
    session_id: str
    agent_id: str
    task_id: str
    directive: str
    swarm_objective: str
    policy: Policy
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    events: list[TrajectoryEvent] = field(default_factory=list)
    records: list[EvaluationRecord] = field(default_factory=list)
    events_since_eval: int = 0
    pending: bool = False
    evaluating: bool = False
    closed: bool = False
    created_at: float = field(default_factory=time.time)

    def add_event(self, e: TrajectoryEvent) -> None:
        if e.agent_id != self.agent_id:
            raise ValueError(f"event agent_id {e.agent_id!r} does not match session agent {self.agent_id!r}")
        self.events.append(e)
        self.events_since_eval += 1

    def build_input(self) -> SupervisorInput:
        proposed = next((e.content for e in reversed(self.events)
                         if e.event_type == EventType.PROPOSED_NEXT_ACTION and isinstance(e.content, str)), None)
        return SupervisorInput(
            task_id=self.task_id, agent_id=self.agent_id, supervisor_session_id=self.session_id,
            swarm_objective=self.swarm_objective, agent_directive=self.directive,
            constraints=list(self.constraints), success_criteria=list(self.success_criteria),
            trajectory=list(self.events), proposed_next_action=proposed,
            relevant_dependencies=list(self.dependencies),
            observation_step=max((e.step for e in self.events), default=0),
        )

    def export(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "agent_id": self.agent_id, "task_id": self.task_id,
            "directive": self.directive, "swarm_objective": self.swarm_objective,
            "constraints": list(self.constraints), "success_criteria": list(self.success_criteria),
            "dependencies": list(self.dependencies),
            "policy": {"name": self.policy.name, "version": self.policy.version, "text": self.policy.text},
            "events": [e.model_dump(mode="json") for e in self.events],
            "records": [r.model_dump(mode="json") for r in self.records],
            "created_at": self.created_at,
        }


class SupervisorManager:
    """Attaches one isolated supervisor session per agent; shares one model client and one policy."""

    def __init__(self, model_client: ModelClient, policy_path: str | Path = "policies/current.txt", *,
                 policy: Policy | None = None, max_concurrent_inferences: int = 4,
                 trigger: TriggerConfig | None = None, max_events_in_context: int = 40,
                 json_mode: bool = False, max_tokens: int = 1024) -> None:
        self.client = model_client
        self.policy = policy or Policy.from_file(policy_path)
        self.trigger = trigger or TriggerConfig()
        self.max_concurrent_inferences = max_concurrent_inferences
        self.max_events_in_context, self.json_mode, self.max_tokens = max_events_in_context, json_mode, max_tokens
        self.sessions: dict[str, SupervisorSession] = {}
        self._supervisors: dict[str, Supervisor] = {}
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None

    # ---- registration -------------------------------------------------------------------------
    def set_policy(self, policy: Policy) -> None:
        """Future registrations use this policy; existing sessions keep theirs (reproducible evaluations)."""
        self.policy = policy

    def register_agent(self, agent_id: str, task_id: str, directive: str, swarm_objective: str,
                       constraints=(), success_criteria=(), dependencies=(), policy: Policy | None = None
                       ) -> SupervisorSession:
        if agent_id in self.sessions:
            raise ValueError(f"agent {agent_id!r} already has an active session")
        s = SupervisorSession(session_id=f"supervisor_{agent_id}_{uuid.uuid4().hex[:8]}", agent_id=agent_id,
                              task_id=task_id, directive=directive, swarm_objective=swarm_objective,
                              policy=policy or self.policy, constraints=list(constraints),
                              success_criteria=list(success_criteria), dependencies=list(dependencies))
        self.sessions[agent_id] = s
        log.info("registered session %s for agent=%s task=%s policy=%s", s.session_id, agent_id, task_id, s.policy.version)
        return s

    def _supervisor_for(self, policy: Policy) -> Supervisor:
        sup = self._supervisors.get(policy.version)
        if sup is None:
            sup = Supervisor(self.client, policy, max_tokens=self.max_tokens, json_mode=self.json_mode,
                             max_events=self.max_events_in_context)
            self._supervisors[policy.version] = sup
        return sup

    # ---- ingestion ----------------------------------------------------------------------------
    def observe_sync(self, agent_id: str, event: TrajectoryEvent | dict[str, Any]) -> bool:
        s = self.sessions[agent_id]  # KeyError for unknown agent is intentional
        e = event if isinstance(event, TrajectoryEvent) else TrajectoryEvent.model_validate(event)
        s.add_event(e)
        if self.trigger.should_trigger(s, e):
            s.pending = True
        return s.pending

    async def observe(self, agent_id: str, event: TrajectoryEvent | dict[str, Any]) -> bool:
        return self.observe_sync(agent_id, event)

    # ---- evaluation ---------------------------------------------------------------------------
    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self.max_concurrent_inferences), loop
        return self._sem

    async def _evaluate(self, s: SupervisorSession) -> EvaluationRecord:
        async with self._semaphore():
            s.evaluating, s.pending, s.events_since_eval = True, False, 0
            try:
                rec = await self._supervisor_for(s.policy).evaluate(s.build_input())
            finally:
                s.evaluating = False
            s.records.append(rec)
            return rec

    async def evaluate_pending(self) -> list[EvaluationRecord]:
        targets = [s for s in self.sessions.values() if s.pending and not s.evaluating and s.events]
        if not targets:
            return []
        return list(await asyncio.gather(*(self._evaluate(s) for s in targets)))

    async def evaluate_now(self, agent_id: str) -> EvaluationRecord:
        return await self._evaluate(self.sessions[agent_id])

    # ---- results / lifecycle ------------------------------------------------------------------
    def decisions(self, agent_id: str) -> list[EvaluationRecord]:
        return list(self.sessions[agent_id].records)

    def latest(self, agent_id: str) -> EvaluationRecord | None:
        recs = self.sessions[agent_id].records
        return recs[-1] if recs else None

    def close_session(self, agent_id: str) -> SupervisorSession:
        s = self.sessions.pop(agent_id)
        s.closed = True
        return s

    def export_session(self, agent_id: str) -> dict[str, Any]:
        return self.sessions[agent_id].export()

    async def replay(self, export: dict[str, Any], agent_id: str | None = None) -> list[EvaluationRecord]:
        """Re-run supervision over an exported session's events with the same triggers (deterministic with a mock)."""
        aid = agent_id or f"{export['agent_id']}__replay_{uuid.uuid4().hex[:4]}"
        pol = Policy.from_text(export["policy"]["text"], name=export["policy"]["name"])
        s = self.register_agent(aid, export["task_id"], export["directive"], export["swarm_objective"],
                                export.get("constraints", ()), export.get("success_criteria", ()),
                                export.get("dependencies", ()), policy=pol)
        out: list[EvaluationRecord] = []
        for raw in export["events"]:
            e = TrajectoryEvent.model_validate({**raw, "agent_id": aid})
            if self.observe_sync(aid, e):
                out.append(await self._evaluate(s))
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "sessions": len(self.sessions),
            "pending": sum(1 for s in self.sessions.values() if s.pending),
            "evaluations": sum(len(s.records) for s in self.sessions.values()),
            "usage": self.client.tracker.snapshot(),
            "policy": {"name": self.policy.name, "version": self.policy.version},
        }
