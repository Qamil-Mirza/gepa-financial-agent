from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..schemas import DatasetExample, TrajectoryEvent
from ..session_manager import SupervisorManager, SupervisorSession
from .base import EventStreamAdapter


@dataclass
class ReplayAgent:
    """Replays a frozen trajectory as a live worker would emit it. Stand-in for the teammate's real agents."""
    agent_id: str
    task_id: str
    directive: str
    swarm_objective: str
    events: list[TrajectoryEvent]
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    delay_s: float = 0.0

    @classmethod
    def from_example(cls, example: DatasetExample, delay_s: float = 0.0, agent_id: str | None = None) -> "ReplayAgent":
        inp = example.supervisor_input
        aid = agent_id or inp.agent_id
        events = [e.model_copy(update={"agent_id": aid}) for e in inp.trajectory]
        return cls(agent_id=aid, task_id=inp.task_id, directive=inp.agent_directive, swarm_objective=inp.swarm_objective,
                   events=events, constraints=list(inp.constraints), success_criteria=list(inp.success_criteria),
                   delay_s=delay_s)

    def register(self, manager: SupervisorManager) -> SupervisorSession:
        return manager.register_agent(self.agent_id, self.task_id, self.directive, self.swarm_objective,
                                      self.constraints, self.success_criteria)

    async def run(self, adapter: EventStreamAdapter) -> None:
        for e in self.events:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            await adapter.emit(self.agent_id, e)


def pick_full_example(examples: list[DatasetExample], kind: str, company: str, variant: int = 0) -> DatasetExample:
    matches = [e for e in examples if f"kind:{kind}" in e.tags and f"company:{company}" in e.tags
               and f"variant:{variant}" in e.tags]
    if not matches:
        raise LookupError(f"no fixture with kind={kind} company={company} variant={variant}")
    return max(matches, key=lambda e: e.observation_step)
