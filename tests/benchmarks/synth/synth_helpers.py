"""Hand-built traces for tests. These are test fixtures, not dataset content."""
from __future__ import annotations

from typing import List, Optional

from synth.schemas import Generation, SourceRef, SteerSpec, TaskSpec, TraceEvent, TraceRecord
from synth.tasks import TASKS

FAMILY = "nvda_datacenter"
SRC = SourceRef(source_id="S1.1", url="https://example.com/q2", title="Q2 results")


class Builder:
    def __init__(self):
        t = TASKS[FAMILY]
        self.task = TaskSpec(task_id=FAMILY, agent_id=t.agent_id, swarm_objective=t.swarm_objective,
                             agent_directive=t.directive, constraints=t.constraints,
                             success_criteria=t.success_criteria)
        self.events: List[TraceEvent] = []

    def _add(self, step, etype, content=None, **kw):
        self.events.append(TraceEvent(event_id=f"evt_{len(self.events) + 1:03d}", task_id=FAMILY,
                                      agent_id=self.task.agent_id, step=step, event_type=etype,
                                      timestamp="2026-10-17T00:00:00Z", content=content, **kw))
        return self

    def search(self, step, query, result="NVIDIA Data Center revenue was $89.0 billion, up 117%."):
        self._add(step, "tool_call", query, tool_name="finance_research", tool_arguments={"query": query})
        return self._add(step, "tool_result", result, tool_name="finance_research", source_references=[SRC])

    def finding(self, step, claim, cited=True):
        return self._add(step, "research_finding", {"claim": claim, "value": None}, tool_name="record_finding",
                         source_references=[SRC] if cited else [])

    def error(self, step):
        return self._add(step, "error", "finance_research failed: upstream timeout", tool_name="finance_research")

    def update(self, step, text):
        return self._add(step, "status_update", text)

    def build(self, steer: SteerSpec, trace_id: str = "tr_test") -> TraceRecord:
        gen = Generation(steer=steer, model="test", temperature=0.0, evidence_source="fixture",
                         created_at="2026-10-17T00:00:00Z")
        return TraceRecord(trace_id=trace_id, scenario_family=FAMILY, task=self.task, events=self.events,
                           generation=gen)


def drift(t: str, x: int, variant: Optional[str] = None) -> SteerSpec:
    return SteerSpec(steer_id=f"drift:{t.lower()}", kind="drift", drift_type=t, step_x=x, variant=variant)


def benign(variant: str, x: int) -> SteerSpec:
    return SteerSpec(steer_id=f"benign:{variant}", kind="benign", variant=variant, step_x=x)


NONE = SteerSpec(steer_id="none", kind="none")


def on_task_prefix(b: Builder, upto: int) -> Builder:
    periods = ["Q2 fiscal 2027", "Q1 fiscal 2027", "Q4 fiscal 2026", "Q3 fiscal 2026", "Q2 fiscal 2026"]
    for s in range(1, upto + 1):
        if s % 2:
            b.search(s, f"NVIDIA Data Center revenue {periods[(s // 2) % len(periods)]}")
        else:
            b.finding(s, "NVIDIA Data Center revenue was $89.0 billion, up 117% year over year")
    return b
