"""Pydantic schemas for traces (the trace DB) and exported dataset examples (the GEPA contract).

Supervisor-visible data lives in TaskSpec + TraceEvent. Everything under TraceRecord.generation
is ground truth / audit material and must never be shown to the supervisor.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

EventType = Literal[
    "agent_message",
    "tool_call",
    "tool_result",
    "research_finding",
    "status_update",
    "error",
    "proposed_next_action",
    "task_completion",
]
Status = Literal["ON_TRACK", "DRIFTING", "UNCERTAIN"]
DriftType = Literal[
    "GOAL_DEVIATION",
    "CONSTRAINT_VIOLATION",
    "UNPRODUCTIVE_LOOP",
    "UNSUPPORTED_CONCLUSION",
    "OBSOLETE_DIRECTIVE",
    "OTHER",
]
SteerKind = Literal["drift", "benign", "none"]


class SourceRef(BaseModel):
    source_id: str
    url: Optional[str] = None
    title: Optional[str] = None
    published_at: Optional[str] = None


class TraceEvent(BaseModel):
    event_id: str
    task_id: str
    agent_id: str
    step: int = Field(ge=0)
    event_type: EventType
    timestamp: str
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    content: Any = None
    source_references: List[SourceRef] = Field(default_factory=list)


class TaskSpec(BaseModel):
    task_id: str
    agent_id: str
    swarm_objective: str
    agent_directive: str
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class SteerSpec(BaseModel):
    steer_id: str
    kind: SteerKind
    drift_type: Optional[DriftType] = None
    step_x: Optional[int] = None
    # Which distractor / detour from the task template the steer uses, if any.
    variant: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "SteerSpec":
        if self.kind == "drift" and (self.drift_type is None or self.step_x is None):
            raise ValueError("drift steers need drift_type and step_x")
        if self.kind == "none" and (self.drift_type or self.step_x is not None):
            raise ValueError("'none' steers take no drift_type/step_x")
        if self.kind == "benign" and (self.drift_type or self.step_x is None):
            raise ValueError("benign steers need step_x and no drift_type")
        return self


class Verification(BaseModel):
    passed: bool
    reason: str
    observed_onset_step: Optional[int] = None
    evidence_event_ids: List[str] = Field(default_factory=list)


class Usage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    youcom_live_calls: int = 0
    youcom_cache_hits: int = 0


class Generation(BaseModel):
    """Ground truth and audit data. Never exported into supervisor_input."""

    steer: SteerSpec
    verification: Optional[Verification] = None
    model: str
    temperature: float
    seed: Optional[int] = None
    evidence_source: Literal["youcom", "fixture", "mixed"]
    usage: Usage = Field(default_factory=Usage)
    private_messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str


class TraceRecord(BaseModel):
    trace_id: str
    scenario_family: str
    task: TaskSpec
    events: List[TraceEvent]
    generation: Generation

    @model_validator(mode="after")
    def _unique_ids(self) -> "TraceRecord":
        ids = [e.event_id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event_id in trace")
        return self


# ---- Exported dataset (the contract consumed by the supervisor/GEPA side) ----


class SupervisorInput(TaskSpec):
    trajectory: List[TraceEvent]
    current_plan: Optional[str] = None
    proposed_next_action: Optional[str] = None
    relevant_dependencies: List[str] = Field(default_factory=list)
    remaining_budget: Optional[int] = None
    observation_step: int


class GroundTruth(BaseModel):
    status: Status
    drift_type: Optional[DriftType] = None
    evidence_event_ids: List[str] = Field(default_factory=list)
    rationale: str


class DatasetExample(BaseModel):
    example_id: str
    scenario_id: str
    scenario_family: str
    split: Literal["train", "val", "test"]
    observation_step: int
    drift_onset_step: Optional[int] = None
    supervisor_input: SupervisorInput
    ground_truth: GroundTruth

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetExample":
        traj = self.supervisor_input.trajectory
        if any(e.step > self.observation_step for e in traj):
            raise ValueError("trajectory contains events past observation_step")
        if self.supervisor_input.observation_step != self.observation_step:
            raise ValueError("observation_step mismatch")
        ids = {e.event_id for e in traj}
        missing = [i for i in self.ground_truth.evidence_event_ids if i not in ids]
        if missing:
            raise ValueError(f"evidence ids not in visible trajectory: {missing}")
        gt = self.ground_truth
        if gt.status == "DRIFTING" and gt.drift_type is None:
            raise ValueError("DRIFTING requires drift_type")
        if gt.status == "ON_TRACK" and gt.drift_type is not None:
            raise ValueError("ON_TRACK must not have drift_type")
        return self
