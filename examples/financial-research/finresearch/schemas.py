"""Normalized schemas shared by the collector, workers, supervisor and (later) GEPA.

SupervisorInput is the contract: the supervisor only ever sees what is in it,
and it only contains events up to the observation cutoff.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventType(str, Enum):
    PLAN = "plan"                    # agent's voluntary plan summary + proposed action
    TOOL_CALL = "tool_call"          # a research query sent to You.com
    TOOL_RESPONSE = "tool_response"  # digest of the You.com answer
    CITATION = "citation"            # one source returned by You.com
    FINDING = "finding"              # a claim the agent chose to record
    AGENT_MESSAGE = "agent_message"  # messages / final report
    ERROR = "error"
    PROGRESS = "progress"
    SUPERVISOR_MESSAGE = "supervisor_message"  # a correction the supervisor sent this agent (visible to it)


class TrajectoryEvent(BaseModel):
    event_id: str
    agent_id: str
    step: int
    seq: int
    type: EventType
    timestamp: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: Optional[str] = None


class ProposedAction(BaseModel):
    action: Literal["research", "finish"]
    query: Optional[str] = None
    rationale: str = ""
    plan_event_id: Optional[str] = None  # the PLAN event that announced it


class Budget(BaseModel):
    steps_used: int
    steps_max: int
    research_calls_used: int
    research_calls_max: int
    est_cost_usd: float = 0.0


class SwarmMessage(BaseModel):
    sender: str
    timestamp: str
    text: str


class SupervisorInput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    agent_id: str
    swarm_objective: str
    agent_directive: str
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    trajectory: list[TrajectoryEvent]
    current_plan: Optional[str] = None
    proposed_next_action: Optional[ProposedAction] = None
    relevant_swarm_messages: list[SwarmMessage] = Field(default_factory=list)
    remaining_budget: Budget
    evaluation_step: int
    observation_cutoff_seq: int
    timestamp: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_trajectory(self) -> "SupervisorInput":
        seen: set[str] = set()
        last_seq = -1
        for ev in self.trajectory:
            if ev.agent_id != self.agent_id:
                raise ValueError(f"event {ev.event_id} belongs to {ev.agent_id}, not {self.agent_id}")
            if ev.seq > self.observation_cutoff_seq:
                raise ValueError(f"event {ev.event_id} (seq {ev.seq}) is beyond cutoff {self.observation_cutoff_seq}")
            if ev.event_id in seen:
                raise ValueError(f"duplicate event id {ev.event_id}")
            if ev.seq <= last_seq:
                raise ValueError("trajectory must be strictly ordered by seq")
            seen.add(ev.event_id)
            last_seq = ev.seq
        pa = self.proposed_next_action
        if pa and pa.plan_event_id and pa.plan_event_id not in seen:
            raise ValueError("proposed_next_action.plan_event_id is not in the visible trajectory")
        return self

    def event_ids(self) -> set[str]:
        return {e.event_id for e in self.trajectory}


class Status(str, Enum):
    ON_TRACK = "ON_TRACK"
    DRIFTING = "DRIFTING"
    UNCERTAIN = "UNCERTAIN"


class DriftType(str, Enum):
    NONE = "none"
    GOAL_DEVIATION = "goal_deviation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    UNPRODUCTIVE_LOOP = "unproductive_loop"
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"
    IGNORED_EVIDENCE = "ignored_evidence"
    OBSOLETE_DIRECTIVE = "obsolete_directive"
    OTHER = "other"


class SupervisorDecision(BaseModel):
    agent_id: str
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    drift_type: DriftType = DriftType.NONE
    evidence_event_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class ValidatedDecision(BaseModel):
    """The model's decision after code-side validation, plus telemetry."""
    decision: SupervisorDecision
    valid_json: bool
    issues: list[str] = Field(default_factory=list)
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    downgraded: bool = False
    raw_text: str = ""
    model: str = ""
    prompt_version: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------- scenario definitions ----------

class DriftInjection(BaseModel):
    """Induced drift for demos and labeled data. Hidden from the supervisor."""
    at_step: int
    instruction: str
    expected_drift_type: DriftType = DriftType.GOAL_DEVIATION


class AgentSpec(BaseModel):
    agent_id: str
    label: Optional[str] = None  # display name for the UI; derived from agent_id when unset
    directive: str
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    max_steps: int = 5
    drift_injection: Optional[DriftInjection] = None


class Scenario(BaseModel):
    task_id: str
    swarm_objective: str
    agents: list[AgentSpec]
