from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESEARCH_FINDING = "research_finding"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PROPOSED_NEXT_ACTION = "proposed_next_action"
    TASK_COMPLETION = "task_completion"


class DriftStatus(str, Enum):
    ON_TRACK = "ON_TRACK"
    DRIFTING = "DRIFTING"
    UNCERTAIN = "UNCERTAIN"


class DriftType(str, Enum):
    GOAL_DEVIATION = "GOAL_DEVIATION"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    UNPRODUCTIVE_LOOP = "UNPRODUCTIVE_LOOP"
    UNSUPPORTED_CONCLUSION = "UNSUPPORTED_CONCLUSION"
    OBSOLETE_DIRECTIVE = "OBSOLETE_DIRECTIVE"
    OTHER = "OTHER"


class RecommendedAction(str, Enum):
    NONE = "NONE"
    CONTINUE_MONITORING = "CONTINUE_MONITORING"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    ESCALATE = "ESCALATE"


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str | None = None
    title: str | None = None
    retrieved_at: str | None = None
    snippet: str | None = None


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    task_id: str
    agent_id: str
    step: int = Field(ge=0)
    event_type: EventType
    timestamp: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    content: str | dict[str, Any] | None = None
    source_references: list[SourceReference] = Field(default_factory=list)


class SupervisorInput(BaseModel):
    """Normalized, framework-independent input. Never contains ground truth."""
    model_config = ConfigDict(extra="forbid")
    task_id: str
    agent_id: str
    supervisor_session_id: str | None = None
    swarm_objective: str
    agent_directive: str
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    trajectory: list[TrajectoryEvent] = Field(default_factory=list)
    current_plan: str | None = None
    proposed_next_action: str | None = None
    relevant_dependencies: list[str] = Field(default_factory=list)
    remaining_budget: dict[str, Any] | None = None
    observation_step: int = 0

    @model_validator(mode="after")
    def _unique_event_ids(self) -> "SupervisorInput":
        ids = [e.event_id for e in self.trajectory]
        if len(ids) != len(set(ids)):
            raise ValueError("trajectory event_ids must be unique")
        return self

    def event_ids(self) -> set[str]:
        return {e.event_id for e in self.trajectory}


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str
    task_id: str
    status: DriftStatus
    confidence: float = Field(ge=0.0, le=1.0)
    drift_type: DriftType | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    recommended_action: RecommendedAction = RecommendedAction.NONE
    policy_version: str | None = None
    observation_step: int | None = None

    @model_validator(mode="after")
    def _normalize_drift_type(self) -> "SupervisorDecision":
        if self.status == DriftStatus.DRIFTING and self.drift_type is None:
            self.drift_type = DriftType.OTHER
        if self.status != DriftStatus.DRIFTING:
            self.drift_type = None
        return self


class EvaluationRecord(BaseModel):
    """One supervisor inference: decision or error, plus telemetry."""
    model_config = ConfigDict(extra="forbid")
    agent_id: str
    task_id: str
    observation_step: int
    policy_version: str
    model: str
    mocked: bool = False
    decision: SupervisorDecision | None = None
    error: str | None = None
    raw_output: str | None = None
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    example_id: str | None = None


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: DriftStatus
    drift_type: DriftType | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    drift_onset_step: int | None = None


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str
    scenario_id: str
    observation_step: int = Field(ge=0)
    supervisor_input: SupervisorInput
    ground_truth: GroundTruth
    split_hint: str | None = None  # "train" | "val" | "test" to force a holdout
    tags: list[str] = Field(default_factory=list)

    @field_validator("split_hint")
    @classmethod
    def _split_hint_ok(cls, v: str | None) -> str | None:
        if v is not None and v not in {"train", "val", "test"}:
            raise ValueError("split_hint must be train, val, or test")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetExample":
        inner = self.supervisor_input
        if "observation_step" not in inner.model_fields_set:
            inner.observation_step = self.observation_step
        elif inner.observation_step != self.observation_step:
            raise ValueError(f"supervisor_input.observation_step {inner.observation_step} != "
                             f"observation_step {self.observation_step}")
        ids = self.supervisor_input.event_ids()
        missing = [i for i in self.ground_truth.evidence_event_ids if i not in ids]
        if missing:
            raise ValueError(f"ground_truth.evidence_event_ids not in trajectory: {missing}")
        if self.supervisor_input.trajectory:
            last = max(e.step for e in self.supervisor_input.trajectory)
            if self.observation_step < last:
                raise ValueError(f"observation_step {self.observation_step} < last event step {last}")
        return self


def split_evidence_ids(ids: list[str], inp: SupervisorInput) -> tuple[list[str], list[str]]:
    """Return (valid, invalid) evidence IDs, order-preserving and deduplicated."""
    known = inp.event_ids()
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        (valid if i in known else invalid).append(i)
    return valid, invalid
