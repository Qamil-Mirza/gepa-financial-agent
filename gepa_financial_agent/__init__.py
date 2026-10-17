"""GEPA Financial Agent: task-scoped drift supervision with GEPA-optimized prompts."""
from .crusoe_client import CrusoeClient, MockModelClient
from .policy_store import Policy, PolicyStore
from .schemas import (DatasetExample, DriftStatus, DriftType, EvaluationRecord, EventType, GroundTruth,
                      SupervisorDecision, SupervisorInput, TrajectoryEvent)
from .session_manager import SupervisorManager, TriggerConfig
from .supervisor import Supervisor

__version__ = "0.1.0"
__all__ = ["CrusoeClient", "MockModelClient", "Policy", "PolicyStore", "Supervisor", "SupervisorManager", "TriggerConfig", "DatasetExample", "DriftStatus",
           "DriftType", "EvaluationRecord", "EventType", "GroundTruth", "SupervisorDecision", "SupervisorInput",
           "TrajectoryEvent"]
