"""P0-P3 urgency levels and the action chosen for a finding."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from ..sdk.context import Criticality
from .l0 import Finding


class Level(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


_ANY = {"high": Level.P0, "medium": Level.P0, "low": Level.P0}
_TABLE: dict[tuple[str, bool | None], dict[str, Level]] = {
    ("budget_exhausted", None): _ANY,
    ("out_of_scope_action", True): _ANY,
    ("out_of_scope_action", False): {"high": Level.P1, "medium": Level.P2, "low": Level.P3},
    ("unproductive_loop", True): _ANY,
    ("unproductive_loop", False): {"high": Level.P0, "medium": Level.P1, "low": Level.P2},
    ("tool_call_breakage", None): {"high": Level.P1, "medium": Level.P1, "low": Level.P2},
    ("refusal_spike", None): {"high": Level.P1, "medium": Level.P1, "low": Level.P2},
}
_OTHER = {"high": Level.P2, "medium": Level.P2, "low": Level.P3}


def classify(drift_type: str, criticality: Criticality, side_effects: bool) -> Level:
    row = _TABLE.get((drift_type, side_effects)) or _TABLE.get((drift_type, None)) or _OTHER
    return row[criticality]


def action_for(finding: Finding, level: Level, criticality: Criticality) -> Literal["block", "steer", "halt", "record"]:
    """The in-the-moment action for a finding. "fallback" is never chosen here: `pitstop.sdk.actions` upgrades a
    P1 "record" to "fallback" afterwards, when the workflow pins a fallback model and the run follows a model or
    provider change (it needs the store, which this table does not read)."""
    if finding.drift_type == "budget_exhausted":
        return "halt"
    if finding.kind == "off_task":
        return "block" if finding.side_effects else "steer"
    return "record"
