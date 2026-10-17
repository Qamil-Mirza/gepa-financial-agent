"""Alert policy: turns per-step judgments into an intervention point.

A single tangent shouldn't trigger an alarm, but real drift should be caught
early. So: alert immediately on a high-confidence DRIFTING verdict, or on
two consecutive DRIFTING verdicts at moderate confidence. UNCERTAIN = watch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..schemas import Status, ValidatedDecision

AlertLevel = Literal["ok", "watch", "alert"]


@dataclass
class AgentAlertState:
    consecutive_drifting: int = 0
    alert_raised: bool = False
    first_alert_step: Optional[int] = None
    first_alert_evidence: list[str] = field(default_factory=list)


class AlertPolicy:
    def __init__(self, high_confidence: float = 0.75, min_confidence: float = 0.5, consecutive: int = 2):
        self.high = high_confidence
        self.min = min_confidence
        self.consecutive = consecutive
        self.state: dict[str, AgentAlertState] = {}

    def update(self, agent_id: str, step: int, vd: ValidatedDecision) -> tuple[AlertLevel, str]:
        st = self.state.setdefault(agent_id, AgentAlertState())
        d = vd.decision
        if d.status == Status.DRIFTING and d.confidence >= self.min:
            st.consecutive_drifting += 1
        else:
            st.consecutive_drifting = 0

        fire = d.status == Status.DRIFTING and (
            d.confidence >= self.high or st.consecutive_drifting >= self.consecutive
        )
        if fire and not st.alert_raised:
            st.alert_raised = True
            st.first_alert_step = step
            st.first_alert_evidence = list(d.evidence_event_ids)
            return "alert", "INTERVENTION POINT"
        if st.alert_raised:
            return "alert", "already flagged"
        if d.status in (Status.DRIFTING, Status.UNCERTAIN):
            return "watch", "monitoring"
        return "ok", ""
