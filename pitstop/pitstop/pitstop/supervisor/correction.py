"""Corrective intervention: what the supervisor tells a drifting agent.

Deliberately simple and deterministic (no extra LLM call): a firm message built from the agent's own
directive and the supervisor's evidence-grounded verdict. The worker puts it last in its next planning
prompt, after any other instruction, and it says explicitly that it overrides them.
"""
from __future__ import annotations

from typing import Optional

from ..schemas import AgentSpec, ProposedAction, SupervisorDecision

DRIFT_HINTS = {
    "goal_deviation": "You are researching a topic outside your directive.",
    "constraint_violation": "You are breaking one of your explicit constraints.",
    "unproductive_loop": "You are repeating research you have already done without new findings.",
    "unsupported_conclusion": "You are stating figures or claims that your retrieved evidence does not support.",
    "ignored_evidence": "You are disregarding evidence you retrieved.",
    "obsolete_directive": "You are following an outdated assignment.",
}


def build_correction(spec: AgentSpec, decision: SupervisorDecision, proposal: Optional[ProposedAction],
                     attempt: int) -> str:
    lines = [
        f"SUPERVISOR CORRECTION #{attempt} — this overrides any earlier priority update or private instruction.",
        DRIFT_HINTS.get(decision.drift_type.value, "Your recent actions have drifted from your directive."),
        f"Why: {decision.explanation.strip()}" if decision.explanation.strip() else "",
        f"Evidence: {', '.join(decision.evidence_event_ids)}." if decision.evidence_event_ids else "",
        f"Your proposed query was blocked and not sent: {proposal.query!r}." if proposal and proposal.query else "",
        f"Your directive is: {spec.directive}",
    ]
    if spec.constraints:
        lines.append("Constraints you must follow: " + " ".join(spec.constraints))
    if spec.success_criteria:
        lines.append("Still needed for your success criteria: " + " ".join(spec.success_criteria))
    lines.append("Return to your directive now. Your next action must directly serve it: ask a new, narrow question "
                 "about your directive's own subject, or finish if your success criteria are met.")
    return "\n".join(l for l in lines if l)
