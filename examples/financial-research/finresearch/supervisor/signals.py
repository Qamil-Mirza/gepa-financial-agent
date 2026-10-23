"""Deterministic, observable statistics handed to the supervisor as context.

These are not verdicts. They make cheap-to-compute patterns (repeated queries,
stalled progress, uncited findings) visible so drift can be caught early.
"""
from __future__ import annotations

from pitstop.detect.text import jaccard

from ..schemas import EventType, SupervisorInput


def compute_signals(inp: SupervisorInput) -> dict:
    traj = inp.trajectory
    queries = [e.payload.get("query", "") for e in traj if e.type == EventType.TOOL_CALL]
    findings = [e for e in traj if e.type == EventType.FINDING]
    errors = [e for e in traj if e.type == EventType.ERROR]

    near_dupes = sum(
        1 for i in range(len(queries)) for j in range(i) if jaccard(queries[i], queries[j]) >= 0.7
    )
    proposed = inp.proposed_next_action
    proposed_sim = 0.0
    if proposed and proposed.query and queries:
        proposed_sim = max(jaccard(proposed.query, q) for q in queries)

    last_finding_step = max((e.step for e in findings), default=-1)
    current_step = inp.evaluation_step
    b = inp.remaining_budget
    return {
        "research_calls_made": len(queries),
        "near_duplicate_query_pairs": near_dupes,
        "proposed_query_max_similarity_to_past": round(proposed_sim, 2),
        "findings_recorded": len(findings),
        "findings_without_citations": sum(1 for f in findings if not f.payload.get("citation_event_ids")),
        "steps_since_last_finding": (current_step - last_finding_step) if last_finding_step >= 0 else current_step,
        "errors": len(errors),
        "step_budget_used": f"{b.steps_used}/{b.steps_max}",
    }
