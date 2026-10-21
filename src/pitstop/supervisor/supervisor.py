"""Supervisor: Crusoe-hosted Nemotron judging one agent's observable trajectory.

The system prompt (the policy GEPA will optimize) lives in a versioned text
file. The output format and all validation live in code, so an optimized
prompt can change *judgment* but cannot break the contract or invent evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..sdk.errors import Halted
from ..sdk.wrap import meter
from ..llm import LLMClient, LLMError, extract_json_object
from ..schemas import (
    DriftType,
    EventType,
    Status,
    SupervisorDecision,
    SupervisorInput,
    TrajectoryEvent,
    ValidatedDecision,
)
from .signals import compute_signals

OUTPUT_CONTRACT = """\
Return ONLY a JSON object, no prose before or after it:
{
  "agent_id": "<the monitored agent id>",
  "status": "ON_TRACK" | "DRIFTING" | "UNCERTAIN",
  "confidence": <number between 0 and 1>,
  "drift_type": "none" | "goal_deviation" | "constraint_violation" | "unproductive_loop" | "unsupported_conclusion" | "ignored_evidence" | "obsolete_directive" | "other",
  "evidence_event_ids": ["<event ids copied exactly from the trajectory above>"],
  "explanation": "<two or three sentences grounded in the cited events>"
}
Use drift_type "none" when status is ON_TRACK. Only cite event IDs that appear in the trajectory above."""

MAX_DIGEST_CHARS = 700
MAX_REPORT_CHARS = 2000
_SHORT_ID = re.compile(r"e\d+")


def load_prompt(path: Path) -> tuple[str, str]:
    text = Path(path).read_text().strip()
    version = f"{Path(path).stem}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    return text, version


def render_event(ev: TrajectoryEvent) -> str:
    head = f"[{ev.event_id}] step {ev.step} | {ev.type.value} | {ev.summary}"
    p = ev.payload
    lines = [head]
    if ev.type == EventType.TOOL_CALL:
        lines.append(f"    query: {p.get('query', '')!r}")
    elif ev.type == EventType.TOOL_RESPONSE:
        digest = (p.get("digest") or "")[:MAX_DIGEST_CHARS]
        lines.append(f"    response excerpt (tool output, not the agent's choice): {digest!r}")
    elif ev.type == EventType.CITATION:
        lines.append(f"    source: {p.get('title', '')} <{p.get('url', '')}> retrieved {p.get('retrieved_at', '')}")
    elif ev.type == EventType.FINDING:
        fields = {k: p.get(k) for k in ("metric", "value", "period") if p.get(k)}
        cites = p.get("citation_event_ids") or []
        lines.append(f"    {json.dumps(fields)} cites: {cites or 'NONE'}")
    elif ev.type == EventType.PLAN and p.get("proposed_query"):
        lines.append(f"    proposed action: {p.get('proposed_action')} query={p.get('proposed_query')!r}")
    elif ev.type == EventType.SUPERVISOR_MESSAGE:
        lines.append(f"    correction sent to the agent: {p.get('message', '')[:MAX_DIGEST_CHARS]!r}")
    elif ev.type == EventType.AGENT_MESSAGE and p.get("report"):
        # The summary is truncated; the final check must see the claims it is judging.
        lines.append(f"    full report: {p['report'][:MAX_REPORT_CHARS]!r}")
    return "\n".join(lines)


def render_user_message(inp: SupervisorInput) -> str:
    b = inp.remaining_budget
    parts = [
        f"## Swarm objective\n{inp.swarm_objective}",
        f"## Monitored agent: {inp.agent_id}",
        f"### Directive\n{inp.agent_directive}",
        "### Constraints\n" + ("\n".join(f"- {c}" for c in inp.constraints) or "- (none)"),
        "### Success criteria\n" + ("\n".join(f"- {c}" for c in inp.success_criteria) or "- (none)"),
        f"### Budget\nsteps used {b.steps_used}/{b.steps_max}; research calls {b.research_calls_used}/{b.research_calls_max}",
    ]
    if inp.relevant_swarm_messages:
        parts.append("### Swarm messages\n" + "\n".join(f"- {m.sender}: {m.text}" for m in inp.relevant_swarm_messages))
    parts.append(
        f"### Observable trajectory (evaluation step {inp.evaluation_step}; events up to seq "
        f"{inp.observation_cutoff_seq} — this is everything you can see)\n"
        + ("\n".join(render_event(e) for e in inp.trajectory) or "(no events yet)")
    )
    if inp.current_plan:
        parts.append(f"### Agent's current plan (self-reported)\n{inp.current_plan}")
    pa = inp.proposed_next_action
    if pa:
        q = f" query={pa.query!r}" if pa.query else ""
        parts.append(
            f"### Proposed next action (NOT yet executed; announced in {pa.plan_event_id})\n"
            f"{pa.action}{q}\nrationale: {pa.rationale}"
        )
    parts.append(
        "### Observable statistics (computed deterministically; context, not a verdict)\n"
        + json.dumps(compute_signals(inp), indent=2)
    )
    parts.append("## Output\n" + OUTPUT_CONTRACT)
    return "\n\n".join(parts)


def _norm_enum(value, enum_cls, default):
    if value is None:
        return default, False
    v = str(value).strip()
    for cand in (v, v.upper(), v.lower(), v.lower().replace(" ", "_").replace("-", "_"), v.upper().replace(" ", "_")):
        try:
            return enum_cls(cand), True
        except ValueError:
            continue
    return default, False


def parse_and_validate(raw_text: str, inp: SupervisorInput) -> ValidatedDecision:
    """Pure function: raw model text -> validated decision. Reused by the GEPA evaluator."""
    issues: list[str] = []
    try:
        obj = extract_json_object(raw_text)
    except ValueError:
        return ValidatedDecision(
            decision=SupervisorDecision(
                agent_id=inp.agent_id, status=Status.UNCERTAIN, confidence=0.0,
                explanation="Supervisor output was not valid JSON.",
            ),
            valid_json=False,
            issues=["invalid_json"],
            raw_text=raw_text,
        )

    if obj.get("agent_id") != inp.agent_id:
        issues.append(f"agent_id_mismatch:{obj.get('agent_id')!r}")
    status, ok = _norm_enum(obj.get("status"), Status, Status.UNCERTAIN)
    if not ok:
        issues.append(f"invalid_status:{obj.get('status')!r}")
    drift_type, ok = _norm_enum(obj.get("drift_type"), DriftType, DriftType.OTHER)
    if not ok:
        issues.append(f"invalid_drift_type:{obj.get('drift_type')!r}")
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf, _ = 0.0, issues.append("invalid_confidence")
    if not 0.0 <= conf <= 1.0:
        issues.append("confidence_out_of_range")
        conf = min(max(conf, 0.0), 1.0)

    raw_ids = obj.get("evidence_event_ids") or []
    if not isinstance(raw_ids, list):
        raw_ids, _ = [], issues.append("evidence_not_a_list")
    visible = inp.event_ids()
    # Models often cite "e005" for "agent_x-e005". Expand only when that exact event is visible;
    # anything else still counts as invented.
    ids, expanded = [], []
    for i in map(str, raw_ids):
        full = f"{inp.agent_id}-{i}"
        if i not in visible and _SHORT_ID.fullmatch(i) and full in visible:
            expanded.append(i)
            i = full
        ids.append(i)
    if expanded:
        issues.append(f"expanded_short_evidence_ids:{expanded}")
    kept = list(dict.fromkeys(i for i in ids if i in visible))
    dropped = [i for i in ids if i not in visible]
    if dropped:
        issues.append(f"invented_or_unseen_evidence:{dropped}")

    downgraded = False
    if status == Status.ON_TRACK and drift_type != DriftType.NONE:
        issues.append("on_track_with_drift_type")
        drift_type = DriftType.NONE
    if status == Status.DRIFTING and drift_type == DriftType.NONE:
        issues.append("drifting_without_drift_type")
        drift_type = DriftType.OTHER
    if status == Status.DRIFTING and not kept:
        issues.append("drift_without_valid_evidence")
        status, downgraded = Status.UNCERTAIN, True

    return ValidatedDecision(
        decision=SupervisorDecision(
            agent_id=inp.agent_id,
            status=status,
            confidence=conf,
            drift_type=drift_type,
            evidence_event_ids=kept,
            explanation=str(obj.get("explanation", ""))[:1000],
        ),
        valid_json=True,
        issues=issues,
        dropped_evidence_ids=dropped,
        downgraded=downgraded,
        raw_text=raw_text,
    )


class Supervisor:
    def __init__(self, llm: LLMClient, prompt_path: Path, *, json_retries: int = 1):
        self.llm = llm
        self.prompt_path = Path(prompt_path)
        self.system_prompt, self.prompt_version = load_prompt(self.prompt_path)
        self.json_retries = json_retries

    @classmethod
    def from_text(cls, llm: LLMClient, prompt_text: str, version: str = "inline") -> "Supervisor":
        """Used by the GEPA evaluator to score a candidate prompt without writing files."""
        obj = cls.__new__(cls)
        obj.llm, obj.prompt_path, obj.json_retries = llm, None, 1
        obj.system_prompt, obj.prompt_version = prompt_text, version
        return obj

    async def evaluate(self, inp: SupervisorInput) -> ValidatedDecision:
        user = render_user_message(inp)
        latency = 0.0
        ptoks = ctoks = 0
        vd = None
        for attempt in range(self.json_retries + 1):
            msg = user if attempt == 0 else user + "\n\nReply with ONLY the JSON object."
            try:
                with meter("watch"):
                    r = await self.llm.complete(self.system_prompt, msg, temperature=0.0)
            except (LLMError, Halted):  # a budget/drift halt must reach the caller, not become UNCERTAIN
                raise
            except Exception as e:  # network/provider failure: fail safe, never crash the swarm
                vd = ValidatedDecision(
                    decision=SupervisorDecision(agent_id=inp.agent_id, status=Status.UNCERTAIN, confidence=0.0,
                                                explanation=f"Supervisor call failed: {e}"),
                    valid_json=False, issues=["supervisor_call_failed"],
                )
                break
            latency += r.latency_s
            ptoks += r.prompt_tokens
            ctoks += r.completion_tokens
            vd = parse_and_validate(r.text, inp)
            if vd.valid_json:
                if attempt:
                    vd.issues.append("needed_json_retry")
                break
        vd.model = self.llm.model
        vd.prompt_version = self.prompt_version
        vd.latency_s, vd.prompt_tokens, vd.completion_tokens = latency, ptoks, ctoks
        return vd
