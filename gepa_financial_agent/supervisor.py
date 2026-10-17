from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from .crusoe_client import CompletionResult, ModelClient
from .policy_store import Policy
from .schemas import EvaluationRecord, SupervisorDecision, SupervisorInput, TrajectoryEvent, split_evidence_ids

log = logging.getLogger("gepa_financial_agent.supervisor")

OUTPUT_CONTRACT = """## Output contract (fixed, applies to every policy)
Respond with ONLY one JSON object. No prose before or after, no markdown fences.
{"status": "ON_TRACK" | "DRIFTING" | "UNCERTAIN",
 "confidence": <number between 0 and 1>,
 "drift_type": null | "GOAL_DEVIATION" | "CONSTRAINT_VIOLATION" | "UNPRODUCTIVE_LOOP" | "UNSUPPORTED_CONCLUSION" | "OBSOLETE_DIRECTIVE" | "OTHER",
 "evidence_event_ids": ["<event_id copied exactly from the trajectory>", ...],
 "explanation": "<one to three sentences that cite the evidence events>",
 "recommended_action": "NONE" | "CONTINUE_MONITORING" | "FLAG_FOR_REVIEW" | "ESCALATE"}
Rules: evidence_event_ids must be event_ids that appear in the trajectory shown in brackets like [event_id]; never invent IDs.
drift_type must be null unless status is DRIFTING. A DRIFTING verdict needs at least one evidence event."""


# Per-event render limit. Long enough for typical cited research answers, because UNSUPPORTED_CONCLUSION is judged
# against the retrieved evidence the supervisor can actually see.
MAX_EVENT_CHARS = 1500


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def event_aliases(inp: SupervisorInput) -> dict[str, str]:
    """Opaque per-prompt event aliases (E1..En by trajectory position).

    Real event IDs never reach the model: datasets often encode the scenario (and so the label) in them.
    """
    return {e.event_id: f"E{i + 1}" for i, e in enumerate(inp.trajectory)}


def render_event(e: TrajectoryEvent, max_chars: int = MAX_EVENT_CHARS, alias: str | None = None) -> str:
    parts = [f"[{alias or e.event_id}] step={e.step} {e.event_type.value}"]
    if e.tool_name:
        parts.append(f"tool={e.tool_name}")
    if e.tool_arguments:
        parts.append("args=" + _clip(json.dumps(e.tool_arguments, sort_keys=True), max_chars // 2))
    if e.content is not None:
        body = e.content if isinstance(e.content, str) else json.dumps(e.content, sort_keys=True)
        parts.append(":: " + _clip(body, max_chars))
    if e.source_references:
        refs = []
        for r in e.source_references[:6]:
            if r.source_id:
                refs.append(f"{r.source_id} {r.url}" if r.url else f"{r.source_id} (never returned)")
            elif r.url:
                refs.append(r.url)
        if refs:
            parts.append("sources=" + ", ".join(refs))
    return " ".join(parts)


def render_input(inp: SupervisorInput, max_events: int = 40, max_chars: int = MAX_EVENT_CHARS) -> str:
    lines = [
        f"Swarm objective: {inp.swarm_objective}",
        f"Agent directive: {inp.agent_directive}",
        "Constraints:" + ("".join(f"\n  - {c}" for c in inp.constraints) or " (none)"),
        "Success criteria:" + ("".join(f"\n  - {c}" for c in inp.success_criteria) or " (none)"),
    ]
    if inp.current_plan:
        lines.append(f"Current plan: {_clip(inp.current_plan, max_chars)}")
    if inp.relevant_dependencies:
        lines.append("Downstream tasks depending on this agent: " + ", ".join(inp.relevant_dependencies))
    lines.append(f"Observation step: {inp.observation_step}")
    events = inp.trajectory
    omitted = max(len(events) - max_events, 0)
    lines.append(f"Trajectory ({len(events)} events" + (f", {omitted} earlier events omitted" if omitted else "") + "):")
    aliases = event_aliases(inp)
    for e in events[omitted:]:
        lines.append("  " + render_event(e, max_chars, aliases[e.event_id]))
    if inp.proposed_next_action:
        lines.append(f"Proposed next action: {_clip(inp.proposed_next_action, max_chars)}")
    lines.append("Classify this agent's trajectory now.")
    return "\n".join(lines)


def build_messages(policy_text: str, inp: SupervisorInput, max_events: int = 40, max_chars: int = MAX_EVENT_CHARS) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": policy_text.rstrip() + "\n\n" + OUTPUT_CONTRACT},
        {"role": "user", "content": render_input(inp, max_events, max_chars)},
    ]


_DECODER = json.JSONDecoder()


def extract_json(text: str) -> dict[str, Any]:
    """Return the last top-level JSON object in `text` that has a "status" key (else the last object).

    Tolerates prose, code fences, reasoning blocks containing braces, and draft objects before the final one.
    """
    objects: list[dict[str, Any]] = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i = text.find("{", end)
    if not objects:
        raise ValueError("no JSON object found in model output")
    with_status = [o for o in objects if "status" in o]
    return (with_status or objects)[-1]


def _upper(v: Any) -> Any:
    return v.strip().upper() if isinstance(v, str) else v


def parse_decision(text: str, inp: SupervisorInput, policy_version: str
                   ) -> tuple[SupervisorDecision | None, str | None, list[str]]:
    try:
        obj = extract_json(text)
    except ValueError as e:
        return None, f"invalid_json: {e}", []
    raw_ids = obj.get("evidence_event_ids") or []
    if not isinstance(raw_ids, list):
        raw_ids = [str(raw_ids)]
    real_by_alias = {a.upper(): real for real, a in event_aliases(inp).items()}
    mapped, unknown = [], []
    for cited in (str(i).strip() for i in raw_ids):
        real = real_by_alias.get(cited.upper())
        if real is None:
            if cited not in unknown:
                unknown.append(cited)
        else:
            mapped.append(real)
    valid, _ = split_evidence_ids(mapped, inp)
    invalid = unknown
    payload = {
        "agent_id": inp.agent_id, "task_id": inp.task_id,
        "status": _upper(obj.get("status")), "confidence": obj.get("confidence", 0.0),
        "drift_type": _upper(obj.get("drift_type")) or None, "evidence_event_ids": valid,
        "explanation": str(obj.get("explanation", "")),
        "recommended_action": _upper(obj.get("recommended_action")) or "NONE",
        "policy_version": policy_version, "observation_step": inp.observation_step,
    }
    try:
        return SupervisorDecision.model_validate(payload), None, invalid
    except ValidationError as e:
        fields = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
        return None, f"schema_error: {fields}", invalid


class Supervisor:
    """Stateless judge: (policy, input) -> EvaluationRecord. Shared across sessions; safe to call concurrently."""

    def __init__(self, client: ModelClient, policy: Policy, *, max_tokens: int = 1024, temperature: float = 0.0,
                 json_mode: bool = False, max_events: int = 40, max_chars: int = MAX_EVENT_CHARS, seed: int | None = None) -> None:
        self.client, self.policy = client, policy
        self.max_tokens, self.temperature, self.json_mode, self.seed = max_tokens, temperature, json_mode, seed
        self.max_events, self.max_chars = max_events, max_chars

    def _messages(self, inp: SupervisorInput) -> list[dict[str, Any]]:
        return build_messages(self.policy.text, inp, self.max_events, self.max_chars)

    def _record(self, inp: SupervisorInput, result: CompletionResult | None, error: str | None,
                example_id: str | None) -> EvaluationRecord:
        rec = EvaluationRecord(agent_id=inp.agent_id, task_id=inp.task_id, observation_step=inp.observation_step,
                               policy_version=self.policy.version, model=result.model if result else self.client.model,
                               mocked=result.mocked if result else False, example_id=example_id)
        if result is not None:
            rec.raw_output, rec.latency_ms = result.text, result.latency_ms
            rec.prompt_tokens, rec.completion_tokens, rec.cached_tokens = result.prompt_tokens, result.completion_tokens, result.cached_tokens
        if error is not None:
            rec.error = error
            return rec
        assert result is not None
        if result.truncated and not result.text.strip():
            rec.error = "truncated_output: finish_reason=length with empty content; raise max_tokens or disable thinking"
            return rec
        decision, perr, invalid = parse_decision(result.text, inp, self.policy.version)
        rec.decision, rec.error, rec.invalid_evidence_ids = decision, perr, invalid
        if perr:
            log.warning("malformed supervisor output for agent=%s step=%d: %s", inp.agent_id, inp.observation_step, perr)
        return rec

    def evaluate_sync(self, inp: SupervisorInput, example_id: str | None = None) -> EvaluationRecord:
        try:
            result = self.client.complete(self._messages(inp), max_tokens=self.max_tokens, temperature=self.temperature,
                                          json_mode=self.json_mode, seed=self.seed)
        except Exception as e:  # noqa: BLE001
            log.error("model error for agent=%s: %s", inp.agent_id, e)
            return self._record(inp, None, f"model_error: {type(e).__name__}: {e}", example_id)
        return self._record(inp, result, None, example_id)

    async def evaluate(self, inp: SupervisorInput, example_id: str | None = None) -> EvaluationRecord:
        try:
            result = await self.client.acomplete(self._messages(inp), max_tokens=self.max_tokens,
                                                 temperature=self.temperature, json_mode=self.json_mode, seed=self.seed)
        except Exception as e:  # noqa: BLE001
            log.error("model error for agent=%s: %s", inp.agent_id, e)
            return self._record(inp, None, f"model_error: {type(e).__name__}: {e}", example_id)
        return self._record(inp, result, None, example_id)
