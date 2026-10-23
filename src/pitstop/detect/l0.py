"""L0 checks: cheap, deterministic drift detection over tool calls and LLM calls."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..sdk.context import Sensitivity, WorkflowSpec
from .text import jaccard

THRESHOLDS: dict[Sensitivity, dict] = {
    "low": dict(loop_window=6, loop_repeats=4, near_dup_jaccard=0.95, parse_fail_rate=0.20,
                refusals_per_run=3, min_calls=5),
    "normal": dict(loop_window=6, loop_repeats=3, near_dup_jaccard=0.90, parse_fail_rate=0.10,
                   refusals_per_run=2, min_calls=5),
    "high": dict(loop_window=8, loop_repeats=2, near_dup_jaccard=0.85, parse_fail_rate=0.05,
                 refusals_per_run=1, min_calls=5),
}

REFUSAL_RE = re.compile(r"^\s*(I'm sorry|I am sorry|I can(?:not|'t) (?:help|assist|comply)|As an AI\b)", re.I)
_REFUSAL_MAX_CHARS = 400
_BREAKAGE_WINDOW = 20


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: str  # canonical JSON, or the raw string if it does not parse
    side_effects: bool


@dataclass(frozen=True)
class LLMCallRecord:
    tool_call_parse_failures: int
    tool_calls: int
    refused: bool


@dataclass(frozen=True)
class Finding:
    drift_type: str
    kind: Literal["off_task", "degraded"]
    side_effects: bool
    evidence: dict = field(default_factory=dict)


def canonical_args(raw: object) -> str:
    """Canonical JSON for tool arguments (a JSON string or an already parsed value); raw text if unparseable."""
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return raw if isinstance(raw, str) else str(raw)


def is_refusal(content: str | None, tool_calls: int) -> bool:
    return bool(content) and tool_calls == 0 and len(content) < _REFUSAL_MAX_CHARS \
        and REFUSAL_RE.match(content) is not None


def check_tool_call(history: Sequence[ToolCallRecord], call: ToolCallRecord, spec: WorkflowSpec) -> list[Finding]:
    """`history` holds the agent's earlier calls (a list or a bounded deque); `call` is the one about to run."""
    out: list[Finding] = []
    if spec.tools is not None and call.name not in spec.tools:
        out.append(Finding("out_of_scope_action", "off_task", call.side_effects,
                           {"tool": call.name, "allowed": list(spec.tools)}))
    t = THRESHOLDS[spec.sensitivity]
    window = (list(history)[-t["loop_window"]:] if t["loop_window"] > 0 else []) + [call]
    same = sum(1 for r in window if r.name == call.name
               and (r.args == call.args or jaccard(r.args, call.args) >= t["near_dup_jaccard"]))
    if same >= t["loop_repeats"]:
        out.append(Finding("unproductive_loop", "off_task", call.side_effects,
                           {"tool": call.name, "args": call.args, "repeats": same,
                            "window": t["loop_window"]}))
    return out


def check_llm_call(history: Sequence[LLMCallRecord], call: LLMCallRecord, sensitivity: Sensitivity,
                   refusals: int) -> list[Finding]:
    """`history` holds the run's recent earlier calls (capped); `call` is the one just made. `refusals` is the
    run's total refusal count including `call`: the capped history cannot count the whole run."""
    t = THRESHOLDS[sensitivity]
    out: list[Finding] = []
    recent = [r for r in [*history, call] if r.tool_calls > 0][-_BREAKAGE_WINDOW:]
    if len(recent) >= t["min_calls"]:
        failures = sum(r.tool_call_parse_failures for r in recent)
        total = sum(r.tool_calls for r in recent)
        rate = failures / total if total else 0.0
        if rate >= t["parse_fail_rate"]:
            out.append(Finding("tool_call_breakage", "degraded", False,
                               {"parse_failures": failures, "tool_calls": total, "rate": round(rate, 3)}))
    if refusals >= t["refusals_per_run"]:
        out.append(Finding("refusal_spike", "degraded", False, {"refusals": refusals}))
    return out
