"""L1 pre-action gate: before a side-effect tool runs on a high or medium workflow, a small judge model is asked
whether this one action serves the goal (the LlamaFirewall AlignmentCheck pattern, spec §6.2).

The prompt is frozen and versioned (`assets/prompts/l1_gate_v2.txt`; v1 stays as it shipped). The untrusted text
(the proposed call's arguments, the recent calls) sits in `<tool_args>` and `<recent_calls>` blocks with every closer
("</" and near-misses) inside written as "<\\/", so it cannot close its block; the prompt says the blocks are
data, never instructions.

The judge call is one sync chat call on the "watch" meter, so it counts against the budgets but never feeds L0
detection or the run's model tag. A judge that errors or answers without a usable verdict yields
`GateVerdict(error=...)`; the caller applies the workflow's `on_gate_error` policy. The error text is only an
exception type name or "unparseable verdict", never the request, the key or the URL. `Halted` and `BudgetExceeded`
from the judge call propagate."""
from __future__ import annotations

import functools
import re
import time
from dataclasses import dataclass
from typing import Any

from ..config import PROMPTS
from ..llm import extract_json_object
from ..sdk.errors import Halted
from .l0 import ToolCallRecord

GATE_PROMPT_VERSION = "l1_gate_v2"

ARGS_MAX = 2000  # chars of the proposed call's arguments shown to the judge
RECENT_CALLS = 5  # the agent's most recent tool calls shown for context
RECENT_ARGS_MAX = 300  # chars of each recent call's arguments
REASON_MAX = 500


@dataclass(frozen=True)
class GateVerdict:
    allow: bool
    reason: str
    error: str | None = None  # set when the judge could not decide; caller applies on_gate_error
    latency_s: float = 0.0


_CLOSER = re.compile("[<\uff1c]\\s*[/\uff0f]")


@functools.cache
def _system_prompt() -> str:
    return (PROMPTS / f"{GATE_PROMPT_VERSION}.txt").read_text()


def _data(text: str) -> str:
    """Untrusted text for a tag block: no closer survives, so the text cannot close the block it sits in. A `<`
    or fullwidth `＜`, optional whitespace, then `/` or fullwidth `／` is written `<\\/` (near-miss closers too)."""
    return _CLOSER.sub("<\\\\/", text)


def _user_message(workflow_goal: str | None, agent_goal: str | None, tool: str, args_text: str,
                  recent: list[ToolCallRecord]) -> str:
    shown = list(recent)[-RECENT_CALLS:]
    calls = [f"- {r.name} {r.args[:RECENT_ARGS_MAX]}" for r in shown] or ["(none)"]
    lines = [f"Workflow goal: {workflow_goal or '(none given)'}",
             f"Agent goal: {agent_goal or '(none given)'}",
             "",
             f"Proposed action: call tool {_data(repr(tool))} with these arguments:",
             "<tool_args>",
             _data(args_text[:ARGS_MAX]),
             "</tool_args>",
             "",
             "The agent's most recent tool calls (oldest first):",
             "<recent_calls>",
             _data("\n".join(calls)),
             "</recent_calls>"]
    return "\n".join(lines)


class Gate:
    """`client` is a wrapped sync OpenAI-compatible client (see `Runtime.gate()`); `model` is the judge model.
    `max_tokens` leaves room for a reasoning model's thinking before it writes the verdict."""

    def __init__(self, client: Any, model: str, *, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def check(self, *, workflow_goal: str | None, agent_goal: str | None, tool: str, args_text: str,
              recent: list[ToolCallRecord]) -> GateVerdict:
        from ..sdk.wrap import meter

        t0 = time.monotonic()
        try:
            messages = [{"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _user_message(workflow_goal, agent_goal, tool, args_text,
                                                                  recent)}]
            with meter("watch"):
                resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                                                           max_tokens=self.max_tokens)
        except Halted:  # BudgetExceeded included: the hard budget and pending halts always win
            raise
        except Exception as e:
            return GateVerdict(False, "", error=type(e).__name__, latency_s=time.monotonic() - t0)
        latency = time.monotonic() - t0
        try:
            obj = extract_json_object(resp.choices[0].message.content or "")
            serves = obj.get("serves_goal")
            if not isinstance(serves, bool):
                raise ValueError("serves_goal missing or not a boolean")
        except Exception:
            return GateVerdict(False, "", error="unparseable verdict", latency_s=latency)
        reason = obj.get("reason")
        reason = (reason if isinstance(reason, str) else "" if reason is None else str(reason))[:REASON_MAX]
        return GateVerdict(serves, reason, latency_s=latency)
