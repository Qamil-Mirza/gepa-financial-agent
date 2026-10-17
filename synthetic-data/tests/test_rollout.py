"""Rollout loop with a scripted fake client and a temp research cache (no network)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from synth.crusoe import Completion
from synth.research_bank import ResearchBank
from synth.rollout import run_rollout
from synth.schemas import SteerSpec
from synth.tasks import TASKS
from synth.textmatch import normalize_query


def _tc(i, name, **args):
    return SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    """Test double. Never used to produce dataset traces."""
    model = "scripted-test-client"

    def __init__(self, script):
        self.script = list(script)
        self.seen_tools = []
        self.seen_messages = []

    async def complete(self, messages, tools=None, **kw):
        self.seen_tools.append([t["function"]["name"] for t in tools or []])
        self.seen_messages.append(list(messages))
        calls = self.script.pop(0)
        return Completion(message=SimpleNamespace(content="", tool_calls=calls), finish_reason="tool_calls",
                          prompt_tokens=10, completion_tokens=5, latency_ms=1)


def _bank(tmp_path):
    q = "NVIDIA data center segment revenue for the last four reported quarters"
    entry = {"query": q, "normalized": normalize_query(q), "response": {"output": {
        "content": "Data Center revenue was $89.0 billion.", "sources": [{"url": "https://x.test/a", "title": "A"}]}}}
    (tmp_path / "a.json").write_text(json.dumps(entry))
    return ResearchBank(cache_dir=tmp_path, live_budget=0)


def test_rollout_records_events_and_hides_steer(tmp_path):
    task = TASKS["nvda_datacenter"]
    steer = SteerSpec(steer_id="drift:constraint_violation", kind="drift", drift_type="CONSTRAINT_VIOLATION", step_x=2)
    client = ScriptedClient([
        [_tc(1, "finance_research", query="NVIDIA data center segment revenue last four reported quarters")],
        [_tc(2, "record_finding", claim="Data Center revenue was $89.0 billion", source_ids=[])],
        [_tc(3, "finish", summary="done")],
    ])
    trace = asyncio.run(run_rollout(client, _bank(tmp_path), task, steer, max_steps=5, today="2026-10-17"))

    types = [e.event_type for e in trace.events]
    assert types == ["tool_call", "tool_result", "research_finding", "task_completion"]
    assert trace.events[1].source_references[0].source_id == "S1.1"
    assert client.seen_tools[1] == ["record_finding"]          # steer turn offers only the target tool
    assert any("Private instruction" in m["content"] for m in client.seen_messages[1] if m["role"] == "user")
    assert not any("Private" in str(e.content) for e in trace.events)
    assert trace.generation.usage.calls == 3 and trace.generation.usage.youcom_cache_hits == 1


def test_obsolete_update_is_visible_but_not_sent(tmp_path):
    task = TASKS["nvda_datacenter"]
    steer = SteerSpec(steer_id="drift:obsolete_directive", kind="drift", drift_type="OBSOLETE_DIRECTIVE", step_x=2)
    client = ScriptedClient([
        [_tc(1, "finance_research", query="NVIDIA data center segment revenue last four reported quarters")],
        [_tc(2, "finish", summary="done")],
    ])
    trace = asyncio.run(run_rollout(client, _bank(tmp_path), task, steer, max_steps=5, today="2026-10-17"))
    assert any(e.event_type == "status_update" and e.step == 2 for e in trace.events)
    assert not any(task.updated_directive in str(m.get("content")) for m in client.seen_messages[1])
