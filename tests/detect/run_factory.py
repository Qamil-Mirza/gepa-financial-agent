"""Write a finished run and its events straight into a store, in the shapes the SDK writes them (Tasks 11-13).

Each `llm_calls` entry `(prompt_tokens, completion_tokens, content_chars)` becomes one agent-meter `llm_call`
event costing $1 per 1M tokens, i.e. `(prompt_tokens + completion_tokens) / 1_000_000` USD. Each `tool_results`
entry becomes a `tool_call` event followed by a `tool_result` event: `{"error": ...}` when True, else
`{"result": ...}`. Event timestamps start at `started_at` and step by 1 ms in write order."""
from __future__ import annotations

import json
import uuid

from pitstop.store import Store

USD_PER_TOKEN = 1e-6


def make_run(store: Store, workflow: str, *, started_at: float, ended_at: float | None, model: str = "m1",
             base_url: str = "https://a/v1", prompt_versions: dict[str, int] | None = None,
             llm_calls: tuple = ((100, 20, 400),), tool_results: tuple = (), status: str = "ok",
             criticality: str = "medium", sensitivity: str = "normal", adapter: str = "wrap") -> str:
    """Insert one run of `workflow` with its llm_call / tool_call / tool_result events; returns the run id."""
    run_id = uuid.uuid4().hex
    events: list[tuple] = []
    ts = started_at
    for prompt, completion, chars in llm_calls:
        payload = {"tool_calls": [], "finish_reason": "stop", "content_chars": chars, "meter": "agent",
                   "prompt_tokens": prompt, "completion_tokens": completion, "refused": False}
        events.append((run_id, ts, "agent", "llm_call", model, json.dumps(payload),
                       (prompt + completion) * USD_PER_TOKEN))
        ts += 0.001
    for i, error in enumerate(tool_results):
        events.append((run_id, ts, "agent", "tool_call", f"tool{i}", json.dumps({"args": "{}"}), None))
        ts += 0.001
        result = {"error": "RuntimeError('boom')"} if error else {"result": "'ok'"}
        events.append((run_id, ts, "agent", "tool_result", f"tool{i}", json.dumps(result), None))
        ts += 0.001
    with store.transaction() as c:
        c.execute("INSERT INTO runs(id, workflow, criticality, sensitivity, started_at, ended_at, status, model,"
                  " base_url, prompt_versions, adapter, pitstop_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (run_id, workflow, criticality, sensitivity, started_at, ended_at, status, model, base_url,
                   json.dumps(prompt_versions) if prompt_versions is not None else None, adapter, "test"))
        c.executemany("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                      events)
    return run_id
