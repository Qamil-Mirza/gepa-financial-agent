"""Confirm the worker model exists, supports tools, and answers one real tool-calling request."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from synth import config

client = OpenAI(base_url=config.CRUSOE_BASE_URL, api_key=config.crusoe_key())
ids = {m.id: m for m in client.models.list().data}
if config.WORKER_MODEL not in ids:
    sys.exit(f"{config.WORKER_MODEL} not found. Available: {sorted(ids)}")
print(f"model found: {config.WORKER_MODEL}")

tools = [{
    "type": "function",
    "function": {
        "name": "finance_research",
        "description": "Research a financial question and return cited evidence.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}]
resp = client.chat.completions.create(
    model=config.WORKER_MODEL,
    messages=[
        {"role": "system", "content": "You are a financial research agent. Use tools to gather evidence."},
        {"role": "user", "content": "Directive: investigate NVIDIA data-center revenue growth. Take your first action."},
    ],
    tools=tools,
    max_tokens=2048,
    temperature=1.0,
    top_p=0.95,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("content:", (msg.content or "")[:300])
for tc in msg.tool_calls or []:
    print("tool_call:", tc.function.name, json.loads(tc.function.arguments))
print("usage:", resp.usage.prompt_tokens, "in /", resp.usage.completion_tokens, "out")
