"""Smoke test: is the key valid, which models exist, does a call work?"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.crusoe_client import PLANNER_MODEL, WORKER_MODEL, chat, client  # noqa: E402

ids = sorted(m.id for m in client.models.list().data)
print("Models available to this key:")
for i in ids:
    print("  ", i)

for m in (PLANNER_MODEL, WORKER_MODEL):
    if m not in ids:
        print(f"\n!! {m} not in /v1/models — fix the ID in .env (case-sensitive)")

for m in (PLANNER_MODEL, WORKER_MODEL):
    reply = chat([{"role": "user", "content": "Reply with exactly: pong"}], model=m)
    print(f"\nTest call to {m}: {reply.strip()[:80]}")
