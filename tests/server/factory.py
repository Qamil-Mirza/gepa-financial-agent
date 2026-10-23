"""Seed dashboard fixtures straight into a store with SQL: runs, events, drifts and ledger rows, in the shapes the
SDK writes them. Each helper takes keyword overrides for any column and returns the new row's id. A `payload` or
`evidence` that is a str is stored as is (to seed malformed rows); anything else is JSON-encoded."""
from __future__ import annotations

import itertools
import json

from pitstop.store import Store

_ids = itertools.count(1)


def _json(value) -> str | None:
    return value if value is None or isinstance(value, str) else json.dumps(value)


def _insert(store: Store, table: str, row: dict) -> int:
    cols = ", ".join(row)
    with store.transaction() as c:
        return c.execute(f"INSERT INTO {table}({cols}) VALUES ({', '.join('?' * len(row))})",
                         tuple(row.values())).lastrowid


def add_run(store: Store, workflow: str | None = "wf", *, started_at: float | None, id: str | None = None,
            **cols) -> str:
    """A run (status "ok", ended 5 s after it started, model m1 at https://a/v1, adapter wrap by default)."""
    run_id = id or f"run-{next(_ids):05d}"
    row = {"id": run_id, "workflow": workflow, "goal": "answer questions", "criticality": "medium",
           "sensitivity": "normal", "fix_mode": "suggest", "started_at": started_at,
           "ended_at": started_at + 5 if isinstance(started_at, (int, float)) else None, "status": "ok",
           "model": "m1", "base_url": "https://a/v1", "adapter": "wrap", "pitstop_version": "test"}
    row.update(cols)
    _insert(store, "runs", row)
    return run_id


def add_event(store: Store, run_id: str | None, kind: str, *, ts: float | None, name: str | None = None,
              payload=None, agent: str | None = "agent", cost_usd: float | None = None, **cols) -> int:
    row = {"run_id": run_id, "ts": ts, "agent": agent, "kind": kind, "name": name,
           "payload": _json(payload if payload is not None else {}), "cost_usd": cost_usd}
    row.update(cols)
    return _insert(store, "events", row)


def add_llm_call(store: Store, run_id: str, *, ts: float, model: str = "m1", prompt: int = 100, completion: int = 20,
                 cost_usd: float | None = 0.01, meter: str = "agent", **extra) -> int:
    payload = {"tool_calls": 0, "finish_reason": "stop", "content_chars": 40, "meter": meter, "prompt_tokens": prompt,
               "completion_tokens": completion, "refused": False, **extra}
    return add_event(store, run_id, "llm_call", ts=ts, name=model, payload=payload, cost_usd=cost_usd)


def add_drift(store: Store, workflow: str | None = "wf", *, ts: float, level: str = "P2",
              drift_type: str = "unproductive_loop", kind: str = "off_task", action: str = "record",
              status: str = "open", run_id: str | None = None, evidence=None, **cols) -> int:
    if evidence is None:
        evidence = [{"run_id": run_id, "ts": ts, "detail": {"tool": "search"}}]
    row = {"run_id": run_id, "workflow": workflow, "ts": ts, "level": level, "kind": kind, "drift_type": drift_type,
           "evidence": _json(evidence), "action": action, "status": status, "last_seen": ts, "occurrences": 1}
    row.update(cols)
    return _insert(store, "drifts", row)


def add_ledger(store: Store, workflow: str | None, usd: float, *, ts: float, meter: str = "agent",
               state: str = "settled", run_id: str | None = None, model: str = "m1",
               base_url: str = "https://a/v1") -> int:
    return _insert(store, "ledger", {"ts": ts, "run_id": run_id, "workflow": workflow, "meter": meter, "model": model,
                                     "base_url": base_url, "prompt_tokens": 100, "completion_tokens": 20, "usd": usd,
                                     "state": state})
