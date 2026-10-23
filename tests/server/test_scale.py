"""A big store stays bounded (Review Focus 2): every list endpoint returns at most its documented limit, and the
event queries are index-backed."""
import json
import time

from pitstop.server.changes import change_series
from pitstop.server.graph import build_graph

WORKFLOWS = ("alpha", "beta", "gamma")
N_RUNS, EVENTS_PER_RUN = 5000, 20
T0 = time.time() - N_RUNS * 10 - 100  # the newest run is recent, inside every window


def _seed(store):
    conn = store.connect()
    runs, events = [], []
    for i in range(N_RUNS):
        started = T0 + i * 10
        runs.append((f"r{i:05d}", WORKFLOWS[i % 3], started, started + 5, "ok", "m1", "https://a/v1", "wrap"))
        for j in range(EVENTS_PER_RUN):
            if j % 2:
                events.append((f"r{i:05d}", started + j / 10, "agent", "tool_call", "search", "{}", None))
            else:
                payload = json.dumps({"meter": "agent", "prompt_tokens": 10, "completion_tokens": 5,
                                      "content_chars": 20, "refused": False})
                events.append((f"r{i:05d}", started + j / 10, "agent", "llm_call", "m1", payload, 0.001))
    with store.transaction() as c:
        c.executemany("INSERT INTO runs(id, workflow, started_at, ended_at, status, model, base_url, adapter)"
                      " VALUES (?,?,?,?,?,?,?,?)", runs)
        c.executemany("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                      events)
        c.executemany("INSERT INTO drifts(workflow, ts, level, kind, drift_type, evidence, action, status, last_seen,"
                      " occurrences) VALUES (?,?,?,?,?,?,?,?,?,1)",
                      [(WORKFLOWS[i % 3], T0 + N_RUNS * 10 - i, "P2", "off_task", f"type{i}", "[]", "record",
                        "open", T0) for i in range(300)])
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == N_RUNS * EVENTS_PER_RUN


def _plans(conn, statements):
    return [" | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + s)) for s in statements]


def _traced(store, fn):
    conn, statements = store.connect(), []
    conn.set_trace_callback(statements.append)
    try:
        fn()
    finally:
        conn.set_trace_callback(None)
    return [s for s in statements if "FROM events" in s]


def test_large_store_responses_are_capped(rt, client):
    _seed(rt.store)
    board = client.get("/api/board").json()
    assert {t["name"] for t in board["workflows"]} == set(WORKFLOWS)
    assert all(len(t["drifts"]) <= 50 for t in board["workflows"] if "drifts" in t)
    assert len(client.get("/api/feed", params={"limit": 10_000}).json()) == 100
    page = client.get("/api/runs", params={"limit": 10_000}).json()
    assert len(page["runs"]) == 200 and page["next_before"] is not None
    assert len(client.get("/api/runs", params={"workflow": "alpha", "limit": 10_000}).json()["runs"]) == 200
    for name in WORKFLOWS:
        detail = client.get("/api/workflow", params={"name": name}).json()
        assert len(detail["drifts"]) == 50
        graph = client.get("/api/graph", params={"workflow": name, "window": "30d"}).json()
        assert graph["runs_in_scope"] == 200
        changes = client.get("/api/changes", params={"workflow": name, "limit": 10_000}).json()
        assert len(changes["points"]) == 200
        assert [p["started_at"] for p in changes["points"]] == sorted(p["started_at"] for p in changes["points"])

    conn = rt.store.connect()
    seen = _traced(rt.store, lambda: build_graph(rt.store, "alpha", window="30d", now=T0 + N_RUNS * 10 + 100))
    seen += _traced(rt.store, lambda: change_series(rt.store, "alpha"))
    assert seen
    for sql, plan in zip(seen, _plans(conn, seen)):
        assert "idx_events_run" in plan, (sql[:80], plan)
        assert "SCAN events" not in plan.replace("SCAN events USING", ""), plan
    runs_plan = _plans(conn, ["SELECT * FROM runs WHERE workflow = 'alpha' AND typeof(started_at) IN ('real')"
                              " ORDER BY started_at DESC LIMIT 200"])[0]
    assert "idx_runs_wf_started" in runs_plan, runs_plan
