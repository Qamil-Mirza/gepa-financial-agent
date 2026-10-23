"""The run graph (`pitstop.server.graph.build_graph`, decision 8) and its endpoint: structure from SDK `agent` events
and OTel span links, node statistics, verdicts, the slowest path and the run scope."""
import time

import pytest

from tests.server.factory import add_event, add_run

T0 = 1_790_000_000.0
HOUR = 3600.0
NOW = T0 + HOUR


def _agent(store, run, name, parent=None, *, ts, **cols):
    return add_event(store, run, "agent", ts=ts, name=name, agent=name,
                     payload={"parent": parent, "source": "otel" if "span_id" in cols else "sdk"}, **cols)


def _llm(store, run, agent, *, ts, model="m1", latency=None, cost=0.01, meter="agent", **cols):
    payload = {"meter": meter, "prompt_tokens": 10, "completion_tokens": 5}
    if latency is not None:
        payload["latency_s"] = latency
    return add_event(store, run, "llm_call", ts=ts, name=model, agent=agent, payload=payload, cost_usd=cost, **cols)


def _tool(store, run, agent, name, *, ts, latency=None, error=None, **cols):
    add_event(store, run, "tool_call", ts=ts, name=name, agent=agent, payload={"args": "{}"}, **cols)
    payload = {"error": error} if error is not None else {"result": "ok"}
    if latency is not None:
        payload["latency_s"] = latency
    return add_event(store, run, "tool_result", ts=ts + 0.5, name=name, agent=agent, payload=payload, **cols)


def _edges(g):
    return {(e.source, e.target) for e in g.edges}


def _nodes(g):
    return {n.id: n for n in g.nodes}


P = "agent:planner"
PR = "agent:planner › researcher"


def test_sdk_run_graph_shape(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    run = add_run(s, "wf", started_at=T0)
    _agent(s, run, "planner", ts=T0 + 1)
    _llm(s, run, "planner", ts=T0 + 2)
    _tool(s, run, "planner", "search", ts=T0 + 3)
    _agent(s, run, "researcher", "planner", ts=T0 + 4)
    _llm(s, run, "researcher", ts=T0 + 5, model="m2")
    _tool(s, run, "researcher", "fetch", ts=T0 + 6)
    _tool(s, run, None, "clock", ts=T0 + 7)  # no agent: hangs under the workflow root

    g = build_graph(s, "wf", run_id=run)
    assert _edges(g) == {
        ("workflow", P), (P, "llm:planner|m1"), (P, "tool:planner|search"), (P, PR),
        (PR, "llm:planner › researcher|m2"), (PR, "tool:planner › researcher|fetch"),
        ("workflow", "tool:|clock")}
    nodes = _nodes(g)
    assert set(nodes) == {"workflow"} | {t for _, t in _edges(g)}
    assert g.nodes[0].id == "workflow"
    root = nodes["workflow"]
    assert (root.kind, root.label, root.agent_path) == ("workflow", "wf", [])
    assert (nodes[PR].kind, nodes[PR].label, nodes[PR].agent_path) == ("agent", "researcher", ["planner", "researcher"])
    leaf = nodes["llm:planner › researcher|m2"]
    assert (leaf.kind, leaf.label, leaf.agent_path) == ("llm", "m2", ["planner", "researcher"])
    tool = nodes["tool:|clock"]
    assert (tool.kind, tool.label, tool.agent_path) == ("tool", "clock", [])
    assert (g.runs_in_scope, g.run_id, g.window) == (1, run, None)
    assert all(n.run_ids == [run] for n in g.nodes)


def test_same_agent_name_under_two_parents_stays_apart(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    a = add_run(s, "wf", started_at=T0)
    _agent(s, a, "planner", ts=T0 + 1)
    _agent(s, a, "researcher", "planner", ts=T0 + 2)
    _llm(s, a, "researcher", ts=T0 + 3)
    b = add_run(s, "wf", started_at=T0 + 10)
    _agent(s, b, "writer", ts=T0 + 11)
    _agent(s, b, "researcher", "writer", ts=T0 + 12)
    _tool(s, b, "researcher", "fetch", ts=T0 + 13)

    g = build_graph(s, "wf", now=NOW)
    WR = "agent:writer › researcher"
    assert _edges(g) == {("workflow", P), ("workflow", "agent:writer"), (P, PR), ("agent:writer", WR),
                         (PR, "llm:planner › researcher|m1"), (WR, "tool:writer › researcher|fetch")}
    nodes = _nodes(g)
    assert nodes[PR].run_ids == [a] and nodes[WR].run_ids == [b]

    # in one run, a leaf goes to the path its agent was most recently entered under
    c = add_run(s, "wf2", started_at=T0)
    _agent(s, c, "planner", ts=T0 + 1)
    _agent(s, c, "writer", ts=T0 + 2)
    _agent(s, c, "researcher", "planner", ts=T0 + 3)
    _llm(s, c, "researcher", ts=T0 + 4)
    _agent(s, c, "researcher", "writer", ts=T0 + 5)
    _llm(s, c, "researcher", ts=T0 + 6, model="m2")
    g = build_graph(s, "wf2", run_id=c)
    assert {"llm:planner › researcher|m1", "llm:writer › researcher|m2"} <= set(_nodes(g))
    assert "llm:writer › researcher|m1" not in _nodes(g)


def test_otel_run_graph_uses_span_links(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    run = add_run(s, "wf", started_at=T0, adapter="otel")
    _agent(s, run, "planner", ts=T0 + 1, span_id="a1", parent_span_id="r0")  # r0: the root span, not recorded
    # an agent called as a tool: researcher's parent span is a tool span under planner
    _tool(s, run, "planner", "delegate", ts=T0 + 2, span_id="t1", parent_span_id="a1")
    _agent(s, run, "researcher", ts=T0 + 3, span_id="b1", parent_span_id="t1")
    _agent(s, run, "researcher", ts=T0 + 4, span_id="b2", parent_span_id="t1")  # a second span: the same node
    _llm(s, run, "researcher", ts=T0 + 5, span_id="c1", parent_span_id="b1")
    _llm(s, run, "researcher", ts=T0 + 6, span_id="c2", parent_span_id="b2", model="m2")
    _llm(s, run, None, ts=T0 + 7, span_id="c3", parent_span_id="a1", model="m3")  # the span link wins over the column
    # the parent span was never recorded (a framework's "other" span): the agent column places it
    _llm(s, run, "researcher", ts=T0 + 8, span_id="c4", parent_span_id="zz", model="m4")
    _llm(s, run, None, ts=T0 + 9, span_id="c5", parent_span_id="r0", model="m5")  # no agent: under the root
    # a nested agent whose parent span was never recorded: the processor's `parent` places it
    add_event(s, run, "agent", ts=T0 + 10, name="critic", agent="critic", span_id="e1", parent_span_id="chain",
              payload={"parent": "researcher", "source": "otel"})
    _llm(s, run, "critic", ts=T0 + 11, span_id="c6", parent_span_id="e1")

    g = build_graph(s, "wf", run_id=run)
    assert _edges(g) == {
        ("workflow", P), (P, "tool:planner|delegate"), (P, PR), (P, "llm:planner|m3"),
        (PR, "llm:planner › researcher|m1"), (PR, "llm:planner › researcher|m2"),
        (PR, "llm:planner › researcher|m4"), ("workflow", "llm:|m5"),
        (PR, PR + " › critic"), (PR + " › critic", "llm:planner › researcher › critic|m1")}


def test_node_stats(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    a = add_run(s, "wf", started_at=T0)
    b = add_run(s, "wf", started_at=T0 + 10)
    for run, ts in ((a, T0), (b, T0 + 10)):
        _agent(s, run, "planner", ts=ts + 1)
    _llm(s, a, "planner", ts=T0 + 2, latency=1.0)
    _llm(s, a, "planner", ts=T0 + 3, latency=3.0)
    _llm(s, b, "planner", ts=T0 + 12, latency=2.0, cost=None)  # unpriced: no cost, still a call
    _llm(s, b, "planner", ts=T0 + 13)  # latency unknown: ignored, never 0
    _llm(s, b, "planner", ts=T0 + 14, latency="slow")  # malformed: unknown
    _tool(s, a, "planner", "search", ts=T0 + 4)  # no latency at all
    _tool(s, b, "planner", "search", ts=T0 + 15)
    add_event(s, b, "llm_call", ts=T0 + 16, name="m1", agent="planner", payload="not json", cost_usd=1.0)

    g = build_graph(s, "wf", now=NOW)
    nodes = _nodes(g)
    llm = nodes["llm:planner|m1"]
    assert (llm.calls, llm.runs) == (5, 2)
    assert llm.cost_usd == pytest.approx(0.04)
    assert llm.latency_p50_s == 2.0
    assert llm.time_per_run_s == pytest.approx(2.0 * 5 / 2)  # the known mean stands in for the unknown calls
    assert llm.run_ids == [b, a]
    tool = nodes["tool:planner|search"]
    assert (tool.calls, tool.runs, tool.cost_usd) == (2, 2, 0.0)
    assert (tool.latency_p50_s, tool.time_per_run_s) == (None, None)
    planner = nodes[P]
    assert (planner.calls, planner.runs, planner.latency_p50_s) == (7, 2, None)
    assert planner.cost_usd == pytest.approx(0.04)
    assert planner.time_per_run_s == pytest.approx(5.0)
    assert (planner.time_partial, llm.time_partial, tool.time_partial) == (True, False, False)
    root = nodes["workflow"]
    assert (root.calls, root.runs, root.run_ids, g.runs_in_scope) == (7, 2, [b, a], 2)
    assert root.time_per_run_s == pytest.approx(5.0)


def test_verdict_counts_on_tool_node(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    run = add_run(s, "wf", started_at=T0)
    _agent(s, run, "planner", ts=T0)
    _llm(s, run, "planner", ts=T0 + 1)
    _tool(s, run, "planner", "send", ts=T0 + 2, error="TimeoutError()")
    _tool(s, run, "planner", "send", ts=T0 + 3)
    for kind, payload in (("block", {"reason": "loop"}), ("steer", {"reason": "off task"}),
                          ("gate", {"allow": False, "reason": "no"}), ("gate", {"allow": True}),
                          ("gate_error", {"error": "timeout", "policy": "allow"}), ("halt", {"reason": "P0"})):
        add_event(s, run, kind, ts=T0 + 4, name="send", agent="planner", payload=payload)
    add_event(s, run, "fallback", ts=T0 + 5, name="m1", agent="planner", payload={"from": "m1", "to": "m0"})
    add_event(s, run, "halt", ts=T0 + 6, name="budget", agent="planner", payload={"reason": "budget"})
    add_event(s, run, "block", ts=T0 + 7, name="delete", agent="planner", payload={"reason": "gate"})  # never ran

    nodes = _nodes(build_graph(s, "wf", run_id=run))
    assert nodes["tool:planner|send"].verdicts.model_dump() == {
        "block": 1, "steer": 1, "halt": 1, "fallback": 0, "gate_reject": 1, "gate_error": 1, "tool_error": 1}
    assert nodes["llm:planner|m1"].verdicts.fallback == 1
    assert nodes[P].verdicts.halt == 1  # names neither a tool nor a model: the agent's own
    blocked = nodes["tool:planner|delete"]
    assert (blocked.calls, blocked.verdicts.block, blocked.runs) == (0, 1, 1)


def test_slowest_path_follows_largest_child(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    run = add_run(s, "wf", started_at=T0)
    _agent(s, run, "planner", ts=T0)
    _llm(s, run, "planner", ts=T0 + 1, latency=1.0)
    _tool(s, run, "planner", "search", ts=T0 + 2, latency=5.0)
    _agent(s, run, "writer", ts=T0 + 3)
    _llm(s, run, "writer", ts=T0 + 4, latency=4.0)  # the largest leaf, under the smaller agent
    _tool(s, run, "writer", "draft", ts=T0 + 5)  # unknown: left out of the writer's time
    _agent(s, run, "idle", ts=T0 + 6)

    g = build_graph(s, "wf", run_id=run)
    nodes = _nodes(g)
    assert g.slowest_path == ["workflow", P, "tool:planner|search"]
    assert {n.id for n in g.nodes if n.on_slowest_path} == set(g.slowest_path)
    assert nodes[P].time_per_run_s == pytest.approx(6.0)
    assert nodes[P].time_partial is False  # every child known
    assert nodes["agent:writer"].time_per_run_s == pytest.approx(4.0)
    assert nodes["agent:writer"].time_partial is True  # a lower bound: its draft tool's time is unknown
    assert (nodes["agent:idle"].time_per_run_s, nodes["agent:idle"].time_partial) == (None, False)
    assert nodes["workflow"].time_per_run_s == pytest.approx(10.0)
    assert nodes["workflow"].time_partial is True  # the writer is partial and the idle agent unknown
    assert not any(n.time_partial for n in g.nodes if n.kind in ("llm", "tool"))

    empty = add_run(s, "quiet", started_at=T0)
    g = build_graph(s, "quiet", run_id=empty)
    assert ([n.id for n in g.nodes], g.edges, g.slowest_path) == (["workflow"], [], [])


def test_window_and_run_scope(rt):
    from pitstop.server.graph import build_graph
    from pitstop.server.views import NotFound

    s = rt.store
    old = add_run(s, "wf", started_at=NOW - 2 * 24 * HOUR)
    _llm(s, old, "planner", ts=NOW - 2 * 24 * HOUR, model="old-model")
    recent = add_run(s, "wf", started_at=NOW - HOUR)
    _llm(s, recent, "planner", ts=NOW - HOUR, model="new-model")
    add_run(s, "wf", started_at=None)  # not numeric: never in a window

    g = build_graph(s, "wf", now=NOW)
    assert (g.runs_in_scope, g.window, g.run_id) == (1, "24h", None)
    assert "llm:planner|new-model" in _nodes(g) and "llm:planner|old-model" not in _nodes(g)
    g = build_graph(s, "wf", window="7d", now=NOW)
    assert g.runs_in_scope == 2 and "llm:planner|old-model" in _nodes(g)
    g = build_graph(s, "wf", window="1h", now=NOW + 1)
    assert g.runs_in_scope == 0 and [n.id for n in g.nodes] == ["workflow"]
    g = build_graph(s, "wf", run_id=old, now=NOW)
    assert (g.runs_in_scope, g.run_id, g.window) == (1, old, None)
    assert "llm:planner|new-model" not in _nodes(g)

    with pytest.raises(ValueError):
        build_graph(s, "wf", window="2d", now=NOW)
    with pytest.raises(NotFound):
        build_graph(s, "nope", now=NOW)
    other = add_run(s, "other", started_at=NOW - HOUR)
    with pytest.raises(NotFound):
        build_graph(s, "wf", run_id=other, now=NOW)
    with pytest.raises(NotFound):
        build_graph(s, "wf", run_id="no-such-run", now=NOW)

    # the cap: only the newest 200 runs in the window
    runs = []
    for i in range(205):
        r = add_run(s, "busy", started_at=NOW - 1000 + i)
        _llm(s, r, None, ts=NOW - 1000 + i)
        runs.append(r)
    g = build_graph(s, "busy", now=NOW)
    nodes = _nodes(g)
    assert g.runs_in_scope == 200 and nodes["llm:|m1"].calls == 200
    assert nodes["workflow"].run_ids == runs[::-1][:20]
    assert build_graph(s, "busy", now=NOW, max_runs=5).runs_in_scope == 5


def test_judge_meter_calls_excluded(rt):
    from pitstop.server.graph import build_graph

    s = rt.store
    run = add_run(s, "wf", started_at=T0)
    _agent(s, run, "planner", ts=T0)
    _llm(s, run, "planner", ts=T0 + 1, cost=0.01)
    for meter in ("watch", "supervisor", "correction"):
        _llm(s, run, "planner", ts=T0 + 2, cost=0.5, meter=meter)
        _llm(s, run, "planner", ts=T0 + 2, cost=0.5, meter=meter, model="judge-m")

    nodes = _nodes(build_graph(s, "wf", run_id=run))
    assert set(nodes) == {"workflow", P, "llm:planner|m1"}
    assert nodes["llm:planner|m1"].calls == 1
    assert nodes["workflow"].cost_usd == pytest.approx(0.01)


def test_graph_endpoint_errors(rt, client):
    s = rt.store
    now = time.time()
    run = add_run(s, "a/b", started_at=now - 60)
    _agent(s, run, "planner", ts=now - 59)
    _llm(s, run, "planner", ts=now - 58, latency=0.5)
    other = add_run(s, "other", started_at=now - 60)

    r = client.get("/api/graph", params={"workflow": "a/b", "window": "2d"})
    assert r.status_code == 422
    assert client.get("/api/graph", params={"workflow": "a/b", "run_id": other}).status_code == 404
    assert client.get("/api/graph", params={"workflow": "nope"}).status_code == 404
    assert client.get("/api/graph").status_code == 422  # the workflow is required

    r = client.get("/api/graph", params={"workflow": "a/b"})
    assert r.status_code == 200
    body = r.json()
    assert (body["runs_in_scope"], body["window"], body["run_id"]) == (1, "24h", None)
    assert body["slowest_path"] == ["workflow", P, "llm:planner|m1"]
    assert {n["id"] for n in body["nodes"]} == {"workflow", P, "llm:planner|m1"}
    assert not any(n["time_partial"] for n in body["nodes"])  # every latency known
    r = client.get("/api/graph", params={"workflow": "a/b", "run_id": run, "window": "7d"})
    assert r.status_code == 200 and r.json()["run_id"] == run
    r = client.get("/api/graph", params={"workflow": "a/b", "run_id": ""})  # an empty run_id scopes nothing
    assert r.status_code == 200 and r.json()["run_id"] is None


def test_graph_survives_malformed_rows(rt):
    from pitstop.server.graph import MAX_DEPTH, build_graph

    s = rt.store
    sdk = add_run(s, "wf", started_at=T0)
    _agent(s, sdk, "a", "b", ts=T0)  # a cycle: a under b under a
    _agent(s, sdk, "b", "a", ts=T0)
    _agent(s, sdk, "c", "c", ts=T0)  # its own parent, with no other entry
    add_event(s, sdk, "agent", ts="soon", name=None, agent=None, payload="{not json")  # nameless: skipped
    add_event(s, sdk, "agent", ts=None, name="d", agent="d", payload={"parent": 5})  # a parent that is no name
    _llm(s, sdk, "a", ts="later", latency=-1.0, cost="free")
    add_event(s, sdk, "tool_result", ts=None, name=None, agent="c", payload='["a", "list"]')
    otel = add_run(s, "wf", started_at=T0 + 1, adapter="otel")
    _agent(s, otel, "x", ts=T0, span_id="s1", parent_span_id="s1")  # its own parent span
    _agent(s, otel, "y", ts=T0, span_id="s2", parent_span_id="s3")
    _agent(s, otel, "z", ts=T0, span_id="s3", parent_span_id="s2")  # a span cycle
    _llm(s, otel, "y", ts=T0, span_id=7, parent_span_id="s2")
    for chain in range(1000):  # a chain far deeper than MAX_DEPTH
        _agent(s, otel, "deep", ts=T0, span_id=f"d{chain + 1}", parent_span_id=f"d{chain}")

    for run in (sdk, otel):
        g = build_graph(s, "wf", run_id=run)
        ids = [n.id for n in g.nodes]
        assert ids[0] == "workflow" and len(ids) == len(set(ids))
        assert {e.target for e in g.edges} == set(ids) - {"workflow"}  # still a tree
        assert max(len(n.agent_path) for n in g.nodes) <= MAX_DEPTH
    nodes = _nodes(build_graph(s, "wf", run_id=sdk))
    assert {"agent:d", "agent:c › c", "tool:c › c|(unknown tool)"} <= set(nodes)
    llm = next(n for n in nodes.values() if n.kind == "llm")
    assert (llm.calls, llm.cost_usd, llm.latency_p50_s) == (1, 0.0, None)


def test_windows_match_the_endpoint():
    from typing import get_args

    from pitstop.server.graph import WINDOWS, Window

    assert set(get_args(Window)) == set(WINDOWS)
