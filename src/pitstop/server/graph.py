"""The run graph (decision 8): a workflow drawn as a tree of agents, LLM calls and tools, with per-node statistics
over the runs in scope (one run, or the newest MAX_RUNS runs started in a window).

Nodes are keyed by agent path, so two agents of one name under different parents stay apart:
- `workflow` is the root;
- `agent:<path>` is an agent, `<path>` its chain of names joined with " › ";
- `llm:<agent path>|<model>` and `tool:<agent path>|<tool name>` are leaves (an empty path: under the root).

Where an event's agent sits:
- SDK runs have no spans. Each `agent` event names its enclosing agent (`payload.parent`), written once per
  (run, parent, name); an event's `agent` column names its agent. A name entered under several parents in one run
  resolves to the one it was most recently entered under at the event's time (the first, before any).
- OTel runs (`runs.adapter = 'otel'`) link events by span: an event's agent is the nearest `agent` event walking up
  `parent_span_id`. When the walk leaves the recorded spans (a framework span PitStop does not record), the event's
  `agent` column (the nearest agent span's name, resolved at record time) places it by name instead.
- An event with no agent hangs directly under the root; an agent without an `agent` event under the root too.

Only agent-meter `llm_call`s count: the judge's, supervisor's and correction calls are PitStop's own. Latency
unknown on an event (older events, a malformed value) is left out, never counted as 0.

Malformed rows never break the graph: a payload that is not JSON is empty, a non-numeric cost or latency is
unknown, and a broken or cyclic parent chain stops at the root."""
from __future__ import annotations

import statistics
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Literal

from .models import Graph, GraphEdge, GraphNode, GraphVerdicts
from .views import NotFound, _num, _payload, _str

Window = Literal["1h", "24h", "7d", "30d"]
WINDOWS = {"1h": 3600.0, "24h": 86400.0, "7d": 7 * 86400.0, "30d": 30 * 86400.0}
MAX_RUNS = 200
RUN_IDS_MAX = 20
SEP = " › "
ROOT = "workflow"
MAX_DEPTH = 64  # an agent path keeps at most this many innermost names (a deeper chain drops its outermost)
_TOOL_VERDICTS = {"block": "block", "steer": "steer", "gate": "gate_reject", "gate_error": "gate_error"}
_LEAF_KINDS = ("llm_call", "tool_call", "tool_result", "block", "steer", "gate", "gate_error", "halt", "fallback")
_NUMERIC = "typeof(started_at) IN ('integer', 'real')"


@dataclass
class _Node:
    id: str
    kind: str
    label: str
    path: tuple[str, ...]
    calls: int = 0
    cost: float = 0.0
    latencies: list[float] = field(default_factory=list)
    timed: int = 0  # events that took time (llm calls, tool results), whether or not their latency is known
    runs: set[str] = field(default_factory=set)
    verdicts: dict[str, int] = field(default_factory=dict)
    children: set[str] = field(default_factory=set)
    time: float | None = None


def _agent_id(path: tuple[str, ...]) -> str:
    return f"agent:{SEP.join(path)}" if path else ROOT


class _Builder:
    def __init__(self, workflow: str):
        self.nodes: dict[str, _Node] = {ROOT: _Node(ROOT, "workflow", workflow, ())}

    def agent(self, path: tuple[str, ...]) -> _Node:
        """The agent node at `path`, created with its ancestors."""
        node_id = _agent_id(path)
        node = self.nodes.get(node_id)
        if node is None:
            node = self.nodes[node_id] = _Node(node_id, "agent", path[-1], path)
            self.agent(path[:-1]).children.add(node_id)
        return node

    def leaf(self, kind: str, path: tuple[str, ...], name: str, *, create: bool = True) -> _Node | None:
        node_id = f"{kind}:{SEP.join(path)}|{name}"
        node = self.nodes.get(node_id)
        if node is None and create:
            node = self.nodes[node_id] = _Node(node_id, kind, name, path)
            self.agent(path).children.add(node_id)
        return node


def _order_key(e) -> tuple[float, int]:
    ts = _num(e["ts"])
    return (ts if ts is not None else float("-inf"), e["id"])


class _RunAgents:
    """Resolves the agent path of each event of one run."""

    def __init__(self, events: list, otel: bool):
        self.otel = otel
        self.agents = [e for e in events if e["kind"] == "agent" and self.name_of(e) is not None]
        self.by_name: dict[str, list] = {}
        for e in sorted(self.agents, key=_order_key):
            self.by_name.setdefault(self.name_of(e), []).append(e)
        self.keys = {name: [_order_key(e) for e in lst] for name, lst in self.by_name.items()}
        self.span_parent: dict[str, str | None] = {}
        self.span_agent: dict[str, object] = {}
        if otel:
            for e in events:
                span = _str(e["span_id"])
                if span is None:
                    continue
                self.span_parent.setdefault(span, _str(e["parent_span_id"]))
                if e["kind"] == "agent" and self.name_of(e) is not None:
                    self.span_agent.setdefault(span, e)
        self.memo: dict[int, tuple[str, ...]] = {}

    @staticmethod
    def name_of(e) -> str | None:
        return _str(e["name"]) if e["name"] is not None else _str(e["agent"])

    def by_name_at(self, name: str, at, exclude=None):
        """The agent event named `name` most recently entered at or before event `at` (the first one before any)."""
        entered = self.by_name.get(name)
        if not entered:
            return None
        skip = exclude["id"] if exclude is not None else None
        i = bisect_right(self.keys[name], _order_key(at))
        for j in (i - 1, i - 2):  # the latest at or before `at`, other than `exclude`
            if j >= 0 and entered[j]["id"] != skip:
                return entered[j]
        return next((c for c in entered[:2] if c["id"] != skip), None)

    def nearest_agent(self, span: str | None, *, exclude=None):
        """The nearest `agent` event from span `span` up; None when the walk leaves the recorded spans."""
        seen = set()
        while span is not None and span not in seen:
            seen.add(span)
            e = self.span_agent.get(span)
            if e is not None and (exclude is None or e["id"] != exclude["id"]):
                return e
            if span not in self.span_parent:
                return None
            span = self.span_parent[span]
        return None

    def path_of_agent(self, e, visiting: frozenset = frozenset()) -> tuple[str, ...]:
        if e["id"] in self.memo:
            return self.memo[e["id"]]
        name = self.name_of(e)
        if self.otel:
            parent = self.nearest_agent(_str(e["parent_span_id"]), exclude=e)
            parent_path = None
        else:
            parent_name = _payload(e["payload"]).get("parent")
            parent_name = parent_name if isinstance(parent_name, str) else None
            parent = self.by_name_at(parent_name, e, exclude=e) if parent_name is not None else None
            parent_path = (parent_name,) if parent_name is not None and parent is None else None
        if parent is not None and parent["id"] not in visiting and len(visiting) < MAX_DEPTH:
            parent_path = self.path_of_agent(parent, visiting | {e["id"]})
        path = (*(parent_path or ()), name)[-MAX_DEPTH:]
        self.memo[e["id"]] = path
        return path

    def path_of(self, e) -> tuple[str, ...]:
        """The agent path of a non-agent event."""
        if self.otel:
            a = self.nearest_agent(_str(e["parent_span_id"]))
            if a is not None:
                return self.path_of_agent(a)
        name = _str(e["agent"])
        if name is None:
            return ()
        a = self.by_name_at(name, e)
        return self.path_of_agent(a) if a is not None else (name,)


def _scope(conn, workflow: str, run_id: str | None, window: str, now: float, max_runs: int) -> list:
    if run_id is not None:
        r = conn.execute("SELECT id, workflow, adapter FROM runs WHERE id = ?", (run_id,)).fetchone()
        if r is None or r["workflow"] != workflow:
            raise NotFound(f"no run {run_id!r} of workflow {workflow!r}")
        return [r]
    if conn.execute("SELECT 1 FROM runs WHERE workflow = ? LIMIT 1", (workflow,)).fetchone() is None:
        raise NotFound(f"no workflow {workflow!r}")
    return conn.execute(f"SELECT id, adapter FROM runs WHERE workflow = ? AND started_at >= ? AND started_at <= ?"
                        f" AND {_NUMERIC} ORDER BY started_at DESC, rowid DESC LIMIT ?",
                        (workflow, now - WINDOWS[window], now, max(0, int(max_runs)))).fetchall()


def _add_event(b: _Builder, e, path: tuple[str, ...], run_id: str) -> None:
    kind = e["kind"]
    p = _payload(e["payload"])
    name = _str(e["name"])
    node = None
    if kind == "llm_call":
        if p.get("meter") != "agent":
            return
        node = b.leaf("llm", path, name if name is not None else "(unknown model)")
        node.calls += 1
        node.cost += _num(e["cost_usd"]) or 0.0
    elif kind in ("tool_call", "tool_result") or kind in _TOOL_VERDICTS:
        node = b.leaf("tool", path, name if name is not None else "(unknown tool)")
        if kind == "tool_call":
            node.calls += 1
        elif kind == "tool_result" and "error" in p:
            node.verdicts["tool_error"] = node.verdicts.get("tool_error", 0) + 1
        elif kind == "gate" and p.get("allow") is not False:
            pass
        elif kind in _TOOL_VERDICTS:
            node.verdicts[_TOOL_VERDICTS[kind]] = node.verdicts.get(_TOOL_VERDICTS[kind], 0) + 1
    elif kind in ("halt", "fallback"):  # named by a tool or a model; otherwise the agent's own
        if name is not None:
            node = b.leaf("tool", path, name, create=False) or b.leaf("llm", path, name, create=False)
        node = node or b.agent(path)
        node.verdicts[kind] = node.verdicts.get(kind, 0) + 1
    if node is None:
        return
    if kind in ("llm_call", "tool_result"):
        node.timed += 1
        latency = _num(p.get("latency_s"))
        if latency is not None and latency >= 0:
            node.latencies.append(latency)
    node.runs.add(run_id)


def _roll_up(b: _Builder, node: _Node, n_runs: int) -> None:
    """Post-order: leaves get their time per run; agents and the root sum their subtree."""
    if node.kind in ("llm", "tool"):
        if node.latencies and n_runs:
            node.time = statistics.fmean(node.latencies) * node.timed / n_runs  # the known mean stands in for unknowns
        return
    times = []
    for child_id in node.children:
        child = b.nodes[child_id]
        _roll_up(b, child, n_runs)
        node.calls += child.calls
        node.cost += child.cost
        node.runs |= child.runs
        if child.time is not None:
            times.append(child.time)
    node.time = sum(times) if times else None


def build_graph(store, workflow: str, *, run_id: str | None = None, window: str = "24h", now: float | None = None,
                max_runs: int = MAX_RUNS) -> Graph:
    """The run graph of `workflow` over one run (`run_id`) or the newest `max_runs` runs started in `window`
    (one of WINDOWS) before `now`. ValueError for an unknown window; NotFound for a workflow without runs or a run
    that is not the workflow's. One events query per scope, over `idx_events_run`."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {', '.join(WINDOWS)}")
    now = time.time() if now is None else now
    conn = store.connect()
    runs = _scope(conn, workflow, run_id, window, now, max_runs)
    order = {r["id"]: i for i, r in enumerate(runs)}  # newest first
    by_run: dict[str, list] = {r["id"]: [] for r in runs}
    if runs:
        marks = ",".join("?" * len(runs))
        for e in conn.execute("SELECT id, run_id, ts, agent, kind, name, payload, cost_usd, span_id, parent_span_id"
                              f" FROM events WHERE run_id IN ({marks})", list(order)):
            by_run[e["run_id"]].append(e)

    b = _Builder(workflow)
    for r in runs:
        events = sorted(by_run[r["id"]], key=lambda e: e["id"])
        agents = _RunAgents(events, r["adapter"] == "otel")
        for e in agents.agents:
            b.agent(agents.path_of_agent(e)).runs.add(r["id"])
        for e in events:
            if e["kind"] in _LEAF_KINDS:
                _add_event(b, e, agents.path_of(e), r["id"])

    root = b.nodes[ROOT]
    _roll_up(b, root, len(runs))
    root.runs = set(order)

    slowest: list[str] = []
    node = root
    while True:
        timed = [b.nodes[c] for c in sorted(node.children) if b.nodes[c].time is not None]
        if not timed:
            break
        node = max(timed, key=lambda c: c.time)  # the first by id on a tie
        slowest.append(node.id)
    slowest = [ROOT, *slowest] if slowest else []
    on_path = set(slowest)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    def emit(n: _Node) -> None:
        nodes.append(GraphNode(
            id=n.id, kind=n.kind, label=n.label, agent_path=list(n.path), calls=n.calls, runs=len(n.runs),
            cost_usd=n.cost, latency_p50_s=statistics.median(n.latencies) if n.latencies else None,
            time_per_run_s=n.time, verdicts=GraphVerdicts(**n.verdicts), on_slowest_path=n.id in on_path,
            run_ids=sorted(n.runs, key=order.__getitem__)[:RUN_IDS_MAX]))
        for child_id in sorted(n.children):
            edges.append(GraphEdge(source=n.id, target=child_id))
            emit(b.nodes[child_id])

    emit(root)
    return Graph(nodes=nodes, edges=edges, runs_in_scope=len(runs), window=None if run_id is not None else window,
                 run_id=run_id, slowest_path=slowest)
