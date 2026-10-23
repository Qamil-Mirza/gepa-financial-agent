// The Run graph tab: the workflow as a DAG of agents, models and tools (decision 8), laid out by ELK and drawn by
// React Flow, over a window of runs or one run. Each node card shows its calls, cost, p50 latency, time per run
// and verdicts as text; the slowest path is heavier and says so in words. Clicking a node opens the NodeDrawer.
// This module is loaded lazily (the graph libraries are the heaviest part of the app).
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
  type ReactFlowState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { getGraph, getRuns, type Graph, type GraphNode, type GraphWindow, type RunRow } from "../api/client";
import { usePoll } from "../api/poll";
import { plural } from "../format";
import { layoutGraph, type Layout } from "../graph/layout";
import { KIND_WORD, nodeName, shortRunId, timePerRun, verdictBadges } from "../graph/nodes";
import { dateTime, duration } from "../time";
import { ErrorState } from "./ErrorState";
import { Money } from "./Money";
import { NodeDrawer } from "./NodeDrawer";
import { windowLabel, WindowPicker } from "./WindowPicker";

const EVERY_MS = 10_000;
const RUN_CHOICES = 20;
const CANVAS_MIN = 360;
const CANVAS_MAX = 720;

export interface RunGraphProps {
  name: string;
  window: GraphWindow;
  runId: string | null;
  onWindow: (window: GraphWindow) => void;
  onRun: (runId: string | null) => void;
}

// ---------- the node card ----------

interface PitData extends Record<string, unknown> {
  node: GraphNode;
  open: boolean;
}

type PitNode = Node<PitData, "pit">;

function PitNodeCard({ data }: NodeProps<PitNode>) {
  const { node, open } = data;
  const badges = verdictBadges(node);
  const cls = ["gnode", `gnode-${node.kind}`, node.on_slowest_path ? "gnode-slow" : "", open ? "gnode-open" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={cls}>
      {node.kind !== "workflow" && <Handle type="target" position={Position.Left} className="ghandle" isConnectable={false} />}
      {/* the click (or Enter on this button) bubbles to the node wrapper, whose onNodeClick opens the drawer */}
      <button type="button" className="gnode-button" aria-label={nodeName(node)} aria-pressed={open} data-node-open={node.id}>
        <span className="gnode-kind">
          {KIND_WORD[node.kind]}
          {node.on_slowest_path && <span className="gnode-slowtag">on slowest path</span>}
        </span>
        <span className="gnode-label" title={node.label}>
          {node.label}
        </span>
        <span className="gnode-facts">
          {plural(node.calls, "call")} · <Money usd={node.cost_usd} />
        </span>
        <span className="gnode-facts">
          p50 {duration(node.latency_p50_s)} · {timePerRun(node)} per run
        </span>
        {badges.length > 0 && (
          <span className="gnode-badges">
            {badges.map((b) => (
              <span key={b.key} className={`vbadge vbadge-${b.tone}`}>
                {b.text}
              </span>
            ))}
          </span>
        )}
      </button>
      {node.kind !== "llm" && node.kind !== "tool" && (
        <Handle type="source" position={Position.Right} className="ghandle" isConnectable={false} />
      )}
    </div>
  );
}

const NODE_TYPES: NodeTypes = { pit: PitNodeCard };

// ---------- the canvas ----------

function reducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

interface CanvasProps {
  nodes: PitNode[];
  edges: Edge[];
  layout: Layout | null;
  onOpen: (id: string) => void;
}

const canvasSize = (s: ReactFlowState) => `${s.width}x${s.height}`;

function Canvas({ nodes, edges, layout, onOpen }: CanvasProps) {
  const { fitView } = useReactFlow();
  const size = useStore(canvasSize); // React Flow's own measurement: the drawer, the layout height and the window change it
  useEffect(() => {
    if (!layout || size.startsWith("0x")) return;
    // a new scope or a resized canvas: show the whole graph. Nodes carry explicit sizes, so no measurement is awaited.
    void fitView({ padding: 0.12, duration: reducedMotion() ? 0 : 300 });
  }, [fitView, layout, size]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => onOpen(node.id)}
      fitView
      fitViewOptions={{ padding: 0.12 }}
      minZoom={0.3}
      maxZoom={1.5}
      nodesDraggable={false}
      nodesConnectable={false}
      nodesFocusable={false}
      edgesFocusable={false}
      elementsSelectable={false}
      zoomOnScroll={false}
      zoomOnDoubleClick={false}
      preventScrolling={false}
      className="graph-flow"
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.5} className="graph-dots" />
      <Controls showInteractive={false} position="bottom-right" />
    </ReactFlow>
  );
}

// ---------- the tab ----------

function pathPairs(path: string[]): Set<string> {
  const pairs = new Set<string>();
  for (let i = 1; i < path.length; i += 1) pairs.add(`${path[i - 1]}\u0001${path[i]}`);
  return pairs;
}

/** "Slowest path: earnings-research → researcher → m-a · 1.3 ms of 3.1 ms per run (43%)". */
export function slowestLine(graph: Graph): string | null {
  if (graph.slowest_path.length < 2) return null;
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const labels = graph.slowest_path.map((id) => byId.get(id)?.label ?? id);
  const root = byId.get(graph.slowest_path[0]);
  const leaf = byId.get(graph.slowest_path[graph.slowest_path.length - 1]);
  let time = "";
  if (leaf && root && leaf.time_per_run_s !== null && root.time_per_run_s !== null && root.time_per_run_s > 0) {
    const share = Math.round((leaf.time_per_run_s / root.time_per_run_s) * 100);
    time = ` · ${timePerRun(leaf)} of ${timePerRun(root)} per run (${share}%)`;
  }
  return `Slowest path: ${labels.join(" → ")}${time}`;
}

function scopeLine(graph: Graph, run: RunRow | undefined): string {
  if (graph.run_id) {
    const status = run?.status ? ` · ${run.status}` : "";
    const when = run?.started_at != null ? ` · ${dateTime(run.started_at)}` : "";
    return `Run ${shortRunId(graph.run_id)}${status}${when}`;
  }
  return `${windowLabel(graph.window)} · ${plural(graph.runs_in_scope, "run")}, averaged`;
}

function runOption(run: RunRow): string {
  const when = run.started_at != null ? ` · ${dateTime(run.started_at)}` : "";
  return `${shortRunId(run.id)} · ${run.status ?? "status unknown"}${when}`;
}

export default function RunGraph({ name, window: win, runId, onWindow, onRun }: RunGraphProps) {
  const graph = usePoll((signal) => getGraph(name, { runId, window: win }, { signal }), EVERY_MS, [name, win, runId]);
  const runs = usePoll((signal) => getRuns(name, null, { signal }), EVERY_MS, [name]);
  const [laid, setLaid] = useState<{ graph: Graph; layout: Layout } | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const data = graph.data;
  useEffect(() => {
    if (!data) return;
    let alive = true;
    layoutGraph(data).then(
      (layout) => {
        if (alive) setLaid({ graph: data, layout });
      },
      () => {
        if (alive) setLaid(null);
      },
    );
    return () => {
      alive = false;
    };
  }, [data]);

  const onOpen = useCallback((id: string) => setOpenId((cur) => (cur === id ? null : id)), []);
  const close = useCallback(() => {
    if (openId !== null) document.querySelector<HTMLElement>(`[data-node-open="${CSS.escape(openId)}"]`)?.focus();
    setOpenId(null);
  }, [openId]);

  const layout = laid?.layout ?? null;
  const { nodes, edges } = useMemo(() => {
    if (!data || !layout) return { nodes: [] as PitNode[], edges: [] as Edge[] };
    const placed = new Map(layout.nodes.map((p) => [p.id, p]));
    const byId = new Map(data.nodes.map((n) => [n.id, n]));
    const hot = pathPairs(data.slowest_path);
    const nodes: PitNode[] = data.nodes.flatMap((n) => {
      const p = placed.get(n.id);
      if (!p) return [];
      return [
        {
          id: n.id,
          type: "pit" as const,
          position: { x: p.x, y: p.y },
          width: p.width,
          height: p.height,
          data: { node: n, open: n.id === openId },
          draggable: false,
          connectable: false,
          selectable: false,
          focusable: false, // the card's button is the focus stop and carries the node's name
        },
      ];
    });
    const edges: Edge[] = data.edges
      .filter((e) => placed.has(e.source) && placed.has(e.target))
      .map((e) => {
        const slow = hot.has(`${e.source}\u0001${e.target}`);
        const from = byId.get(e.source)?.label ?? e.source;
        const to = byId.get(e.target)?.label ?? e.target;
        return {
          id: `${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          className: slow ? "gedge gedge-slow" : "gedge",
          ariaLabel: `${from} to ${to}${slow ? ", on slowest path" : ""}`,
          focusable: false,
          selectable: false,
        };
      });
    return { nodes, edges };
  }, [data, layout, openId]);

  const runsById = useMemo(() => new Map((runs.data?.runs ?? []).map((r) => [r.id, r])), [runs.data]);
  const choices = (runs.data?.runs ?? []).slice(0, RUN_CHOICES);
  const openNode = openId && data ? (data.nodes.find((n) => n.id === openId) ?? null) : null;
  const empty = data !== null && (data.nodes.length === 0 || data.runs_in_scope === 0);
  const slowest = data ? slowestLine(data) : null;
  // the canvas grows with the graph (within reason), so a workflow with many leaves is still readable after fit-view
  const canvasStyle = layout
    ? ({
        "--graph-h": `${Math.min(CANVAS_MAX, Math.max(CANVAS_MIN, layout.height + 56))}px`,
        "--graph-ar": `${Math.max(1, layout.width + 32)} / ${Math.max(1, layout.height + 56)}`, // narrow screens: width-bound
      } as CSSProperties)
    : undefined;

  return (
    <section className="graph" aria-label="Run graph">
      <div className="graph-bar">
        <WindowPicker value={win} onChange={onWindow} disabled={runId !== null} />
        <label className="runpick">
          <span className="runpick-label">Run</span>
          <select className="runpick-select" value={runId ?? ""} onChange={(e) => onRun(e.target.value || null)}>
            <option value="">All runs in the window</option>
            {runId !== null && !choices.some((r) => r.id === runId) && <option value={runId}>{shortRunId(runId)}</option>}
            {choices.map((r) => (
              <option key={r.id} value={r.id}>
                {runOption(r)}
              </option>
            ))}
          </select>
        </label>
        {data && (
          <p className="graph-scope" role="status">
            {scopeLine(data, runId ? runsById.get(runId) : undefined)}
          </p>
        )}
      </div>

      {graph.loading ? (
        <div className="graph-canvas graph-skeleton" aria-busy="true">
          <p className="sr-only" role="status">
            Loading the run graph…
          </p>
        </div>
      ) : !data ? (
        <ErrorState error={graph.error} title="The run graph didn't load" onRetry={graph.refresh} />
      ) : empty ? (
        <div className="graph-canvas graph-empty">
          <p>
            No runs in the {windowLabel(data.window)}.{" "}
            <span className="muted">Widen the window, or pick a run above.</span>
          </p>
        </div>
      ) : (
        <div className={openNode ? "graph-body graph-body-open" : "graph-body"} style={canvasStyle}>
          <div className="graph-canvas">
            {graph.error && (
              <p className="stale graph-stale" role="status" aria-label="Graph not refreshed">
                Couldn't refresh. Retrying.
              </p>
            )}
            <ReactFlowProvider>
              <Canvas nodes={nodes} edges={edges} layout={layout} onOpen={onOpen} />
            </ReactFlowProvider>
          </div>
          {openNode && (
            <NodeDrawer node={openNode} runsInScope={data.runs_in_scope} runsById={runsById} onClose={close} />
          )}
        </div>
      )}

      {slowest && !empty && <p className="graph-slowest">{slowest}</p>}
    </section>
  );
}
