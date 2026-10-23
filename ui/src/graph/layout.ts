// The run graph's layout: ELK's layered algorithm, left to right (decision 8: the workflow root, its agents, their
// models and tools). Pure and async, so it is testable in jsdom; elkjs's bundled build runs on the main thread
// without a worker (the graphs are small: at most a few dozen nodes). The result is memoized per graph structure,
// so a poll that only changes the numbers leaves every node where it was.
import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import type { Graph, GraphNode } from "../api/client";

export interface PlacedNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Layout {
  nodes: PlacedNode[];
  width: number;
  height: number;
}

/** Node card sizes by kind; the agents and the root are a little wider than their models and tools. */
const WIDTH: Record<GraphNode["kind"], number> = { workflow: 208, agent: 208, llm: 188, tool: 188 };
const BASE_HEIGHT = 86; // kind caption, label and two fact lines
const BADGE_ROW = 26; // one more row when the node has verdicts

export function hasVerdicts(node: Pick<GraphNode, "verdicts">): boolean {
  return Object.values(node.verdicts).some((n) => n > 0);
}

/** The size the node card renders at, which ELK lays out around. */
export function nodeSize(node: Pick<GraphNode, "kind" | "verdicts">): { width: number; height: number } {
  return { width: WIDTH[node.kind], height: BASE_HEIGHT + (hasVerdicts(node) ? BADGE_ROW : 0) };
}

/** What the layout depends on: node ids, kinds and sizes, and the edges. Statistics do not move nodes. */
function structureKey(graph: Graph): string {
  const nodes = graph.nodes.map((n) => `${n.id}\u0001${n.kind}\u0001${hasVerdicts(n) ? 1 : 0}`).join("\n");
  const edges = graph.edges.map((e) => `${e.source}\u0001${e.target}`).join("\n");
  return `${nodes}\n\u0002\n${edges}`;
}

const OPTIONS = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.layered.spacing.nodeNodeBetweenLayers": "72",
  "elk.spacing.nodeNode": "20",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES", // keep the server's order: siblings as declared
  "elk.padding": "[top=8,left=8,bottom=8,right=8]",
};

let elk: InstanceType<typeof ELK> | null = null;

async function compute(graph: Graph): Promise<Layout> {
  if (graph.nodes.length === 0) return { nodes: [], width: 0, height: 0 };
  elk ??= new ELK();
  const ids = new Set(graph.nodes.map((n) => n.id));
  const root: ElkNode = {
    id: "root",
    layoutOptions: OPTIONS,
    children: graph.nodes.map((n) => ({ id: n.id, ...nodeSize(n) })),
    edges: graph.edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e, i) => ({ id: `e${i}`, sources: [e.source], targets: [e.target] })),
  };
  const out = await elk.layout(root);
  return {
    nodes: (out.children ?? []).map((c) => ({
      id: c.id,
      x: c.x ?? 0,
      y: c.y ?? 0,
      width: c.width ?? 0,
      height: c.height ?? 0,
    })),
    width: out.width ?? 0,
    height: out.height ?? 0,
  };
}

const CACHE_MAX = 8;
const cache = new Map<string, Promise<Layout>>();

/** Lay `graph` out, or hand back the layout of the last graph with the same structure. */
export function layoutGraph(graph: Graph): Promise<Layout> {
  const key = structureKey(graph);
  const hit = cache.get(key);
  if (hit) return hit;
  const pending = compute(graph).catch((e: unknown) => {
    cache.delete(key);
    throw e;
  });
  cache.set(key, pending);
  if (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  return pending;
}
