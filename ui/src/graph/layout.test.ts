import { describe, expect, it } from "vitest";
import type { Graph } from "../api/client";
import { graph } from "../test/fixtures";
import { layoutGraph, nodeSize } from "./layout";

describe("layout", () => {
  it("layout places root left of agents and leaves", async () => {
    const laid = await layoutGraph(graph);
    const at = (id: string) => {
      const node = laid.nodes.find((n) => n.id === id);
      if (!node) throw new Error(`no laid node ${id}`);
      return node;
    };
    expect(laid.nodes).toHaveLength(graph.nodes.length);

    // left to right: workflow, then the agents, then their models and tools
    expect(at("workflow").x).toBeLessThan(at("agent:researcher").x);
    expect(at("workflow").x).toBeLessThan(at("agent:writer").x);
    expect(at("agent:researcher").x).toBeLessThan(at("llm:researcher|m-a").x);
    expect(at("agent:researcher").x).toBeLessThan(at("tool:researcher|search_filings").x);
    expect(at("agent:writer").x).toBeLessThan(at("tool:writer|publish_note").x);
    // siblings share a column and never overlap vertically
    expect(at("agent:researcher").x).toBe(at("agent:writer").x);
    const [top, bottom] = [at("agent:researcher"), at("agent:writer")].sort((a, b) => a.y - b.y);
    expect(top.y + top.height).toBeLessThanOrEqual(bottom.y);
    // every node keeps the size the node card renders at
    for (const node of graph.nodes) {
      expect({ width: at(node.id).width, height: at(node.id).height }).toEqual(nodeSize(node));
    }
    expect(laid.width).toBeGreaterThan(0);
    expect(laid.height).toBeGreaterThan(0);
  });

  it("memoizes the layout per graph structure", async () => {
    const laid = await layoutGraph(graph);
    expect(await layoutGraph(graph)).toBe(laid);
    // a poll answers with new objects and new numbers, same structure: the nodes stay where they were
    const restated: Graph = { ...graph, nodes: graph.nodes.map((n) => ({ ...n, calls: n.calls + 1, cost_usd: 1 })) };
    expect(await layoutGraph(restated)).toBe(laid);
    // a node gained verdicts: its card grows a badge row, so the layout is computed again
    const grown: Graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === "llm:researcher|m-a" ? { ...n, verdicts: { ...n.verdicts, fallback: 2 } } : n,
      ),
    };
    const relaid = await layoutGraph(grown);
    expect(relaid).not.toBe(laid);
    expect(relaid.nodes.find((n) => n.id === "llm:researcher|m-a")?.height).toBeGreaterThan(
      laid.nodes.find((n) => n.id === "llm:researcher|m-a")?.height ?? Infinity,
    );
  });

  it("lays out an empty graph", async () => {
    const empty: Graph = { ...graph, nodes: [], edges: [], runs_in_scope: 0, slowest_path: [] };
    const laid = await layoutGraph(empty);
    expect(laid.nodes).toEqual([]);
  });
});
