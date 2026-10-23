// Words for a run-graph node, shared by the node card and the node drawer: its kind, where it sits, its verdicts
// as "2 blocked"-style badges, and its time per run (a lower bound when some calls had no recorded latency).
import type { GraphNode } from "../api/client";
import { duration } from "../time";

export const KIND_WORD: Record<GraphNode["kind"], string> = {
  workflow: "workflow",
  agent: "agent",
  llm: "model",
  tool: "tool",
};

/** The agent the node sits under: "under researcher", "under planner › worker"; nothing for the root and top-level agents. */
export function nodeWhere(node: Pick<GraphNode, "kind" | "agent_path">): string {
  const path = node.kind === "agent" ? node.agent_path.slice(0, -1) : node.agent_path;
  return path.length ? `under ${path.join(" › ")}` : "";
}

/** The node's accessible name: "m-a, model under researcher", "researcher, agent", "earnings-research, workflow". */
export function nodeName(node: Pick<GraphNode, "kind" | "label" | "agent_path">): string {
  const where = nodeWhere(node);
  return `${node.label}, ${KIND_WORD[node.kind]}${where ? ` ${where}` : ""}`;
}

/** "Model under researcher", "Agent", "Workflow". */
export function nodeSubtitle(node: Pick<GraphNode, "kind" | "agent_path">): string {
  const kind = KIND_WORD[node.kind];
  const where = nodeWhere(node);
  const text = where ? `${kind} ${where}` : kind;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export interface Badge {
  key: keyof GraphNode["verdicts"];
  text: string;
  /** p0: the run stopped or a call was refused; p1: PitStop steered, fell back or the call errored. */
  tone: "p0" | "p1";
}

const VERDICTS: { key: keyof GraphNode["verdicts"]; word: string; tone: Badge["tone"] }[] = [
  { key: "halt", word: "halted", tone: "p0" },
  { key: "block", word: "blocked", tone: "p0" },
  { key: "gate_reject", word: "gate reject", tone: "p0" },
  { key: "steer", word: "steered", tone: "p1" },
  { key: "fallback", word: "fallback", tone: "p1" },
  { key: "gate_error", word: "gate error", tone: "p1" },
  { key: "tool_error", word: "tool error", tone: "p1" },
];

/** The non-zero verdicts as text badges, most serious first: "1 halted", "2 blocked", "3 steered". */
export function verdictBadges(node: Pick<GraphNode, "verdicts">): Badge[] {
  return VERDICTS.filter((v) => node.verdicts[v.key] > 0).map((v) => ({
    key: v.key,
    text: `${node.verdicts[v.key]} ${v.word}`,
    tone: v.tone,
  }));
}

/** Time per run: "2.0 ms", "≥ 2.0 ms" when partial (a lower bound), "—" when unknown. */
export function timePerRun(node: Pick<GraphNode, "time_per_run_s" | "time_partial">): string {
  const text = duration(node.time_per_run_s);
  return text !== "—" && node.time_partial ? `≥ ${text}` : text;
}

/** The first 8 characters of a run id, enough to tell runs apart on screen. */
export function shortRunId(id: string): string {
  return id.length > 8 ? id.slice(0, 8) : id;
}
