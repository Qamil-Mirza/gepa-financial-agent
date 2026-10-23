// A run's events in order, grouped by the agent that was active (a new group each time the agent changes), each
// with its kind as a glyph and a word, its name, the server's one-line summary, and its latency and cost when
// recorded (unknown is "—", never 0: decision 9). Verdicts stand out in words and tone: a block, halt or denied
// gate in --p0; a steer, fallback, gate error, refusal or tool error in --p1.
import type { EventRow } from "../api/client";
import { plural, words } from "../format";
import { clock, dateTime, duration } from "../time";
import { Money } from "./Money";

export interface EventTimelineProps {
  events: EventRow[];
  /** The run's start, for each event's offset ("+1.2 s"); a wall-clock time when unknown. */
  startedAt: number | null;
  truncated: boolean;
}

const KIND_WORD: Record<string, string> = {
  llm_call: "model call",
  tool_call: "tool call",
  tool_result: "tool result",
  agent: "agent",
  gate: "gate",
  gate_error: "gate error",
  block: "blocked",
  steer: "steered",
  halt: "halted",
  fallback: "fallback",
};

type Tone = "p0" | "p1" | "plain";

/** The event's urgency: what stopped something is p0, what PitStop corrected or what failed is p1. */
export function eventTone(e: Pick<EventRow, "kind" | "summary">): Tone {
  switch (e.kind) {
    case "block":
    case "halt":
      return "p0";
    case "gate":
      return e.summary.startsWith("denied") ? "p0" : "plain";
    case "steer":
    case "fallback":
    case "gate_error":
      return "p1";
    case "tool_result":
      return e.summary.startsWith("error:") ? "p1" : "plain";
    case "llm_call":
      return /\brefused\b/.test(e.summary) ? "p1" : "plain";
    default:
      return "plain";
  }
}

function Glyph({ kind }: { kind: string | null }) {
  const common = { className: "ev-glyph", viewBox: "0 0 12 12", "aria-hidden": true, focusable: "false" } as const;
  switch (kind) {
    case "agent":
      return (
        <svg {...common}>
          <circle cx="6" cy="4" r="2.2" />
          <path d="M1.8 11c.4-2.6 2-3.8 4.2-3.8s3.8 1.2 4.2 3.8" />
        </svg>
      );
    case "llm_call":
      return (
        <svg {...common}>
          <path d="M2 2.5h8v5.5H5.5L3 10V8H2z" />
        </svg>
      );
    case "tool_call":
      return (
        <svg {...common}>
          <path d="M2 6h7M6.5 3l3 3-3 3" />
        </svg>
      );
    case "tool_result":
      return (
        <svg {...common}>
          <path d="M10 6H3M5.5 3l-3 3 3 3" />
        </svg>
      );
    case "gate":
    case "gate_error":
      return (
        <svg {...common}>
          <path d="M6 1.5l4 1.5v3c0 2.4-1.7 4-4 4.8C3.7 10 2 8.4 2 6V3z" />
        </svg>
      );
    case "block":
    case "halt":
      return (
        <svg {...common}>
          <circle cx="6" cy="6" r="4.5" />
          <path d="M3 6h6" />
        </svg>
      );
    case "steer":
    case "fallback":
      return (
        <svg {...common}>
          <path d="M2 8.5c2.5 0 3-5 6-5h2M8.5 2l1.5 1.5L8.5 5" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="6" cy="6" r="2" />
        </svg>
      );
  }
}

interface Group {
  agent: string | null;
  events: EventRow[];
}

/** Consecutive events of one agent make a group; the agent order is the run's order. */
export function groupByAgent(events: EventRow[]): Group[] {
  const groups: Group[] = [];
  for (const e of events) {
    const last = groups[groups.length - 1];
    if (last && last.agent === e.agent) last.events.push(e);
    else groups.push({ agent: e.agent, events: [e] });
  }
  return groups;
}

function when(e: EventRow, startedAt: number | null): { text: string; title?: string } {
  if (e.ts === null) return { text: "—" };
  if (startedAt !== null && e.ts >= startedAt) return { text: `+${duration(e.ts - startedAt)}`, title: dateTime(e.ts) };
  return { text: clock(e.ts), title: dateTime(e.ts) };
}

export function EventTimeline({ events, startedAt, truncated }: EventTimelineProps) {
  if (events.length === 0) return <p className="muted">No events recorded for this run.</p>;
  const groups = groupByAgent(events);
  return (
    <div className="timeline">
      {groups.map((g, gi) => {
        const name = g.agent ?? "no agent";
        const label = `${plural(g.events.length, "event")} of ${name}`;
        return (
          <section key={`${name}-${gi}`} className="ev-group">
            <h3 className="ev-agent">
              {name}
              <span className="muted"> · {plural(g.events.length, "event")}</span>
            </h3>
            <ol className="ev-list" aria-label={label}>
              {g.events.map((e) => {
                const tone = eventTone(e);
                const at = when(e, startedAt);
                return (
                  <li key={e.id} className={tone === "plain" ? "ev" : `ev ev-${tone}`}>
                    <span className="ev-when" title={at.title}>
                      {at.text}
                    </span>
                    <span className="ev-kind">
                      <Glyph kind={e.kind} />
                      {e.kind ? (KIND_WORD[e.kind] ?? words(e.kind).toLowerCase()) : "event"}
                    </span>
                    <span className="ev-body">
                      {e.name && <b className="ev-name">{e.name}</b>}
                      {e.summary && <span className="ev-summary">{e.summary}</span>}
                    </span>
                    <span className="ev-facts">
                      {e.latency_s !== null && <span title="latency">{duration(e.latency_s)}</span>}
                      {e.cost_usd !== null && <Money usd={e.cost_usd} />}
                    </span>
                  </li>
                );
              })}
            </ol>
          </section>
        );
      })}
      {truncated && <p className="ev-truncated">Showing the first 2,000 events</p>}
    </div>
  );
}
