// The live feed: newest first, each item its kind, workflow (linked), level (drifts), text and time ago. An item
// that was not in the previous answer fades in (opacity only, 150 ms; nothing under reduced motion); the first
// answer arrives quietly.
import { useState } from "react";
import { Link } from "react-router";
import type { FeedItem } from "../api/client";
import { workflowHref } from "../routes";
import { timeAgo } from "../time";
import { Level } from "./Level";

const KIND_LABEL: Record<FeedItem["kind"], string> = {
  drift: "Drift",
  halt: "Halt",
  fallback: "Fallback",
  refused: "Refused",
  model_change: "Model change",
};

const ids = (items: FeedItem[]) => new Set(items.map((item) => item.id));

interface Seen {
  items: FeedItem[];
  known: Set<string>;
  /** The ids that were not in the previous answer; they keep their fade-in class until the next answer. */
  fresh: Set<string>;
}

export function Feed({ items, nowMs }: { items: FeedItem[]; nowMs: number }) {
  const [seen, setSeen] = useState<Seen>(() => ({ items, known: ids(items), fresh: new Set() }));
  if (seen.items !== items) {
    // a new answer: derive the fresh set from the previous one (React's adjust-state-on-prop-change pattern)
    setSeen({ items, known: ids(items), fresh: new Set(items.filter((i) => !seen.known.has(i.id)).map((i) => i.id)) });
  }

  if (items.length === 0) {
    return <p className="feed-empty">Nothing yet. Drifts, halts, fallbacks, refusals and model changes land here.</p>;
  }
  return (
    <ol className="feed" aria-label="Live feed">
      {items.map((item) => {
        const fresh = seen.items === items && seen.fresh.has(item.id);
        return (
          <li key={item.id} className={fresh ? "feed-item feed-new" : "feed-item"}>
            <p className="feed-meta">
              <span className={`feed-kind feed-kind-${item.kind}`}>{KIND_LABEL[item.kind]}</span>
              {item.workflow !== null && (
                <Link className="feed-workflow" to={workflowHref(item.workflow)}>
                  {item.workflow}
                </Link>
              )}
              {item.ts !== null && <span className="feed-time">{timeAgo(item.ts, nowMs)}</span>}
            </p>
            <p className="feed-text">{item.text}</p>
            {item.kind === "drift" && <Level level={item.level} />}
          </li>
        );
      })}
    </ol>
  );
}
