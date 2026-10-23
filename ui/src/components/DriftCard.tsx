// One open drift in the right column. It answers the three questions (what drifted, what likely changed, what
// PitStop is doing), then how often and when. Explain opens the attribution inline below; Dismiss asks inline
// ("Dismiss? Yes / No", never a browser dialog) and the page removes the card at once (optimistic).
import { useId, useState } from "react";
import type { DriftCard as DriftData } from "../api/client";
import { plural } from "../format";
import { timeAgo } from "../time";
import { ExplainPanel } from "./ExplainPanel";
import { Level } from "./Level";
import { plainHeadline } from "./Tile";

export interface DriftCardProps {
  drift: DriftData;
  /** The workflow's name, which the drift's title repeats and the card leaves out. */
  workflow: string;
  nowMs: number;
  onDismiss: (id: number) => void;
  /** Why the last dismiss failed, if it did; shown on the restored card. */
  dismissError?: string;
}

export function seenLine(drift: Pick<DriftData, "occurrences" | "first_seen" | "last_seen">, nowMs: number): string {
  const count = plural(drift.occurrences, "occurrence");
  if (drift.occurrences <= 1 || drift.first_seen === null) {
    const at = drift.last_seen ?? drift.first_seen;
    return at === null ? count : `${count} · ${timeAgo(at, nowMs)}`;
  }
  const last = drift.last_seen !== null ? ` · last ${timeAgo(drift.last_seen, nowMs)}` : "";
  return `${count} · first ${timeAgo(drift.first_seen, nowMs)}${last}`;
}

export function DriftCard({ drift, workflow, nowMs, onDismiss, dismissError }: DriftCardProps) {
  const [explaining, setExplaining] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const id = useId();
  const titleId = `${id}-title`;
  const explainId = `${id}-explain`;
  const title = plainHeadline({ headline: drift.title, name: workflow }) ?? drift.title;

  return (
    <article className={`drift drift-${(drift.level ?? "unknown").toLowerCase()}`} aria-labelledby={titleId}>
      <p className="drift-level">
        <Level level={drift.level} />
      </p>
      <h3 id={titleId} className="drift-title">
        {title}
      </h3>
      <dl className="drift-facts">
        <div>
          <dt>Likely cause</dt>
          <dd>{drift.cause}</dd>
        </div>
        <div>
          <dt>What PitStop did</dt>
          <dd>{drift.doing}</dd>
        </div>
      </dl>
      <p className="drift-seen">{seenLine(drift, nowMs)}</p>
      {dismissError && (
        <p className="drift-error" role="alert">
          Couldn't dismiss: {dismissError}
        </p>
      )}
      <div className="drift-actions">
        <button
          type="button"
          className="btn btn-ghost"
          aria-expanded={explaining}
          aria-controls={explainId}
          onClick={() => setExplaining((v) => !v)}
        >
          {explaining ? "Hide explanation" : "Explain"}
        </button>
        {confirming ? (
          <span className="drift-confirm" role="group" aria-label="Dismiss this drift?">
            <span>Dismiss?</span>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                setConfirming(false);
                onDismiss(drift.id);
              }}
            >
              Yes<span className="sr-only">, dismiss it</span>
            </button>
            <button type="button" className="btn" onClick={() => setConfirming(false)}>
              No<span className="sr-only">, keep it</span>
            </button>
          </span>
        ) : (
          <button type="button" className="btn btn-ghost" onClick={() => setConfirming(true)}>
            Dismiss
          </button>
        )}
      </div>
      <div id={explainId}>{explaining && <ExplainPanel driftId={drift.id} />}</div>
    </article>
  );
}
