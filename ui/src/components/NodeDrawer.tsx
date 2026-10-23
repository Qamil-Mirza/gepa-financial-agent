// The panel beside the run graph for one node: its statistics, its verdicts in words, and links to the runs it
// appears in (newest first, up to 20). Not a modal: the graph stays usable; Escape or Close puts focus back on
// the node.
import { useEffect, useRef } from "react";
import { Link } from "react-router";
import type { GraphNode, RunRow } from "../api/client";
import { plural } from "../format";
import { nodeSubtitle, shortRunId, timePerRun, verdictBadges } from "../graph/nodes";
import { runHref } from "../routes";
import { dateTime, duration } from "../time";
import { Money } from "./Money";

export interface NodeDrawerProps {
  node: GraphNode;
  runsInScope: number;
  /** The runs the page knows (the latest page), for a status and a time next to each run id. */
  runsById: ReadonlyMap<string, RunRow>;
  onClose: () => void;
}

const RUN_STATUS: Record<string, string> = {
  ok: "ok",
  halted: "halted",
  refused: "refused",
  error: "errored",
  dropped: "dropped",
  running: "running",
};

export function NodeDrawer({ node, runsInScope, runsById, onClose }: NodeDrawerProps) {
  const ref = useRef<HTMLElement>(null);
  const titleId = `node-${node.id.replace(/[^a-zA-Z0-9_-]/g, "_")}`;

  useEffect(() => {
    ref.current?.focus(); // the drawer took the node's click: tell the keyboard where it is
  }, [node.id]);

  const badges = verdictBadges(node);
  const partialNote = node.time_partial ? " (some calls had no recorded latency)" : "";

  return (
    <aside
      ref={ref}
      className="ndrawer"
      aria-labelledby={titleId}
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <header className="ndrawer-head">
        <div className="ndrawer-title">
          <p className="ndrawer-kind">
            {nodeSubtitle(node)}
            {node.on_slowest_path && <span className="gnode-slowtag">on slowest path</span>}
          </p>
          <h3 id={titleId} className="ndrawer-name">
            {node.label}
          </h3>
        </div>
        <button type="button" className="btn btn-ghost ndrawer-close" onClick={onClose}>
          Close
        </button>
      </header>

      <dl className="ndrawer-facts">
        <div>
          <dt>Calls</dt>
          <dd>{node.calls}</dd>
        </div>
        <div>
          <dt>Runs</dt>
          <dd>
            {node.runs} of {plural(runsInScope, "run")}
          </dd>
        </div>
        <div>
          <dt>Cost</dt>
          <dd>
            <Money usd={node.cost_usd} />
          </dd>
        </div>
        <div>
          <dt>p50 latency</dt>
          <dd>{duration(node.latency_p50_s)}</dd>
        </div>
        <div>
          <dt>Time per run</dt>
          <dd>
            {timePerRun(node)}
            {partialNote && <span className="muted">{partialNote}</span>}
          </dd>
        </div>
      </dl>

      <section className="ndrawer-section" aria-labelledby={`${titleId}-verdicts`}>
        <h4 id={`${titleId}-verdicts`} className="ndrawer-caption">
          Verdicts
        </h4>
        {badges.length === 0 ? (
          <p className="muted">No verdicts</p>
        ) : (
          <ul className="vbadges">
            {badges.map((b) => (
              <li key={b.key} className={`vbadge vbadge-${b.tone}`}>
                {b.text}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="ndrawer-section" aria-labelledby={`${titleId}-runs`}>
        <h4 id={`${titleId}-runs`} className="ndrawer-caption">
          Runs{node.run_ids.length < node.runs ? ` (latest ${node.run_ids.length})` : ""}
        </h4>
        {node.run_ids.length === 0 ? (
          <p className="muted">No runs in scope</p>
        ) : (
          <ol className="ndrawer-runs">
            {node.run_ids.map((id) => {
              const run = runsById.get(id);
              const status = run?.status ? (RUN_STATUS[run.status] ?? run.status) : null;
              return (
                <li key={id}>
                  <Link to={runHref(id)} title={id}>
                    <code>{shortRunId(id)}</code>
                  </Link>
                  {status && <span className={`run-status run-status-${run?.status ?? "unknown"}`}> {status}</span>}
                  {run?.started_at != null && <span className="muted"> · {dateTime(run.started_at)}</span>}
                </li>
              );
            })}
          </ol>
        )}
      </section>
    </aside>
  );
}
