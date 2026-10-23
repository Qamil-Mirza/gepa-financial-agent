// The Runs tab's table: one row per run, newest first, with its status in words, when it started, how long it
// took, the model and adapter, what it cost, its calls and its verdicts. Each row opens the run (the id is the
// link; the whole row is the hit area). "Load more" follows the API's cursor; the page keeps what it has when a
// later page fails and says why.
import { Link, useNavigate } from "react-router";
import type { RunRow } from "../api/client";
import { plural } from "../format";
import { shortRunId } from "../graph/nodes";
import { runHref } from "../routes";
import { dateTime, duration } from "../time";
import { Money } from "./Money";

export type MoreState = { kind: "idle" } | { kind: "loading" } | { kind: "error"; message: string };

export interface RunsTableProps {
  runs: RunRow[];
  hasMore: boolean;
  moreState: MoreState;
  onLoadMore: () => void;
}

const RUN_STATUS: Record<string, string> = { ok: "ok", halted: "halted", refused: "refused", error: "errored", dropped: "dropped", running: "running" };

export function statusWord(status: string | null): string {
  return status ? (RUN_STATUS[status] ?? status) : "unknown";
}

function Row({ run }: { run: RunRow }) {
  const navigate = useNavigate();
  const href = runHref(run.id);
  return (
    <tr
      className="runs-row"
      onClick={(e) => {
        // the link handles its own click; a drag to select text is not a click
        if ((e.target as Element).closest("a, button") || window.getSelection()?.toString()) return;
        navigate(href);
      }}
    >
      <td>
        <Link to={href} title={run.id}>
          <code>{shortRunId(run.id)}</code>
        </Link>
      </td>
      <td className={`run-status run-status-${run.status ?? "unknown"}`}>{statusWord(run.status)}</td>
      <td className="runs-when">{run.started_at !== null ? dateTime(run.started_at) : "—"}</td>
      <td className="num">{duration(run.duration_s)}</td>
      <td className="runs-model" title={run.base_url ?? undefined}>
        {run.model ?? "—"}
      </td>
      <td>{run.adapter ?? "—"}</td>
      <td className="num">
        <Money usd={run.cost_usd} />
      </td>
      <td className="num">{run.llm_calls}</td>
      <td className="num">{run.tool_calls}</td>
      <td>{run.verdicts > 0 ? <span className="vbadge vbadge-p1">{plural(run.verdicts, "verdict")}</span> : <span className="muted">none</span>}</td>
    </tr>
  );
}

export function RunsTable({ runs, hasMore, moreState, onLoadMore }: RunsTableProps) {
  return (
    <div className="runs">
      <div className="runs-scroll">
        <table className="runs-table" aria-label="Runs">
          <thead>
            <tr>
              <th scope="col">Run</th>
              <th scope="col">Status</th>
              <th scope="col">Started</th>
              <th scope="col" className="num">
                Duration
              </th>
              <th scope="col">Model</th>
              <th scope="col">Adapter</th>
              <th scope="col" className="num">
                Cost
              </th>
              <th scope="col" className="num">
                LLM calls
              </th>
              <th scope="col" className="num">
                Tool calls
              </th>
              <th scope="col">Verdicts</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <Row key={r.id} run={r} />
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <div className="runs-more">
          {moreState.kind === "error" && (
            <p className="runs-more-error" role="alert">
              Couldn't load more: {moreState.message}
            </p>
          )}
          <button type="button" className="btn" disabled={moreState.kind === "loading"} onClick={onLoadMore}>
            {moreState.kind === "loading" ? "Loading…" : moreState.kind === "error" ? "Try again" : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}
