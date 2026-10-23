// Run detail at /run?id=…: where the run belongs, its facts, its own run graph (the workflow's DAG scoped to
// this run, without the scope controls) and its events in order. Polls every 10 s (decision 11) so a running
// run fills in; an unknown id is "not found", not an error.
import { lazy, Suspense, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router";
import { ApiError, getRun, type RunDetail as RunDetailData } from "../api/client";
import { usePoll } from "../api/poll";
import { ErrorState } from "../components/ErrorState";
import { EventTimeline } from "../components/EventTimeline";
import { Money } from "../components/Money";
import { statusWord } from "../components/RunsTable";
import { Stale, useStampedAt } from "../components/Stale";
import { plural } from "../format";
import { shortRunId } from "../graph/nodes";
import { PATHS, workflowHref } from "../routes";
import { dateTime, duration } from "../time";
import { MissingParam } from "./Workflow";

const RunGraph = lazy(() => import("../components/RunGraph"));

const EVERY_MS = 10_000;

export function RunDetail() {
  const [params] = useSearchParams();
  const id = params.get("id");
  if (!id) return <MissingParam what="run" />;
  return <RunPage key={id} id={id} />;
}

function NotFoundRun({ id }: { id: string }) {
  return (
    <section className="wf-notfound">
      <h1 className="page-title">Run not found</h1>
      <p>
        No run with the id <b className="wf-notfound-name">{id}</b>. It may have been recorded in another PitStop home, or not
        at all.
      </p>
      <p>
        <Link to={PATHS.pitWall}>Back to the pit wall</Link>
      </p>
    </section>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li className="fact">
      <span className="fact-label">{label}</span>
      <span className="fact-value">{children}</span>
    </li>
  );
}

function RunPage({ id }: { id: string }) {
  const poll = usePoll((signal) => getRun(id, { signal }), EVERY_MS, [id]);
  const stampedAt = useStampedAt(poll.data);

  if (poll.loading) {
    return (
      <div className="wf-skeleton" aria-busy="true">
        <p className="sr-only" role="status">
          Loading run {shortRunId(id)}…
        </p>
        <div className="skeleton skeleton-line" />
        <div className="skeleton skeleton-title" />
        <div className="skeleton skeleton-panel" />
      </div>
    );
  }
  if (!poll.data) {
    if (poll.error instanceof ApiError && poll.error.status === 404) return <NotFoundRun id={id} />;
    return <ErrorState error={poll.error} title="This run didn't load" onRetry={poll.refresh} />;
  }
  const detail: RunDetailData = poll.data;
  const run = detail.run;
  const status = statusWord(run.status);

  return (
    <div className="run">
      <header className="run-head">
        <nav className="run-crumb" aria-label="Breadcrumb">
          <Link to={PATHS.pitWall}>Pit wall</Link>
          <span aria-hidden="true"> › </span>
          {detail.workflow !== null ? (
            <Link to={workflowHref(detail.workflow)}>{detail.workflow}</Link>
          ) : (
            <span className="muted">no workflow</span>
          )}
        </nav>
        <h1 className="page-title run-title">
          Run <code>{shortRunId(run.id)}</code>
        </h1>
        <p className="run-line">
          <span className={`run-status run-status-${run.status ?? "unknown"}`}>{status}</span>
          {run.started_at !== null && <span> · started {dateTime(run.started_at)}</span>}
          <span> · {duration(run.duration_s)}</span>
        </p>
        {detail.goal && <p className="wf-goal">{detail.goal}</p>}
        {poll.error && <Stale what="Run" sinceS={stampedAt !== null ? stampedAt / 1000 : null} />}
      </header>

      <ul className="run-facts" aria-label="Run facts">
        <Fact label="Status">
          <span className={`run-status run-status-${run.status ?? "unknown"}`}>{status}</span>
        </Fact>
        <Fact label="Started">{run.started_at !== null ? dateTime(run.started_at) : "—"}</Fact>
        <Fact label="Ended">{run.ended_at !== null ? dateTime(run.ended_at) : run.status === "running" ? "still running" : "—"}</Fact>
        <Fact label="Duration">{duration(run.duration_s)}</Fact>
        <Fact label="Model">{run.model ?? "—"}</Fact>
        <Fact label="Endpoint">{run.base_url ?? "—"}</Fact>
        <Fact label="Adapter">{run.adapter ?? "—"}</Fact>
        <Fact label="Cost">
          <Money usd={run.cost_usd} />
        </Fact>
        <Fact label="Calls">
          {run.llm_calls} model · {run.tool_calls} tool
        </Fact>
        <Fact label="Verdicts">
          {run.verdicts > 0 ? <span className="vbadge vbadge-p1">{run.verdicts}</span> : <span className="muted">none</span>}
        </Fact>
        <Fact label="Run id">
          <code className="run-id">{run.id}</code>
        </Fact>
      </ul>

      <section className="run-section" aria-labelledby="run-graph-title">
        <h2 id="run-graph-title" className="run-section-title">
          Run graph
        </h2>
        {detail.workflow === null ? (
          <p className="muted">This run recorded no workflow name, so there is no graph to draw.</p>
        ) : (
          <Suspense
            fallback={
              <div className="graph-canvas graph-skeleton" aria-busy="true">
                <p className="sr-only" role="status">
                  Loading the run graph…
                </p>
              </div>
            }
          >
            <RunGraph name={detail.workflow} window="24h" runId={run.id} controls={false} />
          </Suspense>
        )}
      </section>

      <section className="run-section" aria-labelledby="run-events-title">
        <h2 id="run-events-title" className="run-section-title">
          Events <span className="run-section-count">{plural(detail.events.length, "event")}</span>
        </h2>
        <EventTimeline events={detail.events} startedAt={run.started_at} truncated={detail.truncated} />
      </section>
    </div>
  );
}
