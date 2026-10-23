// The workflow page at /w?name=…&tab=…: the tile facts in the header, three tabs (the run graph by default,
// with its scope in the URL too: window= and run=; What changed as small multiples with change markers; Runs as
// a table with a cursor), and the open drifts in a right column that stays on every tab. The page polls
// /api/workflow every 10 s (decision 11); each tab polls its own data.
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams, type SetURLSearchParams } from "react-router";
import {
  ApiError,
  dismissDrift,
  getChanges,
  getRuns,
  getWorkflow,
  type Changes,
  type GraphWindow,
  type RunPage,
  type RunRow,
  type WorkflowDetail,
} from "../api/client";
import { usePoll } from "../api/poll";
import { BudgetBar } from "../components/BudgetBar";
import { DriftCard } from "../components/DriftCard";
import { ErrorState } from "../components/ErrorState";
import { Flag } from "../components/Flag";
import { MarkerLegend } from "../components/MarkerLegend";
import { RunsTable, type MoreState } from "../components/RunsTable";
import { SeriesChart, SeriesTable } from "../components/SeriesChart";
import { Stale, useStampedAt } from "../components/Stale";
import { Criticality, lastRunText } from "../components/Tile";
import { parseWindow } from "../components/WindowPicker";
import { plural } from "../format";
import { PATHS, WORKFLOW_TABS, workflowTab, type WorkflowTab } from "../routes";

const RunGraph = lazy(() => import("../components/RunGraph"));

const EVERY_MS = 10_000;

const TAB_LABEL: Record<WorkflowTab, string> = { graph: "Run graph", changes: "What changed", runs: "Runs" };

export function Workflow() {
  const [params, setParams] = useSearchParams();
  const name = params.get("name");
  if (!name) return <MissingParam what="workflow" />;
  return <WorkflowPage key={name} name={name} params={params} setParams={setParams} />;
}

export function MissingParam({ what }: { what: string }) {
  return (
    <section>
      <h1 className="page-title">No {what} given</h1>
      <p>
        This link is missing its {what}. <Link to={PATHS.pitWall}>Back to the pit wall</Link>
      </p>
    </section>
  );
}

function NotFoundWorkflow({ name }: { name: string }) {
  return (
    <section className="wf-notfound">
      <h1 className="page-title">Workflow not found</h1>
      <p>
        No run has recorded a workflow named <b className="wf-notfound-name">{name}</b>. It may have been renamed, or its
        runs may not have arrived yet.
      </p>
      <p>
        <Link to={PATHS.pitWall}>Back to the pit wall</Link>
      </p>
    </section>
  );
}

function Skeleton({ name }: { name: string }) {
  return (
    <div className="wf-skeleton" aria-busy="true">
      <p className="sr-only" role="status">
        Loading {name}…
      </p>
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-panel" />
    </div>
  );
}

function TabSkeleton({ what }: { what: string }) {
  return (
    <div className="graph-canvas graph-skeleton" aria-busy="true">
      <p className="sr-only" role="status">
        Loading {what}…
      </p>
    </div>
  );
}

// ---------- What changed ----------

function ChangesTab({ name }: { name: string }) {
  const poll = usePoll((signal) => getChanges(name, { signal }), EVERY_MS, [name]);
  const stampedAt = useStampedAt(poll.data);
  if (poll.loading) return <TabSkeleton what="what changed" />;
  if (!poll.data) return <ErrorState error={poll.error} title="What changed didn't load" onRetry={poll.refresh} />;
  const data: Changes = poll.data;
  const charted = data.metrics.filter((m) => data.points.some((p) => Number.isFinite(p.metrics[m.name])));

  if (data.points.length === 0) {
    return (
      <section className="changes" aria-label="What changed">
        <div className="tab-placeholder">
          <p className="tab-placeholder-title">No scored runs yet.</p>
          <p className="muted">
            Each run's tool errors, refusals, tokens per call, calls, cost, duration and output length land here once the
            background worker (or <code>pitstop scan</code>) has scored it.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="changes" aria-label="What changed">
      {poll.error && <Stale what="What changed" sinceS={stampedAt !== null ? stampedAt / 1000 : null} />}
      <p className="changes-key">
        {plural(data.points.length, "scored run")}, one per slot, oldest to newest. A filled point finished ok; a hollow one
        halted, was refused or errored. A dashed line is a change, drawn before the first run that carried it.
      </p>
      <div className="changes-charts">
        {charted.map((m, i) => (
          <SeriesChart
            key={m.name}
            metric={m}
            points={data.points}
            markers={data.markers}
            metrics={data.metrics}
            labelMarkers={i === 0}
            xAxis={i === charted.length - 1}
          />
        ))}
      </div>
      <p className="changes-note muted">Quality score: coming with the run judge.</p>
      <h3 className="changes-title">Changes</h3>
      <MarkerLegend markers={data.markers} metrics={data.metrics} />
      <details className="changes-table">
        <summary>Table view</summary>
        <SeriesTable changes={data} />
      </details>
    </section>
  );
}

// ---------- Runs ----------

/** `incoming` merged into `kept`: a run already there is updated in place, a new one is added, none is dropped;
 *  newest first (the server's order, by `started_at`), ties keeping their place. */
export function mergeRuns(kept: RunRow[], incoming: RunRow[]): RunRow[] {
  const byId = new Map(kept.map((r) => [r.id, r]));
  for (const r of incoming) byId.set(r.id, r);
  return [...byId.values()].sort((a, b) => (b.started_at ?? 0) - (a.started_at ?? 0));
}

interface KeptRuns {
  /** The page-1 answer these rows last merged, so each poll answer is merged once. */
  page: RunPage | null;
  rows: RunRow[];
}

function RunsTab({ name }: { name: string }) {
  const first = usePoll((signal) => getRuns(name, null, { signal }), EVERY_MS, [name]);
  const stampedAt = useStampedAt(first.data);
  // Every run seen so far. Page 1 arrives again on each poll and is merged, never swapped in: new runs go on top,
  // known ones refresh in place, and a run that new arrivals pushed off page 1 stays (it would otherwise be in no
  // page until the next "Load more", which starts after the cursor).
  const [kept, setKept] = useState<KeptRuns>({ page: null, rows: [] });
  if (first.data && kept.page !== first.data) setKept({ page: first.data, rows: mergeRuns(kept.rows, first.data.runs) });
  // the cursor: page 1's until a later page was loaded, then the last loaded page's
  const [loaded, setLoaded] = useState<{ nextBefore: number | null } | null>(null);
  const [moreState, setMoreState] = useState<MoreState>({ kind: "idle" });
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const cursor = loaded ? loaded.nextBefore : (first.data?.next_before ?? null);
  const loadMore = async () => {
    if (cursor === null || moreState.kind === "loading") return;
    setMoreState({ kind: "loading" });
    try {
      const page = await getRuns(name, cursor);
      if (!alive.current) return;
      setKept((k) => ({ ...k, rows: mergeRuns(k.rows, page.runs) }));
      setLoaded({ nextBefore: page.next_before });
      setMoreState({ kind: "idle" });
    } catch (e) {
      if (!alive.current) return;
      setMoreState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  if (first.loading) return <TabSkeleton what="runs" />;
  if (!first.data) return <ErrorState error={first.error} title="The runs didn't load" onRetry={first.refresh} />;

  const rows = kept.page === first.data ? kept.rows : mergeRuns(kept.rows, first.data.runs);

  if (rows.length === 0) {
    return (
      <div className="tab-placeholder">
        <p className="tab-placeholder-title">No runs recorded yet.</p>
        <p className="muted">Runs appear here as soon as the workflow records one.</p>
      </div>
    );
  }
  return (
    <div className="runs-tab">
      {first.error && <Stale what="Runs" sinceS={stampedAt !== null ? stampedAt / 1000 : null} />}
      <p className="runs-count">{plural(rows.length, "run")}, newest first</p>
      <RunsTable runs={rows} hasMore={cursor !== null} moreState={moreState} onLoadMore={() => void loadMore()} />
    </div>
  );
}

interface PageProps {
  name: string;
  params: URLSearchParams;
  setParams: SetURLSearchParams;
}

function WorkflowPage({ name, params, setParams }: PageProps) {
  const wf = usePoll((signal) => getWorkflow(name, { signal }), EVERY_MS, [name]);
  const stampedAt = useStampedAt(wf.data);
  const tab = workflowTab(params.get("tab"));
  const win = parseWindow(params.get("window"));
  const runId = params.get("run");

  // Dismiss is optimistic: the card goes at once and comes back, with the reason, if the server says no.
  const [hidden, setHidden] = useState<ReadonlySet<number>>(new Set());
  const [dismissErrors, setDismissErrors] = useState<ReadonlyMap<number, string>>(new Map());
  const refresh = wf.refresh;
  const dismiss = useCallback(
    async (id: number) => {
      setHidden((s) => new Set(s).add(id));
      setDismissErrors((m) => {
        const next = new Map(m);
        next.delete(id);
        return next;
      });
      try {
        await dismissDrift(id);
        refresh(); // the header's flag may have changed
      } catch (e) {
        if (e instanceof ApiError && (e.status === 404 || e.status === 409)) {
          refresh(); // already gone or no longer open: the next answer no longer lists it
          return;
        }
        setHidden((s) => {
          const next = new Set(s);
          next.delete(id);
          return next;
        });
        setDismissErrors((m) => new Map(m).set(id, e instanceof Error ? e.message : String(e)));
      }
    },
    [refresh],
  );

  const tabHref = (t: WorkflowTab) => {
    const next = new URLSearchParams(params);
    next.set("tab", t);
    return `${PATHS.workflow}?${next}`;
  };
  const setScope = (change: { window?: GraphWindow; run?: string | null }) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (change.window !== undefined) next.set("window", change.window);
        if (change.run !== undefined) {
          if (change.run) next.set("run", change.run);
          else next.delete("run");
        }
        return next;
      },
      { replace: true },
    );
  };

  const drifts = useMemo(() => (wf.data?.drifts ?? []).filter((d) => !hidden.has(d.id)), [wf.data, hidden]);

  if (wf.loading) return <Skeleton name={name} />;
  if (!wf.data) {
    if (wf.error instanceof ApiError && wf.error.status === 404) return <NotFoundWorkflow name={name} />;
    return <ErrorState error={wf.error} title="This workflow didn't load" onRetry={wf.refresh} />;
  }
  const detail: WorkflowDetail = wf.data;
  const tile = detail.tile;
  const nowMs = Date.now();

  return (
    <div className="wf">
      <header className="wf-head">
        <div className="wf-id">
          <p className="wf-flagline">
            <Flag flag={tile.flag} label={tile.flag_label} />
            {tile.fallback_active && <span className="tile-fallback">Fallback active</span>}
          </p>
          <h1 className="page-title wf-name">{name}</h1>
          {detail.goal && <p className="wf-goal">{detail.goal}</p>}
          <p className="wf-meta">
            <Criticality value={tile.criticality} />
            <span className="wf-meta-sep" aria-hidden="true">
              ·
            </span>
            <span>
              Sensitivity <b>{detail.sensitivity ?? "not set"}</b>
            </span>
            <span className="wf-meta-sep" aria-hidden="true">
              ·
            </span>
            <span>
              Fixes <b>{detail.fix_mode ?? "not set"}</b>
            </span>
          </p>
        </div>
        <div className="wf-money">
          <BudgetBar
            spent={tile.spend_today_usd}
            limit={tile.budget_usd}
            label="spend today"
            caption="today"
            noLimitNote="no daily budget"
          />
          <p className="wf-run">{lastRunText(tile, nowMs)}</p>
        </div>
        {wf.error && <Stale what="Workflow" sinceS={stampedAt !== null ? stampedAt / 1000 : null} />}
      </header>

      <nav className="tabs" aria-label="Workflow views">
        {WORKFLOW_TABS.map((t) => (
          <Link key={t} to={tabHref(t)} className={t === tab ? "tab tab-on" : "tab"} aria-current={t === tab ? "page" : undefined}>
            {TAB_LABEL[t]}
          </Link>
        ))}
      </nav>

      <div className="wf-split">
        <div className="wf-main">
          {tab === "graph" && (
            <Suspense
              fallback={
                <div className="graph-canvas graph-skeleton" aria-busy="true">
                  <p className="sr-only" role="status">
                    Loading the run graph…
                  </p>
                </div>
              }
            >
              <RunGraph
                name={name}
                window={win}
                runId={runId}
                onWindow={(window) => setScope({ window })}
                onRun={(run) => setScope({ run })}
              />
            </Suspense>
          )}
          {tab === "changes" && <ChangesTab name={name} />}
          {tab === "runs" && <RunsTab name={name} />}
        </div>

        <aside className="wf-side" aria-labelledby="drifts-title">
          <h2 id="drifts-title" className="wf-side-title">
            Open drifts{drifts.length > 0 && <span className="wf-side-count">{drifts.length}</span>}
          </h2>
          {drifts.length === 0 ? (
            <p className="wf-side-empty">
              Nothing open. {plural(tile.runs_today, "run")} today; new drifts land here as runs are scanned.
            </p>
          ) : (
            <ol className="drifts" aria-label="Open drifts">
              {drifts.map((d) => (
                <li key={d.id}>
                  <DriftCard drift={d} workflow={name} nowMs={nowMs} onDismiss={dismiss} dismissError={dismissErrors.get(d.id)} />
                </li>
              ))}
            </ol>
          )}
        </aside>
      </div>
    </div>
  );
}
