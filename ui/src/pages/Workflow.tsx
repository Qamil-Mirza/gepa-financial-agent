// The workflow page at /w?name=…&tab=…: the tile facts in the header, three tabs (the run graph by default,
// with its scope in the URL too: window= and run=), and the open drifts in a right column that stays on every
// tab. The page polls /api/workflow every 10 s (decision 11); the graph tab polls its own data. What changed and
// Runs are Task 14.
import { lazy, Suspense, useCallback, useMemo, useState } from "react";
import { Link, useSearchParams, type SetURLSearchParams } from "react-router";
import { ApiError, dismissDrift, getWorkflow, type GraphWindow, type WorkflowDetail } from "../api/client";
import { usePoll } from "../api/poll";
import { BudgetBar } from "../components/BudgetBar";
import { DriftCard } from "../components/DriftCard";
import { ErrorState } from "../components/ErrorState";
import { Flag } from "../components/Flag";
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

function Placeholder({ title, text }: { title: string; text: string }) {
  return (
    <div className="tab-placeholder">
      <p className="tab-placeholder-title">{title}</p>
      <p className="muted">{text}</p>
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
          {tab === "changes" && (
            <Placeholder
              title="What changed is on its way."
              text="Quality, errors and cost over time, with model, endpoint and prompt changes drawn as markers."
            />
          )}
          {tab === "runs" && (
            <Placeholder title="Runs are on their way." text="Every run with its status, duration, model, cost and verdicts." />
          )}
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
