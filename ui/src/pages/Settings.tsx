// The settings page, read-only in v1: workflows as their latest run recorded them (they are set in code), global
// budgets, the pre-action judge, alert channels and routes, the background worker and where data lives. Env vars
// appear by name with "set" / "not set" — the API never sends a value. Settings rarely change (config.toml is read
// when the server starts), so both polls run every 30 s, the same pace as the worker status in the top bar.
import type { ReactNode } from "react";
import { Link } from "react-router";
import { getHealth, getSettings, type Settings, type WorkerStatus } from "../api/client";
import { usePoll } from "../api/poll";
import { ErrorState } from "../components/ErrorState";
import { Level } from "../components/Level";
import { Money } from "../components/Money";
import { Stale, useStampedAt } from "../components/Stale";
import { plural } from "../format";
import { workflowHref } from "../routes";
import { dateTime, duration, timeAgo } from "../time";

const EVERY_MS = 30_000;

const PERIOD: Record<string, string> = { day: "today", month: "this month", total: "all time" };

const LEVEL_ORDER = ["P0", "P1", "P2", "P3", "digest"];

function ConfigNote() {
  return (
    <p className="set-source">
      Edit <code>~/.pitstop/config.toml</code>, see docs/config.example.toml
    </p>
  );
}

/** "set" / "not set" in words, with a filled or hollow mark that says the same (never the only cue). */
function EnvState({ set }: { set: boolean }) {
  return (
    <span className={`env ${set ? "env-set" : "env-unset"}`}>
      <span className="env-mark" aria-hidden="true" />
      {set ? "set" : "not set"}
    </span>
  );
}

/** An env var name that may break after each underscore on a narrow screen ("SMTP_|PASSWORD"), not mid-word. */
function EnvName({ name }: { name: string }) {
  const parts = name.split("_");
  return (
    <code>
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && (
            <>
              _<wbr />
            </>
          )}
        </span>
      ))}
    </code>
  );
}

/** A value the latest run didn't record. */
function Unrecorded() {
  return <span className="muted">not recorded</span>;
}

/** `<home>/<file>`, with the home's own separator. */
function inHome(home: string, file: string): string {
  const sep = home.includes("\\") && !home.includes("/") ? "\\" : "/";
  return home.endsWith(sep) ? `${home}${file}` : `${home}${sep}${file}`;
}

interface SectionProps {
  id: string;
  title: string;
  source: ReactNode;
  children: ReactNode;
}

function Section({ id, title, source, children }: SectionProps) {
  return (
    <section className="set-section" aria-labelledby={id}>
      <header className="set-aside">
        <h2 id={id} className="set-title">
          {title}
        </h2>
        {source}
      </header>
      <div className="set-body">{children}</div>
    </section>
  );
}

function Workflows({ rows }: { rows: Settings["workflows"] }) {
  const source = (
    <p className="set-source">
      Set in code: <code>pitstop.workflow(..., criticality=…)</code>
    </p>
  );
  return (
    <Section id="set-workflows" title="Workflows" source={source}>
      {rows.length === 0 ? (
        <p className="set-empty">No workflow has run yet. A workflow's settings show here after its first run.</p>
      ) : (
        <div className="runs-scroll">
          <table className="runs-table set-table" aria-labelledby="set-workflows">
            <thead>
              <tr>
                <th scope="col">Workflow</th>
                <th scope="col">Criticality</th>
                <th scope="col">Sensitivity</th>
                <th scope="col">Fix mode</th>
                <th scope="col" className="num">
                  Daily budget
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((w) => (
                <tr key={w.name}>
                  <th scope="row" className="set-name">
                    <Link to={workflowHref(w.name)}>{w.name}</Link>
                  </th>
                  <td>{w.criticality ?? <Unrecorded />}</td>
                  <td>{w.sensitivity ?? <Unrecorded />}</td>
                  <td>{w.fix_mode ?? <Unrecorded />}</td>
                  <td className="num">{w.budget_usd !== null ? <Money usd={w.budget_usd} /> : <span className="muted">none</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

function Budgets({ rows }: { rows: Settings["budgets"] }) {
  return (
    <Section id="set-budgets" title="Budgets" source={<ConfigNote />}>
      {rows.length === 0 ? (
        <p className="set-empty">
          No global budget. A <code>[budgets.&lt;name&gt;]</code> table caps spend across workflows, or on one endpoint.
        </p>
      ) : (
        <div className="runs-scroll">
          <table className="runs-table set-table" aria-labelledby="set-budgets">
            <thead>
              <tr>
                <th scope="col">Budget</th>
                <th scope="col">Period</th>
                <th scope="col" className="num">
                  Spent
                </th>
                <th scope="col" className="num">
                  Limit
                </th>
                <th scope="col" className="num">
                  Left
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((b) => (
                <tr key={b.scope}>
                  <th scope="row" className="set-name">
                    {b.name}
                  </th>
                  <td className="set-nowrap">{PERIOD[b.period] ?? b.period}</td>
                  <td className="num">
                    <Money usd={b.spent_usd} />
                  </td>
                  <td className="num">
                    <Money usd={b.limit_usd} />
                  </td>
                  <td className="num">
                    <Money usd={b.remaining_usd} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

function Judge({ judge }: { judge: Settings["judge"] }) {
  return (
    <Section id="set-judge" title="Pre-action judge" source={<ConfigNote />}>
      <p className="set-lede">
        Before a <code>side_effects=True</code> tool runs in a high or medium workflow, this model checks that the call
        serves the goal.
      </p>
      <dl className="set-facts">
        <div>
          <dt>Model</dt>
          <dd>
            <code>{judge.model}</code>
          </dd>
        </div>
        <div>
          <dt>Base URL</dt>
          <dd>
            <code>{judge.base_url || "—"}</code>
          </dd>
        </div>
        <div>
          <dt>API key</dt>
          <dd>
            <EnvName name={judge.api_key_env} /> <EnvState set={judge.key_set} />
          </dd>
        </div>
      </dl>
      {!judge.key_set && (
        <p className="set-note">
          Unless your code passes <code>pitstop.init(judge_client=...)</code>, gated calls have no judge and follow each
          workflow's <code>on_gate_error</code>: high workflows block them, medium ones run them.
        </p>
      )}
    </Section>
  );
}

function routeLabel(key: string): ReactNode {
  if (key === "digest") return "Digest (P2, P3)";
  return /^P[0-3]$/.test(key) ? <Level level={key} /> : key;
}

function Alerts({ alerts }: { alerts: Settings["alerts"] }) {
  const ready = new Set(alerts.channels.filter((c) => c.configured && c.env_set).map((c) => c.name));
  const unsent = alerts.channels.filter((c) => c.configured && !c.env_set);
  const routes = Object.entries(alerts.routes).sort(([a], [b]) => {
    const ia = LEVEL_ORDER.indexOf(a);
    const ib = LEVEL_ORDER.indexOf(b);
    return (ia < 0 ? LEVEL_ORDER.length : ia) - (ib < 0 ? LEVEL_ORDER.length : ib) || a.localeCompare(b);
  });
  return (
    <Section id="set-alerts" title="Alert channels" source={<ConfigNote />}>
      <p className="set-lede">A channel sends only when its table is in config.toml and its env var is set.</p>
      <div className="runs-scroll">
        <table className="runs-table set-table" aria-labelledby="set-alerts">
          <thead>
            <tr>
              <th scope="col">Channel</th>
              <th scope="col">Configured</th>
              <th scope="col">Env var</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {alerts.channels.map((c) => (
              <tr key={c.name}>
                <th scope="row" className="set-nowrap">
                  {c.name}
                </th>
                <td>{c.configured ? "yes" : <span className="muted">no</span>}</td>
                <td className="set-env">
                  <EnvName name={c.env_var} />
                </td>
                <td>
                  <EnvState set={c.env_set} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {unsent.map((c) => (
        <p key={c.name} className="set-warn">
          {c.name} is configured but <EnvName name={c.env_var} /> is not set, so it doesn't send.
        </p>
      ))}

      <h3 className="set-subtitle">Routes</h3>
      <ul className="set-routes" aria-label="Alert routes">
        {routes.map(([key, names]) => {
          const live = names.filter((n) => ready.has(n));
          return (
            <li key={key} className="route">
              <span className="route-level">{routeLabel(key)}</span>
              <span className="route-channels">{names.length > 0 ? names.join(", ") : <span className="muted">no channels</span>}</span>
              <span className={live.length > 0 ? "route-live" : "route-live route-none"}>
                {live.length > 0 ? `Sends to ${live.join(", ")}` : "No channel here is ready"}
              </span>
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

function workerState(w: WorkerStatus): string {
  if (!w.enabled) return "Off";
  if (w.last_error) return "Error";
  return w.running ? "Running" : "Stopped";
}

function Worker({ worker }: { worker: WorkerStatus }) {
  const source = (
    <p className="set-source">
      Set when the server starts: <code>pitstop ui --scan-every SECONDS</code>, or <code>--no-worker</code> to turn it off
    </p>
  );
  const lastScan =
    worker.last_scan_at != null
      ? `${dateTime(worker.last_scan_at)} (${timeAgo(worker.last_scan_at)})${worker.last_scan_runs != null ? `, ${plural(worker.last_scan_runs, "run")}` : ""}`
      : null;
  return (
    <Section id="set-worker" title="Background worker" source={source}>
      <dl className="set-facts">
        <div>
          <dt>Status</dt>
          <dd>{workerState(worker)}</dd>
        </div>
        <div>
          <dt>Scans every</dt>
          <dd>{duration(worker.scan_every_s)}</dd>
        </div>
        <div>
          <dt>Last scan</dt>
          <dd>{lastScan ?? <span className="muted">none yet</span>}</dd>
        </div>
        <div>
          <dt>Last error</dt>
          <dd>{worker.last_error ? <code className="set-error">{worker.last_error}</code> : <span className="muted">none</span>}</dd>
        </div>
      </dl>
      {!worker.enabled && (
        <p className="set-note">
          Started with <code>--no-worker</code>: new runs are scanned only when you run <code>pitstop scan</code>.
        </p>
      )}
    </Section>
  );
}

function Home({ home, version }: { home: string; version: string | null }) {
  const source = (
    <p className="set-source">
      Set with the <code>PITSTOP_HOME</code> env var (default <code>~/.pitstop</code>)
    </p>
  );
  return (
    <Section id="set-home" title="Where data lives" source={source}>
      <dl className="set-facts set-facts-wide">
        <div>
          <dt>PITSTOP_HOME</dt>
          <dd>
            <code>{home}</code>
          </dd>
        </div>
        <div>
          <dt>Store</dt>
          <dd>
            <code>{inHome(home, "pitstop.db")}</code>
          </dd>
        </div>
        <div>
          <dt>Config</dt>
          <dd>
            <code>{inHome(home, "config.toml")}</code>
          </dd>
        </div>
        {version && (
          <div>
            <dt>Version</dt>
            <dd>{version}</dd>
          </div>
        )}
      </dl>
    </Section>
  );
}

export function SettingsPage() {
  const settings = usePoll((signal) => getSettings({ signal }), EVERY_MS);
  const health = usePoll((signal) => getHealth({ signal }), EVERY_MS);
  const settingsAt = useStampedAt(settings.data);

  if (settings.loading) {
    return (
      <section className="settings">
        <h1 className="page-title">Settings</h1>
        <p className="set-loading" role="status">
          Loading settings…
        </p>
      </section>
    );
  }
  if (!settings.data) {
    const retry = () => {
      settings.refresh();
      health.refresh();
    };
    return (
      <section className="settings">
        <h1 className="page-title">Settings</h1>
        <ErrorState error={settings.error} title="Settings didn't load" onRetry={retry} />
      </section>
    );
  }
  const s = settings.data;
  const h = health.data;
  const version = h ? `PitStop ${h.version}, store schema ${h.schema_version}` : null;

  return (
    <section className="settings" aria-labelledby="settings-title">
      <header className="settings-head">
        <h1 id="settings-title" className="page-title">
          Settings
        </h1>
        <p className="settings-lede">
          Read-only. Workflows are set in code; the rest comes from <code>config.toml</code> and env vars, read when{" "}
          <code>pitstop ui</code> starts, so restart it after a change.
        </p>
        {settings.error && <Stale what="Settings" sinceS={settingsAt !== null ? settingsAt / 1000 : null} />}
      </header>
      <Workflows rows={s.workflows} />
      <Budgets rows={s.budgets} />
      <Judge judge={s.judge} />
      <Alerts alerts={s.alerts} />
      <Worker worker={h?.worker ?? s.worker} />
      <Home home={s.home} version={version} />
    </section>
  );
}
