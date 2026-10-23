// The app frame: wordmark, nav, the background worker's status (read from /api/health every 30 s) and the theme
// toggle, around the routed page.
import { Link, NavLink, Outlet } from "react-router";
import { ApiError, getHealth, type Health } from "../api/client";
import { usePoll, type Poll } from "../api/poll";
import { PATHS } from "../routes";
import { timeAgo } from "../time";
import { FlagGlyph } from "./Flag";
import { ThemeToggle } from "./ThemeToggle";

const HEALTH_EVERY_MS = 30_000;

type Tone = "ok" | "warn" | "off" | "unknown";

export function workerStatus(poll: Pick<Poll<Health>, "data" | "error">): { tone: Tone; label: string; detail?: string } {
  if (poll.error) {
    const unreachable = poll.error instanceof ApiError && poll.error.status === 0;
    return { tone: "warn", label: unreachable ? "Server unreachable" : "Server error", detail: poll.error.message };
  }
  const worker = poll.data?.worker;
  if (!worker) return { tone: "unknown", label: "Checking…" };
  const lastScan = worker.last_scan_at != null ? `Last scan ${timeAgo(worker.last_scan_at)}` : "No scan yet";
  if (!worker.enabled) return { tone: "off", label: "Worker off", detail: "Started with --no-worker: scans run only when you run pitstop scan" };
  if (worker.last_error) return { tone: "warn", label: "Worker error", detail: `${worker.last_error}. ${lastScan}` };
  if (!worker.running) return { tone: "warn", label: "Worker stopped", detail: lastScan };
  return { tone: "ok", label: "Worker running", detail: lastScan };
}

function WorkerDot({ poll }: { poll: Poll<Health> }) {
  const { tone, label, detail } = workerStatus(poll);
  return (
    <span className={`worker worker-${tone}`} title={detail}>
      <span className="worker-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function Shell() {
  const health = usePoll((signal) => getHealth({ signal }), HEALTH_EVERY_MS);
  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <Link to={PATHS.pitWall} className="brand">
            <FlagGlyph />
            PitStop
          </Link>
          <nav className="nav" aria-label="Main">
            <NavLink to={PATHS.pitWall} end>
              Pit wall
            </NavLink>
            <NavLink to={PATHS.settings}>Settings</NavLink>
          </nav>
          <span className="spacer" />
          <WorkerDot poll={health} />
          <ThemeToggle />
        </div>
      </header>
      <main id="main" className="page" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
