// A large pit-wall tile: a workflow with a red or yellow flag. The flag in words, the name (the link; the whole
// tile is its hit area), the open drifts by level, the headline in plain words, the last run, and today's spend
// against the daily budget.
import { Link } from "react-router";
import type { Tile as TileData } from "../api/client";
import { workflowHref } from "../routes";
import { timeAgo } from "../time";
import { BudgetBar } from "./BudgetBar";
import { Flag } from "./Flag";
import { Level } from "./Level";

const LEVELS = ["P0", "P1", "P2", "P3"] as const;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The headline without the "[P0] name: " the tile already shows, a bare drift type spelled out as words. */
export function plainHeadline(tile: Pick<TileData, "headline" | "name">): string | null {
  if (!tile.headline) return null;
  const rest = tile.headline.replace(new RegExp(`^\\[[^\\]]*\\]\\s*${escapeRegExp(tile.name)}:\\s*`), "").trim();
  if (!rest) return tile.headline;
  if (/^[a-z0-9]+(?:_[a-z0-9]+)+$/.test(rest)) {
    const words = rest.replaceAll("_", " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  }
  return rest;
}

const RUN_STATUS: Record<string, string> = {
  ok: "finished ok",
  halted: "halted",
  refused: "refused",
  error: "errored",
  dropped: "dropped",
  running: "still running",
};

export function lastRunText(tile: Pick<TileData, "last_run" | "runs_today">, nowMs: number): string {
  const run = tile.last_run;
  const today = `${tile.runs_today} ${tile.runs_today === 1 ? "run" : "runs"} today`;
  if (!run) return `No run recorded · ${today}`;
  const status = run.status ? (RUN_STATUS[run.status] ?? run.status) : "status unknown";
  const at = run.ended_at ?? run.started_at;
  const when = at !== null && at !== undefined ? ` · ${timeAgo(at, nowMs)}` : "";
  return `Last run ${status}${when} · ${today}`;
}

export function Criticality({ value }: { value: string | null }) {
  return (
    <span className="tile-crit">
      Criticality <b>{value ?? "not set"}</b>
    </span>
  );
}

export function Tile({ tile, nowMs }: { tile: TileData; nowMs: number }) {
  const headline = plainHeadline(tile);
  const open = LEVELS.filter((level) => tile.open[level] > 0);
  return (
    <article className={`tile tile-big tile-${tile.flag}`}>
      <div className="tile-head">
        <Flag flag={tile.flag} label={tile.flag_label} />
        {tile.fallback_active && <span className="tile-fallback">Fallback active</span>}
      </div>
      <h3 className="tile-name">
        <Link to={workflowHref(tile.name)}>{tile.name}</Link>
      </h3>
      <Criticality value={tile.criticality} />
      {open.length > 0 && (
        <p className="tile-levels" aria-label="Open drifts">
          {open.map((level) => (
            <Level key={level} level={level} count={tile.open[level]} />
          ))}
        </p>
      )}
      {headline && <p className="tile-headline">{headline}</p>}
      <p className="tile-run">{lastRunText(tile, nowMs)}</p>
      <BudgetBar
        spent={tile.spend_today_usd}
        limit={tile.budget_usd}
        label="spend today"
        caption="today"
        noLimitNote="no daily budget"
      />
    </article>
  );
}
