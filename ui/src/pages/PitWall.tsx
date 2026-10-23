// The pit wall (home), direction B of the health-board mockups: the workflows that need you as large tiles, the
// healthy ones as small tiles, the global budgets in the header, and a live feed in the rail. The board polls
// every 10 s and the feed every 5 s (decision 11); a poll that fails keeps the last answer on screen and says so.
import { useEffect, useState } from "react";
import { getBoard, getFeed, type Board, type Tile as TileData } from "../api/client";
import { usePoll } from "../api/poll";
import { BudgetBar } from "../components/BudgetBar";
import { EmptyState } from "../components/EmptyState";
import { ErrorState } from "../components/ErrorState";
import { Feed } from "../components/Feed";
import { Tile } from "../components/Tile";
import { TileSmall } from "../components/TileSmall";
import { clock } from "../time";

const BOARD_EVERY_MS = 10_000;
const FEED_EVERY_MS = 5_000;
const FEED_LIMIT = 50;

const PERIOD: Record<string, string> = { day: "today", month: "this month", total: "all time" };

/** "2 workflows need you · 2 healthy", or "All 4 workflows healthy". */
export function countLine(tiles: TileData[]): string {
  const needs = tiles.filter((t) => t.flag !== "green").length;
  const healthy = tiles.length - needs;
  if (needs === 0) return tiles.length === 1 ? "1 workflow, healthy" : `All ${tiles.length} workflows healthy`;
  const line = `${needs} ${needs === 1 ? "workflow needs" : "workflows need"} you`;
  return healthy > 0 ? `${line} · ${healthy} healthy` : line;
}

/** When `data` last changed (ms), for the staleness line of a poll whose answer carries no time of its own. */
function useStampedAt(data: unknown): number | null {
  const [at, setAt] = useState<number | null>(null);
  useEffect(() => {
    if (data !== null) setAt(Date.now());
  }, [data]);
  return at;
}

function Stale({ what, sinceS }: { what: string; sinceS: number | null }) {
  const since = sinceS !== null ? ` — showing data from ${clock(sinceS)}` : "";
  return (
    <p className="stale" role="status" aria-label={`${what} not refreshed`}>
      Couldn't refresh{since}. Retrying.
    </p>
  );
}

function Budgets({ board }: { board: Board }) {
  if (board.budgets.length === 0) {
    const total = board.workflows.reduce((sum, t) => sum + t.spend_today_usd, 0);
    return (
      <div className="wall-budgets">
        <BudgetBar spent={total} limit={null} label="spend today" caption="spent today" noLimitNote="no global budget" />
      </div>
    );
  }
  return (
    <div className="wall-budgets">
      {board.budgets.map((b) => (
        <BudgetBar
          key={b.scope}
          spent={b.spent_usd}
          limit={b.limit_usd}
          label={`${b.name} budget, ${PERIOD[b.period] ?? b.period}`}
          caption={`${b.name}, ${PERIOD[b.period] ?? b.period}`}
        />
      ))}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="wall-skeleton" aria-busy="true" aria-label="Loading the pit wall">
      <div className="skeleton skeleton-title" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton-tiles">
        <div className="skeleton skeleton-tile" />
        <div className="skeleton skeleton-tile" />
      </div>
      <div className="skeleton-tiles skeleton-tiles-small">
        <div className="skeleton skeleton-tile-small" />
        <div className="skeleton skeleton-tile-small" />
        <div className="skeleton skeleton-tile-small" />
      </div>
    </div>
  );
}

function FeedSkeleton() {
  return (
    <div className="feed-skeleton" aria-busy="true" aria-label="Loading the feed">
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-line" />
    </div>
  );
}

export function PitWall() {
  const board = usePoll((signal) => getBoard({ signal }), BOARD_EVERY_MS);
  const feed = usePoll((signal) => getFeed(FEED_LIMIT, { signal }), FEED_EVERY_MS);
  const feedAt = useStampedAt(feed.data);

  if (board.loading) return <Skeleton />;
  if (!board.data) {
    const retry = () => {
      board.refresh();
      feed.refresh();
    };
    return <ErrorState error={board.error} title="The pit wall didn't load" onRetry={retry} />;
  }
  const tiles = board.data.workflows;
  if (tiles.length === 0) return <EmptyState />;

  const nowMs = Date.now();
  const big = tiles.filter((t) => t.flag !== "green");
  const small = tiles.filter((t) => t.flag === "green");

  return (
    <div className="pitwall">
      <section className="wall" aria-labelledby="wall-title">
        <header className="wall-head">
          <div>
            <h1 id="wall-title" className="page-title">
              Pit wall
            </h1>
            <p className="wall-count">{countLine(tiles)}</p>
            {board.error && <Stale what="Board" sinceS={board.data.now} />}
          </div>
          <Budgets board={board.data} />
        </header>

        {big.length > 0 && (
          <div className="wall-section">
            <h2 className="wall-section-title">Needs you</h2>
            <ol className="tiles tiles-big" aria-label="Workflows that need you">
              {big.map((tile) => (
                <li key={tile.name}>
                  <Tile tile={tile} nowMs={nowMs} />
                </li>
              ))}
            </ol>
          </div>
        )}

        {small.length > 0 && (
          <div className="wall-section">
            <h2 className="wall-section-title">Healthy</h2>
            <ol className="tiles tiles-small" aria-label="Healthy workflows">
              {small.map((tile) => (
                <li key={tile.name}>
                  <TileSmall tile={tile} />
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>

      <aside className="rail" aria-labelledby="feed-title">
        <h2 id="feed-title" className="rail-title">
          Live feed
        </h2>
        {feed.loading ? (
          <FeedSkeleton />
        ) : !feed.data ? (
          <ErrorState error={feed.error} title="The feed didn't load" onRetry={feed.refresh} />
        ) : (
          <>
            {feed.error && <Stale what="Feed" sinceS={feedAt !== null ? feedAt / 1000 : null} />}
            <Feed items={feed.data} nowMs={nowMs} />
          </>
        )}
      </aside>
    </div>
  );
}
