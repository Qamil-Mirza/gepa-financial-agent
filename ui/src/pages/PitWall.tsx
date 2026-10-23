// The pit wall (home). Task 12 builds the tiles and the live feed; for now it lists the workflows, or teaches the
// setup when there are none.
import { Link } from "react-router";
import { getBoard } from "../api/client";
import { usePoll } from "../api/poll";
import { EmptyState } from "../components/EmptyState";
import { ErrorState } from "../components/ErrorState";
import { Flag } from "../components/Flag";
import { workflowHref } from "../routes";

const BOARD_EVERY_MS = 10_000;

export function PitWall() {
  const board = usePoll((signal) => getBoard({ signal }), BOARD_EVERY_MS);
  if (board.loading) return <div className="skeleton skeleton-page" aria-busy="true" aria-label="Loading" />;
  if (!board.data) return <ErrorState error={board.error} onRetry={board.refresh} />;
  if (board.data.workflows.length === 0) return <EmptyState />;
  return (
    <section>
      <h1 className="page-title">Pit wall</h1>
      <ul className="plain-list">
        {board.data.workflows.map((tile) => (
          <li key={tile.name}>
            <Link to={workflowHref(tile.name)}>{tile.name}</Link> <Flag flag={tile.flag} label={tile.flag_label} />
          </li>
        ))}
      </ul>
    </section>
  );
}
