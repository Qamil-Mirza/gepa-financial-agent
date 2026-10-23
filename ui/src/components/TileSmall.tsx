// A small pit-wall tile: a healthy workflow, kept quiet. Name, criticality, runs today and spend.
import { Link } from "react-router";
import type { Tile as TileData } from "../api/client";
import { workflowHref } from "../routes";
import { Flag } from "./Flag";
import { Money } from "./Money";
import { Criticality } from "./Tile";

export function TileSmall({ tile }: { tile: TileData }) {
  return (
    <article className={`tile tile-small tile-${tile.flag}`}>
      <div className="tile-head">
        <Flag flag={tile.flag} label={tile.flag_label} />
        {tile.fallback_active && <span className="tile-fallback">Fallback active</span>}
      </div>
      <h3 className="tile-name">
        <Link to={workflowHref(tile.name)}>{tile.name}</Link>
      </h3>
      <Criticality value={tile.criticality} />
      <p className="tile-stats">
        <span>
          {tile.runs_today} {tile.runs_today === 1 ? "run" : "runs"} today
        </span>
        <span>
          <Money usd={tile.spend_today_usd} />
          {tile.budget_usd !== null && tile.budget_usd > 0 && (
            <span className="meter-cap">
              {" "}
              of <Money usd={tile.budget_usd} />
            </span>
          )}
        </span>
      </p>
    </article>
  );
}
