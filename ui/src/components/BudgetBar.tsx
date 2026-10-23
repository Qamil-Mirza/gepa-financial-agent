// Spend against a limit: the amounts and the share in text, then a thin bar that says the same (the bar is never
// the only cue). Past 80% the bar turns warm, past 100% red. Without a limit it is just the amount and a note.
import { formatUsd, Money } from "./Money";

export interface BudgetBarProps {
  spent: number;
  limit: number | null | undefined;
  /** The meter's accessible name, e.g. "prod budget, today" or "spend today". */
  label: string;
  /** After the amount when there is no limit, e.g. "no daily budget". */
  noLimitNote?: string;
  /** After the amount: "today", "this month"… */
  caption?: string;
}

export function BudgetBar({ spent, limit, label, noLimitNote, caption }: BudgetBarProps) {
  if (limit === null || limit === undefined || !(limit > 0)) {
    return (
      <p className="meter meter-open">
        <span className="meter-amount">
          <Money usd={spent} />
          {caption && <span className="meter-cap"> {caption}</span>}
        </span>
        {noLimitNote && <span className="meter-cap">{noLimitNote}</span>}
      </p>
    );
  }
  const share = spent / limit;
  const pct = Math.round(share * 100);
  const tone = share >= 1 ? " meter-over" : share >= 0.8 ? " meter-hot" : "";
  return (
    <div className={`meter${tone}`}>
      <span className="meter-amount">
        <b>{formatUsd(spent)}</b>
        <span className="meter-cap"> of {formatUsd(limit)}</span>
        {caption && <span className="meter-cap"> {caption}</span>}
      </span>
      <span className="meter-pct">{pct}%</span>
      <div
        className="meter-track"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-valuenow={spent}
        aria-valuetext={`${formatUsd(spent)} of ${formatUsd(limit)}, ${pct}%`}
      >
        <div className="meter-fill" style={{ transform: `scaleX(${Math.min(share, 1)})` }} />
      </div>
    </div>
  );
}
