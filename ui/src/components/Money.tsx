// USD amounts: 2 decimals at $1 or more, 4 below (agent spend is often fractions of a cent), "—" when unknown.
const formatters = new Map<number, Intl.NumberFormat>();

function formatter(decimals: number): Intl.NumberFormat {
  let f = formatters.get(decimals);
  if (!f) {
    f = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    formatters.set(decimals, f);
  }
  return f;
}

export function formatUsd(usd: number | null | undefined): string {
  if (usd === null || usd === undefined || !Number.isFinite(usd)) return "—";
  // decide on the rounded value, so $0.99999 shows as "$1.00", not "$1.0000"
  const decimals = Math.abs(Number(usd.toFixed(4))) >= 1 ? 2 : 4;
  return formatter(decimals).format(usd);
}

export function Money({ usd, className }: { usd: number | null | undefined; className?: string }) {
  const text = formatUsd(usd);
  const cls = className ? `money ${className}` : "money";
  if (text === "—") {
    return (
      <span className={`${cls} money-unknown`} title="unknown">
        <span aria-hidden="true">—</span>
        <span className="sr-only">unknown</span>
      </span>
    );
  }
  return <span className={cls}>{text}</span>;
}
