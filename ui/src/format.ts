// Plain numbers and words as the dashboard shows them.

const formatters = new Map<number, Intl.NumberFormat>();

function formatter(decimals: number): Intl.NumberFormat {
  let f = formatters.get(decimals);
  if (!f) {
    f = new Intl.NumberFormat("en-US", { maximumFractionDigits: decimals });
    formatters.set(decimals, f);
  }
  return f;
}

/** A metric value with the precision its size needs: "2,694", "41.3", "0.86", "0.004"; unknown is "—". */
export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  const decimals = abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  return formatter(decimals).format(n);
}

/** A share as a whole percentage: 0.125 → "13%". */
export function formatPercent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

/** A snake_case identifier as words: "tokens_per_call" → "Tokens per call". */
export function words(identifier: string): string {
  const text = identifier.replaceAll("_", " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** "1 run", "2 runs". */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}
