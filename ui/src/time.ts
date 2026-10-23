// Time as the dashboard shows it. Everything the API sends is epoch seconds.

/** "5s ago", "3 min ago", "2 h ago", "4 d ago"; a time in the future reads as "0s ago". */
export function timeAgo(epochS: number, nowMs = Date.now()): string {
  const s = Math.max(0, Math.round(nowMs / 1000 - epochS));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

const clockFormat = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });

/** The local wall-clock time, "12:04" (or "12:04 PM" where the locale says so). */
export function clock(epochS: number): string {
  return clockFormat.format(new Date(epochS * 1000));
}
