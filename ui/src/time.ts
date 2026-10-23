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
const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

/** The local wall-clock time, "12:04" (or "12:04 PM" where the locale says so). */
export function clock(epochS: number): string {
  return clockFormat.format(new Date(epochS * 1000));
}

/** The local date and time, "Oct 23, 16:35" (locale order and clock). */
export function dateTime(epochS: number): string {
  return dateTimeFormat.format(new Date(epochS * 1000));
}

const trim = (s: string) => s.replace(/\.0$/, "");

/** A duration in seconds as people read it: "0.6 ms", "320 ms", "1.2 s", "48 s", "2 min 5 s", "1 h 4 min".
 *  Unknown (null, NaN, negative) is "—", never 0 (decision 9). */
export function duration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s) || s < 0) return "—";
  const ms = s * 1000;
  if (ms < 10) return `${trim(ms.toFixed(1))} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (s < 60) return `${trim(s.toFixed(1))} s`;
  if (s < 3600) {
    const min = Math.floor(s / 60);
    const rest = Math.round(s - min * 60);
    return rest > 0 ? `${min} min ${rest} s` : `${min} min`;
  }
  const h = Math.floor(s / 3600);
  const min = Math.round((s - h * 3600) / 60);
  return min > 0 ? `${h} h ${min} min` : `${h} h`;
}
