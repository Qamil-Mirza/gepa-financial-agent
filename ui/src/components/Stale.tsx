// The quiet line under a section whose poll failed while older data stays on screen: what, since when, and that
// the page keeps trying. Shared by the pit wall and the workflow page.
import { useEffect, useState } from "react";
import { clock } from "../time";

/** When `data` last changed (ms), for the staleness line of a poll whose answer carries no time of its own. */
export function useStampedAt(data: unknown): number | null {
  const [at, setAt] = useState<number | null>(null);
  useEffect(() => {
    if (data !== null) setAt(Date.now());
  }, [data]);
  return at;
}

export function Stale({ what, sinceS }: { what: string; sinceS: number | null }) {
  const since = sinceS !== null ? ` — showing data from ${clock(sinceS)}` : "";
  return (
    <p className="stale" role="status" aria-label={`${what} not refreshed`}>
      Couldn't refresh{since}. Retrying.
    </p>
  );
}
