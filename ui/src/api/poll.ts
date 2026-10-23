// Live updates by polling (decision 11): fetch now, then every `intervalMs` after each answer. Polling pauses
// while the tab is hidden and refreshes at once when it is shown again; an in-flight request is aborted on
// unmount, on `refresh()` and when `deps` change.
import { useCallback, useEffect, useRef, useState } from "react";

export interface Poll<T> {
  /** The last successful answer; kept while a later poll fails. Null until the first answer. */
  data: T | null;
  /** The last poll's error, cleared by the next success. */
  error: Error | null;
  /** True until the first answer or error. */
  loading: boolean;
  /** Fetch again now (and restart the interval). */
  refresh: () => void;
}

interface State<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

/**
 * Poll `fn` every `intervalMs`. `fn` gets an AbortSignal to pass to the API client. The latest `fn` is always the
 * one called, so an inline arrow is fine; list in `deps` whatever should restart polling from scratch (e.g. the
 * workflow name), which also clears the previous data.
 */
export function usePoll<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  deps: readonly unknown[] = [],
): Poll<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn; // before the polling effect below, so its first tick already sees the latest fn
  });
  const refreshRef = useRef<() => void>(() => {});

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;
    setState({ data: null, error: null, loading: true });

    const stop = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      controller?.abort();
      controller = null;
    };

    const schedule = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = alive && !document.hidden ? setTimeout(tick, intervalMs) : undefined;
    };

    async function tick() {
      timer = undefined;
      if (!alive || document.hidden) return;
      controller?.abort();
      const mine = new AbortController();
      controller = mine;
      try {
        const data = await fnRef.current(mine.signal);
        if (!alive || controller !== mine) return;
        setState({ data, error: null, loading: false });
      } catch (e) {
        if (!alive || controller !== mine || isAbort(e)) return;
        setState((s) => ({ data: s.data, error: e instanceof Error ? e : new Error(String(e)), loading: false }));
      }
      if (controller === mine) controller = null;
      schedule();
    }

    const onVisibility = () => {
      if (document.hidden) {
        stop();
      } else if (timer === undefined && controller === null) {
        void tick();
      }
    };

    refreshRef.current = () => {
      stop();
      void tick();
    };
    document.addEventListener("visibilitychange", onVisibility);
    void tick();
    return () => {
      alive = false;
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intervalMs, ...deps]);

  const refresh = useCallback(() => refreshRef.current(), []);
  return { ...state, refresh };
}
