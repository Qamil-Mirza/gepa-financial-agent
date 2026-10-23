import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePoll } from "./poll";

let hidden = false;

function setHidden(value: boolean) {
  hidden = value;
  document.dispatchEvent(new Event("visibilitychange"));
}

function Probe({ fn, interval = 1000 }: { fn: (signal: AbortSignal) => Promise<string>; interval?: number }) {
  const { data, error, loading } = usePoll(fn, interval);
  return <output>{loading ? "loading" : error ? `error: ${error.message}` : (data ?? "none")}</output>;
}

const flush = () => act(async () => {});

describe("usePoll", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    hidden = false;
    Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
  });
  afterEach(() => {
    vi.useRealTimers();
    Reflect.deleteProperty(document, "hidden");
  });

  it("usePoll pauses when hidden", async () => {
    let n = 0;
    const fn = vi.fn(async () => `tick ${++n}`);
    render(<Probe fn={fn} />);
    await flush();
    expect(fn).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("status")).toHaveTextContent("tick 1");

    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(fn).toHaveBeenCalledTimes(2);

    act(() => setHidden(true));
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(fn).toHaveBeenCalledTimes(2); // nothing while the tab is hidden

    act(() => setHidden(false));
    await flush();
    expect(fn).toHaveBeenCalledTimes(3); // refreshes at once on return
    expect(screen.getByRole("status")).toHaveTextContent("tick 3");

    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(fn).toHaveBeenCalledTimes(4);
  });

  it("does not start while hidden and aborts in flight on unmount", async () => {
    hidden = true;
    const signals: AbortSignal[] = [];
    const fn = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return new Promise<string>(() => {}); // never settles
    });
    const { unmount } = render(<Probe fn={fn} />);
    await flush();
    expect(fn).not.toHaveBeenCalled();

    act(() => setHidden(false));
    await flush();
    expect(fn).toHaveBeenCalledTimes(1);
    unmount();
    expect(signals[0].aborted).toBe(true);
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("reports an error, then recovers", async () => {
    let fail = false;
    const fn = vi.fn(async () => {
      if (fail) throw new Error("store busy, retry");
      return "ok";
    });
    render(<Probe fn={fn} />);
    await flush();
    fail = true;
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByRole("status")).toHaveTextContent("error: store busy, retry");
    fail = false;
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(screen.getByRole("status")).toHaveTextContent("ok");
  });
});
