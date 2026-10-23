import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  dismissDrift,
  explainDrift,
  getBoard,
  getChanges,
  getFeed,
  getGraph,
  getHealth,
  getRun,
  getRuns,
  getSettings,
  getWorkflow,
} from "./client";

type Call = { url: URL; init: RequestInit | undefined };

function stubFetch(status = 200, body: unknown = {}) {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: new URL(String(input), "http://localhost"), init });
      const text = typeof body === "string" ? body : JSON.stringify(body);
      return new Response(text, { status, headers: { "content-type": "application/json" } });
    }),
  );
  return calls;
}

function header(init: RequestInit | undefined, name: string): string | null {
  return new Headers(init?.headers).get(name);
}

describe("api client", () => {
  let calls: Call[];
  beforeEach(() => {
    calls = stubFetch();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("client builds query strings and sends X-PitStop on writes", async () => {
    const name = "a/b ?#🏁";
    await getBoard();
    await getFeed(20);
    await getWorkflow(name);
    await getRuns(name);
    await getRuns(name, 1792773350.5);
    await getRun("r/1");
    await getGraph(name, { window: "7d" });
    await getGraph(name, { runId: "abc", window: "24h" });
    await getChanges(name);
    await getSettings();
    await getHealth();
    await explainDrift(3);
    await dismissDrift(7);

    const seen = calls.map((c) => [c.url.pathname, Object.fromEntries(c.url.searchParams)]);
    expect(seen).toEqual([
      ["/api/board", {}],
      ["/api/feed", { limit: "20" }],
      ["/api/workflow", { name }],
      ["/api/runs", { workflow: name }],
      ["/api/runs", { workflow: name, before: "1792773350.5" }],
      ["/api/runs/r%2F1", {}],
      ["/api/graph", { workflow: name, window: "7d" }],
      ["/api/graph", { workflow: name, run_id: "abc", window: "24h" }],
      ["/api/changes", { workflow: name }],
      ["/api/settings", {}],
      ["/api/health", {}],
      ["/api/drifts/3/explain", {}],
      ["/api/drifts/7/dismiss", {}],
    ]);
    // the raw query never carries a bare "#" or "?" from the name
    expect(calls[2].url.search).toBe("?name=a%2Fb+%3F%23%F0%9F%8F%81");

    const reads = calls.slice(0, -1);
    for (const c of reads) {
      expect(c.init?.method ?? "GET").toBe("GET");
    }
    const write = calls[calls.length - 1];
    expect(write.init?.method).toBe("POST");
    expect(header(write.init, "X-PitStop")).toBe("1");
  });

  it("throws ApiError with the server's message", async () => {
    stubFetch(503, { error: "store busy, retry" });
    await expect(getBoard()).rejects.toEqual(new ApiError(503, "store busy, retry"));

    stubFetch(422, { detail: "drift 4 has no change point" });
    await expect(explainDrift(4)).rejects.toMatchObject({ status: 422, message: "drift 4 has no change point" });

    stubFetch(422, { detail: [{ loc: ["query", "name"], msg: "Field required", type: "missing" }] });
    await expect(getWorkflow("")).rejects.toMatchObject({ status: 422, message: "name: Field required" });

    stubFetch(409, { status: "not_open" });
    await expect(dismissDrift(2)).rejects.toMatchObject({ status: 409, message: "This drift is no longer open." });

    stubFetch(404, { status: "not_found" });
    await expect(dismissDrift(2)).rejects.toMatchObject({ status: 404, message: "This drift no longer exists." });

    stubFetch(502, "<html>bad gateway</html>");
    await expect(getBoard()).rejects.toMatchObject({ status: 502, message: "The server answered 502." });

    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(getBoard()).rejects.toMatchObject({ status: 0, message: "Can't reach the PitStop server." });
  });
});
