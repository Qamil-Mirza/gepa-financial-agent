import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runHref, workflowHref } from "../routes";
import { changes, explain, fetchStub, graph, runs, workflow } from "../test/fixtures";
import { mockReactFlow } from "../test/reactflow";
import { mergeRuns, Workflow } from "./Workflow";

mockReactFlow();

const flush = () => act(async () => {});

const NAME = "earnings-research";
const API = {
  "/api/workflow": workflow,
  "/api/graph": graph,
  "/api/runs": runs,
  "/api/drifts/3/explain": explain,
};

function renderPage(entry = workflowHref(NAME)) {
  const router = createMemoryRouter(
    [
      { path: "/", element: <p>pit wall</p> },
      { path: "/w", element: <Workflow /> },
      { path: "/run", element: <p>run page</p> },
    ],
    { initialEntries: [entry] },
  );
  const view = render(<RouterProvider router={router} />);
  return { router, ...view };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/** The graph node card holding the button named `name`. */
function nodeCard(name: string): HTMLElement {
  const card = screen.getByRole("button", { name }).closest(".gnode");
  if (!(card instanceof HTMLElement)) throw new Error(`no node card for ${name}`);
  return card;
}

/** The request URLs a fetch mock saw, as path + query. */
function requested(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map((call) => {
    const url = new URL(String(call[0]), "http://localhost");
    return url.pathname + url.search;
  });
}

describe("Workflow", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("shows the header facts and the run graph by default", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    renderPage();
    const heading = await screen.findByRole("heading", { level: 1, name: NAME });
    expect(heading).toBeVisible();
    const header = heading.closest("header");
    if (!header) throw new Error("no header");
    expect(within(header).getByText("Halted")).toBeVisible(); // the flag in words
    expect(header).toHaveTextContent("Criticality high");
    expect(header).toHaveTextContent("Summarize each quarter's earnings for the internal research desk");
    expect(header).toHaveTextContent(/Last run halted/);
    const meter = within(header).getByRole("meter", { name: /spend today/i });
    expect(meter).toHaveAttribute("aria-valuemax", "2");
    expect(header).toHaveTextContent("$0.6572");
    expect(header).toHaveTextContent("of $2.00");

    const tabs = screen.getByRole("navigation", { name: "Workflow views" });
    expect(within(tabs).getByRole("link", { name: "Run graph" })).toHaveAttribute("aria-current", "page");
    expect(within(tabs).getByRole("link", { name: "What changed" })).not.toHaveAttribute("aria-current");
    expect(await screen.findByRole("button", { name: "researcher, agent" }, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByText("last 24 hours · 95 runs, averaged")).toBeVisible();
  });

  it("slowest path is labelled", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    renderPage();
    await screen.findByRole("button", { name: "m-a, model under researcher" }, { timeout: 5000 });

    const hot = nodeCard("m-a, model under researcher");
    expect(hot).toHaveClass("gnode-slow");
    expect(hot).toHaveTextContent("on slowest path"); // in words, never weight or color alone
    expect(nodeCard("researcher, agent")).toHaveClass("gnode-slow");
    const cold = nodeCard("m-a, model under writer");
    expect(cold).not.toHaveClass("gnode-slow");
    expect(cold).not.toHaveTextContent("on slowest path");

    // each node carries its facts and its verdicts as text
    expect(hot).toHaveTextContent("180 calls");
    expect(hot).toHaveTextContent("$0.1665");
    expect(nodeCard("publish_note, tool under writer")).toHaveTextContent("1 blocked");
    expect(nodeCard("publish_note, tool under writer")).toHaveTextContent("1 gate reject");
    expect(nodeCard("writer, agent")).toHaveTextContent("1 halted");

    const line = screen.getByText(/^Slowest path/);
    expect(line).toHaveTextContent("Slowest path: earnings-research → researcher → m-a");
  });

  it("node click opens the drawer with stats and run links", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    renderPage();
    const button = await screen.findByRole("button", { name: "m-a, model under researcher" }, { timeout: 5000 });
    expect(screen.queryByRole("complementary", { name: /m-a/ })).toBeNull();

    fireEvent.click(button);
    const drawer = screen.getByRole("complementary", { name: "m-a" });
    expect(drawer).toHaveTextContent("Model under researcher");
    expect(drawer).toHaveTextContent("180");
    expect(drawer).toHaveTextContent("60 of 95 runs");
    expect(drawer).toHaveTextContent("$0.1665");
    expect(drawer).toHaveTextContent("on slowest path");
    expect(drawer).toHaveTextContent("No verdicts");
    const runLinks = within(drawer).getAllByRole("link");
    expect(runLinks).toHaveLength(20);
    expect(runLinks[0]).toHaveAttribute("href", runHref("d54a5aca66ad44ddb2b3d7498e33527b"));
    expect(runLinks[0]).toHaveTextContent("d54a5aca");

    // the drawer closes from its button and from Escape
    fireEvent.click(within(drawer).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("complementary", { name: "m-a" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "writer, agent" }));
    const writer = screen.getByRole("complementary", { name: "writer" });
    expect(writer).toHaveTextContent("1 halted");
    fireEvent.keyDown(writer, { key: "Escape" });
    expect(screen.queryByRole("complementary", { name: "writer" })).toBeNull();
  });

  it("window picker and run selector set the scope", async () => {
    const fetchMock = vi.fn(fetchStub(API));
    vi.stubGlobal("fetch", fetchMock);
    const { router } = renderPage();
    await screen.findByRole("button", { name: "researcher, agent" }, { timeout: 5000 });
    expect(requested(fetchMock)).toContain(`/api/graph?workflow=${NAME}&window=24h`);

    fireEvent.click(screen.getByRole("radio", { name: "7d" }));
    expect(screen.getByRole("radio", { name: "7d" })).toBeChecked();
    await act(async () => {});
    expect(requested(fetchMock)).toContain(`/api/graph?workflow=${NAME}&window=7d`);
    expect(router.state.location.search).toContain("window=7d");

    const select = screen.getByRole("combobox", { name: "Run" });
    const first = runs.runs[0].id;
    fireEvent.change(select, { target: { value: first } });
    await act(async () => {});
    expect(requested(fetchMock)).toContain(`/api/graph?workflow=${NAME}&run_id=${first}&window=7d`);
    expect(router.state.location.search).toContain(`run=${first}`);
    expect(screen.getByRole("radio", { name: "7d" })).toBeDisabled(); // one run has no window
  });

  it("changing the window with a drawer open leaves focus on the window control", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    renderPage();
    const button = await screen.findByRole("button", { name: "writer, agent" }, { timeout: 5000 });
    fireEvent.click(button);
    const drawer = screen.getByRole("complementary", { name: "writer" });
    expect(drawer).toHaveFocus(); // a person opened it: the keyboard follows

    // the person moves to the scope controls and picks another window: the graph reloads and the drawer re-mounts
    const radio = screen.getByRole("radio", { name: "7d" });
    radio.focus();
    fireEvent.click(radio);
    expect(screen.queryByRole("complementary", { name: "writer" })).toBeNull(); // gone while the new graph loads
    const again = await screen.findByRole("complementary", { name: "writer" }, { timeout: 5000 });
    expect(again).toHaveTextContent("1 halted");
    expect(radio).toHaveFocus(); // the re-mounted drawer did not steal it

    // a fresh open still takes focus
    fireEvent.click(screen.getByRole("button", { name: "m-b, model under writer" }));
    expect(screen.getByRole("complementary", { name: "m-b" })).toHaveFocus();
  });

  it("dismiss is optimistic and restores on failure", async () => {
    let fail: (() => void) | undefined;
    const base = fetchStub(API);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        if (new URL(String(input), "http://localhost").pathname === "/api/drifts/2/dismiss" && init?.method === "POST") {
          expect(new Headers(init.headers).get("X-PitStop")).toBe("1");
          return new Promise<Response>((resolve) => {
            fail = () => resolve(json({ error: "store busy, retry" }, 503));
          });
        }
        return base(input);
      }),
    );
    renderPage();
    const list = await screen.findByRole("list", { name: "Open drifts" });
    expect(within(list).getAllByRole("article")).toHaveLength(2);
    const card = within(list).getByRole("article", { name: "Out of scope action" });
    expect(card).toHaveTextContent("P0");
    expect(card).toHaveTextContent("The model changed from m-a to m-b");
    expect(card).toHaveTextContent("Blocked the side-effect tool call");
    expect(card).toHaveTextContent("1 occurrence");

    // "No" keeps the card
    fireEvent.click(within(card).getByRole("button", { name: "Dismiss" }));
    expect(card).toHaveTextContent("Dismiss?");
    fireEvent.click(within(card).getByRole("button", { name: /^No/ }));
    expect(card).not.toHaveTextContent("Dismiss?");
    expect(within(list).getAllByRole("article")).toHaveLength(2);

    // "Yes" removes it at once, before the server answers
    fireEvent.click(within(card).getByRole("button", { name: "Dismiss" }));
    fireEvent.click(within(card).getByRole("button", { name: /^Yes/ }));
    expect(within(list).queryByRole("article", { name: "Out of scope action" })).toBeNull();
    expect(within(list).getAllByRole("article")).toHaveLength(1);
    expect(fail).toBeDefined();

    // the server says no: the card comes back with the reason
    await act(async () => fail?.());
    const restored = await within(list).findByRole("article", { name: "Out of scope action" });
    expect(within(list).getAllByRole("article")).toHaveLength(2);
    expect(within(restored).getByRole("alert")).toHaveTextContent(/Couldn't dismiss/);
    expect(within(restored).getByRole("alert")).toHaveTextContent("store busy");
    expect(within(restored).getByRole("button", { name: "Dismiss" })).toBeVisible();
    expect(screen.queryByRole("dialog")).toBeNull(); // never a browser or modal dialog
  });

  it("explain opens inline", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    renderPage();
    const list = await screen.findByRole("list", { name: "Open drifts" });
    const card = within(list).getByRole("article", { name: "Cost or latency creep" });
    const button = within(card).getByRole("button", { name: "Explain" });
    expect(button).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const panel = await within(card).findByRole("region", { name: /Explanation/ });
    expect(panel).toHaveTextContent("Model changed from m-a to m-b");
    expect(panel).toHaveTextContent("Tokens per call");
    expect(panel).toHaveTextContent("2,694 → 5,363");
    expect(panel).toHaveTextContent("50 runs before · 35 after");
    expect(panel).toHaveTextContent("m-a → m-b");
    expect(panel).toHaveTextContent("0% → 100% of runs");
    expect(screen.queryByRole("dialog")).toBeNull();

    fireEvent.click(within(card).getByRole("button", { name: "Hide explanation" }));
    expect(within(card).queryByRole("region", { name: /Explanation/ })).toBeNull();
  });

  it("tab is reflected in the URL", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub(API)));
    const { router } = renderPage(workflowHref(NAME, "changes"));
    await screen.findByRole("heading", { level: 1, name: NAME });
    const tabs = screen.getByRole("navigation", { name: "Workflow views" });
    expect(within(tabs).getByRole("link", { name: "What changed" })).toHaveAttribute("aria-current", "page");
    expect(within(tabs).getByRole("link", { name: "Run graph" })).not.toHaveAttribute("aria-current");
    expect(screen.queryByRole("radio", { name: "24h" })).toBeNull(); // the graph is not mounted on another tab

    fireEvent.click(within(tabs).getByRole("link", { name: "Runs" }));
    expect(new URLSearchParams(router.state.location.search).get("tab")).toBe("runs");
    expect(new URLSearchParams(router.state.location.search).get("name")).toBe(NAME);
    expect(within(tabs).getByRole("link", { name: "Runs" })).toHaveAttribute("aria-current", "page");

    fireEvent.click(within(tabs).getByRole("link", { name: "Run graph" }));
    expect(new URLSearchParams(router.state.location.search).get("tab")).toBe("graph");
    expect(await screen.findByRole("radio", { name: "24h" }, { timeout: 5000 })).toBeChecked();
    // the right column stays on every tab
    expect(screen.getByRole("list", { name: "Open drifts" })).toBeVisible();
  });

  it("chart draws one path per metric and labels markers", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ ...API, "/api/changes": changes })));
    const { container } = renderPage(workflowHref(NAME, "changes"));
    const figures = await screen.findAllByRole("figure");
    expect(figures).toHaveLength(changes.metrics.length); // every metric has data in the demo
    expect(figures.map((f) => f.getAttribute("aria-label") ?? "")).toEqual([
      "Tool errors",
      "Refusals",
      "Tokens per call",
      "LLM calls",
      "Cost per run",
      "Duration",
      "Output length",
    ]);
    expect(container.querySelectorAll("path.series-line")).toHaveLength(changes.metrics.length);
    // every chart carries the change lines; the words appear once, on the first chart
    expect(container.querySelectorAll("line.series-marker")).toHaveLength(changes.metrics.length * changes.markers.length);
    expect(screen.getAllByText("model m-a → m-b")).toHaveLength(1);
    expect(screen.getAllByText("L2 alarm: tokens per call")).toHaveLength(1);
    // the legend, the table twin and the judge note
    const legend = screen.getByRole("list", { name: "Changes" });
    expect(within(legend).getAllByRole("listitem")).toHaveLength(2);
    expect(within(legend).getByRole("button", { name: "Explain" })).toBeVisible();
    expect(screen.getByText("Quality score: coming with the run judge.")).toBeVisible();
    const table = screen.getByRole("table", { name: "Every run's metrics" });
    expect(within(table).getAllByRole("row")).toHaveLength(changes.points.length + 1);
    expect(within(table).getAllByRole("link")[0]).toHaveAttribute("href", runHref(changes.points[0].run_id));
    expect(screen.queryByRole("radio", { name: "24h" })).toBeNull();
  });

  it("runs table loads more with cursor", async () => {
    const page2 = {
      runs: runs.runs.slice(0, 3).map((r, i) => ({ ...r, id: `older-${i}`, started_at: (r.started_at ?? 0) - 1000 })),
      next_before: null,
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://localhost");
      if (url.pathname === "/api/runs" && url.searchParams.has("before")) return json(page2);
      return fetchStub(API)(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    renderPage(workflowHref(NAME, "runs"));
    const table = await screen.findByRole("table", { name: "Runs" });
    expect(within(table).getAllByRole("row")).toHaveLength(runs.runs.length + 1);
    expect(screen.getByText("50 runs, newest first")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await screen.findByRole("link", { name: /older-0/ });
    expect(requested(fetchMock)).toContain(`/api/runs?workflow=${NAME}&before=${runs.next_before}`);
    expect(within(table).getAllByRole("row")).toHaveLength(runs.runs.length + 3 + 1);
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull(); // the last page
    expect(screen.getByText("53 runs, newest first")).toBeVisible();
  });

  it("a poll refresh keeps every loaded run and puts new ones on top", async () => {
    vi.useFakeTimers();
    const base = runs.runs;
    let firstPages = 0;
    const newer = base.slice(0, 2).map((r, i) => ({ ...r, id: `new-${i}`, started_at: (r.started_at ?? 0) + 10 - i }));
    // two new runs: the refreshed page 1 (the newest 50) no longer holds the two oldest of the first answer
    const refreshed = { runs: [...newer, ...base.slice(0, 48)], next_before: base[47].started_at };
    const page2 = {
      runs: base.slice(0, 3).map((r, i) => ({ ...r, id: `older-${i}`, started_at: (r.started_at ?? 0) - 1000 - i })),
      next_before: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), "http://localhost");
        if (url.pathname === "/api/runs") {
          if (url.searchParams.has("before")) return json(page2);
          firstPages += 1;
          return json(firstPages === 1 ? runs : refreshed);
        }
        return fetchStub(API)(input);
      }),
    );
    renderPage(workflowHref(NAME, "runs"));
    await flush();
    const table = screen.getByRole("table", { name: "Runs" });
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    await flush();
    expect(within(table).getAllByRole("row")).toHaveLength(50 + 3 + 1);

    // 10 s: page 1 polls again with the two new runs on it and the two oldest pushed off
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(firstPages).toBe(2);
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2 + 50 + 3); // nothing dropped
    expect(within(rows[0]).getByRole("link", { name: "new-0" })).toBeInTheDocument();
    expect(within(rows[1]).getByRole("link", { name: "new-1" })).toBeInTheDocument();
    // the two runs that slid off page 1 keep their place between page 1 and page 2
    expect(within(rows[50]).getByRole("link", { name: base[48].id.slice(0, 8) })).toBeInTheDocument();
    expect(within(rows[51]).getByRole("link", { name: base[49].id.slice(0, 8) })).toBeInTheDocument();
    expect(within(rows[52]).getByRole("link", { name: "older-0" })).toBeInTheDocument();
    expect(screen.getByText("55 runs, newest first")).toBeVisible();
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull(); // the cursor stays the last loaded page's
  });

  it("merges a page of runs without dropping or repeating one", () => {
    const [a, b, c] = runs.runs;
    const bNewer = { ...b, status: "ok", cost_usd: 1 };
    const fresh = { ...a, id: "fresh", started_at: (a.started_at ?? 0) + 5 };
    const merged = mergeRuns([a, b, c], [fresh, bNewer]);
    expect(merged.map((r) => r.id)).toEqual(["fresh", a.id, b.id, c.id]);
    expect(merged[2].cost_usd).toBe(1); // updated in place
    expect(mergeRuns([], [c, a])).toEqual([a, c]); // newest first
  });

  it("unknown workflow shows not-found state", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/graph": graph, "/api/runs": runs })));
    renderPage(workflowHref("ghost"));
    expect(await screen.findByRole("heading", { level: 1, name: "Workflow not found" })).toBeVisible();
    expect(screen.getByText(/ghost/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Back to the pit wall" })).toHaveAttribute("href", "/");
    expect(screen.queryByRole("alert")).toBeNull(); // a missing workflow is not an error
  });
});
