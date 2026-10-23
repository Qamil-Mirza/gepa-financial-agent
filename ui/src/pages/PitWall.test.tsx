import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Board, FeedItem } from "../api/client";
import { workflowHref } from "../routes";
import { board, boardEmpty, feed, fetchStub } from "../test/fixtures";
import { PitWall } from "./PitWall";

function renderWall() {
  const router = createMemoryRouter(
    [
      { path: "/", element: <PitWall /> },
      { path: "/w", element: <p>workflow page</p> },
    ],
    { initialEntries: ["/"] },
  );
  return render(<RouterProvider router={router} />);
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const flush = () => act(async () => {});

/** The wall's section (the feed links workflows too, so tile lookups stay inside it). */
const wall = () => screen.getByRole("region", { name: "Pit wall" });

function tileOf(name: string): HTMLElement {
  const tile = within(wall()).getByRole("link", { name }).closest(".tile");
  if (!(tile instanceof HTMLElement)) throw new Error(`no tile for ${name}`);
  return tile;
}

describe("PitWall", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("renders red and yellow as large tiles and green as small ones", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": board, "/api/feed": feed })));
    renderWall();
    expect(await screen.findByRole("heading", { level: 1, name: "Pit wall" })).toBeVisible();
    expect(screen.getByText("2 workflows need you · 2 healthy")).toBeVisible();

    const big = screen.getByRole("list", { name: "Workflows that need you" });
    const bigItems = within(big).getAllByRole("listitem");
    expect(bigItems).toHaveLength(2);
    expect(within(bigItems[0]).getByRole("article")).toHaveClass("tile-big");
    expect(within(bigItems[0]).getByRole("link", { name: "earnings-research" })).toHaveAttribute(
      "href",
      workflowHref("earnings-research"),
    );
    expect(within(bigItems[1]).getByRole("link", { name: "support-triage" })).toBeVisible();

    const small = screen.getByRole("list", { name: "Healthy workflows" });
    const smallItems = within(small).getAllByRole("listitem");
    expect(smallItems).toHaveLength(2);
    expect(within(smallItems[0]).getByRole("article")).toHaveClass("tile-small");
    expect(within(smallItems[0]).getByRole("link", { name: "nightly-digest" })).toHaveAttribute(
      "href",
      workflowHref("nightly-digest"),
    );
    expect(within(smallItems[1]).getByRole("link", { name: "kb-sync" })).toBeVisible();
    // a small tile still carries the money and the day's activity
    expect(smallItems[0]).toHaveTextContent("14 runs today");
    expect(smallItems[0]).toHaveTextContent("$0.0166");
    expect(smallItems[0]).toHaveTextContent("Healthy");
  });

  it("tile shows text flag label and level chips", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": board, "/api/feed": feed })));
    renderWall();
    await screen.findByRole("heading", { level: 1, name: "Pit wall" });

    const tile = tileOf("earnings-research");
    expect(within(tile).getByText("Halted")).toBeVisible(); // the flag in words, never color alone
    expect(tile).toHaveTextContent("Criticality high");
    const levels = tile.querySelectorAll(".level");
    expect(levels).toHaveLength(2); // non-zero levels only
    expect(levels[0]).toHaveClass("level-p0");
    expect(levels[0]).toHaveTextContent("P0: 1 open");
    expect(levels[1]).toHaveClass("level-p1");
    expect(levels[1]).toHaveTextContent("P1: 1 open");
    // the headline in plain words, without repeating the level and the name the tile already shows
    expect(within(tile).getByText("Out of scope action")).toBeVisible();
    expect(tile).toHaveTextContent(/Last run halted/);
    // spend against the daily budget, as text as well as a bar
    const meter = within(tile).getByRole("meter", { name: /spend today/i });
    expect(meter).toHaveAttribute("aria-valuenow", "0.6572427");
    expect(meter).toHaveAttribute("aria-valuemax", "2");
    expect(tile).toHaveTextContent("$0.6572");
    expect(tile).toHaveTextContent("of $2.00");
    expect(tile).toHaveTextContent("33%");
    expect(tile).not.toHaveTextContent("Fallback active");

    // no daily budget: spend only, no bar
    const yellow = tileOf("support-triage");
    expect(within(yellow).getByText("Degraded")).toBeVisible();
    expect(yellow).toHaveTextContent("$0.0331");
    expect(yellow).toHaveTextContent("no daily budget");
    expect(within(yellow).queryByRole("meter")).toBeNull();
  });

  it("shows fallback and the global budgets in the header", async () => {
    const withBudgets: Board = {
      ...board,
      workflows: board.workflows.map((t, i) => (i === 1 ? { ...t, fallback_active: true } : t)),
      budgets: [
        { name: "prod", scope: "global:prod", period: "day", spent_usd: 10.65, limit_usd: 20, remaining_usd: 9.35 },
        { name: "canary", scope: "global:canary", period: "month", spent_usd: 50, limit_usd: 40, remaining_usd: 0 },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": withBudgets, "/api/feed": feed })));
    renderWall();
    await screen.findByRole("heading", { level: 1, name: "Pit wall" });

    expect(tileOf("support-triage")).toHaveTextContent("Fallback active");

    const prod = screen.getByRole("meter", { name: "prod budget, today" });
    expect(prod).toHaveAttribute("aria-valuenow", "10.65");
    expect(prod).toHaveAttribute("aria-valuemax", "20");
    const prodRow = prod.closest(".meter");
    expect(prodRow).toHaveTextContent("$10.65");
    expect(prodRow).toHaveTextContent("of $20.00");
    expect(prodRow).toHaveTextContent("53%");

    const canary = screen.getByRole("meter", { name: "canary budget, this month" });
    expect(canary.closest(".meter")).toHaveClass("meter-over");
    expect(canary.closest(".meter")).toHaveTextContent("125%");
  });

  it("feed lists newest first and links workflows", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": board, "/api/feed": feed })));
    renderWall();
    const list = await screen.findByRole("list", { name: "Live feed" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(feed.length);
    expect(items[0]).toHaveTextContent("Drift opened: cost_or_latency_creep");
    expect(items[0]).toHaveTextContent("Drift");
    expect(items[0].querySelector(".level")).toHaveTextContent("P1");
    expect(within(items[0]).getByRole("link", { name: "earnings-research" })).toHaveAttribute(
      "href",
      workflowHref("earnings-research"),
    );
    expect(items[1]).toHaveTextContent("Halt");
    expect(items[1]).toHaveTextContent(/Halted: out_of_scope_action/);
    expect(items[1].querySelector(".level")).toBeNull();
    expect(items[3]).toHaveTextContent("Model change");
    expect(items[4]).toHaveTextContent("Drift opened: unproductive_loop");
    expect(within(items[4]).getByRole("link", { name: "support-triage" })).toHaveAttribute(
      "href",
      workflowHref("support-triage"),
    );
    // the first load does not fade anything in
    expect(list.querySelector(".feed-new")).toBeNull();
  });

  it("empty board shows EmptyState", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": boardEmpty, "/api/feed": [] })));
    renderWall();
    expect(await screen.findByRole("heading", { name: "No workflows yet" })).toBeVisible();
    expect(screen.queryByRole("list", { name: "Live feed" })).toBeNull();
  });

  it("error shows retry", async () => {
    const fetchMock = vi.fn<(input: RequestInfo | URL) => Promise<Response>>(async () =>
      Promise.reject(new TypeError("Failed to fetch")),
    );
    vi.stubGlobal("fetch", fetchMock);
    renderWall();
    expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the PitStop server");
    const retry = screen.getByRole("button", { name: "Try again" });
    expect(retry).toBeVisible();

    fetchMock.mockImplementation(fetchStub({ "/api/board": board, "/api/feed": feed }));
    fireEvent.click(retry); // one retry fetches the board and the feed again
    expect(await screen.findByRole("region", { name: "Pit wall" })).toBeVisible();
    expect(tileOf("earnings-research")).toBeVisible();
    expect(screen.getByRole("list", { name: "Live feed" })).toBeVisible();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps the wall on a failed poll and says it is stale; new feed items fade in", async () => {
    vi.useFakeTimers();
    let boardCalls = 0;
    let feedCalls = 0;
    const later: FeedItem = {
      id: "halt:9999",
      ts: (feed[0].ts ?? 0) + 60,
      kind: "halt",
      workflow: "support-triage",
      level: null,
      text: "Halted: unproductive_loop",
      run_id: "deadbeef",
      drift_id: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = new URL(String(input), "http://localhost").pathname;
        if (path === "/api/board") {
          boardCalls += 1;
          if (boardCalls > 1) throw new TypeError("Failed to fetch");
          return json(board);
        }
        if (path === "/api/feed") {
          feedCalls += 1;
          return json(feedCalls === 1 ? feed : [later, ...feed]);
        }
        return json({ detail: "Not Found" }, 404);
      }),
    );
    renderWall();
    await flush();
    expect(tileOf("earnings-research")).toBeVisible();
    expect(screen.queryByText(/Couldn't refresh/)).toBeNull();
    const list = screen.getByRole("list", { name: "Live feed" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(feed.length);

    // 5 s: the feed polls again and the new item is first, marked for the fade-in; the old ones are not
    await act(() => vi.advanceTimersByTimeAsync(5_000));
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(feed.length + 1);
    expect(items[0]).toHaveTextContent("Halted: unproductive_loop");
    expect(items[0]).toHaveClass("feed-new");
    expect(items[1]).not.toHaveClass("feed-new");

    // 10 s: the board poll fails; the tiles stay and a quiet line says so
    await act(() => vi.advanceTimersByTimeAsync(5_000));
    expect(boardCalls).toBe(2);
    expect(tileOf("earnings-research")).toBeVisible();
    const stale = screen.getByRole("status", { name: "Board not refreshed" });
    expect(stale).toHaveTextContent(/Couldn't refresh — showing data from \d{1,2}:\d{2}/);
    expect(stale).toHaveTextContent("Retrying.");
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });
});
