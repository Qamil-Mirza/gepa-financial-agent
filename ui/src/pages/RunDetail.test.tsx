import { render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunDetail as RunDetailData } from "../api/client";
import { runHref, workflowHref } from "../routes";
import { fetchStub, graph, run } from "../test/fixtures";
import { mockReactFlow } from "../test/reactflow";
import { RunDetail } from "./RunDetail";

mockReactFlow();

const ID = run.run.id;

function renderPage(detail: RunDetailData = run, entry = runHref(ID)) {
  vi.stubGlobal("fetch", vi.fn(fetchStub({ [`/api/runs/${ID}`]: detail, "/api/graph": graph })));
  const router = createMemoryRouter(
    [
      { path: "/", element: <p>pit wall</p> },
      { path: "/w", element: <p>workflow page</p> },
      { path: "/run", element: <RunDetail /> },
    ],
    { initialEntries: [entry] },
  );
  render(<RouterProvider router={router} />);
  return router;
}

describe("RunDetail", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows the run's facts, its graph and its events", async () => {
    renderPage();
    const heading = await screen.findByRole("heading", { level: 1, name: /Run a5811371/ });
    expect(heading).toBeVisible();
    expect(screen.getByRole("link", { name: "earnings-research" })).toHaveAttribute("href", workflowHref("earnings-research"));
    expect(screen.getByText("Summarize each quarter's earnings for the internal research desk")).toBeVisible();

    const facts = screen.getByRole("list", { name: "Run facts" });
    expect(within(facts).getByText("Status").nextElementSibling).toHaveTextContent("halted");
    expect(within(facts).getByText("Model").nextElementSibling).toHaveTextContent("m-b");
    expect(within(facts).getByText("Endpoint").nextElementSibling).toHaveTextContent("https://llm.offline.invalid/v1");
    expect(within(facts).getByText("Adapter").nextElementSibling).toHaveTextContent("wrap");
    expect(within(facts).getByText("Cost").nextElementSibling).toHaveTextContent("$0.0081");
    expect(within(facts).getByText("Duration").nextElementSibling).toHaveTextContent("3.7 ms");
    expect(within(facts).getByText("Calls").nextElementSibling).toHaveTextContent("4 model · 2 tool");
    expect(within(facts).getByText("Verdicts").nextElementSibling).toHaveTextContent("2");
    expect(facts).toHaveTextContent(ID); // the full id, for copying

    // the run's own graph, without the scope controls
    expect(await screen.findByRole("button", { name: "researcher, agent" }, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.queryByRole("radio", { name: "24h" })).toBeNull();
    expect(screen.queryByRole("combobox", { name: "Run" })).toBeNull();

    expect(screen.getAllByRole("list", { name: /events of/ })).toHaveLength(2);
    expect(screen.queryByText(/Showing the first/)).toBeNull();
  });

  it("run detail shows truncated notice", async () => {
    renderPage({ ...run, truncated: true });
    await screen.findByRole("heading", { level: 1, name: /Run a5811371/ });
    expect(screen.getByText("Showing the first 2,000 events")).toBeVisible();
  });

  it("an unknown run is not found, not an error", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({})));
    const router = createMemoryRouter([{ path: "/run", element: <RunDetail /> }], { initialEntries: [runHref("ghost")] });
    render(<RouterProvider router={router} />);
    expect(await screen.findByRole("heading", { level: 1, name: "Run not found" })).toBeVisible();
    expect(screen.getByText(/ghost/)).toBeVisible();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("a missing id says so", () => {
    const router = createMemoryRouter([{ path: "/run", element: <RunDetail /> }], { initialEntries: ["/run"] });
    render(<RouterProvider router={router} />);
    expect(screen.getByRole("heading", { level: 1, name: "No run given" })).toBeVisible();
  });
});
