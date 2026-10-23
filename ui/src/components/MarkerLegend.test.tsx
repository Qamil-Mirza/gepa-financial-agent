import { fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runHref } from "../routes";
import { changes, explain, fetchStub } from "../test/fixtures";
import { MarkerLegend, markerLabel, metricWords } from "./MarkerLegend";

function renderLegend() {
  const router = createMemoryRouter(
    [
      { path: "/w", element: <MarkerLegend markers={changes.markers} metrics={changes.metrics} /> },
      { path: "/run", element: <p>run page</p> },
    ],
    { initialEntries: ["/w"] },
  );
  return render(<RouterProvider router={router} />);
}

describe("MarkerLegend", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("lists the markers in order with a run link, and explains an L2 alarm inline", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/drifts/3/explain": explain })));
    renderLegend();
    const list = screen.getByRole("list", { name: "Changes" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Model");
    expect(items[0]).toHaveTextContent("m-a → m-b");
    expect(items[0]).toHaveTextContent("SDK runs");
    expect(within(items[0]).getByRole("link", { name: /eda59ed2/ })).toHaveAttribute(
      "href",
      runHref("eda59ed240a64b02a2859e9708ae9a84"),
    );
    expect(within(items[0]).queryByRole("button", { name: "Explain" })).toBeNull();

    expect(items[1]).toHaveTextContent("L2 alarm");
    expect(items[1]).toHaveTextContent("Tokens per call");
    const button = within(items[1]).getByRole("button", { name: "Explain" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const panel = await within(items[1]).findByRole("region", { name: /Explanation/ });
    expect(panel).toHaveTextContent("2,694 → 5,363");
    fireEvent.click(within(items[1]).getByRole("button", { name: "Hide explanation" }));
    expect(within(items[1]).queryByRole("region", { name: /Explanation/ })).toBeNull();
  });

  it("labels every marker kind in words", () => {
    const metrics = changes.metrics;
    expect(markerLabel({ at: 0, kind: "model", before: "m-a", after: "m-b" }, metrics)).toBe("model m-a → m-b");
    expect(markerLabel({ at: 0, kind: "base_url", before: null, after: "https://x" }, metrics)).toBe(
      "endpoint (unset) → https://x",
    );
    expect(markerLabel({ at: 0, kind: "prompt_versions", before: "a=1", after: "a=2" }, metrics)).toBe("prompts a=1 → a=2");
    expect(markerLabel({ at: 0, kind: "l2_alarm", metric: "tokens_per_call" }, metrics)).toBe("L2 alarm: tokens per call");
    expect(markerLabel({ at: 0, kind: "l2_alarm", metric: "odd_metric" }, metrics)).toBe("L2 alarm: odd metric");
    expect(markerLabel({ at: 0, kind: "l2_alarm", metric: "llm_calls" }, metrics)).toBe("L2 alarm: LLM calls");
  });

  it("lowercases a label only when it is not an acronym", () => {
    const metrics = changes.metrics;
    expect(metricWords("llm_calls", metrics)).toBe("LLM calls");
    expect(metricWords("cost_usd", metrics)).toBe("cost per run");
    expect(metricWords("tokens_per_call", metrics)).toBe("tokens per call");
  });
});
