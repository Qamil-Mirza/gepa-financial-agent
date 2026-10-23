import { fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";
import { runHref } from "../routes";
import { changes } from "../test/fixtures";
import { markerSlot, niceCeil, SeriesChart, tickText } from "./SeriesChart";

const found = changes.metrics.find((m) => m.name === "tokens_per_call");
if (!found) throw new Error("fixture has no tokens_per_call metric");
const tokens = found;

function renderChart(labelMarkers = true, markers = changes.markers) {
  const router = createMemoryRouter(
    [
      {
        path: "/w",
        element: (
          <SeriesChart
            metric={tokens}
            points={changes.points}
            markers={markers}
            metrics={changes.metrics}
            labelMarkers={labelMarkers}
            xAxis
          />
        ),
      },
      { path: "/run", element: <p>run page</p> },
    ],
    { initialEntries: ["/w"] },
  );
  const view = render(<RouterProvider router={router} />);
  return { router, ...view };
}

/** The painted dot of `runId` and the plot it sits in. */
function dotOf(container: HTMLElement, runId: string): { dot: SVGCircleElement; svg: SVGSVGElement } {
  const dot = container.querySelector<SVGCircleElement>(`[data-run="${runId}"]`);
  const svg = dot?.closest("svg");
  if (!dot || !svg) throw new Error(`no dot for ${runId}`);
  return { dot, svg };
}

describe("SeriesChart", () => {
  it("draws the series, the markers and a text equivalent", () => {
    const { container } = renderChart();
    const figure = screen.getByRole("figure", { name: /Tokens per call/ });
    expect(figure).toBeVisible();
    expect(container.querySelectorAll("path.series-line")).toHaveLength(1);
    expect(container.querySelectorAll("circle[data-run]")).toHaveLength(changes.points.length);
    // the last run halted: a hollow point
    const last = changes.points[changes.points.length - 1];
    expect(dotOf(container, last.run_id).dot).toHaveClass("series-dot-failed");
    expect(dotOf(container, changes.points[0].run_id).dot).not.toHaveClass("series-dot-failed");
    // one line per marker, between the run before and the run the change was first seen in, and the labels in words
    const lines = container.querySelectorAll("line.series-marker");
    expect(lines).toHaveLength(changes.markers.length);
    const slot = markerSlot(changes.markers[0], changes.points);
    const before = Number(dotOf(container, changes.points[slot - 1].run_id).dot.getAttribute("cx"));
    const at = Number(dotOf(container, changes.points[slot].run_id).dot.getAttribute("cx"));
    expect(Number(lines[0].getAttribute("x1"))).toBeCloseTo((before + at) / 2, 5);
    expect(figure).toHaveTextContent("model m-a → m-b");
    expect(figure).toHaveTextContent("L2 alarm: tokens per call");
    // the summary in words for people who can't see the plot
    expect(figure).toHaveTextContent(/95 runs/);
    expect(figure).toHaveTextContent(/5,100 latest/);
    expect(figure).toHaveTextContent(/from 2,623 to 5,464/);
  });

  it("labels another metric's L2 alarm with that metric's label", () => {
    const at = changes.points[10].started_at;
    renderChart(true, [
      { at, kind: "l2_alarm", metric: "cost_usd", run_id: changes.points[10].run_id, drift_id: 8 },
      { at, kind: "l2_alarm", metric: "llm_calls", run_id: changes.points[10].run_id, drift_id: 9 },
    ]);
    const figure = screen.getByRole("figure", { name: /Tokens per call/ });
    expect(figure).toHaveTextContent("L2 alarm: cost per run");
    expect(figure).toHaveTextContent("L2 alarm: LLM calls");
    expect(figure).not.toHaveTextContent("cost usd");
  });

  it("point click navigates to run", () => {
    const { container, router } = renderChart();
    const last = changes.points[changes.points.length - 1];
    const { dot, svg } = dotOf(container, last.run_id);
    const x = Number(dot.getAttribute("cx"));

    fireEvent.pointerMove(svg, { clientX: x, clientY: 40 });
    const tip = screen.getByTestId("series-tip");
    expect(tip).toHaveTextContent("5,100");
    expect(tip).toHaveTextContent("halted");
    expect(tip).toHaveTextContent(last.run_id.slice(0, 8));

    fireEvent.click(svg, { clientX: x, clientY: 40 });
    expect(router.state.location.pathname + router.state.location.search).toBe(runHref(last.run_id));
  });

  it("points are reachable from the keyboard", () => {
    const { router } = renderChart();
    const plot = screen.getByRole("group", { name: /Tokens per call/ });
    plot.focus();
    fireEvent.keyDown(plot, { key: "End" });
    expect(screen.getByTestId("series-tip")).toHaveTextContent("5,100");
    fireEvent.keyDown(plot, { key: "ArrowLeft" });
    expect(screen.getByTestId("series-tip")).toHaveTextContent(changes.points[changes.points.length - 2].run_id.slice(0, 8));
    fireEvent.keyDown(plot, { key: "Enter" });
    expect(router.state.location.search).toContain(`id=${changes.points[changes.points.length - 2].run_id}`);
  });

  it("formats ticks by unit", () => {
    expect(tickText("%", 0.5)).toBe("50%");
    expect(tickText("$", 0.0051)).toBe("$0.0051");
    expect(tickText("s", 0.005)).toBe("5 ms");
    expect(tickText("", 2500)).toBe("2.5K");
    expect(tickText("", 12)).toBe("12");
    expect(tickText("chars", 29.4)).toBe("29.4");
    expect(tickText("$", 0)).toBe("0");
    expect(tickText("%", 0)).toBe("0%");
  });

  it("tops the y axis at a round number just above the data", () => {
    expect(niceCeil(5464)).toBe(6000);
    expect(niceCeil(0.0081)).toBe(0.01);
    expect(niceCeil(29.4)).toBe(30);
    expect(niceCeil(5)).toBe(5);
    expect(niceCeil(0)).toBe(1);
  });

  it("places a marker before the run it was first seen in", () => {
    const [model] = changes.markers;
    const slot = markerSlot(model, changes.points);
    expect(changes.points[slot].run_id).toBe(model.run_id);
    // no run id: the first run at or after the marker's time; past every run: after the last slot
    expect(markerSlot({ at: model.at, kind: "l2_alarm" }, changes.points)).toBe(slot);
    expect(markerSlot({ at: Number.MAX_SAFE_INTEGER, kind: "model" }, changes.points)).toBe(changes.points.length);
    expect(markerSlot({ at: 0, kind: "model" }, changes.points)).toBe(0);
  });
});
