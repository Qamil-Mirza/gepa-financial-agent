import { fireEvent, render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { runHref } from "../routes";
import { runs } from "../test/fixtures";
import { RunsTable } from "./RunsTable";

function renderTable(props: Partial<Parameters<typeof RunsTable>[0]> = {}) {
  const onLoadMore = vi.fn();
  const router = createMemoryRouter(
    [
      {
        path: "/w",
        element: <RunsTable runs={runs.runs} hasMore moreState={{ kind: "idle" }} onLoadMore={onLoadMore} {...props} />,
      },
      { path: "/run", element: <p>run page</p> },
    ],
    { initialEntries: ["/w"] },
  );
  const view = render(<RouterProvider router={router} />);
  return { router, onLoadMore, ...view };
}

describe("RunsTable", () => {
  it("shows each run's facts and links every row to run detail", () => {
    const { router } = renderTable();
    const table = screen.getByRole("table", { name: "Runs" });
    const rows = within(table).getAllByRole("row").slice(1); // minus the header
    expect(rows).toHaveLength(runs.runs.length);
    const first = rows[0];
    expect(within(first).getByRole("link", { name: /a5811371/ })).toHaveAttribute("href", runHref(runs.runs[0].id));
    expect(first).toHaveTextContent("halted");
    expect(first).toHaveTextContent("m-b");
    expect(first).toHaveTextContent("wrap");
    expect(first).toHaveTextContent("$0.0081");
    expect(within(first).getByRole("cell", { name: "4" })).toBeInTheDocument(); // LLM calls
    expect(within(first).getByRole("cell", { name: "2 verdicts" })).toBeInTheDocument();
    // the whole row is the hit area
    fireEvent.click(within(first).getByText("wrap"));
    expect(router.state.location.pathname + router.state.location.search).toBe(runHref(runs.runs[0].id));
  });

  it("asks for more with its button and says when it is loading or failed", () => {
    const { onLoadMore } = renderTable();
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("hides the button once every run is shown", () => {
    renderTable({ hasMore: false });
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
  });

  it("shows the loading and failed states of the next page", () => {
    const { rerender } = renderTable({ moreState: { kind: "loading" } });
    expect(screen.getByRole("button", { name: /Loading/ })).toBeDisabled();
    rerender(
      <RouterProvider
        router={createMemoryRouter(
          [
            {
              path: "/w",
              element: (
                <RunsTable runs={runs.runs} hasMore moreState={{ kind: "error", message: "store busy, retry" }} onLoadMore={() => {}} />
              ),
            },
          ],
          { initialEntries: ["/w"] },
        )}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("store busy, retry");
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  });
});
