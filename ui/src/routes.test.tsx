import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider, useSearchParams } from "react-router";
import { describe, expect, it } from "vitest";
import { runHref, workflowHref, workflowTab } from "./routes";

// Review Focus 3: names with "/", "?", "#", spaces or emoji travel as query parameters and come back intact.
const ODD_NAMES = ["a/b ?#🏁", "a/b", "x?y#z", "spaced name", "résumé 🏁", "100% & more+"];

function ShowName() {
  const [params] = useSearchParams();
  return <output>{`${params.get("name")}|${workflowTab(params.get("tab"))}`}</output>;
}

describe("routes", () => {
  it("encodes workflow names in links", () => {
    for (const name of ODD_NAMES) {
      const href = workflowHref(name, "changes");
      const url = new URL(href, "http://localhost");
      expect(url.pathname).toBe("/w");
      expect(url.hash).toBe("");
      expect(new URLSearchParams(url.search).get("name")).toBe(name);
      expect(url.searchParams.get("tab")).toBe("changes");
    }
    expect(new URL(workflowHref("a/b ?#🏁"), "http://localhost").searchParams.has("tab")).toBe(false);

    // and through the router: the page reads back exactly the name it was linked with
    const router = createMemoryRouter([{ path: "/w", element: <ShowName /> }], {
      initialEntries: [workflowHref("a/b ?#🏁", "runs")],
    });
    render(<RouterProvider router={router} />);
    expect(screen.getByRole("status")).toHaveTextContent("a/b ?#🏁|runs");
  });

  it("encodes run ids and defaults the tab", () => {
    const url = new URL(runHref("r/1?#x"), "http://localhost");
    expect(url.pathname).toBe("/run");
    expect(url.searchParams.get("id")).toBe("r/1?#x");
    expect(workflowTab(null)).toBe("graph");
    expect(workflowTab("bogus")).toBe("graph");
    expect(workflowTab("runs")).toBe("runs");
  });
});
