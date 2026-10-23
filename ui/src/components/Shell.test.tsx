import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchStub, health } from "../test/fixtures";
import { Shell } from "./Shell";

function renderShell() {
  const router = createMemoryRouter(
    [{ path: "/", element: <Shell />, children: [{ index: true, element: <p>page</p> }] }],
    { initialEntries: ["/"] },
  );
  return render(<RouterProvider router={router} />);
}

describe("Shell", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows the nav and the worker status in words", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/health": health })));
    renderShell();
    expect(screen.getByRole("link", { name: "PitStop" })).toHaveAttribute("href", "/");
    expect(screen.getByRole("link", { name: "Pit wall" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Settings" })).toBeVisible();
    expect(await screen.findByText("Worker off")).toBeVisible();
  });

  it("says when the server is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    renderShell();
    expect(await screen.findByText("Server unreachable")).toBeVisible();
  });
});
