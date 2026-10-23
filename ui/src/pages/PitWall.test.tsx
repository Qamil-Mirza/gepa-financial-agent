import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { boardEmpty, fetchStub } from "../test/fixtures";
import { PitWall } from "./PitWall";

describe("PitWall", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows the empty state when the board has no workflows", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/board": boardEmpty })));
    const router = createMemoryRouter([{ path: "/", element: <PitWall /> }]);
    render(<RouterProvider router={router} />);
    expect(await screen.findByRole("heading", { name: "No workflows yet" })).toBeVisible();
  });
});
