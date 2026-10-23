import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("EmptyState shows the setup snippet", () => {
    const { container } = render(<EmptyState />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("No workflows yet");
    expect(screen.getByText("uv sync")).toBeVisible();
    const published = screen.getByText('pip install "pitstop[ui]"');
    expect(published.parentElement).toHaveTextContent('pip install "pitstop[ui]" once it is published.');
    expect(screen.getByText("pitstop ui")).toBeVisible();
    expect(
      screen.getByText("python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo"),
    ).toBeVisible();

    const snippet = container.querySelector("pre[data-snippet='wrap']")?.textContent ?? "";
    expect(snippet.split("\n")).toHaveLength(6);
    expect(snippet).toContain("client = pitstop.wrap(OpenAI())");
    expect(snippet).toContain('with pitstop.workflow("my-agent", criticality="medium"):');
  });
});
