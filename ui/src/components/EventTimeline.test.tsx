import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { run } from "../test/fixtures";
import { EventTimeline } from "./EventTimeline";

describe("EventTimeline", () => {
  it("groups consecutive events by agent and says each kind in words", () => {
    render(<EventTimeline events={run.events} startedAt={run.run.started_at} truncated={false} />);
    const groups = screen.getAllByRole("list", { name: /events of/ });
    expect(groups).toHaveLength(2);
    expect(groups[0]).toHaveAccessibleName("8 events of researcher");
    expect(groups[1]).toHaveAccessibleName("6 events of writer");

    const first = within(groups[0]).getAllByRole("listitem");
    expect(first[0]).toHaveTextContent("agent");
    expect(first[0]).toHaveTextContent("top-level agent");
    expect(first[1]).toHaveTextContent("model call");
    expect(first[1]).toHaveTextContent("m-b");
    expect(first[1]).toHaveTextContent("2827 prompt + 233 completion tokens");
    expect(first[1]).toHaveTextContent("$0.0011");
    expect(first[1]).toHaveTextContent("0.5 ms");
    expect(first[2]).toHaveTextContent("tool call");
    expect(first[2]).toHaveTextContent("search_filings");
    expect(first[3]).toHaveTextContent("tool result");

    const writer = within(groups[1]).getAllByRole("listitem");
    const gate = writer.find((li) => /denied: publishes unreleased research/.test(li.textContent ?? ""));
    if (!gate) throw new Error("no gate event");
    expect(within(gate).getByText("gate")).toBeInTheDocument();
    expect(gate).toHaveClass("ev-p0"); // a denied gate stopped something
    const block = writer[writer.length - 1];
    expect(block).toHaveTextContent("blocked");
    expect(block).toHaveTextContent("publish_note");
    expect(block).toHaveClass("ev-p0");
    expect(first[1]).not.toHaveClass("ev-p0"); // a plain model call is quiet
    expect(screen.queryByText(/Showing the first/)).toBeNull();
  });

  it("says when only the first 2,000 events are shown", () => {
    render(<EventTimeline events={run.events} startedAt={run.run.started_at} truncated />);
    expect(screen.getByText("Showing the first 2,000 events")).toBeVisible();
  });

  it("says when a run recorded nothing", () => {
    render(<EventTimeline events={[]} startedAt={null} truncated={false} />);
    expect(screen.getByText(/No events/)).toBeVisible();
  });
});
