import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Level } from "./Level";

describe("Level", () => {
  it("Level always renders its text", () => {
    render(
      <>
        <Level level="P0" />
        <Level level="P1" count={2} />
        <Level level="P2" />
        <Level level="P3" />
        <Level level={null} />
      </>,
    );
    for (const level of ["P0", "P2", "P3"]) {
      expect(screen.getByText(level)).toBeVisible();
    }
    // with a count, the level stays readable and the count is spelled out for screen readers
    expect(screen.getByText("P1")).toBeVisible();
    expect(screen.getByText("P1").closest(".level")).toHaveTextContent("P1: 2 open");
    // an unknown level still says so in words, never color alone
    expect(screen.getByText("P?")).toBeVisible();
  });
});
