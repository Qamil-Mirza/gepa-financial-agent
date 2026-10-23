import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { formatUsd, Money } from "./Money";

describe("Money", () => {
  it("Money formats", () => {
    expect(formatUsd(1)).toBe("$1.00");
    expect(formatUsd(8.4)).toBe("$8.40");
    expect(formatUsd(1234.5)).toBe("$1,234.50");
    expect(formatUsd(0.6572427)).toBe("$0.6572");
    expect(formatUsd(0.0330624)).toBe("$0.0331");
    expect(formatUsd(0.99999)).toBe("$1.00"); // rounds up to a dollar: shown as dollars
    expect(formatUsd(0)).toBe("$0.0000");
    expect(formatUsd(null)).toBe("—");
    expect(formatUsd(undefined)).toBe("—");

    render(
      <>
        <Money usd={2} />
        <Money usd={null} />
      </>,
    );
    expect(screen.getByText("$2.00")).toBeVisible();
    expect(screen.getByText("—")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("unknown")).toBeInTheDocument(); // what a screen reader hears
  });
});
