import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ThemeToggle } from "./ThemeToggle";

describe("ThemeToggle", () => {
  it("cycles system, light and dark, and remembers the choice", () => {
    render(<ThemeToggle />);
    const button = screen.getByRole("button");
    expect(button).toHaveTextContent("System");
    expect(document.documentElement).not.toHaveAttribute("data-theme");

    fireEvent.click(button);
    expect(button).toHaveTextContent("Light");
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(localStorage.getItem("pitstop-theme")).toBe("light");

    fireEvent.click(button);
    expect(button).toHaveTextContent("Dark");
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");

    fireEvent.click(button);
    expect(button).toHaveTextContent("System");
    expect(document.documentElement).not.toHaveAttribute("data-theme");
    expect(localStorage.getItem("pitstop-theme")).toBeNull();
  });

  it("still works when storage throws", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("denied");
    });
    render(<ThemeToggle />);
    fireEvent.click(screen.getByRole("button"));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
  });
});
