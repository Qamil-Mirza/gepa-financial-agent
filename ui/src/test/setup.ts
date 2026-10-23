// Vitest setup: jest-dom matchers and a DOM cleanup after each test (globals are off, so RTL can't register it).
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  try {
    window.localStorage.clear();
  } catch {
    // a test may have made storage throw
  }
  document.documentElement.removeAttribute("data-theme");
});
