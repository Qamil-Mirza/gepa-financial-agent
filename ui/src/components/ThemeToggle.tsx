// Theme: follow the system, or pin light or dark. The choice lives in localStorage (when storage works) and is
// applied as [data-theme] on <html>, which tokens.css reads.
import { useEffect, useState } from "react";

export type ThemeChoice = "system" | "light" | "dark";

const KEY = "pitstop-theme";
const NEXT: Record<ThemeChoice, ThemeChoice> = { system: "light", light: "dark", dark: "system" };
const LABEL: Record<ThemeChoice, string> = { system: "System", light: "Light", dark: "Dark" };

export function readTheme(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
}

function storeTheme(choice: ThemeChoice): void {
  try {
    if (choice === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, choice);
  } catch {
    // private mode or blocked storage: the choice lasts for this page only
  }
}

function ThemeIcon({ choice }: { choice: ThemeChoice }) {
  return (
    <svg className="theme-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <circle cx="8" cy="8" r="5.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
      {choice === "system" && <path d="M8 2.75a5.25 5.25 0 0 1 0 10.5z" fill="currentColor" />}
      {choice === "dark" && <circle cx="8" cy="8" r="5.25" fill="currentColor" />}
    </svg>
  );
}

export function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>(readTheme);
  useEffect(() => applyTheme(choice), [choice]);

  function cycle() {
    const next = NEXT[choice];
    storeTheme(next);
    setChoice(next);
  }

  return (
    <button type="button" className="btn btn-ghost theme-toggle" onClick={cycle} title={`Switch to ${LABEL[NEXT[choice]].toLowerCase()} theme`}>
      <ThemeIcon choice={choice} />
      <span className="sr-only">Theme: </span>
      {LABEL[choice]}
    </button>
  );
}
