// The run graph's window: 1h / 24h / 7d / 30d as a segmented group of real radio buttons, so it reads and
// keyboards like the choice it is. Disabled while a single run is the scope.
import { useId } from "react";
import type { GraphWindow } from "../api/client";

export const WINDOWS: readonly GraphWindow[] = ["1h", "24h", "7d", "30d"];
export const DEFAULT_WINDOW: GraphWindow = "24h";

/** The `window` query value as a known window; anything else is the default. */
export function parseWindow(value: string | null): GraphWindow {
  return (WINDOWS as readonly string[]).includes(value ?? "") ? (value as GraphWindow) : DEFAULT_WINDOW;
}

export const WINDOW_LABEL: Record<GraphWindow, string> = {
  "1h": "last hour",
  "24h": "last 24 hours",
  "7d": "last 7 days",
  "30d": "last 30 days",
};

/** The window as words, for whatever the API echoes back ("24h" → "last 24 hours"). */
export function windowLabel(value: string | null | undefined): string {
  return WINDOW_LABEL[parseWindow(value ?? null)] ?? (value ?? "window");
}

export interface WindowPickerProps {
  value: GraphWindow;
  onChange: (window: GraphWindow) => void;
  disabled?: boolean;
}

export function WindowPicker({ value, onChange, disabled }: WindowPickerProps) {
  const group = useId();
  return (
    <fieldset className="winpick" disabled={disabled}>
      <legend className="sr-only">Window</legend>
      {WINDOWS.map((w) => (
        <label key={w} className="winpick-opt">
          <input
            type="radio"
            name={group}
            value={w}
            checked={value === w}
            onChange={() => onChange(w)}
            className="sr-only"
          />
          <span className="winpick-text">{w}</span>
        </label>
      ))}
    </fieldset>
  );
}
