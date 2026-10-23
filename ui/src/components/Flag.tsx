// A pit-wall flag: the flag glyph in the flag's color plus a text label ("Halted", "Degraded", "Healthy"), so the
// state reads in grayscale too.
export type FlagColor = "red" | "yellow" | "green";

export function FlagGlyph() {
  return (
    <svg className="flag-glyph" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M3 1.5v13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M3.8 2.2h8.4l-2 3 2 3H3.8z" fill="currentColor" />
    </svg>
  );
}

export function Flag({ flag, label }: { flag: FlagColor; label: string }) {
  return (
    <span className={`flag flag-${flag}`}>
      <FlagGlyph />
      <span className="sr-only">{`${flag} flag: `}</span>
      {label}
    </span>
  );
}
