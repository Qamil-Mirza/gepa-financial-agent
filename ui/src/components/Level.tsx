// A P-level chip: always the text P0…P3 (urgency is never color alone), tinted with that level's color.
const LEVELS = new Set(["P0", "P1", "P2", "P3"]);

export interface LevelProps {
  level: string | null | undefined;
  /** Open drifts at this level; shown after the level and read out as "P1: 2 open". */
  count?: number;
}

export function Level({ level, count }: LevelProps) {
  const known = level && LEVELS.has(level) ? level : null;
  return (
    <span className={`level level-${known ? known.toLowerCase() : "unknown"}`} title={known ? undefined : "Unknown level"}>
      <span className="level-name">{known ?? "P?"}</span>
      {count !== undefined && (
        <b className="level-count">
          <span className="sr-only">: </span>
          {count}
          <span className="sr-only"> open</span>
        </b>
      )}
    </span>
  );
}
