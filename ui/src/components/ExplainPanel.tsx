// What likely changed before a drift, inline under its card (never a modal): the attribution summary, the metric
// before and after, and each tag change with its share of runs. Fetched when opened; a drift without a workflow
// (422) or one that is gone (404) says so in words.
import { useEffect, useState } from "react";
import { ApiError, explainDrift, type Explain } from "../api/client";
import { formatNumber, formatPercent, plural, words } from "../format";
import { dateTime } from "../time";

const TAG_LABEL: Record<string, string> = {
  model: "Model",
  base_url: "Endpoint",
  prompt_versions: "Prompt versions",
};

type State = { kind: "loading" } | { kind: "error"; message: string } | { kind: "ready"; explain: Explain };

function message(e: unknown): string {
  if (e instanceof ApiError && e.status === 0) return "Can't reach the PitStop server.";
  return e instanceof Error ? e.message : String(e);
}

export function ExplainPanel({ driftId }: { driftId: number }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setState({ kind: "loading" });
    explainDrift(driftId, { signal: controller.signal })
      .then((explain) => setState({ kind: "ready", explain }))
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setState({ kind: "error", message: message(e) });
      });
    return () => controller.abort();
  }, [driftId]);

  return (
    <section className="explain" aria-label="Explanation">
      {state.kind === "loading" && (
        <p className="muted" role="status">
          Looking at what changed…
        </p>
      )}
      {state.kind === "error" && (
        <p className="explain-error" role="alert">
          Couldn't explain this drift: {state.message}
        </p>
      )}
      {state.kind === "ready" && <Body explain={state.explain} />}
    </section>
  );
}

function Body({ explain }: { explain: Explain }) {
  const summary = explain.summary.charAt(0).toUpperCase() + explain.summary.slice(1);
  return (
    <>
      <p className="explain-summary">{summary}</p>
      {explain.metric && (
        <dl className="explain-metric">
          <dt>{words(explain.metric)}</dt>
          <dd>
            <b>
              {formatNumber(explain.before_mean)} → {formatNumber(explain.after_mean)}
            </b>
            <span className="muted">
              {" "}
              · {plural(explain.n_before, "run")} before · {explain.n_after} after
            </span>
          </dd>
        </dl>
      )}
      {explain.changes.length > 0 && (
        <ul className="explain-changes">
          {explain.changes.map((c, i) => (
            <li key={`${c.tag}-${i}`}>
              <span className="chip chip-change">{TAG_LABEL[c.tag] ?? words(c.tag)}</span>
              <b className="explain-change">
                {c.before ?? "(unset)"} → {c.after}
              </b>
              <span className="muted">
                {formatPercent(c.share_before)} → {formatPercent(c.share_after)} of runs
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="explain-at muted">Changed {dateTime(explain.at)}</p>
    </>
  );
}
