// The What changed tab's list of markers, in time order: what changed (model, endpoint, prompt versions, or an
// L2 alarm on a metric), when, for which source's runs, and the run it was first seen in. An L2 alarm opens its
// attribution inline (ExplainPanel), never in a modal.
import { useId, useState } from "react";
import { Link } from "react-router";
import type { Marker, MetricInfo } from "../api/client";
import { words } from "../format";
import { shortRunId } from "../graph/nodes";
import { runHref } from "../routes";
import { dateTime } from "../time";
import { ExplainPanel } from "./ExplainPanel";

const KIND_WORD: Record<Marker["kind"], string> = {
  model: "Model",
  base_url: "Endpoint",
  prompt_versions: "Prompt versions",
  l2_alarm: "L2 alarm",
};

const KIND_SHORT: Record<Marker["kind"], string> = { model: "model", base_url: "endpoint", prompt_versions: "prompts", l2_alarm: "L2 alarm" };

const SOURCE_WORD: Record<string, string> = { sdk: "SDK runs", otel: "OTel runs" };

/** The metric's label in lower case ("tokens per call"), from the tab's metric list or the identifier's words. A
 *  label that starts with an acronym keeps it ("LLM calls"). */
export function metricWords(name: string, metrics: MetricInfo[]): string {
  const label = metrics.find((m) => m.name === name)?.label ?? words(name);
  return /^.\p{Ll}/u.test(label) ? label.charAt(0).toLowerCase() + label.slice(1) : label;
}

/** "model m-a → m-b", "endpoint (unset) → https://x", "prompts a=1 → a=2", "L2 alarm: tokens per call". */
export function markerLabel(marker: Marker, metrics: MetricInfo[]): string {
  if (marker.kind === "l2_alarm") return `L2 alarm: ${metricWords(marker.metric ?? "metric", metrics)}`;
  return `${KIND_SHORT[marker.kind]} ${marker.before ?? "(unset)"} → ${marker.after ?? "(unset)"}`;
}

function Item({ marker, metrics }: { marker: Marker; metrics: MetricInfo[] }) {
  const [explaining, setExplaining] = useState(false);
  const id = useId();
  const alarm = marker.kind === "l2_alarm";
  const change = alarm
    ? (metrics.find((m) => m.name === marker.metric)?.label ?? words(marker.metric ?? "metric"))
    : `${marker.before ?? "(unset)"} → ${marker.after ?? "(unset)"}`;
  const source = marker.source ? SOURCE_WORD[marker.source] : null;
  return (
    <li className="marker">
      <p className="marker-line">
        <span className="chip chip-change">{KIND_WORD[marker.kind]}</span>
        <b className="marker-change">{change}</b>
      </p>
      <p className="marker-meta">
        <span>{dateTime(marker.at)}</span>
        {source && <span>{source}</span>}
        {marker.run_id && (
          <Link to={runHref(marker.run_id)} title={marker.run_id}>
            run <code>{shortRunId(marker.run_id)}</code>
          </Link>
        )}
        {alarm && marker.drift_id != null && (
          <button
            type="button"
            className="btn btn-ghost marker-explain"
            aria-expanded={explaining}
            aria-controls={id}
            onClick={() => setExplaining((v) => !v)}
          >
            {explaining ? "Hide explanation" : "Explain"}
          </button>
        )}
      </p>
      <div id={id}>{explaining && marker.drift_id != null && <ExplainPanel driftId={marker.drift_id} />}</div>
    </li>
  );
}

export function MarkerLegend({ markers, metrics }: { markers: Marker[]; metrics: MetricInfo[] }) {
  if (markers.length === 0) {
    return <p className="muted marker-none">No model, endpoint or prompt change, and no L2 alarm, across these runs.</p>;
  }
  return (
    <ol className="markers" aria-label="Changes">
      {markers.map((m, i) => (
        <Item key={`${m.kind}-${m.at}-${m.drift_id ?? i}`} marker={m} metrics={metrics} />
      ))}
    </ol>
  );
}
