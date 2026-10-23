// One small multiple of the What changed tab: a metric per run, as hand-written SVG (no chart library). The x axis
// is run order, oldest to newest, one slot per run, with the runs' times as its labels: runs arrive in bursts (the
// demo's 95 runs span under a second), and a clock axis would pile them onto one pixel while idle hours took the
// room. Every chart of the stack shares the slots, so the change markers (dashed lines in --change, drawn between
// the run before and the run where the change was first seen) line up down the stack; the first chart carries
// their words, the last the time labels. Points are filled when the run finished ok and hollow (in --p0) when it
// halted, was refused or errored, so status never rides on color alone. The pointer finds the nearest run (a
// crosshair, not a pinpoint); arrow keys do the same from the keyboard; a click or Enter opens the run.
// `SeriesTable` is the text twin: every value, with a link per run.
import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent, type RefObject } from "react";
import { Link, useNavigate } from "react-router";
import type { Changes, Marker, MetricInfo } from "../api/client";
import { formatNumber, plural } from "../format";
import { shortRunId } from "../graph/nodes";
import { runHref } from "../routes";
import { clock, dateTime, duration } from "../time";
import { markerLabel } from "./MarkerLegend";
import { formatUsd } from "./Money";
import { statusWord } from "./RunsTable";

export type Point = Changes["points"][number];

const DEFAULT_WIDTH = 640;
const MIN_WIDTH = 200;
const PLOT_H = 96;
const PAD = { left: 52, right: 14, top: 8 };
const AXIS_H = 20;
const LABEL_ROW = 15;
const LABEL_CHAR = 6.1; // the label font's mean advance, for collision rows
const FAILED = new Set(["halted", "refused", "error", "dropped"]);

export interface SeriesChartProps {
  metric: MetricInfo;
  /** Every point of the tab (oldest first), one slot each; a run without this metric leaves its slot empty. */
  points: Point[];
  markers: Marker[];
  /** Draw the markers' words (the first chart of a stack). */
  labelMarkers?: boolean;
  /** Draw the time labels (the last chart of a stack). */
  xAxis?: boolean;
}

/** The slot boundary a marker sits on: before the run it was first seen in (by id, else the first run at or after
 *  its time), after every run when none matches. */
export function markerSlot(marker: Marker, points: Point[]): number {
  const byId = marker.run_id ? points.findIndex((p) => p.run_id === marker.run_id) : -1;
  if (byId !== -1) return byId;
  const byTime = points.findIndex((p) => p.started_at >= marker.at);
  return byTime === -1 ? points.length : byTime;
}

const trim = (s: string) => s.replace(/\.0$/, "");

/** An axis tick in the metric's unit, compact: "50%", "$0.0051", "5 ms", "2.5K", "12". */
export function tickText(unit: string, v: number): string {
  if (v === 0) return unit === "%" ? "0%" : "0";
  if (unit === "%") {
    const pct = v * 100;
    return `${trim(pct.toFixed(pct > 0 && pct < 10 ? 1 : 0))}%`;
  }
  if (unit === "$") return formatUsd(v);
  if (unit === "s") return duration(v);
  if (Math.abs(v) >= 1000) return `${trim((v / 1000).toFixed(1))}K`;
  return formatNumber(v);
}

/** A value as the tooltip and the table say it: full precision, in the unit. */
export function valueText(unit: string, v: number): string {
  if (unit === "%") return `${trim((v * 100).toFixed(1))}%`;
  if (unit === "$") return formatUsd(v);
  if (unit === "s") return duration(v);
  return formatNumber(v);
}

const NICE = [1, 2, 2.5, 3, 4, 5, 6, 8, 10];

/** The top of the y axis: the smallest 1 / 2 / 2.5 / 3 / 4 / 5 / 6 / 8 x 10^n at or above `max` (1 when there is
 *  nothing to show), so the data fills most of the plot and the halfway tick is still a round number. */
export function niceCeil(max: number): number {
  if (!(max > 0)) return 1;
  const exp = 10 ** Math.floor(Math.log10(max));
  const f = max / exp;
  return (NICE.find((n) => f <= n + 1e-9) ?? 10) * exp;
}

function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w >= MIN_WIDTH) setWidth(Math.round(w));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

interface Placed {
  marker: Marker;
  x: number;
  text: string;
  row: number;
  anchor: "start" | "end";
}

/** Marker labels right of their line (left when there is no room), on the first row where they don't collide. */
function placeLabels(markers: Marker[], metrics: MetricInfo[], xs: number[], left: number, right: number): Placed[] {
  const rows: number[] = []; // the right edge taken on each row
  return markers.map((marker, i) => {
    const px = xs[i];
    const text = markerLabel(marker, metrics);
    const w = text.length * LABEL_CHAR;
    const anchor: Placed["anchor"] = px + 5 + w <= right || px - 5 - w < left ? "start" : "end";
    const start = anchor === "start" ? px + 5 : px - 5 - w;
    let row = rows.findIndex((edge) => start > edge + 10);
    if (row === -1) {
      row = rows.length;
      rows.push(-Infinity);
    }
    rows[row] = Math.max(rows[row], start + w);
    return { marker, x: px, text, row, anchor };
  });
}

type Active = { index: number; via: "pointer" | "key" } | null;

export function SeriesChart({ metric, points, markers, labelMarkers = false, xAxis = false }: SeriesChartProps) {
  const navigate = useNavigate();
  const wrap = useRef<HTMLDivElement>(null);
  const width = useWidth(wrap);
  const id = useId();
  const [active, setActive] = useState<Active>(null);

  const values = points.map((p) => p.metrics[metric.name]).filter((v) => Number.isFinite(v));
  const top = niceCeil(Math.max(0, ...values));
  const left = PAD.left;
  const right = Math.max(width, MIN_WIDTH) - PAD.right;
  const step = (right - left) / Math.max(1, points.length);
  const slotX = (i: number) => left + (i + 0.5) * step; // a run's slot
  const edgeX = (i: number) => left + i * step; // the boundary before slot i
  const markerXs = markers.map((m) => edgeX(markerSlot(m, points)));
  const placed = labelMarkers ? placeLabels(markers, [metric], markerXs, left, right) : [];
  const labelRows = placed.reduce((n, p) => Math.max(n, p.row + 1), 0);
  const plotTop = PAD.top + labelRows * LABEL_ROW;
  const plotBottom = plotTop + PLOT_H;
  const height = plotBottom + (xAxis ? AXIS_H : 10);
  const y = (v: number) => plotBottom - (v / top) * PLOT_H;
  const dots = points.flatMap((p, i) => (Number.isFinite(p.metrics[metric.name]) ? [{ p, cx: slotX(i), cy: y(p.metrics[metric.name]) }] : []));
  const r = Math.min(4, Math.max(2, step * 0.4)); // dense slots get smaller beads so the line still reads
  const path = dots.map((d, i) => `${i === 0 ? "M" : "L"}${d.cx.toFixed(1)} ${d.cy.toFixed(1)}`).join(" ");
  const ticks = [0, top / 2, top];
  const spanS = points.length ? points[points.length - 1].started_at - points[0].started_at : 0;
  const timeText = (t: number) => (spanS < 86400 ? clock(t) : dateTime(t));
  // time labels at four slots; a label that repeats the one before it (a burst of runs) is left out
  const timeTicks = xAxis && points.length ? [0, 1 / 3, 2 / 3, 1].map((f) => Math.round(f * (points.length - 1))) : [];
  const anchors: ("start" | "middle" | "end")[] = ["start", "middle", "middle", "end"];
  const timeLabels = timeTicks
    .map((i, k) => ({ i, text: timeText(points[i].started_at), anchor: anchors[k] }))
    .filter((t, k, all) => k === 0 || t.text !== all[k - 1].text);

  const last = values.length ? values[values.length - 1] : null;
  const min = values.length ? Math.min(...values) : null;
  const max = values.length ? Math.max(...values) : null;
  const unitWord = metric.unit && metric.unit !== "%" && metric.unit !== "$" && metric.unit !== "s" ? ` (${metric.unit})` : "";

  const current = active !== null ? dots[active.index] : undefined;

  const nearest = (clientX: number, el: SVGSVGElement): number | null => {
    if (dots.length === 0) return null;
    const px = clientX - el.getBoundingClientRect().left;
    let best = 0;
    for (let i = 1; i < dots.length; i += 1) if (Math.abs(dots[i].cx - px) < Math.abs(dots[best].cx - px)) best = i;
    return best;
  };
  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const i = nearest(e.clientX, e.currentTarget);
    if (i !== null && (active === null || active.index !== i || active.via !== "pointer")) setActive({ index: i, via: "pointer" });
  };
  const open = (i: number) => navigate(runHref(dots[i].p.run_id));
  const onKeyDown = (e: KeyboardEvent<SVGSVGElement>) => {
    if (dots.length === 0) return;
    const cur = active?.index ?? dots.length; // first arrow press lands on the newest run
    const moves: Record<string, number> = { ArrowLeft: Math.max(0, cur - 1), ArrowRight: Math.min(dots.length - 1, cur === dots.length ? dots.length - 1 : cur + 1), Home: 0, End: dots.length - 1 };
    if (e.key in moves) {
      e.preventDefault();
      setActive({ index: moves[e.key], via: "key" });
    } else if ((e.key === "Enter" || e.key === " ") && active !== null) {
      e.preventDefault();
      open(active.index);
    } else if (e.key === "Escape") {
      setActive(null);
    }
  };

  const tipId = `${id}-tip`;
  const tipLeft = current ? Math.min(Math.max(current.cx, 90), right - 90) : 0;

  return (
    <figure className="series" aria-label={metric.label} ref={wrap}>
      <figcaption className="series-head">
        <span className="series-name">
          {metric.label}
          {unitWord && <span className="muted">{unitWord}</span>}
        </span>
        {last !== null && min !== null && max !== null ? (
          <span className="series-latest">
            <b>{valueText(metric.unit, last)}</b> latest
            <span className="muted"> · {plural(values.length, "run")}</span>
            <span className="sr-only">
              , from {valueText(metric.unit, min)} to {valueText(metric.unit, max)}
            </span>
          </span>
        ) : (
          <span className="muted">no data</span>
        )}
      </figcaption>
      <div className="series-plot">
        <svg
          className="series-svg"
          width={right + PAD.right}
          height={height}
          viewBox={`0 0 ${right + PAD.right} ${height}`} /* 1:1 once measured; scales instead of clipping before that */
          tabIndex={0}
          role="group"
          aria-label={`${metric.label} over time. Arrow keys move between runs, Enter opens the run.`}
          aria-describedby={current ? tipId : undefined}
          onPointerMove={onPointerMove}
          onPointerLeave={() => setActive((a) => (a?.via === "pointer" ? null : a))}
          onBlur={() => setActive((a) => (a?.via === "key" ? null : a))}
          onKeyDown={onKeyDown}
          onClick={(e) => {
            const i = nearest(e.clientX, e.currentTarget);
            if (i !== null) open(i);
          }}
          style={{ cursor: current ? "pointer" : "default" }}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line className="series-grid" x1={left} x2={right} y1={y(t)} y2={y(t)} />
              <text className="series-tick" x={left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">
                {tickText(metric.unit, t)}
              </text>
            </g>
          ))}
          {markerXs.map((mx, i) => (
            <line key={`m${i}`} className="series-marker" x1={mx} x2={mx} y1={PAD.top} y2={plotBottom} />
          ))}
          {placed.map((p, i) => (
            <text key={`l${i}`} className="series-marker-label" x={p.x + (p.anchor === "start" ? 5 : -5)} y={PAD.top + p.row * LABEL_ROW + 10} textAnchor={p.anchor}>
              {p.text}
            </text>
          ))}
          {current && <line className="series-cross" x1={current.cx} x2={current.cx} y1={plotTop} y2={plotBottom} />}
          {path && <path className="series-line" d={path} />}
          {dots.map((d, i) => {
            const failed = FAILED.has(d.p.status ?? "");
            const cls = ["series-dot", failed ? "series-dot-failed" : "", active?.index === i ? "series-dot-on" : ""].filter(Boolean).join(" ");
            return <circle key={d.p.run_id} className={cls} cx={d.cx} cy={d.cy} r={active?.index === i ? r + 1.5 : r} data-run={d.p.run_id} />;
          })}
          {timeLabels.map((t) => (
            <text key={`t${t.i}`} className="series-tick" x={slotX(t.i)} y={plotBottom + 14} textAnchor={t.anchor}>
              {t.text}
            </text>
          ))}
        </svg>
        {current && (
          <div
            id={tipId}
            className={current.cy < plotTop + PLOT_H / 2 ? "series-tip series-tip-below" : "series-tip"}
            data-testid="series-tip"
            style={{ left: tipLeft, top: current.cy }}
          >
            <b>{valueText(metric.unit, current.p.metrics[metric.name])}</b>
            <span className={`run-status run-status-${current.p.status ?? "unknown"}`}>{statusWord(current.p.status)}</span>
            <span className="muted">
              <code>{shortRunId(current.p.run_id)}</code> · {dateTime(current.p.started_at)}
            </span>
          </div>
        )}
        <p className="sr-only" aria-live="polite">
          {current && active?.via === "key"
            ? `Run ${shortRunId(current.p.run_id)}, ${statusWord(current.p.status)}, ${valueText(metric.unit, current.p.metrics[metric.name])}, ${dateTime(current.p.started_at)}`
            : ""}
        </p>
      </div>
    </figure>
  );
}

/** The charts' text twin: one row per run with every metric, and a link to the run. */
export function SeriesTable({ changes }: { changes: Changes }) {
  return (
    <div className="series-table-wrap">
      <table className="series-table" aria-label="Every run's metrics">
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Started</th>
            <th scope="col">Status</th>
            {changes.metrics.map((m) => (
              <th key={m.name} scope="col">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {changes.points.map((p) => (
            <tr key={p.run_id}>
              <td>
                <Link to={runHref(p.run_id)} title={p.run_id}>
                  <code>{shortRunId(p.run_id)}</code>
                </Link>
              </td>
              <td>{dateTime(p.started_at)}</td>
              <td className={`run-status run-status-${p.status ?? "unknown"}`}>{statusWord(p.status)}</td>
              {changes.metrics.map((m) => (
                <td key={m.name} className="num">
                  {Number.isFinite(p.metrics[m.name]) ? valueText(m.unit, p.metrics[m.name]) : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
