"""What changed (decision 10): the L2 per-run metrics of a workflow's newest runs, and the markers to draw on them.

Points are `run_metrics` of the newest `limit` (at most 200) runs that have metrics (refused and dropped runs have
none), oldest first. Markers are inside the points' time range:

- `model`, `base_url` and `prompt_versions` markers: the run's tag differs from the previous run of its workflow and
  source (OTel runs are compared only with OTel runs, SDK runs only with SDK runs, as the feed does). A run redirected
  to a fallback, or without the tag, is skipped.
- `l2_alarm` markers: every L2 drift of the workflow (its newest `limit`), at the `change_at` of its latest evidence.

Queries: one over `idx_runs_wf_started`, `run_metrics` per run (its events read over `idx_events_run`), one on
drifts."""
from __future__ import annotations

import json

from ..detect.attribution import _tag_value
from ..detect.metrics import METRICS, run_metrics
from .models import Changes, Marker, MetricInfo, Point
from .views import _NUMERIC, _num, _str, _tag_text

CHANGES_MAX = 200
_TAGS = ("model", "base_url", "prompt_versions")
_METRIC_INFO = {
    "tool_error_rate": ("Tool errors", "%"), "refusal_rate": ("Refusals", "%"), "cost_usd": ("Cost per run", "$"),
    "tokens_per_call": ("Tokens per call", ""), "llm_calls": ("LLM calls", ""), "duration_s": ("Duration", "s"),
    "output_chars": ("Output length", "chars"),
}


def _tag_markers(runs: list) -> list[Marker]:
    """Tag-change markers over `runs` (oldest first)."""
    last: dict[tuple, str] = {}
    markers = []
    for r in runs:
        if r["fallback_from"] is not None:
            continue
        source = "otel" if r["adapter"] == "otel" else "sdk"
        for tag in _TAGS:
            value = _tag_value(tag, r[tag])
            if value is None:
                continue
            before = last.get((source, tag))
            last[(source, tag)] = value
            if before is not None and before != value:
                markers.append(Marker(at=_num(r["started_at"]), kind=tag, before=_tag_text(tag, before),
                                      after=_tag_text(tag, value), run_id=str(r["id"]), source=source))
    return markers


def _alarm_markers(conn, workflow: str, since: float, limit: int) -> list[Marker]:
    markers = []
    for d in conn.execute("SELECT id, run_id, evidence FROM drifts WHERE workflow = ? AND evidence LIKE '%change_at%'"
                          " ORDER BY ts DESC, id DESC LIMIT ?", (workflow, limit)):
        try:
            entry = json.loads(d["evidence"])[-1]
            detail = entry["detail"]
            at, metric = _num(detail["change_at"]), detail.get("metric")
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if at is None or at < since or not isinstance(metric, str):
            continue
        run_id = entry.get("run_id") if isinstance(entry, dict) else None
        markers.append(Marker(at=at, kind="l2_alarm", run_id=_str(run_id) or _str(d["run_id"]), drift_id=d["id"],
                              metric=metric))
    return markers


def change_series(store, workflow: str, *, limit: int = CHANGES_MAX) -> Changes:
    """The series and markers of `workflow` (see the module docstring); at most `limit` (capped at 200) points."""
    limit = max(1, min(int(limit), CHANGES_MAX))
    conn = store.connect()
    rows = conn.execute(
        "SELECT id, started_at, status, adapter, fallback_from, model, base_url, prompt_versions FROM runs"
        f" WHERE workflow = ? AND {_NUMERIC} AND (status IS NULL OR status NOT IN ('refused', 'dropped'))"
        " ORDER BY started_at DESC, rowid DESC LIMIT ?", (workflow, limit)).fetchall()
    runs, points = [], []
    for r in reversed(rows):  # oldest first
        metrics = run_metrics(store, r["id"])
        if not metrics:
            continue
        runs.append(r)
        points.append(Point(run_id=str(r["id"]), started_at=float(r["started_at"]), status=_str(r["status"]),
                            source="otel" if r["adapter"] == "otel" else "sdk", metrics=metrics))
    markers = _tag_markers(runs)
    if points:
        markers += _alarm_markers(conn, workflow, points[0].started_at, limit)
    markers.sort(key=lambda m: m.at)
    metrics = [MetricInfo(name=n, label=_METRIC_INFO[n][0], unit=_METRIC_INFO[n][1]) for n in METRICS
               if n in _METRIC_INFO]
    return Changes(points=points, markers=markers, metrics=metrics)
