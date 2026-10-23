"""The dashboard's read models: pure functions over the store, each taking `(rt, ...)` and returning the response
model its endpoint (`routes.py`) sends. They only read, except `dismiss`.

Malformed rows never break a view (Review Focus 1): a value of the wrong type (a NULL or text timestamp, an unknown
status, a payload that is not JSON, evidence that is not a list) is shown as null or "unknown", and a row that
cannot be shown at all is skipped. Store errors (`sqlite3.Error`, e.g. a locked store) are never swallowed: the
app's guard turns a busy store into 503.

Lists are bounded: the feed at FEED_MAX items, a runs page at RUNS_MAX runs, a workflow's drift cards at
DRIFT_CARDS_MAX, a run's events at MAX_EVENTS. Secrets never leave: settings name env vars and say whether each is
set, never its value, and URLs are shown without credentials, query or fragment: run base_urls in runs and in the
feed, and every URL in text built from them (alert causes, attribution summaries and tag changes)."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import fields
from urllib.parse import urlsplit, urlunsplit

from ..alerts.model import build_alert
from ..cli_report import _day_start, budget_rows, drift_rows, explain_target, workflow_names, workflow_row
from ..detect.attribution import attribute
from ..detect.drifts import dismiss_drift
from ..sdk import fallback
from ..userconfig import _CHANNEL_SPECS, _SECRET_ENV_KEY, ALERT_CHANNELS
from .models import (AlertSettings, Board, BudgetRow, ChannelSettings, DriftCard, EventRow, Explain, FeedItem,
                     JudgeSettings, LevelCounts, RunDetail, RunPage, RunRef, RunRow, Settings, TagChange, Tile,
                     WorkerStatus, WorkflowDetail, WorkflowSettings)

log = logging.getLogger("pitstop.server")

FEED_MAX = 100
FEED_RUN_WINDOW = 200  # the newest runs the feed reads for halts, fallbacks, refusals and model changes
RUNS_MAX = 200
DRIFT_CARDS_MAX = 50
MAX_EVENTS = 2000
SUMMARY_MAX = 300
ADAPTER_WINDOW = 200  # a tile's adapters are those of the workflow's newest runs
_LEVELS = ("P0", "P1", "P2", "P3")
_FLAG_ORDER = {"red": 0, "yellow": 1, "green": 2}
_FLAG_LABELS = {"red": "Halted", "yellow": "Degraded", "green": "Healthy"}
_RED_STATUSES = ("halted", "refused")
_VERDICTS = ("block", "steer", "halt", "fallback")
_NUMERIC = "typeof(started_at) IN ('integer', 'real')"


class NotFound(LookupError):
    """The workflow, run or drift does not exist (404)."""


class Unprocessable(ValueError):
    """The drift exists but cannot be explained: it has no workflow or no time (422)."""


# -- coercion: a malformed value becomes None, never an exception ---------------------------------------------------

def _num(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _int(v: object) -> int:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else 0


def _str(v: object) -> str | None:
    return None if v is None else str(v)


def _payload(text: object) -> dict:
    try:
        p = json.loads(text) if isinstance(text, (str, bytes)) else {}
    except ValueError:
        return {}
    return p if isinstance(p, dict) else {}


def _one_line(text: str, limit: int = SUMMARY_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _text(v: object) -> str:
    """A payload value as text: strings as they are, anything else as compact JSON."""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, default=str)
    except (TypeError, ValueError):
        return "?"


def redact_url(url: object) -> str | None:
    """`url` without credentials, query or fragment (they may hold secrets); None when it is not a string."""
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port is not None else host
    except ValueError:  # e.g. an unclosed IPv6 bracket or a bad port
        return "(unreadable URL)"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s()<>\"']+")
_URL_TAIL = ".,;:!?"


def redact_text(text: str) -> str:
    """`text` with every URL in it passed through `redact_url`; sentence punctuation right after a URL is kept.
    For text built from stored base_urls (alert causes, attribution summaries)."""
    def redact(m: re.Match) -> str:
        url = m.group(0)
        stripped = url.rstrip(_URL_TAIL)
        return (redact_url(stripped) or "") + url[len(stripped):]

    return _URL.sub(redact, text)


def _alert_text(store, drift: dict) -> tuple[str, str, str]:
    """(title, cause, action) of the drift's alert, URLs redacted (a cause names stored base_urls). A drift whose
    alert cannot be built (a malformed row) gets a plain title and says its cause is unknown."""
    try:
        a = build_alert(store, drift["id"])
        return a.title, redact_text(a.cause), redact_text(a.action)
    except sqlite3.Error:
        raise
    except Exception as e:
        log.debug("pitstop: alert text for drift %s failed (%s)", drift["id"], type(e).__name__)
        workflow = drift.get("workflow")
        title = f"[{drift.get('level')}] {workflow if workflow is not None else '(no workflow)'}: " \
                f"{drift.get('drift_type')}"
        return title, "Unknown: the drift's record could not be read.", f"Action: {drift.get('action')}."


# -- pit wall --------------------------------------------------------------------------------------------------------

def _latest_run(conn, workflow) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE workflow=? ORDER BY started_at DESC, rowid DESC LIMIT 1",
                        (workflow,)).fetchone()


def _workflow_state(rt, workflow, day_start: float) -> tuple[Tile, tuple, sqlite3.Row, list[dict]] | None:
    """(tile, sort key, latest run, open drifts in `drift_rows` order) for a workflow; None when it has no run."""
    store = rt.store
    row = workflow_row(store, workflow, day_start)
    latest = _latest_run(store.connect(), workflow)
    if row is None or latest is None:
        return None
    drifts = drift_rows(store, workflow=workflow)
    levels = [d["level"] for d in drifts]
    if "P0" in levels or latest["status"] in _RED_STATUSES:
        flag = "red"
    else:
        flag = "yellow" if drifts else "green"
    headline = _alert_text(store, drifts[0])[0] if drifts else None
    adapters = sorted({str(r[0]) for r in store.connect().execute(
        "SELECT adapter FROM runs WHERE workflow=? ORDER BY started_at DESC LIMIT ?", (workflow, ADAPTER_WINDOW))
        if r[0] is not None})
    tile = Tile(
        name=str(workflow), criticality=_str(row["criticality"]), flag=flag, flag_label=_FLAG_LABELS[flag],
        open=LevelCounts(P0=row["open_p0"], P1=row["open_p1"], P2=row["open_p2"], P3=row["open_p3"]),
        headline=headline, headline_level=_str(drifts[0]["level"]) if drifts else None,
        last_run=RunRef(id=str(latest["id"]), status=_str(latest["status"]), started_at=_num(latest["started_at"]),
                        ended_at=_num(latest["ended_at"])),
        runs_today=_int(row["runs_today"]), spend_today_usd=_num(row["spend_today_usd"]) or 0.0,
        budget_usd=_num(row["budget_usd"]), adapters=adapters,
        fallback_active=fallback.active(store, workflow) is not None)
    urgency = min((_LEVELS.index(lv) if lv in _LEVELS else len(_LEVELS) for lv in levels), default=len(_LEVELS) + 1)
    times = [_num(latest["started_at"]), _num(latest["ended_at"]), *(_num(d["last_seen"]) for d in drifts)]
    activity = max((t for t in times if t is not None), default=float("-inf"))
    return tile, (_FLAG_ORDER[flag], urgency, -activity), latest, drifts


def board(rt, *, now: float | None = None) -> Board:
    """Every workflow's tile, red first (decision 6), and the global budgets."""
    now = time.time() if now is None else now
    start = _day_start(now, rt.ledger.tz)
    states = [s for wf in workflow_names(rt.store) if (s := _workflow_state(rt, wf, start)) is not None]
    states.sort(key=lambda s: s[1])  # stable: ties keep workflow_names' most-recently-started order
    return Board(workflows=[s[0] for s in states], budgets=_budgets(rt), now=now)


def _budgets(rt) -> list[BudgetRow]:
    return [BudgetRow(name=r["scope"].partition(":")[2], **r) for r in budget_rows(rt.ledger, rt.config)]


def feed(rt, *, limit: int = 50) -> list[FeedItem]:
    """The pit wall's feed, newest first, at most `limit` (capped at FEED_MAX) items: P0-P2 drifts by when they
    opened (whatever their status now; one without a numeric time is skipped), and from the newest FEED_RUN_WINDOW
    runs their `halt` and `fallback` events, refused runs and model changes. A model change is a run whose
    (model, base_url) differs from the previous run of its workflow and source (OTel or SDK, decision 10) that has a
    model and was not redirected to a fallback; the oldest run of each in the window has nothing to compare with.
    The run window keeps every query index-backed on a large store. When the window is full, every item (drifts
    included) older than its oldest run is left out, so the feed may be shorter but never skips an item newer than
    one it shows."""
    limit = max(1, min(int(limit), FEED_MAX))
    conn = rt.store.connect()
    items: list[FeedItem] = []
    for d in conn.execute("SELECT id, ts, workflow, level, drift_type, run_id FROM drifts WHERE level IN ('P0', 'P1',"
                          " 'P2') AND typeof(ts) IN ('integer', 'real') ORDER BY ts DESC, id DESC LIMIT ?", (limit,)):
        items.append(FeedItem(id=f"drift:{d['id']}", ts=_num(d["ts"]), kind="drift", workflow=_str(d["workflow"]),
                              level=d["level"], text=f"Drift opened: {d['drift_type'] or 'unknown'}",
                              run_id=_str(d["run_id"]), drift_id=d["id"]))
    runs = conn.execute("SELECT id, workflow, started_at, status, model, base_url, adapter, fallback_from FROM runs"
                        f" WHERE {_NUMERIC} ORDER BY started_at DESC, rowid DESC LIMIT ?",
                        (FEED_RUN_WINDOW,)).fetchall()
    workflow_of = {r["id"]: r["workflow"] for r in runs}
    previous: dict[tuple, sqlite3.Row] = {}
    for r in reversed(runs):  # oldest first
        if r["status"] == "refused":
            items.append(FeedItem(id=f"refused:{r['id']}", ts=_num(r["started_at"]), kind="refused",
                                  workflow=_str(r["workflow"]), level=None, run_id=_str(r["id"]), drift_id=None,
                                  text="Run refused at its start: a budget is used up"))
        if r["workflow"] is None or r["model"] is None or r["fallback_from"] is not None:
            continue
        key = (r["workflow"], "otel" if r["adapter"] == "otel" else "sdk")
        prev = previous.get(key)
        previous[key] = r
        if prev is None or (prev["model"], prev["base_url"]) == (r["model"], r["base_url"]):
            continue
        if prev["model"] != r["model"]:
            text = f"Model changed: {prev['model']} → {r['model']}"
        else:
            text = f"Provider changed: {redact_url(prev['base_url'])} → {redact_url(r['base_url'])}"
        items.append(FeedItem(id=f"model_change:{r['id']}", ts=_num(r["started_at"]), kind="model_change",
                              workflow=_str(r["workflow"]), level=None, text=_one_line(text), run_id=_str(r["id"]),
                              drift_id=None))
    if runs:
        marks = ",".join("?" * len(runs))
        for e in conn.execute(f"SELECT id, run_id, ts, kind, name, payload FROM events WHERE run_id IN ({marks})"
                              " AND kind IN ('halt', 'fallback')", [r["id"] for r in runs]):
            p = _payload(e["payload"])
            if e["kind"] == "halt":
                reason = p.get("reason")
                text = f"Halted: {_text(reason)}" if reason is not None else "Halted"
            else:
                text = f"Fell back from {_text(p.get('from', 'unknown'))} to {_text(p.get('to', 'unknown'))}"
            items.append(FeedItem(id=f"{e['kind']}:{e['id']}", ts=_num(e["ts"]), kind=e["kind"],
                                  workflow=_str(workflow_of.get(e["run_id"])), level=None, text=_one_line(text),
                                  run_id=_str(e["run_id"]), drift_id=None))
    if len(runs) == FEED_RUN_WINDOW:  # older runs were not read: drop everything older than the window too
        oldest = _num(runs[-1]["started_at"])
        items = [i for i in items if i.ts is not None and oldest is not None and i.ts >= oldest]
    items.sort(key=lambda i: i.id)  # a stable tie-break for equal times
    items.sort(key=lambda i: i.ts if i.ts is not None else float("-inf"), reverse=True)
    return items[:limit]


# -- workflow page ---------------------------------------------------------------------------------------------------

def workflow_detail(rt, name: str, *, now: float | None = None) -> WorkflowDetail:
    """The workflow page: its tile, its latest run's goal and settings, and its open drifts with their alert text
    (most urgent first, at most DRIFT_CARDS_MAX). NotFound when the workflow has no run."""
    now = time.time() if now is None else now
    state = _workflow_state(rt, name, _day_start(now, rt.ledger.tz))
    if state is None:
        raise NotFound(f"no workflow {name!r}")
    tile, _, latest, drifts = state
    cards = []
    for d in drifts[:DRIFT_CARDS_MAX]:
        title, cause, doing = _alert_text(rt.store, d)
        cards.append(DriftCard(id=d["id"], level=_str(d["level"]), drift_type=_str(d["drift_type"]),
                               action=_str(d["action"]), occurrences=_int(d["count"]),
                               first_seen=_num(d["first_seen"]), last_seen=_num(d["last_seen"]), title=title,
                               cause=cause, doing=doing))
    return WorkflowDetail(tile=tile, goal=_str(latest["goal"]), sensitivity=_str(latest["sensitivity"]),
                          fix_mode=_str(latest["fix_mode"]), drifts=cards)


def dismiss(rt, drift_id: int) -> str:
    """"dismissed", "not_found" or "not_open" (`detect.drifts.dismiss_drift`, one BEGIN IMMEDIATE)."""
    return dismiss_drift(rt.store, drift_id)


def explain(rt, drift_id: int) -> Explain:
    """L5 attribution around the drift's change point, as `pitstop explain` prints it. NotFound for an unknown
    drift; Unprocessable for one without a workflow or a time."""
    target = explain_target(rt.store, drift_id)
    if target is None:
        raise NotFound(f"no drift {drift_id}")
    if target["workflow"] is None:
        raise Unprocessable(f"drift {drift_id} has no workflow")
    if target["at"] is None:
        raise Unprocessable(f"drift {drift_id} has no time to compare around")
    a = attribute(rt.store, target["workflow"], target["at"], metric=target["metric"])
    return Explain(summary=redact_text(a.summary), metric=a.metric, before_mean=_num(a.before_mean),
                   after_mean=_num(a.after_mean), n_before=a.n_before, n_after=a.n_after, at=a.at,
                   changes=[TagChange(tag=c.tag, before=_tag_text(c.tag, c.before),
                                      after=_tag_text(c.tag, c.after) or "", share_before=c.share_before,
                                      share_after=c.share_after) for c in a.changes])


def _tag_text(tag: str, value: str | None) -> str | None:
    """A run tag's value for the client: a base_url without credentials, query or fragment."""
    return redact_url(value) if tag == "base_url" and value is not None else value


# -- runs ------------------------------------------------------------------------------------------------------------

def _run_stats(conn, run_ids: list) -> dict:
    """run id -> (cost_usd, llm_calls, tool_calls, verdicts), read over `idx_events_run`. Agent calls are `llm_call`
    events whose payload's `meter` is "agent" (as in `run_metrics`); a payload that is not JSON is no agent call."""
    if not run_ids:
        return {}
    stats = {run_id: [0.0, 0, 0, 0] for run_id in run_ids}
    marks = ",".join("?" * len(run_ids))
    for r in conn.execute(f"SELECT run_id, SUM(kind = 'tool_call'), SUM(kind IN (?, ?, ?, ?)) FROM events"
                          f" WHERE run_id IN ({marks}) GROUP BY run_id", (*_VERDICTS, *run_ids)):
        stats[r[0]][2:] = [_int(r[1]), _int(r[2])]
    for r in conn.execute(f"SELECT run_id, payload, cost_usd FROM events WHERE run_id IN ({marks})"
                          " AND kind = 'llm_call'", run_ids):
        if _payload(r["payload"]).get("meter") == "agent":
            s = stats[r["run_id"]]
            s[0] += _num(r["cost_usd"]) or 0.0
            s[1] += 1
    return {run_id: tuple(s) for run_id, s in stats.items()}


def _run_row(r: sqlite3.Row, stats: tuple | None) -> RunRow:
    cost, llm_calls, tool_calls, verdicts = stats or (0.0, 0, 0, 0)
    started, ended = _num(r["started_at"]), _num(r["ended_at"])
    return RunRow(id=str(r["id"]), status=_str(r["status"]), started_at=started, ended_at=ended,
                  duration_s=ended - started if started is not None and ended is not None and ended >= started
                  else None,
                  model=_str(r["model"]), base_url=redact_url(r["base_url"]), adapter=_str(r["adapter"]),
                  cost_usd=cost, llm_calls=llm_calls, tool_calls=tool_calls, verdicts=verdicts)


def runs_page(rt, workflow: str | None = None, *, limit: int = 50, before: float | None = None) -> RunPage:
    """A page of runs, newest first: those of `workflow` (every run when None) started before `before`, at most
    `limit` (capped at RUNS_MAX). `next_before` is the last run's `started_at` when more remain. Runs started at the
    same instant as the page's last run but cut off by the page end are skipped (the cursor is `started_at` alone).
    A run without a numeric `started_at` is not listed (it has no place in the order); `run_detail` still shows it."""
    limit = max(1, min(int(limit), RUNS_MAX))
    sql, params = f"SELECT * FROM runs WHERE {_NUMERIC}", []
    if workflow is not None:
        sql += " AND workflow = ?"
        params.append(workflow)
    if before is not None:
        sql += " AND started_at < ?"
        params.append(before)
    conn = rt.store.connect()
    rows = conn.execute(sql + " ORDER BY started_at DESC, rowid DESC LIMIT ?", (*params, limit + 1)).fetchall()
    more, rows = len(rows) > limit, rows[:limit]
    stats = _run_stats(conn, [r["id"] for r in rows])
    return RunPage(runs=[_run_row(r, stats.get(r["id"])) for r in rows],
                   next_before=_num(rows[-1]["started_at"]) if more else None)


def _summary(kind: object, name: object, p: dict) -> str:
    """A one-line description of an event built from known payload fields only (never the raw payload)."""
    def get(key: str) -> str | None:
        return _text(p[key]) if p.get(key) is not None else None

    if kind == "llm_call":
        parts = [f"{_text(name) if name is not None else 'unknown model'}: {get('prompt_tokens') or '?'} prompt"
                 f" + {get('completion_tokens') or '?'} completion tokens"]
        if p.get("meter") not in (None, "agent"):
            parts.append(f"{get('meter')} meter")
        if _num(p.get("tool_calls")):
            parts.append(f"{get('tool_calls')} tool call(s)")
        if p.get("refused") is True:
            parts.append("refused")
        if p.get("usage_estimated"):
            parts.append("usage estimated")
        if get("fallback_from"):
            parts.append(f"fallback from {get('fallback_from')}")
        text = ", ".join(parts)
    elif kind == "tool_call":
        text = f"args: {get('args')}" if get("args") is not None else ""
    elif kind == "tool_result":
        text = f"error: {get('error')}" if "error" in p else f"result: {get('result')}" if "result" in p else ""
    elif kind in ("block", "steer", "halt"):
        text = get("reason") or ""
        if get("args"):
            text += f" (args: {get('args')})"
    elif kind == "gate":
        verdict = "allowed" if p.get("allow") is True else "denied" if p.get("allow") is False else "unknown"
        text = f"{verdict}: {get('reason')}" if get("reason") else verdict
    elif kind == "gate_error":
        text = f"gate error: {get('error') or 'unknown'}" + (f" (policy {get('policy')})" if get("policy") else "")
    elif kind == "fallback":
        text = f"{get('from') or 'unknown'} → {get('to') or 'unknown'}"
    elif kind == "agent":
        text = f"under {get('parent')}" if get("parent") else "top-level agent"
    else:
        text = ""
    return _one_line(text)


def run_detail(rt, run_id: str) -> RunDetail:
    """A run, its counts and its first MAX_EVENTS events (by time) with one-line summaries. NotFound for an unknown
    run."""
    conn = rt.store.connect()
    r = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if r is None:
        raise NotFound(f"no run {run_id!r}")
    rows = conn.execute("SELECT id, ts, agent, kind, name, payload, cost_usd FROM events WHERE run_id = ?"
                        " ORDER BY ts, id LIMIT ?", (run_id, MAX_EVENTS + 1)).fetchall()
    events = []
    for e in rows[:MAX_EVENTS]:
        p = _payload(e["payload"])
        events.append(EventRow(id=e["id"], ts=_num(e["ts"]), agent=_str(e["agent"]), kind=_str(e["kind"]),
                               name=_str(e["name"]), cost_usd=_num(e["cost_usd"]),
                               latency_s=_num(p.get("latency_s")), summary=_summary(e["kind"], e["name"], p)))
    return RunDetail(run=_run_row(r, _run_stats(conn, [run_id]).get(run_id)), workflow=_str(r["workflow"]),
                     goal=_str(r["goal"]), events=events, truncated=len(rows) > MAX_EVENTS)


# -- settings --------------------------------------------------------------------------------------------------------

def worker_status(worker) -> WorkerStatus:
    return WorkerStatus(enabled=False, running=False) if worker is None else WorkerStatus(**worker.status())


def _default_env(channel: str) -> str:
    key = _SECRET_ENV_KEY[channel]
    return next(f.default for f in fields(_CHANNEL_SPECS[channel]) if f.name == key)


def _env_set(name: str) -> bool:
    return bool(os.environ.get(name))


def settings(rt, worker=None) -> Settings:
    """Read-only settings. Workflow settings come from each workflow's latest run (they are set in code); the rest
    from config.toml. Env vars appear by name with whether they are set, never with their values."""
    conn = rt.store.connect()
    workflows = []
    for wf in workflow_names(rt.store):
        r = _latest_run(conn, wf)
        if r is not None:
            workflows.append(WorkflowSettings(name=str(wf), criticality=_str(r["criticality"]),
                                              sensitivity=_str(r["sensitivity"]), fix_mode=_str(r["fix_mode"]),
                                              budget_usd=_num(r["budget_usd"])))
    judge = rt.config.judge
    channels = []
    for name in ALERT_CHANNELS:
        spec = getattr(rt.config.alerts, name)
        env_var = getattr(spec, _SECRET_ENV_KEY[name]) if spec is not None else _default_env(name)
        channels.append(ChannelSettings(name=name, configured=spec is not None, env_var=env_var,
                                        env_set=_env_set(env_var)))
    return Settings(
        home=str(rt.home), workflows=workflows, budgets=_budgets(rt),
        judge=JudgeSettings(model=judge.model, base_url=redact_url(judge.base_url) or "",
                            api_key_env=judge.api_key_env, key_set=_env_set(judge.api_key_env)),
        alerts=AlertSettings(channels=channels,
                             routes={k: list(v) for k, v in rt.config.alerts.routes.items()}),
        worker=worker_status(worker))
