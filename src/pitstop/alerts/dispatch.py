"""Routing, dedup, the background sender and the daily digest.

`notify(drift_id)` is called by `pitstop.sdk.actions.apply_findings` after each record of a P0/P1 drift. It only
submits the work to this module's thread pool and returns at once, so a slow or hanging channel never blocks the
agent. The pool's threads are joined at interpreter exit, so an alert submitted just before the process ends (an
uncaught `Halted`, say) is still delivered.

Every send attempt is an `alerts` row (kind `immediate` | `digest` | `test`, status `pending` -> `sent` | `failed`,
error an exception type name or "HTTP <status>"). An immediate alert is sent once per drift and channel: a channel
is skipped while a pending or sent row exists for the drift at the same or a more urgent level, so a drift that
escalates from P1 to P0 alerts again, and a failed send is retried at the drift's next occurrence. Drifts are
themselves deduplicated per (workflow, drift_type) while open, so this is one alert per open drift."""
from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as _wait

from ..userconfig import ALERT_CHANNELS
from . import channels
from .channels import ChannelError
from .model import Alert, build_alert

log = logging.getLogger("pitstop")

IMMEDIATE_LEVELS = ("P0", "P1")
DIGEST_INTERVAL_S = 86400.0
LAST_DIGEST_KEY = "last_digest_ts"
_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pitstop-alerts")
_lock = threading.Lock()
_futures: set[Future] = set()


def _runtime():
    from ..sdk.runtime import get_runtime

    return get_runtime()


def _usable(alerts, names) -> list:
    """The routed channels whose table is configured and whose env var is set, in route order."""
    out = []
    for name in names:
        ch = channels.configured(alerts, name)
        if ch is not None and ch.usable():
            out.append(ch)
    return out


def notify(drift_id: int) -> None:
    """Queue the immediate alert for a P0/P1 drift and return at once. A no-op when no P0/P1 channel is usable."""
    rt = _runtime()
    alerts = rt.config.alerts
    if not any(_usable(alerts, alerts.routes[level]) for level in IMMEDIATE_LEVELS):
        return
    fut = _pool.submit(_deliver, rt.store, alerts, drift_id)
    with _lock:
        _futures.add(fut)
    fut.add_done_callback(_done)


def _done(fut: Future) -> None:
    with _lock:
        _futures.discard(fut)


def wait_idle(timeout: float = 10.0) -> bool:
    """Wait for every queued alert to finish. True when none is left (for tests and the CLI)."""
    with _lock:
        pending = list(_futures)
    return not _wait(pending, timeout=timeout).not_done


def _deliver(store, alerts, drift_id: int) -> None:
    try:
        alert = build_alert(store, drift_id)  # the drift's current level: it may have escalated since
        if alert.level not in IMMEDIATE_LEVELS:
            return
        for ch in _usable(alerts, alerts.routes[alert.level]):
            row_id = _claim(store, alert, ch.name)
            if row_id is not None:
                _attempt(store, [row_id], ch, alert)
    except Exception as e:
        log.warning("pitstop: alert for drift %s failed (%s)", drift_id, type(e).__name__)


def _claim(store, alert: Alert, channel: str) -> int | None:
    """Insert the pending row unless this channel already alerted (or is alerting) at this level or above."""
    with store.transaction(immediate=True) as c:
        rows = c.execute("SELECT level FROM alerts WHERE drift_id=? AND channel=? AND kind='immediate'"
                         " AND status IN ('pending', 'sent')", (alert.drift_id, channel)).fetchall()
        if any(_RANK.get(r["level"], 9) <= _RANK[alert.level] for r in rows):
            return None
        return _insert(c, alert.drift_id, alert.level, channel, "immediate")


def _insert(c, drift_id: int | None, level: str, channel: str, kind: str) -> int:
    return c.execute("INSERT INTO alerts(drift_id, level, channel, kind, ts, status) VALUES (?,?,?,?,?,'pending')",
                     (drift_id, level, channel, kind, time.time())).lastrowid


def _attempt(store, row_ids: list[int], ch, alert: Alert) -> str | None:
    """Send and record the outcome on the rows. Returns None when sent, else the error kind."""
    try:
        ch.send(alert)
        error = None
    except ChannelError as e:
        error = e.kind
    except Exception as e:  # a channel bug: still only its type name
        error = type(e).__name__
    with store.transaction() as c:
        c.executemany("UPDATE alerts SET status=?, error=? WHERE id=?",
                      [("sent" if error is None else "failed", error, i) for i in row_ids])
    if error is not None:
        log.warning("pitstop: %s alert %r failed (%s)", ch.name, alert.title, error)
    return error


def digest(now: float | None = None, *, force: bool = False) -> int:
    """Send one message per usable digest channel listing the open P2/P3 drifts that channel has not had yet,
    record a row per drift and channel, and return how many drifts went out. Does nothing within 24 h of the last
    digest (meta `last_digest_ts`) unless `force`."""
    rt = _runtime()
    store, alerts = rt.store, rt.config.alerts
    now = time.time() if now is None else now
    if not force:
        last = last_digest_ts(store)
        if last is not None and now - last < DIGEST_INTERVAL_S:
            return 0
    included: set[int] = set()
    for ch in _usable(alerts, alerts.routes["digest"]):
        with store.transaction(immediate=True) as c:
            drifts = c.execute(
                "SELECT id, level FROM drifts d WHERE status='open' AND level IN ('P2', 'P3') AND NOT EXISTS"
                " (SELECT 1 FROM alerts a WHERE a.drift_id=d.id AND a.channel=? AND a.kind='digest'"
                " AND a.status IN ('pending', 'sent')) ORDER BY id", (ch.name,)).fetchall()
            row_ids = [_insert(c, d["id"], d["level"], ch.name, "digest") for d in drifts]
        if not drifts:
            continue
        _attempt(store, row_ids, ch, _digest_alert(store, [d["id"] for d in drifts], now))
        included.update(d["id"] for d in drifts)
    if included:
        with store.transaction() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (LAST_DIGEST_KEY, repr(now)))
    return len(included)


def last_digest_ts(store) -> float | None:
    row = store.connect().execute("SELECT value FROM meta WHERE key=?", (LAST_DIGEST_KEY,)).fetchone()
    try:
        return float(row[0]) if row is not None else None
    except (TypeError, ValueError):
        return None


def _digest_alert(store, drift_ids: list[int], now: float) -> Alert:
    lines, levels = [], []
    for i in drift_ids:
        try:
            a = build_alert(store, i)
        except Exception as e:
            lines.append(f"- #{i}: could not be read ({type(e).__name__})")
            continue
        levels.append(a.level)
        lines.append(f"- #{i} [{a.level}] {a.workflow}: {a.what}. Likely change: {a.cause} PitStop: {a.action}")
    level = min(levels, key=lambda lv: _RANK.get(lv, 9)) if levels else "P3"
    return Alert(drift_id=None, workflow="all workflows", level=level, drift_type="digest",
                 what=f"{len(drift_ids)} open P2/P3 drift(s):\n" + "\n".join(lines),
                 cause="Listed with each drift above.",
                 action="Nothing beyond each drift's own action; review them with `pitstop drifts`.",
                 run_id=None, ts=now)


def send_test(channel: str | None = None) -> dict[str, str]:
    """Send a test alert to one channel, or to every channel. Returns channel -> "sent" | "failed: <kind>" |
    "not configured" (no `[alerts.<channel>]` table, or its env var is unset)."""
    if channel is not None and channel not in ALERT_CHANNELS:
        raise ValueError(f"unknown alert channel {channel!r} (expected one of {ALERT_CHANNELS})")
    rt = _runtime()
    store, alerts = rt.store, rt.config.alerts
    alert = Alert(drift_id=None, workflow="pitstop", level="P3", drift_type="test",
                  what="Nothing: this is a test alert from `pitstop alerts test`.", cause="Nothing changed.",
                  action="Nothing; if you can read this, the channel works.", run_id=None, ts=time.time())
    out = {}
    for name in [channel] if channel is not None else ALERT_CHANNELS:
        ch = channels.configured(alerts, name)
        if ch is None or not ch.usable():
            out[name] = channels.NOT_CONFIGURED
            continue
        with store.transaction(immediate=True) as c:
            row_id = _insert(c, None, alert.level, name, "test")
        error = _attempt(store, [row_id], ch, alert)
        out[name] = "sent" if error is None else f"failed: {error}"
    return out
