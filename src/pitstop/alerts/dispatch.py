"""Routing, dedup, the background sender and the daily digest.

`notify(drift_id)` is called by `pitstop.sdk.actions.apply_findings` after each record of a P0/P1 drift. It only
submits the work to this module's thread pool and returns at once, so a slow or hanging channel never blocks the
agent. The pool's threads are joined at interpreter exit, so an alert submitted just before the process ends (an
uncaught `Halted`, say) is still delivered. A forked child gets a fresh pool.

Every send attempt is an `alerts` row (kind `immediate` | `digest` | `test`, status `pending` -> `sent` | `failed`,
error an exception type name or "HTTP <status>"). An immediate alert is sent once per drift and channel: a channel
is skipped while a sent row, or a pending row younger than ALERT_PENDING_STALE_S, exists for the drift at the same or
a more urgent level. So a drift that escalates from P1 to P0 alerts again, a failed send is retried at the drift's
next occurrence, and a row left pending by a killed process stops blocking after ALERT_PENDING_STALE_S. Drifts are
themselves deduplicated per (workflow, drift_type) while open, so this is one alert per open drift."""
from __future__ import annotations

import logging
import os
import threading
import time
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
# A pending row older than this was left by a process that died mid-send (or whose status update failed): it no
# longer blocks a new send. Well above one send's worst case (see RETRY_BACKOFF_S).
ALERT_PENDING_STALE_S = 120.0
# Backoff before each retry of an immediate send that failed with a transport error, a 5xx or a 429. Bounded so an
# alert queued at exit delays the exit by at most 3 attempts + 1.5 s per channel: about 16.5 s over HTTP (5 s
# timeout), 31.5 s over SMTP (10 s timeout). Digest and test sends are not retried.
RETRY_BACKOFF_S: tuple[float, ...] = (0.5, 1.0)
_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pitstop-alerts")
_lock = threading.Lock()
_futures: set[Future] = set()


def _reset_after_fork() -> None:
    # The parent's worker threads do not exist in a forked child, but its pool still counts them as idle: a task
    # submitted there would never run. Start the child with a fresh pool.
    global _pool, _lock, _futures
    _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pitstop-alerts")
    _lock = threading.Lock()
    _futures = set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


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
    """One pool job. The worker thread outlives the job, so its store connection is closed when the job ends."""
    try:
        alert = build_alert(store, drift_id)  # the drift's current level: it may have escalated since
        if alert.level not in IMMEDIATE_LEVELS:
            return
        for ch in _usable(alerts, alerts.routes[alert.level]):
            row_id = _claim(store, alert, ch.name)
            if row_id is not None:
                _attempt(store, [row_id], ch, alert, retries=RETRY_BACKOFF_S)
    except Exception as e:
        log.warning("pitstop: alert for drift %s failed (%s)", drift_id, type(e).__name__)
    finally:
        try:
            store.close_local()
        except Exception as e:
            log.warning("pitstop: could not close the alert worker's store connection (%s)", type(e).__name__)


# A row that still counts: sent, or pending and fresh enough that its sender may still be alive.
_LIVE = "(status = 'sent' OR (status = 'pending' AND ts > ?))"


def _claim(store, alert: Alert, channel: str) -> int | None:
    """Insert the pending row unless this channel already alerted (or is alerting) at this level or above. A plain
    read settles the common case (a repeat occurrence of an alerted drift) without taking the write lock; the
    check is repeated under BEGIN IMMEDIATE before inserting."""
    if _covered(store.connect(), alert, channel):
        return None
    with store.transaction(immediate=True) as c:
        if _covered(c, alert, channel):
            return None
        return _insert(c, alert.drift_id, alert.level, channel, "immediate")


def _covered(c, alert: Alert, channel: str) -> bool:
    rows = c.execute(f"SELECT level FROM alerts WHERE drift_id=? AND channel=? AND kind='immediate' AND {_LIVE}",
                     (alert.drift_id, channel, time.time() - ALERT_PENDING_STALE_S)).fetchall()
    return any(_RANK.get(r["level"], 9) <= _RANK[alert.level] for r in rows)


def _insert(c, drift_id: int | None, level: str, channel: str, kind: str) -> int:
    return c.execute("INSERT INTO alerts(drift_id, level, channel, kind, ts, status) VALUES (?,?,?,?,?,'pending')",
                     (drift_id, level, channel, kind, time.time())).lastrowid


# SMTP replies a retry will not change: the server answered and refused (smtplib.SMTPResponseException and its
# subclasses, plus refused recipients and a missing server feature). SMTPConnectError (a refused greeting) and
# SMTPServerDisconnected are connection errors and are retried.
_SMTP_FINAL = frozenset({"SMTPResponseException", "SMTPAuthenticationError", "SMTPSenderRefused", "SMTPDataError",
                         "SMTPHeloError", "SMTPRecipientsRefused", "SMTPNotSupportedError"})


def _retryable(kind: str) -> bool:
    """Transport errors, 5xx and 429 may pass on a retry; other HTTP statuses, an SMTP refusal (auth, sender,
    recipients, data) and a missing env var will not."""
    if kind == channels.NOT_CONFIGURED or kind in _SMTP_FINAL:
        return False
    if kind.startswith("HTTP "):
        status = kind[5:]
        return status.startswith("5") or status == "429"
    return True


def _attempt(store, row_ids: list[int], ch, alert: Alert, *, retries: tuple[float, ...] = ()) -> str | None:
    """Send (retrying after each delay in `retries` while the failure is retryable) and record the outcome on the
    rows. Returns None when sent, else the error kind."""
    for delay in (None, *retries):
        if delay is not None:
            time.sleep(delay)
        try:
            ch.send(alert)
            error = None
        except ChannelError as e:
            error = e.kind
        except Exception as e:  # a channel bug: still only its type name
            error = type(e).__name__
        if error is None or not _retryable(error):
            break
    status = "sent" if error is None else "failed"
    for last in (False, True):  # one retry: a stuck pending row would block this channel until it goes stale
        try:
            _update(store, row_ids, status, error)
            break
        except Exception as e:
            if last:
                log.warning("pitstop: could not record the %s alert's status (%s)", ch.name, type(e).__name__)
    if error is not None:
        log.warning("pitstop: %s alert %r failed (%s)", ch.name, alert.title, error)
    return error


def _update(store, row_ids: list[int], status: str, error: str | None) -> None:
    with store.transaction() as c:
        c.executemany("UPDATE alerts SET status=?, error=? WHERE id=?", [(status, error, i) for i in row_ids])


def digest(now: float | None = None, *, force: bool = False) -> int:
    """Send the digest of open P2/P3 drifts; returns how many drifts were sent on at least one channel. See
    `run_digest`."""
    return run_digest(now, force=force)[0]


def run_digest(now: float | None = None, *, force: bool = False) -> tuple[int, dict[str, str]]:
    """One message per usable digest channel listing the open P2/P3 drifts that channel has not had yet, with a
    row per drift and channel. Returns (drifts sent on at least one channel, channel -> "sent" | "failed: <kind>"
    for each channel that had something to send). `last_digest_ts` is set only when something was sent; within
    24 h of it nothing is sent unless `force`."""
    rt = _runtime()
    store, alerts = rt.store, rt.config.alerts
    now = time.time() if now is None else now
    if not force:
        last = last_digest_ts(store)
        if last is not None and now - last < DIGEST_INTERVAL_S:
            return 0, {}
    sent: set[int] = set()
    results: dict[str, str] = {}
    for ch in _usable(alerts, alerts.routes["digest"]):
        with store.transaction(immediate=True) as c:
            drifts = c.execute(
                "SELECT id, level FROM drifts d WHERE status='open' AND level IN ('P2', 'P3') AND NOT EXISTS"
                f" (SELECT 1 FROM alerts WHERE drift_id=d.id AND channel=? AND kind='digest' AND {_LIVE})"
                " ORDER BY id", (ch.name, time.time() - ALERT_PENDING_STALE_S)).fetchall()
            row_ids = [_insert(c, d["id"], d["level"], ch.name, "digest") for d in drifts]
        if not drifts:
            continue
        error = _attempt(store, row_ids, ch, _digest_alert(store, [d["id"] for d in drifts], now))
        results[ch.name] = "sent" if error is None else f"failed: {error}"
        if error is None:
            sent.update(d["id"] for d in drifts)
    if sent:
        with store.transaction() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (LAST_DIGEST_KEY, repr(now)))
    return len(sent), results


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
        wf = a.workflow if a.workflow is not None else "(no workflow)"
        lines.append(f"- #{i} [{a.level}] {wf}: {a.what}. Likely change: {a.cause} PitStop: {a.action}")
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
