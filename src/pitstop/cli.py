"""Command line entry points (installed as the `pitstop` console script).

  pitstop status | drifts [--workflow W] [--all] | budget   # SDK runs, drifts and budgets (local DB)
  pitstop budget reconcile [--older-than-min 60] [--release]  # list / free old reservations left by processes
                                                              # that died or calls still running
  pitstop fallback status | clear <workflow>   # workflows pinned to their fallback model
  pitstop alerts test [--channel NAME] | digest [--force]   # check alert channels; send the P2/P3 digest

The local DB and config.toml live under $PITSTOP_HOME (default ~/.pitstop); see pitstop.config. The research
example's commands (run, check, compare, timeline, serve, demo) are on the `finresearch` console script
(examples/financial-research).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .cli_report import budget_rows, drift_rows, explain_target, workflow_rows
from .sdk.runtime import get_runtime
from .userconfig import ALERT_CHANNELS, ConfigInvalid

console = Console(width=None if sys.stdout.isatty() else 140)  # piped output: no 80-column truncation


EMPTY_MSG = "No runs yet. See examples/quickstart."


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _usd(v: float | None) -> str:
    return "-" if v is None else f"${v:.4f}"


def _has_runs(rt) -> bool:
    return rt.store.connect().execute("SELECT 1 FROM runs LIMIT 1").fetchone() is not None


def _budget_table(rt) -> Table:
    t = Table(title="Budgets")
    for col in ("scope", "period", "spent", "limit", "remaining"):
        t.add_column(col)
    for r in budget_rows(rt.ledger, rt.config):
        t.add_row(r["scope"], r["period"], _usd(r["spent_usd"]), _usd(r["limit_usd"]), _usd(r["remaining_usd"]))
    return t


def _runtime():
    """The SDK runtime for the read-only commands; an invalid config.toml is a clean error (exit 2)."""
    try:
        return get_runtime()
    except ConfigInvalid as e:
        console.print(f"[red]Invalid PitStop config:[/] {escape(str(e))}", soft_wrap=True)
        sys.exit(2)


def cmd_status(args) -> None:
    rt = _runtime()
    if not _has_runs(rt):
        console.print(EMPTY_MSG)
        return
    t = Table(title="Workflows")
    for col in ("workflow", "criticality", "runs today", "P0", "P1", "P2", "P3", "spend today", "budget", "last status"):
        t.add_column(col)
    for r in workflow_rows(rt.store, time.time(), tz=rt.ledger.tz):
        t.add_row(r["workflow"], str(r["criticality"]), str(r["runs_today"]), str(r["open_p0"]), str(r["open_p1"]),
                  str(r["open_p2"]), str(r["open_p3"]), _usd(r["spend_today_usd"]), _usd(r["budget_usd"]),
                  str(r["last_status"]))
    console.print(t)
    if rt.config.budgets:
        console.print(_budget_table(rt))


def cmd_drifts(args) -> None:
    rt = _runtime()
    rows = drift_rows(rt.store, workflow=args.workflow, open_only=not args.all)
    if not rows and not _has_runs(rt):
        console.print(EMPTY_MSG)
        return
    t = Table(title="Drifts" + ("" if args.all else " (open)"))
    for col in ("id", "level", "workflow", "drift_type", "action", "occurrences", "first seen", "last seen"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), r["level"], str(r["workflow"]), r["drift_type"], r["action"], str(r["count"]),
                  _fmt_ts(r["first_seen"]), _fmt_ts(r["last_seen"]))
    console.print(t)


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.4g}"


def cmd_explain(args) -> None:
    """L5 attribution for one drift: compares the workflow's runs before and after the drift's change point (an L2
    drift's `change_at`, else when the drift was first recorded); exits 1 for an unknown drift id."""
    from .detect.attribution import attribute

    rt = _runtime()
    target = explain_target(rt.store, args.drift_id)
    if target is None or target["workflow"] is None:
        console.print(f"no drift {args.drift_id}" if target is None else f"drift {args.drift_id} has no workflow")
        sys.exit(1)
    a = attribute(rt.store, target["workflow"], target["at"], metric=target["metric"])
    console.print(escape(a.summary), soft_wrap=True)
    if a.changes:
        t = Table(title=f"Run tags that changed ({escape(a.workflow)})")
        for col in ("tag", "before", "after", "share before", "share after"):
            t.add_column(col)
        for c in a.changes:
            t.add_row(c.tag, escape("(unset)" if c.before is None else c.before), escape(c.after),
                      f"{c.share_before:.0%}", f"{c.share_after:.0%}")
        console.print(t)
    if a.metric is not None:
        console.print(f"{escape(a.metric)} mean: {_num(a.before_mean)} over {a.n_before} runs before, "
                      f"{_num(a.after_mean)} over {a.n_after} runs after", soft_wrap=True)
    else:
        console.print(f"runs compared: {a.n_before} before, {a.n_after} after")


def cmd_budget(args) -> None:
    if getattr(args, "budget_cmd", None) == "reconcile":
        cmd_budget_reconcile(args)
        return
    rt = _runtime()
    if not _has_runs(rt):
        console.print(EMPTY_MSG)
        return
    if not rt.config.budgets:
        console.print("No budgets configured. See docs/config.example.toml.")
        return
    console.print(_budget_table(rt))


def cmd_budget_reconcile(args) -> None:
    """Reservations still `reserved` long after their call should have settled: left by a process that died
    mid-call, or by a call that is still running. Lists them; --release frees them and lists what it freed."""
    rt = _runtime()
    mins = args.older_than_min
    older_than_s = mins * 60
    rows = rt.ledger.release_stale_rows(older_than_s) if args.release else rt.ledger.stale_reservations(older_than_s)
    if not rows:
        console.print(f"No stale reservations older than {mins:g} min.")
        return
    t = Table(title=f"{'Released' if args.release else 'Stale'} reservations (older than {mins:g} min)")
    for col in ("id", "reserved at", "run", "workflow", "meter", "model", "usd"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), _fmt_ts(r["ts"]), str(r["run_id"]), str(r["workflow"]), str(r["meter"]),
                  str(r["model"]), _usd(r["usd"]))
    console.print(t)
    if args.release:
        console.print(f"Released {len(rows)} stale reservation(s) older than {mins:g} min.")
    else:
        console.print(f"{len(rows)} stale reservation(s); re-run with --release to free them.")


def _change_text(change) -> str:
    if not isinstance(change, dict):
        return "-"
    return f"{change.get('field')}: {change.get('before')} -> {change.get('after')}"


def cmd_fallback(args) -> None:
    """`status` (the default) lists the workflows whose agent calls go to their pinned fallback model; `clear`
    disengages one (its calls go back to the requested model at the next call)."""
    from .sdk import fallback

    rt = _runtime()
    if getattr(args, "fallback_cmd", None) == "clear":
        cleared = fallback.clear(rt.store, args.workflow)
        console.print("cleared" if cleared else "no active fallback")
        return
    rows = fallback.list_active(rt.store)
    if not rows:
        console.print("No active fallbacks.")
        return
    t = Table(title="Active fallbacks")
    for col in ("workflow", "model", "since", "change"):
        t.add_column(col)
    for r in rows:
        since = r.get("since")
        model = "(unreadable state; clear it)" if r.get("invalid") else str(r.get("model"))
        t.add_row(escape(r["workflow"]), escape(model),
                  _fmt_ts(since) if isinstance(since, (int, float)) else "-", escape(_change_text(r.get("change"))))
    console.print(t)


def cmd_alerts(args) -> None:
    """`test` sends a test alert to every channel (or `--channel`) and prints each result (exit 1 if one failed);
    `digest` sends the daily digest of open P2/P3 drifts (at most once per 24 h unless `--force`), prints each
    failed channel and exits 1 when every channel that had something to send failed."""
    from .alerts import dispatch

    rt = _runtime()
    if args.alerts_cmd == "test":
        results = dispatch.send_test(args.channel)
        if args.channel is None and set(results.values()) == {"not configured"}:
            console.print("No alert channel is configured: add an [alerts.<channel>] table to config.toml and set "
                          "its env var (see docs/config.example.toml).")
            return
        for name, result in results.items():
            console.print(f"{name:<10} {escape(result)}", soft_wrap=True)
        if any(r.startswith("failed") for r in results.values()):
            sys.exit(1)
        return
    n, results = dispatch.run_digest(force=args.force)
    for name, result in results.items():
        if result != "sent":
            console.print(f"{name:<10} {escape(result)}", soft_wrap=True)
    if n:
        console.print(f"Sent a digest of {n} drift(s).")
        return
    if results:  # every channel that had something to send failed
        sys.exit(1)
    last = dispatch.last_digest_ts(rt.store)
    if not args.force and last is not None and time.time() - last < dispatch.DIGEST_INTERVAL_S:
        console.print(f"No digest sent: the last digest went out at {_fmt_ts(last)} (under 24 h ago); "
                      "use --force to send now.")
    else:
        console.print("No digest sent: no new open P2/P3 drifts, or no digest channel is configured.")


def _minutes(v: str) -> float:
    m = float(v)
    if not m >= 0:  # also rejects nan
        raise argparse.ArgumentTypeError(f"must be a non-negative number of minutes, got {v!r}")
    return m


def main() -> None:
    p = argparse.ArgumentParser(prog="pitstop")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    g = sub.add_parser("drifts")
    g.add_argument("--workflow"); g.add_argument("--all", action="store_true", help="include resolved drifts")
    e = sub.add_parser("explain", help="which run tags changed around a drift (L5 attribution)")
    e.add_argument("drift_id", type=int, help="a drift id from `pitstop drifts --all`")
    b = sub.add_parser("budget")
    bsub = b.add_subparsers(dest="budget_cmd")  # optional: bare `pitstop budget` shows the budgets
    rc = bsub.add_parser("reconcile", help="list (or --release) old reservations, left by processes that died"
                                           " or calls still running")
    rc.add_argument("--older-than-min", type=_minutes, default=60.0,
                    help="only reservations older than this many minutes (default 60)")
    rc.add_argument("--release", action="store_true", help="release them (default: only list them)")
    f = sub.add_parser("fallback")
    fsub = f.add_subparsers(dest="fallback_cmd")  # optional: bare `pitstop fallback` shows the status
    fsub.add_parser("status", help="workflows whose agent calls go to their pinned fallback model")
    fc = fsub.add_parser("clear", help="send the workflow's calls back to the requested model")
    fc.add_argument("workflow")
    a = sub.add_parser("alerts")
    asub = a.add_subparsers(dest="alerts_cmd", required=True)
    at = asub.add_parser("test", help="send a test alert to each configured channel and report the result")
    at.add_argument("--channel", choices=ALERT_CHANNELS, help="only this channel")
    ad = asub.add_parser("digest", help="send the digest of open P2/P3 drifts (once per 24 h; run it from cron)")
    ad.add_argument("--force", action="store_true", help="send even if the last digest was under 24 h ago")
    args = p.parse_args()
    handler = {"status": cmd_status, "drifts": cmd_drifts, "budget": cmd_budget, "explain": cmd_explain,
               "fallback": cmd_fallback, "alerts": cmd_alerts}[args.cmd]
    handler(args)


if __name__ == "__main__":
    main()
