"""Command line entry points (installed as the `pitstop` console script).

  pitstop status | drifts [--workflow W] [--all] | budget   # SDK runs, drifts and budgets (local DB)
  pitstop budget reconcile [--older-than-min 60] [--release]  # list / free old reservations left by processes
                                                              # that died or calls still running
  pitstop fallback status | clear <workflow>   # workflows pinned to their fallback model
  pitstop alerts test [--channel NAME] | digest [--force]   # check alert channels; send the P2/P3 digest
  pitstop explain <drift-id>         # which run tags changed around a drift (L5 attribution)
  pitstop scan [--every SECONDS]     # L2 change detection over finished runs; records attributed drifts
  pitstop ui [--host 127.0.0.1] [--port 8787] [--no-open] [--no-worker] [--scan-every SECONDS]
                                     # the local dashboard (needs the `ui` extra); `serve` = `ui --no-open`

The local DB and config.toml live under $PITSTOP_HOME (default ~/.pitstop); see pitstop.config. The research
example's commands (run, check, compare, timeline, serve, demo) are on that example's own console script
(examples/financial-research).
"""
from __future__ import annotations

import argparse
import errno
import ipaddress
import math
import socket
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
    for col in (
        "workflow",
        "criticality",
        "runs today",
        "P0",
        "P1",
        "P2",
        "P3",
        "spend today",
        "budget",
        "last status",
    ):
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
    if target is None or target["workflow"] is None or target["at"] is None:
        console.print(f"no drift {args.drift_id}" if target is None else f"drift {args.drift_id} has no workflow"
                      if target["workflow"] is None else f"drift {args.drift_id} has no time to compare around")
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


def _scan_once(store) -> None:
    from .detect.scan import scan

    report = scan(store)
    n = len(report.alarms)
    console.print(f"scanned {report.runs} run{'' if report.runs == 1 else 's'}, {n} alarm{'' if n == 1 else 's'}")
    for a in report.alarms:
        console.print(escape(f"{a.level} {a.workflow} {a.drift_type} ({a.metric}): {a.summary}"), soft_wrap=True)
    console.file.flush()


def cmd_scan(args) -> None:
    """One `scan` pass (L2 over finished runs, L5 attribution on each alarm); with --every, a pass every SECONDS
    until Ctrl-C (exit 0)."""
    rt = _runtime()
    try:
        _scan_once(rt.store)
        while args.every is not None:
            time.sleep(args.every)
            _scan_once(rt.store)
    except KeyboardInterrupt:
        return


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


UI_EXTRA_MSG = 'The dashboard needs the ui extra: pip install "pitstop[ui]"'
NO_AUTH_WARNING = ("Warning: listening on {host}, which is not a loopback address. The dashboard shows prompts and "
                   "tool arguments and has no authentication.")


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:  # a host name
        return False


def _listen(host: str, port: int) -> socket.socket:
    """A listening socket on host:port, bound here so a port in use is a clean exit 2 and the browser can connect
    as soon as it opens (connections queue until uvicorn accepts them)."""
    bare = host.strip("[]")
    sock = None
    try:
        try:
            family = socket.AF_INET6 if ipaddress.ip_address(bare).version == 6 else socket.AF_INET
        except ValueError:  # a host name, e.g. localhost
            family = socket.getaddrinfo(bare, port, type=socket.SOCK_STREAM)[0][0]
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn does: rebinding after a restart
        sock.bind((bare, port))
        sock.listen(2048)
    except OSError as e:  # socket.gaierror for an unknown host name is an OSError too
        if sock is not None:
            sock.close()
        if e.errno == errno.EADDRINUSE:
            console.print(f"port {port} is in use; pass --port", soft_wrap=True)
        else:
            console.print(f"cannot listen on {escape(host)}:{port} ({type(e).__name__}: {escape(str(e))})",
                          soft_wrap=True)
        sys.exit(2)
    return sock


def cmd_ui(args) -> None:
    """The local dashboard: the API, the built UI and (unless --no-worker) the background scan worker, served by
    uvicorn until Ctrl-C. Fails fast with exit 2 on a missing `ui` extra, an invalid config, a too-new store or a
    port it cannot listen on."""
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        console.print(UI_EXTRA_MSG, markup=False, soft_wrap=True)
        sys.exit(2)
    rt = _runtime()
    import webbrowser

    from .server.app import DEFAULT_HOSTS, create_app, normalize_host
    from .server.worker import Worker

    host = args.host
    if not _is_loopback(host):
        console.print(NO_AUTH_WARNING.format(host=host), markup=False, soft_wrap=True)
    sock = _listen(host, args.port)
    worker = None if args.no_worker else Worker(rt.store, scan_every_s=args.scan_every)
    app = create_app(rt, allowed_hosts=DEFAULT_HOSTS + (normalize_host(host),), worker=worker)
    url = f"http://{normalize_host(host)}:{sock.getsockname()[1]}"
    console.print(f"PitStop dashboard: {url}", markup=False, soft_wrap=True)
    if not args.no_open:
        webbrowser.open(url)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", proxy_headers=False))
    server.run(sockets=[sock])


def _minutes(v: str) -> float:
    m = float(v)
    if not (m >= 0 and math.isfinite(m)):  # rejects nan and inf
        raise argparse.ArgumentTypeError(f"must be a non-negative, finite number of minutes, got {v!r}")
    return m


def _seconds(v: str) -> float:
    s = float(v)
    if not (s > 0 and math.isfinite(s)):  # rejects nan and inf (time.sleep(inf) overflows)
        raise argparse.ArgumentTypeError(f"must be a positive, finite number of seconds, got {v!r}")
    return s


def _port(v: str) -> int:
    n = int(v)
    if not 0 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"must be a port number from 0 to 65535, got {v!r}")
    return n


def _ui_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default 127.0.0.1; any other address exposes the dashboard, which "
                        "has no login)")
    p.add_argument("--port", type=_port, default=8787, help="port to listen on (default 8787; 0 picks a free one)")
    p.add_argument("--no-worker", action="store_true", help="do not run the background scan and digest worker")
    p.add_argument("--scan-every", type=_seconds, default=60.0, metavar="SECONDS",
                   help="seconds between the worker's scan passes (default 60)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pitstop")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    g = sub.add_parser("drifts")
    g.add_argument("--workflow"); g.add_argument("--all", action="store_true", help="include resolved drifts")
    e = sub.add_parser("explain", help="which run tags changed around a drift (L5 attribution)")
    e.add_argument("drift_id", type=int, help="a drift id from `pitstop drifts --all`")
    sc = sub.add_parser("scan", help="L2 change detection over finished runs; records attributed drifts")
    sc.add_argument("--every", type=_seconds, metavar="SECONDS", help="repeat every SECONDS until Ctrl-C")
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
    u = sub.add_parser("ui", help="the local dashboard; opens it in the browser (needs the ui extra)")
    _ui_args(u)
    u.add_argument("--no-open", action="store_true", help="do not open the browser")
    sv = sub.add_parser("serve", help="the local dashboard without opening the browser (ui --no-open)")
    _ui_args(sv)
    sv.set_defaults(no_open=True)
    args = p.parse_args(argv)
    handler = {"status": cmd_status, "drifts": cmd_drifts, "budget": cmd_budget, "explain": cmd_explain,
               "scan": cmd_scan, "fallback": cmd_fallback, "alerts": cmd_alerts, "ui": cmd_ui,
               "serve": cmd_ui}[args.cmd]
    handler(args)


if __name__ == "__main__":
    main()
