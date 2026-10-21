"""Command line entry points (installed as the `pitstop` console script).

  pitstop check                     # verify Crusoe inference (and optionally You.com)
  pitstop run [--scenario PATH] [--correct-on-drift] [--no-report]   # default: the packaged nvda_amd scenario
  pitstop run --model <crusoe id> --shadow-prompt <prompt.txt>     # 2 supervisors, same inputs
  pitstop compare --run-dir <runs_dir>/<id> --prompt-b <prompt.txt>  # prompt A defaults to supervisor_v0
  pitstop timeline --run-dir <runs_dir>/<id>   # rebuild + print the curated drift timeline
  pitstop serve --port 8765               # timeline API + live SSE stream for the UI
  pitstop demo --port 8770                # GEPA demo page (pitstop.demo)
  pitstop status | drifts [--workflow W] [--all] | budget   # SDK runs, drifts and budgets (local DB)
  pitstop budget reconcile [--older-than-min 60] [--release]  # list / free reservations left by dead processes

Runs and the You.com cache live under $PITSTOP_HOME (default ~/.pitstop); see pitstop.config.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .cli_report import budget_rows, drift_rows, workflow_rows
from .config import PROMPTS, SCENARIOS, load_settings
from .llm import LLMClient, UsageTracker
from .agents.synthesizer import ReportSynthesizer
from .research.youcom import FinanceResearchClient
from .sdk.runtime import get_runtime
from .userconfig import ConfigInvalid
from .schemas import Scenario, SupervisorInput
from .orchestrator import Swarm
from .supervisor.supervisor import Supervisor
from .timeline import build_from_run_dir

console = Console(width=None if sys.stdout.isatty() else 140)  # piped output: no 80-column truncation


def _llm(settings, model: str, temperature: float) -> LLMClient:
    if not settings.crusoe_api_key:
        console.print("[red]CRUSOE_API_KEY is not set (see .env.example)[/]")
        sys.exit(2)
    return LLMClient(model, api_key=settings.crusoe_api_key, base_url=settings.crusoe_base_url,
                     temperature=temperature, usage=UsageTracker())


async def cmd_check(args) -> None:
    s = load_settings()
    for model in {s.supervisor_model, s.worker_model}:
        llm = _llm(s, model, 0.0)
        try:
            r = await llm.complete("You are a test endpoint.", "Reply with exactly: OK")
            console.print(f"[green]✓ {model}[/] replied {r.text.strip()[:40]!r} in {r.latency_s:.1f}s "
                          f"({r.prompt_tokens}+{r.completion_tokens} tokens)")
        except Exception as e:
            console.print(f"[red]✗ {model}: {e}[/]")
    if args.youcom:
        rc = FinanceResearchClient(api_key=s.ydc_api_key, cache_dir=s.cache_dir, mode="live",
                                   effort=s.youcom_effort, max_live_calls=1)
        console.print("calling You.com once (costs ~$0.11, may take up to 2 minutes)...")
        res = await rc.research("What was NVIDIA's data center revenue in fiscal year 2025?")
        console.print(f"[green]✓ You.com[/] {len(res.sources)} sources, citations {res.cited_indices}, "
                      f"{res.latency_s:.0f}s, cached as {res.cache_key}")


async def cmd_run(args) -> None:
    _runtime()  # an invalid config.toml exits 2 before any model or research call
    s = load_settings()
    scenario = Scenario.model_validate_json(Path(args.scenario).read_text())
    worker_llm = _llm(s, s.worker_model, 0.4)
    model = args.model or s.supervisor_model
    sup_llm = _llm(s, model, 0.0)
    # same model, own client (own usage counter), different prompt; never halts or touches the timeline
    shadow = Supervisor(_llm(s, model, 0.0), Path(args.shadow_prompt)) if args.shadow_prompt else None
    research = FinanceResearchClient(
        api_key=s.ydc_api_key, cache_dir=Path(args.cache_dir or s.cache_dir), mode=args.mode,
        effort=s.youcom_effort, max_live_calls=args.max_research_calls or s.max_research_calls,
    )
    synthesizer = None if args.no_report else ReportSynthesizer(
        LLMClient(s.report_model, api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, temperature=0.2,
                  max_tokens=4000, usage=UsageTracker()), PROMPTS / "worker")
    swarm = Swarm(scenario, worker_llm=worker_llm, supervisor=Supervisor(sup_llm, Path(args.prompt)),
                  research=research, prompts_dir=PROMPTS / "worker", runs_dir=s.runs_dir,
                  halt_on_drift=args.halt_on_drift, correct_on_drift=args.correct_on_drift,
                  max_corrections=args.max_corrections, synthesizer=synthesizer, shadow_supervisor=shadow)
    await swarm.run_all()


async def cmd_compare(args) -> None:
    """Re-judge a recorded run's exact supervisor inputs with two prompts (no new research calls)."""
    s = load_settings()
    run_dir = Path(args.run_dir)
    records = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    if args.agent:
        records = [r for r in records if r["agent_id"] == args.agent]
    llm = _llm(s, s.supervisor_model, 0.0)
    sup_a, sup_b = Supervisor(llm, Path(args.prompt_a)), Supervisor(llm, Path(args.prompt_b))
    table = Table(title=f"Same inputs, two supervisors ({run_dir.name})")
    for col in ("agent", "step", "phase", "injected?", sup_a.prompt_version, sup_b.prompt_version):
        table.add_column(col)
    for r in records:
        inp = SupervisorInput.model_validate(r["input"])
        a, b = await asyncio.gather(sup_a.evaluate(inp), sup_b.evaluate(inp))
        fmt = lambda v: f"{v.decision.status.value} {v.decision.confidence:.2f} {v.decision.evidence_event_ids}"
        table.add_row(r["agent_id"], str(r["step"]), r["phase"], "yes" if r["drift_injection_active"] else "no",
                      fmt(a), fmt(b))
    console.print(table)
    console.print("[dim]'injected?' is the unverified injection label; real metrics come from the labeled eval set.[/]")


SEVERITY_STYLE = {"critical": "bold red", "warning": "yellow", "info": "dim"}


async def cmd_timeline(args) -> None:
    """Rebuild timeline.json / timeline.jsonl from a run's raw logs and print the highlights."""
    doc = build_from_run_dir(Path(args.run_dir))
    for e in doc["entries"]:
        if args.highlights and e["severity"] == "info":
            continue
        quiet = f" [dim](+{e['quiet_streak']} quiet)[/]" if e["quiet_streak"] else ""
        reply = f" [dim]↳ {e['responds_to']}[/]" if e["responds_to"] else ""
        who = e["source"]["label"] + (f" → {e['agent_label']}" if e["agent_label"] and e["source"]["role"] != "agent" else "")
        console.print(f"[{SEVERITY_STYLE[e['severity']]}]{e['entry_id']} {e['kind']:<22} \\[{escape(who)}] {escape(e['title'])}[/]"
                      f"{reply}{quiet}")
        if e["evidence"] and e["severity"] != "info":
            for ev in e["evidence"]:
                console.print(f"[dim]      {ev['event_id']}: {escape(ev['summary'][:100])}[/]")
    console.print(f"[dim]wrote {Path(args.run_dir) / 'timeline.json'} ({doc['counts']})[/]")


async def cmd_serve(args) -> None:
    from .server import make_server
    runs_dir = Path(args.runs_dir or load_settings().runs_dir)
    server = make_server(runs_dir, args.host, args.port)
    console.print(f"[green]serving timelines from {runs_dir} on http://{args.host}:{server.server_port}/api/runs[/]")
    try:
        await asyncio.to_thread(server.serve_forever)
    finally:
        server.server_close()


async def cmd_demo(args) -> None:
    from .demo.server import DEFAULT_SNAPSHOT, make_server
    server = make_server(Path(args.snapshot or DEFAULT_SNAPSHOT), host=args.host, port=args.port)
    console.print(f"[green]GEPA demo on http://{args.host}:{server.server_port}/[/]")
    try:
        await asyncio.to_thread(server.serve_forever)
    finally:
        server.server_close()


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
    for col in ("id", "level", "workflow", "drift_type", "action", "evidence", "first seen", "last seen"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), r["level"], str(r["workflow"]), r["drift_type"], r["action"], str(r["count"]),
                  _fmt_ts(r["first_seen"]), _fmt_ts(r["last_seen"]))
    console.print(t)


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
    """Reservations still `reserved` long after their call should have settled (the process died mid-call)."""
    rt = _runtime()
    older_than_s = args.older_than_min * 60
    rows = rt.ledger.stale_reservations(older_than_s)
    if not rows:
        console.print(f"No stale reservations older than {args.older_than_min:g} min.")
        return
    if args.release:
        n = rt.ledger.release_stale(older_than_s)
        console.print(f"Released {n} stale reservation(s) older than {args.older_than_min:g} min.")
        return
    t = Table(title=f"Stale reservations (older than {args.older_than_min:g} min)")
    for col in ("id", "reserved at", "run", "workflow", "meter", "model", "usd"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r["id"]), _fmt_ts(r["ts"]), str(r["run_id"]), str(r["workflow"]), str(r["meter"]),
                  str(r["model"]), _usd(r["usd"]))
    console.print(t)
    console.print(f"{len(rows)} stale reservation(s); re-run with --release to free them.")


def _minutes(v: str) -> float:
    m = float(v)
    if not m >= 0:  # also rejects nan
        raise argparse.ArgumentTypeError(f"must be a non-negative number of minutes, got {v!r}")
    return m


def main() -> None:
    p = argparse.ArgumentParser(prog="pitstop")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--youcom", action="store_true", help="also make one paid You.com call")
    r = sub.add_parser("run")
    r.add_argument("--scenario", default=str(SCENARIOS / "nvda_amd.json"))
    r.add_argument("--prompt", default=str(PROMPTS / "supervisor_v0.txt"))
    r.add_argument("--model", help="supervisor model (default: PITSTOP_SUPERVISOR_MODEL)")
    r.add_argument("--shadow-prompt", help="also judge every input with this prompt (same model); "
                                          "logged to decisions_shadow.jsonl, never halts")
    r.add_argument("--mode", choices=["live", "replay"], default="live",
                   help="live: cache-first, call You.com on miss; replay: cache only")
    r.add_argument("--cache-dir"); r.add_argument("--max-research-calls", type=int)
    r.add_argument("--halt-on-drift", action="store_true", help="stop an agent at its intervention point")
    r.add_argument("--correct-on-drift", action="store_true",
                   help="at an intervention point, block the query and send the agent a correction")
    r.add_argument("--max-corrections", type=int, default=2,
                   help="corrections per agent before --halt-on-drift (if set) halts it")
    r.add_argument("--no-report", action="store_true", help="skip the final report agent")
    m = sub.add_parser("compare")
    m.add_argument("--run-dir", required=True); m.add_argument("--agent")
    m.add_argument("--prompt-a", default=str(PROMPTS / "supervisor_v0.txt"))
    m.add_argument("--prompt-b", required=True)
    t = sub.add_parser("timeline")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--highlights", action="store_true", help="only drift flags, suspicions and responses")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1"); v.add_argument("--port", type=int, default=8765)
    v.add_argument("--runs-dir")
    d = sub.add_parser("demo")
    d.add_argument("--host", default="127.0.0.1"); d.add_argument("--port", type=int, default=8770)
    d.add_argument("--snapshot", help="default: demo/snapshot.json")
    sub.add_parser("status")
    g = sub.add_parser("drifts")
    g.add_argument("--workflow"); g.add_argument("--all", action="store_true", help="include resolved drifts")
    b = sub.add_parser("budget")
    bsub = b.add_subparsers(dest="budget_cmd")  # optional: bare `pitstop budget` shows the budgets
    rc = bsub.add_parser("reconcile", help="list (or --release) reservations left by processes that died mid-call")
    rc.add_argument("--older-than-min", type=_minutes, default=60.0,
                    help="only reservations older than this many minutes (default 60)")
    rc.add_argument("--release", action="store_true", help="release them (default: only list them)")
    args = p.parse_args()
    handler = {"check": cmd_check, "run": cmd_run, "compare": cmd_compare, "timeline": cmd_timeline,
               "serve": cmd_serve, "demo": cmd_demo, "status": cmd_status, "drifts": cmd_drifts,
               "budget": cmd_budget}[args.cmd]
    if inspect.iscoroutinefunction(handler):
        asyncio.run(handler(args))
    else:
        handler(args)


if __name__ == "__main__":
    main()
