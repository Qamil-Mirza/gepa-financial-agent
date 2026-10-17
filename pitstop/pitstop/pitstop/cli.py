"""Command line entry points.

  python -m pitstop.cli check                     # verify Crusoe inference (and optionally You.com)
  python -m pitstop.cli run --scenario scenarios/nvda_amd.json
  python -m pitstop.cli compare --run-dir runs/<id> --prompt-a prompts/supervisor_v0.txt --prompt-b prompts/<new>.txt
  python -m pitstop.cli timeline --run-dir runs/<id>   # rebuild + print the curated drift timeline
  python -m pitstop.cli serve --port 8765               # timeline API + live SSE stream for the UI
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import ROOT, load_settings
from .llm import LLMClient, UsageTracker
from .research.youcom import FinanceResearchClient
from .schemas import Scenario, SupervisorInput
from .orchestrator import Swarm
from .supervisor.supervisor import Supervisor
from .timeline import build_from_run_dir

console = Console()


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
    s = load_settings()
    scenario = Scenario.model_validate_json(Path(args.scenario).read_text())
    worker_llm = _llm(s, s.worker_model, 0.4)
    sup_llm = _llm(s, s.supervisor_model, 0.0)
    research = FinanceResearchClient(
        api_key=s.ydc_api_key, cache_dir=Path(args.cache_dir or s.cache_dir), mode=args.mode,
        effort=s.youcom_effort, max_live_calls=args.max_research_calls or s.max_research_calls,
    )
    swarm = Swarm(scenario, worker_llm=worker_llm, supervisor=Supervisor(sup_llm, Path(args.prompt)),
                  research=research, prompts_dir=ROOT / "prompts" / "worker", runs_dir=s.runs_dir,
                  halt_on_drift=args.halt_on_drift)
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


def main() -> None:
    p = argparse.ArgumentParser(prog="pitstop")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--youcom", action="store_true", help="also make one paid You.com call")
    r = sub.add_parser("run")
    r.add_argument("--scenario", default=str(ROOT / "scenarios" / "nvda_amd.json"))
    r.add_argument("--prompt", default=str(ROOT / "prompts" / "supervisor_v0.txt"))
    r.add_argument("--mode", choices=["live", "replay"], default="live",
                   help="live: cache-first, call You.com on miss; replay: cache only")
    r.add_argument("--cache-dir"); r.add_argument("--max-research-calls", type=int)
    r.add_argument("--halt-on-drift", action="store_true", help="stop an agent at its intervention point")
    m = sub.add_parser("compare")
    m.add_argument("--run-dir", required=True); m.add_argument("--agent")
    m.add_argument("--prompt-a", default=str(ROOT / "prompts" / "supervisor_v0.txt"))
    m.add_argument("--prompt-b", required=True)
    t = sub.add_parser("timeline")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--highlights", action="store_true", help="only drift flags, suspicions and responses")
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1"); v.add_argument("--port", type=int, default=8765)
    v.add_argument("--runs-dir")
    args = p.parse_args()
    asyncio.run({"check": cmd_check, "run": cmd_run, "compare": cmd_compare,
                 "timeline": cmd_timeline, "serve": cmd_serve}[args.cmd](args))


if __name__ == "__main__":
    main()
