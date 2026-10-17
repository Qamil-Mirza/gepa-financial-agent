"""Live terminal demo: three replayed research agents supervised concurrently by task-scoped sessions.

A follows its directive (NVIDIA), B drifts to consumer GPUs (AMD), C takes a legitimate detour (NVIDIA).
Fixture labels are printed as "fixture label" for the audience only; they never reach the supervisor.

  uv run python scripts/demo_supervision.py --mock
  uv run python scripts/demo_supervision.py --policy baseline --policy current --delay 0.3
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gepa_financial_agent.adapters import EventStreamAdapter, ReplayAgent, pick_full_example  # noqa: E402
from gepa_financial_agent.config import Settings  # noqa: E402
from gepa_financial_agent.crusoe_client import CrusoeClient, MockModelClient  # noqa: E402
from gepa_financial_agent.dataset import load_jsonl  # noqa: E402
from gepa_financial_agent.policy_store import PolicyStore  # noqa: E402
from gepa_financial_agent.session_manager import SupervisorManager, TriggerConfig  # noqa: E402

CAST = [("A", "on_track", "NVDA", "follows directive"), ("B", "goal_deviation", "AMD", "drifts to consumer GPUs"),
        ("C", "exploratory_legit", "NVDA", "legitimate detour")]


async def run_policy(policy, client, examples, delay, every_n, mocked_label):
    manager = SupervisorManager(client, policy=policy, max_concurrent_inferences=3, trigger=TriggerConfig(every_n_events=every_n))
    adapter = EventStreamAdapter(manager, eval_interval_s=0.2)
    t0 = time.time()

    def show(rec):
        d = rec.decision
        if d is None:
            print(f"  [{time.time()-t0:5.1f}s] DECISION agent_{rec.agent_id} step={rec.observation_step} ERROR {rec.error}")
            return
        ev = ",".join(d.evidence_event_ids[:4]) or "-"
        print(f"  [{time.time()-t0:5.1f}s] DECISION agent_{rec.agent_id} step={rec.observation_step} {d.status.value:<9} "
              f"conf={d.confidence:.2f} type={d.drift_type.value if d.drift_type else '-':<22} evidence=[{ev}] "
              f"{rec.latency_ms:.0f}ms :: {d.explanation[:140]}")

    adapter.on_decision(show)
    agents = []
    for aid, kind, company, _ in CAST:
        ex = pick_full_example(examples, kind, company)
        a = ReplayAgent.from_example(ex, delay_s=delay, agent_id=aid)
        a.register(manager)
        agents.append((a, ex))
    print(f"\n=== policy={policy.name} ({policy.version}) model={client.model} {mocked_label} ===")
    for a, ex in agents:
        print(f"  agent_{a.agent_id}: {a.directive}  [fixture label: {ex.ground_truth.status.value}"
              f"{'/' + ex.ground_truth.drift_type.value if ex.ground_truth.drift_type else ''}]")

    async def stream(a: ReplayAgent):
        for e in a.events:
            await asyncio.sleep(a.delay_s)
            body = e.content if isinstance(e.content, str) else (e.tool_arguments or {}).get("query", "")
            print(f"  [{time.time()-t0:5.1f}s] agent_{a.agent_id} step={e.step} {e.event_type.value:<20} {str(body)[:90]}")
            await adapter.emit(a.agent_id, e)

    stop = asyncio.Event()
    consumer = asyncio.create_task(adapter.run(stop))
    await asyncio.gather(*(stream(a) for a, _ in agents))
    await adapter.drain()
    stop.set()
    await consumer
    print("  final:")
    for a, ex in agents:
        last = manager.latest(a.agent_id)
        verdict = last.decision.status.value if last and last.decision else f"ERROR({last.error if last else 'none'})"
        print(f"    agent_{a.agent_id}: supervisor={verdict:<9} fixture label={ex.ground_truth.status.value}")
    print(f"  usage: {manager.stats()['usage']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", default=None, help="repeatable; default: baseline, and current if different")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--every-n", type=int, default=3)
    args = ap.parse_args(argv)
    s = Settings.from_env()
    examples = load_jsonl("data/fixtures/dev_examples.jsonl").examples
    store = PolicyStore("policies")
    names = args.policy or ["baseline", "current"]
    policies, seen = [], set()
    for n in names:
        p = store.load(n)
        if p.version not in seen:
            policies.append(p)
            seen.add(p.version)
    if args.mock:
        client, label = MockModelClient(), "[MOCK inference]"
    else:
        if not s.has_key:
            print("CRUSOE_API_KEY not set; use --mock.")
            return 2
        client, label = CrusoeClient.from_settings(s, "supervisor"), "[REAL Crusoe inference]"
    for p in policies:
        asyncio.run(run_policy(p, client, examples, args.delay, args.every_n, label))
    latest = sorted(glob.glob("runs/gepa/*/comparison.txt"))
    if latest:
        print(f"\n=== held-out comparison from {latest[-1]} ===")
        print(Path(latest[-1]).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
