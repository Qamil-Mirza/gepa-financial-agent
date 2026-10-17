"""Verify Crusoe connectivity: list models, show supported parameters, make one real call.

Exit 2 (with a clear message) when CRUSOE_API_KEY is not set (unless --mock). Never prints the key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gepa_financial_agent.config import Settings  # noqa: E402
from gepa_financial_agent.crusoe_client import CrusoeClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--supervise", metavar="EXAMPLE_ID", nargs="?", const="auto",
                    help="run the baseline supervisor on one fixture example (default: first DRIFTING goal_deviation)")
    ap.add_argument("--mock", action="store_true", help="use MockModelClient (clearly labeled) instead of Crusoe")
    ap.add_argument("--policy", default="policies/current.txt")
    args = ap.parse_args(argv)
    s = Settings.from_env()
    client = None
    if not args.mock:
        if not s.has_key:
            print("CRUSOE_API_KEY is not set. export CRUSOE_API_KEY='<key>' (single quotes). Skipping real inference.")
            return 2
        client = CrusoeClient.from_settings(s, "supervisor")
        models = client.list_models()
        ids = sorted(m.get("id", "") for m in models)
        if args.list_models:
            print("\n".join(ids))
        for role, mid in (("supervisor", s.supervisor_model), ("reflection", s.reflection_model)):
            info = next((m for m in models if m.get("id") == mid), None)
            if info is None:
                print(f"[{role}] {mid}: NOT FOUND in /v1/models (check casing/prefix)")
                continue
            sp = info.get("supported_parameters", [])
            print(f"[{role}] {mid}: context={info.get('context_length')} features={info.get('features')}")
            print(f"         response_format supported: {'response_format' in sp}; supported_parameters={sp}")
        r = client.complete([{"role": "system", "content": "You are terse."},
                             {"role": "user", "content": "Reply with exactly: pong"}], max_tokens=args.max_tokens)
        print(json.dumps({"model": r.model, "text": r.text.strip(), "finish_reason": r.finish_reason,
                          "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                          "cached_tokens": r.cached_tokens, "latency_ms": round(r.latency_ms, 1),
                          "reasoning_present": bool(r.reasoning), "mocked": r.mocked}, indent=2))
        print("usage:", json.dumps(client.tracker.snapshot()))
    if args.supervise:
        from gepa_financial_agent.crusoe_client import MockModelClient
        from gepa_financial_agent.dataset import load_jsonl
        from gepa_financial_agent.policy_store import Policy
        from gepa_financial_agent.supervisor import Supervisor
        examples = load_jsonl("data/fixtures/dev_examples.jsonl").examples
        if args.supervise == "auto":
            ex = next(e for e in examples if "kind:goal_deviation" in e.tags and e.ground_truth.status.value == "DRIFTING")
        else:
            ex = next(e for e in examples if e.example_id == args.supervise)
        sup_client = MockModelClient() if args.mock else client
        sup = Supervisor(sup_client, Policy.from_file(args.policy), json_mode=s.json_mode)
        rec = sup.evaluate_sync(ex.supervisor_input, example_id=ex.example_id)
        tag = "MOCK" if rec.mocked else "REAL"
        print(f"[{tag}] example={ex.example_id} expected={ex.ground_truth.status.value}/{ex.ground_truth.drift_type}")
        print(rec.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
