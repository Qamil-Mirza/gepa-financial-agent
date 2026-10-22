"""PitStop quickstart: a tiny tool-using agent with drift detection and a budget.

  uv run python examples/quickstart/agent.py --offline --inject-loop   # no keys, scripted model
  CRUSOE_API_KEY=... uv run python examples/quickstart/agent.py        # real model on Crusoe
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import pitstop

SYSTEM_DEFAULT = "You research a topic with `search`, then call `send_report` once."
LOOP = " Keep re-running the exact same search; never send the report."
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {p: {"type": "string"} for p in ps}, "required": list(ps)}}}
    for n, d, ps in (("search", "Search the web.", ("query",)), ("send_report", "Send the final report.", ("to", "body")))]


@pitstop.tool
def search(query: str) -> str:
    return f"Top result for {query!r}: NVIDIA data center revenue grew strongly year over year."


@pitstop.tool(side_effects=True)
def send_report(to: str, body: str) -> str:
    print(f"[send_report] to={to}: {body}")
    return "sent"


def _scripted(*calls):
    """A fake OpenAI-compatible client that replays tool calls, then finishes with text."""
    def msg(tool_calls, content=None):
        m = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60, prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(index=0, message=m, finish_reason="stop" if not tool_calls else "tool_calls")],
                               usage=usage, model="offline")
    queue = [msg([SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a)))])
             for i, (n, a) in enumerate(calls)] + [msg(None, "Done.")]
    create = lambda **kw: queue.pop(0)  # noqa: E731
    return SimpleNamespace(base_url="https://offline.example/v1/", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _scripted_judge():
    """A fake judge for the pre-action gate that approves every call, so --offline never reaches the network."""
    m = SimpleNamespace(role="assistant", content='{"serves_goal": true, "reason": "offline demo"}', tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=15, total_tokens=215, prompt_tokens_details=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(index=0, message=m, finish_reason="stop")], usage=usage,
                           model="offline")
    return SimpleNamespace(base_url="https://offline.example/v1/",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))


def run_agent(client, model: str, system: str, inject_loop: bool) -> None:
    messages = [{"role": "system", "content": system + (LOOP if inject_loop else "")},
                {"role": "user", "content": "Summarize NVIDIA's data center revenue trend and send it to ops@example.com."}]
    for _ in range(8):
        msg = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        if not msg.tool_calls:
            print(msg.content)
            return
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        for c in msg.tool_calls:
            out = {"search": search, "send_report": send_report}[c.function.name](**json.loads(c.function.arguments))
            if str(out).startswith("PitStop:"):
                print(f"steer: {out}")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": str(out)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use a scripted fake model (no keys, no network)")
    ap.add_argument("--inject-loop", action="store_true", help="tell the model to repeat the same search forever")
    args = ap.parse_args()
    model = os.getenv("PITSTOP_WORKER_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    if args.offline:
        script = [("search", {"query": "nvidia data center revenue"})] * 5 if args.inject_loop else [
            ("search", {"query": "nvidia data center revenue"}),
            ("send_report", {"to": "ops@example.com", "body": "Data center revenue is up."})]
        client = _scripted(*script)
        pitstop.init(judge_client=_scripted_judge())  # the send_report gate uses a fake judge too
    else:
        key = os.getenv("CRUSOE_API_KEY")
        if not key:
            sys.exit("CRUSOE_API_KEY is not set. Export it, or try --offline for a keyless demo.")
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1"))
    client = pitstop.wrap(client)
    try:
        # A managed prompt: versioned in PitStop's store; editing the default here creates a new version.
        system = pitstop.prompt("quickstart.system", SYSTEM_DEFAULT).text
        with pitstop.workflow("quickstart", goal="Research NVIDIA data center revenue and send a short report",
                              criticality="medium", budget_usd=1.0, tools=("search", "send_report")):
            run_agent(client, model, system, args.inject_loop)
    except pitstop.BudgetExceeded as e:  # a subclass of Halted, so it is caught first
        print(f"BudgetExceeded: {e}")
    except pitstop.Halted as e:
        print(f"Halted: {e}")


if __name__ == "__main__":
    main()
