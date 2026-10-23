"""Offline dashboard demo: fill an empty PitStop home with realistic runs through the real SDK, with no keys and no
network, so the local dashboard has something to show.

  uv run python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo
  PITSTOP_HOME=/tmp/pitstop-demo uv run pitstop ui

What it records (every model reply is scripted, every text is SYNTHETIC):

- `earnings-research` (high criticality, $2/day budget): `--runs` runs on model `m-a`, then 35 on `m-b`, which uses
  double the tokens per call. The scan's L2 change detection alarms and L5 attribution names the model change. In
  the last run the writer tries to publish to a public channel; the scripted pre-action gate judge rejects it (P0)
  and the run halts.
- `support-triage` (medium): healthy runs, and one where the agent repeats the same knowledge-base search; PitStop
  steers it twice (a P1 drift).
- `nightly-digest` and `kb-sync` (low): healthy runs only.

The model and the gate's judge are real `openai.OpenAI` clients whose HTTP transport is `httpx.MockTransport`: each
request is answered in-process with a scripted reply, so nothing leaves the machine and no env var key is read.
The demo writes a `config.toml` that prices its made-up models, and refuses a home that already has a store.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import httpx
import openai

import pitstop
from pitstop.detect.scan import scan

BASE_URL = "https://llm.offline.invalid/v1"  # `.invalid` never resolves; MockTransport answers every request
JUDGE_MODEL = "m-judge"
DEFAULT_RUNS = 60
POST_CHANGE_RUNS = 35  # fixed: L2 relearns for 30 runs after its alarm, and attribution needs the runs after
MIN_RUNS = 35  # the L2 warm-up (30) plus room to watch before the change
OFF_TASK_CHANNEL = "#public-social"
CONFIG = f"""# Written by examples/quickstart/dashboard_demo.py: USD per 1M tokens for the demo's made-up models.
[prices."m-a"]
input = 0.30
output = 1.20

[prices."m-b"]
input = 0.30
output = 1.20

[prices."{JUDGE_MODEL}"]
input = 0.05
output = 0.20
"""


# -- scripted model ----------------------------------------------------------------------------------------------------

def _completion(model: str, content: str | None, tool_calls: list[tuple[str, dict]], prompt_tokens: int,
                completion_tokens: int) -> dict:
    """A `chat.completion` body in the provider's wire format."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}}
                             for i, (name, args) in enumerate(tool_calls)]
    return {"id": "chatcmpl-demo", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def _client(handler) -> openai.OpenAI:
    """A real OpenAI SDK client over an in-process transport (no sockets). The key is a placeholder, never read
    from the environment."""
    return openai.OpenAI(base_url=BASE_URL, api_key="offline-demo", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))


class ScriptedModel:
    """The worker model: answers each request with the next reply of the current script."""

    def __init__(self) -> None:
        self.replies: list[tuple] = []
        self.client = _client(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content).get("model", "unknown")
        content, tool_calls, prompt_tokens, completion_tokens = self.replies.pop(0)
        return httpx.Response(200, json=_completion(model, content, tool_calls, prompt_tokens, completion_tokens))


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[-1].split(end, 1)[0] if start in text else ""


def scripted_judge() -> openai.OpenAI:
    """The pre-action gate's judge: rejects a proposed call whose arguments name the public channel, approves the
    rest. It reads only the proposed action (not the `<recent_calls>` block)."""
    def handle(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content).get("messages", [])
        user = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        proposed = _between(user, "Proposed action:", "</tool_args>")
        verdict = ({"serves_goal": False, "reason": "publishes unreleased research to a public channel"}
                   if OFF_TASK_CHANNEL in proposed else {"serves_goal": True, "reason": "serves the goal (demo)"})
        return httpx.Response(200, json=_completion(JUDGE_MODEL, json.dumps(verdict), [], 420, 28))
    return _client(handle)


def say(text: str, prompt_tokens: int, completion_tokens: int) -> tuple:
    return (text, [], prompt_tokens, completion_tokens)


def call(name: str, args: dict, prompt_tokens: int, completion_tokens: int, text: str | None = None) -> tuple:
    return (text, [(name, args)], prompt_tokens, completion_tokens)


def _tool_spec(fn) -> dict:
    params = list(inspect.signature(fn).parameters)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": (fn.__doc__ or fn.__name__).strip(),
        "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in params}, "required": params}}}


def run_agent(model: ScriptedModel, client, model_name: str, system: str, task: str, tools: list,
              replies: list[tuple]) -> str | None:
    """A plain tool-calling loop: the scripted model answers, tools run through `@pitstop.tool`, and their results
    (or PitStop's block or steer replies) go back to the model, until it answers with text."""
    model.replies = list(replies)
    by_name = {t.__name__: t for t in tools}
    specs = [_tool_spec(t) for t in tools]
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    while True:
        msg = client.chat.completions.create(model=model_name, messages=messages, tools=specs,
                                             max_tokens=1024).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        for c in msg.tool_calls:
            out = by_name[c.function.name](**json.loads(c.function.arguments))
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": out if isinstance(out, str) else json.dumps(out)})


# -- tools (all synthetic) ---------------------------------------------------------------------------------------------

@pitstop.tool
def search_filings(ticker: str, quarter: str) -> str:
    """Search SEC filings."""
    return f"SYNTHETIC 10-Q for {ticker} {quarter}: revenue $12.4B, up 18% year over year."


@pitstop.tool
def fetch_transcript(ticker: str, quarter: str) -> str:
    """Fetch an earnings call transcript."""
    return f"SYNTHETIC call transcript for {ticker} {quarter}: guidance raised on data center demand."


@pitstop.tool(side_effects=True)
def publish_note(channel: str, title: str, body: str) -> str:
    """Publish a research note to a channel."""
    return f"published to {channel}"


@pitstop.tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket."""
    return f"SYNTHETIC ticket {ticket_id}: customer cannot export reports to CSV."


@pitstop.tool
def search_kb(query: str) -> str:
    """Search the knowledge base."""
    return f"SYNTHETIC KB result for {query!r}: no exact match."


@pitstop.tool(side_effects=True)
def reply_customer(ticket_id: str, body: str) -> str:
    """Reply to the customer on a ticket."""
    return f"replied on {ticket_id}"


@pitstop.tool
def fetch_metrics(date: str) -> str:
    """Fetch yesterday's product metrics."""
    return f"SYNTHETIC metrics for {date}: 1,204 active teams, p95 latency 410 ms."


@pitstop.tool(side_effects=True)
def post_digest(channel: str, body: str) -> str:
    """Post the digest to a channel."""
    return f"posted to {channel}"


@pitstop.tool
def list_articles(since: str) -> str:
    """List knowledge-base articles changed since a date."""
    return f"SYNTHETIC: 3 articles changed since {since}."


@pitstop.tool(side_effects=True)
def upsert_article(article_id: str, title: str) -> str:
    """Create or update a knowledge-base article in the search index."""
    return f"indexed {article_id}"


# -- workflows ---------------------------------------------------------------------------------------------------------

TICKERS = ("NVDA", "AMD", "AVGO", "MSFT", "TSM", "ASML", "AAPL", "GOOGL")
QUARTERS = ("Q1 FY2026", "Q2 FY2026", "Q3 FY2026")


def _jitter(rng: random.Random, base: int, scale: float = 1.0) -> int:
    return round(base * scale * rng.uniform(0.95, 1.05))


def earnings_run(model: ScriptedModel, client, i: int, model_name: str, scale: float, off_task: bool) -> None:
    rng = random.Random(f"earnings-{i}")
    t = lambda base: _jitter(rng, base, scale)  # noqa: E731
    ticker, quarter = TICKERS[i % len(TICKERS)], QUARTERS[i % len(QUARTERS)]
    channel = OFF_TASK_CHANNEL if off_task else "#research-desk"
    with pitstop.workflow("earnings-research", goal="Summarize each quarter's earnings for the internal research desk",
                          criticality="high", budget_usd=2.0,
                          tools=("search_filings", "fetch_transcript", "publish_note")):
        system = pitstop.prompt("earnings-research.system", "You are an equity research analyst. Use the tools; "
                                "cite the filing and the call.").text
        with pitstop.agent("researcher", goal="Collect the quarter's reported figures from filings and the call"):
            run_agent(model, client, model_name, system, f"Collect {ticker} {quarter} figures.",
                      [search_filings, fetch_transcript], [
                          call("search_filings", {"ticker": ticker, "quarter": quarter}, t(1400), t(120),
                               "Checking the 10-Q first."),
                          call("fetch_transcript", {"ticker": ticker, "quarter": quarter}, t(2200), t(160),
                               "Now the call transcript."),
                          say(f"SYNTHETIC: {ticker} {quarter} revenue $12.4B (+18% y/y); guidance raised.",
                              t(3000), t(380))])
        with pitstop.agent("writer", goal="Write the earnings note and publish it to the internal research desk"):
            run_agent(model, client, model_name, system, f"Write and publish the {ticker} {quarter} note.",
                      [publish_note], [
                          call("publish_note", {"channel": channel, "title": f"{ticker} {quarter} earnings",
                                                "body": f"SYNTHETIC: {ticker} beat on revenue; guidance raised."},
                               t(2600), t(450), "Publishing the note."),
                          say("Published.", t(3100), t(40))])


def support_run(model: ScriptedModel, client, i: int, loop: bool) -> None:
    rng = random.Random(f"support-{i}")
    t = lambda base: _jitter(rng, base)  # noqa: E731
    ticket = f"T-{4100 + i}"
    search = call("search_kb", {"query": "export reports csv"}, t(1500), t(60), "Searching the knowledge base.")
    searches = [search] * (4 if loop else 1)  # four identical searches: the third and fourth are steered
    with pitstop.workflow("support-triage", goal="Triage incoming support tickets and answer the customer",
                          criticality="medium", tools=("lookup_ticket", "search_kb", "reply_customer")):
        system = pitstop.prompt("support-triage.system", "You triage support tickets. Look the ticket up, search "
                                "the knowledge base once, then reply.").text
        with pitstop.agent("triager", goal="Find the answer to the ticket's question"):
            run_agent(model, client, "m-a", system, f"Triage ticket {ticket}.", [lookup_ticket, search_kb], [
                call("lookup_ticket", {"ticket_id": ticket}, t(900), t(50), "Reading the ticket."),
                *searches,
                say("SYNTHETIC: export is under Reports > Download > CSV.", t(1800), t(90))])
        with pitstop.agent("responder", goal="Answer the customer on the ticket"):
            run_agent(model, client, "m-a", system, f"Reply on {ticket}.", [reply_customer], [
                call("reply_customer", {"ticket_id": ticket, "body": "SYNTHETIC: use Reports > Download > CSV."},
                     t(1200), t(110)),
                say("Replied.", t(1400), t(20))])


def digest_run(model: ScriptedModel, client, i: int) -> None:
    rng = random.Random(f"digest-{i}")
    t = lambda base: _jitter(rng, base)  # noqa: E731
    date = f"2026-10-{(i % 28) + 1:02d}"
    with pitstop.workflow("nightly-digest", goal="Post the nightly product metrics digest", criticality="low",
                          tools=("fetch_metrics", "post_digest")):
        with pitstop.agent("digest"):
            run_agent(model, client, "m-a", "You write the nightly metrics digest.", f"Digest for {date}.",
                      [fetch_metrics, post_digest], [
                          call("fetch_metrics", {"date": date}, t(700), t(40)),
                          call("post_digest", {"channel": "#metrics", "body": "SYNTHETIC: 1,204 active teams."},
                               t(1100), t(160)),
                          say("Posted.", t(1300), t(15))])


def kb_sync_run(model: ScriptedModel, client, i: int) -> None:
    rng = random.Random(f"kb-{i}")
    t = lambda base: _jitter(rng, base)  # noqa: E731
    with pitstop.workflow("kb-sync", goal="Keep the knowledge-base search index in sync", criticality="low",
                          tools=("list_articles", "upsert_article")):
        with pitstop.agent("indexer"):
            run_agent(model, client, "m-a", "You keep the search index in sync.", "Sync changed articles.",
                      [list_articles, upsert_article], [
                          call("list_articles", {"since": "yesterday"}, t(600), t(40)),
                          call("upsert_article", {"article_id": f"kb-{100 + i}", "title": "Exporting reports"},
                               t(800), t(50)),
                          call("upsert_article", {"article_id": f"kb-{200 + i}", "title": "SSO setup"},
                               t(950), t(50)),
                          say("Index in sync.", t(1050), t(15))])


def _quietly(job) -> None:
    """Run one workflow run; a halt or a used-up budget is part of the story, not an error of the demo."""
    try:
        job()
    except pitstop.BudgetExceeded as e:
        print(f"  refused: {e}")
    except pitstop.Halted as e:
        print(f"  halted: {e}")


def seed(home: Path, runs: int) -> int:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(CONFIG)
    rt = pitstop.init(home=home, judge_client=scripted_judge(), judge_model=JUDGE_MODEL)
    model = ScriptedModel()
    client = pitstop.wrap(model.client)

    queues = [[lambda i=i: kb_sync_run(model, client, i) for i in range(10)],
              [lambda i=i: digest_run(model, client, i) for i in range(14)],
              [lambda i=i: support_run(model, client, i, loop=i == 10) for i in range(13)]]  # the loop run: 11th
    others = []
    while any(queues):  # round robin, each workflow's runs in order
        others += [q.pop(0) for q in queues if q]
    schedule: dict[int, list] = {}
    for j, job in enumerate(others):  # spread over the time before the model change
        schedule.setdefault(j * runs // len(others), []).append(job)

    print(f"earnings-research: {runs} runs on m-a, then {POST_CHANGE_RUNS} on m-b (double the tokens per call)")
    for i in range(runs):
        _quietly(lambda: earnings_run(model, client, i, "m-a", 1.0, off_task=False))
        for job in schedule.get(i, []):
            _quietly(job)
    for i in range(runs, runs + POST_CHANGE_RUNS):
        last = i == runs + POST_CHANGE_RUNS - 1
        _quietly(lambda: earnings_run(model, client, i, "m-b", 2.0, off_task=last))

    while (report := scan(rt.store)).runs:  # one pass reads at most 500 runs
        for a in report.alarms:
            print(f"  scan: {a.level} {a.workflow} {a.metric}: {a.summary}")
    return rt.store.connect().execute("SELECT count(*) FROM runs").fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill an empty PitStop home with offline demo data for `pitstop ui`.")
    ap.add_argument("--home", required=True, type=Path, help="an empty directory for the demo's PitStop home")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help=f"earnings-research runs before its model change (at least {MIN_RUNS}; default "
                         f"{DEFAULT_RUNS}); {POST_CHANGE_RUNS} more follow on the new model")
    args = ap.parse_args()
    if args.runs < MIN_RUNS:
        ap.error(f"--runs must be at least {MIN_RUNS} (the change detector learns its baseline over 30 runs)")
    home = args.home.expanduser().absolute()
    if home.exists() and not home.is_dir():
        print(f"{home} is not a directory; pick an empty directory", file=sys.stderr)
        sys.exit(2)
    for name in ("pitstop.db", "config.toml"):  # never touch an existing PitStop home
        if (home / name).exists():
            print(f"{home} already has a {name}; pick an empty directory", file=sys.stderr)
            sys.exit(2)
    n = seed(home, args.runs)
    print(f"Seeded {n} runs in {home}. Run: PITSTOP_HOME={home} pitstop ui")


if __name__ == "__main__":
    main()
