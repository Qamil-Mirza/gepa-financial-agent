"""wrap meters the Anthropic SDK: `messages.create` / `parse` / `stream`, raw and streaming responses.

Every test drives a real `anthropic` client over `httpx2.MockTransport` (anthropic 2.x runs on httpx2; see
http_fakes): no sockets, no network.
"""
import json
import math

import anthropic
import pytest

import pitstop
from pitstop.budget.prices import Price, cost_usd
from pitstop.detect.drifts import list_drifts
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import BudgetExceeded
from pitstop.sdk.runtime import get_runtime
from pitstop.sdk.wrap import wrap
from tests.sdk.http_fakes import (
    CLAUDE,
    MODEL,
    anthropic_client,
    chat_completion,
    json_response,
    message_body,
    message_events,
    openai_client,
    sse_response,
)

PRICE = Price(15.0, 75.0, 15.0)  # the claude-* conservative fallback
MSGS = [{"role": "user", "content": "x" * 400}]  # 100 estimated prompt tokens
TOOL_USE = {"name": "search", "input": {"q": "x"}}


def _init(tmp_path, config: str = ""):
    (tmp_path / "config.toml").write_text(config)
    return pitstop.init(home=tmp_path)


@pytest.fixture(autouse=True)
def _close_db():
    """The store keeps one connection per thread; close this thread's before the runtime is reset."""
    yield
    from pitstop.sdk import runtime

    rt = runtime._runtime
    conn = getattr(getattr(rt.store, "_local", None), "conn", None) if rt is not None else None
    if conn is not None:
        conn.close()


def _db():
    return get_runtime().store.connect()


def _ledger_rows():
    return [dict(r) for r in _db().execute("SELECT * FROM ledger ORDER BY id")]


def _events():
    rows = [dict(r) for r in _db().execute("SELECT * FROM events WHERE kind='llm_call' ORDER BY id")]
    for r in rows:
        r["payload"] = json.loads(r["payload"])
    return rows


def _recording(seen, respond):
    def handle(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))
        return respond(body)
    return handle


def _handler(seen, *, tool_use=None, stop_reason=None):
    """Non-stream -> a message with usage (13, 5); stream -> the SSE events with usage (17 in, 6 out)."""
    def respond(body):
        if body.get("stream"):
            return sse_response(message_events("Hello", tool_use=tool_use, stop_reason=stop_reason), named=True)
        return json_response(message_body("hi", tool_uses=[tool_use] if tool_use else None, stop_reason=stop_reason))
    return _recording(seen, respond)


def _assert_settled(prompt, completion, cached=0, row=None):
    row = row or _ledger_rows()[-1]
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (prompt, completion)
    assert row["usd"] == pytest.approx(cost_usd(PRICE, prompt, completion, cached))


def _run_row(run_id):
    return _db().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


# -- client detection -------------------------------------------------------------------------------------------


def test_wrapped_anthropic_keeps_type(tmp_path):
    from pitstop.sdk.wrap.anthropic import ANTHROPIC_METHODS
    from pitstop.sdk.wrap.shapes import METHODS

    w = wrap(anthropic.Anthropic(api_key="test"))
    assert isinstance(w, anthropic.Anthropic)
    assert w._pitstop_table is ANTHROPIC_METHODS
    assert isinstance(wrap(anthropic.AsyncAnthropic(api_key="test")), anthropic.AsyncAnthropic)
    assert wrap(openai_client(_handler([])))._pitstop_table is METHODS


# -- messages.create ----------------------------------------------------------------------------------------------


def test_messages_create_settles_with_cache_tokens(tmp_path):
    _init(tmp_path)
    seen = []
    body = message_body("hi", input_tokens=100, output_tokens=20, cache_creation=40, cache_read=60)
    w = wrap(anthropic_client(_recording(seen, lambda b: json_response(body))))
    with workflow("wf") as ctx:
        r = w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert r.content[0].text == "hi"
    assert len(seen) == 1 and seen[0][0] == "/v1/messages" and seen[0][1]["max_tokens"] == 50
    # no cache_creation breakdown: writes at the 1-hour premium (2x); cache reads billed at the cached rate
    _assert_settled(100 + 80 + 60, 20, cached=60)
    [ev] = _events()
    assert (ev["run_id"], ev["name"]) == (ctx.run_id, CLAUDE)
    assert ev["payload"]["finish_reason"] == "end_turn" and ev["payload"]["content_chars"] == 2
    assert ev["payload"]["tool_calls"] == [] and "usage_estimated" not in ev["payload"]
    run = _run_row(ctx.run_id)
    assert run["base_url"] == "https://api.anthropic.com" and run["price_known"] == 0  # family fallback price


def test_cache_writes_charged_by_breakdown(tmp_path):
    """5-minute writes at 1.25x, 1-hour writes at 2x, any write the breakdown misses at 2x."""
    _init(tmp_path)
    bodies = [message_body("hi", input_tokens=100, output_tokens=20, cache_creation=40, cache_read=60,
                           cache_breakdown=(30, 10)),
              message_body("hi", input_tokens=100, output_tokens=20, cache_creation=40, cache_read=60,
                           cache_breakdown=(30, 6))]  # 4 writes not in the breakdown
    w = wrap(anthropic_client(_recording([], lambda b: json_response(bodies.pop(0)))))
    w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    _assert_settled(100 + math.ceil(1.25 * 30 + 2 * 10) + 60, 20, cached=60)
    w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    _assert_settled(100 + math.ceil(1.25 * 30 + 2 * 6) + 2 * 4 + 60, 20, cached=60)


def test_cache_writes_without_breakdown_charged_at_2x(tmp_path):
    from pitstop.sdk.wrap.anthropic import ANTHROPIC
    from pitstop.sdk.wrap.shapes import Observed

    obs = Observed()
    ANTHROPIC.read(message_body("hi", input_tokens=10, output_tokens=2, cache_creation=7), obs)
    assert (obs.prompt_tokens, obs.cached_tokens, obs.completion_tokens) == (10 + 14, 0, 2)


def test_streamed_cache_writes_raised_by_message_delta(tmp_path):
    """The breakdown comes only on message_start; a larger total on message_delta is charged the excess at 2x."""
    _init(tmp_path)
    events = message_events("Hello", input_tokens=30, output_tokens=9, cache_creation=10, cache_read=4,
                            cache_breakdown=(6, 4), delta_usage={"cache_creation_input_tokens": 14})
    w = wrap(anthropic_client(_recording([], lambda b: sse_response(events, named=True))))
    list(w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True))
    _assert_settled(30 + math.ceil(1.25 * 6 + 2 * 4) + 2 * 4 + 4, 9, cached=4)


def test_messages_stream_true_settles_from_events(tmp_path):
    _init(tmp_path)
    seen = []
    events = message_events("Hello", tool_use=TOOL_USE, input_tokens=30, output_tokens=9, cache_creation=8,
                            cache_read=4)
    raw_client = anthropic_client(_recording(seen, lambda b: sse_response(events, named=True)))
    w = wrap(raw_client)
    stream = w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True)
    assert isinstance(stream, anthropic.Stream)
    got = [e.type for e in stream]
    unwrapped = [e.type for e in raw_client.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True)]
    assert got == unwrapped and got[0] == "message_start" and got[-1] == "message_stop"
    assert seen[0][1] == seen[1][1]  # nothing injected into the request
    _assert_settled(30 + 16 + 4, 9, cached=4)  # output_tokens from message_delta, not message_start's 1
    [ev] = _events()
    assert ev["payload"]["tool_calls"] == [{"name": "search", "arguments": json.dumps({"q": "x"})}]
    assert ev["payload"]["finish_reason"] == "tool_use" and ev["payload"]["content_chars"] == 5
    assert "usage_estimated" not in ev["payload"]


def test_messages_stream_dropped_early_settles_by_estimate(tmp_path):
    """No message_delta: completion by character estimate, prompt from message_start's usage."""
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([])))
    stream = w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True)
    it = iter(stream)
    kinds = [next(it).type for _ in range(3)]  # message_start, content_block_start, the "Hel" delta
    assert kinds[-1] == "content_block_delta"
    stream.close()
    _assert_settled(17, 1)
    assert _events()[0]["payload"]["usage_estimated"] is True


async def test_async_messages_create_metered(tmp_path):
    _init(tmp_path)
    seen = []
    w = wrap(anthropic_client(_handler(seen), async_=True))
    r = await w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert r.content[0].text == "hi"
    _assert_settled(13, 5)
    stream = await w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True)
    assert isinstance(stream, anthropic.AsyncStream)
    kinds = [e.type async for e in stream]
    assert kinds[-1] == "message_stop"
    _assert_settled(17, 6)
    assert [e["payload"]["content_chars"] for e in _events()] == [2, 5]


def test_messages_parse_is_metered(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    _init(tmp_path)
    body = message_body('{"answer": "hi"}', input_tokens=9, output_tokens=4)
    w = wrap(anthropic_client(_recording([], lambda b: json_response(body))))
    r = w.messages.parse(model=CLAUDE, max_tokens=50, messages=MSGS, output_format=Answer)
    assert r.parsed_output.answer == "hi"
    _assert_settled(9, 4)


def test_prompt_estimate_and_max_tokens():
    from pitstop.sdk.wrap.anthropic import ANTHROPIC

    tools = [{"name": "search", "input_schema": {"type": "object"}}]
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": [{"type": "text", "text": "b" * 100},
                                          {"type": "tool_use", "id": "t", "name": "search", "input": {"q": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "c" * 100},
                                     {"type": "tool_result", "tool_use_id": "t",
                                      "content": [{"type": "text", "text": "d" * 100}]}]},
    ]
    kwargs = dict(model=CLAUDE, max_tokens=77, system="e" * 100, messages=messages, tools=tools)
    expected = math.ceil((500 + len(json.dumps({"q": "x"})) + len(json.dumps(tools))) / 4)
    assert ANTHROPIC.estimate_prompt_tokens(kwargs) == expected
    assert ANTHROPIC.estimate_prompt_tokens({"system": [{"type": "text", "text": "f" * 40}], "messages": []}) == 10
    assert ANTHROPIC.max_output_tokens(kwargs) == 77
    assert ANTHROPIC.max_output_tokens({"messages": []}) == 4096  # required by the API; fall back if missing


# -- raw and streaming responses ----------------------------------------------------------------------------------


def test_raw_response_create_settles_from_usage(tmp_path):
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([])))
    raw = w.messages.with_raw_response.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert raw.parse().content[0].text == "hi"
    _assert_settled(13, 5)
    raw = w.with_raw_response.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert raw.parse().content[0].text == "hi"
    assert [r["state"] for r in _ledger_rows()] == ["settled"] * 2


def test_streaming_response_charged_worst_case(tmp_path):
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([])))
    with w.messages.with_streaming_response.create(model=CLAUDE, max_tokens=50, messages=MSGS) as r:
        assert json.loads(r.read())["content"][0]["text"] == "hi"
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(cost_usd(PRICE, 100, 50))
    assert _events()[0]["payload"]["usage_estimated"] == "worst_case"


def test_unlisted_methods_delegate_unchanged(tmp_path):
    raw_client = anthropic_client(_handler([]))
    w = wrap(raw_client)
    assert w.messages.count_tokens == raw_client.messages.count_tokens
    assert w.models is raw_client.models


# -- messages.stream (the stream helper) --------------------------------------------------------------------------


def test_messages_stream_helper_matches_unwrapped(tmp_path):
    """Review Focus 4: events, text_stream, get_final_message(), until_done() and isinstance as unwrapped."""
    from anthropic.lib.streaming import MessageStream, MessageStreamManager

    _init(tmp_path)
    seen = []
    raw_client = anthropic_client(_handler(seen))
    results = []
    for c in (raw_client, wrap(raw_client)):
        manager = c.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS)
        assert isinstance(manager, MessageStreamManager)
        with manager as s:
            assert isinstance(s, MessageStream)
            texts = list(s.text_stream)
            final = s.get_final_message()
            done = s.until_done()
            text = s.get_final_text()
        with c.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
            kinds = [e.type for e in s]
            snapshot = s.current_message_snapshot
        results.append((texts, final.model_dump(), done, text, kinds, snapshot.model_dump()))
    assert results[0] == results[1]
    assert results[1][0] == ["Hel", "lo"] and results[1][3] == "Hello"
    assert "text" in results[1][4]  # the helper's derived events pass through
    assert seen[0][1] == seen[2][1]  # the same request, nothing injected
    rows = _ledger_rows()
    assert len(rows) == 2
    for row in rows:
        _assert_settled(17, 6, row=row)
    evs = _events()
    assert [e["payload"]["content_chars"] for e in evs] == [5, 5]
    assert all("usage_estimated" not in e["payload"] for e in evs)


def test_messages_stream_helper_final_message_without_iterating(tmp_path):
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([], tool_use=TOOL_USE)))
    with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
        final = s.get_final_message()
    assert final.content[0].text == "Hello" and final.content[1].input == {"q": "x"}
    _assert_settled(17, 6)
    [ev] = _events()
    assert ev["payload"]["tool_calls"] == [{"name": "search", "arguments": json.dumps({"q": "x"})}]
    with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
        assert s.get_final_text() == "Hello"
    assert [r["state"] for r in _ledger_rows()] == ["settled"] * 2


def test_messages_stream_helper_early_exit_settles(tmp_path):
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([])))
    with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
        next(iter(s))
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["prompt_tokens"] == 17  # message_start usage was seen
    assert _events()[0]["payload"]["usage_estimated"] is True


def test_messages_stream_helper_never_entered_releases(tmp_path):
    import gc

    _init(tmp_path)
    seen = []
    w = wrap(anthropic_client(_handler(seen)))
    manager = w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS)
    [row] = _ledger_rows()
    assert row["state"] == "reserved" and (row["prompt_tokens"], row["completion_tokens"]) == (100, 50)
    del manager
    gc.collect()
    assert [r["state"] for r in _ledger_rows()] == ["released"] and seen == []


async def test_async_messages_stream_helper_metered(tmp_path):
    from anthropic.lib.streaming import AsyncMessageStream

    _init(tmp_path)
    w = wrap(anthropic_client(_handler([]), async_=True))
    async with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
        assert isinstance(s, AsyncMessageStream)
        texts = [t async for t in s.text_stream]
        final = await s.get_final_message()
    assert texts == ["Hel", "lo"] and final.content[0].text == "Hello"
    _assert_settled(17, 6)
    async with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
        assert await s.get_final_text() == "Hello"
        assert await s.until_done() is None  # as the SDK's own
    assert [r["state"] for r in _ledger_rows()] == ["settled"] * 2


# -- beta.messages ------------------------------------------------------------------------------------------------


def test_beta_messages_create_metered(tmp_path):
    _init(tmp_path)
    seen = []
    w = wrap(anthropic_client(_handler(seen, tool_use=TOOL_USE)))
    r = w.beta.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert r.content[0].text == "hi" and seen[0][0] == "/v1/messages"
    _assert_settled(13, 5)
    assert _events()[0]["payload"]["tool_calls"] == [{"name": "search", "arguments": json.dumps({"q": "x"})}]


def test_beta_messages_stream_helper_matches_unwrapped(tmp_path):
    from anthropic.lib.streaming import BetaMessageStream, BetaMessageStreamManager

    _init(tmp_path)
    raw_client = anthropic_client(_handler([]))
    results = []
    for c in (raw_client, wrap(raw_client)):
        manager = c.beta.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS)
        assert isinstance(manager, BetaMessageStreamManager)
        with manager as s:
            assert isinstance(s, BetaMessageStream)
            texts = list(s.text_stream)
            final = s.get_final_message()
            done = s.until_done()
        with c.beta.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
            kinds = [e.type for e in s]
        results.append((texts, final.model_dump(), done, kinds))
    assert results[0] == results[1] and results[1][0] == ["Hel", "lo"]
    rows = _ledger_rows()
    assert len(rows) == 2
    for row in rows:
        _assert_settled(17, 6, row=row)
    assert [e["payload"]["content_chars"] for e in _events()] == [5, 5]


# -- detection ------------------------------------------------------------------------------------------------


def test_refusal_stop_reason_counts_as_refusal(tmp_path):
    """stop_reason "refusal" is a refusal whatever the text says; normal sensitivity: 2 per run -> spike."""
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([], stop_reason="refusal")))
    with workflow("wf") as ctx:
        w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
        assert ctx.refusals == 1 and list_drifts(get_runtime().store) == []
        with w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS) as s:
            s.until_done()
    assert ctx.refusals == 2
    assert [r["drift_type"] for r in list_drifts(get_runtime().store)] == ["refusal_spike"]
    assert [e["payload"]["finish_reason"] for e in _events()] == ["refusal", "refusal"]


def test_tool_use_blocks_recorded_as_tool_calls(tmp_path):
    _init(tmp_path)
    w = wrap(anthropic_client(_handler([], tool_use=TOOL_USE)))
    with workflow("wf") as ctx:
        for _ in range(3):
            w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
        list(w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True))
    expected = [{"name": "search", "arguments": json.dumps({"q": "x"})}]
    assert [e["payload"]["tool_calls"] for e in _events()] == [expected] * 4
    assert [(r.tool_calls, r.tool_call_parse_failures) for r in ctx.llm_history] == [(1, 0)] * 4


def test_streamed_tool_use_without_input_records_empty_object(tmp_path):
    _init(tmp_path)
    events = message_events("Hello", tool_use={"name": "now", "input": {}})
    events = [e for e in events if (e.get("delta") or {}).get("type") != "input_json_delta"]
    w = wrap(anthropic_client(_recording([], lambda b: sse_response(events, named=True))))
    with workflow("wf") as ctx:
        list(w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS, stream=True))
    assert _events()[0]["payload"]["tool_calls"] == [{"name": "now", "arguments": "{}"}]
    assert ctx.llm_history[0].tool_call_parse_failures == 0


# -- budgets ------------------------------------------------------------------------------------------------------


def test_global_budget_by_anthropic_base_url(tmp_path):
    _init(tmp_path, '[budgets.anthropic]\nlimit_usd = 0.000001\nperiod = "total"\n'
                    'base_url = "https://api.anthropic.com"\n')
    seen = []
    w = wrap(anthropic_client(_handler(seen)))
    with pytest.raises(BudgetExceeded):
        w.messages.create(model=CLAUDE, max_tokens=50, messages=MSGS)
    with pytest.raises(BudgetExceeded):
        w.messages.stream(model=CLAUDE, max_tokens=50, messages=MSGS)
    assert seen == [] and _ledger_rows() == []
    # Another provider's base_url is outside that budget.
    o = wrap(openai_client(_recording([], lambda b: json_response(chat_completion()))))
    o.chat.completions.create(model=MODEL, messages=MSGS)
    assert [r["state"] for r in _ledger_rows()] == ["settled"]
