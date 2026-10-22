"""wrap meters the OpenAI Responses API (`responses.create` / `parse` / `stream`) and legacy `completions.create`.

Every test drives a real `openai` client over `httpx.MockTransport` (see http_fakes): no sockets, no network.
"""
import json
import math

import pytest

import pitstop
from pitstop.budget.prices import PRICES_PER_M, cost_usd
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import BudgetExceeded
from pitstop.sdk.runtime import get_runtime
from tests.sdk.http_fakes import (
    MODEL,
    json_response,
    openai_client,
    response_body,
    response_events,
    sse_response,
    text_completion,
    text_completion_chunk,
)

PRICE = PRICES_PER_M[MODEL]
INPUT = "x" * 400  # 100 estimated prompt tokens
TOOL_CALL = {"name": "search", "arguments": '{"q": "x"}'}


@pytest.fixture(autouse=True)
def _forget_endpoints():
    yield
    from pitstop.sdk.wrap import call

    call._NO_STREAM_OPTIONS.clear()


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


def _responses_handler(seen, *, tool_call=None):
    """Non-stream -> a response with usage (13, 5); stream -> the SSE events ending in response.completed."""
    def respond(body):
        if body.get("stream"):
            return sse_response(response_events("Hello", tool_call=tool_call, input_tokens=17, output_tokens=6))
        return json_response(response_body("hi", tool_calls=[tool_call] if tool_call else None, input_tokens=13,
                                           output_tokens=5, cached_tokens=4))
    return _recording(seen, respond)


def _assert_settled(prompt, completion, cached=0):
    row = _ledger_rows()[-1]
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (prompt, completion)
    assert row["usd"] == pytest.approx(cost_usd(PRICE, prompt, completion, cached))


# -- Responses API ----------------------------------------------------------------------------------------------


def test_responses_create_settles_usage_and_tool_calls(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_responses_handler(seen, tool_call=TOOL_CALL)))
    with workflow("wf") as ctx:
        r = w.responses.create(model=MODEL, input=INPUT)
    assert r.output_text == "hi"
    assert len(seen) == 1 and seen[0][0] == "/v1/responses" and "stream_options" not in seen[0][1]
    _assert_settled(13, 5, cached=4)
    [ev] = _events()
    assert (ev["run_id"], ev["name"]) == (ctx.run_id, MODEL)
    assert ev["payload"]["tool_calls"] == [TOOL_CALL]
    assert ev["payload"]["finish_reason"] == "completed" and ev["payload"]["content_chars"] == 2
    assert "usage_estimated" not in ev["payload"]


def test_responses_prompt_estimate_and_max_output(tmp_path):
    """instructions + input items (string content or text parts) + tools JSON, and max_output_tokens."""
    _init(tmp_path)
    tools = [{"type": "function", "name": "search", "parameters": {"type": "object"}}]
    items = [{"role": "user", "content": "a" * 100},
             {"role": "user", "content": [{"type": "input_text", "text": "b" * 100}]}]
    kwargs = dict(model=MODEL, instructions="c" * 100, input=items, tools=tools, max_output_tokens=77)
    est = math.ceil((300 + len(json.dumps(tools))) / 4)
    from pitstop.sdk.wrap.shapes import RESPONSES

    assert RESPONSES.estimate_prompt_tokens(kwargs) == est
    assert RESPONSES.max_output_tokens(kwargs) == 77
    assert RESPONSES.max_output_tokens({"input": "x"}) == 4096
    assert RESPONSES.estimate_prompt_tokens({"input": INPUT}) == 100
    outputs = [{"type": "function_call_output", "call_id": "c", "output": "d" * 40},
               {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_text", "text": "e" * 40}]},
               {"type": "function_call", "call_id": "c", "name": "f", "arguments": "g" * 40}]
    assert RESPONSES.estimate_prompt_tokens({"input": outputs}) == 30


def test_responses_parse_is_metered(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    _init(tmp_path)
    handler = _recording([], lambda body: json_response(response_body('{"answer": "hi"}', input_tokens=9,
                                                                      output_tokens=4)))
    w = pitstop.wrap(openai_client(handler))
    r = w.responses.parse(model=MODEL, input=INPUT, text_format=Answer)
    assert r.output_parsed.answer == "hi"
    _assert_settled(9, 4)


async def test_async_responses_create_is_metered(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([]), async_=True))
    r = await w.responses.create(model=MODEL, input=INPUT)
    assert r.output_text == "hi"
    _assert_settled(13, 5, cached=4)


def test_responses_stream_settles_from_completed_event(tmp_path):
    import openai

    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([], tool_call=TOOL_CALL)))
    stream = w.responses.create(model=MODEL, input=INPUT, stream=True)
    assert isinstance(stream, openai.Stream)
    events = list(stream)
    assert events[-1].type == "response.completed" and len(events) == len(response_events(tool_call=TOOL_CALL))
    _assert_settled(17, 6)
    [ev] = _events()
    assert ev["payload"]["tool_calls"] == [TOOL_CALL] and ev["payload"]["content_chars"] == 5
    assert ev["payload"]["finish_reason"] == "completed" and "usage_estimated" not in ev["payload"]


def test_responses_stream_helper_metered(tmp_path):
    from openai.lib.streaming.responses import ResponseStream, ResponseStreamManager

    _init(tmp_path)
    seen = []
    raw_client = openai_client(_responses_handler(seen, tool_call=TOOL_CALL))
    results = []
    for c in (raw_client, pitstop.wrap(raw_client)):
        manager = c.responses.stream(model=MODEL, input=INPUT)
        assert isinstance(manager, ResponseStreamManager)
        with manager as s:
            assert isinstance(s, ResponseStream)
            kinds = [e.type for e in s]
            final = s.get_final_response()
        results.append((kinds, final.output_text))
    assert results[0] == results[1] and results[0][1] == "Hello"
    assert seen[0][1] == seen[1][1]  # nothing injected
    _assert_settled(17, 6)
    [ev] = _events()
    assert ev["payload"]["tool_calls"] == [TOOL_CALL] and ev["payload"]["content_chars"] == 5


def test_responses_stream_helper_final_response_without_iterating(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([])))
    with w.responses.stream(model=MODEL, input=INPUT) as s:
        assert s.get_final_response().output_text == "Hello"
    _assert_settled(17, 6)


async def test_async_responses_stream_helper_metered(tmp_path):
    from openai.lib.streaming.responses import AsyncResponseStream

    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([]), async_=True))
    async with w.responses.stream(model=MODEL, input=INPUT) as s:
        assert isinstance(s, AsyncResponseStream)
        assert (await s.get_final_response()).output_text == "Hello"
    _assert_settled(17, 6)


def _worst(est_prompt=100, max_output=4096):
    row = _ledger_rows()[-1]
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (est_prompt, max_output)
    assert row["usd"] == pytest.approx(cost_usd(PRICE, est_prompt, max_output))
    assert _events()[-1]["payload"]["usage_estimated"] == "worst_case"


def test_responses_background_charged_worst_case(tmp_path):
    """A background response returns while still queued (usage null); its output is billed later."""
    _init(tmp_path)
    handler = _recording([], lambda body: json_response(response_body(None, status="queued", usage=False)))
    w = pitstop.wrap(openai_client(handler))
    assert w.responses.create(model=MODEL, input=INPUT, background=True).status == "queued"
    _worst()


def test_responses_unfinished_status_charged_worst_case(tmp_path):
    _init(tmp_path)
    handler = _recording([], lambda body: json_response(response_body(None, status="in_progress", usage=False)))
    w = pitstop.wrap(openai_client(handler))
    w.responses.create(model=MODEL, input=INPUT)
    _worst()


def test_responses_background_stream_dropped_early_charged_worst_case(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([])))
    stream = w.responses.create(model=MODEL, input=INPUT, stream=True, background=True)
    next(iter(stream))
    stream.close()
    _worst()
    # Read to its end, a background stream settles from the response.completed usage.
    list(w.responses.create(model=MODEL, input=INPUT, stream=True, background=True))
    _assert_settled(17, 6)


def test_responses_incomplete_event_usage_settled(tmp_path):
    _init(tmp_path)
    events = response_events("Hel", input_tokens=19, output_tokens=7, status="incomplete")
    w = pitstop.wrap(openai_client(_recording([], lambda body: sse_response(events))))
    with w.responses.stream(model=MODEL, input=INPUT, max_output_tokens=7) as s:
        assert [e.type for e in s][-1] == "response.incomplete"
    _assert_settled(19, 7)
    assert _events()[0]["payload"]["finish_reason"] == "incomplete"


def test_responses_stream_helper_error_in_body_settles_and_propagates(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(openai_client(_responses_handler([])))
    with pytest.raises(KeyError):
        with w.responses.stream(model=MODEL, input=INPUT) as s:
            for e in s:
                if e.type == "response.output_text.delta":
                    raise KeyError("caller's own")
    [row] = _ledger_rows()
    assert row["state"] == "settled" and _events()[0]["payload"]["usage_estimated"] is True


def test_responses_budget_exceeded_before_request(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    seen = []
    w = pitstop.wrap(openai_client(_responses_handler(seen)))
    with pytest.raises(BudgetExceeded):
        w.responses.create(model=MODEL, input=INPUT)
    with pytest.raises(BudgetExceeded):
        w.responses.stream(model=MODEL, input=INPUT)
    with pytest.raises(BudgetExceeded):
        w.chat.completions.stream(model=MODEL, messages=[{"role": "user", "content": INPUT}])
    with pytest.raises(BudgetExceeded):
        w.completions.create(model=MODEL, prompt=INPUT)
    assert seen == []  # 0 requests reach the transport
    assert _ledger_rows() == [] and _events() == []


# -- legacy completions -------------------------------------------------------------------------------------------


def _completions_handler(seen):
    def respond(body):
        if body.get("stream"):
            events = [text_completion_chunk("Hel"), text_completion_chunk("lo", finish_reason="stop")]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append(text_completion_chunk(choices=False, usage=(8, 3)))
            return sse_response(events)
        return json_response(text_completion("hi", prompt_tokens=7, completion_tokens=2))
    return _recording(seen, respond)


def test_legacy_completions_metered_and_stream_usage_hidden(tmp_path):
    _init(tmp_path)
    seen = []
    raw_client = openai_client(_completions_handler(seen))
    w = pitstop.wrap(raw_client)
    assert w.completions.create(model=MODEL, prompt=INPUT).choices[0].text == "hi"
    _assert_settled(7, 2)
    chunks = list(w.completions.create(model=MODEL, prompt=[INPUT], stream=True))
    assert seen[1][1]["stream_options"] == {"include_usage": True}
    unwrapped = list(raw_client.completions.create(model=MODEL, prompt=[INPUT], stream=True))
    assert [c.choices[0].text for c in chunks] == [c.choices[0].text for c in unwrapped] == ["Hel", "lo"]
    _assert_settled(8, 3)
    evs = _events()
    assert [e["payload"]["content_chars"] for e in evs] == [2, 5]
    assert [e["payload"]["finish_reason"] for e in evs] == ["stop", "stop"]


def test_legacy_completions_estimate_and_max_tokens(tmp_path):
    from pitstop.sdk.wrap.shapes import COMPLETIONS

    assert COMPLETIONS.estimate_prompt_tokens({"prompt": INPUT}) == 100
    assert COMPLETIONS.estimate_prompt_tokens({"prompt": [INPUT, INPUT]}) == 200
    assert COMPLETIONS.max_output_tokens({"prompt": "x"}) == 4096  # the API default is 16; reserve conservatively
    assert COMPLETIONS.max_output_tokens({"prompt": "x", "max_tokens": 9}) == 9


def test_resource_proxies_keep_isinstance_and_weakref(tmp_path):
    import weakref

    from openai.resources.chat import Chat, Completions
    from openai.resources.responses import Responses

    _init(tmp_path)
    raw = openai_client(lambda r: json_response({}))
    w = pitstop.wrap(raw)
    assert isinstance(w.chat, Chat) and isinstance(w.chat.completions, Completions)
    assert isinstance(w.responses, Responses)
    assert isinstance(w.with_raw_response.chat, type(raw.with_raw_response.chat))
    chat = w.chat
    ref = weakref.ref(chat)
    assert ref() is chat
