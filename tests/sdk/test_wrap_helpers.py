"""wrap meters the OpenAI SDK's chat helpers: `parse`, raw and streaming responses, and the stream_options retry.

Every test drives a real `openai` client over `httpx.MockTransport` (see http_fakes): no sockets, no network.
"""
import json
import math

import pytest

import pitstop
from pitstop.budget.prices import PRICES_PER_M, cost_usd
from pitstop.sdk.context import workflow
from pitstop.sdk.runtime import get_runtime
from tests.sdk.http_fakes import (
    BASE_URL,
    MODEL,
    chat_chunk,
    chat_completion,
    json_response,
    openai_client,
    sse_response,
)

PRICE = PRICES_PER_M[MODEL]
MSGS = [{"role": "user", "content": "x" * 400}]  # 100 estimated prompt tokens
WORST = cost_usd(PRICE, 100, 4096)  # the reservation for MSGS with no max_tokens


@pytest.fixture(autouse=True)
def _forget_endpoints():
    """The no-stream_options endpoint memory is process-level; no test may see another test's endpoints."""
    yield
    from pitstop.sdk.wrap import call

    call._NO_STREAM_OPTIONS.clear()


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _db():
    return get_runtime().store.connect()


def _ledger_rows():
    return [dict(r) for r in _db().execute("SELECT * FROM ledger ORDER BY id")]


def _events():
    rows = [dict(r) for r in _db().execute("SELECT * FROM events WHERE kind='llm_call' ORDER BY id")]
    for r in rows:
        r["payload"] = json.loads(r["payload"])
    return rows


def _sse_text(text="Hello", usage=None):
    """A chat stream of `text` in two chunks; with `usage`, the include_usage chunk too."""
    events = [chat_chunk(text[:3]), chat_chunk(text[3:]), chat_chunk(finish_reason="stop")]
    if usage is not None:
        events.append(chat_chunk(choices=False, usage=usage))
    return events


def _recording(seen, respond):
    def handle(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))
        return respond(body)
    return handle


def _chat_handler(seen):
    """Non-stream -> a completion with usage (11, 3); stream -> SSE, with the usage chunk only if asked for."""
    def respond(body):
        if body.get("stream"):
            wants = (body.get("stream_options") or {}).get("include_usage", False)
            return sse_response(_sse_text(usage=(21, 2) if wants else None))
        return json_response(chat_completion("hi", prompt_tokens=11, completion_tokens=3))
    return _recording(seen, respond)


# -- parse ----------------------------------------------------------------------------------------------------


def test_parse_is_metered(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    _init(tmp_path)
    seen = []
    handler = _recording(seen, lambda body: json_response(
        chat_completion('{"answer": "hi"}', prompt_tokens=11, completion_tokens=4)))
    w = pitstop.wrap(openai_client(handler))
    with workflow("wf") as ctx:
        out = w.chat.completions.parse(model=MODEL, messages=MSGS, response_format=Answer)
    assert out.choices[0].message.parsed.answer == "hi"
    assert len(seen) == 1
    [row] = _ledger_rows()
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (11, 4)
    assert row["usd"] == pytest.approx(cost_usd(PRICE, 11, 4))
    [ev] = _events()
    assert (ev["run_id"], ev["name"]) == (ctx.run_id, MODEL)
    assert ev["payload"]["content_chars"] == len('{"answer": "hi"}') and "usage_estimated" not in ev["payload"]


# -- raw responses ----------------------------------------------------------------------------------------------


def test_raw_response_create_settles_from_usage(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_chat_handler(seen)))
    raw = w.chat.completions.with_raw_response.create(model=MODEL, messages=MSGS)
    assert raw.parse().choices[0].message.content == "hi"  # the caller still gets a working raw response
    assert raw.headers["content-type"].startswith("application/json")
    assert len(seen) == 1
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["prompt_tokens"] == 11 and row["completion_tokens"] == 3
    [ev] = _events()
    assert ev["payload"]["content_chars"] == 2 and "usage_estimated" not in ev["payload"]


def test_client_level_with_raw_response_is_metered(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_chat_handler(seen)))
    raw = w.with_raw_response.chat.completions.create(model=MODEL, messages=MSGS)
    assert raw.parse().choices[0].message.content == "hi"
    [row] = _ledger_rows()
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (11, 3)
    assert len(_events()) == 1


def test_raw_response_stream_charged_worst_case(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_chat_handler(seen)))
    raw = w.chat.completions.with_raw_response.create(model=MODEL, messages=MSGS, stream=True)
    assert "stream_options" not in seen[0][1]  # the caller reads the raw stream: nothing is injected into it
    assert "".join(c.choices[0].delta.content or "" for c in raw.parse()) == "Hello"
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(WORST)
    assert _events()[0]["payload"]["usage_estimated"] == "worst_case"


def test_streaming_response_charged_worst_case(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_chat_handler(seen)))
    with w.chat.completions.with_streaming_response.create(model=MODEL, messages=MSGS) as resp:
        assert resp.parse().choices[0].message.content == "hi"
    assert len(seen) == 1
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(WORST)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (100, 4096)
    [ev] = _events()
    assert ev["payload"]["usage_estimated"] == "worst_case" and ev["cost_usd"] == pytest.approx(WORST)


async def test_async_raw_response_is_metered(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_chat_handler(seen), async_=True))
    raw = await w.chat.completions.with_raw_response.create(model=MODEL, messages=MSGS)
    assert raw.parse().choices[0].message.content == "hi"
    raw2 = await w.with_raw_response.chat.completions.create(model=MODEL, messages=MSGS)
    assert raw2.parse().choices[0].message.content == "hi"
    async with w.chat.completions.with_streaming_response.create(model=MODEL, messages=MSGS) as resp:
        assert (await resp.parse()).choices[0].message.content == "hi"
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled"] * 3
    assert [(r["prompt_tokens"], r["completion_tokens"]) for r in rows] == [(11, 3), (11, 3), (100, 4096)]


# -- stream_options retry ---------------------------------------------------------------------------------------


def _rejects_stream_options(seen):
    def respond(body):
        if "stream_options" in body:
            return json_response({"error": {"message": "unknown field stream_options", "type": "invalid_request"}},
                                 status=400)
        return sse_response(_sse_text())
    return _recording(seen, respond)


def test_stream_options_400_retries_once_without_injection(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_rejects_stream_options(seen)))
    stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in stream) == "Hello"
    assert len(seen) == 2 and "stream_options" in seen[0][1] and "stream_options" not in seen[1][1]
    [row] = _ledger_rows()
    assert row["state"] == "settled" and (row["prompt_tokens"], row["completion_tokens"]) == (100, math.ceil(5 / 4))
    assert _events()[0]["payload"]["usage_estimated"] is True
    # The endpoint is remembered: the next stream call goes out without the injection (1 request).
    assert "".join(c.choices[0].delta.content or "" for c in
                   w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)) == "Hello"
    assert len(seen) == 3 and "stream_options" not in seen[2][1]
    assert [r["state"] for r in _ledger_rows()] == ["settled", "settled"]


def test_stream_options_400_not_retried_when_caller_set_it(tmp_path):
    import openai

    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_rejects_stream_options(seen)))
    with pytest.raises(openai.BadRequestError):
        w.chat.completions.create(model=MODEL, messages=MSGS, stream=True, stream_options={"include_usage": True})
    assert len(seen) == 1
    assert [r["state"] for r in _ledger_rows()] == ["released"]


def test_stream_options_400_retry_error_propagates(tmp_path):
    import openai

    _init(tmp_path)
    seen = []
    handler = _recording(seen, lambda body: json_response({"error": {"message": "bad model"}}, status=400))
    w = pitstop.wrap(openai_client(handler))
    with pytest.raises(openai.BadRequestError):
        w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    assert len(seen) == 2  # one retry only
    assert [r["state"] for r in _ledger_rows()] == ["released"]
    from pitstop.sdk.wrap.call import _NO_STREAM_OPTIONS

    assert BASE_URL not in _NO_STREAM_OPTIONS  # a failed retry proves nothing about the endpoint


async def test_async_stream_options_400_retries_once(tmp_path):
    _init(tmp_path)
    seen = []
    w = pitstop.wrap(openai_client(_rejects_stream_options(seen), async_=True))
    stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    assert "".join([c.choices[0].delta.content or "" async for c in stream]) == "Hello"
    assert len(seen) == 2
    assert [r["state"] for r in _ledger_rows()] == ["settled"]


# -- stream identity and pass-through ---------------------------------------------------------------------------


def test_wrapped_stream_isinstance_openai_stream(tmp_path):
    import openai

    _init(tmp_path)
    w = pitstop.wrap(openai_client(_chat_handler([])))
    stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    assert isinstance(stream, openai.Stream)
    assert type(stream) is not openai.Stream  # still PitStop's proxy underneath
    list(stream)
    assert [r["state"] for r in _ledger_rows()] == ["settled"]


async def test_wrapped_async_stream_isinstance_openai_async_stream(tmp_path):
    import openai

    _init(tmp_path)
    w = pitstop.wrap(openai_client(_chat_handler([]), async_=True))
    stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    assert isinstance(stream, openai.AsyncStream)
    [c async for c in stream]
    assert [r["state"] for r in _ledger_rows()] == ["settled"]


def test_unlisted_methods_delegate_unchanged(tmp_path):
    _init(tmp_path)
    seen = []

    def respond(body):
        if body is None:  # GET /models
            return json_response({"object": "list", "data": [{"id": "m1", "object": "model", "created": 0,
                                                              "owned_by": "x"}]})
        return json_response({"object": "list", "model": "emb", "data": [
            {"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}})

    client = openai_client(_recording(seen, respond))
    w = pitstop.wrap(client)
    assert w.models is client.models and w.embeddings is client.embeddings
    assert [m.id for m in w.models.list()] == ["m1"]
    assert w.embeddings.create(model="emb", input="hi").data[0].embedding == [0.5, 0.25]
    assert w.with_raw_response.models.list().parse().data[0].id == "m1"
    assert [p for p, _ in seen] == ["/v1/models", "/v1/embeddings", "/v1/models"]
    assert _ledger_rows() == [] and _events() == []
