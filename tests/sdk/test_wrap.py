import asyncio
import gc
import inspect
import json
import math
import sqlite3
from types import SimpleNamespace

import pytest

import pitstop
from pitstop.budget.ledger import Ledger
from pitstop.budget.prices import CONSERVATIVE_PRICE, PRICES_PER_M, cost_usd
from pitstop.sdk.context import agent, current_run, workflow
from pitstop.sdk.errors import BudgetExceeded, Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import (
    BASE_URL,
    FakeAsyncOpenAI,
    FakeOpenAI,
    make_chunk,
    make_response,
    make_tool_call,
    make_tool_delta,
    make_usage_chunk,
)

MODEL = "openai/gpt-oss-120b"
PRICE = PRICES_PER_M[MODEL]
MSGS = [{"role": "user", "content": "x" * 400}]  # 100 estimated prompt tokens


def _init(tmp_path, budgets: str = ""):
    (tmp_path / "config.toml").write_text(budgets)
    return pitstop.init(home=tmp_path)


def _db():
    return get_runtime().store.connect()


def _ledger_rows():
    return [dict(r) for r in _db().execute("SELECT * FROM ledger ORDER BY id")]


def _states():
    return [r["state"] for r in _ledger_rows()]


def _events():
    rows = [dict(r) for r in _db().execute("SELECT * FROM events WHERE kind='llm_call' ORDER BY id")]
    for r in rows:
        r["payload"] = json.loads(r["payload"])
    return rows


def _spy_reserve(monkeypatch):
    seen = []
    orig = Ledger.reserve

    def spy(self, **kw):
        seen.append(kw)
        return orig(self, **kw)

    monkeypatch.setattr(Ledger, "reserve", spy)
    return seen


# -- delegation / identity ----------------------------------------------------------------------------


def test_wrap_delegates_unknown_attributes(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([])
    w = pitstop.wrap(fake)
    assert w.models is fake.models
    assert w.base_url == BASE_URL
    assert w.chat.completions is not fake.chat.completions  # intercepted
    assert w.calls is fake.calls
    with pytest.raises(AttributeError):
        w.does_not_exist


def test_double_wrap_is_idempotent(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([make_response("hi")])
    w = pitstop.wrap(fake)
    assert pitstop.wrap(w) is w
    pitstop.wrap(w).chat.completions.create(model=MODEL, messages=MSGS)
    assert len(_ledger_rows()) == 1 and len(_events()) == 1


def test_wrapped_real_openai_clients_keep_type_and_async_shape(tmp_path):
    import openai

    _init(tmp_path)
    a = pitstop.wrap(openai.AsyncOpenAI(api_key="k", base_url="http://localhost:1/v1"))
    s = pitstop.wrap(openai.OpenAI(api_key="k", base_url="http://localhost:1/v1"))
    assert isinstance(a, openai.AsyncOpenAI) and isinstance(s, openai.OpenAI)
    # openai's create is a sync def wrapping an async def; wrap must still treat it as async.
    assert inspect.iscoroutinefunction(a.chat.completions.create)
    assert not inspect.iscoroutinefunction(s.chat.completions.create)
    # Derived clients stay metered.
    assert type(s.with_options(timeout=5)).__name__ == "_Wrapped"
    assert type(a.copy()).__name__ == "_Wrapped"


# -- non-stream ---------------------------------------------------------------------------------------


def test_sync_call_settles_usage_and_records_event(tmp_path):
    _init(tmp_path)
    tc = make_tool_call("search", '{"q": "nvda"}')
    resp = make_response(None, [tc], prompt_tokens=1000, completion_tokens=500, cached_tokens=200)
    fake = FakeOpenAI([resp])
    w = pitstop.wrap(fake)
    with workflow("wf") as ctx:
        with agent("researcher", goal="find"):
            out = w.chat.completions.create(model=MODEL, messages=MSGS, max_tokens=50)
    assert out is resp
    assert fake.calls == [{"model": MODEL, "messages": MSGS, "max_tokens": 50}]
    want = cost_usd(PRICE, 1000, 500, 200)
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(want)
    assert (row["run_id"], row["workflow"], row["meter"], row["base_url"]) == (ctx.run_id, "wf", "agent",
                                                                               BASE_URL.rstrip("/"))
    [ev] = _events()
    assert (ev["run_id"], ev["agent"], ev["name"]) == (ctx.run_id, "researcher", MODEL)
    assert ev["cost_usd"] == pytest.approx(want)
    assert ev["payload"]["tool_calls"] == [{"name": "search", "arguments": '{"q": "nvda"}'}]
    assert ev["payload"]["finish_reason"] == "tool_calls"
    assert ev["payload"]["content_chars"] == 0
    assert "unmetered" not in ev["payload"]
    run = dict(_db().execute("SELECT * FROM runs WHERE id=?", (ctx.run_id,)).fetchone())
    assert (run["model"], run["base_url"], run["adapter"], run["pitstop_version"], run["price_known"]) == (
        MODEL, BASE_URL, "wrap", pitstop.__version__, 1)


def test_prompt_estimate_and_max_tokens(tmp_path, monkeypatch):
    _init(tmp_path)
    seen = _spy_reserve(monkeypatch)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    msgs = [{"role": "system", "content": "abc"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}]},
            {"role": "assistant", "content": None}]
    w = pitstop.wrap(FakeOpenAI([make_response("a")] * 3))
    w.chat.completions.create(model=MODEL, messages=msgs, tools=tools)
    w.chat.completions.create(model=MODEL, messages=msgs, max_completion_tokens=77)
    w.chat.completions.create(model=MODEL, messages=msgs, max_tokens=33)
    assert seen[0]["est_prompt_tokens"] == math.ceil((3 + 5 + len(json.dumps(tools))) / 4)
    assert [s["max_completion_tokens"] for s in seen] == [4096, 77, 33]
    assert seen[1]["est_prompt_tokens"] == 2
    assert all(s["meter"] == "agent" for s in seen)


async def test_async_call_settles_usage(tmp_path):
    _init(tmp_path)
    resp = make_response("hello", prompt_tokens=100, completion_tokens=20)
    fake = FakeAsyncOpenAI([resp])
    w = pitstop.wrap(fake)
    assert inspect.iscoroutinefunction(w.chat.completions.create)
    async with workflow("wf"):
        out = await w.chat.completions.create(model=MODEL, messages=MSGS)
    assert out is resp
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(cost_usd(PRICE, 100, 20))
    [ev] = _events()
    assert ev["payload"]["content_chars"] == 5 and ev["payload"]["finish_reason"] == "stop"


def test_non_stream_without_usage_settles_by_estimate(tmp_path):
    """Ruling 3: minimal response objects (no usage, no tool_calls, no finish_reason) are settled by estimate."""
    _init(tmp_path)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="y" * 40))])
    w = pitstop.wrap(FakeOpenAI([resp]))
    assert w.chat.completions.create(model=MODEL, messages=MSGS) is resp
    [row] = _ledger_rows()
    assert row["state"] == "settled"
    assert (row["prompt_tokens"], row["completion_tokens"]) == (100, 10)
    assert row["usd"] == pytest.approx(cost_usd(PRICE, 100, 10))
    [ev] = _events()
    assert ev["payload"]["tool_calls"] == [] and ev["payload"]["finish_reason"] is None
    assert ev["payload"]["usage_estimated"] is True


def test_coroutine_returned_from_undetected_sync_create_is_still_metered(tmp_path):
    _init(tmp_path)
    resp = make_response("hi", prompt_tokens=7, completion_tokens=3)

    async def acreate(**kw):
        return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: acreate(**kw))))
    w = pitstop.wrap(client)
    assert asyncio.run(w.chat.completions.create(model=MODEL, messages=MSGS)) is resp
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["base_url"] is None
    assert row["usd"] == pytest.approx(cost_usd(PRICE, 7, 3))


# -- failures -----------------------------------------------------------------------------------------


def test_failed_call_releases_reservation(tmp_path):
    _init(tmp_path)
    err = RuntimeError("upstream 500")
    w = pitstop.wrap(FakeOpenAI([err]))
    with pytest.raises(RuntimeError) as ei:
        w.chat.completions.create(model=MODEL, messages=MSGS)
    assert ei.value is err
    assert _states() == ["released"]
    assert _events() == []


async def test_cancelled_async_call_releases_reservation(tmp_path):
    _init(tmp_path)
    started = asyncio.Event()

    async def create(**kw):
        started.set()
        await asyncio.Event().wait()  # never returns

    w = pitstop.wrap(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    task = asyncio.create_task(w.chat.completions.create(model=MODEL, messages=MSGS))
    await started.wait()
    assert _states() == ["reserved"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _states() == ["released"]


def test_keyboard_interrupt_during_sync_call_releases_reservation(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([KeyboardInterrupt()]))
    with pytest.raises(KeyboardInterrupt):
        w.chat.completions.create(model=MODEL, messages=MSGS)
    assert _states() == ["released"]


def test_budget_exceeded_raises_before_calling_client(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    fake = FakeOpenAI([make_response("hi")])
    w = pitstop.wrap(fake)
    with pytest.raises(BudgetExceeded) as ei:
        w.chat.completions.create(model=MODEL, messages=MSGS)
    assert ei.value.scope == "global:all"
    assert fake.calls == []
    assert _ledger_rows() == [] and _events() == []


def test_workflow_budget_is_enforced(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([make_response("hi")])
    w = pitstop.wrap(fake)
    with pytest.raises(BudgetExceeded) as ei:
        with workflow("wf", budget_usd=0.000001):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    assert ei.value.scope == "workflow:wf"
    assert fake.calls == []


def test_pending_halt_raises_before_calling_client(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([make_response("hi")])
    w = pitstop.wrap(fake)
    with pytest.raises(Halted, match="stop now"):
        with workflow("wf"):
            current_run().halt_reason = "stop now"
            w.chat.completions.create(model=MODEL, messages=MSGS)
    assert fake.calls == [] and _ledger_rows() == []


def test_wrap_outside_workflow_counts_global_budget(tmp_path):
    rt = _init(tmp_path, '[budgets.all]\nlimit_usd = 1.0\nperiod = "total"\n')
    w = pitstop.wrap(FakeOpenAI([make_response("hi", prompt_tokens=1000, completion_tokens=100)]))
    assert current_run() is None
    w.chat.completions.create(model=MODEL, messages=MSGS)
    [row] = _ledger_rows()
    assert (row["state"], row["run_id"], row["workflow"]) == ("settled", None, None)
    assert rt.ledger.spent("global:all") == pytest.approx(cost_usd(PRICE, 1000, 100))
    [ev] = _events()
    assert ev["run_id"] is None and ev["agent"] is None


def test_store_failure_does_not_break_user_call(tmp_path, monkeypatch):
    _init(tmp_path)

    def boom(self, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "reserve", boom)
    resp = make_response("hi")
    fake = FakeOpenAI([resp])
    w = pitstop.wrap(fake)
    with workflow("wf"):
        assert w.chat.completions.create(model=MODEL, messages=MSGS) is resp
    assert len(fake.calls) == 1
    assert _ledger_rows() == []
    [ev] = _events()
    assert ev["payload"]["unmetered"] is True


def test_settle_failure_keeps_reservation_and_returns_response(tmp_path, monkeypatch):
    _init(tmp_path)

    def boom(self, res, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "settle", boom)
    resp = make_response("hi")
    w = pitstop.wrap(FakeOpenAI([resp]))
    assert w.chat.completions.create(model=MODEL, messages=MSGS) is resp
    assert _states() == ["reserved"]  # fail safe: over-counted, never under-counted


def test_runtime_init_failure_does_not_break_user_call(tmp_path, monkeypatch):
    import pitstop.sdk.wrap as wrap_mod

    def boom():
        raise RuntimeError("bad config")

    monkeypatch.setattr(wrap_mod, "get_runtime", boom)
    resp = make_response("hi")
    w = pitstop.wrap(FakeOpenAI([resp]))
    assert w.chat.completions.create(model=MODEL, messages=MSGS) is resp


def test_unknown_model_uses_conservative_price_and_flags_run(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("hi", prompt_tokens=100, completion_tokens=10)]))
    with workflow("wf") as ctx:
        w.chat.completions.create(model="acme/mystery-1", messages=MSGS)
    [row] = _ledger_rows()
    assert row["usd"] == pytest.approx(cost_usd(CONSERVATIVE_PRICE, 100, 10))
    run = _db().execute("SELECT model, price_known FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    assert (run["model"], run["price_known"]) == ("acme/mystery-1", 0)


def test_later_unknown_model_still_flags_run(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("a"), make_response("b")]))
    with workflow("wf") as ctx:
        w.chat.completions.create(model=MODEL, messages=MSGS)
        w.chat.completions.create(model="acme/mystery-1", messages=MSGS)
    run = _db().execute("SELECT model, price_known FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    assert (run["model"], run["price_known"]) == (MODEL, 0)


# -- streams ------------------------------------------------------------------------------------------


def _chunks():
    return [
        make_chunk("Hel"),
        make_chunk("lo"),
        make_chunk(tool_calls=[make_tool_delta(0, "search", '{"q": ')]),
        make_chunk(tool_calls=[make_tool_delta(0, None, '"x"}')]),
        make_chunk(finish_reason="tool_calls"),
    ]


def test_stream_yields_all_chunks_and_settles_from_usage_chunk(tmp_path):
    _init(tmp_path)
    chunks = _chunks() + [make_usage_chunk(300, 40, cached_tokens=100)]
    fake = FakeOpenAI([chunks])
    w = pitstop.wrap(fake)
    with workflow("wf"), agent("streamer"):
        stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
        assert _states() == ["reserved"]
        got = list(stream)
    # Every provider chunk is yielded unchanged, except the usage-only chunk PitStop itself requested: the
    # caller did not ask for it, and code like chunk.choices[0] would break on its empty choices.
    assert len(got) == len(chunks) - 1 and all(a is b for a, b in zip(got, chunks))
    assert fake.calls[0]["stream_options"] == {"include_usage": True}
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(cost_usd(PRICE, 300, 40, 100))
    [ev] = _events()
    assert ev["agent"] == "streamer"
    assert ev["payload"]["tool_calls"] == [{"name": "search", "arguments": '{"q": "x"}'}]
    assert ev["payload"]["content_chars"] == 5 and ev["payload"]["finish_reason"] == "tool_calls"
    assert ev["cost_usd"] == pytest.approx(row["usd"])


def test_stream_usage_chunk_is_yielded_when_caller_asked_for_it(tmp_path):
    _init(tmp_path)
    chunks = [make_chunk("ab"), make_usage_chunk(30, 4)]
    fake = FakeOpenAI([chunks])
    w = pitstop.wrap(fake)
    opts = {"include_usage": True}
    got = list(w.chat.completions.create(model=MODEL, messages=MSGS, stream=True, stream_options=opts))
    assert len(got) == 2 and all(a is b for a, b in zip(got, chunks))
    assert fake.calls[0]["stream_options"] is opts
    [row] = _ledger_rows()
    assert row["usd"] == pytest.approx(cost_usd(PRICE, 30, 4))


def test_stream_keeps_caller_stream_options_and_delegates(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([[make_chunk("a")]])
    w = pitstop.wrap(fake)
    with w.chat.completions.create(model=MODEL, messages=MSGS, stream=True, stream_options=None) as stream:
        assert stream.response.status_code == 200
        assert [c.choices[0].delta.content for c in stream] == ["a"]
    assert fake.calls[0]["stream_options"] is None
    assert _states() == ["settled"]


def test_stream_without_usage_settles_by_estimate(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([_chunks()]))
    list(w.chat.completions.create(model=MODEL, messages=MSGS, stream=True))
    rows = _ledger_rows()
    assert [r["state"] for r in rows].count("settled") == 1
    assert [r["state"] for r in rows].count("reserved") == 0
    # 5 content chars + "search" + '{"q": "x"}' = 21 chars -> 6 tokens; prompt estimate 100.
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (100, 6)
    assert rows[0]["usd"] > 0
    assert _events()[0]["payload"]["usage_estimated"] is True


def test_stream_early_break_then_close_settles_from_seen(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([_chunks()])
    w = pitstop.wrap(fake)
    stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    for _ in stream:
        break
    assert _states() == ["reserved"]
    stream.close()
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled"]
    assert rows[0]["completion_tokens"] == 1  # "Hel" -> ceil(3/4)
    stream.close()  # idempotent
    assert len(_events()) == 1


def test_stream_early_break_then_dropped_settles_from_seen(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([_chunks()]))

    def consume():
        for c in w.chat.completions.create(model=MODEL, messages=MSGS, stream=True):
            return c.choices[0].delta.content

    assert consume() == "Hel"
    gc.collect()
    assert _states() == ["settled"]


def test_stream_error_midway_settles_and_propagates(tmp_path):
    _init(tmp_path)
    err = ConnectionError("reset")
    w = pitstop.wrap(FakeOpenAI([[make_chunk("abcd"), err]]))
    stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    with pytest.raises(ConnectionError) as ei:
        list(stream)
    assert ei.value is err
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled"] and rows[0]["completion_tokens"] == 1


async def test_async_stream_yields_and_settles(tmp_path):
    _init(tmp_path)
    chunks = _chunks() + [make_usage_chunk(300, 40)]
    fake = FakeAsyncOpenAI([chunks])
    w = pitstop.wrap(fake)
    async with workflow("wf"):
        stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
        got = [c async for c in stream]
    assert len(got) == len(chunks) - 1 and all(a is b for a, b in zip(got, chunks))
    assert fake.calls[0]["stream_options"] == {"include_usage": True}
    [row] = _ledger_rows()
    assert row["state"] == "settled" and row["usd"] == pytest.approx(cost_usd(PRICE, 300, 40))
    assert _events()[0]["payload"]["tool_calls"] == [{"name": "search", "arguments": '{"q": "x"}'}]


async def test_async_stream_without_usage_and_context_manager(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeAsyncOpenAI([_chunks()]))
    stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    async with stream as s:
        assert s.response.status_code == 200
        n = 0
        async for _ in s:
            n += 1
    assert n == 5
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled"] and rows[0]["completion_tokens"] == 6


async def test_async_stream_early_break_settles_from_seen(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeAsyncOpenAI([_chunks(), _chunks()]))
    stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    async for _ in stream:
        break
    assert _states() == ["reserved"]
    await stream.close()
    assert _states() == ["settled"]
    stream2 = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    async for _ in stream2:
        break
    del stream2
    gc.collect()
    assert _states() == ["settled", "settled"]


async def test_parallel_agents_attribute_their_own_calls(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeAsyncOpenAI([make_response("a"), make_response("b")]))

    async def worker(name, delay):
        with agent(name):
            await asyncio.sleep(delay)
            await w.chat.completions.create(model=MODEL, messages=MSGS)

    async with workflow("wf"):
        await asyncio.gather(worker("first", 0.01), worker("second", 0.03))
    assert [e["agent"] for e in _events()] == ["first", "second"]


# -- the real openai SDK over an in-process HTTP transport (no network) -----------------------------------


def _completion_json(content="hi", usage=(12, 3)):
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def _sse(include_usage: bool) -> bytes:
    def chunk(delta, finish=None, choices=True, usage=None):
        return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else [],
                "usage": usage}

    parts = [chunk({"role": "assistant", "content": "Hel"}), chunk({"content": "lo"}), chunk({}, "stop")]
    if include_usage:
        parts.append(chunk({}, choices=False, usage={"prompt_tokens": 21, "completion_tokens": 2,
                                                     "total_tokens": 23}))
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in parts) + "data: [DONE]\n\n"
    return body.encode()


def _handler(seen):
    import httpx

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            usage = (body.get("stream_options") or {}).get("include_usage", False)
            return httpx.Response(200, content=_sse(usage), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion_json())
    return handle


def test_real_openai_sync_client_non_stream_and_stream(tmp_path):
    import httpx
    import openai

    _init(tmp_path)
    seen = []
    client = openai.OpenAI(api_key="k", base_url="http://pitstop.test/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(_handler(seen))))
    w = pitstop.wrap(client)
    out = w.chat.completions.create(model=MODEL, messages=MSGS)
    assert out.choices[0].message.content == "hi"
    stream = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    text = "".join(c.choices[0].delta.content or "" for c in stream)  # must not hit an empty-choices chunk
    assert text == "Hello"
    assert seen[1]["stream_options"] == {"include_usage": True}
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled", "settled"]
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (12, 3)
    assert (rows[1]["prompt_tokens"], rows[1]["completion_tokens"]) == (21, 2)
    assert rows[0]["base_url"] == "http://pitstop.test/v1"


async def test_real_openai_async_client_non_stream_and_stream(tmp_path):
    import httpx
    import openai

    _init(tmp_path)
    seen = []
    client = openai.AsyncOpenAI(api_key="k", base_url="http://pitstop.test/v1",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(seen))))
    w = pitstop.wrap(client)
    out = await w.chat.completions.create(model=MODEL, messages=MSGS)
    assert out.choices[0].message.content == "hi"
    stream = await w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
    text = "".join([c.choices[0].delta.content or "" async for c in stream])
    assert text == "Hello"
    rows = _ledger_rows()
    assert [r["state"] for r in rows] == ["settled", "settled"]
    assert (rows[1]["prompt_tokens"], rows[1]["completion_tokens"]) == (21, 2)


def test_wrapped_client_supports_weakref():
    import weakref

    w = pitstop.wrap(FakeOpenAI([]))
    ref = weakref.ref(w)
    assert ref() is w
