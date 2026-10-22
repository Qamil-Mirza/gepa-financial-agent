import asyncio
import json

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.sdk.context import agent, workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import FakeOpenAI, make_response

MODEL = "openai/gpt-oss-120b"
MSGS = [{"role": "user", "content": "hi"}]


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _events(kind):
    rows = get_runtime().store.connect().execute(
        "SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    return [dict(r) for r in rows]


def test_tool_runs_normally_and_records_events(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return f"r:{q}"

    with workflow("wf"):
        assert search("a") == "r:a"
    assert len(_events("tool_call")) == 1
    res = _events("tool_result")
    assert len(res) == 1 and res[0]["name"] == "search"


def test_out_of_scope_side_effect_tool_is_blocked(tmp_path):
    _init(tmp_path)
    called = []

    @pitstop.tool(side_effects=True)
    def send_email(to):
        called.append(to)

    with workflow("wf", criticality="medium", tools=["search"]):
        out = send_email("x")
    assert called == []
    assert out["blocked_by"] == "pitstop" and out["reason"].startswith("out_of_scope_action: ")
    assert set(out) == {"blocked_by", "reason"}
    d = list_drifts(get_runtime().store)
    assert [(r["drift_type"], r["level"], r["action"]) for r in d] == [("out_of_scope_action", "P0", "block")]
    assert len(_events("block")) == 1


def test_high_criticality_p0_halts_at_next_boundary(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response(content="ok")]))

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    with pytest.raises(Halted):
        with workflow("wf", criticality="high", tools=["search"]):
            out = send_email("x")
            assert out["blocked_by"] == "pitstop"
            w.chat.completions.create(model=MODEL, messages=MSGS)


def test_read_only_loop_is_steered_twice_then_halts(tmp_path):
    _init(tmp_path)
    calls = []

    @pitstop.tool
    def search(q):
        calls.append(q)
        return "r"

    with pytest.raises(Halted, match="steer limit exceeded"):
        with workflow("wf", goal="wf goal"):
            assert search("a") == "r"
            assert search("a") == "r"
            assert search("a").startswith("PitStop: this step drifts")
            assert search("a").startswith("PitStop: this step drifts")
            search("a")
    assert len(calls) == 2
    assert len(_events("steer")) == 2


def test_steer_message_uses_agent_goal_then_workflow_goal(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return "r"

    with workflow("wf", goal="wf goal"):
        with agent("a", goal="agent goal"):
            for _ in range(3):
                m = search("a")
        assert m.endswith("Return to: agent goal.")
    with workflow("wf2", goal="wf goal"):
        for _ in range(3):
            m = search("a")
        assert m.endswith("Return to: wf goal.")
    with workflow("wf3"):
        for _ in range(3):
            m = search("a")
        assert m.endswith("Return to: your original task.")


def test_parallel_agents_similar_calls_are_not_a_loop(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return "r"

    async def run(name):
        with agent(name):
            await asyncio.sleep(0)
            for _ in range(2):
                assert search("same") == "r"

    async def main():
        async with workflow("wf"):
            await asyncio.gather(run("a"), run("b"))

    asyncio.run(main())
    assert list_drifts(get_runtime().store) == []


def test_async_tool(tmp_path):
    _init(tmp_path)

    @pitstop.tool(side_effects=True, name="mail")
    async def send(to):
        return "sent"

    async def main():
        async with workflow("wf", tools=["mail"]):
            assert await send("x") == "sent"
        async with workflow("wf", tools=["other"]):
            return await send("x")

    out = asyncio.run(main())
    assert out["blocked_by"] == "pitstop"


def test_tool_outside_workflow_just_runs(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def f(x):
        return x + 1

    assert f(1) == 2
    assert _events("tool_call") == []


def test_tool_exception_propagates(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def boom():
        raise ValueError("bad")

    with workflow("wf"):
        with pytest.raises(ValueError, match="bad"):
            boom()
    res = _events("tool_result")
    assert "error" in json.loads(res[0]["payload"])


def _broken_drift_store(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("drift store down")

    monkeypatch.setattr("pitstop.detect.drifts.record_drift", boom)


def test_block_does_not_depend_on_the_drift_write(tmp_path, monkeypatch):
    _init(tmp_path)
    _broken_drift_store(monkeypatch)
    called = []

    @pitstop.tool(side_effects=True)
    def send_email(to):
        called.append(to)

    with workflow("wf", criticality="medium", tools=["search"]):
        out = send_email("x")
    assert called == [] and out["blocked_by"] == "pitstop"
    assert len(_events("block")) == 1


def test_p0_halt_does_not_depend_on_the_drift_write(tmp_path, monkeypatch):
    _init(tmp_path)
    _broken_drift_store(monkeypatch)

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    with pytest.raises(Halted) as ei:
        with workflow("wf", criticality="high", tools=["search"]):
            assert send_email("x")["blocked_by"] == "pitstop"
            send_email("x")
    assert ei.value.drift_id is None


def test_steer_does_not_depend_on_the_drift_write(tmp_path, monkeypatch):
    _init(tmp_path)
    _broken_drift_store(monkeypatch)

    @pitstop.tool
    def search(q):
        return "r"

    with workflow("wf", goal="wf goal"):
        search("a")
        search("a")
        assert search("a").startswith("PitStop: this step drifts")


def test_steer_limit_halt_is_recorded_as_an_event(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def search(q):
        return "r"

    with pytest.raises(Halted, match="steer limit exceeded"):
        with workflow("wf", goal="wf goal"):
            for _ in range(5):
                search("a")
    [ev] = _events("halt")
    payload = json.loads(ev["payload"])
    assert ev["name"] == "search" and payload["reason"] == "steer limit exceeded"
    assert payload["drift"].startswith("unproductive_loop: ")


def test_tool_error_payload_is_capped(tmp_path):
    _init(tmp_path)

    @pitstop.tool
    def boom():
        raise ValueError("x" * 5000)

    with workflow("wf"):
        with pytest.raises(ValueError):
            boom()
    err = json.loads(_events("tool_result")[0]["payload"])["error"]
    assert len(err) == 500


def test_pending_p0_halt_records_halt_event(tmp_path):
    _init(tmp_path)

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    with pytest.raises(Halted):
        with workflow("wf", criticality="high", tools=["search"]) as ctx:
            assert send_email("x")["blocked_by"] == "pitstop"
            [ev] = _events("halt")
            assert (ev["run_id"], ev["name"]) == (ctx.run_id, "out_of_scope_action")
            assert json.loads(ev["payload"]) == {"reason": ctx.halt_reason, "drift_id": ctx.halt_drift_id,
                                                 "pending": True}
            assert ctx.halt_drift_id is not None
            send_email("x")  # the next boundary raises the pending halt
    assert len(_events("halt")) == 1


def test_histories_are_bounded(tmp_path):
    import uuid
    from collections import deque

    from pitstop.sdk.context import HISTORY_MAX

    _init(tmp_path)
    assert HISTORY_MAX == 64

    @pitstop.tool
    def search(q):
        return "r"

    with workflow("wf") as ctx:
        with agent("a"):
            for _ in range(200):
                assert search(uuid.uuid4().hex) == "r"
    hist = ctx.tool_history["a"]
    assert isinstance(hist, deque) and hist.maxlen == HISTORY_MAX and len(hist) == HISTORY_MAX
    assert isinstance(ctx.llm_history, deque) and ctx.llm_history.maxlen == HISTORY_MAX


class _SlowLimit:
    """MAX_STEERS stand-in whose comparison yields the GIL, widening the steer check-to-increment window."""

    def __le__(self, steers):  # `ctx.steers >= MAX_STEERS` falls back to this reflected comparison
        import time

        time.sleep(0.002)
        return 2 <= steers


def test_steer_limit_is_atomic_across_threads(tmp_path, monkeypatch):
    import contextvars
    import threading

    _init(tmp_path)
    monkeypatch.setattr("pitstop.sdk.tool.MAX_STEERS", _SlowLimit())

    @pitstop.tool
    def search(q):
        return "r"

    outcomes = []
    with workflow("wf", goal="g") as ctx:
        assert isinstance(ctx.lock, type(threading.Lock()))
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            for _ in range(50):
                try:
                    search("a")
                except Halted:
                    outcomes.append("halted")
                    return
            outcomes.append("never halted")

        threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert ctx.steers == 2 and len(_events("steer")) == 2
    assert outcomes == ["halted"] * 8
