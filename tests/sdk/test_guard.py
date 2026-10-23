import asyncio
import json

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.detect.l0 import Finding
from pitstop.sdk.actions import apply_findings
from pitstop.sdk.context import WorkflowSpec, agent, workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.guard import (GuardDecision, aguard_call, checked, guard_call, record_tool_result,
                               tool_args_text)
from pitstop.sdk.runtime import get_runtime


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _events(kind):
    rows = get_runtime().store.connect().execute(
        "SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["payload"] = json.loads(r["payload"])
    return out


def test_tool_args_text_is_canonical():
    assert tool_args_text(("a",), {"z": 1, "b": 2}) == '{"args":["a"],"kwargs":{"b":2,"z":1}}'
    assert json.loads(tool_args_text((object(),), {}))["args"][0].startswith("<object object")


def test_guard_call_block_steer_run(tmp_path):
    _init(tmp_path)

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    @pitstop.tool
    def search(q):
        return "r"

    # the decorator's answers on one run ...
    with workflow("wf", goal="the goal", tools=["search"]):
        tool_block = send_email("x")
        tool_steps = [search("a") for _ in range(3)]
    # ... and the guard's on an identical one
    with workflow("wf", goal="the goal", tools=["search"]) as ctx:
        block = guard_call(ctx, "send_email", True, tool_args_text(("x",), {}))
        steps = [guard_call(ctx, "search", False, tool_args_text(("a",), {})) for _ in range(3)]

    assert block == GuardDecision("block", tool_block)
    assert [s.action for s in steps] == ["run", "run", "steer"]
    assert steps[2].reply == tool_steps[2] and steps[2].reply.startswith("PitStop: this step drifts")
    assert [s.reply for s in steps[:2]] == [None, None]
    assert len(_events("block")) == 2 and len(_events("steer")) == 2
    assert len(_events("tool_call")) == 4  # two runs of search per workflow


def test_guard_call_steer_limit_halts(tmp_path):
    _init(tmp_path)
    with pytest.raises(Halted, match="steer limit exceeded"):
        with workflow("wf") as ctx:
            for _ in range(5):
                guard_call(ctx, "search", False, tool_args_text(("a",), {}))
    assert len(_events("steer")) == 2 and len(_events("halt")) == 1


async def test_aguard_call_matches_guard_call(tmp_path):
    _init(tmp_path)
    async with workflow("wf", tools=["search"]) as ctx:
        d = await aguard_call(ctx, "send_email", True, tool_args_text(("x",), {}))
        ok = await aguard_call(ctx, "search", False, tool_args_text(("x",), {}))
    assert d.action == "block" and d.reply["blocked_by"] == "pitstop"
    assert ok == GuardDecision("run")


def test_guard_agent_override_keys_history(tmp_path):
    _init(tmp_path)
    with workflow("wf") as ctx:
        with agent("ctx-agent"):
            for _ in range(2):
                assert guard_call(ctx, "search", False, "{}", agent="framework-agent").action == "run"
            record_tool_result(ctx, "search", result="r", agent="framework-agent")
            assert len(ctx.tool_history_for("framework-agent")) == 2
            assert len(ctx.tool_history_for("ctx-agent")) == 0
            # a second agent's identical calls are not a loop with the first's
            assert guard_call(ctx, "search", False, "{}", agent="other").action == "run"
    assert {e["agent"] for e in _events("tool_call")} == {"framework-agent", "other"}
    [res] = _events("tool_result")
    assert res["agent"] == "framework-agent" and res["payload"] == {"result": "'r'"}


def test_record_tool_result_error(tmp_path):
    _init(tmp_path)
    with workflow("wf") as ctx:
        record_tool_result(ctx, "search", error=ValueError("boom"))
    [res] = _events("tool_result")
    assert res["payload"] == {"error": "ValueError('boom')"}


def test_nested_tool_checked_once(tmp_path):
    _init(tmp_path)
    calls = []

    @pitstop.tool(name="t")
    def t(q):
        calls.append(q)
        return "r"

    with workflow("wf") as ctx:
        assert guard_call(ctx, "t", False, tool_args_text(("a",), {})).action == "run"
        with checked(ctx, "t"):
            assert t("a") == "r"
        assert len(ctx.tool_history_for(None)) == 1
        # outside the mark the same tool is guarded again: a repeat counts
        assert t("a") == "r"
        assert len(ctx.tool_history_for(None)) == 2
    assert calls == ["a", "a"]
    assert len(_events("tool_call")) == 2  # the guard's and the unmarked call's, not the nested one's
    assert len(_events("tool_result")) == 2


def test_tool_marks_its_own_call_checked(tmp_path):
    _init(tmp_path)

    @pitstop.tool(name="inner")
    def inner(q):
        return q

    @pitstop.tool(name="inner")
    def outer(q):
        return inner(q)

    @pitstop.tool(name="inner")
    async def aouter(q):
        await asyncio.sleep(0)
        return inner(q)

    with workflow("wf") as ctx:
        assert outer("a") == "a"
        assert asyncio.run(aouter("b")) == "b"
        assert len(ctx.tool_history_for(None)) == 2
    assert len(_events("tool_call")) == 2


def test_checked_mark_is_per_run_and_tool(tmp_path):
    _init(tmp_path)

    @pitstop.tool(name="other")
    def other(q):
        return q

    with workflow("wf") as ctx:
        with checked(ctx, "t"):
            other("a")  # a different tool is still guarded
        assert len(ctx.tool_history_for(None)) == 1


def test_observe_only_records_without_acting(tmp_path):
    _init(tmp_path)
    finding = Finding("out_of_scope_action", "off_task", True, {"tool": "send_email"})
    with workflow("wf", criticality="high") as ctx:
        [a] = apply_findings(ctx, [finding], observe_only=True)
        assert ctx.halt_reason is None
    assert a.action == "record" and a.level.value == "P0"
    [d] = list_drifts(get_runtime().store)
    assert (d["drift_type"], d["level"], d["action"]) == ("out_of_scope_action", "P0", "record")
    assert _events("block") == [] and _events("halt") == []


def test_observe_only_without_ctx_uses_spec(tmp_path):
    _init(tmp_path)
    spec = WorkflowSpec("otel-wf", criticality="high")
    finding = Finding("out_of_scope_action", "off_task", True, {"tool": "send_email"})
    [a] = apply_findings(None, [finding], run_id="otel-abc", spec=spec, observe_only=True)
    assert a.action == "record"
    [d] = list_drifts(get_runtime().store)
    assert (d["workflow"], d["level"]) == ("otel-wf", "P0")


def test_checked_exit_in_another_context_never_leaks_marks(tmp_path):
    import contextvars

    from pitstop.sdk.guard import Mark, _checked

    _init(tmp_path)
    with workflow("wf") as ctx:
        entering = contextvars.copy_context()
        cm = checked(ctx, "t")

        def enter():
            _checked.set({(ctx.run_id, "outer"): Mark()})  # an outer mark in the entering context only
            cm.__enter__()

        entering.run(enter)
        foreign = contextvars.Context()
        foreign.run(cm.__exit__, None, None, None)
        assert foreign.run(_checked.get) == {}  # neither "outer" nor "t" installed there
        assert set(entering.run(_checked.get)) == {(ctx.run_id, "outer"), (ctx.run_id, "t")}


def test_checked_mark_tells_whether_the_result_was_recorded(tmp_path):
    """One call, one tool_result: an adapter records the result only when an inner @pitstop.tool did not. The
    mark is shared with nested checked() blocks of the same tool and with worker threads started inside."""
    _init(tmp_path)

    @pitstop.tool(name="t")
    def inner(x):
        return x

    with workflow("wf") as ctx:
        with checked(ctx, "t") as mark:
            assert mark.recorded is False
            with checked(ctx, "t") as again:
                assert again is mark
            assert asyncio.run(asyncio.to_thread(inner, 1)) == 1
            assert mark.recorded is True
        with checked(ctx, "t") as fresh:
            assert fresh is not mark and fresh.recorded is False
            record_tool_result(ctx, "other", result=1)
            assert fresh.recorded is False  # another tool's result is not this call's
    assert len(_events("tool_result")) == 2 and _events("tool_call") == []
