import asyncio

import pytest

import pitstop
from pitstop.sdk.context import WorkflowSpec, agent, check_halt, current_run, workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime


def _row(run_id):
    return get_runtime().store.connect().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def test_decorator_records_run_ok(tmp_path):
    pitstop.init(home=tmp_path)
    seen = []

    @workflow("wf", goal="g", criticality="high", sensitivity="high")
    def fn(x):
        seen.append(current_run())
        return x + 1

    assert fn.__name__ == "fn"
    assert fn(1) == 2
    assert current_run() is None
    row = _row(seen[0].run_id)
    assert (row["workflow"], row["goal"], row["criticality"], row["sensitivity"]) == ("wf", "g", "high", "high")
    assert row["status"] == "ok" and row["ended_at"] is not None and row["started_at"] is not None
    assert row["fix_mode"] == "manual"


def test_context_manager_sync_and_async(tmp_path):
    pitstop.init(home=tmp_path)
    with workflow("a") as _:
        r1 = current_run().run_id
    assert _row(r1)["status"] == "ok"

    async def go():
        async with workflow("b"):
            rid = current_run().run_id
            await asyncio.sleep(0)
            assert current_run().run_id == rid
        return rid

    assert _row(asyncio.run(go()))["status"] == "ok"

    @workflow("c")
    async def afn():
        return current_run().run_id

    assert _row(asyncio.run(afn()))["status"] == "ok"


def test_user_exception_propagates_and_run_marked_error(tmp_path):
    pitstop.init(home=tmp_path)
    boom = ValueError("boom")
    ids = []
    with pytest.raises(ValueError) as ei:
        with workflow("w"):
            ids.append(current_run().run_id)
            raise boom
    assert ei.value is boom
    assert _row(ids[0])["status"] == "error"


def test_halted_marks_run_halted(tmp_path):
    pitstop.init(home=tmp_path)
    ids = []
    with pytest.raises(Halted):
        with workflow("w"):
            ids.append(current_run().run_id)
            current_run().halt_reason, current_run().halt_drift_id = "bad", 3
            check_halt()
    assert _row(ids[0])["status"] == "halted"


def test_check_halt_noop_without_pending(tmp_path):
    pitstop.init(home=tmp_path)
    check_halt()
    with workflow("w"):
        check_halt()


def test_agent_sets_and_restores_name_and_goal(tmp_path):
    pitstop.init(home=tmp_path)
    with workflow("w"):
        ctx = current_run()
        assert ctx.agent is None
        with agent("outer", goal="og"):
            assert (ctx.agent, ctx.agent_goal) == ("outer", "og")
            with agent("inner"):
                assert (ctx.agent, ctx.agent_goal) == ("inner", None)
            assert (ctx.agent, ctx.agent_goal) == ("outer", "og")

        @agent("dec", goal="dg")
        def f():
            return (ctx.agent, ctx.agent_goal)

        assert f() == ("dec", "dg")
        assert ctx.agent is None
    with agent("noop"):  # outside a workflow: no-op
        assert current_run() is None


def test_nested_workflow_raises(tmp_path):
    pitstop.init(home=tmp_path)
    with workflow("a"):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with workflow("b"):
                pass
        assert current_run().spec.name == "a"


def test_concurrent_async_workflows_are_isolated(tmp_path):
    pitstop.init(home=tmp_path)

    async def one(name):
        async with workflow(name):
            with agent(name + "-agent"):
                await asyncio.sleep(0.01)
                await asyncio.sleep(0)
                ctx = current_run()
                return ctx.spec.name, ctx.run_id, ctx.agent

    async def main():
        return await asyncio.gather(one("x"), one("y"))

    (n1, id1, a1), (n2, id2, a2) = asyncio.run(main())
    assert (n1, a1) == ("x", "x-agent") and (n2, a2) == ("y", "y-agent")
    assert id1 != id2


def test_fix_mode_default_by_criticality(tmp_path):
    pitstop.init(home=tmp_path)
    got = {}
    for crit, fm in [("high", None), ("medium", None), ("low", None), ("high", "auto")]:
        with workflow("w", criticality=crit, fix_mode=fm):
            got[(crit, fm)] = _row(current_run().run_id)["fix_mode"]
    assert got == {("high", None): "manual", ("medium", None): "auto", ("low", None): "auto", ("high", "auto"): "auto"}
    assert WorkflowSpec("w").tools is None


def test_store_failure_never_surfaces(tmp_path, monkeypatch):
    rt = pitstop.init(home=tmp_path)

    def bad(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(rt.store, "transaction", bad)
    with workflow("w"):
        assert current_run() is not None
    assert current_run() is None


def test_init_idempotent_per_home(tmp_path):
    a = pitstop.init(home=tmp_path)
    assert pitstop.init(home=tmp_path) is a
    assert get_runtime() is a


def test_parallel_agents_in_one_workflow_are_isolated(tmp_path):
    pitstop.init(home=tmp_path)

    async def worker(name, goal, delay):
        with agent(name, goal=goal):
            ctx = current_run()
            seen = [(ctx.agent, ctx.agent_goal)]
            await asyncio.sleep(delay)
            seen.append((ctx.agent, ctx.agent_goal))
            await asyncio.sleep(0)
            seen.append((ctx.agent, ctx.agent_goal))
            return seen

    async def main():
        async with workflow("w"):
            ctx = current_run()
            a, b = await asyncio.gather(worker("a", "ga", 0.01), worker("b", "gb", 0.03))
            return a, b, (ctx.agent, ctx.agent_goal)

    a, b, after = asyncio.run(main())
    assert a == [("a", "ga")] * 3
    assert b == [("b", "gb")] * 3
    assert after == (None, None)
