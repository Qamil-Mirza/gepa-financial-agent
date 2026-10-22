import asyncio
import gc

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


@pytest.mark.parametrize("kw", [
    {"criticality": "High"}, {"criticality": None}, {"sensitivity": "medium"}, {"fix_mode": "Manual"},
    {"fix_mode": ""}, {"on_gate_error": "deny"}, {"on_gate_error": "Block"},
])
def test_workflow_rejects_invalid_settings_at_definition(kw):
    name = next(iter(kw))
    with pytest.raises(ValueError, match=f"invalid {name}"):
        workflow("w", **kw)
    with pytest.raises(ValueError, match=f"invalid {name}"):
        WorkflowSpec("w", **kw)


def test_workflow_accepts_every_valid_setting():
    for crit in ("high", "medium", "low"):
        for sens in ("low", "normal", "high"):
            for fm in ("manual", "auto", None):
                for oge in ("block", "allow", None):
                    workflow("w", criticality=crit, sensitivity=sens, fix_mode=fm, on_gate_error=oge)


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


def test_shared_agent_instance_under_gather_is_isolated(tmp_path):
    pitstop.init(home=tmp_path)
    researcher = agent("researcher", goal="rg")
    writer = agent("writer", goal="wg")

    async def worker(scope, delay):
        async with scope:
            ctx = current_run()
            first = (ctx.agent, ctx.agent_goal)
            await asyncio.sleep(delay)
            with researcher:  # the same instance re-entered inside an overlapping task
                inner = ctx.agent
            return first, (ctx.agent, ctx.agent_goal), inner

    async def main():
        async with workflow("w"):
            ctx = current_run()
            got = await asyncio.gather(worker(researcher, 0.01), worker(researcher, 0.03),
                                       worker(writer, 0.02))
            return got, (ctx.agent, ctx.agent_goal)

    got, after = asyncio.run(main())
    assert got[0] == (("researcher", "rg"), ("researcher", "rg"), "researcher")
    assert got[1] == (("researcher", "rg"), ("researcher", "rg"), "researcher")
    assert got[2] == (("writer", "wg"), ("writer", "wg"), "researcher")
    assert after == (None, None)


def test_shared_workflow_instance_under_gather_gets_separate_runs(tmp_path):
    pitstop.init(home=tmp_path)
    wf = workflow("shared")

    async def one(delay, fail):
        async with wf:
            rid = current_run().run_id
            await asyncio.sleep(delay)
            assert current_run().run_id == rid
            if fail:
                raise KeyError("mine")
            return rid

    async def main():
        return await asyncio.gather(one(0.01, False), one(0.03, False), one(0.02, True), return_exceptions=True)

    r1, r2, err = asyncio.run(main())
    assert isinstance(err, KeyError) and err.args == ("mine",)
    assert r1 != r2
    assert _row(r1)["status"] == "ok" and _row(r2)["status"] == "ok"
    statuses = [r["status"] for r in get_runtime().store.connect().execute("SELECT status FROM runs")]
    assert sorted(statuses) == ["error", "ok", "ok"]
    assert current_run() is None


def test_workflow_on_generator_records_run_and_does_not_leak(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("g")
    def gen():
        yield current_run().run_id
        yield current_run().run_id

    it = gen()
    a = next(it)
    assert current_run() is None  # consumer code between yields sees no run
    assert _row(a)["status"] == "running"
    b = next(it)
    assert a == b
    assert list(it) == []
    row = _row(a)
    assert (row["workflow"], row["status"]) == ("g", "ok") and row["ended_at"] is not None


def test_generator_workflow_passes_send_throw_and_return_value(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("g")
    def gen():
        got = yield current_run().run_id
        try:
            yield got
        except KeyError:
            yield "caught"
        return "done"

    it = gen()
    rid = next(it)
    assert it.send("hello") == "hello"
    assert it.throw(KeyError("k")) == "caught"
    with pytest.raises(StopIteration) as ei:
        next(it)
    assert ei.value.value == "done"
    assert _row(rid)["status"] == "ok"


def test_generator_workflow_error_and_halt_status(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("g")
    def bad():
        yield current_run().run_id
        raise ValueError("boom")

    it = bad()
    rid = next(it)
    with pytest.raises(ValueError):
        next(it)
    assert _row(rid)["status"] == "error"

    @workflow("h")
    def halts():
        yield current_run().run_id
        raise Halted("stop")

    it = halts()
    rid = next(it)
    with pytest.raises(Halted):
        next(it)
    assert _row(rid)["status"] == "halted"


def test_generator_closed_early_finishes_run(tmp_path):
    pitstop.init(home=tmp_path)
    cleaned = []

    @workflow("g")
    def gen():
        try:
            while True:
                yield current_run().run_id
        finally:
            cleaned.append(current_run().run_id)

    it = gen()
    rid = next(it)
    it.close()
    assert cleaned == [rid]  # the generator's own finally still runs inside the run
    row = _row(rid)
    assert row["status"] == "ok" and row["ended_at"] is not None
    assert current_run() is None

    it = gen()  # dropped unfinished: best effort, the run still ends when the generator is collected
    rid = next(it)
    del it
    gc.collect()
    assert _row(rid)["status"] == "ok"


def test_generator_workflow_cannot_nest(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("g")
    def gen():
        yield 1

    with workflow("outer"):
        with pytest.raises(RuntimeError, match="cannot be nested"):
            next(gen())


async def test_workflow_on_async_generator(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("ag")
    async def agen():
        yield current_run().run_id
        await asyncio.sleep(0)
        got = yield current_run().run_id
        yield got

    ids = []
    it = agen()
    async for rid in it:
        assert current_run() is None
        ids.append(rid)
        if len(ids) == 2:
            break
    assert ids[0] == ids[1] and _row(ids[0])["status"] == "running"
    assert await it.asend("x") == "x"
    with pytest.raises(StopAsyncIteration):
        await it.__anext__()
    assert _row(ids[0])["status"] == "ok"

    @workflow("ag2")
    async def bad():
        yield current_run().run_id
        raise ValueError("boom")

    it = bad()
    rid = await it.__anext__()
    with pytest.raises(ValueError):
        await it.__anext__()
    assert _row(rid)["status"] == "error"

    it = agen()
    rid = await it.__anext__()
    await it.aclose()  # closed early
    row = _row(rid)
    assert row["status"] == "ok" and row["ended_at"] is not None


def test_agent_on_generator_scopes_each_step(tmp_path):
    pitstop.init(home=tmp_path)

    @agent("a", goal="ag")
    def steps():
        yield current_run().agent
        yield current_run().agent_goal

    with workflow("wf") as ctx:
        it = steps()
        assert next(it) == "a"
        assert ctx.agent is None  # the consumer is outside the agent scope
        assert list(it) == ["ag"]


def test_async_generator_left_open_closes_inside_the_run_at_loop_shutdown(tmp_path):
    pitstop.init(home=tmp_path)
    seen, keep, errors = [], [], []

    @workflow("ag")
    async def agen():
        try:
            while True:
                yield current_run().run_id
        finally:
            await asyncio.sleep(0.05)  # cleanup that suspends while asyncio shuts its generators down
            r = current_run()
            seen.append(r.run_id if r else None)

    async def main():
        asyncio.get_running_loop().set_exception_handler(lambda loop, c: errors.append(c))
        it = agen()
        keep.append(it)  # still referenced when asyncio.run ends: closed by loop.shutdown_asyncgens()
        keep.append(await it.__anext__())

    asyncio.run(main())
    rid = keep[1]
    assert errors == []
    assert seen == [rid]  # the generator's cleanup ran inside the run
    row = _row(rid)
    assert row["status"] == "ok" and row["ended_at"] is not None


async def test_async_generator_timeout_spanning_a_yield_still_fires(tmp_path):
    pitstop.init(home=tmp_path)

    @workflow("t")
    async def agen():
        async with asyncio.timeout(0.05):
            yield current_run().run_id
            await asyncio.sleep(5)  # the timeout set before the yield must cancel this
            yield "late"

    it = agen()
    rid = await it.__anext__()
    with pytest.raises(TimeoutError):
        await it.__anext__()
    assert _row(rid)["status"] == "error"


async def test_async_generator_steps_see_consumer_context_and_keep_their_own_agent(tmp_path):
    import contextvars

    pitstop.init(home=tmp_path)
    user_var = contextvars.ContextVar("user_var", default="unset")

    @workflow("w")
    async def agen():
        with agent("inner"):
            yield (current_run().agent, user_var.get())
            yield (current_run().agent, user_var.get())

    it = agen()
    assert await it.__anext__() == ("inner", "unset")
    assert current_run() is None
    user_var.set("set-by-consumer")  # set between steps: the next step runs in the consumer's context
    assert await it.__anext__() == ("inner", "set-by-consumer")
    await it.aclose()
