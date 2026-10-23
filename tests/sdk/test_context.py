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


def test_meter_spanning_a_yield_does_not_leak_to_the_consumer(tmp_path):
    from pitstop.sdk.wrap import _meter, meter

    pitstop.init(home=tmp_path)

    @workflow("g")
    def gen():
        with meter("judge"):
            yield _meter.get()
            yield _meter.get()  # restored when the step resumes
        yield _meter.get()

    it = gen()
    assert next(it) == "judge"
    assert _meter.get() == "agent"  # the consumer between yields keeps the default meter
    assert next(it) == "judge"
    assert _meter.get() == "agent"
    assert next(it) == "agent"
    assert list(it) == [] and _meter.get() == "agent"


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


async def test_async_gen_workflow_in_reference_cycle(tmp_path, monkeypatch, caplog):
    """Abandoned inside a reference cycle, the wrapper and the inner generator are finalized by the same
    gc.collect(). Only the wrapper may close the inner generator: the loop must not also schedule its own aclose(),
    which would race the wrapper's and fail with "already running"."""
    from pitstop.sdk import context

    pitstop.init(home=tmp_path)
    exits, seen, errors = [], [], []
    real_exit = context._Workflow.__exit__

    def spy(self, et, ev, tb):
        exits.append(et)
        return real_exit(self, et, ev, tb)

    monkeypatch.setattr(context._Workflow, "__exit__", spy)

    @workflow("ag")
    async def agen():
        try:
            while True:
                yield current_run().run_id
        finally:
            await asyncio.sleep(0.01)  # cleanup that suspends: a second aclose() would find it running
            r = current_run()
            seen.append(r.run_id if r else None)

    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, c: errors.append(c))
    try:
        it = agen()
        rid = await it.__anext__()
        cycle = [it]
        cycle.append(cycle)
        del it, cycle
        gc.collect()  # the loop's finalizer hook schedules the wrapper's aclose()
        for _ in range(20):
            await asyncio.sleep(0.01)
        gc.collect()  # a failed close task that nobody awaited reports "never retrieved" when it is collected
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(old_handler)
    assert exits == [None]  # the run ended exactly once
    assert seen == [rid]  # the inner generator's cleanup ran inside the run
    assert _row(rid)["status"] == "ok"
    assert errors == []  # no "aclose(): asynchronous generator is already running" from a second close task
    assert "already running" not in caplog.text


# -- refusing a run on an exhausted budget ----------------------------------------------------------------

_FLAT = '[prices."acme/flat"]\ninput = 1.0\noutput = 0.0\n'
_A, _B = "https://a.example/v1", "https://b.example/v1"


def _spend(usd, *, workflow=None, budget=None, base_url=_A):
    """Settle `usd` of agent spend on the runtime's ledger (today, outside any run)."""
    ledger = get_runtime().ledger
    res = ledger.reserve(model="acme/flat", base_url=base_url, est_prompt_tokens=round(usd * 1_000_000),
                         max_completion_tokens=0, run_id="earlier", workflow=workflow, workflow_budget_usd=budget)
    ledger.settle(res, prompt_tokens=round(usd * 1_000_000), completion_tokens=0)


def _runs(name):
    return get_runtime().store.connect().execute(
        "SELECT * FROM runs WHERE workflow=? ORDER BY started_at", (name,)).fetchall()


def test_exhausted_workflow_budget_refuses_run(tmp_path):
    from pitstop.detect.drifts import list_drifts
    from pitstop.sdk.errors import BudgetExceeded

    (tmp_path / "config.toml").write_text(_FLAT)
    pitstop.init(home=tmp_path)
    _spend(1.0, workflow="wf", budget=1.0)
    entered = []
    with pytest.raises(BudgetExceeded) as ei:
        with workflow("wf", budget_usd=1.0):
            entered.append(1)
    assert entered == [] and current_run() is None
    assert ei.value.scope == "workflow:wf" and ei.value.limit_usd == 1.0
    [row] = _runs("wf")
    assert row["status"] == "refused" and row["ended_at"] is not None and row["budget_usd"] == 1.0
    [d] = list_drifts(get_runtime().store)
    assert (d["drift_type"], d["level"], d["action"], d["workflow"], d["run_id"]) == \
        ("budget_exhausted", "P0", "halt", "wf", row["id"])
    assert d["evidence"][0]["detail"]["scope"] == "workflow:wf"
    # another workflow, and the same one without a budget, still start
    with workflow("wf2", budget_usd=1.0):
        pass
    with workflow("wf"):
        pass


def test_refused_generator_and_async_workflows_raise_on_entry(tmp_path):
    from pitstop.sdk.errors import BudgetExceeded

    (tmp_path / "config.toml").write_text(_FLAT)
    pitstop.init(home=tmp_path)
    _spend(1.0, workflow="wf", budget=1.0)
    body = []

    @workflow("wf", budget_usd=1.0)
    def gen():
        body.append(1)
        yield 1

    g = gen()
    with pytest.raises(BudgetExceeded):
        next(g)

    @workflow("wf", budget_usd=1.0)
    async def afn():
        body.append(2)

    with pytest.raises(BudgetExceeded):
        asyncio.run(afn())
    assert body == [] and current_run() is None
    assert [r["status"] for r in _runs("wf")] == ["refused", "refused"]


def test_endpoint_budget_checked_from_previous_run(tmp_path):
    import time

    from pitstop.sdk.errors import BudgetExceeded

    (tmp_path / "config.toml").write_text(_FLAT + f'[budgets.a]\nlimit_usd = 1.0\nperiod = "total"\nbase_url = "{_A}"\n')
    pitstop.init(home=tmp_path)
    _spend(1.0, base_url=_A)

    def previous(url):
        with get_runtime().store.transaction() as c:
            c.execute("INSERT INTO runs(id, workflow, started_at, ended_at, status, base_url) VALUES (?,?,?,?,?,?)",
                      (f"prev-{url}", "wf", time.time(), time.time(), "ok", url))

    previous(_B)
    with workflow("wf"):  # the previous run used B: B's budget is not used up
        pass
    previous(_A + "/")
    with pytest.raises(BudgetExceeded) as ei:
        with workflow("wf"):
            pass
    assert ei.value.scope == "global:a"
    assert _runs("wf")[-1]["status"] == "refused"
    with pytest.raises(BudgetExceeded):  # the refused run has no endpoint: the one before it still counts
        with workflow("wf"):
            pass


def test_endpoint_budget_ignores_otel_runs(tmp_path):
    import time

    (tmp_path / "config.toml").write_text(_FLAT + f'[budgets.a]\nlimit_usd = 1.0\nperiod = "total"\nbase_url = "{_A}"\n')
    pitstop.init(home=tmp_path)
    _spend(1.0, base_url=_A)
    with get_runtime().store.transaction() as c:
        c.execute("INSERT INTO runs(id, workflow, started_at, ended_at, status, base_url) VALUES (?,?,?,?,?,?)",
                  ("prev-sdk", "wf", time.time(), time.time(), "ok", _B))
        # a later observed run on A: OTel spend never reaches the ledger, so it never decides a refusal
        c.execute("INSERT INTO runs(id, workflow, started_at, ended_at, status, base_url, adapter)"
                  " VALUES (?,?,?,?,?,?,?)", ("otel-1", "wf", time.time() + 1, time.time() + 1, "ok", _A, "otel"))
    with workflow("wf") as ctx:
        pass
    assert _row(ctx.run_id)["status"] == "ok"


def test_refusal_check_fails_open(tmp_path, monkeypatch, caplog):
    import sqlite3

    from pitstop.budget.ledger import Ledger

    pitstop.init(home=tmp_path)

    def locked(self, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "exhausted", locked)
    with caplog.at_level("WARNING", logger="pitstop"):
        with workflow("wf", budget_usd=1.0) as ctx:
            assert current_run() is ctx
    assert _row(ctx.run_id)["status"] == "ok"
    assert [r for r in caplog.records if r.levelname == "WARNING"]


def test_refusal_check_keeps_config_invalid_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text("[budgets.a]\nlimit_usd = -1\n")
    with pytest.raises(Halted, match="pitstop config invalid"):
        with workflow("wf"):
            pass


async def test_framework_agent_names_the_calling_scope(tmp_path):
    """set_framework_agent (for framework hooks that run in a child task) names the agent of every task whose
    innermost scope is the one the hook saw; an inner agent() scope still names its own calls."""
    pitstop.init(home=tmp_path)
    with workflow("wf") as ctx:
        with agent("outer", goal="og"):
            await asyncio.create_task(_name("lead"))  # like a hook: a child task, its context a copy
            assert (ctx.agent, ctx.agent_goal) == ("lead", "og")  # keeps the enclosing scope's goal
            with agent("sub", goal="sg"):
                assert (ctx.agent, ctx.agent_goal) == ("sub", "sg")
            await asyncio.create_task(_name("next"))  # replaces the name set before
            assert ctx.agent == "next"
            await asyncio.create_task(_name(None))
            assert (ctx.agent, ctx.agent_goal) == ("outer", "og")
        assert ctx.agent is None  # the top level was never named
        await asyncio.create_task(_name("top"))
        assert (ctx.agent, ctx.agent_goal) == ("top", None)  # no enclosing scope: the workflow goal applies


async def _name(name):
    current_run().set_framework_agent(name)
