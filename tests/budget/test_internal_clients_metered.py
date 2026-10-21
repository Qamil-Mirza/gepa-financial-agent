import asyncio

import pytest

import pitstop
from pitstop.correct.client import CrusoeClient
from pitstop.llm import LLMClient
from pitstop.sdk.errors import BudgetExceeded
from pitstop.sdk.runtime import get_runtime
from pitstop.sdk.wrap import meter
from tests.sdk.fakes import FakeAsyncOpenAI, FakeOpenAI, make_response

MODEL = "openai/gpt-oss-120b"
MSGS = [{"role": "user", "content": "hello"}]


def _init(tmp_path, budgets=""):
    (tmp_path / "config.toml").write_text(budgets)
    return pitstop.init(home=tmp_path)


def _rows():
    return [dict(r) for r in get_runtime().store.connect().execute("SELECT * FROM ledger ORDER BY id")]


def _inp():
    from pitstop.schemas import Budget, SupervisorInput

    return SupervisorInput(task_id="t", agent_id="a", swarm_objective="o", agent_directive="d", trajectory=[],
                           remaining_budget=Budget(steps_used=0, steps_max=1, research_calls_used=0,
                                                   research_calls_max=1),
                           evaluation_step=0, observation_cutoff_seq=0)


def test_runtime_llm_client_is_metered(tmp_path):
    _init(tmp_path)
    llm = LLMClient(MODEL, client=FakeAsyncOpenAI([make_response("hi")]))
    r = asyncio.run(llm.complete("sys", "user"))
    assert r.text == "hi"
    [row] = _rows()
    assert row["state"] == "settled" and row["workflow"] is None and row["meter"] == "agent"


def test_correct_client_hits_global_budget(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    fake = FakeOpenAI([make_response("hi")])
    c = CrusoeClient(MODEL, sync_client=fake, max_retries=3)
    with pytest.raises(BudgetExceeded):
        c.complete(MSGS)
    assert fake.calls == []
    assert c.tracker.retries == 0


def test_correct_client_async_budget_not_retried_or_wrapped(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    fake = FakeAsyncOpenAI([make_response("hi")])
    c = CrusoeClient(MODEL, async_client=fake, max_retries=3)
    with pytest.raises(BudgetExceeded):
        asyncio.run(c.acomplete(MSGS))
    assert fake.calls == [] and c.tracker.retries == 0


def test_correct_client_uses_correction_meter(tmp_path):
    _init(tmp_path)
    CrusoeClient(MODEL, sync_client=FakeOpenAI([make_response("hi")])).complete(MSGS)
    assert [r["meter"] for r in _rows()] == ["correction"]


def test_supervisor_calls_use_watch_meter(tmp_path):
    from pitstop.supervisor.supervisor import Supervisor

    _init(tmp_path)
    llm = LLMClient(MODEL, client=FakeAsyncOpenAI([make_response("not json")] * 4))
    sup = Supervisor.from_text(llm, "sys")
    seen = []
    orig = llm.complete

    async def spy(*a, **k):
        seen.append(True)
        return await orig(*a, **k)

    llm.complete = spy
    asyncio.run(sup.evaluate(_inp()))
    rows = _rows()
    assert rows and {r["meter"] for r in rows} == {"watch"}


def test_supervisor_propagates_budget_exceeded(tmp_path):
    from pitstop.supervisor.supervisor import Supervisor

    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    sup = Supervisor.from_text(LLMClient(MODEL, client=FakeAsyncOpenAI([make_response("{}")])), "sys")
    with pytest.raises(BudgetExceeded):
        asyncio.run(sup.evaluate(_inp()))


def test_meter_context_manager_nests_and_restores(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("a"), make_response("b"), make_response("c")]))
    with meter("x"):
        w.chat.completions.create(model=MODEL, messages=MSGS)
        with meter("y"):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    w.chat.completions.create(model=MODEL, messages=MSGS)
    assert [r["meter"] for r in _rows()] == ["x", "y", "agent"]
