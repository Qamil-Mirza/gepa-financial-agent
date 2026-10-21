import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import BudgetExceeded, Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import FakeOpenAI, make_response, make_tool_call

MODEL = "openai/gpt-oss-120b"
MSGS = [{"role": "user", "content": "hello"}]


def _init(tmp_path, budgets=""):
    (tmp_path / "config.toml").write_text(budgets)
    return pitstop.init(home=tmp_path)


def _bad():
    return make_response(tool_calls=[make_tool_call("search", '{"q": ')])


def test_wrap_records_breakage_drift_from_bad_tool_args(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([_bad() for _ in range(5)]))
    with workflow("wf", criticality="medium"):
        for _ in range(5):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    rows = list_drifts(get_runtime().store)
    assert len(rows) == 1
    r = rows[0]
    assert (r["workflow"], r["drift_type"], r["kind"], r["level"], r["action"]) == \
        ("wf", "tool_call_breakage", "degraded", "P1", "record")
    assert r["run_id"]


def test_wrap_breakage_outside_workflow_has_null_workflow(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([_bad() for _ in range(5)]))
    for _ in range(5):
        w.chat.completions.create(model=MODEL, messages=MSGS)
    # no run context -> no per-run history, so nothing accumulates; the call must still succeed
    assert list_drifts(get_runtime().store) == []


def test_p0_on_high_workflow_sets_pending_halt(tmp_path):
    _init(tmp_path)
    refusal = lambda: make_response("I'm sorry, I can't do that")  # noqa: E731
    w = pitstop.wrap(FakeOpenAI([refusal(), refusal(), refusal()]))
    with workflow("wf", criticality="high", sensitivity="high") as ctx:  # high: 1 refusal -> spike (P1)
        w.chat.completions.create(model=MODEL, messages=MSGS)
        assert ctx.halt_reason is None  # P1 never halts
    # breakage on a high workflow is P1 too; only P0 rows halt: simulate via a direct helper
    from pitstop.sdk import wrap as wrapmod

    with workflow("wf2", criticality="high") as ctx:
        wrapmod._apply_findings(ctx, [__import__("pitstop.detect.l0", fromlist=["Finding"]).Finding(
            "unproductive_loop", "off_task", False, {"x": 1})])
        assert ctx.halt_reason and ctx.halt_drift_id
        with pytest.raises(Halted):
            w.chat.completions.create(model=MODEL, messages=MSGS)


def test_budget_exhausted_records_p0_drift(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("hi")]))
    with pytest.raises(BudgetExceeded):
        with workflow("wf", budget_usd=0.000001):
            w.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "x" * 400}])
    rows = list_drifts(get_runtime().store)
    assert [(r["drift_type"], r["level"], r["action"], r["workflow"]) for r in rows] == \
        [("budget_exhausted", "P0", "halt", "wf")]
    assert rows[0]["evidence"][0]["scope"] == "workflow:wf"


def test_budget_exhausted_outside_workflow(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.000001\nperiod = "total"\n')
    w = pitstop.wrap(FakeOpenAI([make_response("hi")]))
    with pytest.raises(BudgetExceeded):
        w.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "x" * 400}])
    rows = list_drifts(get_runtime().store)
    assert [(r["drift_type"], r["workflow"], r["level"]) for r in rows] == [("budget_exhausted", None, "P0")]


def test_drift_store_failure_never_breaks_call(tmp_path, monkeypatch):
    _init(tmp_path)
    import pitstop.detect.drifts as d

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(d, "record_drift", boom)
    w = pitstop.wrap(FakeOpenAI([_bad() for _ in range(5)]))
    with workflow("wf"):
        for _ in range(5):
            assert w.chat.completions.create(model=MODEL, messages=MSGS) is not None
