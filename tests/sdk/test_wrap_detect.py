import json

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import BudgetExceeded, Halted
from pitstop.sdk.runtime import get_runtime
from pitstop.sdk.wrap import meter
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
    from pitstop.detect.l0 import Finding
    from pitstop.sdk.actions import Applied, apply_findings

    with workflow("wf2", criticality="high") as ctx:
        [applied] = apply_findings(ctx, [Finding("unproductive_loop", "off_task", False, {"x": 1})])
        assert isinstance(applied, Applied) and applied.action == "steer" and applied.drift_id == ctx.halt_drift_id
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
    assert rows[0]["evidence"][0]["detail"]["scope"] == "workflow:wf"


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


def _run_row(run_id):
    return get_runtime().store.connect().execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def test_refusal_spike_counts_whole_run_past_cap(tmp_path):
    from pitstop.sdk.context import HISTORY_MAX

    _init(tmp_path)
    refusal = make_response("I'm sorry, I can't do that")
    script = [refusal] + [make_response("fine") for _ in range(HISTORY_MAX)] + [refusal]
    w = pitstop.wrap(FakeOpenAI(script))
    with workflow("wf") as ctx:  # normal: 2 refusals per run -> spike
        for _ in range(len(script)):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    assert len(ctx.llm_history) == HISTORY_MAX and ctx.refusals == 2
    assert [r["drift_type"] for r in list_drifts(get_runtime().store)] == ["refusal_spike"]


def test_judge_meter_calls_skip_detection_and_run_tags(tmp_path):
    _init(tmp_path)
    judge = "google/gemma-4-31b-it"
    w = pitstop.wrap(FakeOpenAI([make_response("I'm sorry, I can't do that"), make_response("ok")]))
    with workflow("wf", sensitivity="high") as ctx:  # high: a single refusal is a spike
        with meter("watch"):
            w.chat.completions.create(model=judge, messages=MSGS)
        assert len(ctx.llm_history) == 0 and ctx.refusals == 0
        assert _run_row(ctx.run_id)["model"] is None
        w.chat.completions.create(model=MODEL, messages=MSGS)
    assert list_drifts(get_runtime().store) == []
    assert len(ctx.llm_history) == 1
    run = _run_row(ctx.run_id)
    assert (run["model"], run["adapter"], run["price_known"]) == (MODEL, "wrap", 1)


def test_llm_event_payload_has_meter(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("a"), make_response("b"), make_response("c")]))
    with workflow("wf"):
        w.chat.completions.create(model=MODEL, messages=MSGS)
        with meter("watch"):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    w.chat.completions.create(model=MODEL, messages=MSGS)  # outside a workflow too
    rows = get_runtime().store.connect().execute(
        "SELECT payload FROM events WHERE kind='llm_call' ORDER BY id").fetchall()
    assert [json.loads(r["payload"])["meter"] for r in rows] == ["agent", "watch", "agent"]


def test_unknown_priced_watch_call_marks_price_unknown_without_tagging_the_model(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([make_response("verdict"), make_response("ok")]))
    with workflow("wf") as ctx:
        with meter("watch"):
            w.chat.completions.create(model="acme/mystery-1", messages=MSGS)
        run = _run_row(ctx.run_id)
        assert (run["model"], run["price_known"]) == (None, 0)
        w.chat.completions.create(model=MODEL, messages=MSGS)  # a known-priced agent call never raises it back
    run = _run_row(ctx.run_id)
    assert (run["model"], run["price_known"]) == (MODEL, 0)
