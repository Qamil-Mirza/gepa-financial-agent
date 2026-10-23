"""L2 per-run metrics: what `run_metrics` reads from a run's events, and what it leaves out."""
import json

import pytest

from pitstop.detect.metrics import METRICS, MetricSpec, run_metrics
from pitstop.store import Store
from tests.detect.run_factory import make_run


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def _event(store, run_id, kind, payload, cost=None, name="x"):
    with store.transaction() as c:
        c.execute("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd) VALUES (?,?,?,?,?,?,?)",
                  (run_id, 1.0, "agent", kind, name, json.dumps(payload), cost))


def _llm(content_chars=400, prompt=100, completion=20, refused=False, meter="agent", **extra):
    return {"tool_calls": [], "finish_reason": "stop", "content_chars": content_chars, "meter": meter,
            "prompt_tokens": prompt, "completion_tokens": completion, "refused": refused, **extra}


def test_metric_table():
    assert METRICS == {
        "tool_error_rate": MetricSpec("tool_call_breakage", ("up",), 0.15, False),
        "refusal_rate": MetricSpec("refusal_spike", ("up",), 0.15, False),
        "tokens_per_call": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 1.0),
        "llm_calls": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 1.0),
        "cost_usd": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 0.0001),
        "duration_s": MetricSpec("cost_or_latency_creep", ("up",), 0.05, True, 0.1),
        "output_chars": MetricSpec("quality_drop", ("up", "down"), 0.05, True, 1.0),
    }


def test_run_metrics_from_events(store):
    rid = make_run(store, "wf", started_at=100.0, ended_at=112.5, llm_calls=((100, 20, 400), (200, 40, 600)),
                   tool_results=(False, True, False))
    m = run_metrics(store, rid)
    assert set(m) == set(METRICS)
    assert m["tool_error_rate"] == pytest.approx(1 / 3)
    assert m["refusal_rate"] == 0.0
    assert m["tokens_per_call"] == pytest.approx(180.0)
    assert m["llm_calls"] == 2.0
    assert m["cost_usd"] == pytest.approx(360e-6)
    assert m["duration_s"] == pytest.approx(12.5)
    assert m["output_chars"] == pytest.approx(500.0)


def test_missing_metrics_omitted(store):
    # no LLM calls, no tools, not ended: only the call count is known
    rid = make_run(store, "wf", started_at=100.0, ended_at=None, llm_calls=(), tool_results=())
    assert run_metrics(store, rid) == {"llm_calls": 0.0}
    # LLM calls but no tool calls: no tool_error_rate
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0)
    assert "tool_error_rate" not in run_metrics(store, rid)
    assert run_metrics(store, "no-such-run") == {}


@pytest.mark.parametrize("status", ["refused", "dropped"])
def test_refused_and_dropped_runs_have_no_metrics(store, status):
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0, tool_results=(True,), status=status)
    assert run_metrics(store, rid) == {}


def test_null_fields_skipped_in_per_call_means(store):
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0, llm_calls=((100, 20, 400),))
    _event(store, rid, "llm_call", _llm(content_chars=None, prompt=None, completion=None, source="otel"), cost=None)
    m = run_metrics(store, rid)
    assert m["output_chars"] == 400.0 and m["tokens_per_call"] == 120.0  # the null call is skipped in both means
    assert m["llm_calls"] == 2.0 and m["cost_usd"] == pytest.approx(120e-6)  # but counted; its null cost is 0
    # every value null: the metric is left out
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0, llm_calls=())
    _event(store, rid, "llm_call", _llm(content_chars=None, prompt=None, completion=None, source="otel"))
    m = run_metrics(store, rid)
    assert "output_chars" not in m and "tokens_per_call" not in m
    assert m["llm_calls"] == 1.0 and m["cost_usd"] == 0.0


def test_two_tool_results_for_one_call_capped(store):
    # a nested @pitstop.tool can record two error results for one guarded call
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0, tool_results=(True,))
    _event(store, rid, "tool_result", {"error": "RuntimeError('boom')"}, name="tool0")
    assert run_metrics(store, rid)["tool_error_rate"] == 1.0


def test_otel_tool_results_and_refusals(store):
    rid = make_run(store, "wf", started_at=100.0, ended_at=101.0, llm_calls=((100, 20, 400),) * 3)
    for err in (None, "TimeoutError"):
        _event(store, rid, "tool_call", {"args": "", "source": "otel"})
        _event(store, rid, "tool_result", {"source": "otel"} | ({"error": err} if err else {}))
    _event(store, rid, "llm_call", _llm(refused=True), cost=0.5)
    _event(store, rid, "llm_call", _llm(meter="judge", refused=True), cost=9.0)  # judge calls never count
    m = run_metrics(store, rid)
    assert m["tool_error_rate"] == 0.5
    assert m["refusal_rate"] == 0.25 and m["llm_calls"] == 4.0
    assert m["cost_usd"] == pytest.approx(0.5 + 360e-6)


def test_duration_omitted_on_clock_skew(store):
    skewed = make_run(store, "wf", started_at=200.0, ended_at=100.0)
    assert "duration_s" not in run_metrics(store, skewed)
    ok = make_run(store, "wf", started_at=100.0, ended_at=105.0)
    assert run_metrics(store, ok)["duration_s"] == 5.0
