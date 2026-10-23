"""L5 attribution: which run tags changed between the runs before and after a change point."""
from datetime import datetime, timezone

import pytest

from pitstop.detect.attribution import (MAX_SHARE_BEFORE, NEW_SHARE_AFTER, TAGS, WINDOW, attribute,
                                        explains_change)
from pitstop.detect.urgency import Level, escalate_for_change
from pitstop.store import Store

from tests.detect.run_factory import make_run

T0 = datetime(2026, 10, 22, 14, 5, 30, tzinfo=timezone.utc).timestamp()
ISO = "2026-10-22T14:05Z"


def _runs(store, n_before, n_after, *, before=None, after=None, wf="wf"):
    """`n_before` runs ending one minute apart before T0, then `n_after` runs from T0 on; `before` / `after` are
    make_run keyword arguments, or a function of the run's index returning them."""
    for i in range(n_before):
        kw = before(i) if callable(before) else (before or {})
        s = T0 - (n_before - i) * 60
        make_run(store, wf, started_at=s, ended_at=s + 5, **kw)
    for i in range(n_after):
        kw = after(i) if callable(after) else (after or {})
        s = T0 + i * 60
        make_run(store, wf, started_at=s, ended_at=s + 5, **kw)


def test_constants():
    assert TAGS == ("model", "base_url", "prompt_versions", "pitstop_version", "adapter")
    assert (WINDOW, NEW_SHARE_AFTER, MAX_SHARE_BEFORE) == (50, 0.5, 0.1)


def test_model_change_attributed(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 30, 20, after={"model": "m2"})
    a = attribute(store, "wf", T0)
    assert (a.workflow, a.at, a.n_before, a.n_after) == ("wf", T0, 30, 20)
    [c] = a.changes
    assert (c.tag, c.before, c.after, c.share_before, c.share_after) == ("model", "m1", "m2", 0.0, 1.0)
    assert a.summary == f"model changed from m1 to m2 (100% of runs since {ISO})"
    assert (a.metric, a.before_mean, a.after_mean) == (None, None, None)
    assert explains_change(a)


def test_several_changes_joined(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 10, after=lambda i: {"model": "m2", "base_url": "https://b/v1" if i < 6 else "https://a/v1"})
    a = attribute(store, "wf", T0)
    assert [c.tag for c in a.changes] == ["model", "base_url"]
    assert a.summary == (f"model changed from m1 to m2 (100% of runs since {ISO}); "
                         f"base_url changed from https://a/v1 to https://b/v1 (60% of runs since {ISO})")


def test_no_change_summary(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 30, 20)
    a = attribute(store, "wf", T0)
    assert a.changes == ()
    assert a.summary == f"no run tag changed around {ISO}; likely an input or upstream shift"
    assert not explains_change(a)


def test_no_runs_before(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 0, 5, after={"model": "m2"})
    _runs(store, 5, 0, wf="other")  # another workflow's runs are never compared
    a = attribute(store, "wf", T0)
    assert (a.n_before, a.n_after, a.changes) == (0, 5, ())
    assert a.summary == f"no runs before {ISO} to compare"


def test_minority_value_not_a_change(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 30, 20, after=lambda i: {"model": "m2" if i % 5 == 0 else "m1"})  # 20% on m2
    a = attribute(store, "wf", T0)
    assert a.changes == ()
    assert not explains_change(a)


def test_value_common_before_is_not_new(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    # m2 had 20% of the runs before (over MAX_SHARE_BEFORE), so taking over afterwards is not a new tag value
    _runs(store, 10, 10, before=lambda i: {"model": "m2" if i < 2 else "m1"}, after={"model": "m2"})
    assert attribute(store, "wf", T0).changes == ()


def test_prompt_version_change_explains(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 10, before={"prompt_versions": {"researcher": 1}}, after={"prompt_versions": {"researcher": 2}})
    a = attribute(store, "wf", T0)
    [c] = a.changes
    assert (c.tag, c.before, c.after) == ("prompt_versions", '{"researcher": 1}', '{"researcher": 2}')
    assert a.summary == ('prompt_versions changed from {"researcher": 1} to {"researcher": 2}'
                         f" (100% of runs since {ISO})")
    assert explains_change(a)


def test_prompt_versions_normalized_and_unset(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 10, after={"prompt_versions": {"b": 1, "a": 2}})
    with store.transaction() as c:  # a writer that did not sort its keys or used other spacing: same value
        c.execute("""UPDATE runs SET prompt_versions='{"b":1,"a":2}' WHERE started_at >= ? AND rowid % 2 = 0""",
                  (T0,))
    a = attribute(store, "wf", T0)
    [c] = a.changes
    assert (c.before, c.after, c.share_after) == (None, '{"a": 2, "b": 1}', 1.0)
    assert a.summary.startswith('prompt_versions changed from (unset) to {"a": 2, "b": 1} (100%')


def test_adapter_change_does_not_explain(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 10, after={"adapter": "langchain"})
    a = attribute(store, "wf", T0)
    assert [c.tag for c in a.changes] == ["adapter"]
    assert not explains_change(a)


def test_refused_and_dropped_runs_excluded(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 4, after={"model": "m2"})
    _runs(store, 3, 6, before={"status": "dropped"}, after={"status": "refused", "model": None})
    a = attribute(store, "wf", T0)
    assert (a.n_before, a.n_after) == (10, 4)
    assert [(c.tag, c.share_after) for c in a.changes] == [("model", 1.0)]


def test_window_takes_runs_nearest_the_change(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    # old runs on m0, the 5 just before T0 on m1; after: 5 on m2, then later runs back on m1
    _runs(store, 20, 20, before=lambda i: {"model": "m0" if i < 15 else "m1"},
          after=lambda i: {"model": "m2" if i < 5 else "m1"})
    a = attribute(store, "wf", T0, window=5)
    assert (a.n_before, a.n_after) == (5, 5)
    assert [(c.before, c.after) for c in a.changes] == [("m1", "m2")]


def test_metric_means(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _runs(store, 10, 5, before={"llm_calls": ((100, 20, 400),)}, after={"llm_calls": ((200, 40, 400),)})
    a = attribute(store, "wf", T0, metric="tokens_per_call")
    assert (a.metric, a.before_mean, a.after_mean) == ("tokens_per_call", 120.0, 240.0)
    b = attribute(store, "wf", T0, metric="tool_error_rate")  # no run has the metric
    assert (b.before_mean, b.after_mean) == (None, None)


P0, P1, P2, P3 = Level.P0, Level.P1, Level.P2, Level.P3


@pytest.mark.parametrize("level,criticality,expected", [
    (P0, "high", P0), (P1, "high", P1), (P2, "high", P1), (P3, "high", P1),
    (P0, "medium", P0), (P1, "medium", P1), (P2, "medium", P1), (P3, "medium", P1),
    (P0, "low", P0), (P1, "low", P1), (P2, "low", P2), (P3, "low", P2),
])
def test_escalate_for_change(level, criticality, expected):
    assert escalate_for_change(level, criticality) is expected
