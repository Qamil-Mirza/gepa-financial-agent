import pytest

from pitstop.detect.changes import CHANGE_WINDOW_S, Change, recent_change
from pitstop.store import Store

NOW = 10.0  # every run below starts well inside the change window of NOW


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def _run(store, run_id, *, workflow="wf", model="m-a", base_url="https://a.example/v1", started_at=1.0,
         fallback_from=None):
    with store.transaction() as c:
        c.execute("INSERT INTO runs(id, workflow, started_at, status, model, base_url, fallback_from)"
                  " VALUES (?,?,?,?,?,?,?)", (run_id, workflow, started_at, "ok", model, base_url, fallback_from))


def test_recent_change_detects_model_and_base_url(store):
    _run(store, "r1", model="m-a", started_at=1.0)
    _run(store, "r2", model="m-b", base_url="https://b.example/v1", started_at=2.0)
    # the model is checked first, even when the base_url changed too
    assert recent_change(store, "wf", "r2", now=NOW) == Change("model", "m-a", "m-b", "r1", 2.0)

    _run(store, "r3", model="m-b", base_url="https://c.example/v1", started_at=3.0)
    assert recent_change(store, "wf", "r3", now=NOW) == Change("base_url", "https://b.example/v1",
                                                               "https://c.example/v1", "r2", 3.0)

    # a run without a model (no agent call yet) is skipped
    _run(store, "r4", model=None, base_url=None, started_at=4.0)
    _run(store, "r5", model="m-a", base_url="https://c.example/v1", started_at=5.0)
    assert recent_change(store, "wf", "r5", now=NOW) == Change("model", "m-b", "m-a", "r3", 5.0)


def test_later_runs_on_the_new_setup_still_see_the_change(store):
    _run(store, "r1", model="m-a", started_at=1.0)
    _run(store, "r2", model="m-b", started_at=2.0)
    _run(store, "r3", model="m-b", started_at=3.0)
    _run(store, "r4", model="m-b", started_at=4.0)
    # `at` is when the new setup started (its first run), not this run's start
    assert recent_change(store, "wf", "r4", now=NOW) == Change("model", "m-a", "m-b", "r1", 2.0)


def test_change_older_than_the_window_is_not_recent(store):
    _run(store, "r1", model="m-a", started_at=1.0)
    _run(store, "r2", model="m-b", started_at=2.0)
    _run(store, "r3", model="m-b", started_at=3.0)
    assert recent_change(store, "wf", "r3", now=2.0 + CHANGE_WINDOW_S) == Change("model", "m-a", "m-b", "r1", 2.0)
    assert recent_change(store, "wf", "r3", now=2.0 + CHANGE_WINDOW_S + 1) is None
    # the default clock is the real one: these runs started in 1970
    assert recent_change(store, "wf", "r3") is None


def test_ignore_model_skips_pinned_runs(store):
    _run(store, "r1", model="m-b", started_at=1.0)
    _run(store, "r2", model="pinned", started_at=2.0)  # runs while a fallback was active
    _run(store, "r3", model="m-b", started_at=3.0)
    assert recent_change(store, "wf", "r3", now=NOW) == Change("model", "pinned", "m-b", "r2", 3.0)
    assert recent_change(store, "wf", "r3", ignore_model="pinned", now=NOW) is None  # back on m-b: no change

    # a real change made while the fallback was active (m-b -> m-c) is still seen once it is cleared
    _run(store, "r4", model="pinned", started_at=4.0)
    _run(store, "r5", model="m-c", started_at=5.0)
    assert recent_change(store, "wf", "r5", ignore_model="pinned", now=NOW) == Change("model", "m-b", "m-c",
                                                                                      "r3", 5.0)


def test_fallback_runs_are_not_a_setup(store):
    _run(store, "r1", model="m-a", started_at=1.0)
    _run(store, "r2", model="m-b", started_at=2.0)
    _run(store, "r3", model="pinned", started_at=3.0, fallback_from="m-b")  # redirected to the pinned fallback
    _run(store, "r4", model="m-b", started_at=4.0)  # the fallback was cleared
    # no ignore_model needed: r3 is skipped by its fallback_from, so r4 is still on the m-a -> m-b change
    assert recent_change(store, "wf", "r4", now=NOW) == Change("model", "m-a", "m-b", "r1", 2.0)


def test_recent_change_none_without_history_or_change(store):
    _run(store, "r1", started_at=1.0)
    assert recent_change(store, "wf", "r1", now=NOW) is None  # no earlier run
    _run(store, "other", workflow="wf2", model="m-z", started_at=0.5)
    assert recent_change(store, "wf", "r1", now=NOW) is None  # another workflow's run is not history
    _run(store, "r2", started_at=2.0)
    assert recent_change(store, "wf", "r2", now=NOW) is None  # same model and base_url
    _run(store, "r3", model=None, base_url=None, started_at=3.0)
    assert recent_change(store, "wf", "r3", now=NOW) is None  # this run has no model yet
    assert recent_change(store, "wf", "missing", now=NOW) is None
    # a run started later is not "before" this one
    _run(store, "r0", model="m-q", started_at=0.1)
    _run(store, "r9", model="m-late", started_at=9.0)
    assert recent_change(store, "wf", "r0", now=NOW) is None
