import pytest

from pitstop.detect.changes import Change, recent_change
from pitstop.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "s.db")


def _run(store, run_id, *, workflow="wf", model="m-a", base_url="https://a.example/v1", started_at=1.0):
    with store.transaction() as c:
        c.execute("INSERT INTO runs(id, workflow, started_at, status, model, base_url) VALUES (?,?,?,?,?,?)",
                  (run_id, workflow, started_at, "ok", model, base_url))


def test_recent_change_detects_model_and_base_url(store):
    _run(store, "r1", model="m-a", started_at=1.0)
    _run(store, "r2", model="m-b", base_url="https://b.example/v1", started_at=2.0)
    ch = recent_change(store, "wf", "r2")
    # the model is checked first, even when the base_url changed too
    assert ch == Change("model", "m-a", "m-b", "r1", 2.0)

    _run(store, "r3", model="m-b", base_url="https://c.example/v1", started_at=3.0)
    assert recent_change(store, "wf", "r3") == Change("base_url", "https://b.example/v1",
                                                      "https://c.example/v1", "r2", 3.0)

    # the newest earlier run with a model wins; a run without a model (no agent call yet) is skipped
    _run(store, "r4", model=None, base_url=None, started_at=4.0)
    _run(store, "r5", model="m-a", base_url="https://c.example/v1", started_at=5.0)
    assert recent_change(store, "wf", "r5") == Change("model", "m-b", "m-a", "r3", 5.0)


def test_recent_change_none_without_history_or_change(store):
    _run(store, "r1", started_at=1.0)
    assert recent_change(store, "wf", "r1") is None  # no earlier run
    _run(store, "other", workflow="wf2", model="m-z", started_at=0.5)
    assert recent_change(store, "wf", "r1") is None  # another workflow's run is not history
    _run(store, "r2", started_at=2.0)
    assert recent_change(store, "wf", "r2") is None  # same model and base_url
    _run(store, "r3", model=None, base_url=None, started_at=3.0)
    assert recent_change(store, "wf", "r3") is None  # this run has no model yet
    assert recent_change(store, "wf", "missing") is None
    # a run started later is not "before" this one
    _run(store, "r0", model="m-q", started_at=0.1)
    _run(store, "r9", model="m-late", started_at=9.0)
    assert recent_change(store, "wf", "r0") is None
