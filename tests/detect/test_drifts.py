import json

from pitstop.detect.drifts import list_drifts, record_drift
from pitstop.detect.l0 import Finding
from pitstop.detect.urgency import Level
from pitstop.store import Store


def _store(tmp_path):
    return Store(tmp_path / "s.db")


def f(**ev):
    return Finding("unproductive_loop", "off_task", False, ev)


def test_record_drift_dedups_open_drift(tmp_path):
    s = _store(tmp_path)
    a = record_drift(s, run_id="r1", workflow="wf", finding=f(n=1), level=Level.P1, action="steer")
    b = record_drift(s, run_id="r2", workflow="wf", finding=f(n=2), level=Level.P1, action="steer")
    assert a == b
    rows = list_drifts(s)
    assert len(rows) == 1
    assert rows[0]["evidence"] == [{"n": 1}, {"n": 2}]
    assert (rows[0]["level"], rows[0]["kind"], rows[0]["status"]) == ("P1", "off_task", "open")
    # another workflow / drift type is a separate drift
    c = record_drift(s, run_id="r3", workflow="other", finding=f(), level="P2", action="steer")
    d = record_drift(s, run_id="r3", workflow="wf", finding=Finding("refusal_spike", "degraded", False, {}),
                     level=Level.P1, action="record")
    assert len({a, c, d}) == 3
    # closed drifts do not dedup
    s.connect().execute("UPDATE drifts SET status='closed' WHERE id=?", (a,))
    e = record_drift(s, run_id="r4", workflow="wf", finding=f(), level=Level.P1, action="steer")
    assert e not in {a, c, d}
    assert {r["id"] for r in list_drifts(s, open_only=False)} == {a, c, d, e}
    assert [r["id"] for r in list_drifts(s, workflow="wf")] == [d, e]


def test_record_drift_null_workflow_dedups(tmp_path):
    s = _store(tmp_path)
    a = record_drift(s, run_id=None, workflow=None, finding=f(n=1), level=Level.P2, action="steer")
    b = record_drift(s, run_id=None, workflow=None, finding=f(n=2), level=Level.P2, action="steer")
    assert a == b
    raw = s.connect().execute("SELECT evidence FROM drifts WHERE id=?", (a,)).fetchone()[0]
    assert json.loads(raw) == [{"n": 1}, {"n": 2}]


def test_record_drift_dedup_escalates_level_and_action(tmp_path):
    s = _store(tmp_path)
    a = record_drift(s, run_id="r1", workflow="wf", finding=f(n=1), level=Level.P1, action="steer")
    b = record_drift(s, run_id="r2", workflow="wf", finding=f(n=2), level=Level.P0, action="block")
    assert a == b
    [row] = list_drifts(s)
    assert (row["level"], row["action"]) == ("P0", "block")
    # a later, milder occurrence never downgrades the open drift
    record_drift(s, run_id="r3", workflow="wf", finding=f(n=3), level="P3", action="record")
    [row] = list_drifts(s)
    assert (row["level"], row["action"]) == ("P0", "block")
    assert row["evidence"] == [{"n": 1}, {"n": 2}, {"n": 3}]
    record_drift(s, run_id="r4", workflow="wf", finding=f(n=4), level="P2", action="halt")
    [row] = list_drifts(s)
    assert (row["level"], row["action"]) == ("P0", "halt")
