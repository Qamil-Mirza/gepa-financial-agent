import json

from pitstop.detect.drifts import MAX_EVIDENCE, dismiss_drift, list_drifts, record_drift
from pitstop.detect.l0 import Finding
from pitstop.detect.urgency import Level
from pitstop.store import Store


def _store(tmp_path):
    return Store(tmp_path / "s.db")


def _details(row):
    return [e["detail"] for e in row["evidence"]]


def f(**ev):
    return Finding("unproductive_loop", "off_task", False, ev)


def test_record_drift_dedups_open_drift(tmp_path):
    s = _store(tmp_path)
    a = record_drift(s, run_id="r1", workflow="wf", finding=f(n=1), level=Level.P1, action="steer")
    b = record_drift(s, run_id="r2", workflow="wf", finding=f(n=2), level=Level.P1, action="steer")
    assert a == b
    rows = list_drifts(s)
    assert len(rows) == 1
    assert _details(rows[0]) == [{"n": 1}, {"n": 2}]
    assert [e["run_id"] for e in rows[0]["evidence"]] == ["r1", "r2"]
    assert rows[0]["occurrences"] == 2
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
    assert [e["detail"] for e in json.loads(raw)] == [{"n": 1}, {"n": 2}]


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
    assert _details(row) == [{"n": 1}, {"n": 2}, {"n": 3}]
    record_drift(s, run_id="r4", workflow="wf", finding=f(n=4), level="P2", action="halt")
    [row] = list_drifts(s)
    assert (row["level"], row["action"]) == ("P0", "halt")


def test_dedup_caps_evidence_and_counts_occurrences(tmp_path):
    s = _store(tmp_path)
    assert MAX_EVIDENCE == 20
    ids = {record_drift(s, run_id=f"r{i}", workflow="wf", finding=f(n=i), level=Level.P1, action="steer")
           for i in range(25)}
    assert len(ids) == 1
    [row] = list_drifts(s)
    assert row["occurrences"] == 25
    assert len(row["evidence"]) == MAX_EVIDENCE
    assert _details(row) == [{"n": i} for i in range(5, 25)]  # newest kept, oldest dropped
    for i, e in zip(range(5, 25), row["evidence"]):
        assert set(e) == {"run_id", "ts", "detail"}
        assert e["run_id"] == f"r{i}" and isinstance(e["ts"], float)
    assert row["last_seen"] > row["ts"]
    assert row["last_seen"] == row["evidence"][-1]["ts"]
    assert row["run_id"] == "r0"  # the drift row keeps the run that opened it


def test_new_drift_has_one_occurrence(tmp_path):
    s = _store(tmp_path)
    record_drift(s, run_id="r1", workflow="wf", finding=f(n=1), level=Level.P1, action="steer")
    [row] = list_drifts(s)
    assert row["occurrences"] == 1 and row["last_seen"] == row["ts"]
    assert row["evidence"] == [{"run_id": "r1", "ts": row["ts"], "detail": {"n": 1}}]


def test_fallback_ranks_between_record_and_steer(tmp_path):
    store = _store(tmp_path)
    brk = Finding("tool_call_breakage", "degraded", False, {})
    for action in ("record", "fallback", "record"):
        record_drift(store, run_id="r", workflow="wf", finding=brk, level="P1", action=action)
    assert [r["action"] for r in list_drifts(store)] == ["fallback"]
    record_drift(store, run_id="r", workflow="wf", finding=brk, level="P1", action="steer")
    assert [r["action"] for r in list_drifts(store)] == ["steer"]


def test_dismiss_drift_states(tmp_path):
    s = _store(tmp_path)
    a = record_drift(s, run_id="r1", workflow="wf", finding=f(n=1), level=Level.P1, action="steer")
    assert dismiss_drift(s, a) == "dismissed"
    assert dismiss_drift(s, a) == "not_open"
    assert dismiss_drift(s, a + 100) == "not_found"
    assert list_drifts(s, workflow="wf") == []
    b = record_drift(s, run_id="r2", workflow="wf", finding=f(n=2), level=Level.P1, action="steer")
    assert b != a
