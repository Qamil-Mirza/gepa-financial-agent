import os
import subprocess
import sys
import time
from pathlib import Path

from pitstop.budget.ledger import Ledger
from pitstop.cli_report import budget_rows, drift_rows, workflow_rows
from pitstop.detect.drifts import record_drift
from pitstop.detect.l0 import Finding
from pitstop.store import Store
from pitstop.userconfig import BudgetSpec, UserConfig

REPO = Path(__file__).resolve().parents[1]


def _run(store, rid, wf, status="ok", crit="medium", started=None):
    with store.transaction(immediate=True) as c:
        c.execute("INSERT INTO runs(id, workflow, criticality, started_at, status) VALUES (?,?,?,?,?)",
                  (rid, wf, crit, started or time.time(), status))


def test_workflow_rows_counts_open_drifts_by_level(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _run(store, "r1", "wf", crit="high")
    _run(store, "r2", "wf", status="halted")
    for level, dt in (("P0", "a"), ("P1", "b"), ("P1", "c"), ("P3", "d")):
        record_drift(store, run_id="r2", workflow="wf", finding=Finding(dt, "off_task", False, {"x": 1}),
                     level=level, action="record")
    rows = workflow_rows(store, time.time())
    assert len(rows) == 1
    r = rows[0]
    assert (r["workflow"], r["runs_today"], r["open_p0"], r["open_p1"], r["open_p2"], r["open_p3"]) == \
        ("wf", 2, 1, 2, 0, 1)
    assert r["last_status"] == "halted"
    assert r["criticality"] == "medium"
    assert r["spend_today_usd"] == 0.0
    assert r["budget_usd"] is None


def test_budget_rows_show_remaining(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    config = UserConfig((BudgetSpec("crusoe", 5.0, "total", None),), {}, None)
    ledger = Ledger(store, config)
    with store.transaction(immediate=True) as c:
        c.execute("INSERT INTO ledger(ts, workflow, meter, model, usd, state) VALUES (?,?,?,?,?,?)",
                  (time.time(), "wf", "agent", "m", 1.25, "settled"))
    rows = budget_rows(ledger, config)
    assert rows == [{"scope": "global:crusoe", "period": "total", "spent_usd": 1.25, "limit_usd": 5.0,
                     "remaining_usd": 3.75}]


def test_drift_rows_sorted_by_level(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    record_drift(store, run_id=None, workflow="wf", finding=Finding("b", "off_task", False, {}), level="P2",
                 action="record")
    record_drift(store, run_id=None, workflow="wf", finding=Finding("a", "off_task", False, {}), level="P0",
                 action="halt")
    rows = drift_rows(store)
    assert [r["level"] for r in rows] == ["P0", "P2"]
    assert rows[0]["count"] == 1 and rows[0]["drift_type"] == "a"


def test_cli_status_empty_db_is_friendly(tmp_path):
    env = {**os.environ, "PITSTOP_HOME": str(tmp_path)}
    for cmd in ("status", "drifts", "budget"):
        out = subprocess.run([sys.executable, "-m", "pitstop.cli", cmd], cwd=REPO, env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "No runs yet" in out.stdout
