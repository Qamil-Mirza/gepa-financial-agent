import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pitstop.budget.ledger import Ledger
from pitstop.cli_report import budget_rows, drift_rows, workflow_rows
from pitstop.detect.drifts import record_drift
from pitstop.detect.l0 import Finding
from pitstop.store import Store
from pitstop.userconfig import BudgetSpec, UserConfig

REPO = Path(__file__).resolve().parents[1]


def _run(store, rid, wf, status="ok", crit="medium", started=None):
    with store.transaction(immediate=True) as c:
        c.execute("INSERT INTO runs(id, workflow, criticality, started_at, status, budget_usd) VALUES (?,?,?,?,?,?)",
                  (rid, wf, crit, started or time.time(), status, 2.5))


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
    assert r["budget_usd"] == 2.5


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


def test_drifts_shows_cleared_fallback(tmp_path):
    from pitstop.detect.changes import Change
    from pitstop.sdk import fallback

    store = Store(tmp_path / "pitstop.db")
    for wf in ("wf", "wf2"):
        record_drift(store, run_id=None, workflow=wf, finding=Finding("tool_call_breakage", "degraded", False, {}),
                     level="P1", action="fallback")
    fallback.activate(store, "wf2", model="pinned", drift_id=None, change=Change("model", "a", "b", "r0", 1.0))
    assert {r["workflow"]: r["action"] for r in drift_rows(store)} == {"wf": "fallback (cleared)",
                                                                       "wf2": "fallback"}
    fallback.clear(store, "wf2")
    assert {r["action"] for r in drift_rows(store)} == {"fallback (cleared)"}


def test_cli_status_empty_db_is_friendly(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    for cmd in ("status", "drifts", "budget"):
        out = subprocess.run([sys.executable, "-m", "pitstop.cli", cmd], cwd=tmp_path, env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "No runs yet" in out.stdout


def test_workflow_rows_spend_today_uses_configured_timezone(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = 1_760_000_000.0  # fixed instant; only the configured timezone decides where "today" starts
    for name in ("Pacific/Kiritimati", "Pacific/Pago_Pago"):
        tz = ZoneInfo(name)
        store = Store(tmp_path / f"{name.replace('/', '_')}.db")
        start = datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        _run(store, "r1", "wf", started=start + 1)
        _run(store, "r0", "wf", started=start - 1)
        with store.transaction(immediate=True) as c:
            for ts, usd in ((start + 1, 1.0), (start - 1, 10.0)):
                c.execute("INSERT INTO ledger(ts, workflow, meter, model, usd, state) VALUES (?,?,?,?,?,?)",
                          (ts, "wf", "agent", "m", usd, "settled"))
        [row] = workflow_rows(store, now, tz=tz)
        assert (row["spend_today_usd"], row["runs_today"]) == (1.0, 1), name


def _cli(tmp_path, cmd):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    return subprocess.run([sys.executable, "-m", "pitstop.cli", cmd], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def test_cli_invalid_config_is_a_clean_error(tmp_path):
    (tmp_path / "config.toml").write_text('[budgets.all]\nlimit_usd = 5\nperiod = "weekly"\n')
    for cmd in ("status", "drifts", "budget"):
        out = _cli(tmp_path, cmd)
        assert out.returncode == 2, (cmd, out.stdout, out.stderr)
        assert "Traceback" not in out.stderr + out.stdout
        assert "invalid period 'weekly'" in out.stdout + out.stderr


def test_drift_rows_newest_first_by_last_seen_within_a_level(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    for dt in ("a", "b"):
        record_drift(store, run_id=None, workflow="wf", finding=Finding(dt, "off_task", False, {}), level="P1",
                     action="record")
    with store.transaction(immediate=True) as c:  # "a" opened earlier but was seen most recently
        c.execute("UPDATE drifts SET ts=100.0, last_seen=500.0 WHERE drift_type='a'")
        c.execute("UPDATE drifts SET ts=200.0, last_seen=300.0 WHERE drift_type='b'")
    assert [r["drift_type"] for r in drift_rows(store)] == ["a", "b"]


def test_drift_rows_use_occurrences_and_last_seen(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    for i in range(3):
        record_drift(store, run_id=f"r{i}", workflow="wf", finding=Finding("loop", "off_task", False, {"i": i}),
                     level="P1", action="steer")
    with store.transaction(immediate=True) as c:
        c.execute("UPDATE drifts SET ts=100.0, last_seen=200.0, occurrences=42")
    [row] = drift_rows(store)
    assert (row["count"], row["first_seen"], row["last_seen"]) == (42, 100.0, 200.0)


def test_cli_drifts_shows_occurrences_column(tmp_path, monkeypatch):
    import io

    from rich.console import Console

    import pitstop
    from pitstop import cli

    rt = pitstop.init(home=tmp_path)
    _run(rt.store, "r1", "wf")
    for _ in range(3):
        record_drift(rt.store, run_id="r1", workflow="wf", finding=Finding("loop", "off_task", False, {}),
                     level="P1", action="steer")
    out = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=200))
    _main(monkeypatch, "drifts")
    header, row = [line.split() for line in out.getvalue().splitlines() if "action" in line or " loop " in line]
    assert "occurrences" in header and "evidence" not in header
    assert row[header.index("occurrences")] == "3"


def _main(monkeypatch, *argv):
    from pitstop import cli

    monkeypatch.setattr(sys, "argv", ["pitstop", *argv])
    cli.main()


def _ledger_states(ledger):
    return [r[0] for r in ledger.store.connect().execute("SELECT state FROM ledger ORDER BY id")]


def test_budget_reconcile_cli_dry_run_then_release(capsys, monkeypatch):
    from pitstop.sdk.runtime import get_runtime

    ledger = get_runtime().ledger
    now = time.time()
    with ledger.store.transaction(immediate=True) as c:  # stale and live rows far either side of 60 minutes
        for ts, state, wf in ((now - 7200, "reserved", "stale-wf"), (now - 7200, "settled", "done-wf"),
                              (now - 10, "reserved", "live-wf")):
            c.execute("INSERT INTO ledger(ts, run_id, workflow, meter, model, usd, state) VALUES (?,?,?,?,?,?,?)",
                      (ts, "r", wf, "agent", "m", 1.5, state))

    _main(monkeypatch, "budget", "reconcile")
    out = capsys.readouterr().out
    assert "stale-wf" in out and "live-wf" not in out and "done-wf" not in out
    assert "re-run with --release to free them" in out
    assert _ledger_states(ledger) == ["reserved", "settled", "reserved"]  # dry run changes nothing

    _main(monkeypatch, "budget", "reconcile", "--older-than-min", "0")  # threshold is honoured: both open rows
    out = capsys.readouterr().out
    assert "stale-wf" in out and "live-wf" in out and "done-wf" not in out
    assert _ledger_states(ledger) == ["reserved", "settled", "reserved"]

    _main(monkeypatch, "budget", "reconcile", "--older-than-min", "60", "--release")
    out = capsys.readouterr().out
    assert "stale-wf" in out and "live-wf" not in out  # the rows it freed are listed
    assert "Released 1 stale reservation" in out
    assert _ledger_states(ledger) == ["released", "settled", "reserved"]

    _main(monkeypatch, "budget", "reconcile")
    assert "No stale reservations" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["-5", "nan"])
def test_budget_reconcile_rejects_bad_age(monkeypatch, capsys, value):
    with pytest.raises(SystemExit) as ei:
        _main(monkeypatch, "budget", "reconcile", "--older-than-min", value)
    assert ei.value.code == 2
    assert "must be a non-negative number of minutes" in capsys.readouterr().err


def test_bare_budget_still_works_in_process(monkeypatch, capsys):
    _main(monkeypatch, "budget")
    assert "No runs yet" in capsys.readouterr().out
