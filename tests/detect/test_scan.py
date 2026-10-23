"""The background `scan` pass: L2 over finished runs, L5 attribution on an alarm, attributed drifts."""
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone

import pytest

from pitstop.detect.drifts import list_drifts
from pitstop.detect.scan import ScanAlarm, ScanReport, scan
from pitstop.store import Store

from tests.detect.run_factory import make_run

T0 = datetime(2026, 10, 22, 14, 5, tzinfo=timezone.utc).timestamp()
ISO = "2026-10-22T14:05Z"


def _runs(store, n, *, start=0, model="m1", tokens=120, seed=7, workflow="wf", **kw) -> list[str]:
    """`n` runs one minute apart (from T0 + start minutes), each with one agent call of about `tokens` tokens."""
    rng = random.Random(seed + start)
    ids = []
    for i in range(start, start + n):
        prompt = tokens - 20 + rng.randint(-4, 4)
        ids.append(make_run(store, workflow, started_at=T0 + i * 60, ended_at=T0 + i * 60 + 5, model=model,
                            llm_calls=((prompt, 20, 400),), **kw))
    return ids


def _state(store, metric, workflow="wf") -> dict:
    row = store.connect().execute("SELECT state FROM l2_state WHERE workflow=? AND metric=?",
                                  (workflow, metric)).fetchone()
    return json.loads(row["state"])


def _model_change(store, **kw):
    _runs(store, 30, **kw)  # the baseline completes exactly at run 30
    _runs(store, 15, start=30, model="m2", tokens=300, **kw)


def test_scan_alarm_with_failed_attribution(tmp_path, monkeypatch):
    from pitstop.detect import scan as scan_mod

    def boom(*a, **k):
        raise RuntimeError("attribution exploded")

    monkeypatch.setattr(scan_mod, "attribute", boom)
    store = Store(tmp_path / "p.db")
    _model_change(store)
    report = scan(store)
    assert report.alarms
    assert {a.summary for a in report.alarms} == {"attribution failed (RuntimeError)"}
    assert all(a.drift_id is not None for a in report.alarms)  # the drift is still recorded


def test_stationary_workflow_no_alarms(tmp_path):
    store = Store(tmp_path / "p.db")
    _runs(store, 60)
    assert scan(store) == ScanReport(runs=60, alarms=())
    assert list_drifts(store) == []


def test_token_jump_after_model_change_is_p1(tmp_path):
    store = Store(tmp_path / "p.db")
    _model_change(store)
    report = scan(store)
    assert report.runs == 45
    assert report.alarms and all(isinstance(a, ScanAlarm) for a in report.alarms)
    first = report.alarms[0]
    assert (first.workflow, first.metric, first.drift_type, first.level) == (
        "wf", "tokens_per_call", "cost_or_latency_creep", "P1")
    assert first.summary.startswith("model changed from m1 to m2")
    drifts = list_drifts(store)
    assert [(d["drift_type"], d["level"], d["action"]) for d in drifts] == [("cost_or_latency_creep", "P1", "record")]
    assert {a.drift_id for a in report.alarms} == {drifts[0]["id"]}
    detail = drifts[0]["evidence"][0]["detail"]
    assert detail["layer"] == "L2" and detail["metric"] == "tokens_per_call" and detail["direction"] == "up"
    assert detail["change_at"] == T0 + 30 * 60  # the first m2 run
    assert detail["baseline_mean"] < 130 < 280 < detail["value"]
    assert detail["attribution"]["summary"] == f"model changed from m1 to m2 (100% of runs since {ISO[:-3]}35Z)"
    assert {"tag": "model", "before": "m1", "after": "m2", "share_before": 0.0, "share_after": 1.0} \
        in detail["attribution"]["changes"]
    for entry in drifts[0]["evidence"]:
        assert entry["detail"]["attribution"]["summary"].startswith("model changed from m1 to m2")


def test_low_criticality_escalates_only_to_p2(tmp_path):
    store = Store(tmp_path / "p.db")
    _model_change(store, criticality="low")
    report = scan(store)
    assert report.alarms and {a.level for a in report.alarms} == {"P2"}  # P3 without the model change
    assert [d["level"] for d in list_drifts(store)] == ["P2"]


def test_shift_without_tag_change_is_not_escalated(tmp_path):
    store = Store(tmp_path / "p.db")
    _runs(store, 30)
    _runs(store, 15, start=30, tokens=300)  # same model: an input or upstream shift
    report = scan(store)
    assert report.alarms and {a.level for a in report.alarms} == {"P2"}
    assert report.alarms[0].summary.startswith("no run tag changed around")


def test_rescan_is_noop(tmp_path):
    store = Store(tmp_path / "p.db")
    _model_change(store)
    scan(store)
    states = store.connect().execute("SELECT * FROM l2_state ORDER BY metric").fetchall()
    drifts = list_drifts(store)
    assert scan(store) == ScanReport(runs=0, alarms=())
    assert store.connect().execute("SELECT * FROM l2_state ORDER BY metric").fetchall() == states
    assert list_drifts(store) == drifts


def test_limit_caps_one_pass(tmp_path):
    store = Store(tmp_path / "p.db")
    _runs(store, 12)
    assert scan(store, limit=5).runs == 5
    assert scan(store, limit=5).runs == 5
    assert scan(store).runs == 2
    assert _state(store, "tokens_per_call")["n"] == 12


def test_unfinished_runs_wait(tmp_path):
    store = Store(tmp_path / "p.db")
    _runs(store, 3)
    open_id = make_run(store, "wf", started_at=T0 + 600, ended_at=None)
    assert scan(store).runs == 3
    assert store.connect().execute("SELECT scanned_at FROM runs WHERE id=?", (open_id,)).fetchone()[0] is None
    with store.transaction() as c:
        c.execute("UPDATE runs SET ended_at=? WHERE id=?", (T0 + 605, open_id))
    assert scan(store).runs == 1
    assert _state(store, "tokens_per_call")["n"] == 4


def test_runs_without_metrics_are_marked_scanned(tmp_path):
    store = Store(tmp_path / "p.db")
    ids = [make_run(store, "wf", started_at=T0, ended_at=T0, status=status, llm_calls=())
           for status in ("refused", "dropped")]
    assert scan(store) == ScanReport(runs=2, alarms=())
    marks = [store.connect().execute("SELECT scanned_at FROM runs WHERE id=?", (i,)).fetchone()[0] for i in ids]
    assert all(m is not None for m in marks)
    assert store.connect().execute("SELECT COUNT(*) FROM l2_state").fetchone()[0] == 0


def test_concurrent_scans_process_each_run_once(tmp_path):
    path = tmp_path / "p.db"
    _runs(Store(path), 15)
    barrier, reports, errors = threading.Barrier(2), [], []

    def worker():
        try:
            store = Store(path)
            barrier.wait()
            reports.append(scan(store))
            store.close_local()
        except BaseException as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sum(r.runs for r in reports) == 15
    store = Store(path)
    assert store.connect().execute("SELECT COUNT(*) FROM runs WHERE scanned_at IS NULL").fetchone()[0] == 0
    assert _state(store, "tokens_per_call")["n"] == 15  # one detector state, each run fed once
    assert store.connect().execute("SELECT COUNT(*) FROM l2_state WHERE metric='tokens_per_call'").fetchone()[0] == 1


def test_scan_state_survives_restart(tmp_path):
    path = tmp_path / "p.db"
    store = Store(path)
    _runs(store, 25)
    assert scan(store).runs == 25
    assert _state(store, "tokens_per_call")["n"] == 25
    store.close_local()
    store = Store(path)
    _runs(store, 10, start=25)
    assert scan(store).runs == 10
    assert _state(store, "tokens_per_call")["n"] == 30  # continued (the baseline froze at WARMUP), not relearned


def test_corrupt_l2_state_restarts_detector(tmp_path, caplog):
    store = Store(tmp_path / "p.db")
    with store.transaction() as c:
        c.executemany("INSERT INTO l2_state(workflow, metric, state, updated_at) VALUES ('wf', ?, ?, 0)",
                      [("tokens_per_call", "{not json"), ("cost_usd", '{"n": "seven"}')])
    _runs(store, 3)
    with caplog.at_level(logging.WARNING, logger="pitstop"):
        assert scan(store) == ScanReport(runs=3, alarms=())
    assert _state(store, "tokens_per_call")["n"] == 3 and _state(store, "cost_usd")["n"] == 3
    warnings = [r.getMessage() for r in caplog.records if "l2_state" in r.getMessage()]
    assert len(warnings) == 2  # once per (workflow, metric), not once per run
    assert "{not json" not in caplog.text


def _cli_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    return env


def _cli(tmp_path, *argv):
    return subprocess.run([sys.executable, "-m", "pitstop.cli", *argv], cwd=tmp_path, env=_cli_env(tmp_path),
                          capture_output=True, text=True, timeout=60)


def test_scan_cli_once(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    _model_change(store)
    out = _cli(tmp_path, "scan")
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "scanned 45 runs, 2 alarms"
    summary = f"model changed from m1 to m2 (100% of runs since {ISO[:-3]}35Z)"
    assert lines[1] == f"P1 wf cost_or_latency_creep (tokens_per_call): {summary}"
    assert lines[2].startswith("P1 wf cost_or_latency_creep (cost_usd): model changed from m1 to m2")
    again = _cli(tmp_path, "scan")
    assert again.returncode == 0 and again.stdout.splitlines() == ["scanned 0 runs, 0 alarms"]


def test_scan_cli_counts_one_run_in_the_singular(tmp_path):
    _runs(Store(tmp_path / "pitstop.db"), 1)
    out = _cli(tmp_path, "scan")
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.splitlines() == ["scanned 1 run, 0 alarms"]


@pytest.mark.parametrize("every", ["inf", "nan", "0", "-1", "soon"])
def test_scan_cli_rejects_a_bad_interval(tmp_path, every):
    out = _cli(tmp_path, "scan", f"--every={every}")
    assert out.returncode == 2
    assert "--every" in out.stderr and "Traceback" not in out.stderr


def test_scan_cli_every_loops_until_ctrl_c(tmp_path):
    _runs(Store(tmp_path / "pitstop.db"), 2)
    proc = subprocess.Popen([sys.executable, "-m", "pitstop.cli", "scan", "--every", "0.2"], cwd=tmp_path,
                            env=_cli_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "scanned 2 runs, 0 alarms"
        assert proc.stdout.readline().strip() == "scanned 0 runs, 0 alarms"  # the next pass
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == 0, err
    assert "Traceback" not in err


def test_scan_cli_invalid_config_exits_2(tmp_path):
    (tmp_path / "config.toml").write_text("[budgets\n")
    out = _cli(tmp_path, "scan")
    assert out.returncode == 2
    assert "Invalid PitStop config" in out.stdout


def test_scan_alerts_use_scan_store(tmp_path, monkeypatch):
    import httpx

    import pitstop
    from pitstop.alerts import channels, dispatch

    monkeypatch.setenv("PITSTOP_WEBHOOK_URL", "https://hook.test/in")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.toml").write_text("[alerts.webhook]\n")
    rt = pitstop.init(home=tmp_path / "a")  # the runtime is on store A
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    store_b = Store(tmp_path / "b.db")
    _model_change(store_b)
    report = scan(store_b)
    assert report.alarms and dispatch.wait_idle()
    assert len(seen) == 1 and json.loads(seen[0].content)["drift_id"] == report.alarms[0].drift_id
    rows = [dict(r) for r in store_b.connect().execute("SELECT * FROM alerts")]
    assert [(r["channel"], r["status"]) for r in rows] == [("webhook", "sent")]
    assert rt.store.connect().execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0
