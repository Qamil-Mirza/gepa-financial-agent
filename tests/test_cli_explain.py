"""`pitstop explain <drift-id>`, run as a subprocess against a store under PITSTOP_HOME."""
import os
import subprocess
import sys
from datetime import datetime, timezone

from pitstop.detect.drifts import record_drift
from pitstop.detect.l0 import Finding
from pitstop.store import Store

from tests.detect.run_factory import make_run

T0 = datetime(2026, 10, 22, 14, 5, tzinfo=timezone.utc).timestamp()
ISO = "2026-10-22T14:05Z"


def _cli(tmp_path, *argv):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    return subprocess.run([sys.executable, "-m", "pitstop.cli", *argv], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.replace("┃", "│").strip().strip("│").split("│")]


def _store_with_model_change(tmp_path) -> Store:
    store = Store(tmp_path / "pitstop.db")
    for i in range(30):
        make_run(store, "wf", started_at=T0 - (30 - i) * 60, ended_at=T0 - (30 - i) * 60 + 5)
    for i in range(20):
        make_run(store, "wf", started_at=T0 + i * 60, ended_at=T0 + i * 60 + 5, model="m2",
                 llm_calls=((200, 40, 400),))
    return store


def test_explain_cli_prints_summary(tmp_path):
    store = _store_with_model_change(tmp_path)
    # an L2 drift: the newest evidence entry names the change point and the metric
    did = record_drift(store, run_id=None, workflow="wf", level="P2", action="record",
                       finding=Finding("cost_or_latency_creep", "degraded", False,
                                       {"change_at": T0 - 3600, "metric": "duration_s"}))
    record_drift(store, run_id=None, workflow="wf", level="P2", action="record",
                 finding=Finding("cost_or_latency_creep", "degraded", False,
                                 {"change_at": T0, "metric": "tokens_per_call"}))
    out = _cli(tmp_path, "explain", str(did))
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert f"model changed from m1 to m2 (100% of runs since {ISO})" in lines
    rows = [_cells(line) for line in lines if "│" in line or "┃" in line]
    assert rows[0] == ["tag", "before", "after", "share before", "share after"]
    assert ["model", "m1", "m2", "0%", "100%"] in rows
    assert "tokens_per_call mean: 120 over 30 runs before, 240 over 20 runs after" in lines


def test_explain_without_change_point_uses_drift_ts(tmp_path):
    store = _store_with_model_change(tmp_path)
    did = record_drift(store, run_id=None, workflow="wf", level="P1", action="record",
                       finding=Finding("tool_call_breakage", "degraded", False, {"rate": 0.5}))
    with store.transaction() as c:
        c.execute("UPDATE drifts SET ts=? WHERE id=?", (T0 + 10 * 60, did))  # m2 has 25% of the runs before
    out = _cli(tmp_path, "explain", str(did))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no run tag changed around 2026-10-22T14:15Z; likely an input or upstream shift" in out.stdout
    assert "runs compared: 40 before, 10 after" in out.stdout.splitlines()


def test_explain_unknown_drift_exits_1(tmp_path):
    Store(tmp_path / "pitstop.db")
    out = _cli(tmp_path, "explain", "42")
    assert out.returncode == 1
    assert "no drift 42" in out.stdout + out.stderr
    assert "Traceback" not in out.stderr


def test_explain_exit_2_on_invalid_config(tmp_path):
    (tmp_path / "config.toml").write_text("[budgets\n")
    out = _cli(tmp_path, "explain", "1")
    assert out.returncode == 2
    assert "Invalid PitStop config" in out.stdout
    assert "Traceback" not in out.stderr


def test_explain_workflowless_drift_exits_1(tmp_path):
    store = Store(tmp_path / "pitstop.db")
    did = record_drift(store, run_id=None, workflow=None, level="P2", action="record",
                       finding=Finding("tool_call_breakage", "degraded", False, {"rate": 0.5}))
    out = _cli(tmp_path, "explain", str(did))
    assert out.returncode == 1
    assert f"drift {did} has no workflow" in out.stdout
    assert "Traceback" not in out.stderr
