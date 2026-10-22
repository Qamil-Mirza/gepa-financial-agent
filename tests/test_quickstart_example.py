import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_quickstart_offline_inject_loop_steers_then_halts(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--offline", "--inject-loop"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("steer:") == 2
    assert "Halted: steer limit exceeded" in out.stdout
    d = subprocess.run([sys.executable, "-m", "pitstop.cli", "drifts", "--all"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert d.returncode == 0, d.stderr
    line = next(ln for ln in d.stdout.splitlines() if "unproductive_loop" in ln)
    assert "P1" in line and "steer" in line


def test_quickstart_online_without_key_exits_clearly(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py")], cwd=tmp_path, env=env,
                         capture_output=True, text=True)  # cwd has no .env, so no real key can be loaded
    assert out.returncode != 0
    assert "CRUSOE_API_KEY" in (out.stdout + out.stderr)


def test_quickstart_help_creates_no_db(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path / "home")
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--help"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert not (tmp_path / "home" / "pitstop.db").exists()


def test_quickstart_reports_budget_exceeded_specifically(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    (tmp_path / "config.toml").write_text('[budgets.all]\nlimit_usd = 0.0000001\nperiod = "total"\n')
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--offline"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "BudgetExceeded: PitStop budget exceeded" in out.stdout


def test_quickstart_offline_gate_uses_scripted_judge_even_with_a_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    env["CRUSOE_API_KEY"] = "not-a-real-key"
    # Were the env key used, the judge would hit this closed loopback port (no outside network either way).
    (tmp_path / "config.toml").write_text('[judge]\nbase_url = "http://127.0.0.1:9/v1"\ntimeout_s = 1\n')
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--offline"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "[send_report] to=ops@example.com" in out.stdout
    import json
    import sqlite3

    db = sqlite3.connect(tmp_path / "pitstop.db")
    gates = [json.loads(p) for (p,) in db.execute("SELECT payload FROM events WHERE kind='gate'")]
    assert [(g["allow"], g["reason"]) for g in gates] == [(True, "offline demo")]
    assert db.execute("SELECT count(*) FROM events WHERE kind='gate_error'").fetchone()[0] == 0


def test_quickstart_offline_reports_invalid_config_as_halted(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    (tmp_path / "config.toml").write_text('[judge]\ntimeout_s = 0\n')
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--offline"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Halted: pitstop config invalid:" in out.stdout and "timeout_s" in out.stdout
