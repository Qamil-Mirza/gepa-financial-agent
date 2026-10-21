import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_quickstart_offline_inject_loop_steers_then_halts(tmp_path):
    env = {**os.environ, "PITSTOP_HOME": str(tmp_path)}
    env.pop("CRUSOE_API_KEY", None)
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py"), "--offline", "--inject-loop"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("steer:") == 2
    assert "Halted: steer limit exceeded" in out.stdout
    d = subprocess.run([sys.executable, "-m", "pitstop.cli", "drifts", "--all"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert d.returncode == 0, d.stderr
    line = next(ln for ln in d.stdout.splitlines() if "unproductive_loop" in ln)
    assert "P1" in line and "steer" in line


def test_quickstart_online_without_key_exits_clearly(tmp_path):
    env = {**os.environ, "PITSTOP_HOME": str(tmp_path)}
    env.pop("CRUSOE_API_KEY", None)
    out = subprocess.run([sys.executable, str(REPO / "examples/quickstart/agent.py")], cwd=tmp_path, env=env,
                         capture_output=True, text=True)  # cwd has no .env, so no real key can be loaded
    assert out.returncode != 0
    assert "CRUSOE_API_KEY" in (out.stdout + out.stderr)
