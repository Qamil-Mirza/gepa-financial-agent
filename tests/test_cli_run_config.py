from __future__ import annotations

import os
import subprocess
import sys


def test_pitstop_run_with_invalid_config_exits_2(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text("this is [not valid toml")
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(home)
    r = subprocess.run([sys.executable, "-m", "pitstop.cli", "run", "--mode", "replay"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "Invalid PitStop config" in r.stdout and "Traceback" not in r.stderr
