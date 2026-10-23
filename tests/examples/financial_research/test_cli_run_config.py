from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_finresearch_run_with_invalid_config_exits_2(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text("this is [not valid toml")
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(home)
    r = subprocess.run([sys.executable, "-m", "finresearch.cli", "run", "--mode", "replay"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "Invalid PitStop config" in r.stdout and "Traceback" not in r.stderr


def _cli(tmp_path, *argv):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    env["PITSTOP_HOME"] = str(tmp_path)
    return subprocess.run([sys.executable, "-m", "finresearch.cli", "run", *argv], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("flags,named", [
    (["--no-report"], "--no-report"),
    (["--mode", "replay"], "--mode replay"),
    (["--cache-dir", "x"], "--cache-dir"),
    (["--max-research-calls", "3"], "--max-research-calls"),
])
def test_offline_rejects_unsupported_flags(tmp_path, flags, named):
    r = _cli(tmp_path, "--offline", *flags)
    assert r.returncode == 2
    assert f"--offline does not support {named}" in r.stderr and "Traceback" not in r.stderr


def test_prompt_alias_points_to_legacy_supervisor(tmp_path):
    r = _cli(tmp_path, "--prompt", "p.txt")
    assert r.returncode == 2
    assert "--prompt was replaced by --legacy-supervisor PROMPT" in r.stderr and "Traceback" not in r.stderr
