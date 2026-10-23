"""The `pitstop` wheel and CLI carry only the SDK; the research swarm lives in examples/financial-research."""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_NO_KEYS = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}


def test_wheel_ships_only_the_sdk(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; run the suite via `uv run pytest`")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], check=True, cwd=REPO_ROOT, env=_NO_KEYS)
    names = zipfile.ZipFile(next(tmp_path.glob("pitstop-*.whl"))).namelist()
    for required in ("pitstop/assets/prompts/l1_gate_v1.txt", "pitstop/assets/prompts/l1_gate_v2.txt",
                     "pitstop/store/schema.sql"):
        assert required in names
    for gone in ("pitstop/demo/", "pitstop/ui/", "pitstop/orchestrator.py", "pitstop/assets/scenarios/",
                 "pitstop/supervisor/", "pitstop/assets/prompts/worker/", "pitstop/assets/prompts/gepa/",
                 "finresearch/"):
        assert not any(n.startswith(gone) for n in names), gone


def test_core_does_not_import_example(tmp_path):
    code = ("import sys, pitstop, pitstop.cli; "
            "print(sorted(m for m in sys.modules if m == 'finresearch' or m.startswith('finresearch.')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=_NO_KEYS, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_pitstop_console_script_lists_only_sdk_commands():
    if shutil.which("pitstop") is None:
        pytest.skip("pitstop console script not on PATH; run the suite via `uv run pytest`")
    out = subprocess.run(["pitstop", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    commands = set(re.search(r"\{([a-z,]+)\}", out.stdout).group(1).split(","))
    assert commands == {"status", "drifts", "budget", "explain", "fallback", "alerts"}
