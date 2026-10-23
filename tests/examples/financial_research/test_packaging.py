import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_NO_KEYS = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}


def test_finresearch_assets_resolve_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from finresearch import config
    assert (config.PROMPTS / "supervisor_v0.txt").is_file()
    assert (config.SCENARIOS / "nvda_amd.json").is_file()
    assert (config.PROMPTS / "gepa" / "pitstop_nano_800.txt").is_file()
    assert (config.PROMPTS / "worker" / "planner.txt").is_file()
    assert config.SNAPSHOT.is_file()


def test_settings_paths_follow_pitstop_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
    monkeypatch.delenv("PITSTOP_RUNS_DIR", raising=False)
    monkeypatch.delenv("PITSTOP_CACHE_DIR", raising=False)
    from finresearch.config import load_settings
    s = load_settings()
    assert s.runs_dir == tmp_path / "runs"
    assert s.cache_dir == tmp_path / "cache" / "youcom"
    monkeypatch.setenv("PITSTOP_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert load_settings().runs_dir == tmp_path / "elsewhere"


def test_dotenv_loaded_from_cwd(tmp_path):
    # run in a subprocess so module-level load_dotenv runs fresh
    (tmp_path / ".env").write_text("PITSTOP_WORKER_MODEL=from-dotenv\n")
    env = {k: v for k, v in _NO_KEYS.items() if k != "PITSTOP_WORKER_MODEL"}
    out = subprocess.run([sys.executable, "-c", "from finresearch.config import load_settings; "
                          "print(load_settings().worker_model)"], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "from-dotenv"


def test_finresearch_wheel_contains_package_data(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; run the suite via `uv run pytest`")
    subprocess.run(["uv", "build", "--wheel", "--package", "pitstop-financial-research", "--out-dir", str(tmp_path)],
                   check=True, cwd=REPO_ROOT, env=_NO_KEYS)
    names = zipfile.ZipFile(next(tmp_path.glob("pitstop_financial_research-*.whl"))).namelist()
    for required in ("finresearch/assets/prompts/supervisor_v0.txt", "finresearch/assets/prompts/worker/planner.txt",
                     "finresearch/assets/prompts/gepa/pitstop_nano_800.txt",
                     "finresearch/assets/scenarios/nvda_amd.json", "finresearch/demo/snapshot.json",
                     "finresearch/demo/static/demo.html", "finresearch/ui/static/index.html"):
        assert required in names
    assert not any(n.startswith("pitstop/") for n in names)


def test_console_script_help():
    if shutil.which("finresearch") is None:
        pytest.skip("finresearch console script not on PATH; run the suite via `uv run pytest`")
    out = subprocess.run(["finresearch", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    commands = set(re.search(r"\{([a-z,]+)\}", out.stdout).group(1).split(","))
    assert commands == {"check", "run", "compare", "timeline", "serve", "demo"}
