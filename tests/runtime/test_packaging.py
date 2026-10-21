import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_assets_resolve_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pitstop import config
    assert (config.PROMPTS / "supervisor_v0.txt").is_file()
    assert (config.SCENARIOS / "nvda_amd.json").is_file()
    assert (config.PROMPTS / "gepa" / "pitstop_nano_800.txt").is_file()
    assert config.SNAPSHOT.is_file()


def test_settings_paths_follow_pitstop_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
    monkeypatch.delenv("PITSTOP_RUNS_DIR", raising=False)
    monkeypatch.delenv("PITSTOP_CACHE_DIR", raising=False)
    from pitstop.config import load_settings
    s = load_settings()
    assert s.runs_dir == tmp_path / "runs"
    assert s.cache_dir == tmp_path / "cache" / "youcom"
    monkeypatch.setenv("PITSTOP_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert load_settings().runs_dir == tmp_path / "elsewhere"


def test_dotenv_loaded_from_cwd(tmp_path):
    # run in a subprocess so module-level load_dotenv runs fresh
    (tmp_path / ".env").write_text("PITSTOP_WORKER_MODEL=from-dotenv\n")
    out = subprocess.run([sys.executable, "-c", "from pitstop.config import load_settings; print(load_settings().worker_model)"],
                         cwd=tmp_path, capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PITSTOP_WORKER_MODEL"})
    assert out.stdout.strip() == "from-dotenv"


def test_wheel_contains_package_data(tmp_path):
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; run the suite via `uv run pytest`")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path)], check=True, cwd=REPO_ROOT)
    names = zipfile.ZipFile(next(tmp_path.glob("pitstop-*.whl"))).namelist()
    for required in ("pitstop/assets/prompts/supervisor_v0.txt", "pitstop/assets/prompts/worker/planner.txt",
                     "pitstop/assets/scenarios/nvda_amd.json", "pitstop/demo/snapshot.json",
                     "pitstop/demo/static/demo.html", "pitstop/ui/static/index.html",
                     "pitstop/store/schema.sql"):
        assert required in names


def test_console_script_help():
    if shutil.which("pitstop") is None:
        pytest.skip("pitstop console script not on PATH; run the suite via `uv run pytest`")
    out = subprocess.run(["pitstop", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "demo" in out.stdout
