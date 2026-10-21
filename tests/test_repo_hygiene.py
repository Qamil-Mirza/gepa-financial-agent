import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GONE = ["agent", "main.py", "gepa", "policies", "gepa_financial_agent", "synthetic-data", "financial-agent",
        "pitstop/pitstop", "requirements.txt", "data"]


@pytest.mark.parametrize("rel", GONE)
def test_superseded_paths_are_gone(rel):
    assert not (REPO / rel).exists()


def test_no_stale_references_in_code():
    hits = subprocess.run(["git", "grep", "-nE", r"gepa_financial_agent|pitstop/pitstop|synthetic-data/|policies/versions",
                           "--", "*.py", "*.toml", "*.yml", "*.yaml", "*.ini", "*.cfg", ":!tests/test_repo_hygiene.py"],
                          cwd=REPO, capture_output=True, text=True).stdout
    assert hits == ""


def test_suite_runs_without_keys():
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/runtime", "tests/correct", "-p", "no:cacheprovider"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:]
