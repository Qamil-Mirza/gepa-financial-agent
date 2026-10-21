import os
import re
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
    hits = subprocess.run(["git", "grep", "-nE", r"gepa_financial_agent|(^|[^.~/])pitstop/pitstop|synthetic-data/|policies/versions",
                           "--", "*.py", "*.toml", "*.yml", "*.yaml", "*.ini", "*.cfg", ":!tests/test_repo_hygiene.py"],
                          cwd=REPO, capture_output=True, text=True).stdout
    assert hits == ""


def test_suite_runs_without_keys(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("CRUSOE_API_KEY", "YDC_API_KEY")}
    # cwd is an empty temp dir so the repo-root .env is never loaded; repo config is passed explicitly
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-c", str(REPO / "pyproject.toml"), "--rootdir", str(REPO),
                          str(REPO / "tests/runtime"), str(REPO / "tests/correct"), "-p", "no:cacheprovider"],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:]


def test_no_stale_references_in_docs():
    hits = subprocess.run(["git", "grep", "-nE", r"gepa_financial_agent|(^|[^.~/])pitstop/pitstop|synthetic-data/|policies/versions",
                           "--", "*.md", ":!docs/history/**", ":!docs/superpowers/**", ":!CHANGELOG.md"],
                          cwd=REPO, capture_output=True, text=True).stdout
    assert hits == ""


@pytest.mark.parametrize("doc", ["README.md", "CONTRIBUTING.md"])
def test_relative_links_resolve(doc):
    text = (REPO / doc).read_text()
    targets = re.findall(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", text)
    missing = []
    for target in targets:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        if not (REPO / doc).parent.joinpath(path).exists():
            missing.append(target)
    assert missing == []


def test_check_crusoe_script_meters_its_client():
    """The live smoke script spends real money, so its client must go through the hard budget."""
    text = (REPO / "benchmarks/synth/scripts/check_crusoe.py").read_text()
    assert re.search(r"pitstop\.wrap\(\s*OpenAI\(", text)
    assert not re.search(r"(?<!wrap\()\bOpenAI\(", text)
