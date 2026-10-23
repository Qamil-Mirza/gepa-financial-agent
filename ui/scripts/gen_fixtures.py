"""Regenerate the dashboard's Vitest fixtures (`ui/src/test/fixtures/*.json`) from real API responses.

    uv run python ui/scripts/gen_fixtures.py

It seeds a throwaway PitStop home with the offline demo (`examples/quickstart/dashboard_demo.py`: no keys, no
network), calls each endpoint in-process through FastAPI's TestClient, and writes one JSON file per response. The
home lives in a temp directory that is deleted afterwards; `~/.pitstop` is never read or written. The temp path is
replaced by `/home/you/.pitstop` in the output. Run ids and timestamps change on every run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples/quickstart/dashboard_demo.py"
OUT = REPO / "ui/src/test/fixtures"
PLACEHOLDER_HOME = "/home/you/.pitstop"
WORKFLOW = "earnings-research"
# the root conftest's list: no real key or alert destination may reach the demo or the settings fixture
SCRUBBED = ("CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
            "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD")


def _runtime(home: Path):
    from pitstop.budget.ledger import Ledger
    from pitstop.sdk.runtime import Runtime
    from pitstop.store import Store
    from pitstop.userconfig import load_user_config

    store = Store(home / "pitstop.db")
    config = load_user_config(home / "config.toml")
    return Runtime(store, config, Ledger(store, config), home)


def _client(home: Path, static: Path):
    from fastapi.testclient import TestClient

    from pitstop.server.app import create_app

    return TestClient(create_app(_runtime(home), static_dir=static), base_url="http://localhost:8787")


def _get(client, path: str, **params):
    r = client.get(path, params=params)
    if r.status_code != 200:
        raise SystemExit(f"{path} {params} answered {r.status_code}: {r.text}")
    return r.json()


def main() -> int:
    for var in SCRUBBED:
        os.environ.pop(var, None)
    with tempfile.TemporaryDirectory(prefix="pitstop-fixtures-") as d:
        tmp = Path(d)
        os.chdir(tmp)  # pitstop.config loads a .env from the cwd: keep the repo's out of the fixtures
        try:
            _write(tmp)
        finally:
            os.chdir(REPO)
    return 0


def _write(tmp: Path) -> None:
    home, empty = tmp / "home", tmp / "empty"
    os.environ["PITSTOP_HOME"] = str(home)
    demo = subprocess.run([sys.executable, str(DEMO), "--home", str(home)], cwd=tmp, env=dict(os.environ),
                          capture_output=True, text=True, timeout=300)
    if demo.returncode != 0:
        raise SystemExit(f"the demo failed:\n{demo.stderr}")
    empty.mkdir()

    client = _client(home, tmp / "static")
    board = _get(client, "/api/board")
    runs = _get(client, "/api/runs", workflow=WORKFLOW)
    workflow = _get(client, "/api/workflow", name=WORKFLOW)
    alarm = next(c for c in workflow["drifts"] if c["drift_type"] == "cost_or_latency_creep")
    fixtures = {
        "health": _get(client, "/api/health"),
        "board": board,
        "board_empty": _get(_client(empty, tmp / "static"), "/api/board"),
        "feed": _get(client, "/api/feed", limit=50),
        "workflow": workflow,
        "explain": _get(client, f"/api/drifts/{alarm['id']}/explain"),
        "runs": runs,
        "run": _get(client, f"/api/runs/{runs['runs'][0]['id']}"),
        "graph": _get(client, "/api/graph", workflow=WORKFLOW, window="24h"),
        "changes": _get(client, "/api/changes", workflow=WORKFLOW),
        "settings": _get(client, "/api/settings"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in fixtures.items():
        text = json.dumps(body, indent=2, ensure_ascii=False) + "\n"
        # longest first: on macOS the resolved /private/var/... path contains the unresolved /var/... one
        for path in sorted({str(p) for p in (home, home.resolve(), empty, empty.resolve())}, key=len, reverse=True):
            text = text.replace(path, PLACEHOLDER_HOME)
        if str(tmp) in text or str(tmp.resolve()) in text:
            raise SystemExit(f"{name}: the temp path survived the placeholder")
        (OUT / f"{name}.json").write_text(text, encoding="utf-8")
        print(f"wrote {(OUT / f'{name}.json').relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
