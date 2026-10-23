"""`examples/quickstart/dashboard_demo.py`: fills an empty PitStop home through the real SDK (no keys, no network), so
the dashboard has something to show. Run as a subprocess, as a user would, then read back through the API."""
import os
import subprocess
import sys
from pathlib import Path

from tests.server.conftest import BASE_URL

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples/quickstart/dashboard_demo.py"
# the root conftest's list: no real key or alert destination may reach the subprocess
SCRUBBED = ("CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
            "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD")


def _demo(tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env["PITSTOP_HOME"] = str(tmp_path)
    return subprocess.run([sys.executable, str(DEMO), *args], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=180)


def test_dashboard_demo_seeds_expected_board(tmp_path):
    from fastapi.testclient import TestClient

    from pitstop.sdk.runtime import init
    from pitstop.server.app import create_app

    home = tmp_path / "home"
    out = _demo(tmp_path, "--home", str(home), "--runs", "40")
    assert out.returncode == 0, out.stderr
    n_runs = 40 + 35 + 13 + 14 + 10  # earnings-research before and after the model change, then the other three
    assert f"Seeded {n_runs} runs in {home}. Run: PITSTOP_HOME={home} pitstop ui" in out.stdout

    client = TestClient(create_app(init(home=home), static_dir=tmp_path / "static"), base_url=BASE_URL)
    tiles = {t["name"]: t for t in client.get("/api/board").json()["workflows"]}
    assert {name: t["flag"] for name, t in tiles.items()} == {
        "earnings-research": "red", "support-triage": "yellow", "nightly-digest": "green", "kb-sync": "green"}
    earnings = tiles["earnings-research"]
    assert earnings["last_run"]["status"] == "halted" and earnings["open"]["P0"] == 1
    assert earnings["budget_usd"] == 2.0 and 0 < earnings["spend_today_usd"] < 2.0
    assert tiles["support-triage"]["open"]["P0"] == 0 and tiles["support-triage"]["open"]["P1"] == 1

    markers = client.get("/api/changes", params={"workflow": "earnings-research"}).json()["markers"]
    model = [m for m in markers if m["kind"] == "model"]
    assert [(m["before"], m["after"]) for m in model] == [("m-a", "m-b")]
    alarm = next(m for m in markers if m["kind"] == "l2_alarm" and m["metric"] == "tokens_per_call")
    assert alarm["at"] == model[0]["at"]  # the change point is the first run on m-b
    for name in ("support-triage", "nightly-digest", "kb-sync"):
        assert not [m for m in client.get("/api/changes", params={"workflow": name}).json()["markers"]
                    if m["kind"] == "l2_alarm"]


def test_dashboard_demo_refuses_existing_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "pitstop.db").write_bytes(b"the owner's store")
    out = _demo(tmp_path, "--home", str(home))
    assert out.returncode == 2
    assert "pick an empty directory" in out.stderr
    assert (home / "pitstop.db").read_bytes() == b"the owner's store"
    assert sorted(p.name for p in home.iterdir()) == ["pitstop.db"]


def test_dashboard_demo_refuses_too_few_runs(tmp_path):
    out = _demo(tmp_path, "--home", str(tmp_path / "home"), "--runs", "34")
    assert out.returncode == 2
    assert "--runs" in out.stderr
    assert not (tmp_path / "home").exists()
