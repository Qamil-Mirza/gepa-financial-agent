"""The dashboard's JSON endpoints over HTTP (in-process TestClient): status codes, caps, odd workflow names,
malformed rows, a locked store and secrets."""
import json
import sqlite3
import time

import pytest

from pitstop.budget.ledger import Ledger
from pitstop.sdk.runtime import Runtime, init
from pitstop.store import Store
from pitstop.userconfig import load_user_config

from tests.server.conftest import BASE_URL
from tests.server.factory import add_drift, add_event, add_ledger, add_llm_call, add_run

T0 = 1_790_000_000.0
WRITE = {"X-PitStop": "1"}
ODD_NAMES = ("a/b", "x?y#z", "spaced name", "résumé 🏁")


def _client(rt, tmp_path):
    from fastapi.testclient import TestClient

    from pitstop.server.app import create_app

    return TestClient(create_app(rt, static_dir=tmp_path / "static"), base_url=BASE_URL)


def _drift_status(store, drift_id):
    return store.connect().execute("SELECT status FROM drifts WHERE id=?", (drift_id,)).fetchone()[0]


def test_dismiss_endpoint_states(rt, client):
    drift = add_drift(rt.store, "wf", ts=T0)
    r = client.post(f"/api/drifts/{drift}/dismiss", headers=WRITE)
    assert (r.status_code, r.json()) == (200, {"status": "dismissed"})
    assert _drift_status(rt.store, drift) == "dismissed"
    r = client.post(f"/api/drifts/{drift}/dismiss", headers=WRITE)
    assert (r.status_code, r.json()) == (409, {"status": "not_open"})
    r = client.post("/api/drifts/9999/dismiss", headers=WRITE)
    assert (r.status_code, r.json()) == (404, {"status": "not_found"})
    assert client.post(f"/api/drifts/{drift}/dismiss").status_code == 403  # the write guard still applies


class _QuickBusyStore(Store):
    """A store whose connections give up on a held write lock after 200 ms instead of 5 s."""
    BUSY_S = 0.2

    def connect(self):
        conn = super().connect()
        conn.execute(f"PRAGMA busy_timeout={int(self.BUSY_S * 1000)}")
        return conn


def test_dismiss_on_locked_store_returns_503(tmp_path):
    home = tmp_path / "busy-home"
    store = _QuickBusyStore(home / "pitstop.db")
    config = load_user_config(home / "config.toml")
    rt = Runtime(store, config, Ledger(store, config), home)
    run = add_run(store, "wf", started_at=T0)
    drift = add_drift(store, "wf", ts=T0, run_id=run)
    client = _client(rt, tmp_path)

    holder = sqlite3.connect(store.path, isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        r = client.post(f"/api/drifts/{drift}/dismiss", headers=WRITE)
        elapsed = time.monotonic() - started
        assert (r.status_code, r.json()) == (503, {"error": "store busy, retry"})
        assert elapsed < _QuickBusyStore.BUSY_S + 1
        # WAL readers are not blocked by the held write lock
        assert client.get("/api/board").status_code == 200
        assert client.get("/api/workflow", params={"name": "wf"}).status_code == 200
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert _drift_status(store, drift) == "open"  # nothing changed
    assert client.post(f"/api/drifts/{drift}/dismiss", headers=WRITE).json() == {"status": "dismissed"}
    store.close_local()


def test_explain_endpoint(rt, client):
    change_at = T0 + 1000
    for i in range(6):
        add_run(rt.store, "wf", started_at=T0 + i * 10, model="m1")
        add_run(rt.store, "wf", started_at=change_at + i * 10, model="m2")
    drift = add_drift(rt.store, "wf", ts=change_at + 100, level="P1", drift_type="cost_or_latency_creep",
                      kind="degraded", evidence=[{"run_id": None, "ts": change_at + 100,
                                                  "detail": {"change_at": change_at, "metric": "tokens_per_call"}}])
    body = client.get(f"/api/drifts/{drift}/explain").json()
    assert (body["at"], body["metric"], body["n_before"], body["n_after"]) == (change_at, "tokens_per_call", 6, 6)
    assert {"tag": "model", "before": "m1", "after": "m2", "share_before": 0.0, "share_after": 1.0} in body["changes"]
    assert "model changed from m1 to m2" in body["summary"]
    assert set(body) == {"summary", "changes", "metric", "before_mean", "after_mean", "n_before", "n_after", "at"}

    r = client.get("/api/drifts/9999/explain")
    assert r.status_code == 404 and r.headers["content-type"] == "application/json"
    orphan = add_drift(rt.store, None, ts=T0)
    assert client.get(f"/api/drifts/{orphan}/explain").status_code == 422


def test_runs_pagination_cursor(rt, client):
    ids = [add_run(rt.store, "wf", started_at=T0 + i) for i in range(75)]
    add_run(rt.store, "other", started_at=T0 + 1000)
    first = client.get("/api/runs", params={"workflow": "wf"}).json()
    assert [r["id"] for r in first["runs"]] == ids[::-1][:50]
    assert first["next_before"] == T0 + 25
    second = client.get("/api/runs", params={"workflow": "wf", "before": first["next_before"]}).json()
    assert [r["id"] for r in second["runs"]] == ids[::-1][50:]
    assert second["next_before"] is None
    assert len(client.get("/api/runs", params={"workflow": "wf", "limit": 10}).json()["runs"]) == 10
    assert len(client.get("/api/runs", params={"workflow": "wf", "limit": 5000}).json()["runs"]) == 75  # max 200
    assert len(client.get("/api/runs").json()["runs"]) == 50  # every workflow when none is named


def test_run_detail_endpoint(rt, client):
    run = add_run(rt.store, "wf", started_at=T0)
    add_llm_call(rt.store, run, ts=T0 + 1)
    body = client.get(f"/api/runs/{run}").json()
    assert body["run"]["id"] == run and body["workflow"] == "wf" and len(body["events"]) == 1
    assert "payload" not in json.dumps(body["events"])
    assert client.get("/api/runs/nope").status_code == 404


def test_settings_never_returns_env_values(tmp_path, monkeypatch):
    home = tmp_path / "cfg-home"
    home.mkdir()
    (home / "config.toml").write_text(
        '[budgets.daily]\nlimit_usd = 10\nperiod = "day"\n'
        '[judge]\nbase_url = "https://judge-user:pw-in-url@judge.example/v1?token=tok-in-url"\n'
        '[alerts.slack]\nwebhook_url_env = "SLACK_WEBHOOK_URL"\n'
        '[alerts.routes]\nP0 = ["slack"]\n')
    monkeypatch.setenv("CRUSOE_API_KEY", "sk-test-123")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/services/T0/B0/slack-secret-456")
    rt = init(home=home)
    add_run(rt.store, "wf", started_at=T0, criticality="high", sensitivity="low", fix_mode="auto", budget_usd=2.5)
    add_ledger(rt.store, "wf", 1.5, ts=time.time())
    r = _client(rt, tmp_path).get("/api/settings")
    assert r.status_code == 200
    for secret in ("sk-test-123", "slack-secret-456", "hooks.slack.example", "pw-in-url", "tok-in-url",
                   "judge-user"):
        assert secret not in r.text, secret
    body = r.json()
    assert body["home"] == str(home)
    assert body["judge"] == {"model": "openai/gpt-oss-120b", "base_url": "https://judge.example/v1",
                             "api_key_env": "CRUSOE_API_KEY", "key_set": True}
    channels = {c["name"]: c for c in body["alerts"]["channels"]}
    assert list(channels) == ["pagerduty", "opsgenie", "slack", "email", "webhook"]
    assert channels["slack"] == {"name": "slack", "configured": True, "env_var": "SLACK_WEBHOOK_URL",
                                 "env_set": True}
    assert channels["pagerduty"] == {"name": "pagerduty", "configured": False, "env_var": "PAGERDUTY_ROUTING_KEY",
                                     "env_set": False}
    assert channels["email"]["env_var"] == "SMTP_PASSWORD"
    assert body["alerts"]["routes"]["P0"] == ["slack"]
    assert body["alerts"]["routes"]["P1"] == ["slack", "email", "webhook"]
    assert body["workflows"] == [{"name": "wf", "criticality": "high", "sensitivity": "low", "fix_mode": "auto",
                                  "budget_usd": 2.5}]
    (budget,) = body["budgets"]
    assert (budget["name"], budget["period"], budget["limit_usd"], budget["spent_usd"]) == ("daily", "day", 10.0, 1.5)
    assert body["worker"]["enabled"] is False

    monkeypatch.delenv("CRUSOE_API_KEY")
    assert _client(rt, tmp_path).get("/api/settings").json()["judge"]["key_set"] is False


def _get_everything(client, store) -> list:
    """Every read endpoint, for each workflow, run and drift in the store."""
    conn = store.connect()
    names = [r[0] for r in conn.execute("SELECT DISTINCT workflow FROM runs WHERE workflow IS NOT NULL")]
    run_ids = [r[0] for r in conn.execute("SELECT id FROM runs")]
    drift_ids = [r[0] for r in conn.execute("SELECT id FROM drifts")]
    responses = [client.get(p) for p in ("/api/board", "/api/feed", "/api/runs", "/api/settings", "/api/health")]
    for name in names:
        responses.append(client.get("/api/workflow", params={"name": name}))
        responses.append(client.get("/api/runs", params={"workflow": name}))
    responses += [client.get(f"/api/runs/{run_id}") for run_id in run_ids]
    responses += [client.get(f"/api/drifts/{d}/explain") for d in drift_ids]
    return responses


def test_endpoints_survive_malformed_rows(rt, client):
    good = add_run(rt.store, "wf", started_at=T0)
    add_run(rt.store, None, started_at=None, ended_at=None, status=None)  # no workflow, no times
    odd = add_run(rt.store, "wf", started_at=T0 + 10, ended_at=None, status="exploded", model=None, base_url=None,
                  adapter=None, criticality=None)
    texty = add_run(rt.store, "wf", started_at="not a time", ended_at="later", status="ok", budget_usd="lots")
    add_run(rt.store, "nulls", started_at=None, status="halted")
    for kind in ("llm_call", "tool_call", "tool_result", "halt", "fallback", "block", "gate", "agent"):
        add_event(rt.store, good, kind, ts=T0 + 1, name=kind, payload="{not json")
        add_event(rt.store, good, kind, ts=None, name=None, payload='["a", "list"]', agent=None)
        add_event(rt.store, odd, kind, ts="soon", name=kind, payload='{"reason": 5, "args": null, "latency_s": "x"}',
                  cost_usd="free")
    add_event(rt.store, None, "llm_call", ts=T0, payload={"meter": "agent"})  # outside any run
    add_drift(rt.store, "wf", ts=T0, level="P0", run_id=good, evidence='{"not": "a list"}')
    add_drift(rt.store, "wf", ts=T0, level="P1", drift_type="other", run_id="missing-run", evidence="not json")
    add_drift(rt.store, "wf", ts=T0, level="P9", drift_type=None, kind=None, action=None, evidence=None)
    add_drift(rt.store, "wf", ts=None, level=None, drift_type="no-time", last_seen=None, evidence="[1, 2]")
    add_drift(rt.store, None, ts=T0, level="P2", drift_type="loose")
    add_drift(rt.store, "wf", ts=T0, level="P1", drift_type="texty", run_id=texty)  # its alert text cannot be built
    add_ledger(rt.store, "wf", 0.5, ts=time.time())

    responses = _get_everything(client, rt.store)
    explains = [r for r in responses if r.request.url.path.endswith("/explain")]
    for r in responses:
        if r in explains:
            assert r.status_code in (200, 422), (r.request.url, r.status_code, r.text)
        else:
            assert r.status_code == 200, (r.request.url, r.status_code, r.text)
    assert sum(r.status_code == 200 for r in explains) >= 3  # only the drifts without a workflow or time are 422
    board = responses[0].json()
    assert {t["name"] for t in board["workflows"]} == {"wf", "nulls"}
    assert {t["name"]: t["flag"] for t in board["workflows"]}["nulls"] == "red"
    cards = {c["drift_type"]: c for c in client.get("/api/workflow", params={"name": "wf"}).json()["drifts"]}
    assert cards["texty"]["title"] == "[P1] wf: texty"
    assert cards["texty"]["cause"].startswith("Unknown")


def test_workflow_names_round_trip(rt, client):
    runs, drifts = {}, {}
    for i, name in enumerate(ODD_NAMES):
        runs[name] = add_run(rt.store, name, started_at=T0 + i, budget_usd=1.0)
        drifts[name] = add_drift(rt.store, name, ts=T0 + i, level="P1", run_id=runs[name])
        add_event(rt.store, runs[name], "halt", ts=T0 + i, payload={"reason": "stop"})
    add_run(rt.store, "a", started_at=T0)  # a prefix of "a/b": never confused with it

    assert {t["name"] for t in client.get("/api/board").json()["workflows"]} == {*ODD_NAMES, "a"}
    feed = client.get("/api/feed").json()
    settings = client.get("/api/settings").json()
    assert {w["name"] for w in settings["workflows"]} == {*ODD_NAMES, "a"}
    for name in ODD_NAMES:
        detail = client.get("/api/workflow", params={"name": name}).json()
        assert detail["tile"]["name"] == name
        assert [d["id"] for d in detail["drifts"]] == [drifts[name]]
        page = client.get("/api/runs", params={"workflow": name}).json()
        assert [r["id"] for r in page["runs"]] == [runs[name]]
        assert client.get(f"/api/runs/{runs[name]}").json()["workflow"] == name
        assert {i["kind"] for i in feed if i["workflow"] == name} == {"drift", "halt"}
        assert client.get(f"/api/drifts/{drifts[name]}/explain").status_code == 200
    assert client.get("/api/workflow", params={"name": "a/"}).status_code == 404
    assert client.get("/api/workflow", params={"name": "résumé"}).status_code == 404
    assert client.get("/api/runs", params={"workflow": "a/"}).json()["runs"] == []


def test_unknown_workflow_is_404(client):
    r = client.get("/api/workflow", params={"name": "nope"})
    assert r.status_code == 404 and r.headers["content-type"] == "application/json"
    assert client.get("/api/workflow").status_code == 422  # name is required


def test_feed_limit_is_capped(rt, client):
    for i in range(130):
        add_drift(rt.store, f"wf{i}", ts=T0 + i, level="P2")
    assert len(client.get("/api/feed").json()) == 50
    assert len(client.get("/api/feed", params={"limit": 500}).json()) == 100
    assert len(client.get("/api/feed", params={"limit": 0}).json()) == 1


def test_requests_leave_no_store_connection_open(rt, client):
    """Connection policy: each endpoint closes the store connection of the thread that served it."""
    run = add_run(rt.store, "wf", started_at=T0)
    drift = add_drift(rt.store, "wf", ts=T0, run_id=run)
    before = rt.store.open_connections()
    _get_everything(client, rt.store)
    client.post(f"/api/drifts/{drift}/dismiss", headers=WRITE)
    client.post("/api/drifts/9999/dismiss", headers=WRITE)
    assert rt.store.open_connections() == before


def test_endpoints_are_in_the_openapi_schema():
    from pitstop.server.app import export_openapi

    schema = export_openapi()
    paths = {"/api/board", "/api/feed", "/api/workflow", "/api/drifts/{drift_id}/dismiss",
             "/api/drifts/{drift_id}/explain", "/api/runs", "/api/runs/{run_id}", "/api/settings"}
    assert paths <= set(schema["paths"])
    models = {"Board", "Tile", "BudgetRow", "FeedItem", "WorkflowDetail", "DriftCard", "DismissResult", "Explain",
              "TagChange", "RunPage", "RunRow", "RunDetail", "EventRow", "Settings"}
    assert models <= set(schema["components"]["schemas"])


@pytest.mark.parametrize("path", ["/api/runs/{run}", "/api/workflow?name=wf"])
def test_responses_carry_no_raw_payload(rt, client, path):
    run = add_run(rt.store, "wf", started_at=T0)
    add_event(rt.store, run, "tool_result", ts=T0 + 1, name="t", payload={"result": "RAW-" + "y" * 5000})
    assert "y" * 300 not in client.get(path.format(run=run)).text  # at most a 300-character summary of it


SECRET_URL = "https://user-x:pw-secret@host.example/v1?key=key-secret#frag-secret"
URL_SECRETS = ("user-x", "pw-secret", "key-secret", "frag-secret")


def test_run_base_url_secrets_never_reach_the_client(rt, client):
    """A credentialed run base_url is redacted everywhere it surfaces: alert causes, explain, runs, feed."""
    now = time.time()
    for i in range(4):
        add_run(rt.store, "wf", started_at=now - 600 + i, base_url="https://a/v1")
    changed = [add_run(rt.store, "wf", started_at=now - 300 + i, base_url=SECRET_URL) for i in range(4)]
    drift = add_drift(rt.store, "wf", ts=now - 200, level="P0", run_id=changed[-1],
                      evidence=[{"run_id": changed[-1], "ts": now - 200,
                                 "detail": {"change_at": now - 300, "metric": "llm_calls"}}])
    lone = add_run(rt.store, "lone", started_at=now - 100, base_url=SECRET_URL)  # cause names its base_url
    add_drift(rt.store, "lone", ts=now - 100, level="P1", run_id=lone)

    responses = [client.get("/api/board"), client.get("/api/feed"), client.get(f"/api/drifts/{drift}/explain"),
                 client.get("/api/runs"), client.get(f"/api/runs/{changed[-1]}"), client.get("/api/settings")]
    responses += [client.get("/api/workflow", params={"name": n}) for n in ("wf", "lone")]
    assert all(r.status_code == 200 for r in responses)
    for r in responses:
        for secret in URL_SECRETS:
            assert secret not in r.text, (r.request.url, secret)
    explain = responses[2].json()
    assert {"tag": "base_url", "before": "https://a/v1", "after": "https://host.example/v1", "share_before": 0.0,
            "share_after": 1.0} in explain["changes"]
    assert "https://host.example/v1" in explain["summary"]
    cause = responses[-2].json()["drifts"][0]["cause"]
    assert "from https://a/v1 to https://host.example/v1 (" in cause  # the sentence around the URL is intact
    assert "https://host.example/v1)." in responses[-1].json()["drifts"][0]["cause"]
