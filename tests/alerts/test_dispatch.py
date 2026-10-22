import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

import pitstop
from pitstop.alerts import channels, dispatch
from pitstop.alerts.model import build_alert
from pitstop.detect.drifts import record_drift
from pitstop.detect.l0 import Finding
from pitstop.sdk.actions import apply_findings
from pitstop.sdk.context import workflow
from pitstop.sdk.runtime import get_runtime

REPO = Path(__file__).resolve().parents[2]
P0_CONFIG = "[alerts.pagerduty]\n[alerts.opsgenie]\n[alerts.slack]\n[alerts.webhook]\n"
OK = {"events.pagerduty.com": 202, "api.opsgenie.com": 202}  # everything else answers 200


def _init(tmp_path, monkeypatch, config=P0_CONFIG, env=("PAGERDUTY_ROUTING_KEY", "SLACK_WEBHOOK_URL",
                                                         "PITSTOP_WEBHOOK_URL")):
    values = {"PAGERDUTY_ROUTING_KEY": "rk", "OPSGENIE_API_KEY": "og", "SLACK_WEBHOOK_URL": "https://hooks.slack.test/x",
              "PITSTOP_WEBHOOK_URL": "https://hook.test/in", "SMTP_PASSWORD": "pw"}
    for name in env:
        monkeypatch.setenv(name, values[name])
    (tmp_path / "config.toml").write_text(config)
    return pitstop.init(home=tmp_path)


def _capture(monkeypatch, status=None):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status or OK.get(request.url.host, 200))

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return seen


def _hosts(seen):
    return sorted(r.url.host for r in seen)


def _rows():
    return [dict(r) for r in get_runtime().store.connect().execute("SELECT * FROM alerts ORDER BY id")]


def _apply(*findings, criticality="medium"):
    with workflow("wf", criticality=criticality) as ctx:
        return apply_findings(ctx, list(findings))


def _p0():
    return Finding("out_of_scope_action", "off_task", True, {"tool": "send_email"})


def _p2(drift_type="out_of_scope_action"):
    return Finding(drift_type, "off_task", False, {"tool": "search"})


def test_p0_drift_alerts_routed_channels_once(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)  # opsgenie is configured but its key is not set: skipped
    seen = _capture(monkeypatch)
    applied = _apply(_p0(), _p0(), _p0())
    assert len({a.drift_id for a in applied}) == 1
    assert dispatch.wait_idle()
    assert _hosts(seen) == ["events.pagerduty.com", "hook.test", "hooks.slack.test"]
    rows = _rows()
    assert sorted(r["channel"] for r in rows) == ["pagerduty", "slack", "webhook"]
    assert {(r["kind"], r["status"], r["level"], r["error"]) for r in rows} == {("immediate", "sent", "P0", None)}
    _apply(_p0())  # a later occurrence of the same open drift does not alert again
    assert dispatch.wait_idle() and len(seen) == 3


def test_p1_routes_to_slack_email_webhook(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)
    _apply(Finding("tool_call_breakage", "degraded", False, {}))
    assert dispatch.wait_idle()
    assert _hosts(seen) == ["hook.test", "hooks.slack.test"]  # email is not configured


def test_escalation_realerts(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "[alerts.slack]\n", env=("SLACK_WEBHOOK_URL",))
    seen = _capture(monkeypatch)
    [p1] = _apply(Finding("unproductive_loop", "off_task", False, {}))
    assert p1.level.value == "P1"
    assert dispatch.wait_idle() and len(seen) == 1
    [p0] = _apply(Finding("unproductive_loop", "off_task", True, {}))
    assert p0.drift_id == p1.drift_id and p0.level.value == "P0"
    assert dispatch.wait_idle() and len(seen) == 2
    assert json.loads(seen[0].content)["text"].startswith("*[P1] wf: unproductive_loop*")
    assert json.loads(seen[1].content)["text"].startswith("*[P0] wf: unproductive_loop*")
    _apply(Finding("unproductive_loop", "off_task", False, {}))  # P1 again on the P0 drift: nothing new
    assert dispatch.wait_idle() and len(seen) == 2
    assert [r["level"] for r in _rows()] == ["P1", "P0"]


def test_p2_not_sent_immediately_but_in_digest(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch)
    seen = _capture(monkeypatch)
    [a] = _apply(_p2())
    assert a.level.value == "P2"
    assert dispatch.wait_idle() and seen == [] and _rows() == []
    assert dispatch.digest() == 1
    assert _hosts(seen) == ["hook.test", "hooks.slack.test"]  # digest route: email, slack, webhook
    hook = json.loads([r for r in seen if r.url.host == "hook.test"][0].content)
    assert "out_of_scope_action" in hook["what"] and hook["level"] == "P2"
    assert {(r["drift_id"], r["kind"], r["status"]) for r in _rows()} == {(a.drift_id, "digest", "sent")}
    assert dispatch.digest(force=True) == 0  # every open P2/P3 drift has been in a digest
    assert len(seen) == 2


def test_digest_respects_24h_unless_forced(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "[alerts.webhook]\n", env=("PITSTOP_WEBHOOK_URL",))
    seen = _capture(monkeypatch)
    t = 1_800_000_000.0
    _apply(_p2())
    assert dispatch.digest(now=t) == 1
    _apply(_p2("goal_decay"))
    assert dispatch.digest(now=t + 3600) == 0  # under 24 h since the last digest
    assert dispatch.digest(now=t + 3600, force=True) == 1
    _apply(_p2("topic_shift"))
    assert dispatch.digest(now=t + 3600 + 86399) == 0
    assert dispatch.digest(now=t + 3600 + 86401) == 1
    assert len(seen) == 3


def test_digest_without_channels_sends_nothing(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "")
    seen = _capture(monkeypatch)
    _apply(_p2())
    assert dispatch.digest() == 0 and seen == [] and _rows() == []


def test_failed_send_recorded(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "[alerts.webhook]\n", env=("PITSTOP_WEBHOOK_URL",))
    seen = _capture(monkeypatch, status=500)
    _apply(_p0())
    assert dispatch.wait_idle()
    [row] = _rows()
    assert (row["channel"], row["status"], row["error"]) == ("webhook", "failed", "HTTP 500")
    _apply(_p0())  # a failed alert does not block the next occurrence from retrying
    assert dispatch.wait_idle() and len(seen) == 2


def test_hanging_channel_does_not_block_agent(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "[alerts.webhook]\n", env=("PITSTOP_WEBHOOK_URL",))
    release = threading.Event()
    seen = []

    def handler(request):
        seen.append(request)
        release.wait(10)
        return httpx.Response(200)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        start = time.monotonic()
        _apply(_p0())
        assert time.monotonic() - start < 0.5
        assert not dispatch.wait_idle(0.2)  # still sending
    finally:
        release.set()
    assert dispatch.wait_idle()
    assert len(seen) == 1 and _rows()[0]["status"] == "sent"


def test_secret_never_stored_or_logged(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _init(tmp_path, monkeypatch, "[alerts.slack]\n")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/SECRET123")
    fail = [True]

    def handler(request):
        if fail[0]:
            raise httpx.ConnectError(f"cannot connect to {request.url}")
        return httpx.Response(200)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    _apply(Finding("tool_call_breakage", "degraded", False, {}))
    assert dispatch.wait_idle()
    assert _rows()[0]["error"] == "ConnectError"
    fail[0] = False
    _apply(Finding("tool_call_breakage", "degraded", False, {}))  # retried and sent: httpx logs its request
    assert dispatch.wait_idle()
    assert dispatch.send_test("slack") == {"slack": "sent"}
    assert [r["status"] for r in _rows()] == ["failed", "sent", "sent"]
    conn = get_runtime().store.connect()
    for table in ("alerts", "events", "drifts", "meta", "runs"):
        for row in conn.execute(f"SELECT * FROM {table}"):
            assert "SECRET123" not in json.dumps(tuple(row), default=str), table
    assert "SECRET123" not in caplog.text


def test_alert_body_names_model_change(tmp_path, monkeypatch):
    rt = _init(tmp_path, monkeypatch, "")
    now = time.time()
    with rt.store.transaction() as c:
        for rid, model, ts in (("r1", "model-a", now - 100), ("r2", "model-b", now - 10)):
            c.execute("INSERT INTO runs(id, workflow, criticality, started_at, model, base_url) VALUES (?,?,?,?,?,?)",
                      (rid, "wf", "medium", ts, model, "https://api.example/v1"))
    did = record_drift(rt.store, run_id="r2", workflow="wf", finding=Finding("tool_call_breakage", "degraded", False,
                                                                              {"failures": 3}),
                       level="P1", action="record")
    alert = build_alert(rt.store, did)
    assert "model changed from model-a to model-b" in alert.cause
    assert "What likely changed: " + alert.cause in alert.body
    assert (alert.drift_id, alert.workflow, alert.level, alert.run_id) == (did, "wf", "P1", "r2")
    assert "tool_call_breakage" in alert.what and alert.action
    with rt.store.transaction() as c:
        c.execute("UPDATE runs SET model='model-b' WHERE id='r1'")
    assert build_alert(rt.store, did).cause == ("No model or provider change detected for this run "
                                                "(model model-b at base_url https://api.example/v1).")


def test_notify_without_channels_starts_no_work(tmp_path, monkeypatch):
    _init(tmp_path, monkeypatch, "")
    seen = _capture(monkeypatch)
    _apply(_p0())
    assert dispatch.wait_idle() and seen == [] and _rows() == []


class _Hook(BaseHTTPRequestHandler):
    posts: list = []

    def do_POST(self):
        _Hook.posts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_p0_alert_delivered_when_process_exits(tmp_path):
    _Hook.posts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.toml").write_text("[alerts.webhook]\n")
        script = tmp_path / "agent.py"
        script.write_text(textwrap.dedent("""
            import pitstop
            from pitstop.sdk.context import check_halt

            @pitstop.tool(side_effects=True)
            def send_email(to):
                return "sent"

            with pitstop.workflow("payments", criticality="high", tools=["search"]):
                print(send_email("x@example.com"))  # out of scope: blocked, P0, pending halt
                check_halt()  # the next call boundary raises Halted; nothing catches it
        """))
        scrub = {"CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
                 "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD"}
        env = {k: v for k, v in os.environ.items() if k not in scrub}
        env.update(PITSTOP_HOME=str(home), PYTHONPATH=str(REPO / "src"),
                   PITSTOP_WEBHOOK_URL=f"http://127.0.0.1:{server.server_address[1]}/hook")
        out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                             timeout=60)
        assert out.returncode != 0 and "Halted" in out.stderr, out.stderr
        assert "blocked_by" in out.stdout
    finally:
        server.shutdown()
        server.server_close()
    [post] = _Hook.posts
    assert post["level"] == "P0" and post["workflow"] == "payments"
    assert post["drift_type"] == "out_of_scope_action" and post["title"].startswith("[P0] payments")
