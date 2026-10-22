import sys

import httpx
import pytest

from pitstop.alerts import channels
from pitstop.detect.l0 import Finding
from pitstop.sdk.actions import apply_findings
from pitstop.sdk.context import workflow
from pitstop.sdk.runtime import get_runtime


def _main(monkeypatch, *argv):
    from pitstop import cli

    monkeypatch.setattr(sys, "argv", ["pitstop", *argv])
    cli.main()


def _setup(tmp_path, monkeypatch, config):
    monkeypatch.setenv("PITSTOP_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(config)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(500 if request.url.host == "hook.test" else 200)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return seen


def _rows():
    return [dict(r) for r in get_runtime().store.connect().execute("SELECT * FROM alerts ORDER BY id")]


def test_alerts_test_cli_reports_per_channel(tmp_path, capsys, monkeypatch):
    seen = _setup(tmp_path, monkeypatch, "[alerts.slack]\n[alerts.webhook]\n[alerts.opsgenie]\n")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/SECRET123")
    monkeypatch.setenv("PITSTOP_WEBHOOK_URL", "https://hook.test/in")
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, "alerts", "test")
    assert e.value.code == 1  # a channel failed
    out = capsys.readouterr().out
    lines = {line.split()[0]: " ".join(line.split()[1:]) for line in out.splitlines() if line.strip()}
    assert lines["slack"] == "sent"
    assert lines["webhook"] == "failed: HTTP 500"
    assert lines["opsgenie"] == "not configured" and lines["pagerduty"] == "not configured"
    assert lines["email"] == "not configured"
    assert "SECRET123" not in out
    assert len(seen) == 2
    assert {(r["channel"], r["kind"], r["status"]) for r in _rows()} == {("slack", "test", "sent"),
                                                                        ("webhook", "test", "failed")}
    _main(monkeypatch, "alerts", "test", "--channel", "slack")  # exit 0
    assert capsys.readouterr().out.split() == ["slack", "sent"]


def test_alerts_test_cli_nothing_configured(tmp_path, capsys, monkeypatch):
    _setup(tmp_path, monkeypatch, "")
    _main(monkeypatch, "alerts", "test")
    assert "No alert channel is configured" in capsys.readouterr().out


def test_alerts_digest_cli(tmp_path, capsys, monkeypatch):
    seen = _setup(tmp_path, monkeypatch, "[alerts.slack]\n")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    with workflow("wf") as ctx:
        apply_findings(ctx, [Finding("out_of_scope_action", "off_task", False, {})])  # P2
    _main(monkeypatch, "alerts", "digest")
    assert "1 drift(s)" in capsys.readouterr().out
    with workflow("wf") as ctx:
        apply_findings(ctx, [Finding("goal_decay", "off_task", False, {})])
    _main(monkeypatch, "alerts", "digest")
    assert "last digest" in capsys.readouterr().out  # under 24 h: nothing sent
    _main(monkeypatch, "alerts", "digest", "--force")
    assert "1 drift(s)" in capsys.readouterr().out
    assert len(seen) == 2


def test_alerts_cli_invalid_config_exits_2(tmp_path, capsys, monkeypatch):
    _setup(tmp_path, monkeypatch, '[alerts.slack]\nwebhook_url = "https://hooks.slack.test/SECRET123"\n')
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, "alerts", "test")
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "webhook_url_env" in out and "SECRET123" not in out
