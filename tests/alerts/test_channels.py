import json

import httpx
import pytest

from pitstop.alerts import channels
from pitstop.alerts.channels import ChannelError
from pitstop.alerts.model import Alert
from pitstop.userconfig import EmailSpec, OpsgenieSpec, PagerDutySpec, SlackSpec, WebhookSpec

ALERT = Alert(drift_id=7, workflow="support", level="P0", drift_type="out_of_scope_action",
              what="send_email is outside the workflow's tools", cause="The model changed from a to b.",
              action="Blocked the call.", run_id="r1", ts=1000.0)


def _capture(monkeypatch, status=202):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return seen


def test_alert_text():
    assert ALERT.title == "[P0] support: out_of_scope_action"
    assert ALERT.body == ("What drifted: send_email is outside the workflow's tools\n"
                          "What likely changed: The model changed from a to b.\nWhat PitStop did: Blocked the call.")
    assert ALERT.dedup_key == "pitstop:support:out_of_scope_action"


def test_alert_text_outside_a_workflow():
    from dataclasses import replace

    a = replace(ALERT, workflow=None, drift_type="budget_exhausted")
    assert a.title == "[P0] (no workflow): budget_exhausted"
    assert a.dedup_key == "pitstop:-:budget_exhausted"


@pytest.mark.parametrize("level,severity", [("P0", "critical"), ("P1", "error"), ("P2", "warning"), ("P3", "info")])
def test_pagerduty_payload(monkeypatch, level, severity):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "rk-123")
    seen = _capture(monkeypatch, 202)
    alert = Alert(**{**ALERT.__dict__, "level": level})
    channels.PagerDuty(PagerDutySpec()).send(alert)
    [req] = seen
    assert (req.method, str(req.url)) == ("POST", "https://events.pagerduty.com/v2/enqueue")
    body = json.loads(req.content)
    assert body["routing_key"] == "rk-123" and body["event_action"] == "trigger"
    assert body["dedup_key"] == "pitstop:support:out_of_scope_action"
    p = body["payload"]
    assert (p["summary"], p["source"], p["severity"]) == (alert.title, "pitstop", severity)
    assert p["custom_details"]["what"] == alert.what and p["custom_details"]["drift_id"] == 7


def test_pagerduty_rejects_non_202(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "rk-123")
    _capture(monkeypatch, 200)
    with pytest.raises(ChannelError) as e:
        channels.PagerDuty(PagerDutySpec()).send(ALERT)
    assert e.value.kind == "HTTP 200"


def test_opsgenie_payload_and_eu_region(monkeypatch):
    monkeypatch.setenv("OPSGENIE_API_KEY", "og-key")
    seen = _capture(monkeypatch, 202)
    long = Alert(**{**ALERT.__dict__, "level": "P1", "what": "x" * 20000, "workflow": "w" * 200})
    channels.Opsgenie(OpsgenieSpec()).send(long)
    channels.Opsgenie(OpsgenieSpec(region="eu")).send(ALERT)
    us, eu = seen
    assert str(us.url) == "https://api.opsgenie.com/v2/alerts"
    assert str(eu.url) == "https://api.eu.opsgenie.com/v2/alerts"
    assert us.headers["Authorization"] == "GenieKey og-key"
    b = json.loads(us.content)
    assert b["message"] == long.title[:130] and len(b["message"]) == 130
    assert b["alias"] == long.dedup_key and b["description"] == long.body[:15000]
    assert (b["priority"], b["source"]) == ("P2", "pitstop")
    assert json.loads(eu.content)["priority"] == "P1"


def test_slack_and_webhook_payloads(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T/B/X")
    monkeypatch.setenv("HOOK", "https://hook.test/in")
    seen = _capture(monkeypatch, 200)
    channels.Slack(SlackSpec()).send(ALERT)
    channels.Webhook(WebhookSpec(url_env="HOOK")).send(ALERT)
    slack, hook = seen
    assert str(slack.url) == "https://hooks.slack.test/T/B/X"
    assert json.loads(slack.content) == {"text": f"*{ALERT.title}*\n{ALERT.body}"}
    assert str(hook.url) == "https://hook.test/in"
    body = json.loads(hook.content)
    assert body == {**ALERT.__dict__, "title": ALERT.title, "body": ALERT.body}


def test_webhook_accepts_any_2xx_slack_only_200(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T/B/X")
    monkeypatch.setenv("PITSTOP_WEBHOOK_URL", "https://hook.test/in")
    _capture(monkeypatch, 204)
    channels.Webhook(WebhookSpec()).send(ALERT)
    with pytest.raises(ChannelError, match="HTTP 204"):
        channels.Slack(SlackSpec()).send(ALERT)


def test_transport_error_kind_is_type_name_only(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/SECRET123")

    def handler(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ChannelError) as e:
        channels.Slack(SlackSpec()).send(ALERT)
    assert e.value.kind == "ConnectError" and "SECRET123" not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


def test_missing_env_is_not_usable(monkeypatch):
    assert not channels.Slack(SlackSpec()).usable()
    with pytest.raises(ChannelError, match="not configured"):
        channels.Slack(SlackSpec()).send(ALERT)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/x")
    assert channels.Slack(SlackSpec()).usable()


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append(("quit",))
        return False

    def starttls(self, **kw):
        self.calls.append(("starttls",))

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append(("send", msg))


def _email(**kw):
    return EmailSpec(**{"smtp_host": "smtp.example.com", "from_addr": "pitstop@example.com",
                        "to": ("a@example.com", "b@example.com"), **kw})


def test_email_uses_starttls_and_login(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(channels, "_smtp", FakeSMTP)
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    channels.Email(_email(username="bot", smtp_port=2525)).send(ALERT)
    [s] = FakeSMTP.instances
    assert (s.host, s.port) == ("smtp.example.com", 2525)
    assert [c[0] for c in s.calls] == ["starttls", "login", "send", "quit"]
    assert s.calls[1] == ("login", "bot", "pw")
    msg = s.calls[2][1]
    assert msg["Subject"] == ALERT.title and msg["From"] == "pitstop@example.com"
    assert msg["To"] == "a@example.com, b@example.com"
    assert ALERT.body in msg.get_content()


def test_email_without_tls_or_login(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(channels, "_smtp", FakeSMTP)
    ch = channels.Email(_email(starttls=False))
    assert ch.usable()  # no username: no password needed
    ch.send(ALERT)
    assert [c[0] for c in FakeSMTP.instances[0].calls] == ["send", "quit"]
    assert not channels.Email(_email(username="bot")).usable()  # login needs SMTP_PASSWORD


def test_email_failure_kind_is_type_name(monkeypatch):
    import smtplib

    class Failing(FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad password pw for bot")

    monkeypatch.setattr(channels, "_smtp", Failing)
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    with pytest.raises(ChannelError) as e:
        channels.Email(_email(username="bot")).send(ALERT)
    assert e.value.kind == "SMTPAuthenticationError"
