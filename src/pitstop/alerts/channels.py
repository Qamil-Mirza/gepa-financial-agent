"""Alert channels: PagerDuty Events v2, Opsgenie, Slack incoming webhook, SMTP email and a generic webhook.

Each channel reads its secret from the env var named in its config at send time, and `send(alert)` raises
ChannelError on any failure. A ChannelError's `kind` is only an exception type name or "HTTP <status>": never the
exception's message, a URL or a response body, which can hold the secret (Slack and webhook URLs are secrets)."""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
import threading
from dataclasses import asdict
from email.message import EmailMessage

import httpx

from ..userconfig import EmailSpec, OpsgenieSpec, PagerDutySpec, SlackSpec, WebhookSpec
from .model import Alert

PAGERDUTY_URL = "https://events.pagerduty.com/v2/enqueue"
OPSGENIE_URLS = {"us": "https://api.opsgenie.com/v2/alerts", "eu": "https://api.eu.opsgenie.com/v2/alerts"}
_PD_SEVERITY = {"P0": "critical", "P1": "error", "P2": "warning", "P3": "info"}
_OG_PRIORITY = {"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4"}
NOT_CONFIGURED = "not configured"


class ChannelError(Exception):
    """`retryable` is set when the channel itself knows whether a retry may help (SMTP: see `_smtp_retryable`);
    None leaves it to the dispatcher's rule for the kind."""

    def __init__(self, kind: str, *, retryable: bool | None = None):
        super().__init__(kind)
        self.kind = kind
        self.retryable = retryable


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=5.0)


def _smtp(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=10)


def _smtp_retryable(e: BaseException) -> bool:
    """A retry may pass after a dropped connection or a 4xx (transient) reply. Anything else is final: a 5xx reply
    (auth, sender, recipients, data, a refused greeting), recipients refused unless all with a 4xx, a failed
    certificate check, a missing feature, a generic
    SMTPException. SMTPException subclasses OSError, so it is checked first."""
    if isinstance(e, smtplib.SMTPServerDisconnected):
        return True
    if isinstance(e, smtplib.SMTPRecipientsRefused):  # {recipient: (code, message)}: retry only if all are 4xx
        codes = [r[0] for r in e.recipients.values()]
        return bool(codes) and all(isinstance(c, int) and 400 <= c < 500 for c in codes)
    if isinstance(e, smtplib.SMTPResponseException):
        return isinstance(e.smtp_code, int) and 400 <= e.smtp_code < 500
    if isinstance(e, smtplib.SMTPException):
        return False
    if isinstance(e, ssl.SSLCertVerificationError):  # an untrusted or mismatched certificate: a retry sees the same
        return False
    return isinstance(e, OSError)  # timeouts, refused or reset connections, DNS failures


_sending = threading.local()


class _QuietWhileSending(logging.Filter):
    """httpx logs every request URL at INFO; a Slack or webhook URL is a secret, so drop its records while an
    alert is being sent (only on this thread: the app's own httpx logging is untouched)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(_sending, "active", False)


logging.getLogger("httpx").addFilter(_QuietWhileSending())


def _post(url: str, body: dict, *, ok, headers: dict | None = None) -> None:
    _sending.active = True
    try:
        with _http_client() as client:
            r = client.post(url, json=body, headers=headers)
    except Exception as e:
        raise ChannelError(type(e).__name__) from None  # no chained context: its message can hold the URL
    finally:
        _sending.active = False
    if not ok(r.status_code):
        raise ChannelError(f"HTTP {r.status_code}")


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ChannelError(NOT_CONFIGURED)
    return value


class PagerDuty:
    name = "pagerduty"

    def __init__(self, spec: PagerDutySpec):
        self.spec = spec

    def usable(self) -> bool:
        return bool(os.environ.get(self.spec.routing_key_env))

    def send(self, alert: Alert) -> None:
        details = {"what": alert.what, "cause": alert.cause, "action": alert.action, "workflow": alert.workflow,
                   "drift_id": alert.drift_id, "run_id": alert.run_id, "level": alert.level}
        body = {"routing_key": _env(self.spec.routing_key_env), "event_action": "trigger",
                "dedup_key": alert.dedup_key,
                "payload": {"summary": alert.title[:1024], "source": "pitstop",
                            "severity": _PD_SEVERITY[alert.level], "custom_details": details}}
        _post(PAGERDUTY_URL, body, ok=lambda s: s == 202)


class Opsgenie:
    name = "opsgenie"

    def __init__(self, spec: OpsgenieSpec):
        self.spec = spec

    def usable(self) -> bool:
        return bool(os.environ.get(self.spec.api_key_env))

    def send(self, alert: Alert) -> None:
        key = _env(self.spec.api_key_env)
        body = {"message": alert.title[:130], "alias": alert.dedup_key, "description": alert.body[:15000],
                "priority": _OG_PRIORITY[alert.level], "source": "pitstop"}
        _post(OPSGENIE_URLS[self.spec.region], body, ok=lambda s: s == 202,
              headers={"Authorization": f"GenieKey {key}"})


class Slack:
    name = "slack"

    def __init__(self, spec: SlackSpec):
        self.spec = spec

    def usable(self) -> bool:
        return bool(os.environ.get(self.spec.webhook_url_env))

    def send(self, alert: Alert) -> None:
        _post(_env(self.spec.webhook_url_env), {"text": f"*{alert.title}*\n{alert.body}"}, ok=lambda s: s == 200)


class Webhook:
    name = "webhook"

    def __init__(self, spec: WebhookSpec):
        self.spec = spec

    def usable(self) -> bool:
        return bool(os.environ.get(self.spec.url_env))

    def send(self, alert: Alert) -> None:
        body = asdict(alert) | {"title": alert.title, "body": alert.body}
        _post(_env(self.spec.url_env), body, ok=lambda s: 200 <= s < 300)


class Email:
    """SMTP with STARTTLS (when `starttls`) and login (when `username`; the password comes from `password_env`)."""
    name = "email"

    def __init__(self, spec: EmailSpec):
        self.spec = spec

    def usable(self) -> bool:
        return self.spec.username is None or bool(os.environ.get(self.spec.password_env))

    def send(self, alert: Alert) -> None:
        s = self.spec
        password = _env(s.password_env) if s.username is not None else None
        try:
            msg = EmailMessage()
            msg["Subject"] = alert.title
            msg["From"] = s.from_addr
            msg["To"] = ", ".join(s.to)
            msg.set_content(alert.body)
            with _smtp(s.smtp_host, s.smtp_port) as smtp:
                if s.starttls:
                    smtp.starttls(context=ssl.create_default_context())
                if s.username is not None:
                    smtp.login(s.username, password)
                smtp.send_message(msg)
        except Exception as e:
            raise ChannelError(type(e).__name__, retryable=_smtp_retryable(e)) from None


_CLASSES = {"pagerduty": PagerDuty, "opsgenie": Opsgenie, "slack": Slack, "email": Email, "webhook": Webhook}


def configured(alerts, name: str):
    """The channel `name` built from its `[alerts.<name>]` table, or None when the table is absent."""
    spec = getattr(alerts, name, None)
    return _CLASSES[name](spec) if spec is not None else None
