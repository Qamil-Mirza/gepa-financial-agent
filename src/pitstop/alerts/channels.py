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
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=5.0)


def _smtp(host: str, port: int) -> smtplib.SMTP:
    return smtplib.SMTP(host, port, timeout=10)


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
            raise ChannelError(type(e).__name__) from None


_CLASSES = {"pagerduty": PagerDuty, "opsgenie": Opsgenie, "slack": Slack, "email": Email, "webhook": Webhook}


def configured(alerts, name: str):
    """The channel `name` built from its `[alerts.<name>]` table, or None when the table is absent."""
    spec = getattr(alerts, name, None)
    return _CLASSES[name](spec) if spec is not None else None
