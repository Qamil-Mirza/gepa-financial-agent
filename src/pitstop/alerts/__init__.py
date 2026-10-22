"""Alerts: P0/P1 drifts go straight to PagerDuty, Opsgenie, Slack, email or a webhook (routes in config.toml,
`[alerts.routes]`); P2/P3 drifts go in a daily digest (`pitstop alerts digest`). Each alert says what drifted,
what likely changed and what PitStop did. Secrets come only from env vars named in config.toml."""
from .dispatch import digest, notify, send_test, wait_idle
from .model import Alert, build_alert

__all__ = ["Alert", "build_alert", "notify", "digest", "send_test", "wait_idle"]
