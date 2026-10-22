"""User config (~/.pitstop/config.toml): budgets, price overrides, timezone, the L1 gate's judge and alerts."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import pitstop_home

_PERIODS = ("total", "day", "month")


class ConfigInvalid(ValueError):
    """config.toml cannot be used (bad TOML, period, limit, price or timezone).

    Distinct from transient store errors: callers on the model-call path fail closed on it (the hard budget
    must never silently disappear), while sqlite busy/locked errors stay fail-open."""


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    limit_usd: float
    period: Literal["total", "day", "month"]
    base_url: str | None


@dataclass(frozen=True)
class PriceOverride:
    input: float
    output: float
    cached_input: float


@dataclass(frozen=True)
class JudgeSpec:
    """The L1 pre-action gate's judge. The key itself never lives in config: only the env var's name."""
    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.inference.crusoecloud.com/v1"
    api_key_env: str = "CRUSOE_API_KEY"
    timeout_s: float = 10.0


# Alert channels. Secrets (routing keys, API keys, webhook URLs, SMTP passwords) never live in config: only the
# names of the env vars that hold them. A channel is used only when its table is present and its env var is set.
@dataclass(frozen=True)
class PagerDutySpec:
    routing_key_env: str = "PAGERDUTY_ROUTING_KEY"


@dataclass(frozen=True)
class OpsgenieSpec:
    api_key_env: str = "OPSGENIE_API_KEY"
    region: Literal["us", "eu"] = "us"


@dataclass(frozen=True)
class SlackSpec:
    webhook_url_env: str = "SLACK_WEBHOOK_URL"


@dataclass(frozen=True)
class EmailSpec:
    smtp_host: str
    from_addr: str
    to: tuple[str, ...]
    smtp_port: int = 587
    starttls: bool = True
    username: str | None = None
    password_env: str = "SMTP_PASSWORD"


@dataclass(frozen=True)
class WebhookSpec:
    url_env: str = "PITSTOP_WEBHOOK_URL"


ALERT_CHANNELS = ("pagerduty", "opsgenie", "slack", "email", "webhook")
DEFAULT_ROUTES: dict[str, tuple[str, ...]] = {
    "P0": ("pagerduty", "opsgenie", "slack", "webhook"),
    "P1": ("slack", "email", "webhook"),
    "digest": ("email", "slack", "webhook"),
}
_INLINE_SECRETS = ("routing_key", "api_key", "webhook_url", "password", "url")


@dataclass(frozen=True)
class AlertsSpec:
    """`[alerts.<channel>]` tables (None when absent) and `[alerts.routes]` (P0, P1, digest -> channel names)."""
    pagerduty: PagerDutySpec | None = None
    opsgenie: OpsgenieSpec | None = None
    slack: SlackSpec | None = None
    email: EmailSpec | None = None
    webhook: WebhookSpec | None = None
    routes: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_ROUTES))


@dataclass(frozen=True)
class UserConfig:
    budgets: tuple[BudgetSpec, ...]
    prices: dict[str, PriceOverride]
    timezone: str | None
    judge: JudgeSpec = JudgeSpec()
    alerts: AlertsSpec = field(default_factory=AlertsSpec)


def load_user_config(path: Path | None = None) -> UserConfig:
    """Parse config.toml. Any problem with its content raises ConfigInvalid (a ValueError)."""
    path = Path(path) if path is not None else pitstop_home() / "config.toml"
    if not path.exists():
        return UserConfig((), {}, None)
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigInvalid(f"{path}: cannot read config: {type(e).__name__}") from e
    try:
        return _parse(tomllib.loads(text))
    except ConfigInvalid as e:
        raise ConfigInvalid(f"{path}: {e}") from e
    except (ValueError, TypeError, KeyError, AttributeError) as e:  # bad TOML, wrong types, missing keys
        raise ConfigInvalid(f"{path}: {type(e).__name__}: {e}") from e


def _parse(data: dict) -> UserConfig:
    budgets = []
    for name, raw in data.get("budgets", {}).items():
        period = raw.get("period", "total")
        if period not in _PERIODS:
            raise ConfigInvalid(f"budget {name!r}: invalid period {period!r} (expected one of {_PERIODS})")
        try:
            limit = float(raw.get("limit_usd", -1))
        except (TypeError, ValueError, OverflowError):
            limit = -1.0
        if not limit >= 0:  # also rejects NaN
            raise ConfigInvalid(f"budget {name!r}: limit_usd must be a non-negative number")
        base_url = raw.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ConfigInvalid(f"budget {name!r}: base_url must be a string")
        budgets.append(BudgetSpec(name, limit, period, base_url))
    prices = {}
    for model, p in data.get("prices", {}).items():
        inp = _finite_nonneg(p["input"], f"price {model!r} input")
        out = _finite_nonneg(p["output"], f"price {model!r} output")
        cached = _finite_nonneg(p["cached_input"], f"price {model!r} cached_input") if "cached_input" in p else inp
        prices[model] = PriceOverride(inp, out, cached)
    timezone = data.get("timezone") or None  # "" means unset (system local time)
    if timezone is not None:
        _check_timezone(timezone)
    return UserConfig(tuple(budgets), prices, timezone, _parse_judge(data.get("judge", {})),
                      _parse_alerts(data.get("alerts", {})))


def _parse_judge(raw: object) -> JudgeSpec:
    if not isinstance(raw, dict):
        raise ConfigInvalid("[judge] must be a table")
    default = JudgeSpec()
    fields = {}
    for key in ("model", "base_url", "api_key_env"):
        value = raw.get(key, getattr(default, key))
        if not isinstance(value, str):
            raise ConfigInvalid(f"judge {key} must be a string")
        fields[key] = value
    timeout = _finite_nonneg(raw.get("timeout_s", default.timeout_s), "judge timeout_s")
    if not timeout > 0:
        raise ConfigInvalid("judge timeout_s must be greater than 0")
    return JudgeSpec(timeout_s=timeout, **fields)


def _parse_alerts(raw: object) -> AlertsSpec:
    if not isinstance(raw, dict):
        raise ConfigInvalid("[alerts] must be a table")
    unknown = set(raw) - set(ALERT_CHANNELS) - {"routes"}
    if unknown:
        raise ConfigInvalid(f"alerts: unknown channel(s) {sorted(unknown)} (expected {ALERT_CHANNELS} or routes)")
    channels = {}
    for name in ALERT_CHANNELS:
        if name in raw:
            channels[name] = _parse_channel(name, raw[name])
    return AlertsSpec(**channels, routes=_parse_routes(raw.get("routes", {})))


_CHANNEL_SPECS = {"pagerduty": PagerDutySpec, "opsgenie": OpsgenieSpec, "slack": SlackSpec, "email": EmailSpec,
                  "webhook": WebhookSpec}


def _parse_channel(name: str, raw: object):
    what = f"alerts.{name}"
    if not isinstance(raw, dict):
        raise ConfigInvalid(f"[{what}] must be a table")
    for key in _INLINE_SECRETS:  # the value is never echoed: it is a secret
        if key in raw:
            raise ConfigInvalid(f"{what}: {key} is not allowed here; put secrets in environment variables; "
                                f"use {key}_env")
    spec_cls = _CHANNEL_SPECS[name]
    allowed = set(spec_cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigInvalid(f"{what}: unknown key(s) {sorted(unknown)} (expected {sorted(allowed)})")
    fields = {}
    for key, value in raw.items():
        if key.endswith("_env") or key in ("smtp_host", "from_addr", "username"):
            if not isinstance(value, str) or not value:
                raise ConfigInvalid(f"{what} {key} must be a non-empty string")
        fields[key] = value
    if name == "opsgenie" and fields.get("region", "us") not in ("us", "eu"):
        raise ConfigInvalid(f"{what} region must be \"us\" or \"eu\"")
    if name == "email":
        for key in ("smtp_host", "from_addr"):
            if key not in fields:
                raise ConfigInvalid(f"{what}: {key} is required")
        to = fields.get("to")
        if not isinstance(to, list) or not to or not all(isinstance(a, str) and a for a in to):
            raise ConfigInvalid(f"{what} to must be a non-empty list of addresses")
        fields["to"] = tuple(to)
        if "smtp_port" in fields:
            port = _finite_nonneg(fields["smtp_port"], f"{what} smtp_port")
            if not (isinstance(fields["smtp_port"], int) and 1 <= port <= 65535):
                raise ConfigInvalid(f"{what} smtp_port must be an integer from 1 to 65535")
        if not isinstance(fields.get("starttls", True), bool):
            raise ConfigInvalid(f"{what} starttls must be true or false")
    return spec_cls(**fields)


def _parse_routes(raw: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        raise ConfigInvalid("[alerts.routes] must be a table")
    unknown = set(raw) - set(DEFAULT_ROUTES)
    if unknown:
        raise ConfigInvalid(f"alerts.routes: unknown route(s) {sorted(unknown)} (expected {list(DEFAULT_ROUTES)})")
    routes = dict(DEFAULT_ROUTES)
    for key, names in raw.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ConfigInvalid(f"alerts.routes {key} must be a list of channel names")
        bad = [n for n in names if n not in ALERT_CHANNELS]
        if bad:
            raise ConfigInvalid(f"alerts.routes {key}: unknown channel(s) {bad} (expected {ALERT_CHANNELS})")
        routes[key] = tuple(dict.fromkeys(names))
    return routes


def _finite_nonneg(value: object, what: str) -> float:
    """A finite, non-negative real number (bool, NaN, +-inf, negatives and non-numbers raise ConfigInvalid)."""
    try:
        ok = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        result = float(value) if ok else 0.0  # a huge int raises OverflowError here
    except OverflowError:
        ok = False
    if not ok:
        raise ConfigInvalid(f"{what} must be a finite non-negative number")
    return result


def _check_timezone(name: object) -> None:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not isinstance(name, str):
        raise ConfigInvalid(f"timezone must be an IANA name like 'America/New_York', got {name!r}")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ConfigInvalid(f"unknown timezone {name!r}") from e
