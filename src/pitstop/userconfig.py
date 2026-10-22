"""User config (~/.pitstop/config.toml): budgets, price overrides, timezone and the L1 gate's judge."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class UserConfig:
    budgets: tuple[BudgetSpec, ...]
    prices: dict[str, PriceOverride]
    timezone: str | None
    judge: JudgeSpec = JudgeSpec()


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
    return UserConfig(tuple(budgets), prices, timezone, _parse_judge(data.get("judge", {})))


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
