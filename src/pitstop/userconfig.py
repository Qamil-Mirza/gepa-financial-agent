"""User config (~/.pitstop/config.toml): budgets, price overrides, timezone."""
from __future__ import annotations

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
class UserConfig:
    budgets: tuple[BudgetSpec, ...]
    prices: dict[str, PriceOverride]
    timezone: str | None


def load_user_config(path: Path | None = None) -> UserConfig:
    """Parse config.toml. Any problem with its content raises ConfigInvalid (a ValueError)."""
    path = Path(path) if path is not None else pitstop_home() / "config.toml"
    if not path.exists():
        return UserConfig((), {}, None)
    try:
        return _parse(tomllib.loads(path.read_text()))
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
        except (TypeError, ValueError):
            limit = -1.0
        if not limit >= 0:  # also rejects NaN
            raise ConfigInvalid(f"budget {name!r}: limit_usd must be a non-negative number")
        budgets.append(BudgetSpec(name, limit, period, raw.get("base_url")))
    prices = {
        model: PriceOverride(float(p["input"]), float(p["output"]), float(p.get("cached_input", p["input"])))
        for model, p in data.get("prices", {}).items()
    }
    timezone = data.get("timezone") or None  # "" means unset (system local time)
    if timezone is not None:
        _check_timezone(timezone)
    return UserConfig(tuple(budgets), prices, timezone)


def _check_timezone(name: object) -> None:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not isinstance(name, str):
        raise ConfigInvalid(f"timezone must be an IANA name like 'America/New_York', got {name!r}")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ConfigInvalid(f"unknown timezone {name!r}") from e
