"""User config (~/.pitstop/config.toml): budgets, price overrides, timezone."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import pitstop_home

_PERIODS = ("total", "day", "month")


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
    path = Path(path) if path is not None else pitstop_home() / "config.toml"
    if not path.exists():
        return UserConfig((), {}, None)
    data = tomllib.loads(path.read_text())
    budgets = []
    for name, raw in data.get("budgets", {}).items():
        period = raw.get("period", "total")
        if period not in _PERIODS:
            raise ValueError(f"budget {name!r}: invalid period {period!r} (expected one of {_PERIODS})")
        limit = float(raw.get("limit_usd", -1))
        if limit < 0:
            raise ValueError(f"budget {name!r}: limit_usd must be a non-negative number")
        budgets.append(BudgetSpec(name, limit, period, raw.get("base_url")))
    prices = {
        model: PriceOverride(float(p["input"]), float(p["output"]), float(p.get("cached_input", p["input"])))
        for model, p in data.get("prices", {}).items()
    }
    return UserConfig(tuple(budgets), prices, data.get("timezone"))
