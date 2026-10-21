from __future__ import annotations

import pytest

from pitstop.budget.prices import (
    CONSERVATIVE_PRICE, PRICES_PER_M, Price, cost_usd, estimate_cost_usd, price_for,
)
from pitstop.userconfig import PriceOverride

SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"


def test_price_override_wins_then_table_then_conservative():
    overrides = {SUPER: PriceOverride(1.0, 2.0, 0.5)}
    assert price_for(SUPER, overrides) == (Price(1.0, 2.0, 0.5), True)
    assert price_for(SUPER, {}) == (Price(0.30, 2.40, 0.15), True)
    assert price_for("acme/unknown", {}) == (CONSERVATIVE_PRICE, False)
    assert CONSERVATIVE_PRICE == Price(5.00, 15.00, 5.00)
    # An override can also price a model the table does not know.
    assert price_for("acme/custom", {"acme/custom": PriceOverride(0.1, 0.2, 0.1)}) == (Price(0.1, 0.2, 0.1), True)


def test_price_table_entries_are_prices():
    assert all(isinstance(p, Price) for p in PRICES_PER_M.values())
    assert PRICES_PER_M["deepseek-ai/Deepseek-V4-Flash"] == Price(0.14, 0.28, 0.03)


def test_cost_usd_matches_old_estimate():
    assert estimate_cost_usd(SUPER, 1_000_000, 1_000_000) == pytest.approx(0.30 + 2.40)
    assert estimate_cost_usd("unknown/model", 10, 10) is None
    # cached tokens are billed at the cached rate and never go negative
    p = Price(1.0, 2.0, 0.5)
    assert cost_usd(p, 1_000_000, 0, cached_tokens=400_000) == pytest.approx(0.6 + 0.2)
    assert cost_usd(p, 100, 0, cached_tokens=1_000_000) == pytest.approx(0.5)


def test_client_reexports_estimate_cost_usd():
    from pitstop.correct.client import estimate_cost_usd as reexported

    assert reexported is estimate_cost_usd
