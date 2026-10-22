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


def test_unknown_claude_model_uses_family_conservative_price():
    from pitstop.budget import prices

    assert prices._CONSERVATIVE_BY_PREFIX == {"claude-": Price(15.00, 75.00, 15.00)}
    assert price_for("claude-mystery-9", {}) == (Price(15.0, 75.0, 15.0), False)
    # user overrides still win over the family fallback; a non-matching id keeps the global fallback
    assert price_for("claude-mystery-9", {"claude-mystery-9": PriceOverride(3.0, 15.0, 0.3)}) == \
        (Price(3.0, 15.0, 0.3), True)
    assert price_for("acme/claude-x", {}) == (CONSERVATIVE_PRICE, False)


def test_longest_matching_prefix_wins(monkeypatch):
    from pitstop.budget import prices

    monkeypatch.setattr(prices, "_CONSERVATIVE_BY_PREFIX",
                        {"claude-": Price(15.0, 75.0, 15.0), "claude-haiku-": Price(1.0, 5.0, 1.0)})
    assert price_for("claude-haiku-9", {}) == (Price(1.0, 5.0, 1.0), False)
    assert price_for("claude-opus-9", {}) == (Price(15.0, 75.0, 15.0), False)
    # an exact table entry beats any prefix
    monkeypatch.setitem(prices.PRICES_PER_M, "claude-haiku-9", Price(0.8, 4.0, 0.08))
    assert price_for("claude-haiku-9", {}) == (Price(0.8, 4.0, 0.08), True)


def test_bedrock_style_claude_ids_use_family_conservative_price():
    for model in ("anthropic.claude-mystery-9-v1:0", "us.anthropic.claude-mystery-9-v1:0",
                  "global.anthropic.claude-mystery-9-v1:0"):
        assert price_for(model, {}) == (Price(15.0, 75.0, 15.0), False)
    assert price_for("anthropic.titan-x", {}) == (CONSERVATIVE_PRICE, False)
