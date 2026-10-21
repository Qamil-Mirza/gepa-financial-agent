"""Model prices (USD per 1M tokens) and cost arithmetic."""
from __future__ import annotations

from dataclasses import dataclass

from ..userconfig import PriceOverride


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cached_input: float


# Source: docs/skills/crusoe-recursive-hack-skill.md §4.1
PRICES_PER_M: dict[str, Price] = {
    "deepseek-ai/Deepseek-V4-Flash": Price(0.14, 0.28, 0.03),
    "deepseek-ai/DeepSeek-V4.1-Flash": Price(0.29, 0.20, 0.007),
    "deepseek-ai/DeepSeek-V4-Pro": Price(1.74, 3.48, 0.15),
    "nvidia/Nemotron-3-Nano-30B-A3B": Price(0.05, 0.20, 0.03),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": Price(0.05, 0.20, 0.03),  # id as served by /v1/models
    "nvidia/nemotron-3.5-lightning-30b-a3b": Price(0.05, 0.20, 0.03),
    "nvidia/Nemotron-3-Super-120B-A12B": Price(0.30, 2.40, 0.15),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": Price(0.30, 2.40, 0.15),  # id as served by /v1/models
    "openai/gpt-oss-120b": Price(0.05, 0.20, 0.05),
    "zai-org/GLM-5.3-Flash": Price(0.15, 0.50, 0.03),
    "google/gemma-4-31b-it": Price(0.14, 0.40, 0.14),
}

# Used for models we cannot price, so budgets fail safe (over-reserve) rather than open.
CONSERVATIVE_PRICE = Price(5.00, 15.00, 5.00)


def price_for(model: str, overrides: dict[str, PriceOverride]) -> tuple[Price, bool]:
    """(price, known): user overrides win, then the built-in table, then CONSERVATIVE_PRICE (known=False)."""
    o = overrides.get(model)
    if o is not None:
        return Price(o.input, o.output, o.cached_input), True
    p = PRICES_PER_M.get(model)
    if p is not None:
        return p, True
    return CONSERVATIVE_PRICE, False


def cost_usd(price: Price, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * price.input + completion_tokens * price.output + cached_tokens * price.cached_input) / 1_000_000


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float | None:
    p = PRICES_PER_M.get(model)
    if p is None:
        return None
    return cost_usd(p, prompt_tokens, completion_tokens, cached_tokens)
