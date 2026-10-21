"""Exceptions the SDK raises into the user's agent code."""
from __future__ import annotations


class Halted(Exception):
    """PitStop stopped the workflow (a hard drift or an exhausted budget)."""

    def __init__(self, reason: str, drift_id: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.drift_id = drift_id


class BudgetExceeded(Halted):
    def __init__(self, scope: str, limit_usd: float, spent_usd: float):
        self.scope = scope
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        super().__init__(f"PitStop budget exceeded: {scope} spent ${spent_usd:.2f} of ${limit_usd:.2f}")
