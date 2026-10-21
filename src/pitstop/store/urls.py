"""Base URL normalization shared by the budget ledger and the `runs` table."""
from __future__ import annotations


def normalize_base_url(url: str | None) -> str | None:
    """Strip trailing slashes, so `https://host/v1/` and `https://host/v1` are the same provider."""
    return url.rstrip("/") if url is not None else None
