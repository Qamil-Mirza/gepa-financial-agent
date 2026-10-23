"""Small text helpers shared by the detectors (and the research example's supervisor signals)."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "what", "how", "is", "was", "its", "by", "with", "did"}


def _tokens(s: str) -> set[str]:
    return {w for w in _WORD.findall((s or "").lower()) if w not in _STOP}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
