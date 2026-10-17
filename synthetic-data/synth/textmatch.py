"""Small deterministic text helpers shared by the research bank and the verifier."""
from __future__ import annotations

import re
from typing import Iterable, List, Set

_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "by", "with", "what", "is", "are",
    "how", "does", "do", "its", "it", "most", "recent", "latest", "last", "vs", "versus", "from",
}


def tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


_PERIOD_WORDS = {"q1", "q2", "q3", "q4", "first", "second", "third", "fourth", "h1", "h2"}


ENTITIES = {"nvidia", "amd", "microsoft", "alphabet", "google", "amazon", "aws", "meta", "tsmc", "intel",
            "qualcomm", "apple", "broadcom"}


def _is_period(tok: str) -> bool:
    return tok in _PERIOD_WORDS or tok in ENTITIES or any(c.isdigit() for c in tok)


def same_question(a: str, b: str, threshold: float = 0.5) -> bool:
    """Similar wording AND same periods, numbers and companies (Q2 vs Q3, NVIDIA vs AMD are different questions)."""
    ta, tb = tokens(a), tokens(b)
    if any(_is_period(t) for t in ta ^ tb):
        return False
    return jaccard(a, b) >= threshold


def normalize_query(q: str) -> str:
    return " ".join(sorted(tokens(q)))


def keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    """Keywords that occur in text as whole words/phrases (case-insensitive)."""
    low = text.lower()
    return [k for k in keywords if re.search(r"(?<![a-z0-9])" + re.escape(k.lower()) + r"(?![a-z0-9])", low)]


def numbers(text: str) -> Set[str]:
    """Numeric values in text, normalized ("1,234.50" -> "1234.5"). Years and small ints are dropped."""
    out = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        try:
            v = float(n)
        except ValueError:
            continue
        if v < 10 and "." not in n:
            continue  # quarter numbers, list indices
        if 1990 <= v <= 2040 and "." not in n:
            continue  # years / fiscal years
        out.add(n)
    return out
