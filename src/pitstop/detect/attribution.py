"""L5 attribution (spec 6.2): what changed around a change point in a workflow's runs.

`attribute` compares the runs just before `at` with the runs from `at` on, tag by tag (`TAGS`, columns of the
`runs` table). Before: up to `window` runs started before `at`, newest first; after: up to `window` runs started at
or after `at`, oldest first. A `refused` run (it never started) or a `dropped` one (a partial OTel trace) is in
neither window, as in `run_metrics`. A tag value is a change when it has at least NEW_SHARE_AFTER of the runs
after and at most MAX_SHARE_BEFORE of the runs before; its `before` is the tag's most common value before (ties go
to the newest run). A NULL tag value never counts as the new value.

`prompt_versions` is stored as JSON text (prompt name -> version); it is compared as JSON re-dumped with sorted
keys, so `{"b":1,"a":2}` and `{"a": 2, "b": 1}` are the same value and read `{"a": 2, "b": 1}` in the summary.
Text that is not JSON is compared as stored. Times in the summary are UTC, ISO 8601 to the minute
(`2026-10-22T14:05Z`); a NULL `before` reads `(unset)`."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from .metrics import run_metrics

TAGS = ("model", "base_url", "prompt_versions", "pitstop_version", "adapter")
WINDOW = 50
NEW_SHARE_AFTER = 0.5
MAX_SHARE_BEFORE = 0.1
_EXPLAINING_TAGS = ("model", "base_url", "prompt_versions")  # changes that escalate an L2 drift (urgency)
_EXCLUDED_STATUSES = ("refused", "dropped")


@dataclass(frozen=True)
class TagChange:
    tag: str
    before: str | None  # the most common value before
    after: str  # the new value
    share_before: float  # the new value's share of the runs before
    share_after: float  # ... and of the runs after


@dataclass(frozen=True)
class Attribution:
    workflow: str
    at: float
    changes: tuple[TagChange, ...]
    n_before: int
    n_after: int
    metric: str | None
    before_mean: float | None  # mean of run_metrics()[metric] over the runs that have it; None when none does
    after_mean: float | None
    summary: str


def _iso_minute(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _tag_value(tag: str, value) -> str | None:
    if value is None:
        return None
    if tag == "prompt_versions":
        try:
            return json.dumps(json.loads(value), sort_keys=True)
        except (TypeError, ValueError):
            pass
    return str(value)


def _window(store, workflow: str, at: float, window: int, *, before: bool) -> list[dict]:
    op, order = ("<", "DESC") if before else (">=", "ASC")
    rows = store.connect().execute(
        f"SELECT id, {', '.join(TAGS)} FROM runs WHERE workflow = ? AND started_at {op} ?"
        f" AND (status IS NULL OR status NOT IN ({', '.join('?' * len(_EXCLUDED_STATUSES))}))"
        f" ORDER BY started_at {order}, rowid {order} LIMIT ?",
        (workflow, at, *_EXCLUDED_STATUSES, max(window, 0))).fetchall()
    return [{"id": r["id"], **{t: _tag_value(t, r[t]) for t in TAGS}} for r in rows]


def _changes(before: list[dict], after: list[dict]) -> tuple[TagChange, ...]:
    if not before or not after:
        return ()
    out = []
    for tag in TAGS:
        counts_before = Counter(r[tag] for r in before)  # insertion order is newest first: ties go to the newest
        counts_after = Counter(r[tag] for r in after)
        usual = counts_before.most_common(1)[0][0]
        for value, n in counts_after.most_common():
            share_after, share_before = n / len(after), counts_before[value] / len(before)
            if value is not None and share_after >= NEW_SHARE_AFTER and share_before <= MAX_SHARE_BEFORE:
                out.append(TagChange(tag, usual, value, share_before, share_after))
    return tuple(out)


def _mean(store, runs: list[dict], metric: str | None) -> float | None:
    if metric is None:
        return None
    values = [m[metric] for r in runs if metric in (m := run_metrics(store, r["id"]))]
    return sum(values) / len(values) if values else None


def _summary(changes: tuple[TagChange, ...], n_before: int, at: float) -> str:
    iso = _iso_minute(at)
    if n_before == 0:
        return f"no runs before {iso} to compare"
    if not changes:
        return f"no run tag changed around {iso}; likely an input or upstream shift"
    return "; ".join(f"{c.tag} changed from {'(unset)' if c.before is None else c.before} to {c.after}"
                     f" ({c.share_after:.0%} of runs since {iso})" for c in changes)


def attribute(store, workflow: str, at: float, *, metric: str | None = None, window: int = WINDOW) -> Attribution:
    """Compare the workflow's runs before and after `at` (see the module docstring). Reads the store only."""
    before = _window(store, workflow, at, window, before=True)
    after = _window(store, workflow, at, window, before=False)
    changes = _changes(before, after)
    return Attribution(workflow, at, changes, len(before), len(after), metric, _mean(store, before, metric),
                       _mean(store, after, metric), _summary(changes, len(before), at))


def explains_change(a: Attribution) -> bool:
    """True when the attribution names a model, base_url or prompt_versions change (spec 5: a quality step right
    after one of these is P1; see `urgency.escalate_for_change`)."""
    return any(c.tag in _EXPLAINING_TAGS for c in a.changes)
