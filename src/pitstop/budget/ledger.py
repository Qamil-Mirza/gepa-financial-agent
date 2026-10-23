"""Hard-stop budget ledger: atomic reserve -> settle/release against the local SQLite store.

Every model call first *reserves* its worst-case cost (estimated prompt tokens + max completion tokens).
The check against every applicable budget and the insert of the `reserved` row happen in one
`BEGIN IMMEDIATE` transaction, so concurrent threads and processes sharing the DB file can never
jointly reserve past a limit. Ledger writes therefore refuse to run inside another transaction (they would
become a savepoint without the write lock, and an outer rollback would erase a reservation whose call ran). After the call, `settle` replaces the estimate with the actual cost;
on failure `release` frees it. A reservation left `reserved` by a process that died mid-call keeps counting
until `release_stale` (`pitstop budget reconcile --release`) frees it. Once a budget is used up, `exhausted`
names it, so a new run is refused at its start instead of at its first call.

Scopes are derived at query time (the ledger table has no scope column):
  global:<budget>   rows whose base_url matches the BudgetSpec's (all rows if it has none), every meter
  workflow:<name>   rows of that workflow with meter == 'agent', calendar-day window
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..sdk.errors import BudgetExceeded
from ..store import Store
from ..store.urls import normalize_base_url
from ..userconfig import BudgetSpec, UserConfig
from .prices import Price, cost_usd, price_for

# Float slack so that e.g. 50 x $0.10 against $5.00 is not rejected by representation error.
_EPS = 1e-9
_LIVE = ("reserved", "settled")
_NESTED = "ledger operations must not run inside another transaction"
_STALE_COLS = "id, ts, run_id, workflow, meter, model, usd"


@dataclass
class Reservation:
    id: int
    usd: float
    scopes: tuple[str, ...]
    model: str
    base_url: str | None
    price: Price
    price_known: bool


class Ledger:
    def __init__(self, store: Store, config: UserConfig, now: Callable[[], float] = time.time):
        self.store = store
        self.config = config
        self._now = now
        self._tz = ZoneInfo(config.timezone) if config.timezone else None
        self._budgets: dict[str, BudgetSpec] = {b.name: b for b in config.budgets}

    # -- public API -------------------------------------------------------------------------------

    @property
    def tz(self) -> ZoneInfo | None:
        """The configured timezone for day/month windows (None: system local time)."""
        return self._tz

    def reserve(self, *, model: str, base_url: str | None, est_prompt_tokens: int, max_completion_tokens: int,
                run_id: str | None, workflow: str | None, workflow_budget_usd: float | None,
                meter: str = "agent") -> Reservation:
        price, known = price_for(model, self.config.prices)
        usd = cost_usd(price, est_prompt_tokens, max_completion_tokens)
        url = normalize_base_url(base_url)
        checks: list[tuple[str, float]] = [
            (f"global:{b.name}", b.limit_usd) for b in self.config.budgets
            if b.base_url is None or normalize_base_url(b.base_url) == url
        ]
        if workflow_budget_usd is not None and meter == "agent":
            # Workflow budgets only count agent-meter spend, so only agent calls are checked against them.
            if workflow is None:
                raise ValueError("workflow_budget_usd requires a workflow name")
            checks.append((f"workflow:{workflow}", workflow_budget_usd))
        now = self._now()
        with self._write() as conn:
            for scope, limit in checks:
                spent = self._spent(conn, scope, now)
                if spent + usd > limit + _EPS:
                    raise BudgetExceeded(scope, limit, spent)
            cur = conn.execute(
                "INSERT INTO ledger(ts, run_id, workflow, meter, model, base_url, prompt_tokens, completion_tokens,"
                " usd, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved')",
                (now, run_id, workflow, meter, model, url, est_prompt_tokens, max_completion_tokens, usd),
            )
            rid = cur.lastrowid
        return Reservation(rid, usd, tuple(s for s, _ in checks), model, url, price, known)

    def settle(self, res: Reservation, *, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        """Replace the reservation with the actual cost (at the price fixed at reserve time). Returns the USD."""
        actual = cost_usd(res.price, prompt_tokens, completion_tokens, cached_tokens)
        with self._write() as conn:
            conn.execute(
                "UPDATE ledger SET usd=?, prompt_tokens=?, completion_tokens=?, state='settled' WHERE id=?",
                (actual, prompt_tokens, completion_tokens, res.id),
            )
        return actual

    def release(self, res: Reservation) -> None:
        """Free a reservation whose call did not happen. A settled row is never released."""
        with self._write() as conn:
            conn.execute("UPDATE ledger SET state='released' WHERE id=? AND state='reserved'", (res.id,))

    def stale_reservations(self, older_than_s: float) -> list[dict]:
        """Rows still `reserved` after `older_than_s` seconds: left by a process that died between reserve and
        settle/release (they hold budget no call will ever settle), or by a call that is still running."""
        rows = self.store.connect().execute(
            f"SELECT {_STALE_COLS} FROM ledger WHERE state='reserved' AND ts < ? ORDER BY id",
            (self._now() - older_than_s,)).fetchall()
        return [dict(r) for r in rows]

    def release_stale(self, older_than_s: float) -> int:
        """Release every reservation older than `older_than_s` seconds. Returns how many were released."""
        return len(self.release_stale_rows(older_than_s))

    def release_stale_rows(self, older_than_s: float) -> list[dict]:
        """Release every reservation older than `older_than_s` seconds and return exactly the rows released
        (selected and updated under one write lock, so a concurrent settle cannot slip in between)."""
        with self._write() as conn:
            rows = [dict(r) for r in conn.execute(
                f"SELECT {_STALE_COLS} FROM ledger WHERE state='reserved' AND ts < ? ORDER BY id",
                (self._now() - older_than_s,))]
            conn.executemany("UPDATE ledger SET state='released' WHERE id=? AND state='reserved'",
                             [(r["id"],) for r in rows])
        return rows

    def exhausted(self, *, workflow: str, workflow_budget_usd: float | None,
                  base_url: str | None) -> BudgetExceeded | None:
        """The first budget a new run of `workflow` would find used up (spent >= limit), as the exception to
        raise, else None. Checked in order: the workflow's own budget (when set), every global budget without a
        base_url, then the global budgets for `base_url` (none when it is None). Read-only."""
        checks: list[tuple[str, float]] = []
        if workflow_budget_usd is not None:
            checks.append((f"workflow:{workflow}", workflow_budget_usd))
        checks += [(f"global:{b.name}", b.limit_usd) for b in self.config.budgets if b.base_url is None]
        if base_url is not None:
            url = normalize_base_url(base_url)
            checks += [(f"global:{b.name}", b.limit_usd) for b in self.config.budgets
                       if b.base_url is not None and normalize_base_url(b.base_url) == url]
        conn, now = self.store.connect(), self._now()
        for scope, limit in checks:
            spent = self._spent(conn, scope, now)
            if spent >= limit - _EPS:
                return BudgetExceeded(scope, limit, spent)
        return None

    def spent(self, scope: str) -> float:
        """Settled + reserved USD for `scope` in its current period window."""
        return self._spent(self.store.connect(), scope, self._now())

    # -- internals --------------------------------------------------------------------------------

    def _write(self):
        """The outermost BEGIN IMMEDIATE transaction for a ledger write; never a savepoint of a caller's."""
        if self.store.connect().in_transaction:
            raise RuntimeError(_NESTED)
        return self.store.transaction(immediate=True)

    def _spent(self, conn: sqlite3.Connection, scope: str, now: float) -> float:
        kind, sep, name = scope.partition(":")
        if not sep or not name:
            raise ValueError(f"invalid budget scope {scope!r}")
        sql = "SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE state IN (?, ?)"
        params: list = list(_LIVE)
        if kind == "global":
            spec = self._budgets.get(name)
            if spec is None:
                raise ValueError(f"unknown budget {name!r}")
            period = spec.period
            if spec.base_url is not None:
                sql += " AND rtrim(base_url, '/') = ?"
                params.append(normalize_base_url(spec.base_url))
        elif kind == "workflow":
            period = "day"
            sql += " AND workflow = ? AND meter = 'agent'"
            params.append(name)
        else:
            raise ValueError(f"invalid budget scope {scope!r}")
        start = self._period_start(period, now)
        if start is not None:
            sql += " AND ts >= ?"
            params.append(start)
        return float(conn.execute(sql, params).fetchone()[0])

    def _period_start(self, period: str, now: float) -> float | None:
        if period == "total":
            return None
        # Naive local datetime when no timezone is configured: .timestamp() then interprets it as system local time.
        dt = datetime.fromtimestamp(now, self._tz) if self._tz else datetime.fromtimestamp(now)
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "month":
            dt = dt.replace(day=1)
        elif period != "day":
            raise ValueError(f"invalid budget period {period!r}")
        return dt.timestamp()
