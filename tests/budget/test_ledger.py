from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from pitstop.budget.ledger import Ledger, Reservation
from pitstop.sdk.errors import BudgetExceeded, Halted
from pitstop.store import Store
from pitstop.userconfig import BudgetSpec, PriceOverride, UserConfig

CRUSOE = "https://api.crusoe.ai/v1"
MODEL = "acme/flat"  # $1 per 1M input tokens, $2 per 1M output, $0.5 per 1M cached
PRICES = {MODEL: PriceOverride(1.0, 2.0, 0.5)}


def _cfg(*budgets: BudgetSpec, tz: str | None = "UTC") -> UserConfig:
    return UserConfig(tuple(budgets), dict(PRICES), tz)


def _ledger(tmp_path, *budgets: BudgetSpec, now=None, tz: str | None = "UTC") -> Ledger:
    store = Store(tmp_path / "pitstop.db")
    if now is None:
        return Ledger(store, _cfg(*budgets, tz=tz))
    return Ledger(store, _cfg(*budgets, tz=tz), now=now)


def _reserve(ledger: Ledger, usd: float, **kw) -> Reservation:
    """Reserve exactly `usd` dollars of input tokens on MODEL."""
    args = dict(model=MODEL, base_url=CRUSOE, est_prompt_tokens=round(usd * 1_000_000), max_completion_tokens=0,
                run_id="r1", workflow=None, workflow_budget_usd=None)
    args.update(kw)
    return ledger.reserve(**args)


def _states(ledger: Ledger) -> list[str]:
    return [r["state"] for r in ledger.store.connect().execute("SELECT state FROM ledger ORDER BY id")]


def test_reserve_then_settle_records_actual(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 10.0, "total", CRUSOE))
    res = _reserve(ledger, 0.5, max_completion_tokens=100_000)
    assert res.scopes == ("global:crusoe",)
    assert res.price_known is True
    assert res.usd == pytest.approx(0.5 + 0.2)
    assert ledger.spent("global:crusoe") == pytest.approx(0.7)
    actual = ledger.settle(res, prompt_tokens=200_000, completion_tokens=50_000, cached_tokens=100_000)
    assert actual == pytest.approx(0.1 + 0.05 + 0.1)
    assert ledger.spent("global:crusoe") == pytest.approx(actual)
    assert _states(ledger) == ["settled"]


def test_reserve_over_limit_raises_budget_exceeded(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 0.01, "total", CRUSOE))
    with pytest.raises(BudgetExceeded) as ei:
        _reserve(ledger, 0.02)
    err = ei.value
    assert isinstance(err, Halted)
    assert err.scope == "global:crusoe"
    assert err.limit_usd == 0.01
    assert err.spent_usd == 0.0
    assert err.drift_id is None
    assert str(err).startswith("PitStop budget exceeded: global:crusoe spent $")
    assert err.reason == str(err)
    assert _states(ledger) == []  # nothing was written


def test_unknown_model_reserves_at_conservative_price(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 1.0, "total", None))
    res = ledger.reserve(model="acme/mystery", base_url=CRUSOE, est_prompt_tokens=100_000,
                         max_completion_tokens=10_000, run_id=None, workflow=None, workflow_budget_usd=None)
    assert res.price_known is False
    assert res.usd == pytest.approx(0.5 + 0.15)


def test_release_frees_reservation(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 1.0, "total", CRUSOE))
    res = _reserve(ledger, 0.8)
    with pytest.raises(BudgetExceeded):
        _reserve(ledger, 0.5)
    ledger.release(res)
    assert ledger.spent("global:crusoe") == 0.0
    _reserve(ledger, 0.5)  # room again
    assert _states(ledger) == ["released", "reserved"]


def test_release_does_not_undo_a_settled_row(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 1.0, "total", CRUSOE))
    res = _reserve(ledger, 0.3)
    ledger.settle(res, prompt_tokens=300_000, completion_tokens=0)
    ledger.release(res)
    assert _states(ledger) == ["settled"]
    assert ledger.spent("global:crusoe") == pytest.approx(0.3)


def test_base_url_scoping(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 1.0, "total", CRUSOE + "/"), BudgetSpec("all", 10.0, "total", None))
    res = _reserve(ledger, 0.4, base_url="https://api.openai.com/v1")
    assert res.scopes == ("global:all",)
    assert ledger.spent("global:crusoe") == 0.0
    assert ledger.spent("global:all") == pytest.approx(0.4)
    res = _reserve(ledger, 0.3, base_url=CRUSOE)  # no trailing slash still matches
    assert set(res.scopes) == {"global:crusoe", "global:all"}
    assert ledger.spent("global:crusoe") == pytest.approx(0.3)
    assert ledger.spent("global:all") == pytest.approx(0.7)
    # a budget scoped to another provider never blocks this one
    _reserve(ledger, 0.9, base_url="https://api.openai.com/v1")


def test_day_period_resets_at_local_midnight(tmp_path):
    clock = {"t": datetime(2026, 10, 20, 23, 59, tzinfo=timezone.utc).timestamp()}
    ledger = _ledger(tmp_path, BudgetSpec("daily", 1.0, "day", None), now=lambda: clock["t"], tz="UTC")
    _reserve(ledger, 0.6)
    assert ledger.spent("global:daily") == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        _reserve(ledger, 0.6)
    clock["t"] = datetime(2026, 10, 21, 0, 1, tzinfo=timezone.utc).timestamp()
    assert ledger.spent("global:daily") == 0.0
    _reserve(ledger, 0.6)


def test_day_period_uses_configured_timezone(tmp_path):
    # 23:30 UTC on Oct 20 is already 08:30 Oct 21 in Tokyo; 00:30 UTC Oct 21 is the same Tokyo day.
    clock = {"t": datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc).timestamp()}
    ledger = _ledger(tmp_path, BudgetSpec("daily", 1.0, "day", None), now=lambda: clock["t"], tz="Asia/Tokyo")
    _reserve(ledger, 0.6)
    clock["t"] = datetime(2026, 10, 21, 0, 30, tzinfo=timezone.utc).timestamp()
    assert ledger.spent("global:daily") == pytest.approx(0.6)


def test_month_and_total_periods(tmp_path):
    clock = {"t": datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc).timestamp()}
    ledger = _ledger(tmp_path, BudgetSpec("monthly", 5.0, "month", None), BudgetSpec("forever", 5.0, "total", None),
                     now=lambda: clock["t"], tz="UTC")
    _reserve(ledger, 1.0)
    clock["t"] = datetime(2026, 11, 1, 0, 1, tzinfo=timezone.utc).timestamp()
    assert ledger.spent("global:monthly") == 0.0
    assert ledger.spent("global:forever") == pytest.approx(1.0)


def test_system_local_timezone_when_unset(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("daily", 1.0, "day", None), tz=None)
    _reserve(ledger, 0.2)
    assert ledger.spent("global:daily") == pytest.approx(0.2)


def test_workflow_budget_counts_agent_meter_only(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("all", 100.0, "total", None))
    res = _reserve(ledger, 0.6, workflow="wf", workflow_budget_usd=1.0)
    assert set(res.scopes) == {"global:all", "workflow:wf"}
    _reserve(ledger, 5.0, workflow="wf", meter="supervisor", workflow_budget_usd=1.0)
    _reserve(ledger, 5.0, workflow="other", workflow_budget_usd=10.0)
    assert ledger.spent("workflow:wf") == pytest.approx(0.6)
    assert ledger.spent("global:all") == pytest.approx(10.6)  # global counts every meter
    with pytest.raises(BudgetExceeded) as ei:
        _reserve(ledger, 0.5, workflow="wf", workflow_budget_usd=1.0)
    assert ei.value.scope == "workflow:wf"
    assert ei.value.limit_usd == 1.0


def test_unknown_scope_raises(tmp_path):
    ledger = _ledger(tmp_path, BudgetSpec("crusoe", 1.0, "total", CRUSOE))
    with pytest.raises(ValueError):
        ledger.spent("global:nope")
    with pytest.raises(ValueError):
        ledger.spent("bogus")


def test_concurrent_reservations_never_exceed_limit(tmp_path):
    path = tmp_path / "pitstop.db"
    Store(path)  # create the schema up front
    cfg = _cfg(BudgetSpec("crusoe", 5.0, "total", CRUSOE))
    exceeded = []
    granted = []
    errors = []
    start = threading.Barrier(8, timeout=10)
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            ledger = Ledger(Store(path), cfg)
            start.wait()
            for j in range(20):
                try:
                    res = _reserve(ledger, 0.10, run_id=f"t{i}")
                except BudgetExceeded:
                    with lock:
                        exceeded.append((i, j))
                    continue
                with lock:
                    granted.append(res.id)
                if j % 2 == 0:  # settle half at exactly the estimate, leave the rest reserved
                    ledger.settle(res, prompt_tokens=100_000, completion_tokens=0)
        except BaseException as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    conn = Store(path).connect()
    total = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE state IN ('reserved','settled')").fetchone()[0]
    assert total <= 5.00 + 1e-9
    assert len(granted) == 50  # the limit is used exactly, never overshot
    assert len(exceeded) == 160 - 50
    assert len(exceeded) >= 1
