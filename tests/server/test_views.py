"""The dashboard's read models (`pitstop.server.views`), called directly on seeded stores."""
from datetime import datetime

import pytest

from pitstop.alerts.model import build_alert

from tests.server.factory import add_drift, add_event, add_ledger, add_llm_call, add_run

NOW = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()  # local noon: "today" is safe
HOUR = 3600.0


def test_board_flags_and_sort(rt):
    from pitstop.server.views import board

    # the green one ran most recently, the red one least recently: the flag sorts first
    red_run = add_run(rt.store, "red-wf", started_at=NOW - 3 * HOUR, criticality="high")
    add_run(rt.store, "yellow-wf", started_at=NOW - 2 * HOUR)
    add_run(rt.store, "green-wf", started_at=NOW - HOUR, criticality="low")
    p0 = add_drift(rt.store, "red-wf", ts=NOW - 3 * HOUR, level="P0", drift_type="budget_exhausted", run_id=red_run)
    add_drift(rt.store, "red-wf", ts=NOW - 3 * HOUR, level="P2", drift_type="unproductive_loop", run_id=red_run)
    add_drift(rt.store, "yellow-wf", ts=NOW - 2 * HOUR, level="P2")
    add_drift(rt.store, "green-wf", ts=NOW - HOUR, level="P1", status="dismissed")

    b = board(rt, now=NOW)
    assert [t.name for t in b.workflows] == ["red-wf", "yellow-wf", "green-wf"]
    assert [(t.flag, t.flag_label) for t in b.workflows] == [
        ("red", "Halted"), ("yellow", "Degraded"), ("green", "Healthy")]
    red, yellow, green = b.workflows
    assert red.headline == build_alert(rt.store, p0).title
    assert red.headline_level == "P0"
    assert red.open.model_dump() == {"P0": 1, "P1": 0, "P2": 1, "P3": 0}
    assert red.criticality == "high"
    assert red.last_run.id == red_run and red.last_run.status == "ok"
    assert yellow.headline_level == "P2"
    assert (green.headline, green.headline_level) == (None, None)
    assert green.open.model_dump() == {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    assert all(t.adapters == ["wrap"] and t.fallback_active is False for t in b.workflows)
    assert b.now == NOW and b.budgets == []


def test_board_sorts_by_level_then_latest_activity(rt):
    from pitstop.server.views import board

    for name, started, level in (("p2-old", 5, "P2"), ("p1", 4, "P1"), ("p2-new", 1, "P2")):
        add_run(rt.store, name, started_at=NOW - started * HOUR)
        add_drift(rt.store, name, ts=NOW - started * HOUR, level=level)
    assert [t.name for t in board(rt, now=NOW).workflows] == ["p1", "p2-new", "p2-old"]


def test_board_halted_last_run_is_red_without_drift(rt):
    from pitstop.server.views import board

    add_run(rt.store, "halted-wf", started_at=NOW - 2 * HOUR, status="ok")
    halted = add_run(rt.store, "halted-wf", started_at=NOW - HOUR, status="halted")
    add_run(rt.store, "refused-wf", started_at=NOW - HOUR, status="refused", model=None)
    add_run(rt.store, "recovered-wf", started_at=NOW - 2 * HOUR, status="halted")
    add_run(rt.store, "recovered-wf", started_at=NOW - HOUR, status="ok")
    tiles = {t.name: t for t in board(rt, now=NOW).workflows}
    assert (tiles["halted-wf"].flag, tiles["halted-wf"].flag_label) == ("red", "Halted")
    assert tiles["halted-wf"].last_run.id == halted and tiles["halted-wf"].headline is None
    assert tiles["refused-wf"].flag == "red"
    assert tiles["recovered-wf"].flag == "green"


def test_tile_spend_against_workflow_budget(rt):
    from pitstop.server.views import board

    add_run(rt.store, "budgeted", started_at=NOW - 30 * HOUR, budget_usd=1.0)
    add_run(rt.store, "budgeted", started_at=NOW - HOUR, budget_usd=5.0)  # the latest run's budget counts
    add_run(rt.store, "unbudgeted", started_at=NOW - HOUR)
    add_ledger(rt.store, "budgeted", 0.25, ts=NOW - HOUR)
    add_ledger(rt.store, "budgeted", 0.5, ts=NOW - 2 * HOUR, state="reserved")  # a live reservation counts
    add_ledger(rt.store, "budgeted", 7.0, ts=NOW - 30 * HOUR)  # yesterday
    add_ledger(rt.store, "budgeted", 3.0, ts=NOW - HOUR, meter="judge")  # not agent spend
    add_ledger(rt.store, "budgeted", 2.0, ts=NOW - HOUR, state="released")
    add_ledger(rt.store, "unbudgeted", 0.1, ts=NOW - HOUR)
    tiles = {t.name: t for t in board(rt, now=NOW).workflows}
    assert tiles["budgeted"].spend_today_usd == 0.75
    assert tiles["budgeted"].budget_usd == 5.0
    assert tiles["budgeted"].runs_today == 1
    assert (tiles["unbudgeted"].spend_today_usd, tiles["unbudgeted"].budget_usd) == (0.1, None)


def test_feed_kinds_and_order(rt):
    from pitstop.server.views import feed

    t = NOW - 10 * HOUR
    r1 = add_run(rt.store, "wf", started_at=t)
    add_event(rt.store, r1, "halt", ts=t + 1, name="search", payload={"reason": "spend spike", "args": "{}"})
    add_event(rt.store, r1, "tool_call", ts=t + 2, name="search", payload={"args": "{}"})  # not a feed kind
    r2 = add_run(rt.store, "wf", started_at=t + 100, model="m2")  # model change from m1
    add_event(rt.store, r2, "fallback", ts=t + 101, name="tool_call_breakage",
              payload={"from": "m2", "to": "m1", "change": {"field": "model"}})
    refused = add_run(rt.store, "wf", started_at=t + 200, status="refused", model=None)
    p1 = add_drift(rt.store, "wf", ts=t + 300, level="P1", drift_type="tool_call_breakage", run_id=r2)
    add_drift(rt.store, "wf", ts=t + 400, level="P3", drift_type="quality_drop")  # P3: not in the feed
    dismissed = add_drift(rt.store, "wf", ts=t + 500, level="P0", status="dismissed")  # opened: still in the feed

    items = feed(rt)
    assert [(i.kind, i.id) for i in items] == [
        ("drift", f"drift:{dismissed}"), ("drift", f"drift:{p1}"), ("refused", f"refused:{refused}"),
        ("fallback", items[3].id), ("model_change", f"model_change:{r2}"), ("halt", items[5].id)]
    assert [i.ts for i in items] == sorted((i.ts for i in items), reverse=True)
    drift, _, ref, fb, change, halt = items
    assert (drift.level, drift.drift_id, drift.workflow) == ("P0", dismissed, "wf")
    assert (ref.run_id, ref.drift_id) == (refused, None)
    assert "m2" in fb.text and "m1" in fb.text and fb.run_id == r2
    assert "m1" in change.text and "m2" in change.text and change.run_id == r2
    assert "spend spike" in halt.text and halt.run_id == r1 and halt.id.startswith("halt:")
    assert len(feed(rt, limit=2)) == 2
    assert len(feed(rt, limit=1000)) == 6  # capped at 100, fewer exist


def test_feed_model_change_same_source_only(rt):
    from pitstop.server.views import feed

    t = NOW - 10 * HOUR
    # SDK and OTel runs of one workflow on different models, interleaved: no change
    for i in range(4):
        add_run(rt.store, "wf", started_at=t + i * 10, model="sdk-m", adapter="wrap")
        add_run(rt.store, "wf", started_at=t + i * 10 + 5, model="otel-m", adapter="otel", id=f"otel-{i:04x}")
    add_run(rt.store, "other", started_at=t + 50, model="x")  # another workflow: not compared with wf
    assert [i for i in feed(rt) if i.kind == "model_change"] == []

    changed = add_run(rt.store, "wf", started_at=t + 100, model="sdk-m2", adapter="wrap")
    add_run(rt.store, "wf", started_at=t + 110, model=None, status="refused")  # no model: skipped, no change
    same = add_run(rt.store, "wf", started_at=t + 120, model="sdk-m2", adapter="wrap")
    base = add_run(rt.store, "wf", started_at=t + 130, model="sdk-m2", base_url="https://b/v1")
    changes = [i for i in feed(rt) if i.kind == "model_change"]
    assert [i.run_id for i in changes] == [base, changed]
    assert same not in {i.run_id for i in changes}
    assert "sdk-m" in changes[1].text and "sdk-m2" in changes[1].text
    assert "https://a/v1" in changes[0].text and "https://b/v1" in changes[0].text


def test_workflow_detail_drift_cards_use_alert_text(rt):
    from pitstop.server.views import workflow_detail

    run = add_run(rt.store, "wf", started_at=NOW - HOUR, goal="summarize filings", sensitivity="high",
                  fix_mode="auto")
    p2 = add_drift(rt.store, "wf", ts=NOW - 2 * HOUR, level="P2", run_id=run)
    p0 = add_drift(rt.store, "wf", ts=NOW - 3 * HOUR, level="P0", drift_type="spend_spike", action="halt",
                   run_id=run, occurrences=3, last_seen=NOW - HOUR)
    add_drift(rt.store, "wf", ts=NOW - HOUR, level="P1", drift_type="refusal_spike", status="dismissed")
    for i in range(55):  # open drifts are capped at 50
        add_drift(rt.store, "wf", ts=NOW - 4 * HOUR, level="P3", drift_type=f"t{i}")
    d = workflow_detail(rt, "wf", now=NOW)
    assert (d.goal, d.sensitivity, d.fix_mode) == ("summarize filings", "high", "auto")
    assert d.tile.name == "wf" and d.tile.flag == "red"
    assert len(d.drifts) == 50
    assert [c.id for c in d.drifts[:2]] == [p0, p2]
    card = d.drifts[0]
    alert = build_alert(rt.store, p0)
    assert (card.title, card.cause, card.doing) == (alert.title, alert.cause, alert.action)
    assert (card.level, card.drift_type, card.action, card.occurrences) == ("P0", "spend_spike", "halt", 3)
    assert (card.first_seen, card.last_seen) == (NOW - 3 * HOUR, NOW - HOUR)


def test_workflow_detail_unknown_is_not_found(rt):
    from pitstop.server.views import NotFound, workflow_detail

    add_drift(rt.store, "ghost", ts=NOW)  # a drift alone does not make a workflow
    with pytest.raises(NotFound):
        workflow_detail(rt, "ghost")


def test_runs_page_counts(rt):
    from pitstop.server.views import runs_page

    run = add_run(rt.store, "wf", started_at=NOW, ended_at=NOW + 12.5)
    add_llm_call(rt.store, run, ts=NOW + 1, cost_usd=0.02)
    add_llm_call(rt.store, run, ts=NOW + 2, cost_usd=0.03)
    add_llm_call(rt.store, run, ts=NOW + 3, cost_usd=0.5, meter="judge")  # not the agent's
    add_event(rt.store, run, "tool_call", ts=NOW + 4, name="search", payload={"args": "{}"})
    for kind in ("block", "steer", "halt", "fallback", "gate"):
        add_event(rt.store, run, kind, ts=NOW + 5, name="search", payload={"reason": "x"})
    add_event(rt.store, run, "llm_call", ts=NOW + 6, name="m1", payload="{not json", cost_usd=9.0)
    page = runs_page(rt, "wf")
    assert page.next_before is None
    (row,) = page.runs
    assert (row.id, row.status, row.duration_s, row.model, row.base_url, row.adapter) == (
        run, "ok", 12.5, "m1", "https://a/v1", "wrap")
    assert row.cost_usd == pytest.approx(0.05)
    assert (row.llm_calls, row.tool_calls, row.verdicts) == (2, 1, 4)


def test_run_detail_summaries_never_include_raw_payload(rt):
    from pitstop.server.views import run_detail

    run = add_run(rt.store, "wf", started_at=NOW)
    big = "x" * 10_000
    add_llm_call(rt.store, run, ts=NOW + 1, model="m1", prompt=1200, completion=80, latency_s=1.25,
                 fallback_from="m0")
    add_event(rt.store, run, "tool_call", ts=NOW + 2, name="search", payload={"args": '{"q": "acme 10-K"}'})
    add_event(rt.store, run, "tool_result", ts=NOW + 3, name="search", payload={"result": big, "latency_s": 0.5})
    add_event(rt.store, run, "tool_result", ts=NOW + 4, name="search",
              payload={"error": "RuntimeError('boom')", "latency_s": 0.1})
    add_event(rt.store, run, "block", ts=NOW + 5, name="send_email",
              payload={"reason": "off task: emails a stranger", "args": big})
    add_event(rt.store, run, "gate", ts=NOW + 6, name="send_email",
              payload={"allow": False, "reason": "not in the goal", "latency_s": 0.2})
    add_event(rt.store, run, "agent", ts=NOW + 7, name="writer", payload={"parent": "planner", "source": "sdk"})
    add_event(rt.store, run, "mystery", ts=NOW + 8, name="?", payload={"secret_field": big})
    add_event(rt.store, run, "tool_result", ts=NOW + 9, name="search", payload="{not json " + big)

    d = run_detail(rt, run)
    assert (d.workflow, d.goal, d.truncated, d.run.id) == ("wf", "answer questions", False, run)
    assert all(len(e.summary) <= 300 and "\n" not in e.summary for e in d.events)
    assert all(big[:400] not in e.summary for e in d.events)
    by_kind = {}
    for e in d.events:
        by_kind.setdefault(e.kind, []).append(e)
    llm = by_kind["llm_call"][0]
    assert "m1" in llm.summary and "1200" in llm.summary and "80" in llm.summary and "m0" in llm.summary
    assert llm.latency_s == 1.25 and llm.cost_usd == 0.01
    assert "acme 10-K" in by_kind["tool_call"][0].summary
    assert by_kind["tool_result"][0].latency_s == 0.5
    assert "RuntimeError('boom')" in by_kind["tool_result"][1].summary
    assert "off task: emails a stranger" in by_kind["block"][0].summary
    assert "not in the goal" in by_kind["gate"][0].summary
    assert "planner" in by_kind["agent"][0].summary
    assert "secret_field" not in by_kind["mystery"][0].summary
    assert by_kind["tool_result"][2].summary == "" and by_kind["tool_result"][2].latency_s is None


def test_run_detail_caps_events(rt):
    from pitstop.server.views import MAX_EVENTS, run_detail

    run = add_run(rt.store, "wf", started_at=NOW)
    with rt.store.transaction() as c:
        c.executemany("INSERT INTO events(run_id, ts, agent, kind, name, payload) VALUES (?,?,?,?,?,?)",
                      [(run, NOW + i, "a", "tool_call", "t", '{"args": "{}"}') for i in range(MAX_EVENTS + 5)])
    d = run_detail(rt, run)
    assert MAX_EVENTS == 2000 and len(d.events) == 2000 and d.truncated is True
    assert d.events[0].ts == NOW  # the first 2,000
