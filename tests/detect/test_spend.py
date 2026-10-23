"""The spend-rate P0: a run whose agent spend passes N x its workflow's baseline halts at its next call."""
import time

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.detect.spend import (BASELINE_RUNS, MIN_BASELINE_RUNS, MIN_BASELINE_USD, SPEND_MULTIPLIER,
                                  baseline_run_cost, check_spend)
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime
from pitstop.sdk.wrap import meter
from tests.sdk.fakes import FakeOpenAI, make_chunk, make_response, make_usage_chunk

MODEL = "acme/flat"  # $1 per 1M input tokens, free output: N cents = N * 10_000 prompt tokens
MSGS = [{"role": "user", "content": "hello"}]


def _init(tmp_path):
    (tmp_path / "config.toml").write_text(f'[prices."{MODEL}"]\ninput = 1.0\noutput = 0.0\n')
    return pitstop.init(home=tmp_path)


def _resp(usd: float):
    return make_response("ok", prompt_tokens=round(usd * 1_000_000), completion_tokens=0)


def _history(n: int, usd: float = 0.02, name: str = "wf") -> None:
    """`n` finished ok runs of `name`, each settling one agent call of `usd`."""
    w = pitstop.wrap(FakeOpenAI([_resp(usd) for _ in range(n)]))
    for _ in range(n):
        with workflow(name):
            w.chat.completions.create(model=MODEL, messages=MSGS)


def _spikes():
    return [d for d in list_drifts(get_runtime().store) if d["drift_type"] == "spend_spike"]


# -- check_spend / baseline_run_cost ----------------------------------------------------------------------


def test_constants():
    assert SPEND_MULTIPLIER == {"low": 10.0, "normal": 5.0, "high": 3.0}
    assert (BASELINE_RUNS, MIN_BASELINE_RUNS, MIN_BASELINE_USD) == (20, 10, 0.01)


def test_check_spend_thresholds():
    assert check_spend(0.11, None, "normal") is None  # no baseline yet: never a spike
    assert check_spend(0.10, 0.02, "normal") is None  # exactly 5x is not above it
    f = check_spend(0.11, 0.02, "normal")
    assert (f.drift_type, f.kind, f.side_effects) == ("spend_spike", "degraded", False)
    assert f.evidence == {"run_usd": 0.11, "baseline_usd": 0.02, "multiplier": 5.0}
    assert check_spend(0.07, 0.02, "high") is not None and check_spend(0.07, 0.02, "low") is None
    # the baseline is floored at $0.01: a near-free workflow does not spike on its first real cent
    assert check_spend(0.04, 0.0, "normal") is None
    assert check_spend(0.06, 0.0, "normal") is not None


def test_baseline_is_median_of_last_ok_runs(tmp_path):
    rt = _init(tmp_path)
    _history(5, 0.01)
    _history(BASELINE_RUNS, 0.03)  # the last 20 ok runs: only these count
    assert baseline_run_cost(rt.store, "wf", before=time.time()) == pytest.approx(0.03)
    assert baseline_run_cost(rt.store, "other", before=time.time()) is None


def test_baseline_ignores_runs_not_ok_and_judge_spend(tmp_path):
    rt = _init(tmp_path)
    _history(MIN_BASELINE_RUNS - 1)
    with pytest.raises(ValueError):  # an error run never joins the baseline
        with workflow("wf"):
            raise ValueError("x")
    assert baseline_run_cost(rt.store, "wf", before=time.time()) is None
    w = pitstop.wrap(FakeOpenAI([_resp(0.02), _resp(5.0)]))
    with workflow("wf"):
        w.chat.completions.create(model=MODEL, messages=MSGS)
        with meter("judge"):  # only agent-meter spend counts
            w.chat.completions.create(model=MODEL, messages=MSGS)
    assert baseline_run_cost(rt.store, "wf", before=time.time()) == pytest.approx(0.02)
    assert baseline_run_cost(rt.store, "wf", before=0.0) is None  # only runs started before `before`


def test_baseline_ignores_otel_runs(tmp_path):
    rt = _init(tmp_path)
    _history(MIN_BASELINE_RUNS, 0.02)
    now = time.time()
    with rt.store.transaction() as c:  # observed OTel runs are `ok` with no ledger rows: never a $0 baseline
        for i in range(15):
            c.execute("INSERT INTO runs(id, workflow, started_at, ended_at, status, adapter) VALUES (?,?,?,?,?,?)",
                      (f"otel-{i:032x}", "wf", now + i, now + i, "ok", "otel"))
    assert baseline_run_cost(rt.store, "wf", before=now + 100) == pytest.approx(0.02)


# -- the inline P0 ----------------------------------------------------------------------------------------


def test_spend_spike_halts_next_call(tmp_path):
    _init(tmp_path)
    _history(10)
    w = pitstop.wrap(FakeOpenAI([_resp(0.11), _resp(0.01)]))
    with pytest.raises(Halted) as ei:
        with workflow("wf") as ctx:  # medium criticality, normal sensitivity: 5x
            w.chat.completions.create(model=MODEL, messages=MSGS)  # the call that settled is never raised
            assert ctx.spend_usd == pytest.approx(0.11) and ctx.spend_spiked
            w.chat.completions.create(model=MODEL, messages=MSGS)
    [d] = _spikes()
    assert (d["level"], d["action"], d["workflow"], d["run_id"]) == ("P0", "halt", "wf", ctx.run_id)
    detail = d["evidence"][0]["detail"]
    assert detail["run_usd"] == pytest.approx(0.11) and detail["baseline_usd"] == pytest.approx(0.02)
    assert detail["multiplier"] == 5.0
    assert ei.value.drift_id == d["id"]
    row = get_runtime().store.connect().execute("SELECT status FROM runs WHERE id=?", (ctx.run_id,)).fetchone()
    assert row["status"] == "halted"


def test_no_spike_below_min_baseline_runs(tmp_path):
    _init(tmp_path)
    _history(MIN_BASELINE_RUNS - 1)
    w = pitstop.wrap(FakeOpenAI([_resp(0.11), _resp(0.01)]))
    with workflow("wf") as ctx:
        w.chat.completions.create(model=MODEL, messages=MSGS)
        w.chat.completions.create(model=MODEL, messages=MSGS)
    assert ctx.spend_baseline is None and not ctx.spend_spiked
    assert _spikes() == []


def test_spend_below_threshold_accumulates_without_spike(tmp_path):
    _init(tmp_path)
    _history(10)
    w = pitstop.wrap(FakeOpenAI([_resp(0.05), _resp(0.05)]))
    with workflow("wf") as ctx:
        w.chat.completions.create(model=MODEL, messages=MSGS)
        w.chat.completions.create(model=MODEL, messages=MSGS)  # $0.10 in all: exactly 5x, not above
    assert ctx.spend_usd == pytest.approx(0.10) and ctx.spend_baseline == pytest.approx(0.02)
    assert _spikes() == []


def test_spike_applied_once_per_run(tmp_path):
    _init(tmp_path)
    _history(10)

    def stream(usd):
        return [make_chunk("ok"), make_chunk(finish_reason="stop"), make_usage_chunk(round(usd * 1_000_000), 0)]

    w = pitstop.wrap(FakeOpenAI([stream(0.11), stream(0.11)]))
    with workflow("wf") as ctx:
        # both calls pass the halt check before either settles; each settle is over the threshold
        s1 = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
        s2 = w.chat.completions.create(model=MODEL, messages=MSGS, stream=True)
        list(s1)
        list(s2)
        assert ctx.spend_usd == pytest.approx(0.22)
    [d] = _spikes()
    assert d["occurrences"] == 1
    halts = get_runtime().store.connect().execute(
        "SELECT COUNT(*) FROM events WHERE run_id=? AND kind='halt'", (ctx.run_id,)).fetchone()[0]
    assert halts == 1


def test_spike_on_low_criticality_still_halts(tmp_path):
    _init(tmp_path)
    w = pitstop.wrap(FakeOpenAI([_resp(0.02) for _ in range(10)]))
    for _ in range(10):
        with workflow("wf", criticality="low"):
            w.chat.completions.create(model=MODEL, messages=MSGS)
    w = pitstop.wrap(FakeOpenAI([_resp(0.11), _resp(0.01)]))
    with pytest.raises(Halted):
        with workflow("wf", criticality="low"):
            w.chat.completions.create(model=MODEL, messages=MSGS)
            w.chat.completions.create(model=MODEL, messages=MSGS)
    [d] = _spikes()
    assert (d["level"], d["action"]) == ("P0", "halt")


def test_spend_check_failure_never_breaks_the_call(tmp_path, monkeypatch, caplog):
    import pitstop.detect.spend as spend

    _init(tmp_path)
    _history(10)

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(spend, "baseline_run_cost", boom)
    w = pitstop.wrap(FakeOpenAI([_resp(0.11), _resp(0.01)]))
    with workflow("wf"):
        assert w.chat.completions.create(model=MODEL, messages=MSGS) is not None
        assert w.chat.completions.create(model=MODEL, messages=MSGS) is not None
    assert _spikes() == []
    assert "internal error ignored" in caplog.text
