import json
import sys

import pytest

import pitstop
from pitstop.detect.changes import Change
from pitstop.detect.drifts import list_drifts
from pitstop.sdk import fallback
from pitstop.sdk.context import workflow
from pitstop.sdk.runtime import get_runtime
from pitstop.sdk.wrap import meter
from tests.sdk.fakes import FakeOpenAI, make_response, make_tool_call

MODEL_A = "model-a"
MODEL_B = "model-b"
FALLBACK = "pinned-model"
MSGS = [{"role": "user", "content": "hello"}]


def _init(tmp_path):
    (tmp_path / "config.toml").write_text("")
    return pitstop.init(home=tmp_path)


def _ok():
    return make_response("fine")


def _bad():
    return make_response(tool_calls=[make_tool_call("search", '{"q": ')])


def _events(kind):
    rows = get_runtime().store.connect().execute(
        "SELECT run_id, name, payload FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    return [(r["run_id"], r["name"], json.loads(r["payload"])) for r in rows]


def _breakage_after(tmp_path, first_model, second_model, **wf):
    """Run 1 on `first_model` (one good call), then run 2 on `second_model` with 5 malformed tool calls
    (P1 tool_call_breakage on a medium workflow). Returns the fake client and run 2's id."""
    _init(tmp_path)
    fake = FakeOpenAI([_ok()] + [_bad() for _ in range(5)] + [_ok() for _ in range(5)])
    w = pitstop.wrap(fake)
    with workflow("wf", **wf):
        w.chat.completions.create(model=first_model, messages=MSGS)
    with workflow("wf", **wf) as ctx:
        for _ in range(5):
            w.chat.completions.create(model=second_model, messages=MSGS)
    return w, fake, ctx.run_id


def _activate(model=FALLBACK, workflow_name="wf"):
    return fallback.activate(get_runtime().store, workflow_name, model=model, drift_id=None,
                             change=Change("model", MODEL_A, MODEL_B, "r-prev", 1.0))


def test_breakage_after_model_change_activates_fallback(tmp_path):
    w, fake, run2 = _breakage_after(tmp_path, MODEL_A, MODEL_B, fallback_model=FALLBACK)
    assert [c["model"] for c in fake.calls] == [MODEL_A] + [MODEL_B] * 5
    [drift] = list_drifts(get_runtime().store)
    assert (drift["drift_type"], drift["level"], drift["action"]) == ("tool_call_breakage", "P1", "fallback")
    [(run_id, name, payload)] = _events("fallback")
    assert (run_id, name) == (run2, "tool_call_breakage")
    assert payload["from"] == MODEL_B and payload["to"] == FALLBACK
    assert payload["change"]["field"] == "model"
    assert (payload["change"]["before"], payload["change"]["after"]) == (MODEL_A, MODEL_B)

    state = fallback.active(get_runtime().store, "wf")
    assert state["model"] == FALLBACK and state["drift_id"] == drift["id"] and state["since"] > 0
    assert state["change"]["prev_run_id"] != run2

    # every call of the next run goes to the pinned model (the same run: test_next_call_in_the_same_run_is_swapped)
    with workflow("wf", fallback_model=FALLBACK) as ctx3:
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert [c["model"] for c in fake.calls[-2:]] == [FALLBACK, FALLBACK]
    calls = [e for e in _events("llm_call") if e[0] == ctx3.run_id]
    assert [(name, p.get("fallback_from")) for _, name, p in calls] == [(FALLBACK, MODEL_B)] * 2


def test_next_call_in_the_same_run_is_swapped(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([_ok()] + [_bad() for _ in range(5)] + [_ok()])
    w = pitstop.wrap(fake)
    with workflow("wf", fallback_model=FALLBACK):
        w.chat.completions.create(model=MODEL_A, messages=MSGS)
    with workflow("wf", fallback_model=FALLBACK):
        for _ in range(6):
            w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert [c["model"] for c in fake.calls] == [MODEL_A] + [MODEL_B] * 5 + [FALLBACK]


def test_refusal_spike_after_base_url_change_activates_fallback(tmp_path):
    _init(tmp_path)
    fake_a = FakeOpenAI([_ok()], base_url="https://a.example/v1/")
    fake_b = FakeOpenAI([make_response("I'm sorry, I can't do that"), _ok()], base_url="https://b.example/v1/")
    with workflow("wf", sensitivity="high", fallback_model=FALLBACK):  # high: one refusal is a spike
        pitstop.wrap(fake_a).chat.completions.create(model=MODEL_B, messages=MSGS)
    w = pitstop.wrap(fake_b)
    with workflow("wf", sensitivity="high", fallback_model=FALLBACK):
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert [c["model"] for c in fake_b.calls] == [MODEL_B, FALLBACK]
    [drift] = list_drifts(get_runtime().store)
    assert (drift["drift_type"], drift["level"], drift["action"]) == ("refusal_spike", "P1", "fallback")
    [(_, _, payload)] = _events("fallback")
    assert payload["change"]["field"] == "base_url"
    assert (payload["change"]["before"], payload["change"]["after"]) == ("https://a.example/v1",
                                                                         "https://b.example/v1")


def test_no_change_no_fallback(tmp_path):
    _breakage_after(tmp_path, MODEL_B, MODEL_B, fallback_model=FALLBACK)
    [drift] = list_drifts(get_runtime().store)
    assert (drift["level"], drift["action"]) == ("P1", "record")
    assert fallback.active(get_runtime().store, "wf") is None
    assert _events("fallback") == []


def test_no_fallback_model_declared_no_fallback(tmp_path):
    _breakage_after(tmp_path, MODEL_A, MODEL_B)
    [drift] = list_drifts(get_runtime().store)
    assert (drift["level"], drift["action"]) == ("P1", "record")
    assert fallback.list_active(get_runtime().store) == []


def test_low_criticality_breakage_is_p2_no_fallback(tmp_path):
    _breakage_after(tmp_path, MODEL_A, MODEL_B, fallback_model=FALLBACK, criticality="low")
    [drift] = list_drifts(get_runtime().store)
    assert (drift["level"], drift["action"]) == ("P2", "record")
    assert fallback.active(get_runtime().store, "wf") is None


def test_breakage_on_the_fallback_model_itself_does_not_engage(tmp_path):
    _breakage_after(tmp_path, MODEL_A, FALLBACK, fallback_model=FALLBACK)
    [drift] = list_drifts(get_runtime().store)
    assert drift["action"] == "record"
    assert fallback.active(get_runtime().store, "wf") is None


def test_watch_meter_calls_not_swapped(tmp_path):
    _init(tmp_path)
    assert _activate() is True
    fake = FakeOpenAI([_ok(), _ok(), _ok()])
    w = pitstop.wrap(fake)
    with workflow("wf", fallback_model=FALLBACK):
        with meter("watch"):
            w.chat.completions.create(model=MODEL_B, messages=MSGS)
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    w.chat.completions.create(model=MODEL_B, messages=MSGS)  # outside a workflow: never swapped
    assert [c["model"] for c in fake.calls] == [MODEL_B, FALLBACK, MODEL_B]
    payloads = [p for _, _, p in _events("llm_call")]
    assert "fallback_from" not in payloads[0] and payloads[1]["fallback_from"] == MODEL_B
    assert "fallback_from" not in payloads[2]


def test_other_workflow_and_workflow_without_fallback_model_not_swapped(tmp_path):
    _init(tmp_path)
    _activate()
    fake = FakeOpenAI([_ok(), _ok()])
    w = pitstop.wrap(fake)
    with workflow("other", fallback_model=FALLBACK):
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    with workflow("wf"):  # active state, but this definition pins no fallback model
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert [c["model"] for c in fake.calls] == [MODEL_B, MODEL_B]


def test_clear_restores_requested_model(tmp_path):
    _init(tmp_path)
    _activate()
    fake = FakeOpenAI([_ok(), _ok()])
    w = pitstop.wrap(fake)
    with workflow("wf", fallback_model=FALLBACK):
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
        assert fallback.clear(get_runtime().store, "wf") is True
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert [c["model"] for c in fake.calls] == [FALLBACK, MODEL_B]
    assert fallback.clear(get_runtime().store, "wf") is False


def test_store_read_failure_keeps_requested_model(tmp_path, monkeypatch):
    _init(tmp_path)
    _activate()

    def boom(store, workflow):
        raise RuntimeError("db is locked")

    monkeypatch.setattr(fallback, "active", boom)
    fake = FakeOpenAI([_ok()])
    w = pitstop.wrap(fake)
    with workflow("wf", fallback_model=FALLBACK):
        w.chat.completions.create(model=MODEL_B, messages=MSGS)
    assert fake.calls[0]["model"] == MODEL_B


def test_activate_is_once_until_cleared(tmp_path):
    _init(tmp_path)
    store = get_runtime().store
    assert _activate() is True
    assert _activate(model="another") is False
    assert fallback.active(store, "wf")["model"] == FALLBACK
    _activate(workflow_name="wf2")
    assert [(r["workflow"], r["model"]) for r in fallback.list_active(store)] == [("wf", FALLBACK),
                                                                                 ("wf2", FALLBACK)]
    assert fallback.clear(store, "wf") is True
    assert fallback.active(store, "wf") is None
    assert [r["workflow"] for r in fallback.list_active(store)] == ["wf2"]


@pytest.mark.parametrize("bad", [123, ""])
def test_fallback_model_is_validated(bad):
    with pytest.raises(ValueError, match="fallback_model"):
        workflow("wf", fallback_model=bad)


def _main(monkeypatch, *argv):
    from pitstop import cli

    monkeypatch.setattr(sys, "argv", ["pitstop", *argv])
    cli.main()


def test_fallback_cli_status_and_clear(tmp_path, capsys, monkeypatch):
    _main(monkeypatch, "fallback", "status")
    assert "No active fallback" in capsys.readouterr().out
    _activate()
    _main(monkeypatch, "fallback", "status")
    out = capsys.readouterr().out
    assert "wf" in out and FALLBACK in out and MODEL_A in out and MODEL_B in out
    _main(monkeypatch, "fallback", "clear", "wf")
    assert "cleared" in capsys.readouterr().out
    _main(monkeypatch, "fallback", "clear", "wf")  # exit 0 either way
    assert "no active fallback" in capsys.readouterr().out
    assert fallback.active(get_runtime().store, "wf") is None


def _real_clients():
    from tests.sdk.http_fakes import (anthropic_client, chat_completion, json_response, message_body,
                                      message_events, openai_client, response_body, sse_response)

    def chat(seen):
        c = openai_client(lambda r: seen.append(json.loads(r.content)) or json_response(chat_completion()))
        c = pitstop.wrap(c)
        return lambda: c.chat.completions.create(model=MODEL_B, messages=MSGS)

    def responses(seen):
        c = openai_client(lambda r: seen.append(json.loads(r.content)) or json_response(response_body()))
        c = pitstop.wrap(c)
        return lambda: c.responses.create(model=MODEL_B, input="hi")

    def messages(seen):
        c = anthropic_client(lambda r: seen.append(json.loads(r.content)) or json_response(message_body()))
        c = pitstop.wrap(c)
        return lambda: c.messages.create(model=MODEL_B, max_tokens=50, messages=MSGS)

    def message_stream(seen):
        c = anthropic_client(lambda r: seen.append(json.loads(r.content))
                             or sse_response(message_events(), named=True))
        c = pitstop.wrap(c)

        def go():
            with c.messages.stream(model=MODEL_B, max_tokens=50, messages=MSGS) as s:
                s.get_final_message()
        return go

    return {"chat": chat, "responses": responses, "messages": messages, "messages.stream": message_stream}


@pytest.mark.parametrize("api", ["chat", "responses", "messages", "messages.stream"])
def test_swap_applies_to_every_shape(tmp_path, api):
    _init(tmp_path)
    _activate()
    seen: list[dict] = []
    call = _real_clients()[api](seen)
    with workflow("wf", fallback_model=FALLBACK):
        call()
    assert [b["model"] for b in seen] == [FALLBACK]
    [(model,)] = get_runtime().store.connect().execute("SELECT model FROM ledger").fetchall()
    assert model == FALLBACK  # reserved and settled at the pinned model's price
    [(_, name, payload)] = _events("llm_call")
    assert (name, payload["fallback_from"]) == (FALLBACK, MODEL_B)


# -- the change window and pinned runs (recent_change with ignore_model) ---------------------------------------


class _Script:
    """A wrapped fake whose responses are queued per run."""

    def __init__(self):
        self.fake = FakeOpenAI([])
        self.w = pitstop.wrap(self.fake)

    def run(self, model, *, ok=0, bad=0, **wf):
        self.fake._responses += [_ok() for _ in range(ok)] + [_bad() for _ in range(bad)]
        with workflow("wf", fallback_model=FALLBACK, **wf) as ctx:
            for _ in range(ok + bad):
                self.w.chat.completions.create(model=model, messages=MSGS)
        return ctx.run_id

    def sent(self, n):
        return [c["model"] for c in self.fake.calls[-n:]]


def _drift_actions():
    return [d["action"] for d in list_drifts(get_runtime().store)]


def test_later_run_on_the_new_model_still_engages(tmp_path):
    _init(tmp_path)
    s = _Script()
    s.run(MODEL_A, ok=1)
    s.run(MODEL_B, ok=1)  # the first run on the new model did not break
    s.run(MODEL_B, bad=5)
    assert _drift_actions() == ["fallback"]
    assert fallback.active(get_runtime().store, "wf")["change"]["before"] == MODEL_A
    s.run(MODEL_B, ok=1)
    assert s.sent(1) == [FALLBACK]


def test_change_older_than_a_day_does_not_engage(tmp_path):
    _init(tmp_path)
    s = _Script()
    s.run(MODEL_A, ok=1)
    s.run(MODEL_B, ok=1)
    with get_runtime().store.transaction() as c:  # both runs (and so the switch to MODEL_B) were 2 days ago
        c.execute("UPDATE runs SET started_at = started_at - 2 * 86400")
    s.run(MODEL_B, bad=5)
    assert _drift_actions() == ["record"]
    assert fallback.active(get_runtime().store, "wf") is None


def test_after_clear_pinned_runs_are_not_a_change(tmp_path):
    _init(tmp_path)
    s = _Script()
    s.run(MODEL_B, ok=1)
    _activate()
    s.run(MODEL_B, ok=1)  # sent to the pinned model: this run's model is FALLBACK
    assert s.sent(1) == [FALLBACK]
    assert fallback.clear(get_runtime().store, "wf")
    s.run(MODEL_B, bad=5)  # back on MODEL_B, as before the fallback: FALLBACK -> MODEL_B is no change
    assert s.sent(5) == [MODEL_B] * 5
    assert _drift_actions() == ["record"]
    assert fallback.active(get_runtime().store, "wf") is None


def test_change_made_while_the_fallback_was_active_is_seen_after_clear(tmp_path):
    _init(tmp_path)
    s = _Script()
    s.run(MODEL_B, ok=1)
    _activate()
    s.run("model-c", ok=1)  # the code moved to model-c while every call still went to the pinned model
    assert s.sent(1) == [FALLBACK]
    assert fallback.clear(get_runtime().store, "wf")
    s.run("model-c", bad=5)
    assert _drift_actions() == ["fallback"]
    change = fallback.active(get_runtime().store, "wf")["change"]
    assert (change["field"], change["before"], change["after"]) == ("model", MODEL_B, "model-c")


def test_unreadable_state_is_listed_and_replaced(tmp_path, capsys, monkeypatch):
    _init(tmp_path)
    store = get_runtime().store
    with store.transaction() as c:
        c.execute("INSERT INTO meta(key, value) VALUES ('fallback:wf', 'not json')")
    assert fallback.active(store, "wf") is None  # nothing is swapped on an unreadable state
    [row] = fallback.list_active(store)
    assert row["workflow"] == "wf" and row["invalid"] is True and row["model"] is None
    _main(monkeypatch, "fallback", "status")
    assert "unreadable" in capsys.readouterr().out
    assert _activate() is True  # replaced, not blocked forever
    assert fallback.active(store, "wf")["model"] == FALLBACK
    assert "invalid" not in fallback.list_active(store)[0]
