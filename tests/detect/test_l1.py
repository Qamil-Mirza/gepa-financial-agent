import json

import pytest

import pitstop
from pitstop.detect.l0 import ToolCallRecord
from pitstop.detect.l1 import GATE_PROMPT_VERSION, Gate, GateVerdict
from pitstop.sdk.context import workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import FakeOpenAI, make_response

JUDGE = "openai/gpt-oss-120b"


def _init(tmp_path, text=""):
    (tmp_path / "config.toml").write_text(text)
    return pitstop.init(home=tmp_path)


def _verdict(serves: bool, reason: str = "fine") -> object:
    return make_response(content=json.dumps({"serves_goal": serves, "reason": reason}))


def _check(gate, **kw):
    args = dict(workflow_goal="Answer the ticket", agent_goal="Draft a reply", tool="send_email",
                args_text='{"to":"a@b"}', recent=[])
    args.update(kw)
    return gate.check(**args)


def test_allow_verdict_and_request_shape(tmp_path):
    _init(tmp_path)
    fake = FakeOpenAI([_verdict(True, "sends the drafted reply")])
    gate = Gate(pitstop.wrap(fake), JUDGE)
    recent = [ToolCallRecord(f"t{i}", "x" * 1000, False) for i in range(7)]
    v = _check(gate, args_text="A" * 5000, recent=recent)
    assert isinstance(v, GateVerdict)
    assert (v.allow, v.reason, v.error) == (True, "sends the drafted reply", None)
    assert v.latency_s >= 0
    [kw] = fake.calls
    assert kw["model"] == JUDGE and kw["temperature"] == 0 and kw["max_tokens"] == 2048
    text = "\n".join(m["content"] for m in kw["messages"])
    assert "serves_goal" in text  # the frozen prompt asks for the JSON verdict
    user = kw["messages"][-1]["content"]
    assert "Answer the ticket" in user and "Draft a reply" in user and "send_email" in user
    assert "A" * 2000 in user and "A" * 2001 not in user  # proposed args truncated to 2000 chars
    assert "t2" in user and "t6" in user and "t1" not in user  # only the last 5 calls
    assert "x" * 300 in user and "x" * 301 not in user  # each recent call truncated to 300 chars
    assert GATE_PROMPT_VERSION == "l1_gate_v1"


def test_reject_verdict(tmp_path):
    _init(tmp_path)
    gate = Gate(pitstop.wrap(FakeOpenAI([_verdict(False, "emails a stranger")])), JUDGE)
    v = _check(gate)
    assert (v.allow, v.reason, v.error) == (False, "emails a stranger", None)


def test_fenced_and_reasoning_wrapped_verdicts_parse(tmp_path):
    _init(tmp_path)
    body = '<think>hmm</think>\n```json\n{"serves_goal": true, "reason": "ok"}\n```'
    gate = Gate(pitstop.wrap(FakeOpenAI([make_response(content=body)])), JUDGE)
    assert _check(gate).allow is True


@pytest.mark.parametrize("content", [
    "I'm sorry, I can't help with that",
    '{"reason": "no verdict key"}',
    '{"serves_goal": "yes", "reason": "string, not bool"}',
    None,
])
def test_unparseable_verdicts_are_errors(tmp_path, content):
    _init(tmp_path)
    gate = Gate(pitstop.wrap(FakeOpenAI([make_response(content=content)])), JUDGE)
    v = _check(gate)
    assert (v.allow, v.reason, v.error) == (False, "", "unparseable verdict")


def test_exception_verdict_names_only_the_type(tmp_path):
    _init(tmp_path)
    secret = "sk-SECRET https://judge.example/v1 Answer the ticket"
    gate = Gate(pitstop.wrap(FakeOpenAI([TimeoutError(secret)])), JUDGE)
    v = _check(gate)
    assert (v.allow, v.reason, v.error) == (False, "", "TimeoutError")


def test_halted_and_budget_exceeded_propagate(tmp_path):
    _init(tmp_path, '[budgets.all]\nlimit_usd = 0.0000001\n')
    gate = Gate(pitstop.wrap(FakeOpenAI([_verdict(True)])), JUDGE)
    with pytest.raises(pitstop.BudgetExceeded):
        _check(gate)
    gate = Gate(pitstop.wrap(FakeOpenAI([Halted("stop")])), JUDGE)
    with pytest.raises(Halted):
        _check(gate)


def test_judge_call_runs_on_watch_meter(tmp_path):
    _init(tmp_path)
    gate = Gate(pitstop.wrap(FakeOpenAI([_verdict(True)])), JUDGE)
    with workflow("wf"):
        _check(gate)
    [row] = get_runtime().store.connect().execute("SELECT meter, state FROM ledger").fetchall()
    assert (row["meter"], row["state"]) == ("watch", "settled")
