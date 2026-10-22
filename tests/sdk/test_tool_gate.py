import json
import threading

import pytest

import pitstop
from pitstop.detect.drifts import list_drifts
from pitstop.detect.l1 import GATE_PROMPT_VERSION
from pitstop.sdk.context import agent, workflow
from pitstop.sdk.errors import Halted
from pitstop.sdk.runtime import get_runtime
from tests.sdk.fakes import FakeOpenAI, make_response

MODEL = "openai/gpt-oss-120b"
AGENT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
MSGS = [{"role": "user", "content": "hi"}]
NO_JUDGE = "no judge configured: set CRUSOE_API_KEY or [judge] in config.toml"


def _verdict(serves: bool, reason: str = "fine"):
    return make_response(content=json.dumps({"serves_goal": serves, "reason": reason}))


def _init(tmp_path, judge=None, config=""):
    (tmp_path / "config.toml").write_text(config)
    return pitstop.init(home=tmp_path, judge_client=judge)


def _db():
    return get_runtime().store.connect()


def _events(kind):
    rows = _db().execute("SELECT * FROM events WHERE kind=? ORDER BY id", (kind,)).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["payload"] = json.loads(r["payload"])
    return out


def _tools(calls):
    @pitstop.tool(side_effects=True)
    def send_email(to, body):
        calls.append(to)
        return "sent"

    @pitstop.tool
    def search(q):
        calls.append(q)
        return "found"

    return send_email, search


def test_gate_allows_and_tool_runs(tmp_path):
    judge = FakeOpenAI([_verdict(True, "the reply is the task")])
    _init(tmp_path, judge)
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="Answer the ticket", criticality="medium"):
        with agent("writer", goal="Reply to the customer"):
            assert send_email("a@b", "hi") == "sent"
    assert calls == ["a@b"]
    [g] = _events("gate")
    assert g["payload"] == {"allow": True, "reason": "the reply is the task", "latency_s": g["payload"]["latency_s"],
                            "prompt": GATE_PROMPT_VERSION}
    assert g["name"] == "send_email" and g["agent"] == "writer"
    user = judge.calls[0]["messages"][-1]["content"]
    assert "Answer the ticket" in user and "Reply to the customer" in user and "a@b" in user
    [row] = _db().execute("SELECT meter, state FROM ledger").fetchall()
    assert (row["meter"], row["state"]) == ("watch", "settled")
    assert len(_events("tool_call")) == 1 and _events("gate_error") == []


def test_gate_rejects_blocks_and_records_p0(tmp_path):
    _init(tmp_path, FakeOpenAI([_verdict(False, "emails an outsider")]))
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="Answer the ticket", criticality="medium") as ctx:
        out = send_email("x@evil", "data")
        assert ctx.halt_reason is None  # medium: block, no halt
    assert calls == []
    assert out["blocked_by"] == "pitstop" and set(out) == {"blocked_by", "reason"}
    assert out["reason"].startswith("out_of_scope_action: ") and "emails an outsider" in out["reason"]
    [d] = list_drifts(get_runtime().store)
    assert (d["drift_type"], d["level"], d["action"]) == ("out_of_scope_action", "P0", "block")
    assert d["evidence"][0]["detail"] == {"tool": "send_email", "gate": "emails an outsider",
                                         "prompt": GATE_PROMPT_VERSION}
    [g] = _events("gate")
    assert g["payload"]["allow"] is False and g["payload"]["reason"] == "emails an outsider"
    assert len(_events("block")) == 1 and _events("tool_call") == []


def test_gate_reject_on_high_sets_pending_halt(tmp_path):
    _init(tmp_path, FakeOpenAI([_verdict(False, "off task")]))
    agent_client = pitstop.wrap(FakeOpenAI([make_response("ok")]))
    calls = []
    send_email, _ = _tools(calls)
    with pytest.raises(Halted):
        with workflow("wf", goal="g", criticality="high") as ctx:
            assert send_email("a@b", "x")["blocked_by"] == "pitstop"
            assert ctx.halt_reason
            agent_client.chat.completions.create(model=MODEL, messages=MSGS)
    assert calls == []


def test_high_without_judge_blocks_with_fix_hint(tmp_path):
    _init(tmp_path)
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="high"):
        r = send_email("a@b", "hi")
    assert calls == []
    assert r["blocked_by"] == "pitstop" and "[judge]" in r["reason"]
    assert r["reason"] == (f"pre-action check unavailable ({NO_JUDGE}); blocked by on_gate_error=block. "
                           "Configure [judge] in config.toml or set on_gate_error=\"allow\".")
    [e] = _events("gate_error")
    assert e["payload"] == {"error": NO_JUDGE, "policy": "block"}
    assert _events("gate") == []


def test_medium_without_judge_allows_and_records(tmp_path):
    _init(tmp_path)
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="medium"):
        assert send_email("a@b", "hi") == "sent"
    assert calls == ["a@b"]
    [e] = _events("gate_error")
    assert e["payload"] == {"error": NO_JUDGE, "policy": "allow"}
    assert list_drifts(get_runtime().store) == []


@pytest.mark.parametrize("crit,override,expect_run", [("high", "allow", True), ("medium", "block", False)])
def test_on_gate_error_override(tmp_path, crit, override, expect_run):
    _init(tmp_path)
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality=crit, on_gate_error=override) as ctx:
        assert ctx.spec.effective_on_gate_error == override
        out = send_email("a@b", "hi")
    assert (calls == ["a@b"]) is expect_run
    assert (out == "sent") is expect_run
    assert _events("gate_error")[0]["payload"]["policy"] == override


def test_default_gate_error_policy_by_criticality():
    assert workflow("w", criticality="high").spec.effective_on_gate_error == "block"
    assert workflow("w", criticality="medium").spec.effective_on_gate_error == "allow"
    assert workflow("w", criticality="low").spec.effective_on_gate_error == "allow"
    assert workflow("w").spec.on_gate_error is None


def test_unparseable_and_exception_verdicts_are_errors(tmp_path):
    secret = "sk-SECRET https://judge.example/v1"
    _init(tmp_path, FakeOpenAI([make_response(content="I'm sorry, I can't help"), ConnectionError(secret)]))
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="high"):
        r1 = send_email("a@b", "1")
        r2 = send_email("c@d", "2")
    assert calls == []
    assert "(unparseable verdict)" in r1["reason"] and "(ConnectionError)" in r2["reason"]
    errs = [e["payload"] for e in _events("gate_error")]
    assert errs == [{"error": "unparseable verdict", "policy": "block"},
                    {"error": "ConnectionError", "policy": "block"}]
    dump = json.dumps([dict(r) for r in _db().execute("SELECT * FROM events").fetchall()])
    assert "SECRET" not in dump and "judge.example" not in dump
    assert list_drifts(get_runtime().store) == []  # an error is not a verdict: no drift


def test_judge_budget_exceeded_halts(tmp_path):
    _init(tmp_path, FakeOpenAI([_verdict(True)]), config="[budgets.all]\nlimit_usd = 0.0000001\n")
    calls = []
    send_email, _ = _tools(calls)
    with pytest.raises(Halted):
        with workflow("wf", goal="g", criticality="medium"):
            send_email("a@b", "hi")
    assert calls == []


def test_low_criticality_and_read_only_tools_not_gated(tmp_path):
    judge = FakeOpenAI([])
    _init(tmp_path, judge)
    calls = []
    send_email, search = _tools(calls)
    with workflow("low", goal="g", criticality="low"):
        assert send_email("a@b", "hi") == "sent"
    with workflow("med", goal="g", criticality="high"):
        assert search("q") == "found"
    assert judge.calls == [] and calls == ["a@b", "q"]
    assert _events("gate") == [] and _events("gate_error") == []


def test_l0_block_skips_gate(tmp_path):
    judge = FakeOpenAI([])
    _init(tmp_path, judge)
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="medium", tools=["search"]):
        out = send_email("a@b", "hi")
    assert out["blocked_by"] == "pitstop" and judge.calls == [] and calls == []
    assert _events("gate") == [] and _events("gate_error") == []


class _ThreadRecordingJudge(FakeOpenAI):
    def __init__(self, responses):
        super().__init__(responses)
        self.threads = []

    def _create(self, **kw):
        self.threads.append(threading.get_ident())
        return super()._create(**kw)


async def test_async_tool_gate_runs_off_loop(tmp_path):
    judge = _ThreadRecordingJudge([_verdict(True)])
    _init(tmp_path, judge)
    ran = []

    @pitstop.tool(side_effects=True)
    async def send_email(to):
        ran.append(threading.get_ident())
        return "sent"

    loop_thread = threading.get_ident()
    with workflow("wf", goal="g", criticality="medium") as ctx:
        with agent("writer"):
            assert await send_email("a@b") == "sent"
    assert ran == [loop_thread]
    assert len(judge.threads) == 1 and judge.threads[0] != loop_thread
    [g] = _events("gate")
    assert (g["run_id"], g["agent"]) == (ctx.run_id, "writer")  # the run context reached the thread
    [row] = _db().execute("SELECT meter, run_id, workflow FROM ledger").fetchall()
    assert (row["meter"], row["run_id"], row["workflow"]) == ("watch", ctx.run_id, "wf")


def test_judge_calls_do_not_run_l0_or_tag_run(tmp_path):
    judge = FakeOpenAI([make_response(content="I'm sorry, I can't help") for _ in range(3)])
    _init(tmp_path, judge)
    agent_client = pitstop.wrap(FakeOpenAI([make_response("ok")]))
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="medium", sensitivity="high") as ctx:  # high: 1 refusal spikes
        for i in range(3):
            send_email(f"a{i}@b", "hi")
        agent_client.chat.completions.create(model=AGENT_MODEL, messages=MSGS)
    assert ctx.refusals == 0 and len(ctx.llm_history) == 1
    assert [d["drift_type"] for d in list_drifts(get_runtime().store)] == []
    run = dict(_db().execute("SELECT model FROM runs WHERE id=?", (ctx.run_id,)).fetchone())
    assert run["model"] == AGENT_MODEL


def test_judge_model_override_and_cached_gate(tmp_path):
    judge = FakeOpenAI([_verdict(True), _verdict(True)])
    (tmp_path / "config.toml").write_text("")
    rt = pitstop.init(home=tmp_path, judge_client=judge, judge_model="acme/judge-1")
    assert rt.gate() is rt.gate()
    calls = []
    send_email, _ = _tools(calls)
    with workflow("wf", goal="g", criticality="medium"):
        send_email("a@b", "1")
        send_email("c@d", "2")
    assert [c["model"] for c in judge.calls] == ["acme/judge-1", "acme/judge-1"]


def test_gate_from_env_key_uses_config(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[judge]\nmodel = "acme/judge-2"\nbase_url = "https://judge.example/v1"\n'
                                          'api_key_env = "MY_JUDGE_KEY"\ntimeout_s = 3.5\n')
    rt = pitstop.init(home=tmp_path)
    assert rt.gate() is None  # no key in the environment
    from pitstop.sdk.runtime import reset_runtime

    reset_runtime()
    monkeypatch.setenv("MY_JUDGE_KEY", "sk-test-not-real")
    rt = pitstop.init(home=tmp_path)
    gate = rt.gate()
    assert gate is not None and gate.model == "acme/judge-2"
    assert str(gate.client.base_url).rstrip("/") == "https://judge.example/v1"
    assert gate.client.timeout == 3.5 and gate.client.max_retries == 0
