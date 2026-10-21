import json

import pytest

from pitstop.detect.l0 import (
    THRESHOLDS,
    Finding,
    LLMCallRecord,
    ToolCallRecord,
    canonical_args,
    check_llm_call,
    check_tool_call,
)
from pitstop.sdk.context import WorkflowSpec


def tc(name="search", args='{"q":"apples"}', side=False):
    return ToolCallRecord(name, args, side)


def spec(**kw):
    return WorkflowSpec("wf", **kw)


def test_thresholds_exact():
    assert THRESHOLDS["normal"] == dict(loop_window=6, loop_repeats=3, near_dup_jaccard=0.90,
                                        parse_fail_rate=0.10, refusals_per_run=2, min_calls=5)
    assert THRESHOLDS["low"]["loop_repeats"] == 4 and THRESHOLDS["high"]["loop_window"] == 8


@pytest.mark.parametrize("sens,calls_to_flag", [("normal", 3), ("low", 4), ("high", 2)])
def test_loop_detected_at_threshold_per_sensitivity(sens, calls_to_flag):
    s = spec(sensitivity=sens)
    hist: list[ToolCallRecord] = []
    for n in range(1, 7):
        found = [f.drift_type for f in check_tool_call(hist, tc(), s)]
        assert ("unproductive_loop" in found) == (n >= calls_to_flag), n
        hist.append(tc())


def test_loop_ignores_other_tools_and_old_calls():
    s = spec()
    hist = [tc("a")] * 2 + [tc("b")] * 6
    assert check_tool_call(hist, tc("a"), s) == []  # the two earlier "a" fell out of the window


def test_near_duplicate_args_count_as_loop():
    a = '{"q":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"}'
    b = '{"q":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"}'
    hist = [tc(args=a), tc(args=b)]
    out = check_tool_call(hist, tc(args=a), spec())
    assert [f.drift_type for f in out] == ["unproductive_loop"]
    # clearly different args do not
    assert check_tool_call([tc(args='{"q":"x1"}'), tc(args='{"q":"y2"}')], tc(args='{"q":"z3"}'), spec()) == []


def test_loop_carries_side_effects():
    out = check_tool_call([tc(side=True)] * 2, tc(side=True), spec())
    assert out[0].side_effects is True and out[0].kind == "off_task"


def test_out_of_scope_tool():
    s = spec(tools=("search",))
    out = check_tool_call([], tc("delete_all", side=True), s)
    assert out == [Finding("out_of_scope_action", "off_task", True, out[0].evidence)]
    assert check_tool_call([], tc("search"), s) == []
    assert check_tool_call([], tc("anything"), spec()) == []  # no allow-list -> no scope check


def test_canonical_args():
    assert canonical_args('{"b": 1, "a": 2}') == '{"a":2,"b":1}'
    assert canonical_args("not json{") == "not json{"


def _llm(fail=0, calls=1, refused=False):
    return LLMCallRecord(fail, calls, refused)


def test_parse_failure_rate_needs_min_calls():
    bad = _llm(1, 1)
    assert check_llm_call([bad] * 3, bad, "normal") == []  # 4 calls < min_calls
    out = check_llm_call([bad] * 4, bad, "normal")
    assert [f.drift_type for f in out] == ["tool_call_breakage"] and out[0].kind == "degraded"
    # 1 failure in 10 calls = 10% >= normal threshold; 1 in 11 is below it; low needs 20%
    ok = _llm(0, 1)
    assert check_llm_call([ok] * 9 + [bad], ok, "normal") == []
    assert check_llm_call([bad] + [ok] * 8, ok, "normal")[0].drift_type == "tool_call_breakage"
    assert check_llm_call([bad] + [ok] * 8, ok, "low") == []
    # calls without tool calls are ignored
    assert check_llm_call([_llm(0, 0)] * 30, bad, "normal") == []


def test_refusal_regex_and_limit():
    from pitstop.detect.l0 import REFUSAL_RE, is_refusal

    for t in ["I'm sorry, but no", "  I am sorry", "I can't help with that", "I cannot assist", "As an AI model"]:
        assert REFUSAL_RE.match(t), t
    assert not REFUSAL_RE.match("Sure, I'm sorry later")
    assert is_refusal("I'm sorry, no", 0)
    assert not is_refusal("I'm sorry, no", 1)
    assert not is_refusal("I'm sorry " + "x" * 400, 0)
    assert not is_refusal(None, 0)
    r = _llm(0, 0, True)
    assert check_llm_call([], r, "normal") == []
    out = check_llm_call([r], r, "normal")
    assert [f.drift_type for f in out] == ["refusal_spike"]
    assert check_llm_call([], r, "high")[0].drift_type == "refusal_spike"
    assert check_llm_call([r, r], r, "low")[0].evidence["refusals"] == 3
    assert json.dumps(out[0].evidence)
