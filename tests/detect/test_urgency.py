import pytest

from pitstop.detect.l0 import Finding
from pitstop.detect.urgency import Level, action_for, classify

P0, P1, P2, P3 = Level.P0, Level.P1, Level.P2, Level.P3
ROWS = [
    ("budget_exhausted", True, (P0, P0, P0)),
    ("budget_exhausted", False, (P0, P0, P0)),
    ("out_of_scope_action", True, (P0, P0, P0)),
    ("out_of_scope_action", False, (P1, P2, P3)),
    ("unproductive_loop", True, (P0, P0, P0)),
    ("unproductive_loop", False, (P0, P1, P2)),
    ("tool_call_breakage", True, (P1, P1, P2)),
    ("tool_call_breakage", False, (P1, P1, P2)),
    ("refusal_spike", True, (P1, P1, P2)),
    ("refusal_spike", False, (P1, P1, P2)),
    ("something_else", True, (P2, P2, P3)),
    ("something_else", False, (P2, P2, P3)),
]


@pytest.mark.parametrize("drift_type,side,expected", ROWS)
def test_classify_table(drift_type, side, expected):
    got = tuple(classify(drift_type, c, side) for c in ("high", "medium", "low"))
    assert got == expected


def test_action_for_order():
    f = lambda t, k, s: Finding(t, k, s)  # noqa: E731
    assert action_for(f("budget_exhausted", "degraded", True), P0, "high") == "halt"
    assert action_for(f("budget_exhausted", "off_task", False), P0, "low") == "halt"
    assert action_for(f("unproductive_loop", "off_task", True), P0, "high") == "block"
    assert action_for(f("unproductive_loop", "off_task", False), P0, "high") == "steer"
    assert action_for(f("refusal_spike", "degraded", False), P1, "high") == "record"
    assert action_for(f("tool_call_breakage", "degraded", True), P1, "medium") == "record"
