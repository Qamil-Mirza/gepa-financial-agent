import pickle

from pitstop.sdk.errors import BudgetExceeded, Halted


def test_halted_pickle_round_trip():
    e = pickle.loads(pickle.dumps(Halted("stop now", 7)))
    assert type(e) is Halted
    assert (e.reason, e.drift_id, str(e)) == ("stop now", 7, "stop now")


def test_budget_exceeded_pickle_round_trip():
    orig = BudgetExceeded("global:x", 5.0, 5.25)
    e = pickle.loads(pickle.dumps(orig))
    assert type(e) is BudgetExceeded
    assert (e.scope, e.limit_usd, e.spent_usd) == ("global:x", 5.0, 5.25)
    assert e.reason == orig.reason and str(e) == str(orig)
