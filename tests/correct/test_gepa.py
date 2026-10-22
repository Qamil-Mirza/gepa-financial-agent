import json

import pytest

from pitstop.correct.client import MockModelClient
from pitstop.correct.gepa import CrusoeReflectionLM, OptimizeConfig, run_optimization
from pitstop.correct.policy import Policy
from pitstop.sdk.errors import BudgetExceeded, Halted


def test_reflection_lm_wraps_client():
    lm = CrusoeReflectionLM(MockModelClient(lambda m: "NEW PROMPT"))
    assert lm("improve this") == "NEW PROMPT" and lm([{"role": "user", "content": "x"}]) == "NEW PROMPT"
    assert lm.total_cost >= 0.0


def test_run_optimization_requires_and_uses_caller_evaluator(tmp_path):
    seen = []
    def evaluator(candidate: str, example: str):
        seen.append(example); return (1.0 if "x" in example else 0.0), {"feedback": "ok"}
    art = run_optimization(Policy.from_text("seed prompt"), ["x1", "y1"], ["x2"], MockModelClient(),
                           CrusoeReflectionLM(MockModelClient()), OptimizeConfig(max_metric_calls=6, max_workers=1,
                           run_dir=str(tmp_path / "run")), evaluator=evaluator, objective="o", background="b")
    assert seen and (tmp_path / "run" / "artifacts.json").is_file()
    assert art.seed_policy.text == "seed prompt"


def test_run_optimization_has_no_default_evaluator():
    import inspect
    p = inspect.signature(run_optimization).parameters
    assert p["evaluator"].default is inspect.Parameter.empty
    assert p["objective"].default is inspect.Parameter.empty


def _run(tmp_path, evaluator, lm=None, calls_budget=30):
    return run_optimization(Policy.from_text("seed prompt"), ["x1", "y1", "z1"], ["x2", "y2"], MockModelClient(),
                            lm or CrusoeReflectionLM(MockModelClient(lambda m: "NEW PROMPT")),
                            OptimizeConfig(max_metric_calls=calls_budget, max_workers=1, run_dir=str(tmp_path / "r")),
                            evaluator=evaluator, objective="o", background="b")


def test_budget_exceeded_aborts_optimization(tmp_path):
    calls = []
    def evaluator(candidate, example):
        calls.append(1)
        if len(calls) == 3:
            raise BudgetExceeded("global:crusoe", 50.0, 50.0)
        return 0.5, {}
    with pytest.raises(BudgetExceeded):
        _run(tmp_path, evaluator)
    assert len(calls) == 3  # nothing called through after the halt
    aborted = json.loads((tmp_path / "r/aborted.json").read_text())
    assert aborted["type"] == "BudgetExceeded" and aborted["drift_id"] is None and aborted["reason"]
    for name in ("result.json", "best_policy.txt", "artifacts.json"):
        assert not (tmp_path / "r" / name).exists()


def test_halt_in_reflection_lm_aborts(tmp_path):
    halt = Halted("stop now", drift_id="d1")
    lm_calls = []
    def lm(prompt):
        lm_calls.append(1)
        raise halt
    def evaluator(candidate, example):
        return 0.5, {}
    with pytest.raises(Halted) as ei:
        _run(tmp_path, evaluator, lm=lm)
    assert ei.value is halt and len(lm_calls) == 1
    aborted = json.loads((tmp_path / "r/aborted.json").read_text())
    assert aborted == {"reason": "stop now", "drift_id": "d1", "type": "Halted"}
    assert not (tmp_path / "r/best_policy.txt").exists()


def test_unhalted_run_unchanged(tmp_path):
    art = _run(tmp_path, lambda c, e: (0.5, {}), calls_budget=6)
    assert (tmp_path / "r/artifacts.json").is_file() and (tmp_path / "r/best_policy.txt").is_file()
    assert not (tmp_path / "r/aborted.json").exists() and art.seed_policy.text == "seed prompt"
