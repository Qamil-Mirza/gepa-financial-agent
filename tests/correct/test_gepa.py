from pitstop.correct.client import MockModelClient
from pitstop.correct.gepa import CrusoeReflectionLM, OptimizeConfig, run_optimization
from pitstop.correct.policy import Policy


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
