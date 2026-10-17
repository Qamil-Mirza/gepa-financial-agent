import json

from gepa_financial_agent.crusoe_client import MockModelClient
from gepa_financial_agent.dataset import load_jsonl, split_by_scenario
from gepa_financial_agent.optimizer import CrusoeReflectionLM, OptimizeConfig, make_evaluator, run_optimization
from gepa_financial_agent.policy_store import Policy


def drifty_responder(messages):
    user = messages[1]["content"]
    status = "DRIFTING" if ("reddit" in user or "triple" in user) else "ON_TRACK"
    return json.dumps({"status": status, "confidence": 0.7, "drift_type": "OTHER" if status == "DRIFTING" else None,
                       "evidence_event_ids": [], "explanation": "stub", "recommended_action": "NONE"})


def test_make_evaluator_returns_score_and_side_info():
    ex = load_jsonl("data/fixtures/dev_examples.jsonl").examples[0]
    score, info = make_evaluator(MockModelClient(drifty_responder))("seed prompt", ex)
    assert -0.5 <= score <= 1.0 and "feedback" in info and "input_excerpt" in info and "ground_truth" not in info["input_excerpt"]


def test_reflection_lm_wraps_client():
    lm = CrusoeReflectionLM(MockModelClient(lambda m: "NEW PROMPT"))
    assert lm("improve this") == "NEW PROMPT" and lm([{"role": "user", "content": "x"}]) == "NEW PROMPT"
    assert lm.total_cost >= 0.0


def test_run_optimization_end_to_end_with_mocks(tmp_path):
    exs = load_jsonl("data/fixtures/dev_examples.jsonl").examples
    sp = split_by_scenario(exs, seed=0)
    train, val = sp.train[:8], sp.val[:4]
    seed = Policy.from_text("You are a drift supervisor. Decide ON_TRACK or DRIFTING.", name="seed")
    fake_reflection = lambda prompt: "```\nYou are a careful drift supervisor. Flag reddit sourcing and unsupported tripling claims.\n```"
    cfg = OptimizeConfig(max_metric_calls=20, reflection_minibatch_size=2, max_workers=1, run_dir=str(tmp_path / "run"))
    art = run_optimization(seed, train, val, MockModelClient(drifty_responder), fake_reflection, cfg)
    run = tmp_path / "run"
    assert (run / "result.json").exists() and (run / "best_policy.txt").exists() and (run / "config.json").exists()
    assert (run / "candidates.json").exists()  # written by GEPA itself
    assert art.best_policy.text.strip() and art.num_candidates >= 1 and art.total_metric_calls <= 20 + len(val)
    res = json.loads((run / "result.json").read_text())
    assert "val_aggregate_scores" in res and json.loads((run / "config.json").read_text())["dataset"]["train"] == 8


def _load_optimize_script():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("optimize_supervisor", root / "scripts" / "optimize_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def _policies(tmp_path, root):
    pol = tmp_path / "policies"
    pol.mkdir()
    (pol / "baseline.txt").write_text((root / "policies" / "baseline.txt").read_text())
    (pol / "current.txt").write_text((root / "policies" / "baseline.txt").read_text())
    return pol


def _args(tmp_path, root, pol, *extra):
    return ["--mock", "--max-metric-calls", "6", "--dataset", str(root / "data/fixtures/dev_examples.jsonl"),
            "--policies", str(pol), "--run-dir", str(tmp_path / "run"), *extra]


def test_mock_run_refuses_promote_and_is_named_mock(tmp_path):
    mod, root = _load_optimize_script()
    pol = _policies(tmp_path, root)
    before = (pol / "current.txt").read_text()
    assert mod.main(_args(tmp_path, root, pol, "--promote")) != 0
    assert (pol / "current.txt").read_text() == before
    manifest = json.loads((pol / "versions" / "manifest.json").read_text())
    assert all(name.startswith("mock_") for name in manifest["versions"]) and manifest["versions"]


def test_test_split_untouched_unless_final_test(tmp_path):
    mod, root = _load_optimize_script()
    pol = _policies(tmp_path, root)
    assert mod.main(_args(tmp_path, root, pol)) == 0
    labels = set(json.loads((tmp_path / "run" / "comparison.json").read_text())["runs"])
    assert labels == {"baseline/val", "gepa/val"}
