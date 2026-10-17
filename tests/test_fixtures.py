import importlib.util
import sys
from pathlib import Path

from gepa_financial_agent.dataset import label_distribution, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("make_dev_fixtures", ROOT / "scripts" / "make_dev_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_dev_fixtures"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fixtures_valid_balanced_and_deterministic(tmp_path):
    mod = load_script()
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    mod.main(["--out", str(out1)])
    mod.main(["--out", str(out2)])
    assert out1.read_bytes() == out2.read_bytes()
    rep = load_jsonl(out1)
    assert rep.invalid == []
    dist = label_distribution(rep.examples)
    assert all(v > 0 for v in dist.values()), dist
    assert 0.3 <= dist["DRIFTING"] / len(rep.examples) <= 0.6
    kinds = {t.split(":", 1)[1] for e in rep.examples for t in e.tags if t.startswith("kind:")}
    assert kinds == set(mod.KINDS)
    drift_types = {e.ground_truth.drift_type for e in rep.examples if e.ground_truth.drift_type}
    assert len(drift_types) == 5  # every category except OTHER is exercised
    assert len({e.example_id for e in rep.examples}) == len(rep.examples)


def test_committed_fixture_file_matches_generator(tmp_path):
    mod = load_script()
    out = tmp_path / "regen.jsonl"
    mod.main(["--out", str(out)])
    committed = ROOT / "data" / "fixtures" / "dev_examples.jsonl"
    assert committed.read_bytes() == out.read_bytes(), "run: uv run python scripts/make_dev_fixtures.py"
