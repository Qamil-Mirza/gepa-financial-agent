import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("summarize_pitstop",
                                               Path(__file__).resolve().parents[1] / "scripts" / "summarize_pitstop.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def _run(tmp_path, name, correct_by_case, trials=3, infra=0):
    d = tmp_path / name
    d.mkdir()
    rows = []
    for cid, ok_flags in correct_by_case.items():
        label = cid.rsplit(".", 1)[1]
        for t in range(trials):
            ok = ok_flags[t]
            fired = (label == "drift") == ok
            rows.append({"id": cid, "category": cid.split(".")[1], "label": label, "expect_type": "goal_deviation",
                         "trial": t, "fired": fired, "correct": ok, "drift_type": "goal_deviation" if fired else "none",
                         "valid_json": True, "downgraded": False, "infra_error": False})
    rows += [dict(rows[0], infra_error=True)] * infra
    (d / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "summary.json").write_text(json.dumps({"summary": {"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                                                            "prompt": name, "usage": {"prompt_tokens": 1000,
                                                                                      "completion_tokens": 100}}}))
    return d


def test_summarize_and_paired_sign_test(tmp_path):
    base = _run(tmp_path, "base", {"x.detour.drift": [0, 0, 1], "x.detour.ok": [1, 1, 1],
                                   "x.uncited.drift": [1, 1, 1], "x.uncited.ok": [1, 1, 0]}, infra=2)
    gepa = _run(tmp_path, "gepa", {"x.detour.drift": [1, 1, 1], "x.detour.ok": [1, 1, 1],
                                   "x.uncited.drift": [1, 1, 1], "x.uncited.ok": [0, 0, 1]})
    a, b = sp.summarize_arm(base), sp.summarize_arm(gepa)
    assert a["judgments"] == 12 and a["infra_errors_excluded"] == 2 and a["trials"] == 3
    assert a["accuracy"]["per_trial"] == [0.75, 0.75, 0.75] and b["accuracy"]["mean"] == 0.833
    assert a["pair_accuracy"]["per_trial"] == [0.5, 0.5, 0.5]
    p = sp.paired(a, b)
    assert p["fixed"] == ["x.detour.drift"] and p["broken"] == ["x.uncited.ok"] and p["sign_test_p"] == 1.0
    assert sp.main(["--arm", f"base={base}", "--arm", f"gepa={gepa}", "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["reference"] == "base"
