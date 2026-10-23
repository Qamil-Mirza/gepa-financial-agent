"""A benchmark swarm stage whose run ends halted must fail loudly, naming the run."""
from types import SimpleNamespace

import finresearch.orchestrator as orch
from benchmarks.calibration import loop
from finresearch.config import PROMPTS


def test_benchmark_fails_on_halted_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    class HaltedSwarm:
        def __init__(self, *a, **kw):
            pass

        async def run_all(self):
            return SimpleNamespace(run_id="nvda_amd-abc123", halted="budget exceeded")

    monkeypatch.setattr(orch, "Swarm", HaltedSwarm)
    s = SimpleNamespace(crusoe_api_key="k", base_url="https://offline.invalid/v1")
    state = SimpleNamespace(current="x", current_path=lambda: PROMPTS / "supervisor_v0.txt")
    args = SimpleNamespace(round=0, supervisor_model="m", mode="replay", no_correct=True, no_report=True,
                           scenarios=["nvda_amd"], max_research_calls=None)
    (tmp_path / "runs/recursive/round_00").mkdir(parents=True)
    assert loop.cmd_swarm(args, s, state) != 0
    out = capsys.readouterr()
    assert "nvda_amd-abc123" in out.out + out.err and "halted" in (out.out + out.err).lower()
