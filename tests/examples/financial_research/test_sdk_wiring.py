"""The swarm watched by the PitStop SDK alone: the scenario's injected topic drift is caught by the L1 pre-action
gate on the paid `youcom_research` tool, offline (scripted worker, canned research, scripted judge)."""
import asyncio
import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from finresearch import offline
from finresearch.config import SCENARIOS
from finresearch.schemas import EventType, Scenario
from pitstop.sdk.runtime import get_runtime

SCRUB = ("CRUSOE_API_KEY", "YDC_API_KEY", "PAGERDUTY_ROUTING_KEY", "OPSGENIE_API_KEY", "SLACK_WEBHOOK_URL",
         "PITSTOP_WEBHOOK_URL", "SMTP_PASSWORD")
FINRESEARCH = Path(sys.executable).with_name("finresearch")  # the console script in this environment


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env["PITSTOP_HOME"] = str(home)
    return env


def _offline_run(tmp_path, *, inject_drift=True):
    scenario = Scenario.model_validate_json((SCENARIOS / "nvda_amd.json").read_text())
    swarm = offline.make_swarm(scenario, runs_dir=tmp_path / "runs", inject_drift=inject_drift,
                               console=Console(file=io.StringIO()))
    return swarm, asyncio.run(swarm.run_all())


def test_injected_drift_blocked_and_recorded(tmp_path):
    swarm, run = _offline_run(tmp_path)
    c = get_runtime().store.connect()
    drifts = c.execute("SELECT * FROM drifts WHERE status='open'").fetchall()
    assert [(d["drift_type"], d["level"], d["action"]) for d in drifts] == [("out_of_scope_action", "P0", "block")]
    blocks = c.execute("SELECT agent, name FROM events WHERE kind='block'").fetchall()
    assert blocks, "the gaming queries were not blocked"
    assert {(b["agent"], b["name"]) for b in blocks} == {("agent_nvda_dc", "youcom_research")}
    runs = c.execute("SELECT id, workflow, status FROM runs").fetchall()
    assert [(r["workflow"], r["status"]) for r in runs] == [("financial-research", "ok")]  # medium: block, not halt
    assert run.pitstop_run_id == runs[0]["id"] and run.halted is None

    # the block reached the worker as an ERROR event naming the reason; no gaming query was ever researched
    errors = [e for e in swarm.collector.history("agent_nvda_dc") if e.type == EventType.ERROR]
    assert errors and all("blocked by PitStop" in e.summary for e in errors)
    assert all("about consumer gaming, not the directive" in e.payload["reason"] for e in errors)
    assert len(errors) == len(blocks)
    assert swarm.research.queries and not any("gaming" in q.lower() or "geforce" in q.lower()
                                               for q in swarm.research.queries)
    meta = json.loads((run.run_dir / "meta.json").read_text())
    assert meta["supervisor"] is None and meta["pitstop"]["run_id"] == run.pitstop_run_id


def test_finresearch_offline_run_blocks_injected_drift(tmp_path):
    assert FINRESEARCH.exists(), "finresearch console script missing; run the suite via `uv run pytest`"
    env = _env(tmp_path)
    out = subprocess.run([str(FINRESEARCH), "run", "--offline"], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Traceback" not in out.stderr
    assert "research blocked by PitStop" in out.stdout and "PitStop blocked 2 research call(s)" in out.stdout
    d = subprocess.run([sys.executable, "-m", "pitstop.cli", "drifts"], cwd=tmp_path, env=env, capture_output=True,
                       text=True, timeout=60)
    assert d.returncode == 0, d.stderr
    line = next(ln for ln in d.stdout.splitlines() if "out_of_scope_action" in ln)
    assert "P0" in line and "block" in line and "financial-research" in line


def test_clean_offline_run_records_no_drift(tmp_path):
    env = _env(tmp_path)
    out = subprocess.run([str(FINRESEARCH), "run", "--offline", "--no-inject-drift"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "research blocked by PitStop" not in out.stdout
    c = sqlite3.connect(tmp_path / "pitstop.db")
    assert c.execute("SELECT COUNT(*) FROM drifts").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM events WHERE kind IN ('block', 'gate_error')").fetchone()[0] == 0
    assert c.execute("SELECT status FROM runs").fetchall() == [("ok",)]
    allowed = c.execute("SELECT COUNT(*) FROM events WHERE kind='gate'").fetchone()[0]
    assert allowed > 0  # every research call went through the gate and was allowed


def test_halted_ends_the_swarm_and_records_the_run_halted(tmp_path, FakeLLMClass):
    from pitstop import Halted

    scenario = Scenario.model_validate_json((SCENARIOS / "nvda_amd.json").read_text())
    out = io.StringIO()
    swarm = offline.make_swarm(scenario, runs_dir=tmp_path / "runs", console=Console(file=out, width=200))

    def halt(system, user):
        raise Halted("budget hit (test)")

    swarm.workers[1].llm = FakeLLMClass(halt)
    run = asyncio.run(swarm.run_all())
    assert run.halted == "budget hit (test)"
    assert "PitStop halted the run: budget hit (test)" in out.getvalue()
    c = get_runtime().store.connect()
    assert [r["status"] for r in c.execute("SELECT status FROM runs")] == ["halted"]
    assert json.loads((run.run_dir / "meta.json").read_text())["pitstop"]["halted"] == "budget hit (test)"


def test_blocked_worker_is_told_why_when_it_replans(tmp_path, monkeypatch):
    seen = []
    reply = offline.ScriptedWorkerLLM._reply
    monkeypatch.setattr(offline.ScriptedWorkerLLM, "_reply",
                        staticmethod(lambda system, user: seen.append(user) or reply(system, user)))
    _offline_run(tmp_path)
    told = [u for u in seen if "PitStop blocked your last research call" in u]
    assert told and all("Your directive: Investigate NVIDIA's data-center" in u for u in told)


@pytest.mark.parametrize("argv, message", [
    (["run", "--shadow-prompt", "x.txt"], "--shadow-prompt need --legacy-supervisor"),
    (["run", "--correct-on-drift"], "--correct-on-drift need --legacy-supervisor"),
    (["run", "--offline", "--legacy-supervisor"], "not available with --offline"),
])
def test_run_flag_combinations_rejected(argv, message, monkeypatch, capsys):
    from finresearch import cli

    monkeypatch.setattr(sys, "argv", ["finresearch", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2 and message in capsys.readouterr().err


def test_unwatched_swarm_records_no_pitstop_run_and_makes_no_judge_call(tmp_path):
    """watch=False (the benchmarks and the demo pair): no workflow, no gated tool, no judge call."""
    import pitstop
    from finresearch.config import PROMPTS
    from finresearch.orchestrator import Swarm

    judge, judged = offline.scripted_judge(), []
    create = judge.chat.completions.create
    judge.chat.completions.create = lambda **kw: judged.append(kw) or create(**kw)
    pitstop.init(judge_client=judge)
    scenario = Scenario.model_validate_json((SCENARIOS / "nvda_amd.json").read_text())
    swarm = Swarm(scenario, worker_llm=offline.ScriptedWorkerLLM(), research=offline.CannedResearch(),
                  prompts_dir=PROMPTS / "worker", runs_dir=tmp_path / "runs", console=Console(file=io.StringIO()),
                  watch=False)
    run = asyncio.run(swarm.run_all())
    assert judged == []
    c = get_runtime().store.connect()
    for table in ("runs", "events", "drifts"):
        assert c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert run.pitstop_run_id is None
    assert any("GeForce" in q for q in swarm.research.queries)  # the injected drift ran: nothing gated it
    assert json.loads((run.run_dir / "meta.json").read_text())["pitstop"] is None
