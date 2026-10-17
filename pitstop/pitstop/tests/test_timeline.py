"""Curated run timeline: drift flags and responses are highlighted, routine verdicts are folded."""
import json

import httpx
import pytest
from rich.console import Console

from pitstop.config import ROOT
from pitstop.orchestrator import Swarm
from pitstop.research.youcom import FinanceResearchClient
from pitstop.schemas import Scenario
from pitstop.supervisor.supervisor import Supervisor
from pitstop.timeline import TimelineBuilder, TimelineKind, build_from_run_dir

from test_pipeline_offline import SCENARIO, supervisor_handler, worker_handler


def _record(status, *, step=0, phase="pre_action", note="", level="ok", evidence=(), dropped=(), downgraded=False):
    traj = [{"event_id": f"a-e{i:03d}", "agent_id": "a", "step": i, "seq": i, "type": "plan",
             "timestamp": "t", "summary": f"plan {i}", "payload": {}} for i in range(step + 1)]
    return {
        "agent_id": "a", "step": step, "phase": phase, "alert_level": level, "alert_note": note,
        "drift_injection_active": False,
        "input": {"timestamp": f"ts{step}", "trajectory": traj,
                  "proposed_next_action": {"action": "research", "query": f"q{step}", "rationale": "r"}},
        "result": {"decision": {"agent_id": "a", "status": status, "confidence": 0.9,
                                "drift_type": "none" if status == "ON_TRACK" else "goal_deviation",
                                "evidence_event_ids": list(evidence), "explanation": "x"},
                   "dropped_evidence_ids": list(dropped), "downgraded": downgraded, "issues": []},
    }


def _kinds(tb):
    return [e.kind for e in tb.entries]


def test_routine_verdicts_are_folded_into_quiet_streak():
    tb = TimelineBuilder()
    for i in range(3):
        assert tb.on_decision(_record("ON_TRACK", step=i), i) == []
    fin = tb.agent_finished("a", 3)
    assert fin.quiet_streak == 3
    assert len(tb.document()["agents"]["a"]["verdicts"]) == 3


def test_flag_then_recovery_is_linked():
    tb = TimelineBuilder()
    tb.on_decision(_record("ON_TRACK", step=0), 0)
    flag, = tb.on_decision(_record("DRIFTING", step=1, note="INTERVENTION POINT", level="alert",
                                   evidence=["a-e001"]), 1)
    assert flag.kind == TimelineKind.DRIFT_FLAGGED and flag.severity == "critical"
    assert flag.quiet_streak == 1
    assert flag.evidence[0].summary == "plan 1" and flag.data["blocked_query"] == "q1"
    resp, = tb.on_decision(_record("ON_TRACK", step=2, note="already flagged", level="alert"), 2)
    assert resp.kind == TimelineKind.DRIFT_RESPONSE and resp.responds_to == flag.entry_id
    assert resp.data["outcome"] == "recovered"


def test_persisting_drift_after_flag():
    tb = TimelineBuilder()
    flag, = tb.on_decision(_record("DRIFTING", note="INTERVENTION POINT", level="alert"), 0)
    out = tb.on_decision(_record("DRIFTING", step=1, note="already flagged", level="alert"), 1)
    assert [e.kind for e in out] == [TimelineKind.DRIFT_RESPONSE, TimelineKind.DRIFT_SUSPECTED]
    assert out[0].data["outcome"] == "persisted" and out[0].responds_to == flag.entry_id


def test_status_change_suspicion_and_correction():
    tb = TimelineBuilder()
    tb.on_decision(_record("UNCERTAIN", step=0, level="watch", note="monitoring",
                           dropped=["a-e999"], downgraded=True), 0)
    tb.on_decision(_record("DRIFTING", step=1, level="watch", note="monitoring"), 1)
    tb.on_decision(_record("UNCERTAIN", step=2, level="watch", note="monitoring"), 2)  # DRIFTING -> UNCERTAIN
    tb.on_decision(_record("UNCERTAIN", step=3, level="watch", note="monitoring"), 3)  # unchanged: folded
    assert _kinds(tb) == [TimelineKind.SUPERVISOR_CORRECTION, TimelineKind.STATUS_CHANGED,
                          TimelineKind.DRIFT_SUSPECTED, TimelineKind.STATUS_CHANGED]
    assert tb.entries[2].severity == "warning"


@pytest.mark.asyncio
async def test_swarm_writes_timeline_and_offline_rebuild_matches(tmp_path, FakeLLMClass, fixture_response):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture_response)))
    research = FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http, max_live_calls=10)
    swarm = Swarm(
        Scenario.model_validate(SCENARIO), worker_llm=FakeLLMClass(worker_handler),
        supervisor=Supervisor(FakeLLMClass(supervisor_handler), ROOT / "prompts" / "supervisor_v0.txt"),
        research=research, prompts_dir=ROOT / "prompts" / "worker", runs_dir=tmp_path / "runs",
        halt_on_drift=True, console=Console(file=open(tmp_path / "out.txt", "w")),
    )
    run = await swarm.run_all()

    doc = json.loads((run.run_dir / "timeline.json").read_text())
    entries = doc["entries"]
    assert entries[0]["kind"] == "run_started" and entries[-1]["kind"] == "run_finished"
    assert len(entries) == len((run.run_dir / "timeline.jsonl").read_text().splitlines())

    bad = [e for e in entries if e["agent_id"] == "bad"]
    flag = next(e for e in bad if e["kind"] == "drift_flagged")
    assert flag["step"] == 1 and flag["data"]["blocked_query"].startswith("Example Corp gaming")
    assert flag["evidence"] and flag["evidence"][0]["type"] == "plan"
    halt = next(e for e in bad if e["kind"] == "drift_response")
    assert halt["responds_to"] == flag["entry_id"] and halt["data"]["outcome"] == "halted"

    good = [e for e in entries if e["agent_id"] == "good"]
    assert [e["kind"] for e in good] == ["agent_finished"] and good[0]["quiet_streak"] >= 3
    assert doc["agents"]["bad"]["halted"] and doc["agents"]["bad"]["intervention"]["step"] == 1

    rebuilt = build_from_run_dir(run.run_dir, write=False)
    strip = lambda es: [(e["kind"], e["agent_id"], e["step"], e["responds_to"], e["quiet_streak"]) for e in es]
    assert strip(rebuilt["entries"]) == strip(entries)
