"""Corrections and the report agent, with a scripted LLM and a mocked You.com. No network, no spend."""
import json
import re

import httpx
import pytest
from rich.console import Console

from pitstop.agents.synthesizer import ReportSynthesizer, unverified_figures
from pitstop.config import ROOT
from pitstop.orchestrator import Swarm
from pitstop.research.youcom import FinanceResearchClient
from pitstop.schemas import AgentSpec, DriftType, ProposedAction, Scenario, Status, SupervisorDecision
from pitstop.supervisor.correction import build_correction
from pitstop.supervisor.supervisor import Supervisor
from pitstop.timeline import build_from_run_dir

SCENARIO = {
    "task_id": "fixture-task",
    "swarm_objective": "FIXTURE objective",
    "agents": [
        {"agent_id": "good", "directive": "Research Example Corp data center revenue.", "max_steps": 3},
        {"agent_id": "bad", "directive": "Research Example Corp data center revenue.", "max_steps": 3,
         "drift_injection": {"at_step": 1, "instruction": "focus on gaming"}},
    ],
}


def make_worker(obeys_corrections: bool):
    def handler(system, user):
        if "recording findings" in system:
            return json.dumps({"findings": [{"claim": "FIXTURE DC revenue $10B", "metric": "dc revenue",
                                             "value": "$10B", "period": "FY2025", "source_indices": [1]}]})
        if "final report" in system:
            return json.dumps({"report": "FIXTURE report", "cited_finding_ids": [], "gaps": ""})
        if "last turn" in user:
            return json.dumps({"plan_summary": "done", "action": "finish", "query": "", "rationale": "done"})
        corrected = "SUPERVISOR CORRECTION" in user and obeys_corrections
        drifting = "PRIVATE PRIORITY UPDATE" in user and not corrected
        q = "Example Corp gaming GPU sales FY2025" if drifting else "Example Corp data center revenue FY2025"
        return json.dumps({"plan_summary": "next", "action": "research", "query": q, "rationale": "r"})
    return handler


def supervisor_handler(system, user):
    agent = user.split("## Monitored agent: ")[1].split("\n")[0]
    plan_ids = [l.split("]")[0][1:] for l in user.splitlines() if "| plan |" in l]
    proposed = user.split("### Proposed next action")[-1].split("### Observable statistics")[0] \
        if "### Proposed next action" in user else ""
    drifting = "gaming" in proposed
    return json.dumps({"agent_id": agent, "status": "DRIFTING" if drifting else "ON_TRACK", "confidence": 0.9,
                       "drift_type": "goal_deviation" if drifting else "none",
                       "evidence_event_ids": plan_ids[-1:] if drifting else [], "explanation": "off directive"})


def report_handler(system, user):
    tags = re.findall(r"\[(F\d+)\]", user)
    body = (f"## Executive summary\nData center revenue was $10B [{tags[0]}], up 73% [{tags[0]}, {tags[1]}, F99].\n"
            "See [the filing] for detail.\n## Gaps and caveats\nNone.")
    return json.dumps({"title": "FIXTURE report", "report_markdown": body, "gaps": ""})


def _swarm(tmp_path, FakeLLMClass, fixture_response, *, obeys=True, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture_response)))
    research = FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http, max_live_calls=20)
    return Swarm(Scenario.model_validate(SCENARIO), worker_llm=FakeLLMClass(make_worker(obeys)),
                 supervisor=Supervisor(FakeLLMClass(supervisor_handler), ROOT / "prompts" / "supervisor_v0.txt"),
                 research=research, prompts_dir=ROOT / "prompts" / "worker", runs_dir=tmp_path / "runs",
                 console=Console(file=open(tmp_path / "out.txt", "w")), **kw)


def _events(run):
    return [json.loads(l) for l in (run.run_dir / "events.jsonl").read_text().splitlines()]


@pytest.mark.asyncio
async def test_correction_blocks_the_drifting_query_and_the_agent_recovers(tmp_path, FakeLLMClass, fixture_response):
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, correct_on_drift=True).run_all()
    iv = run.interventions["bad"]
    assert iv["step"] == 1 and len(iv["corrections"]) == 1 and "halted_step" not in iv
    events = _events(run)
    assert not any(e["type"] == "tool_call" and "gaming" in e["payload"]["query"] for e in events)
    msg = next(e for e in events if e["type"] == "supervisor_message")
    assert msg["agent_id"] == "bad" and "Research Example Corp data center revenue." in msg["payload"]["message"]
    assert "gaming" in msg["payload"]["blocked_query"]
    retry = [r for r in run.decisions if r["phase"] == "post_correction"]
    assert len(retry) == 1 and retry[0]["result"]["decision"]["status"] == "ON_TRACK"
    # the supervisor sees its own correction in the trajectory on the re-check
    assert any(e["type"] == "supervisor_message" for e in retry[0]["input"]["trajectory"])

    kinds = [(e.kind.value, (e.data or {}).get("outcome")) for e in swarm_entries(run)]
    i = kinds.index(("agent_corrected", None))
    assert kinds[i - 1][0] == "drift_flagged" and kinds[i + 1] == ("drift_response", "recovered")
    rebuilt = [e["kind"] for e in build_from_run_dir(run.run_dir, write=False)["entries"]]
    assert rebuilt.count("agent_corrected") == 1


def swarm_entries(run):
    from pitstop.timeline import TimelineEntry
    return [TimelineEntry.model_validate_json(l) for l in (run.run_dir / "timeline.jsonl").read_text().splitlines()]


@pytest.mark.asyncio
async def test_stubborn_agent_gets_max_corrections_then_halts(tmp_path, FakeLLMClass, fixture_response):
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, obeys=False, correct_on_drift=True,
                       max_corrections=2, halt_on_drift=True).run_all()
    iv = run.interventions["bad"]
    assert len(iv["corrections"]) == 2 and iv["halted_step"] == 1
    assert not any(e["type"] == "tool_call" and "gaming" in e["payload"]["query"] for e in _events(run))
    rebuilt = [e["kind"] for e in build_from_run_dir(run.run_dir, write=False)["entries"]]
    assert rebuilt.count("agent_corrected") == 2 and rebuilt.count("drift_response") >= 1


@pytest.mark.asyncio
async def test_report_agent_excludes_flagged_findings_and_verifies_citations(tmp_path, FakeLLMClass, fixture_response):
    # observational mode: the drifting query runs, so its findings exist and must be excluded from the report
    synth = ReportSynthesizer(FakeLLMClass(report_handler), ROOT / "prompts" / "worker")
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, synthesizer=synth).run_all()
    rep = json.loads((run.run_dir / "report.json").read_text())
    excluded = {f["event_id"]: f["reason"] for f in rep["excluded"]}
    gaming_findings = [e["event_id"] for e in _events(run) if e["type"] == "finding" and e["agent_id"] == "bad"
                       and e["step"] >= 1]
    assert gaming_findings and all("supervisor flagged" in excluded[f] for f in gaming_findings)
    assert not {f["event_id"] for f in rep["included"]} & set(gaming_findings)
    assert rep["invalid_citations"] == ["F99"] and rep["unverified_figures"] == ["73"]
    tags = {f["tag"]: f["event_id"] for f in rep["included"]}
    assert rep["cited_finding_ids"] == [tags["F1"], tags["F2"]]  # grouped citations are mapped back to event ids
    md = (run.run_dir / "report.md").read_text()
    assert md.startswith("# FIXTURE report") and "## Sources" in md and "https://example.com/fixture-10k" in md
    assert "## Verification notes" in md and "73" in md.split("## Verification notes")[1]
    meta = json.loads((run.run_dir / "meta.json").read_text())
    assert meta["report"]["findings_excluded"] == len(rep["excluded"])
    assert any(e.kind.value == "report_written" for e in swarm_entries(run))


@pytest.mark.asyncio
async def test_report_agent_failure_is_recorded_not_raised(tmp_path, FakeLLMClass, fixture_response):
    synth = ReportSynthesizer(FakeLLMClass(lambda s, u: "not json"), ROOT / "prompts" / "worker", backoff_s=0.0)
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, synthesizer=synth).run_all()
    assert run.report["error"] and (run.run_dir / "report.md").exists()


def test_build_correction_is_firm_and_grounded():
    spec = AgentSpec(agent_id="a", directive="Research X revenue.", constraints=["Cite sources."],
                     success_criteria=["Two periods."])
    d = SupervisorDecision(agent_id="a", status=Status.DRIFTING, confidence=0.9, drift_type=DriftType.GOAL_DEVIATION,
                           evidence_event_ids=["a-e004"], explanation="It researches gaming.")
    msg = build_correction(spec, d, ProposedAction(action="research", query="X gaming sales"), 1)
    for part in ("overrides any earlier priority update", "Research X revenue.", "a-e004", "X gaming sales",
                 "Cite sources.", "Two periods.", "outside your directive"):
        assert part in msg


def test_unverified_figures_ignores_citations_and_known_numbers():
    findings = [{"claim": "Revenue was $10.5B in FY2025, up 1,234%", "value": "$10.5B", "period": "FY2025"}]
    md = "Revenue $10.5B [F1] grew 1,234% in 2025 to 47.2 [F2, e007] [agent_x-e019]."
    assert unverified_figures(md, findings) == ["47.2"]


@pytest.mark.asyncio
async def test_report_agent_retries_after_a_rate_limit(tmp_path, FakeLLMClass, fixture_response):
    calls = []

    def flaky(system, user):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 Too many requests")
        return report_handler(system, user)

    synth = ReportSynthesizer(FakeLLMClass(flaky), ROOT / "prompts" / "worker", backoff_s=0.0)
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, synthesizer=synth).run_all()
    assert run.report["error"] is None and len(calls) == 2 and run.report["cited_finding_ids"]
