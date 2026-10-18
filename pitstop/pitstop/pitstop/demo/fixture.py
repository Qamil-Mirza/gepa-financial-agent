"""Offline FIXTURE swarm run for building the demo page without spending anything.

Scripted LLMs (workers + two supervisors) and a mocked You.com, the same pattern as
tests/test_pipeline_offline.py. It mirrors scenarios/nvda_amd.json so the page looks right, but every
text the page shows starts with "FIXTURE" and the view is marked source="offline-fixture".
The shadow ("GEPA") supervisor is scripted to flag the NVIDIA drift one step before the primary:
this is a layout aid, not a result.

make_fixture_pair writes the same thing as a live-mode pair (demo/pair.py): one swarm per supervisor, each
correcting its own drifting agent and writing a FIXTURE team report. Corrected agents return to the directive.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import httpx
from rich.console import Console

from ..agents.synthesizer import ReportSynthesizer
from ..config import ROOT
from ..llm import LLMClient, LLMResult
from ..orchestrator import Swarm, SwarmRun
from ..research.youcom import FinanceResearchClient
from ..schemas import Scenario
from ..supervisor.supervisor import Supervisor, load_prompt
from .pair import LIVE_MAX_CORRECTIONS, Pair

SCENARIO = ROOT / "scenarios" / "nvda_amd.json"
GEPA_PROMPT = ROOT / "prompts" / "gepa" / "pitstop_nano_800.txt"
FALLBACK_GEPA_PROMPT = ROOT.parent.parent / "policies" / "versions" / "gepa_nano_150.txt"

QUERIES = {  # per agent, by step; the NVIDIA agent's drift injection starts at step 2
    "agent_nvda_dc": ["FIXTURE: NVIDIA data center revenue FY2025",
                      "FIXTURE: NVIDIA data center revenue growth drivers FY2025",
                      "FIXTURE: NVIDIA GeForce RTX gaming revenue",
                      "FIXTURE: NVIDIA RTX 50 series launch gamer demand"],
    "agent_amd_ai": ["FIXTURE: AMD data center segment revenue 2024",
                     "FIXTURE: AMD Instinct MI300 accelerator revenue 2024",
                     "FIXTURE: AMD data center segment growth drivers 2025",
                     "FIXTURE: AMD Instinct accelerator guidance 2025"],
    "agent_supply": ["FIXTURE: TSMC CoWoS advanced packaging capacity for NVIDIA and AMD",
                     "FIXTURE: US export restrictions on AI accelerators to China",
                     "FIXTURE: HBM memory supply constraints for AI accelerators"],
}
RECOVERY_QUERIES = {"agent_nvda_dc": ["FIXTURE: NVIDIA data center revenue by quarter FY2025",
                                     "FIXTURE: NVIDIA Blackwell data center demand FY2026"]}
DRIFT_WORDS = ("GeForce", "gaming", "gamer")

YOUCOM_RESPONSE = {"output": {
    "content": "FIXTURE: synthetic research answer, not real financial data. Segment revenue was $10.0B, "
               "up 50%.[[1]] Demand for accelerators was the main driver.[[1, 2]]",
    "content_type": "text",
    "sources": [{"url": "https://example.com/fixture-10k", "title": "FIXTURE 10-K filing"},
                {"url": "https://example.com/fixture-earnings", "title": "FIXTURE earnings call"}],
}}


class ScriptedLLM(LLMClient):
    """Stand-in for Crusoe: handler(system, user) -> reply text. No network."""

    def __init__(self, handler, model: str):
        super().__init__(model, api_key=None)
        self.handler = handler

    async def complete(self, system, user, *, temperature=None):
        r = LLMResult(text=self.handler(system, user), model=self.model, latency_s=0.0,
                      prompt_tokens=len(user) // 4, completion_tokens=60)
        self.usage.add(r)
        return r


def _agent_from_directive(user: str) -> str:
    d = user.split("Your directive: ", 1)[-1].split("\n", 1)[0]
    return "agent_supply" if "supply-side" in d else "agent_nvda_dc" if "NVIDIA's data-center" in d else "agent_amd_ai"


def worker_handler(system: str, user: str) -> str:
    agent = _agent_from_directive(user)
    if "recording findings" in system:
        q = user.split("Question asked: ", 1)[-1].split("\n", 1)[0].removeprefix("FIXTURE: ")
        return json.dumps({"findings": [{"claim": f"FIXTURE: synthetic figure for {q}", "metric": "FIXTURE metric",
                                         "value": "$10.0B (FIXTURE)", "period": "FY2025", "source_indices": [1]}]})
    if "final report" in system:
        ids = re.findall(r"\[([\w-]+-e\d+)\]", user)
        return json.dumps({"report": f"FIXTURE report for {agent}: synthetic summary of the recorded findings, "
                                     f"not real data.", "cited_finding_ids": ids[:3], "gaps": "FIXTURE"})
    if "last turn" in user:
        return json.dumps({"plan_summary": "FIXTURE: success criteria covered; finishing.", "action": "finish",
                           "query": "", "rationale": "FIXTURE: done"})
    m = re.search(r"Turns remaining including this one: (\d+)", user)
    max_steps = {"agent_supply": 4}.get(agent, 5)
    qs = QUERIES[agent]
    step = min(max_steps - int(m.group(1)) if m else 0, len(qs) - 1)
    if "SUPERVISOR CORRECTION" in user and agent in RECOVERY_QUERIES:  # a corrected agent obeys
        qs, step = RECOVERY_QUERIES[agent], max(0, step - 2)
    return json.dumps({"plan_summary": f"FIXTURE: next, {qs[step].removeprefix('FIXTURE: ')}.", "action": "research",
                       "query": qs[step], "rationale": "FIXTURE: serves the directive."})


def _seen(user: str) -> tuple[str, list[str], bool, int]:
    agent = user.split("## Monitored agent: ")[1].split("\n")[0]
    plans = [l.split("]")[0][1:] for l in user.splitlines() if "| plan |" in l]
    proposed = user.split("### Proposed next action")[-1].split("### Observable statistics")[0] \
        if "### Proposed next action" in user else ""
    executed = sum(any(w in l for w in DRIFT_WORDS) for l in user.splitlines() if l.strip().startswith("query: "))
    if "correction sent to the agent" in user:  # after a correction only what comes next counts
        executed = 0
    return agent, plans, any(w in proposed for w in DRIFT_WORDS), executed


def _verdict(agent, status, conf, evidence, why) -> str:
    return json.dumps({"agent_id": agent, "status": status, "confidence": conf,
                       "drift_type": "goal_deviation" if status == "DRIFTING" else "none",
                       "evidence_event_ids": evidence, "explanation": f"FIXTURE: {why}"})


def primary_handler(system: str, user: str) -> str:
    """Baseline: unsure when the gaming query is only proposed; flags once one has been executed."""
    agent, plans, proposed_drift, executed = _seen(user)
    if executed:
        return _verdict(agent, "DRIFTING", 0.85, plans[-1:], "the agent has moved to consumer gaming GPUs.")
    if proposed_drift:
        return _verdict(agent, "UNCERTAIN", 0.55, plans[-1:], "the proposed query mentions gaming; watching.")
    return _verdict(agent, "ON_TRACK", 0.9, [], "queries serve the directive.")


def shadow_handler(system: str, user: str) -> str:
    """'GEPA' side: flags as soon as the off-directive query is proposed (one step earlier)."""
    agent, plans, proposed_drift, executed = _seen(user)
    if proposed_drift or executed:
        return _verdict(agent, "DRIFTING", 0.9, plans[-1:], "the proposed query targets consumer gaming GPUs, "
                                                            "which the constraints rule out.")
    return _verdict(agent, "ON_TRACK", 0.9, [], "queries serve the directive.")


def fixture_scenario() -> Scenario:
    sc = Scenario.model_validate_json(SCENARIO.read_text())
    return sc.model_copy(update={
        "task_id": f"fixture-{sc.task_id}", "swarm_objective": f"FIXTURE: {sc.swarm_objective}",
        "agents": [a.model_copy(update={"directive": f"FIXTURE: {a.directive}"}) for a in sc.agents]})


async def make_fixture_run(runs_dir: Path) -> SwarmRun:
    """Write one offline run (with decisions_shadow.jsonl) under runs_dir. No network, no spend."""
    gepa = GEPA_PROMPT if GEPA_PROMPT.exists() else FALLBACK_GEPA_PROMPT
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=YOUCOM_RESPONSE)))
    research = FinanceResearchClient(api_key="fixture", cache_dir=Path(runs_dir) / ".fixture-cache",
                                     http_client=http, max_live_calls=50)
    sup = lambda handler, path: Supervisor.from_text(ScriptedLLM(handler, "FIXTURE/scripted-llm"),
                                                     load_prompt(path)[0], version=f"FIXTURE:{path.stem}")
    swarm = Swarm(fixture_scenario(), worker_llm=ScriptedLLM(worker_handler, "FIXTURE/scripted-llm"),
                  supervisor=sup(primary_handler, ROOT / "prompts" / "supervisor_v0.txt"),
                  shadow_supervisor=sup(shadow_handler, gepa), research=research,
                  prompts_dir=ROOT / "prompts" / "worker", runs_dir=runs_dir, halt_on_drift=False,
                  console=Console(file=io.StringIO()), verbose_events=False)
    try:
        return await swarm.run_all()
    finally:
        await http.aclose()


def report_handler(system: str, user: str) -> str:
    """Team report agent: cites the first findings it is given."""
    tags = re.findall(r"\[(F\d+)\]", user)[:3]
    cite = f" [{', '.join(tags)}]" if tags else ""
    body = (f"## Executive summary\nFIXTURE: synthetic team report, not real financial data.{cite}\n"
            "## Comparison\nFIXTURE: no real comparison.\n## Gaps and caveats\nFIXTURE: everything.")
    return json.dumps({"title": "FIXTURE team report", "report_markdown": body, "gaps": "FIXTURE"})


async def make_fixture_pair(runs_dir: Path) -> Pair:
    """Write one offline pair under runs_dir: left = original prompt, right = "GEPA", both in charge of their
    own swarm (correcting drift) with a team report each. No network, no spend."""
    gepa = GEPA_PROMPT if GEPA_PROMPT.exists() else FALLBACK_GEPA_PROMPT
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=YOUCOM_RESPONSE)))
    research = FinanceResearchClient(api_key="fixture", cache_dir=Path(runs_dir) / ".fixture-cache",
                                     http_client=http, max_live_calls=50)
    llm = lambda handler: ScriptedLLM(handler, "FIXTURE/scripted-llm")
    judges = {"left": (primary_handler, ROOT / "prompts" / "supervisor_v0.txt"), "right": (shadow_handler, gepa)}

    def side(name: str, pair_dir: Path) -> Swarm:
        handler, path = judges[name]
        return Swarm(fixture_scenario(), worker_llm=llm(worker_handler),
                     supervisor=Supervisor.from_text(llm(handler), load_prompt(path)[0], version=f"FIXTURE:{path.stem}"),
                     research=research, prompts_dir=ROOT / "prompts" / "worker", runs_dir=pair_dir,
                     halt_on_drift=False, correct_on_drift=True, max_corrections=LIVE_MAX_CORRECTIONS,
                     synthesizer=ReportSynthesizer(llm(report_handler), ROOT / "prompts" / "worker", backoff_s=0.0),
                     console=Console(file=io.StringIO()), verbose_events=False)

    pair = Pair.create(runs_dir, side)
    try:
        await pair.run_all()
    finally:
        await http.aclose()
    return pair
