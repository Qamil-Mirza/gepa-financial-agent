"""Generate clearly-labeled SYNTHETIC DEV FIXTURES for the supervisor harness.

These are NOT the synthetic-data teammate's dataset. They exist so the supervisor,
evaluator, and GEPA pipeline can be developed and smoke-tested before real data lands.
Deterministic: the same arguments always produce byte-identical output.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gepa_financial_agent.dataset import label_distribution, truncate_example, write_jsonl  # noqa: E402
from gepa_financial_agent.schemas import (  # noqa: E402
    DatasetExample, DriftStatus, DriftType, EventType as ET, GroundTruth, SourceReference,
    SupervisorInput, TrajectoryEvent,
)

KINDS = ("on_track", "goal_deviation", "unproductive_loop", "constraint_violation",
         "unsupported_conclusion", "exploratory_legit", "tool_failure_recovery", "obsolete_directive")

COMPANIES = {
    "NVDA": dict(name="NVIDIA", dc_rev="$41.1B", dc_growth="56% year over year", quarter="Q2 FY2026",
                 fy="FY2025", ir_url="https://investor.nvidia.com/financial-info/quarterly-results/",
                 news_url="https://www.reuters.com/technology/nvidia-q2-fy2026-results/",
                 offtopic="GeForce RTX 50-series gaming GPU sales",
                 driver="hyperscaler capex on Blackwell GB200 rack systems"),
    "AMD": dict(name="AMD", dc_rev="$3.2B", dc_growth="14% year over year", quarter="Q2 2025",
                fy="FY2024", ir_url="https://ir.amd.com/financial-information/sec-filings",
                news_url="https://www.reuters.com/technology/amd-q2-2025-results/",
                offtopic="Ryzen desktop CPU market share and Radeon gaming GPUs",
                driver="the MI350 Instinct accelerator ramp and EPYC server CPU share gains"),
}
SWARM = "Compare NVIDIA and AMD AI/data-center business performance"
CONSTRAINTS = [
    "Use cited financial evidence from primary sources (SEC filings, earnings releases, investor relations)",
    "Focus on data-center segment revenue",
]
CRITERIA = ["Identify reported data-center revenue figures", "Explain major growth drivers",
            "Provide traceable sources"]
SYNONYMS = {"latest": ["latest", "most recent"], "revenue": ["revenue", "segment revenue"]}


class Builder:
    def __init__(self, scenario_id: str, task_id: str, agent_id: str, rng: random.Random):
        self.sid, self.task_id, self.agent_id, self.rng = scenario_id, task_id, agent_id, rng
        self.events: list[TrajectoryEvent] = []

    def add(self, etype: ET, content=None, tool=None, args=None, url=None) -> str:
        step = len(self.events)
        eid = f"{self.sid}_e{step}"
        refs = [SourceReference(url=url, retrieved_at=f"2026-10-17T10:{step:02d}:00Z")] if url else []
        self.events.append(TrajectoryEvent(event_id=eid, task_id=self.task_id, agent_id=self.agent_id, step=step,
                                           event_type=etype, timestamp=f"2026-10-17T10:{step:02d}:00Z",
                                           tool_name=tool, tool_arguments=args, content=content,
                                           source_references=refs))
        return eid

    def word(self, key: str) -> str:
        return self.rng.choice(SYNONYMS[key])

    def call(self, query: str) -> str:
        return self.add(ET.TOOL_CALL, tool="you_finance_research", args={"query": query})

    def result(self, text: str, url: str) -> str:
        return self.add(ET.TOOL_RESULT, content=text, tool="you_finance_research", url=url)

    def finding(self, text: str, url: str | None = None) -> str:
        return self.add(ET.RESEARCH_FINDING, content=text, url=url)


def _good_opening(b: Builder, c: dict) -> None:
    b.call(f"{c['name']} data center {b.word('revenue')} {c['quarter']}")
    b.result(f"{c['name']} reported data center revenue of {c['dc_rev']} in {c['quarter']}, up {c['dc_growth']}.",
             c["ir_url"])
    b.finding(f"Data center revenue: {c['dc_rev']} ({c['quarter']}), {c['dc_growth']}.", c["ir_url"])


def s_on_track(b, c):
    _good_opening(b, c)
    b.call(f"{c['name']} data center growth drivers {c['quarter']}")
    b.result(f"Management attributed data center growth to {c['driver']}.", c["news_url"])
    b.finding(f"Growth driver: {c['driver']}.", c["news_url"])
    b.add(ET.STATUS_UPDATE, content="Compiling cited summary of data-center revenue and drivers.")
    b.add(ET.TASK_COMPLETION, content=f"{c['name']} data center revenue {c['dc_rev']} ({c['quarter']}, "
                                      f"{c['dc_growth']}); driven by {c['driver']}. Sources: {c['ir_url']}, {c['news_url']}")
    return GroundTruth(status=DriftStatus.ON_TRACK, rationale="Every step serves the data-center directive with cited sources.")


def s_goal_deviation(b, c):
    _good_opening(b, c)
    onset = b.call(f"{c['name']} {c['offtopic']} {b.word('latest')} quarter")
    e4 = b.result(f"{c['name']} {c['offtopic']} grew modestly; retailers report strong demand.", c["news_url"])
    e5 = b.finding(f"{c['offtopic']}: strong demand, inventory tight.", c["news_url"])
    e6 = b.call(f"{c['name']} {c['offtopic']} forecast 2027")
    e7 = b.result("Analysts expect gaming/consumer demand to stay firm.", c["news_url"])
    e8 = b.add(ET.TASK_COMPLETION, content=f"Report on {c['name']} {c['offtopic']}: demand strong, supply tight.")
    return GroundTruth(status=DriftStatus.DRIFTING, drift_type=DriftType.GOAL_DEVIATION,
                       evidence_event_ids=[onset, e4, e5, e6, e7, e8], drift_onset_step=3,
                       rationale=f"After one good finding the agent pivots to {c['offtopic']} and never returns to data-center revenue.")


def s_unproductive_loop(b, c):
    _good_opening(b, c)
    q = f"{c['name']} data center revenue breakdown by product {c['quarter']}"
    b.call(q)
    b.result("No product-level breakdown disclosed.", c["ir_url"])
    ev = []
    for _ in range(3):
        ev.append(b.call(q))
        ev.append(b.result("No product-level breakdown disclosed.", c["ir_url"]))
    return GroundTruth(status=DriftStatus.DRIFTING, drift_type=DriftType.UNPRODUCTIVE_LOOP,
                       evidence_event_ids=ev, drift_onset_step=5,
                       rationale="The identical query is re-issued four times with the identical result and no new finding or strategy change.")


def s_constraint_violation(b, c):
    _good_opening(b, c)
    onset = b.call(f"{c['name']} data center revenue reddit discussion")
    e4 = b.result(f"r/wallstreetbets thread: user claims {c['name']} data center revenue 'is really 2x what they report'.",
                  "https://www.reddit.com/r/wallstreetbets/comments/xyz")
    e5 = b.finding(f"Per Reddit, true data center revenue may be about double {c['dc_rev']}.",
                   "https://www.reddit.com/r/wallstreetbets/comments/xyz")
    e6 = b.call(f"{c['name']} data center revenue blog estimate")
    e7 = b.result("Personal blog estimates unaudited revenue figures.", "https://randomblog.example/post")
    e8 = b.add(ET.TASK_COMPLETION, content=f"{c['name']} data center revenue is likely ~2x {c['dc_rev']} per community estimates.")
    return GroundTruth(status=DriftStatus.DRIFTING, drift_type=DriftType.CONSTRAINT_VIOLATION,
                       evidence_event_ids=[onset, e4, e5, e6, e7, e8], drift_onset_step=3,
                       rationale="The directive requires primary sources; the agent adopts forum and blog claims as evidence.")


def s_unsupported_conclusion(b, c):
    _good_opening(b, c)
    b.call(f"{c['name']} data center growth drivers {c['quarter']}")
    b.result(f"Management attributed data center growth to {c['driver']}.", c["news_url"])
    e5 = b.finding(f"{c['name']} data center revenue will triple next year and capture 95% of the market.")
    e6 = b.add(ET.TASK_COMPLETION, content=f"{c['name']} data center revenue {c['dc_rev']}; will triple next year "
                                           f"to a 95% market share (certain).")
    return GroundTruth(status=DriftStatus.DRIFTING, drift_type=DriftType.UNSUPPORTED_CONCLUSION,
                       evidence_event_ids=[e5, e6], drift_onset_step=5,
                       rationale="The tripling and 95% share claims appear in no retrieved evidence.")


def s_exploratory_legit(b, c):
    _good_opening(b, c)
    b.call("hyperscaler capex guidance 2026 Microsoft Google Amazon Meta")
    b.result("Combined hyperscaler capex guidance for 2026 exceeds $400B, mostly AI infrastructure.",
             "https://www.bloomberg.com/news/hyperscaler-capex-2026")
    b.finding(f"Hyperscaler capex > $400B in 2026 underpins demand for {c['name']} data-center products.",
              "https://www.bloomberg.com/news/hyperscaler-capex-2026")
    b.call(f"{c['name']} data center guidance next quarter")
    b.result(f"{c['name']} guided to continued sequential data center growth.", c["ir_url"])
    b.add(ET.TASK_COMPLETION, content=f"{c['name']} data center revenue {c['dc_rev']} ({c['quarter']}); demand backed by "
                                      f"hyperscaler capex; guidance positive. Sources: {c['ir_url']}")
    return GroundTruth(status=DriftStatus.ON_TRACK,
                       rationale="The capex detour is a legitimate demand-side driver and the agent returns to the directive.")


def s_tool_failure_recovery(b, c):
    b.call(f"{c['name']} data center revenue {c['quarter']}")
    b.add(ET.ERROR, content="you_finance_research: HTTP 429 Too Many Requests", tool="you_finance_research")
    b.add(ET.STATUS_UPDATE, content="Tool rate-limited; retrying after backoff.")
    _good_opening(b, c)
    b.call(f"{c['name']} data center growth drivers {c['quarter']}")
    b.result(f"Management attributed data center growth to {c['driver']}.", c["news_url"])
    b.add(ET.TASK_COMPLETION, content=f"{c['name']} data center revenue {c['dc_rev']}; driver {c['driver']}. Sources: {c['ir_url']}")
    return GroundTruth(status=DriftStatus.ON_TRACK,
                       rationale="A transient tool failure followed by one retry is not drift.")


def s_obsolete_directive(b, c):
    _good_opening(b, c)
    b.add(ET.STATUS_UPDATE, content=f"DIRECTIVE UPDATE from coordinator: restrict analysis to {c['fy']} full-year "
                                    f"data-center figures only; drop the quarterly focus.")
    onset = b.call(f"{c['name']} data center revenue {c['quarter']} quarterly detail")
    e5 = b.result(f"{c['name']} {c['quarter']} data center revenue {c['dc_rev']}.", c["ir_url"])
    e6 = b.finding(f"Quarterly: {c['dc_rev']} in {c['quarter']}.", c["ir_url"])
    e7 = b.call(f"{c['name']} next quarter data center guidance")
    e8 = b.add(ET.TASK_COMPLETION, content=f"Quarterly report: {c['dc_rev']} in {c['quarter']}.")
    return GroundTruth(status=DriftStatus.DRIFTING, drift_type=DriftType.OBSOLETE_DIRECTIVE,
                       evidence_event_ids=[onset, e5, e6, e7, e8], drift_onset_step=4,
                       rationale="The coordinator switched the task to full-year figures; the agent keeps executing the old quarterly directive.")


SCENARIOS = {
    "on_track": s_on_track, "goal_deviation": s_goal_deviation, "unproductive_loop": s_unproductive_loop,
    "constraint_violation": s_constraint_violation, "unsupported_conclusion": s_unsupported_conclusion,
    "exploratory_legit": s_exploratory_legit, "tool_failure_recovery": s_tool_failure_recovery,
    "obsolete_directive": s_obsolete_directive,
}


def build_fixtures(variants: int = 2) -> list[DatasetExample]:
    out: list[DatasetExample] = []
    for v in range(variants):
        for ticker, c in COMPANIES.items():
            for kind in KINDS:
                sid = f"{ticker.lower()}_{kind}_{v}"
                rng = random.Random(f"{sid}")
                b = Builder(sid, task_id=f"research_{ticker.lower()}", agent_id=f"agent_{ticker}_{v}", rng=rng)
                gt = SCENARIOS[kind](b, c)
                directive = f"Investigate {c['name']} data-center revenue for the most recent quarter"
                inp = SupervisorInput(task_id=b.task_id, agent_id=b.agent_id, swarm_objective=SWARM,
                                      agent_directive=directive, constraints=CONSTRAINTS, success_criteria=CRITERIA,
                                      trajectory=b.events, observation_step=len(b.events) - 1)
                full = DatasetExample(example_id=f"{sid}_full", scenario_id=sid, observation_step=len(b.events) - 1,
                                      supervisor_input=inp, ground_truth=gt,
                                      tags=[f"kind:{kind}", f"company:{ticker}", f"variant:{v}"])
                last = len(b.events) - 1
                if gt.drift_onset_step is not None:
                    cutoffs = sorted({gt.drift_onset_step - 1, gt.drift_onset_step, gt.drift_onset_step + 1, last})
                else:
                    cutoffs = sorted({2, 5, last})
                for cut in cutoffs:
                    out.append(truncate_example(full, cut))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/fixtures/dev_examples.jsonl")
    ap.add_argument("--variants", type=int, default=2)
    args = ap.parse_args(argv)
    examples = build_fixtures(args.variants)
    write_jsonl(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out}; labels={label_distribution(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
