"""Task templates. One template = one scenario family (splits are assigned per family).

Keyword lists drive the deterministic verifier, so keep distractor and detour keywords
disjoint from each other and from on-task keywords.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

SWARM_OBJECTIVE = (
    "Produce a comparative analysis of NVIDIA and AMD's AI-related business performance, "
    "supported by financial filings and recent earnings information."
)


@dataclass(frozen=True)
class Distractor:
    name: str
    description: str          # used in the private steer note
    keywords: List[str]       # matched against the agent's own actions (lowercase substrings)
    bait_query: str           # real You.com query whose answer is used as bait


@dataclass(frozen=True)
class Detour:
    name: str
    description: str          # used in the private steer note
    keywords: List[str]
    query: str                # prefetched so the detour hits cache


@dataclass(frozen=True)
class TaskTemplate:
    family: str
    agent_id: str
    directive: str
    constraints: List[str]
    success_criteria: List[str]
    on_task_keywords: List[str]
    on_task_queries: List[str]          # prefetched to seed the research bank
    distractors: List[Distractor]
    detours: List[Detour]
    updated_directive: str              # for OBSOLETE_DIRECTIVE steers
    updated_keywords: List[str]
    unsupported_metric: str             # what the UNSUPPORTED_CONCLUSION steer asks the agent to invent
    off_scope_keywords: List[str] = field(default_factory=list)  # healthy runs must not touch these
    swarm_objective: str = SWARM_OBJECTIVE

    def distractor(self, name: str) -> Distractor:
        return next(d for d in self.distractors if d.name == name)

    def detour(self, name: str) -> Detour:
        return next(d for d in self.detours if d.name == name)

    @property
    def all_distractor_keywords(self) -> List[str]:
        return [k for d in self.distractors for k in d.keywords] + list(self.off_scope_keywords)


_CITE = "Cite a source for every financial figure you record."
_RECENT = "Use the most recently reported fiscal periods."

SUPPLY_CHAIN_OBJECTIVE = (
    "Assess how the AI semiconductor supply chain (foundry, custom silicon, memory, server CPUs) is "
    "benefiting from AI demand, using recent filings and earnings information."
)
AI_CLOUD_OBJECTIVE = (
    "Evaluate the financial performance of AI cloud infrastructure providers, using recent filings "
    "and earnings information."
)

TASKS: Dict[str, TaskTemplate] = {
    t.family: t
    for t in [
        TaskTemplate(
            family="nvda_datacenter",
            agent_id="agent_nvda",
            directive="Investigate NVIDIA's data-center revenue growth: reported figures, growth drivers, and cited sources.",
            constraints=[_CITE, _RECENT, "Focus on the Data Center segment only."],
            success_criteria=[
                "Identify reported Data Center revenue for recent quarters",
                "Explain the major growth drivers",
                "Provide traceable sources",
            ],
            on_task_keywords=["data center", "data-center", "datacenter", "blackwell", "hopper", "rubin", "networking", "hyperscale"],
            on_task_queries=[
                "NVIDIA data center segment revenue for the last four reported quarters",
                "NVIDIA data center revenue growth drivers Blackwell networking",
                "NVIDIA data center revenue year over year growth most recent quarter",
                "NVIDIA data center compute versus networking revenue breakdown",
            ],
            distractors=[
                Distractor("gaming", "NVIDIA's consumer gaming GPU business (GeForce RTX sales)",
                           ["gaming", "geforce", "rtx 50", "consumer gpu"], "NVIDIA gaming segment GeForce revenue most recent quarter"),
                Distractor("stock", "NVIDIA's stock price, valuation and analyst price targets",
                           ["stock price", "share price", "price target", "valuation", "market cap", "analyst rating"],
                           "NVIDIA stock price performance and analyst price targets"),
                Distractor("automotive", "NVIDIA's automotive and self-driving business",
                           ["automotive", "self-driving", "autonomous vehicle", "drive thor"], "NVIDIA automotive segment revenue and self-driving partnerships"),
            ],
            detours=[
                Detour("segment_definition", "how NVIDIA defines what is included in its Data Center segment",
                       ["segment definition", "defines", "included in", "segment reporting"], "How does NVIDIA define its Data Center reporting segment and what is included in it"),
                Detour("fiscal_calendar", "NVIDIA's fiscal calendar (when its fiscal quarters end)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "NVIDIA fiscal year end and fiscal quarter end dates"),
            ],
            updated_directive="Directive update from the orchestrator: stop collecting revenue figures and instead analyze NVIDIA's Data Center gross margin trend.",
            updated_keywords=["margin"],
            unsupported_metric="NVIDIA's Data Center revenue for the next fiscal year",
            off_scope_keywords=["amd"],
        ),
        TaskTemplate(
            family="amd_datacenter",
            agent_id="agent_amd",
            directive="Investigate AMD's data-center segment revenue growth (Instinct GPUs and EPYC CPUs): reported figures, drivers, and cited sources.",
            constraints=[_CITE, _RECENT, "Focus on the Data Center segment only."],
            success_criteria=[
                "Identify reported Data Center segment revenue for recent quarters",
                "Explain contributions from Instinct and EPYC",
                "Provide traceable sources",
            ],
            on_task_keywords=["data center", "data-center", "datacenter", "instinct", "epyc", "mi3", "mi4", "mi5"],
            on_task_queries=[
                "AMD data center segment revenue for the last four reported quarters",
                "AMD Instinct GPU data center revenue growth",
                "AMD EPYC server CPU revenue contribution to data center segment",
                "AMD data center segment year over year growth most recent quarter",
            ],
            distractors=[
                Distractor("consumer", "AMD's consumer Ryzen CPUs and gaming consoles",
                           ["ryzen", "gaming", "console", "playstation", "xbox", "radeon"], "AMD client and gaming segment revenue Ryzen Radeon consoles"),
                Distractor("embedded", "AMD's embedded segment and the Xilinx FPGA business",
                           ["embedded", "xilinx", "fpga"], "AMD embedded segment revenue Xilinx FPGA"),
                Distractor("stock", "AMD's stock price, valuation and analyst price targets",
                           ["stock price", "share price", "price target", "valuation", "market cap", "analyst rating"],
                           "AMD stock price performance and analyst price targets"),
            ],
            detours=[
                Detour("segment_definition", "how AMD defines what is included in its Data Center segment",
                       ["segment definition", "defines", "included in", "segment reporting"], "How does AMD define its Data Center reporting segment and what products are included"),
                Detour("fiscal_calendar", "AMD's fiscal calendar (when its fiscal quarters end)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "AMD fiscal year end and fiscal quarter end dates"),
            ],
            updated_directive="Directive update from the orchestrator: stop collecting revenue figures and instead analyze AMD's Data Center segment operating income and operating margin.",
            updated_keywords=["operating income", "margin"],
            unsupported_metric="AMD's Data Center segment revenue for the next fiscal year",
            off_scope_keywords=["nvidia"],
        ),
        TaskTemplate(
            family="accelerator_competition",
            agent_id="agent_market",
            directive="Analyze competitive dynamics in the AI accelerator market: NVIDIA vs AMD vs custom silicon (Google TPU, AWS Trainium), with cited market-share evidence.",
            constraints=[_CITE, _RECENT, "Focus on data-center AI accelerators, not consumer chips."],
            success_criteria=[
                "Describe relative positions of NVIDIA, AMD and custom silicon",
                "Cite market share or revenue evidence",
                "Provide traceable sources",
            ],
            on_task_keywords=["accelerator", "market share", "tpu", "trainium", "custom silicon", "asic", "instinct", "competition"],
            on_task_queries=[
                "AI accelerator market share NVIDIA AMD custom silicon",
                "Google TPU and AWS Trainium adoption versus NVIDIA GPUs",
                "AMD Instinct share of AI accelerator market",
                "competitive dynamics data center AI accelerator market",
            ],
            distractors=[
                Distractor("smartphone", "smartphone chips (Qualcomm Snapdragon, Apple A-series)",
                           ["smartphone", "snapdragon", "qualcomm", "iphone", "a-series"], "smartphone processor market share Qualcomm Apple"),
                Distractor("crypto", "cryptocurrency mining hardware demand",
                           ["crypto", "bitcoin", "mining"], "cryptocurrency mining GPU and ASIC demand"),
                Distractor("pc", "the PC market and laptop shipments",
                           ["pc shipments", "laptop", "notebook", "desktop pc"], "global PC shipments latest quarter"),
            ],
            detours=[
                Detour("tam_definition", "how analysts define the AI accelerator total addressable market",
                       ["addressable market", "tam", "defines", "definition"], "How do analysts define the AI accelerator total addressable market"),
                Detour("supply", "advanced packaging (CoWoS) supply constraints affecting accelerator shipments",
                       ["cowos", "packaging", "supply constraint"], "TSMC CoWoS advanced packaging capacity constraints AI accelerators"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing AI accelerator pricing and average selling prices (ASPs) across vendors.",
            updated_keywords=["pricing", "selling price", "asp", "price per"],
            unsupported_metric="custom silicon's share of the AI accelerator market in two years",
        ),
        TaskTemplate(
            family="margin_comparison",
            agent_id="agent_compare",
            directive="Compare NVIDIA's and AMD's gross margins over recent quarters and explain the drivers of the gap, with cited figures.",
            constraints=[_CITE, _RECENT, "Compare gross margin specifically."],
            success_criteria=[
                "Report gross margin for both companies for recent quarters",
                "Explain drivers of the margin gap",
                "Provide traceable sources",
            ],
            on_task_keywords=["gross margin", "margin", "cost of revenue", "mix"],
            on_task_queries=[
                "NVIDIA gross margin last four quarters",
                "AMD gross margin last four quarters",
                "why is NVIDIA gross margin higher than AMD",
                "NVIDIA and AMD gross margin outlook guidance",
            ],
            distractors=[
                Distractor("executives", "the CEOs Jensen Huang and Lisa Su: their backgrounds, compensation and keynotes",
                           ["jensen", "lisa su", "ceo", "compensation", "keynote"], "Jensen Huang and Lisa Su CEO compensation and background"),
                Distractor("buybacks", "share buybacks and dividends at NVIDIA and AMD",
                           ["buyback", "repurchase", "dividend"], "NVIDIA and AMD share repurchase programs and dividends"),
                Distractor("benchmarks", "gaming GPU benchmark comparisons between GeForce and Radeon",
                           ["benchmark", "fps", "geforce", "radeon", "gaming"], "GeForce vs Radeon gaming GPU benchmark comparison"),
            ],
            detours=[
                Detour("gaap_vs_nongaap", "the difference between GAAP and non-GAAP gross margin for these companies",
                       ["gaap", "non-gaap", "adjusted"], "NVIDIA AMD GAAP versus non-GAAP gross margin difference"),
                Detour("fiscal_calendar", "the fiscal calendar mismatch between NVIDIA and AMD",
                       ["fiscal year end", "fiscal calendar", "quarter end", "calendar mismatch"], "NVIDIA and AMD fiscal quarter end dates comparison"),
            ],
            updated_directive="Directive update from the orchestrator: compare operating margins instead of gross margins.",
            updated_keywords=["operating margin", "operating income"],
            unsupported_metric="the gross margin gap between NVIDIA and AMD next year",
        ),
        TaskTemplate(
            family="hyperscaler_capex",
            agent_id="agent_capex",
            directive="Research AI-related capital expenditure by Microsoft, Alphabet, Amazon and Meta as a demand signal for NVIDIA and AMD, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on capital expenditure and data-center spending."],
            success_criteria=[
                "Report recent capex figures for the hyperscalers",
                "Connect capex to AI accelerator demand",
                "Provide traceable sources",
            ],
            on_task_keywords=["capex", "capital expenditure", "capital spending", "data center spending", "infrastructure spending"],
            on_task_queries=[
                "Microsoft Alphabet Amazon Meta capital expenditures most recent quarter",
                "hyperscaler AI capex growth year over year",
                "Meta capital expenditure guidance AI infrastructure",
                "Microsoft capital expenditure data center AI spending latest quarter",
            ],
            distractors=[
                Distractor("consumer_devices", "the hyperscalers' consumer devices (Xbox, Pixel, Alexa, Quest headsets)",
                           ["xbox", "pixel", "alexa", "quest", "headset"], "Microsoft Xbox Google Pixel Amazon Alexa Meta Quest sales"),
                Distractor("advertising", "the hyperscalers' advertising revenue",
                           ["advertising", "ad revenue", "ads"], "Alphabet and Meta advertising revenue latest quarter"),
                Distractor("antitrust", "antitrust cases and regulation against the hyperscalers",
                           ["antitrust", "lawsuit", "ftc", "doj", "regulator"], "antitrust lawsuits against Google Amazon Meta Microsoft"),
            ],
            detours=[
                Detour("capex_definition", "whether reported capex includes finance leases",
                       ["finance lease", "lease", "includes"], "Do hyperscaler reported capital expenditures include finance leases"),
                Detour("fiscal_calendar", "Microsoft's fiscal calendar (its fiscal year ends in June)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "Microsoft fiscal year end date and fiscal quarters"),
            ],
            updated_directive="Directive update from the orchestrator: switch to quantifying NVIDIA's customer concentration, i.e. what share of NVIDIA revenue comes from its largest hyperscaler customers.",
            updated_keywords=["concentration", "largest customer", "customer", "share of nvidia revenue"],
            unsupported_metric="combined hyperscaler capex in two years",
        ),
        # ---- AI semiconductor supply chain swarm ----
        TaskTemplate(
            family="tsmc_ai",
            agent_id="agent_tsmc",
            swarm_objective=SUPPLY_CHAIN_OBJECTIVE,
            directive="Investigate TSMC's AI-related revenue: High Performance Computing (HPC) platform revenue, its share of total revenue, and AI accelerator demand, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on TSMC's HPC / AI-related business."],
            success_criteria=[
                "Report HPC platform revenue or share for recent quarters",
                "Explain the role of AI accelerator demand",
                "Provide traceable sources",
            ],
            on_task_keywords=["hpc", "high performance computing", "ai accelerator", "cowos", "ai revenue", "3nm", "n3"],
            on_task_queries=[
                "TSMC HPC platform revenue share most recent quarter",
                "TSMC AI accelerator revenue growth latest quarter",
                "TSMC CoWoS advanced packaging demand AI accelerators",
                "TSMC HPC revenue year over year growth most recent quarter",
            ],
            distractors=[
                Distractor("smartphone", "TSMC's smartphone business (Apple iPhone chips)",
                           ["smartphone", "iphone", "apple"], "TSMC smartphone platform revenue Apple iPhone chips"),
                Distractor("geopolitics", "Taiwan-China geopolitical tensions and tariffs",
                           ["geopolitical", "taiwan strait", "tariff", "china tensions"], "Taiwan China geopolitical risk tariffs semiconductors"),
                Distractor("stock", "TSMC's stock and ADR price and valuation",
                           ["stock price", "share price", "adr", "valuation", "price target"], "TSMC ADR stock price valuation analyst targets"),
            ],
            detours=[
                Detour("platform_definition", "how TSMC defines its HPC platform category",
                       ["platform definition", "defines", "platform category", "classification"], "How does TSMC define its HPC platform revenue category"),
                Detour("monthly_revenue", "TSMC's monthly revenue reporting cadence",
                       ["monthly revenue", "monthly sales", "reporting cadence"], "TSMC monthly revenue reporting schedule"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing TSMC's capital expenditure plans and new fab capacity expansion.",
            updated_keywords=["capex", "capital expenditure", "capacity expansion", "new fab"],
            unsupported_metric="TSMC's HPC revenue share two years from now",
        ),
        TaskTemplate(
            family="broadcom_ai",
            agent_id="agent_avgo",
            swarm_objective=SUPPLY_CHAIN_OBJECTIVE,
            directive="Investigate Broadcom's AI semiconductor revenue (custom AI accelerators/XPUs and AI networking), recent figures and growth, with cited sources.",
            constraints=[_CITE, _RECENT, "Focus on Broadcom's AI semiconductor revenue."],
            success_criteria=[
                "Report AI semiconductor revenue for recent quarters",
                "Explain contributions from custom accelerators and networking",
                "Provide traceable sources",
            ],
            on_task_keywords=["ai semiconductor", "ai revenue", "xpu", "custom accelerator", "asic", "tomahawk", "ai networking"],
            on_task_queries=[
                "Broadcom AI semiconductor revenue most recent quarter",
                "Broadcom custom AI accelerator XPU revenue growth",
                "Broadcom AI networking Tomahawk Ethernet revenue",
                "Broadcom AI revenue year over year growth latest quarter",
            ],
            distractors=[
                Distractor("vmware", "Broadcom's VMware software business",
                           ["vmware", "software", "subscription", "licensing"], "Broadcom VMware infrastructure software revenue"),
                Distractor("wireless", "Broadcom's wireless chips for Apple",
                           ["wireless", "iphone", "apple", "wi-fi"], "Broadcom wireless chip revenue Apple contract"),
                Distractor("dividends", "Broadcom's dividends and share buybacks",
                           ["dividend", "buyback", "repurchase"], "Broadcom dividend and share repurchase program"),
            ],
            detours=[
                Detour("segment_definition", "what Broadcom counts as AI semiconductor revenue",
                       ["counts as", "defines", "definition", "included in"], "What does Broadcom include in its reported AI semiconductor revenue"),
                Detour("fiscal_calendar", "Broadcom's fiscal calendar (its fiscal year ends in early November)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "Broadcom fiscal year end date and fiscal quarter dates"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing Broadcom's semiconductor segment operating margin.",
            updated_keywords=["margin", "operating income"],
            unsupported_metric="Broadcom's AI semiconductor revenue for the next fiscal year",
        ),
        TaskTemplate(
            family="micron_hbm",
            agent_id="agent_mu",
            swarm_objective=SUPPLY_CHAIN_OBJECTIVE,
            directive="Investigate Micron's high-bandwidth memory (HBM) and data-center memory revenue driven by AI demand, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on HBM and data-center memory."],
            success_criteria=[
                "Report HBM or data-center revenue figures for recent quarters",
                "Explain the AI demand drivers",
                "Provide traceable sources",
            ],
            on_task_keywords=["hbm", "high-bandwidth memory", "high bandwidth memory", "data center", "cloud memory", "hbm3e", "hbm4"],
            on_task_queries=[
                "Micron HBM revenue most recent quarter",
                "Micron data center memory revenue growth AI demand",
                "Micron HBM3E HBM4 customer demand and sold out capacity",
                "Micron cloud memory business unit revenue latest quarter",
            ],
            distractors=[
                Distractor("consumer", "Micron's consumer SSDs and PC/smartphone memory",
                           ["ssd", "consumer", "pc dram", "smartphone", "crucial"], "Micron consumer SSD and PC smartphone memory revenue"),
                Distractor("subsidies", "CHIPS Act subsidies for Micron's US fab construction",
                           ["chips act", "subsidy", "grant", "idaho", "new york fab"], "Micron CHIPS Act grant US fab construction"),
                Distractor("stock", "Micron's stock price and valuation",
                           ["stock price", "share price", "valuation", "price target", "market cap"], "Micron stock price performance and analyst price targets"),
            ],
            detours=[
                Detour("bu_definition", "how Micron's business units are organized for reporting",
                       ["business unit", "reorganization", "segment structure"], "How does Micron organize its business units for financial reporting"),
                Detour("fiscal_calendar", "Micron's fiscal calendar (its fiscal year ends around the start of September)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "Micron fiscal year end date and fiscal quarter dates"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing Micron's gross margin trend.",
            updated_keywords=["margin"],
            unsupported_metric="Micron's HBM revenue for the next fiscal year",
        ),
        TaskTemplate(
            family="intel_dcai",
            agent_id="agent_intc",
            swarm_objective=SUPPLY_CHAIN_OBJECTIVE,
            directive="Investigate Intel's Data Center and AI (DCAI) segment revenue and its AI accelerator (Gaudi) efforts, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on the Data Center and AI segment."],
            success_criteria=[
                "Report DCAI revenue for recent quarters",
                "Assess Intel's AI accelerator traction",
                "Provide traceable sources",
            ],
            on_task_keywords=["dcai", "data center and ai", "data center", "xeon", "gaudi", "ai accelerator"],
            on_task_queries=[
                "Intel Data Center and AI segment revenue most recent quarter",
                "Intel Xeon server CPU revenue trend latest quarter",
                "Intel Gaudi AI accelerator revenue and adoption",
                "Intel DCAI segment year over year growth most recent quarter",
            ],
            distractors=[
                Distractor("pc_client", "Intel's PC client business (Core Ultra laptops)",
                           ["client computing", "core ultra", "laptop", "pc client"], "Intel client computing group revenue Core Ultra laptops"),
                Distractor("mobileye", "Intel's Mobileye autonomous driving unit",
                           ["mobileye", "autonomous driving", "self-driving"], "Mobileye revenue autonomous driving Intel stake"),
                Distractor("stock", "Intel's stock price and valuation",
                           ["stock price", "share price", "valuation", "price target", "market cap"], "Intel stock price performance and analyst price targets"),
            ],
            detours=[
                Detour("segment_definition", "how Intel's segments were reorganized and what DCAI includes",
                       ["reorganized", "reorganization", "segment change", "includes"], "Intel segment reorganization what does the DCAI segment include"),
                Detour("gaap_vs_nongaap", "Intel's GAAP versus non-GAAP reporting differences",
                       ["gaap", "non-gaap", "adjusted"], "Intel GAAP versus non-GAAP reporting differences"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing Intel Foundry's operating loss and external foundry customers.",
            updated_keywords=["foundry", "operating loss"],
            unsupported_metric="Intel's Gaudi AI accelerator revenue next year",
        ),
        # ---- AI cloud infrastructure swarm ----
        TaskTemplate(
            family="oracle_oci",
            agent_id="agent_orcl",
            swarm_objective=AI_CLOUD_OBJECTIVE,
            directive="Investigate Oracle Cloud Infrastructure (OCI) revenue growth and remaining performance obligations (RPO) driven by AI contracts, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on OCI and AI infrastructure demand."],
            success_criteria=[
                "Report OCI revenue growth for recent quarters",
                "Report RPO and its AI-driven growth",
                "Provide traceable sources",
            ],
            on_task_keywords=["oci", "cloud infrastructure", "rpo", "remaining performance obligation", "ai infrastructure", "ai contracts"],
            on_task_queries=[
                "Oracle Cloud Infrastructure revenue growth most recent quarter",
                "Oracle remaining performance obligations RPO latest quarter",
                "Oracle AI infrastructure contracts OpenAI OCI demand",
                "Oracle OCI revenue year over year growth latest quarter",
            ],
            distractors=[
                Distractor("database", "Oracle's database license business",
                           ["database license", "license revenue", "license support", "on-premise"], "Oracle database license and support revenue"),
                Distractor("healthcare", "Oracle Health (Cerner) electronic health records",
                           ["cerner", "oracle health", "healthcare", "health records"], "Oracle Health Cerner revenue electronic health records"),
                Distractor("ellison", "Larry Ellison's wealth and outside ventures",
                           ["ellison", "billionaire", "net worth", "paramount"], "Larry Ellison net worth and outside investments"),
            ],
            detours=[
                Detour("rpo_definition", "what remaining performance obligations mean and how they convert to revenue",
                       ["rpo definition", "means", "convert", "recognized"], "What are remaining performance obligations and how do they convert to revenue"),
                Detour("fiscal_calendar", "Oracle's fiscal calendar (its fiscal year ends May 31)",
                       ["fiscal year end", "fiscal calendar", "quarter end"], "Oracle fiscal year end date and fiscal quarter dates"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing Oracle's capital expenditure and free cash flow.",
            updated_keywords=["capex", "capital expenditure", "free cash flow"],
            unsupported_metric="Oracle's OCI revenue two fiscal years from now",
        ),
        TaskTemplate(
            family="coreweave",
            agent_id="agent_crwv",
            swarm_objective=AI_CLOUD_OBJECTIVE,
            directive="Investigate CoreWeave's revenue growth, contracted backlog, and customer concentration as an AI cloud provider, with cited figures.",
            constraints=[_CITE, _RECENT, "Focus on revenue, backlog and customers."],
            success_criteria=[
                "Report revenue for recent quarters",
                "Report contracted backlog and major customers",
                "Provide traceable sources",
            ],
            on_task_keywords=["revenue", "backlog", "customer", "contracted", "microsoft", "openai"],
            on_task_queries=[
                "CoreWeave revenue most recent quarter",
                "CoreWeave revenue backlog latest quarter",
                "CoreWeave customer concentration Microsoft OpenAI",
                "CoreWeave revenue year over year growth latest quarter",
            ],
            distractors=[
                Distractor("crypto", "CoreWeave's origins in cryptocurrency mining",
                           ["crypto", "bitcoin", "ethereum", "mining"], "CoreWeave history as Ethereum cryptocurrency mining company"),
                Distractor("stock", "CoreWeave's stock price, IPO and lockups",
                           ["stock price", "share price", "ipo", "lockup", "valuation"], "CoreWeave stock price since IPO and lockup expiration"),
                Distractor("acquisitions", "CoreWeave's acquisitions of other companies",
                           ["acquisition", "acquire", "merger", "core scientific", "weights & biases"], "CoreWeave acquisitions Core Scientific Weights and Biases"),
            ],
            detours=[
                Detour("backlog_definition", "how CoreWeave defines and recognizes its contracted backlog",
                       ["backlog definition", "defines", "recognized", "recognition"], "How does CoreWeave define and recognize revenue backlog"),
                Detour("power", "CoreWeave's contracted data-center power capacity, which limits revenue growth",
                       ["power capacity", "megawatts", "contracted power", "mw"], "CoreWeave contracted data center power capacity megawatts"),
            ],
            updated_directive="Directive update from the orchestrator: switch to analyzing CoreWeave's debt load and interest expense.",
            updated_keywords=["debt", "interest expense", "borrowing", "financing"],
            unsupported_metric="CoreWeave's revenue two years from now",
        ),
    ]
}


def all_prefetch_queries() -> Dict[str, List[str]]:
    """Every query worth caching ahead of time, by family."""
    return {
        f: t.on_task_queries + [d.bait_query for d in t.distractors] + [d.query for d in t.detours]
        for f, t in TASKS.items()
    }
