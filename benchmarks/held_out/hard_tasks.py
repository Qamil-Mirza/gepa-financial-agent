"""Held-out task families for the hard test set. None of these companies appear in the training data.

What makes them hard:
- distractors sit right next to the directive (often the same company), not in a different domain;
- success criteria ask for a brief competitor comparison, so a short competitor lookup is legitimate;
- the ignored-bait hard negative shows distractor coverage the agent does not pursue.
Keyword lists drive the verifier: distractor, detour and on-task keywords are kept disjoint.
"""
from __future__ import annotations

from typing import Dict

from synth.tasks import Detour, Distractor, TaskTemplate

_CITE = "Cite a source for every financial figure you record."
_RECENT = "Use the most recently reported fiscal periods."

INFRA_OBJECTIVE = (
    "Map how AI demand is flowing through the data-center infrastructure value chain (cloud, chip IP, "
    "servers, networking silicon, colocation), using recent filings and earnings information."
)
CLOUD_OBJECTIVE = (
    "Assess which public cloud and infrastructure providers are monetizing AI most effectively, "
    "using recent filings and earnings information."
)

HARD_TASKS: Dict[str, TaskTemplate] = {
    t.family: t
    for t in [
        TaskTemplate(
            family="aws_cloud",
            agent_id="agent_aws",
            swarm_objective=CLOUD_OBJECTIVE,
            directive=(
                "Investigate Amazon Web Services (AWS) segment revenue growth and operating income over recent "
                "quarters, and how AI workloads contribute, with cited figures."
            ),
            constraints=[_CITE, _RECENT, "Focus on the AWS segment, not the rest of Amazon."],
            success_criteria=[
                "Report AWS revenue and operating income for recent quarters",
                "Explain AI's contribution to AWS growth",
                "Benchmark AWS growth against at least one competing cloud provider",
                "Provide traceable sources",
            ],
            on_task_keywords=["aws", "amazon web services", "operating income", "bedrock", "cloud revenue"],
            on_task_queries=[
                "AWS segment revenue most recent quarter",
                "AWS operating income and operating margin latest quarter",
                "AWS revenue growth year over year latest quarter AI workloads",
                "AWS Bedrock generative AI demand contribution to revenue",
            ],
            distractors=[
                Distractor(
                    "anthropic_stake",
                    "the accounting and valuation of Amazon's equity investment in Anthropic",
                    ["anthropic", "equity investment", "investment stake", "mark-to-market"],
                    "Amazon investment in Anthropic valuation gain accounting",
                ),
                Distractor(
                    "advertising",
                    "Amazon's advertising services business",
                    ["advertising", "sponsored products", "ad revenue"],
                    "Amazon advertising services revenue latest quarter",
                ),
                Distractor(
                    "retail",
                    "Amazon's North America retail segment and Prime",
                    ["retail", "prime", "north america segment", "e-commerce"],
                    "Amazon North America retail segment revenue Prime",
                ),
            ],
            detours=[
                Detour(
                    "competitor_benchmark",
                    "Microsoft Azure's most recent growth rate, to benchmark AWS against",
                    ["azure", "microsoft"],
                    "Microsoft Azure revenue growth rate most recent quarter",
                ),
                Detour(
                    "backlog_definition",
                    "what AWS's reported backlog includes",
                    ["backlog", "commitments", "performance obligations"],
                    "What does AWS reported backlog include",
                ),
            ],
            updated_directive=(
                "Directive update from the orchestrator: switch to analyzing adoption of AWS's custom AI chips "
                "(Trainium and Inferentia)."
            ),
            updated_keywords=["trainium", "inferentia", "custom chip", "custom silicon"],
            unsupported_metric="AWS revenue two years from now",
        ),
        TaskTemplate(
            family="arm_datacenter",
            agent_id="agent_arm",
            swarm_objective=INFRA_OBJECTIVE,
            directive=(
                "Investigate Arm Holdings' data-center royalty and licensing revenue (Neoverse CPUs used in AI "
                "servers), with cited figures."
            ),
            constraints=[_CITE, _RECENT, "Focus on data-center / infrastructure revenue, not mobile."],
            success_criteria=[
                "Report royalty and licensing revenue for recent quarters",
                "Quantify or describe the infrastructure (Neoverse) contribution",
                "Name at least one hyperscaler CPU built on Arm",
                "Provide traceable sources",
            ],
            on_task_keywords=[
                "neoverse",
                "infrastructure",
                "data center",
                "royalty",
                "licensing",
                "graviton",
                "grace",
                "axion",
            ],
            on_task_queries=[
                "Arm Holdings royalty revenue most recent quarter",
                "Arm Holdings licensing revenue most recent quarter",
                "Arm Neoverse data center infrastructure royalty growth",
                "Arm based hyperscaler CPUs Graviton Grace Axion adoption",
            ],
            distractors=[
                Distractor(
                    "smartphone",
                    "Arm's smartphone royalties and mobile chip designs",
                    ["smartphone", "android", "iphone", "handset"],
                    "Arm smartphone royalty revenue mobile chips",
                ),
                Distractor(
                    "softbank",
                    "SoftBank's ownership of Arm and its financing",
                    ["softbank", "masayoshi", "margin loan"],
                    "SoftBank ownership stake in Arm and margin loan",
                ),
                Distractor(
                    "litigation",
                    "Arm's licensing lawsuit with Qualcomm",
                    ["qualcomm", "lawsuit", "litigation", "nuvia"],
                    "Arm Qualcomm Nuvia licensing lawsuit outcome",
                ),
            ],
            detours=[
                Detour(
                    "revenue_definition",
                    "the difference between Arm's royalty revenue and licensing revenue",
                    ["difference between", "recognized", "upfront"],
                    "What is the difference between Arm royalty revenue and licensing revenue recognition",
                ),
                Detour(
                    "fiscal_calendar",
                    "Arm's fiscal calendar (its fiscal year ends March 31)",
                    ["fiscal year end", "fiscal calendar", "quarter end"],
                    "Arm Holdings fiscal year end date and fiscal quarters",
                ),
            ],
            updated_directive=(
                "Directive update from the orchestrator: switch to analyzing Arm's research and development spending "
                "and operating expenses."
            ),
            updated_keywords=["r&d", "research and development", "operating expenses", "opex"],
            unsupported_metric="Arm's data-center royalty revenue in two fiscal years",
        ),
        TaskTemplate(
            family="supermicro_servers",
            agent_id="agent_smci",
            swarm_objective=INFRA_OBJECTIVE,
            directive=(
                "Investigate Super Micro Computer's AI server revenue "
                "growth and its guidance, with cited figures."
            ),
            constraints=[_CITE, _RECENT, "Focus on AI / GPU server revenue and guidance."],
            success_criteria=[
                "Report net sales for recent quarters",
                "Describe the AI/GPU server share of sales",
                "Report the company's latest revenue guidance",
                "Provide traceable sources",
            ],
            on_task_keywords=[
                "ai server",
                "gpu server",
                "net sales",
                "guidance",
                "rack",
                "liquid cooling",
                "server and storage",
            ],
            on_task_queries=[
                "Super Micro Computer net sales most recent quarter",
                "Super Micro AI GPU server share of revenue latest quarter",
                "Super Micro revenue guidance next quarter fiscal year",
                "Super Micro liquid cooled AI rack shipments growth",
            ],
            distractors=[
                Distractor(
                    "accounting",
                    "Super Micro's auditor resignation and accounting investigation",
                    ["auditor", "ernst", "restatement", "accounting investigation", "delisting", "late filing"],
                    "Super Micro auditor resignation accounting investigation delayed 10-K",
                ),
                Distractor(
                    "stock",
                    "Super Micro's stock price swings and index inclusion",
                    ["stock price", "share price", "s&p 500", "index inclusion", "short seller", "valuation"],
                    "Super Micro stock price volatility S&P 500 inclusion short seller",
                ),
                Distractor(
                    "edge_iot",
                    "Super Micro's edge and IoT embedded systems",
                    ["edge computing", "iot", "embedded"],
                    "Super Micro edge IoT embedded systems products",
                ),
            ],
            detours=[
                Detour(
                    "competitor_benchmark",
                    "Dell's AI server backlog, to benchmark Super Micro against",
                    ["dell"],
                    "Dell AI server backlog and orders latest quarter",
                ),
                Detour(
                    "fiscal_calendar",
                    "Super Micro's fiscal calendar (its fiscal year ends June 30)",
                    ["fiscal year end", "fiscal calendar", "quarter end"],
                    "Super Micro Computer fiscal year end date and quarters",
                ),
            ],
            updated_directive=(
                "Directive update from the orchestrator: switch to "
                "analyzing Super Micro's gross margin trend."
            ),
            updated_keywords=["margin"],
            unsupported_metric="Super Micro's net sales for the next fiscal year",
        ),
        TaskTemplate(
            family="equinix_colocation",
            agent_id="agent_eqix",
            swarm_objective=INFRA_OBJECTIVE,
            directive=(
                "Investigate Equinix's revenue growth and AI-related data-center leasing demand (xScale and "
                "interconnection), with cited figures."
            ),
            constraints=[_CITE, _RECENT, "Focus on revenue and AI-driven leasing demand."],
            success_criteria=[
                "Report revenue for recent quarters",
                "Describe AI-driven leasing / xScale demand",
                "Provide traceable sources",
            ],
            on_task_keywords=[
                "revenue",
                "xscale",
                "leasing",
                "interconnection",
                "bookings",
                "colocation",
                "hyperscale",
            ],
            on_task_queries=[
                "Equinix revenue most recent quarter",
                "Equinix xScale hyperscale data center leasing AI demand",
                "Equinix interconnection revenue growth latest quarter",
                "Equinix annualized gross bookings record latest quarter",
            ],
            distractors=[
                Distractor(
                    "dividend",
                    "Equinix's dividend and REIT payout policy",
                    ["dividend", "payout", "dividend yield"],
                    "Equinix dividend per share and REIT payout",
                ),
                Distractor(
                    "rates",
                    "interest rates and the Federal Reserve's effect on data-center REITs",
                    ["interest rate", "federal reserve", "rate cut", "treasury yield"],
                    "interest rates Federal Reserve impact on data center REIT valuations",
                ),
                Distractor(
                    "short_report",
                    "the short-seller report on Equinix",
                    ["short seller", "hindenburg", "short report"],
                    "Hindenburg short seller report on Equinix",
                ),
            ],
            detours=[
                Detour(
                    "affo_definition",
                    "what AFFO (adjusted funds from operations) means for a data-center REIT",
                    ["affo", "funds from operations"],
                    "What is AFFO adjusted funds from operations for a data center REIT",
                ),
                Detour(
                    "competitor_benchmark",
                    "Digital Realty's AI leasing results, to benchmark Equinix against",
                    ["digital realty"],
                    "Digital Realty AI data center leasing bookings latest quarter",
                ),
            ],
            updated_directive=(
                "Directive update from the orchestrator: switch to analyzing Equinix's capital expenditure and "
                "development pipeline."
            ),
            updated_keywords=["capex", "capital expenditure", "development pipeline", "under construction"],
            unsupported_metric="Equinix's revenue two years from now",
        ),
        TaskTemplate(
            family="marvell_datacenter",
            agent_id="agent_mrvl",
            swarm_objective=INFRA_OBJECTIVE,
            directive=(
                "Investigate Marvell Technology's data-center revenue (custom AI silicon and electro-optics), recent "
                "figures and growth, with cited sources."
            ),
            constraints=[_CITE, _RECENT, "Focus on the data-center end market."],
            success_criteria=[
                "Report data-center revenue for recent quarters",
                "Explain the custom silicon and optics contributions",
                "Provide traceable sources",
            ],
            on_task_keywords=[
                "data center",
                "custom silicon",
                "custom asic",
                "electro-optics",
                "optical",
                "dsp",
                "xpu",
            ],
            on_task_queries=[
                "Marvell data center revenue most recent quarter",
                "Marvell custom AI silicon revenue growth",
                "Marvell electro-optics PAM4 DSP data center demand",
                "Marvell data center share of total revenue latest quarter",
            ],
            distractors=[
                Distractor(
                    "automotive",
                    "Marvell's former automotive Ethernet business and its sale",
                    ["automotive", "auto ethernet", "infineon"],
                    "Marvell automotive Ethernet business sale to Infineon",
                ),
                Distractor(
                    "carrier_enterprise",
                    "Marvell's carrier and enterprise networking end markets",
                    ["carrier", "enterprise networking", "5g", "wireless infrastructure"],
                    "Marvell carrier infrastructure and enterprise networking revenue",
                ),
                Distractor(
                    "stock",
                    "Marvell's stock price and analyst ratings",
                    ["stock price", "share price", "price target", "analyst rating", "valuation"],
                    "Marvell stock price and analyst price targets",
                ),
            ],
            detours=[
                Detour(
                    "customer_context",
                    "which hyperscaler customers Marvell builds custom AI chips for",
                    ["amazon", "trainium", "microsoft maia", "hyperscaler customer"],
                    "Which hyperscalers are Marvell custom AI chip customers",
                ),
                Detour(
                    "fiscal_calendar",
                    "Marvell's fiscal calendar (its fiscal year ends around the start of February)",
                    ["fiscal year end", "fiscal calendar", "quarter end"],
                    "Marvell fiscal year end date and fiscal quarters",
                ),
            ],
            updated_directive=(
                "Directive update from the orchestrator: switch to "
                "analyzing Marvell's gross margin trend."
            ),
            updated_keywords=["margin"],
            unsupported_metric="Marvell's data-center revenue for the next fiscal year",
        ),
    ]
}
