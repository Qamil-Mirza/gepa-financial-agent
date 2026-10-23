# PitStop: drift detection and correction for agent workflows

**Date:** 2026-10-20 · **Status:** design, pending approval · **Owner:** Qamil Mirza

This document turns the hackathon project (GEPA-based drift supervision on Crusoe managed inference) into an open-source Python package with a product focus. Decisions the owner made are marked **(owner)**. Decisions made on the owner's behalf during design are marked **(design call)** and can be revisited.

---

## 1. Why

Developers running LLM agents in production lose time and money when an agent quietly does something other than what they intended. That happens in two ways:

- **Off-task drift:** the agent leaves its goal or constraints. It loops, takes out-of-scope actions, or draws unsupported conclusions.
- **Degraded drift:** the agent does the right task, but worse than before. This often follows a silent model, provider, quantization or harness change.

Both kinds vary in urgency. A loop in a low-stakes summarizer is a trend to watch. The same loop in an agent that can send email needs a page now.

The research behind this design (summarized in §11) found:

- Existing tools (Langfuse, LangSmith, Datadog, Arize, Braintrust) stop at observation and alerting.
- The hardest part of an incident is usually knowing *what changed*.
- No vendor combines all three of the following:
  1. endpoint-level change detection tied to agent metrics;
  2. per-workflow alert sensitivity driven by criticality;
  3. gated, closed-loop prompt correction.

That combination is PitStop's position.

**Who it's for (owner):** developers who run agent workflows and want visibility into drift, with urgency labels that reflect how critical each workflow is.

## 2. What PitStop is

An open-source **(owner)** Python package. A developer adds it to an existing agent with a few lines of code **(owner)**. It then:

1. **Watches** every run, using cheap code checks inline plus deeper checks in the background.
2. **Labels** each drift P0–P3 from the drift's kind and the workflow's declared criticality.
3. **Acts in the moment** with plain code: blocks a tool call, halts a run, enforces a hard dollar budget, or falls back to a pinned model.
4. **Alerts** through PagerDuty, Opsgenie, Slack, email or a webhook, routed by urgency.
5. **Corrects** using GEPA in the background. GEPA rewrites the *agent's* prompt **(owner)**, gates the candidate on held-out runs, and then ships it automatically or after approval, depending on the workflow's mode.
6. **Shows** it all in a local dashboard (`pitstop ui`).

Everything runs on the developer's own machine or servers **(owner)**. Developers bring their own API keys for the judge, GEPA, canaries and alert channels **(owner)**. There is no hosted service and there are no accounts.

### Goals
- Catch both kinds of drift and say what likely caused each one.
- Make urgency reflect blast radius, using the P0–P3 levels in §5.
- Integrate in a few lines with the majority of agent stacks **(owner)**.
- Close the loop with prompt fixes that are trusted, versioned and reversible.
- Enforce spend limits as hard stops, not alerts.

### Non-goals
- A general tracing or observability platform. PitStop reads traces the user already has and sends its own verdicts back into those tools. It does not compete with them.
- A hosted SaaS, multi-tenant accounts, or authentication for v1.
- Correcting anything other than prompts (no fine-tuning, no code changes).
- Using GEPA as the response to P0 incidents. P0 responses are deterministic.

### Anti-goals (it fails even if it technically works)
- **Painful adoption.** If it takes more than a few lines, or breaks async or streaming in the user's agent, it has failed. The integration should work for most systems without framework-specific work **(owner)**.
- **Costing more than it saves.** Monitoring that burns significant tokens defeats the purpose. See the watch budget in §6.4.
- **Untrusted auto-fixes.** A prompt that changes in production without a record, a reason and a one-click rollback. The owner considers this solvable with manual and auto modes plus rollback, modeled on Claude Code's manual and auto modes **(owner)**.
- **Crying wolf.** The owner considers this solvable with per-workflow sensitivity and better detection **(owner)**. The design treats false-alarm rate as a first-class metric (§6.5).

## 3. Developer experience (the few lines)

```python
import pitstop
from openai import OpenAI

pitstop.init()                                       # reads ~/.pitstop/config.toml or PITSTOP_* env

client = pitstop.wrap(OpenAI(base_url=CRUSOE_URL))   # inline: budget, halts, fallback, telemetry

RESEARCHER = pitstop.prompt("researcher", default=open("researcher.txt").read())  # managed prompt (see §7.3)

@pitstop.tool(side_effects=True)                     # off-task calls to this tool are P0 and blockable
def send_email(to, body): ...

@pitstop.workflow(
    "earnings-research",
    goal="Compare quarterly margins across peers; cite sources",
    criticality="high",          # high | medium | low
    budget_usd=10.00,            # per day, hard stop
    sensitivity="normal",        # low | normal | high
    fix_mode="manual",           # manual | auto
)
def run(question: str) -> str:
    ...  # the user's agent, unchanged, using `client` and `RESEARCHER.text`
```

- **Both forms supported.** `pitstop.workflow` works as a decorator and as a context manager. The decorated-function form is needed for correction, because GEPA must be able to replay the workflow (§7.2).
- **OpenTelemetry (design call, from research).** Users who already trace with OTel (Langfuse v3+, Logfire, OpenLLMetry, OpenInference) can add `pitstop.otel.SpanProcessor()` to their tracer provider. PitStop then reads the GenAI semantic-convention spans for detection and the run graph. OTel is observe-only, so the inline features still need `wrap` and `@tool`.
- **Framework adapters (design call).** These hook into each framework's native guardrail points rather than wrapping beneath the framework. Milestone 2 ships adapters for the **OpenAI Agents SDK** (tool input guardrails) and **LangGraph / LangChain** (middleware). Google ADK and Claude Agent SDK adapters come later.
- **Correctness requirements.** `wrap` must support sync, async and streaming calls, counting usage at the end of a stream. It delegates through `__getattr__` and never monkeypatches globally.
- **Emit, don't compete.** PitStop's verdicts are written back as OTel span events, so they also show up in the user's existing Langfuse or Datadog views.
- **Agents inside a workflow (design call).** `pitstop.agent(name, goal=None)` is an optional decorator or context manager nested inside a workflow, for example `planner`, `worker · NVDA`, `synthesizer`.
  - It names run-graph nodes for users who don't use OTel.
  - It gives L1 and L3 a per-agent goal. If `goal` is omitted, the agent inherits the workflow goal.
  - OTel and adapter users get agent nodes automatically from `invoke_agent` spans.
- **Client coverage (design call).** In milestone 2, `wrap` supports OpenAI-compatible clients (Crusoe, vLLM, OpenRouter, Together and others) and the Anthropic SDK. LiteLLM (through its callback hook) and Bedrock come later.

### 3.1 Inline actions: what the agent experiences (design call)

PitStop has four inline actions. None of them interrupts a stream mid-flight. Each takes effect at the next call boundary, before the next LLM call or tool call.

| Action | When | What the agent sees |
|---|---|---|
| **Steer** | Off-task drift on a step with no side effects. This is the hackathon supervisor's main mechanism, kept **(owner)**. | The flagged action does not run. The agent receives a corrective message ("PitStop: this step drifts from your goal: <evidence>. Return to: <goal>.") and re-plans. At most 2 steers per run (the existing `max_corrections`), after which PitStop halts the run. |
| **Block** | Off-task call to a `side_effects=True` tool, or the L1 gate rejects the call | The tool does not run. The agent gets a synthetic tool result `{"blocked_by": "pitstop", "reason": ...}`, so it can recover instead of crashing. |
| **Fallback** | P1 breakage tied to a model or provider change, when a pinned fallback is configured | Subsequent LLM calls go to the pinned model. The agent sees no difference. |
| **Halt** | Budget hit, P0 on a high-criticality workflow, or steer limit exceeded | The next wrapped call raises `pitstop.Halted`, an exception carrying the drift id and reason. It propagates out of the workflow function, and the user's code can catch it. |

Every action is recorded as an event, so the run graph shows it.

### 3.2 Budgets (design call)
- **Scopes.** Budgets can be set per workflow (`budget_usd` on the workflow) or globally in config, either as a total or per endpoint:
  ```toml
  [budgets.crusoe]
  base_url = "https://api.inference.crusoecloud.com/v1"
  limit_usd = 50
  period = "total"   # total | day | month
  ```
  The owner's $50 Crusoe cap is a global `period = "total"` budget.
- **Periods.** `day` and `month` are calendar periods in the configured timezone, which defaults to the system local timezone.
- **Enforcement.**
  - Before each call, PitStop estimates its cost: prompt tokens plus `max_tokens`, or a conservative default if `max_tokens` is unset.
  - If the estimate would cross the limit, the call is refused and the run halts.
  - After each call, PitStop settles the actual usage.
  - Once a budget is exhausted, new runs are refused at the start.
  - Spend from all processes is summed in the SQLite ledger.
- **Prices.** PitStop ships a built-in price table for common hosted models. User config overrides it, and that is where Crusoe prices go. A model with no known price falls back to a conservative default, and the workflow page shows a warning that its budget may be inaccurate. A spend limit is only as good as its price source.
- **Separate meters.** Agent spend, watch spend (§6.4) and correction spend (§7.2) are tracked separately. Global budgets count all three; workflow budgets count only agent spend.

## 4. Architecture

```
 user's process                                   pitstop server process (`pitstop ui`)
 ┌──────────────────────────────────┐             ┌───────────────────────────────────────┐
 │ wrap / @tool / workflow / adapters│  writes     │ background worker                      │
 │  • L0 code checks (inline, <5ms)  │────────────▶│  • L2 change-point stats                │
 │  • L1 pre-action gate (P0/P1 wf)  │  SQLite     │  • L3 run judge (sampled / flagged)     │
 │  • budget ledger (hard stop)      │  (WAL)      │  • L4 endpoint canaries (scheduled)     │
 │  • P0 alerts sent directly        │             │  • L5 change attribution                │
 │ otel SpanProcessor (observe-only) │             │  • GEPA correction + gate               │
 └──────────────────────────────────┘             │  • alert routing (P1–P3)                │
                                                  │ local API + bundled React dashboard     │
                                                  └───────────────────────────────────────┘
```

- **Storage (design call).** A single local SQLite file in WAL mode, at `~/.pitstop/pitstop.db` by default, configurable with `PITSTOP_HOME`. The SDK writes to it directly, so it works whether or not the server is running.
- **Server (design call).** FastAPI plus a background worker in one process, started by `pitstop ui` or `pitstop serve`. If the server is not running, events build up and the background layers process them on the next start.
- **What never depends on the server.** Inline protections (L0, L1, budget) and P0 alerts. The SDK sends P0 alerts directly.
- **Scope (design call).** v1 assumes the SDK and server share a filesystem: one machine, or one host per deployment. Collecting from remote agents over HTTP is a later milestone (§10).
- **Models.** The judge, GEPA reflection and canaries use any OpenAI-compatible endpoint the user configures with their own key. Crusoe is the documented default **(owner: Crusoe-first, BYO keys)**.
- **Run-level tags.** Every run records model id, provider/base URL, prompt version hash, PitStop version and SDK adapter. This powers attribution (L5).

## 5. Urgency model (P0–P3)

Urgency is derived from three inputs: the **drift signal**, the **workflow's criticality**, and whether a **side-effecting tool** is involved. These levels come from the research synthesis. The exact mapping table is a design call.

| Level | Meaning | Typical triggers | In-the-moment response | Alert route (design call) |
|---|---|---|---|---|
| **P0** | Act now | Off-task call to a `side_effects=True` tool; spend rate above N× baseline or budget hit; loop on a high-criticality workflow | Block the tool / halt the run. GEPA is never the P0 response. | PagerDuty or Opsgenie (phone/SMS), plus Slack |
| **P1** | Hours | Tool-call/schema failure spike, refusal spike, garbage output, or a sudden quality step change right after a model/provider/prompt change | Fall back to the pinned model if configured. Queue a GEPA candidate. | Slack + email |
| **P2** | This week | Right task, quality degrading within contract (judge score drift, goal-adherence decay in long runs) | GEPA runs in the background. | Daily digest |
| **P3** | Trend | Input/topic shift without quality loss, slow token or latency creep | Log it; add examples to the eval set. | Daily digest |

**How criticality shifts levels.** A `high` workflow escalates borderline signals one level, up to P1. Escalation to P0 requires side effects or spend. A `low` workflow de-escalates one level, down to P3. P0 triggers from side-effecting tools and budgets are never de-escalated.

**Sensitivity.** `low`, `normal` and `high` move the thresholds and judge sample rates (§6.3) together. Crying wolf is handled here **(owner)**.

## 6. Detection

### 6.1 Drift taxonomy
- **Off-task.** Keep the six types from the hackathon supervisor: goal_deviation, constraint_violation, unproductive_loop, unsupported_conclusion, ignored_evidence, obsolete_directive. Add **out_of_scope_action** for tool calls outside the declared goal.
- **Degraded.** quality_drop, tool_call_breakage, schema_breakage, refusal_spike, cost_or_latency_creep.
- **Shift.** input_shift (P3 only).

### 6.2 Layers (design call, from research)

| Layer | Where | Signal | Catches | Cost |
|---|---|---|---|---|
| **L0 code checks** | inline, every step | Near-duplicate (tool, args) in the last k steps; step/token counts vs the workflow's p95; tool allowlist / scope; schema and tool-call parse failures; refusal patterns; budget | loops, out-of-scope tools, breakage, spend | ~0 |
| **L1 pre-action gate** | inline, before `side_effects=True` tools on high/medium workflows only | Small judge model, from a different model family than the agent where possible: "does this action serve the goal?" (LlamaFirewall AlignmentCheck pattern) | off-task actions | ~0.5–1.5 s per gated call |
| **L2 change-point stats** | background | CUSUM / Page-Hinkley on per-workflow streams (tool error rate, tokens/step, output length, refusal rate, judge pass rate), split by run tags | sudden degradation, slow off-task creep | ~0 |
| **L3 run judge** | background | Judges whole runs (hierarchical + sliding-window), not single steps. 3 votes, threshold calibrated on labeled runs. Triggered by L0/L2 flags plus a base sample rate by criticality (default high 25%, medium 5%, low 1%) | off-task and quality drop | bounded by the watch budget |
| **L4 endpoint canaries** | scheduled, plus on detected model/provider tag change | Fixed probe set against each configured endpoint: an MMD two-sample test (Model Equality Testing) plus tool-call trigger F1 and JSON-schema accuracy (Kimi Vendor Verifier-style) | silent model/quantization/serving swaps | < $1 per run |
| **L5 attribution** | background, on alarm | Compare metrics before and after the change point, broken down by run tags | answers "what changed" | ~0 |

### 6.3 What this replaces
The hackathon supervisor runs an LLM judge (Nemotron) on every step. That is the main source of false alarms: drift is rare, so per-step judging produces mostly false positives. In the new design the judge is L3, which runs on whole runs, is gated and sampled, and is calibrated. Its prompt is **frozen and versioned**. GEPA does not optimize the detector in production **(owner agreed)**. Calibration is a deliberate offline step using labeled runs.

### 6.4 Watch budget
Monitoring cost is tracked separately from agent cost. Each workflow has a **watch budget** (design call: default 10% of the agent's own spend, configurable as a percentage or in dollars). When the budget is close to running out, L3 sampling drops first. L0, L1 and budget enforcement are never throttled. The dashboard shows watch cost next to agent cost, so the "costs more than it saves" anti-goal is visible. The owner has not settled further cost strategy; this is the v1 mechanism.

### 6.5 Measuring the detector
PitStop has to meet its own bar. The hard held-out eval set (from `financial-agent/test-synthetic-data`) and the hackathon hard-pairs eval move into `benchmarks/`. They report drift recall and false-alarm rate per layer. A detector change ships only if false alarms do not rise. This is the same rule as the existing gate.

## 7. Correction

### 7.1 What gets optimized
Only **managed prompts**: those registered with `pitstop.prompt(name, default=...)`. `pitstop.prompt` returns the currently promoted version. A fix therefore takes effect without a redeploy, and rollback is instant. Prompts that are not managed still get fix suggestions, shown as a diff to copy, but never auto-apply.

**When the developer changes the code default (design call).** Suppose GEPA has promoted v3 and the developer then edits the `default=` text.
- The developer's edit wins. It becomes a new version, tagged as a developer edit, and supersedes v3.
- The dashboard notes "your edit replaced fix v3", and v3 stays available to roll back to.
- Any pending candidates built on the old text are marked stale.

Silently ignoring an explicit edit would break trust more than silently replacing a fix the developer can see in the history.

### 7.2 Replay
GEPA needs to run the workflow with candidate prompts:

- **Inputs.** It replays recorded inputs through the decorated `@pitstop.workflow` function.
- **Tools.** During replay, `side_effects=True` tools are **stubbed** by default and return the recorded output for matching calls, or a neutral stub otherwise. Read-only tools run live unless the user marks them `replay="recorded"`.
- **Without a replayable function.** If a workflow is not replayable (context-manager form only), correction is unavailable for it and the dashboard says so.
- **Where replay runs (design call).** Replay runs in the user's environment, not inside the server process.
  - Config names an entrypoint (`module:function`) and the Python interpreter to use.
  - The server launches `pitstop correct --workflow <name>` as a subprocess with that interpreter. The subprocess sees the user's code, dependencies and environment variables.
  - Developers can also run `pitstop correct` by hand.
  - The server never imports user code.
- **Who pays for replay (design call).** Replay LLM calls and live read-only tools count toward a separate **correction budget**, by default $5 per optimization run, configurable per workflow. They do not count toward the workflow's own `budget_usd`. That way GEPA can't trip the workflow's halt, and it can't spend without a ceiling.

### 7.3 Loop
1. **Trigger.** A confirmed P1/P2 drift, meaning L2 or L4 agrees with L3.
2. **Build data.** Use the flagged runs plus a sample of healthy runs, split into train and held-out sets.
3. **Optimize.** GEPA (`gepa` library, existing `optimize_anything` integration) rewrites the managed prompt. Reflection runs on the user's configured model.
4. **Score with something other than the detector.** Candidates are scored by a task-quality scorer (a rubric judge plus any golden examples the user provides), **never by the drift detector**. Optimizing against the monitor teaches the agent to hide misbehavior (OpenAI, 2025).
5. **Gate.** Promote only if the candidate:
   - beats the current prompt on held-out runs;
   - has zero negative flips beyond tolerance;
   - does not increase off-task flags;
   - grows the prompt by no more than a configurable limit (the hackathon GEPA prompts grew about 5×).
6. **Ship by mode (owner: modeled on Claude Code's manual and auto modes):**
   - `manual`: the fix appears in the dashboard with a diff, held-out score, negative flips and size change. The developer can Approve, Canary (route X% of runs) or Dismiss.
   - `auto`: candidates that pass the gate go to a canary automatically, then are promoted if the canary holds.
   - Design-call defaults: `high` → manual, `medium`/`low` → auto.
7. **Rollback.** Every promotion is a new prompt version with its reason, linked drift, scores and timestamp. Rollback is one click (or `pitstop prompts rollback <name>`) and is always available. A canary that underperforms rolls itself back.

## 8. Alerts
Routing per §5. The integrations are PagerDuty Events v2, Opsgenie, Slack incoming webhook, SMTP email and a generic webhook, all configured with the user's own keys **(owner: email/page/text)**. SMS and phone calls go through PagerDuty or Opsgenie rather than a custom SMS integration (design call). Alerts are deduplicated per (workflow, drift type) while a drift is open. Every alert answers: what drifted, what likely changed, and what is being done.

## 9. Dashboard

Design context lives in `/.impeccable.md` **(owner)**: a calm control room, Sentry/PagerDuty clarity with Vercel/Stripe cleanliness, a subtle racing nod (PitStop name, flags), light and dark themes following the system, and urgency never shown by color alone. The chosen mockups are in `docs/superpowers/specs/2026-10-20-pitstop-mockups/`.

- **Home: pit wall (owner, direction B).**
  - Workflows with open drifts get large tiles: flag (red = halted, yellow = degraded), criticality, P-level labels, one-line description, and spend against the hard budget.
  - Healthy workflows shrink to small green tiles.
  - A live feed on the right lists halts, model changes, fallbacks and fixes ready to review.
- **Workflow page, default tab: Run graph (owner).**
  - The workflow drawn as a DAG of agents, LLM calls and tools, built from span parent/child links and aggregated over a time window or for a single run.
  - Each node shows cost, latency and drift badges. The slowest path is called out.
  - Clicking a node shows its runs and verdicts.
- **Workflow page, other tabs.** *What changed* (quality, errors and cost over time with model/provider/prompt changes drawn on it), *Runs*, *Prompt versions* (history, diffs, rollback).
- **Right column on the workflow page.** Always shows open drifts and any fix awaiting review, with Review diff / Canary / Dismiss.
- **Also needed:**
  - an Endpoints page (canary results per model/provider);
  - settings showing sensitivity, fix mode, budget and watch budget per workflow (read-only in v1; configured in code or config);
  - an empty state that teaches the few-lines setup.
- **Tech (design call).** React + TypeScript + Vite in `ui/`, built at release time and bundled into the Python wheel, so users never need Node. The run graph uses React Flow with ELK layout. The two existing vanilla-HTML UIs are retired.

### 9.1 Landing page (owner request)
This is the public front door for the open-source project.

- **Hosting (design call).** A static site in `site/`, deployed with GitHub Pages.
- **Hero (owner choice: "Racing line" direction).** The mockup is `2026-10-20-pitstop-mockups/landing-hero.html`.
  - Headline: "Keep your agents on the *racing line*."
  - To the right, a top-down circuit drawn on canvas over SVG. The agent laps the racing line. On lap 3 it drifts into the infield, a flag drops with a "P1 · Off-task" callout, and it is steered back ("Back on line. Prompt fix queued for review.").
  - A small HUD shows workflow, lap, drifts caught, and spend against budget.
  - Calls to action: copy `pip install pitstop`, "Read the docs", "Star on GitHub".
- **Below the hero (owner).** Simple, static sections; scroll-driven motion is out for now.
- **Motion rules.**
  - Respect `prefers-reduced-motion`: show a static frame of the drift-and-recovery moment.
  - Pause when off-screen.
  - Cap DPR at 2.
  - Follow the light/dark tokens in `.impeccable.md`.

### 9.2 README and open-source repo basics (owner request)
The README should look and work like a well-kept open-source project. It must always describe what actually works at that point (design call). Features that aren't built yet appear only in a clearly labeled roadmap, so nobody runs `pip install pitstop` before the package exists.

**README contents, in order:**
1. Logo/wordmark, a one-line pitch, and badges (CI, license, Python version, and PyPI once published).
2. A hero GIF or screenshot (from milestone 3 on; until then, a still from the landing hero).
3. **Why PitStop**: the problem, in three or four lines.
4. **Quickstart**: install, the few lines of code, `pitstop ui`, and what you'll see. Until milestone 2, this section is "Run the example" instead.
5. **How it works**: Mermaid diagrams, which GitHub renders natively.
   - a runtime architecture diagram (user process ↔ SQLite ↔ server/worker ↔ dashboard, as in §4);
   - the detection-layer flow, L0–L5 → P-level → action;
   - the correction loop (flagged runs → GEPA → gate → manual/auto → canary → promote/rollback);
   - a repo map showing how the codebase is laid out (§12).
6. **Concepts**: workflows, criticality, P0–P3, inline actions, budgets, managed prompts.
7. **Configuration**: an example `config.toml` (keys, endpoints, budgets, alerts).
8. **Benchmarks**: detector recall and false alarms, with honest limits. This carries over the hackathon results and their caveats.
9. **Roadmap**: the milestones, with status.
10. **Contributing**, **License**, **Acknowledgements** (the hackathon, Crusoe, GEPA, the original teammates).

**Repo files (design call).**
- `LICENSE` (Apache-2.0)
- an updated `CONTRIBUTING.md` (dev setup with `uv`, tests, the UI build)
- `CODE_OF_CONDUCT.md` (Contributor Covenant)
- `SECURITY.md`, which says how to report issues; the repo is public, and PitStop handles API keys
- `.github/ISSUE_TEMPLATE/` (bug, feature) and a pull-request template
- `CHANGELOG.md`
- `.env.example`
- a CI workflow that runs tests and lint

**When it lands.**
- Milestone 1: the README is rewritten for the cleaned repo, with the repo map, architecture diagrams, a run-the-example guide and the roadmap. The repo files above are added.
- Milestones 2 and 3: the Quickstart and screenshots are updated as those features ship.

## 10. Milestones
Each ends in something demoable. There is no deadline **(owner)**. Order (design call, owner wants both SDK and dashboard):

1. **Cleanup and restructure** (§12). One installable package, the example agent still runs, tests pass in one command.
2. **SDK and detection.**
   - `wrap`, `workflow`, `tool`, `prompt`, the budget ledger, the OTel span processor, L0–L2 and L5, P0 alerts, plus the two framework adapters.
   - The example agent (financial research swarm) is wired in with a few lines and catches injected drift. A CLI (`pitstop status`, `pitstop drifts`) shows P-labeled drifts.
   - The owner's own Crusoe usage gets a $50 hard budget from day one (a global `period = "total"` budget, §3.2).
3. **Dashboard** (status: 3a done: server, worker, pit wall, workflow, run detail, settings; 3b next: L3, L4, Endpoints page; 3c: landing site, screenshots). Pit wall, run graph, what changed, runs, settings, all on real data from milestone 2. L3 run judge and L4 canaries land here, with the Endpoints page. **First end-to-end demo.**
4. **Correction.** Managed prompts, replay, GEPA loop with gate, manual/auto modes, canary, rollback, Prompt versions tab.
5. **Later (not designed here).** Remote collection over HTTP, more adapters (Google ADK, Claude Agent SDK, CrewAI), editing settings from the UI, a LiteLLM gateway plugin for teams that won't touch code.

## 11. Research basis (summary)
- **Complaints.** Silent provider regressions (Anthropic postmortems Aug–Sep 2025 and Apr 2026; OpenAI's sycophantic GPT-4o rollback, Apr 2025), quantization and serving differences across hosts (Kimi Vendor Verifier), runaway agent loops (a widely repeated but unverified $47k story), side-effect incidents (Replit production database deletion).
- **Detection.** Base-rate problem of per-step judges; trajectory monitors (Kale et al., ICLR 2026); LlamaFirewall AlignmentCheck; Model Equality Testing (Gao et al., ICLR 2025); monitor obfuscation under optimization pressure (OpenAI, 2025).
- **Integration.** Vendors converging on OTel for observation, with hooks, wrappers or gateways for enforcement. GenAI semantic conventions are still at Development stability, so attribute names should be pinned and mapped.
- **Auto-correction.** Willison's GEPA experiment (Jul 2026): gains on train data, held-out regression. Practitioner consensus: automate the work, gate the decision.

## 12. Repo cleanup map
The current repo has two parallel supervisor stacks. PitStop (`pitstop/pitstop/pitstop/`) is live. The older `gepa_financial_agent/` supervisor stack is superseded, except for its GEPA pieces.

**Target layout**
```
pyproject.toml            # package name `pitstop` (src layout), Apache-2.0
src/pitstop/
  sdk/       wrap, workflow, tool, prompt, adapters/
  otel/      span processor + semconv mapping
  detect/    l0_checks, l1_gate, l2_changepoint, l3_judge, l4_canary, attribution
  correct/   replay, gepa loop (from gepa_financial_agent/{optimizer,recursive}), gate
  budget/    ledger (estimate-before, settle-after)
  alerts/    routing + integrations
  store/     sqlite schema + migrations
  server/    FastAPI app, background worker, static UI bundle
  llm.py     the one OpenAI-compatible client (Crusoe default)
  cli.py     init, ui/serve, status, drifts, prompts, canary
ui/                         # React source
examples/financial-research/  # the PitStop research swarm, rewired onto the SDK; keeps drift injection for demos
benchmarks/                 # hard held-out set + hard-pairs eval + supervisor calibration tooling (incl. hackathon GEPA-on-detector)
tests/
docs/
```

**Keep and move**
- PitStop core: `orchestrator`, agents, research client, `supervisor/signals.py` logic (feeds L0), schemas, timeline severity concepts, `llm.py`.
- `supervisor/correction.py` and `supervisor/policy.py` become the **steer** action and its alert policy (§3.1). `supervisor/supervisor.py` (the per-step judge) becomes the basis for the L1 gate and L3 run judge, and stops running on every step. Its evidence-id validation (dropping invented ids, downgrading DRIFTING with no evidence) is kept.
- From `gepa_financial_agent/`: `optimizer.py`, `recursive.py` (gate logic), `crusoe_client.py` (CrusoeReflectionLM), `config.py`, `policy_store.py` (ideas for prompt versioning), and `session_manager.py` / `adapters/base.py` (ideas for the SDK's observe API).
- Eval data generator `synthetic-data/synth`. It moves into `benchmarks/`, and generated traces are produced on demand rather than committed.
- Results docs (`docs/RESULTS.md`, `PITSTOP_RESULTS.md`) are kept as history.

**Delete**
- Dead code: `agent/`, `main.py`, `data/sources.py`, `gepa/`, `scripts/check_crusoe.py`.
- The older supervisor stack: `gepa_financial_agent/{supervisor,session_manager,evaluator,dataset,schemas}.py`, `adapters/{base,mock_agent}.py` (after harvesting ideas), `policies/`, `data/fixtures/`, and `scripts/{optimize_supervisor,evaluate_supervisor,demo_supervision,smoke_test,summarize_repeats,make_dev_fixtures}.py`.
- Duplicate Crusoe clients (4 → 1) and requirements files (4 → 1 `pyproject.toml`).
- Committed generated data: `synthetic-data/{traces,traces_archive,evidence,data}`, `financial-agent/test-synthetic-data/{traces,data,evidence}`, the committed You.com cache, `demo/snapshot.json`, `evals/results`. The hard held-out *case definitions* stay. Raw traces are regenerated or attached to a GitHub release.
- The two old UIs and their servers (`demo/`, `ui/`, `pitstop/server.py`), once milestone 3 lands. Until then they stay so the demo keeps working.
- The `sys.path` hack in `adapters/pitstop.py`; the three-level `pitstop/pitstop/pitstop/` nesting; the confusing in-repo `financial-agent/` folder.

**Also**
- Add `.superpowers/` to `.gitignore`.
- The repo is public, so double-check that no keys are in history before the open-source announcement.
- The cleanup lands as a PR on a branch, because teammates (Arjun, Gautam) have open remote branches.

## 13. Left to the builder
- Exact SQLite schema, CUSUM/Page-Hinkley parameters, L0 thresholds and default sample rates (tuned against `benchmarks/`).
- The probe set for L4 canaries and the MMD test details.
- Internal module boundaries beyond §12, CLI command details, test strategy.
- Dashboard component structure, within `.impeccable.md` and the mockups.
- Packaging and release mechanics (bundling the UI build into the wheel).
- Landing-page sections below the hero, and the site build tooling.

## 14. Resolved after design review
1. **L1 gate failure policy (owner).** When the pre-action judge times out, errors, or has no key configured:
   - on `high` workflows, the side-effecting tool call is **blocked** (fail closed);
   - on `medium` workflows, it **goes ahead** (fail open);
   - either can be overridden per workflow (`on_gate_error="block" | "allow"`).
   Every gate failure is recorded and shown on the workflow page.
2. **In-run steering is kept (owner).** It is the default response to off-task steps that have no side effects. PitStop escalates to halt after the steer limit (§3.1).
