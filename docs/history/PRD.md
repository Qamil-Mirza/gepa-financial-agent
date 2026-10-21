# GEPA Financial Agent — Scalable Supervisor for Multi-Agent Financial Research

You are the lead engineer responsible for building the **GEPA Financial Agent**, a pluggable, GEPA-optimized supervision layer for long-running multi-agent financial research systems.

This project is being built for the **Crusoe Recursive Agents Hackathon** on October 17, 2026.

We have three teammates working in parallel:

1. **Financial Agent Stack:** One teammate is implementing a multi-agent financial research system using Crusoe Managed Inference and You.com's Finance Research API.
2. **Synthetic Dataset:** Another teammate is generating labeled synthetic execution trajectories to train and evaluate drift detection using GEPA.
3. **Supervisor (our responsibility):** We are implementing the pluggable, concurrent, GEPA-optimized supervisor that attaches to the financial agent stack.

**You are responsible for the third workstream only.**

Do not duplicate the financial agent stack or synthetic-data generator. Build stable integration interfaces so both components can connect as they become available.

## 1. Product vision

We want a reusable supervision layer that developers can attach to an existing multi-agent workflow.

Each agent has an assigned task or directive. As agents execute, they generate observable trajectories consisting of actions, tool calls, observations, intermediate findings, and communication with other agents.

Our supervisor evaluates these trajectories and detects whether the agents are drifting away from their assigned objectives.

Potential failure modes include:

- Pursuing an irrelevant research direction.
- Repeatedly executing unproductive actions.
- Ignoring explicit requirements or constraints.
- Drawing conclusions unsupported by retrieved evidence.
- Following outdated instructions after legitimate task changes.
- Producing outputs that do not satisfy assigned deliverables.
- Deviating from a task in ways that threaten downstream agents.

The system should distinguish genuine drift from legitimate exploration, temporary uncertainty, or reasonable changes in strategy.

**The long-term goal is online trajectory correction.**

However, our immediate objective is to build a working drift detector whose decision policy is optimized using GEPA.

## 2. Core research hypothesis

We hypothesize that:

**A GEPA-optimized supervisor can detect trajectory drift more accurately than a fixed-prompt supervisor while remaining sufficiently efficient to supervise many concurrently executing agents.**

We want to measure this claim rather than assume it is true.

Our system should support direct comparisons between:

- A fixed, manually written supervisor prompt.
- A GEPA-optimized supervisor prompt.
- A simple deterministic drift-detection baseline, if feasible.

All comparisons must use the same underlying model, evaluation examples, and output schemas.

## 3. First task: inspect documentation and repository

Before implementing:

1. Inspect the repository's current structure, dependencies, and existing code.
2. Locate and read the Crusoe hackathon skill file supplied in the documentation.
3. Review the relevant Crusoe developer examples:
   https://github.com/crusoecloud/crusoe-developer-hub
4. Review the official GEPA implementation:
   https://github.com/gepa-ai/gepa
5. Review the GEPA `optimize_anything` API:
   https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/
6. Identify the exact interface needed for Crusoe Managed Inference.
7. Identify any existing worker-agent interfaces or event schemas defined by the other teammate.
8. Check for concurrent changes before modifying shared files.

Prefer working, tested integrations over speculative abstractions.

Do not assume a package API or model name without checking the installed version and official documentation.

Use the Crusoe hackathon skill instructions wherever applicable.

## 4. Architectural decision: cloned supervisors per task

We do NOT want a single centralized LLM supervisor sequentially inspecting every agent.

Instead, implement a **concurrent, task-scoped supervisor architecture**.

Each active task or agent-task assignment receives its own lightweight supervisor session.

For example:

Financial research swarm:

- Agent A: Investigate NVIDIA's data-center revenue.
- Agent B: Investigate AMD's data-center revenue.
- Agent C: Analyze competitive market dynamics.
- Agent D: Compare financial performance.
- Agent E: Synthesize the final report.

Each agent should have an independently maintained supervisor context.

Conceptually:

```text
                 FINANCIAL RESEARCH SWARM
                           |
          +----------------+----------------+
          |                |                |
       Agent A          Agent B          Agent C
          |                |                |
    Supervisor A     Supervisor B     Supervisor C
          |                |                |
          +----------------+----------------+
                           |
                 Shared GEPA Policy
                           |
                  Evaluation Logs
                           |
                  GEPA Optimization
                           |
                 Improved Policy vN+1
```

### Important implementation distinction

A supervisor clone is **not** a separate model deployment or a copy of model weights.

A clone is a lightweight, independently maintained supervision session containing:

- A unique supervisor/session ID.
- Its assigned worker/task ID.
- The worker's directive.
- Relevant execution history.
- Recent drift decisions.
- A reference to a shared, immutable, versioned supervisor prompt.
- Relevant task constraints and context.

All supervisor instances share the same Crusoe-hosted inference model and current validated GEPA policy.

Supervisor sessions must be isolated from one another so that unrelated agents' trajectories do not accidentally contaminate their evaluations.

### Concurrency requirements

- Use asynchronous event processing where useful.
- Support multiple supervisor sessions concurrently.
- Use bounded concurrency rather than unlimited inference calls.
- Respect API rate limits and token budgets.
- Avoid holding up healthy worker agents unnecessarily.
- Support configurable inspection frequency and event-triggered evaluation.
- Deduplicate or coalesce redundant evaluation requests.
- Record inference latency, token usage, and supervision overhead.

For V1, supervision should be nonblocking and observational by default.

A future implementation may intercept an agent's proposed next action, but automatic blocking or correction is out of scope unless explicitly requested.

## 5. Supervisor inputs

Define a normalized, framework-independent input schema.

Each evaluation should contain:

```json
{
  "task_id": "task_001",
  "agent_id": "agent_A",
  "supervisor_session_id": "supervisor_A",
  "swarm_objective": "Compare NVIDIA and AMD AI business performance",
  "agent_directive": "Investigate NVIDIA data-center revenue",
  "constraints": [
    "Use cited financial evidence",
    "Focus on data-center revenue"
  ],
  "success_criteria": [
    "Identify reported revenue figures",
    "Explain major growth drivers",
    "Provide traceable sources"
  ],
  "trajectory": [],
  "current_plan": null,
  "proposed_next_action": null,
  "relevant_dependencies": [],
  "remaining_budget": null,
  "observation_step": 0
}
```

This is a proposed internal schema and may be adapted to the existing repository.

### Trajectory event schema

Each event should carry:

- `event_id`
- `task_id`
- `agent_id`
- `step`
- `event_type`
- `timestamp`
- `tool_name`, if applicable
- `tool_arguments`, if applicable
- `content` or structured observation
- `source_references`, if applicable

Supported event types should include:

- Agent message.
- Tool call.
- Tool result.
- Research finding.
- Task or status update.
- Error.
- Proposed next action.
- Task completion.

Use externally observable messages and execution events.

Do not assume access to private chain-of-thought. Explicit agent-generated reasoning summaries may be included if the underlying system exposes them.

### Financial research context

The worker agents will use You.com's Finance Research API.

The supervisor should be able to inspect the resulting research questions, returned evidence, sources, and subsequent worker decisions.

Preserve source URLs and timestamps when available.

Do not directly perform financial research inside the supervisor unless necessary for a clearly defined verification feature.

The supervisor's primary job is **trajectory alignment**, not generating financial advice or replacing the research agents.

## 6. Supervisor outputs

The supervisor must return structured, schema-validated JSON.

Example:

```json
{
  "agent_id": "agent_A",
  "task_id": "task_001",
  "status": "DRIFTING",
  "confidence": 0.86,
  "drift_type": "GOAL_DEVIATION",
  "evidence_event_ids": [
    "event_012",
    "event_014"
  ],
  "explanation": "The agent repeatedly investigates consumer GPU gaming sales despite its directive to analyze data-center revenue.",
  "recommended_action": "FLAG_FOR_REVIEW"
}
```

Supported statuses:

- `ON_TRACK`
- `DRIFTING`
- `UNCERTAIN`

Suggested drift categories:

- `GOAL_DEVIATION`
- `CONSTRAINT_VIOLATION`
- `UNPRODUCTIVE_LOOP`
- `UNSUPPORTED_CONCLUSION`
- `OBSOLETE_DIRECTIVE`
- `OTHER`

These categories should be configurable.

### Rules

- Require evidence-grounded judgments.
- Validate all returned event IDs.
- Never invent citations or trajectory events.
- Do not equate tool failure with objective drift.
- Do not treat legitimate exploratory research as automatically off-track.
- Use `UNCERTAIN` when evidence is insufficient.
- Record malformed outputs and model errors explicitly.
- Keep the final classification separate from any proposed intervention.

Confidence values are model estimates, not calibrated probabilities unless calibration has been evaluated.

## 7. Supervisor session manager

Implement a supervisor manager that can attach supervision to multiple active agents.

The primary public API should resemble:

```python
from gepa_financial_agent import SupervisorManager

manager = SupervisorManager(
    model_client=crusoe_client,
    policy_path="policies/current.txt",
    max_concurrent_inferences=5,
)

manager.register_agent(
    agent_id="agent_A",
    task_id="research_nvda",
    directive="Research NVIDIA data-center revenue",
    swarm_objective="Compare NVIDIA and AMD AI businesses",
)

await manager.observe(
    agent_id="agent_A",
    event=agent_event,
)

decisions = await manager.evaluate_pending()
```

This is a proposed interface. Implement an equivalent clean API that fits the existing codebase.

Support:

- Agent registration.
- Task registration.
- Event ingestion.
- Per-task supervision sessions.
- Asynchronous evaluations.
- Supervisor result retrieval.
- Session cleanup.
- Session replay.
- Policy version selection.

The supervisor package must not depend on a particular financial-agent framework.

Expose a narrow adapter interface that the other teammate can implement or directly call.

## 8. GEPA optimization — must work in V1

GEPA optimization is a mandatory deliverable.

We are optimizing the **supervisor's system prompt**, not the base model weights.

Use the official GEPA library and its `optimize_anything` API, subject to verification against the installed version.

The candidate artifact is a text prompt containing instructions for drift detection and evidence-grounded classification.

Conceptually:

```python
from gepa.optimize_anything import (
    optimize_anything,
    OptimizeAnythingConfig,
)

def evaluate(candidate_prompt, example):
    prediction = run_supervisor(
        prompt=candidate_prompt,
        example=example,
    )

    score, diagnostics = score_prediction(
        prediction,
        example,
    )

    return score, diagnostics

result = optimize_anything(
    seed_candidate=baseline_prompt,
    evaluator=evaluate,
    dataset=train_examples,
    valset=validation_examples,
    objective="Improve reliable, evidence-grounded drift detection",
    config=OptimizeAnythingConfig(
        engine="gepa",
        max_evals=20,
    ),
)
```

This is an architectural example, not code to copy without checking the installed GEPA API.

### Requirements

- Actual GEPA optimization must execute.
- Start with a hand-written baseline supervisor prompt.
- Evaluate candidate prompts against labeled trajectories.
- Return scalar feedback and detailed diagnostic feedback.
- Respect a configurable maximum evaluation budget.
- Save the candidate history and selected policy.
- Save evaluation metrics and optimizer configuration.
- Evaluate the optimized policy on held-out trajectories.
- Allow supervisor instances to load the optimized policy.

The optimizer should not need the live financial swarm to run.

**Use frozen, replayable trajectories during GEPA optimization.**

Never call the You.com Finance Research API again for each GEPA candidate if the underlying research evidence is already recorded.

### GEPA reflection model

Determine which model GEPA uses to propose and reflect on candidate prompts.

Prefer a Crusoe-hosted model where compatible.

Verify the model's endpoint, authentication, and compatibility with GEPA's reflection interface.

If the desired model cannot be connected directly, document the limitation and implement the simplest supported configuration without pretending it uses Crusoe.

Keep supervisor inference and GEPA reflection-model configuration separately adjustable.

## 9. Synthetic data integration

Another teammate is generating synthetic training data.

We need to define an ingestion contract immediately.

Create:

`docs/DATASET_CONTRACT.md`

The dataset should be representable as JSONL.

Each example must contain:

```json
{
  "example_id": "example_001",
  "scenario_id": "scenario_001",
  "observation_step": 8,
  "supervisor_input": {
    "task_id": "task_001",
    "agent_id": "agent_A",
    "swarm_objective": "...",
    "agent_directive": "...",
    "constraints": [],
    "success_criteria": [],
    "trajectory": []
  },
  "ground_truth": {
    "status": "DRIFTING",
    "drift_type": "GOAL_DEVIATION",
    "evidence_event_ids": [],
    "rationale": "..."
  }
}
```

Implement a Pydantic validator and dataset loader.

Support:

- JSONL loading.
- Schema validation.
- Invalid-example reporting.
- Scenario-aware train/validation/test splitting.
- Labeled positive and negative examples.
- Prefix-level evaluation for early drift detection.
- Loading new teammate-generated data without changing optimizer code.

The teammate owns data generation.

We only own the dataset interface and evaluation harness.

### Data leakage prevention

Never place prefixes from the same underlying synthetic trajectory in different splits.

Hold out complete scenario families where practical.

Do not feed ground-truth labels into supervisor inference.

Do not expose held-out test results to GEPA during optimization.

Keep the final test set untouched until a candidate policy has been selected.

## 10. Drift detection evaluation

Implement an evaluation harness to compare the baseline and GEPA-optimized supervisors.

### Primary metrics

- Macro-F1 across `ON_TRACK`, `DRIFTING`, and `UNCERTAIN`.
- Precision and recall for `DRIFTING`.
- False-positive rate on genuinely on-track trajectories.
- False-negative rate on drifting trajectories.

### Secondary metrics

- Accuracy.
- Per-drift-category accuracy.
- Evidence-reference validity.
- Invalid-output rate.
- Inference latency.
- Tokens consumed.
- Estimated inference cost.
- Detection delay, when labeled onset information exists.

### GEPA optimization score

Use a clearly documented per-example scalar score.

A reasonable starting point is:

- Correct drift status: primary reward.
- Correct drift category: smaller additional reward when applicable.
- Valid, correctly grounded evidence: additional reward.
- Incorrectly flagging healthy agents: explicit penalty.
- Unsupported or fabricated evidence: explicit penalty.

Return descriptive feedback identifying the exact error.

Prevent the optimizer from exploiting a majority-class strategy or always predicting `UNCERTAIN`.

Report macro-F1 and false-positive rate separately on full evaluation sets even if GEPA uses a scalar per-example reward.

### Critical baseline comparison

Compare the two supervisor policies on:

- The same held-out examples.
- The same model.
- The same decoding configuration.
- The same schemas.
- The same observation cutoffs.

Persist individual predictions so errors can be inspected manually.

## 11. Scaling to N agents

The system should support N concurrently active supervisor sessions without N independent model deployments.

Use:

- Shared Crusoe inference client.
- Shared immutable GEPA policy.
- Per-task session state.
- Asynchronous inference.
- Concurrency limits.
- Event-triggered evaluation.
- Backpressure and bounded queues.
- Bounded trajectory context.
- Prompt and token accounting.

Do not automatically run a supervisor inference on every event.

Use configurable triggers such as:

- Every N meaningful actions.
- Explicit tool failure.
- Repeated or suspicious behavior.
- Proposed next action.
- Completion of an important milestone.
- Final output verification.

For the MVP, use simple event-based triggers and configurable inspection intervals.

### Shared policy versus isolated context

All supervisors begin with the same validated GEPA-optimized policy.

Their individual task contexts and histories remain isolated.

When GEPA produces an improved policy, future sessions may load it. Existing sessions should keep their current policy version until an explicit version switch, making evaluations reproducible.

Do not allow every task-scoped supervisor to independently run GEPA optimization. That would produce substantial unnecessary inference overhead.

GEPA optimization is shared across the fleet.

## 12. DAG failsafe — optional secondary mechanism

We are retaining a directed acyclic graph as a **failsafe**, not as the primary monitoring architecture.

Do not make a DAG mandatory for supervisor operation.

If the host swarm exposes task dependencies, allow them to be recorded as optional metadata.

The DAG may be used to:

- Identify downstream tasks consuming a suspicious output.
- Estimate potential error propagation.
- Identify tasks blocked by dependencies.
- Prioritize alerts involving critical dependencies.
- Trigger checks before unverified outputs are consumed downstream.

Avoid sophisticated graph algorithms in V1.

A basic dependency representation and downstream traversal are sufficient.

The primary system remains **task-scoped parallel supervision**.

## 13. Financial agent integration contract

Our teammate's worker stack may evolve independently.

Define a small integration contract so that their implementation only needs to:

1. Register an agent and task.
2. Emit normalized execution events.
3. Optionally submit proposed actions.
4. Read supervisor decisions.
5. Optionally provide dependency metadata.

The supervisor must not assume ownership of worker execution, financial research tools, or orchestration.

Expose the API in a way that can support event callbacks or an asynchronous event stream.

Write:

`docs/INTEGRATION_CONTRACT.md`

Include minimal working examples showing how the financial research stack can attach supervision.

Until the real stack is available, implement a small mock adapter and replay existing fixture events.

Do not build a second financial swarm.

## 14. Repository organization

Prefer a clean structure resembling:

```text
gepa_financial_agent/
├── __init__.py
├── schemas.py
├── supervisor.py
├── session_manager.py
├── event_collector.py
├── scheduler.py
├── crusoe_client.py
├── policy_store.py
├── optimizer.py
├── evaluator.py
├── dataset.py
└── adapters/
    ├── __init__.py
    └── mock_agent.py

policies/
├── baseline.txt
└── current.txt

data/
├── fixtures/
└── README.md

scripts/
├── smoke_test.py
├── optimize_supervisor.py
├── evaluate_supervisor.py
└── demo_supervision.py

tests/
├── test_schemas.py
├── test_sessions.py
├── test_evaluator.py
└── test_dataset.py

docs/
├── DATASET_CONTRACT.md
├── INTEGRATION_CONTRACT.md
└── ARCHITECTURE.md

README.md
.env.example
```

Adapt this layout to the actual repository.

Avoid modifying your teammates' implementation files unless required for an agreed integration contract.

## 15. Demo requirements

The hackathon demonstration should show supervision of a financial research swarm.

Example:

- Agent A researches NVIDIA.
- Agent B researches AMD.
- Agent C researches market dynamics.

Agent A follows its original directive.

Agent B begins deviating from its assigned research objective.

Agent C explores a tangential but legitimately relevant topic.

We should demonstrate that the supervisor can identify Agent B while avoiding false alarms on A and C.

### Demo flow

1. Start or replay several concurrent research agents.
2. Show their assigned objectives.
3. Stream actual execution events.
4. Spawn or attach task-specific supervisor sessions.
5. Show supervisors producing drift judgments concurrently.
6. Highlight the exact evidence supporting detected drift.
7. Show baseline-versus-GEPA policy decisions.
8. Display measured held-out detection performance and inference cost.

Avoid a dashboard-first product. A simple terminal-based live trace or lightweight visualization is sufficient.

Do not fabricate improvements or use hidden labels to drive demo predictions.

## 16. Engineering constraints

- Python-first.
- Use async concurrency where appropriate.
- Use Pydantic for schemas.
- Use a shared Crusoe inference client.
- Keep API keys in environment variables.
- Provide mock and real inference modes.
- Make trajectory replay deterministic.
- Make optimizer results reproducible where possible.
- Persist policy versions and evaluation results.
- Keep code simple, testable, and modular.
- Avoid needless dependencies.
- Do not store API credentials in logs.
- Bound API spending.
- Fail gracefully when inference is unavailable.
- Avoid breaking teammates' concurrent work.

Use structured logging and explicit error handling.

Never present mocked inference as real model execution.

## 17. Required implementation order

We have a hard hackathon deadline. Prioritize a working end-to-end system.

### Milestone A — Supervisor core

- Inspect documentation.
- Establish the shared dataset and integration contracts.
- Implement schemas.
- Connect one supervisor to Crusoe inference.
- Produce a valid drift judgment from one trajectory.
- Demonstrate independently from the worker stack.

### Milestone B — Parallel supervisors

- Implement per-task sessions.
- Support concurrent event ingestion.
- Evaluate multiple agents concurrently.
- Enforce inference concurrency limits.
- Demonstrate that task contexts remain isolated.

### Milestone C — Evaluation harness

- Load labeled synthetic examples.
- Run baseline supervisor evaluations.
- Compute detection metrics.
- Produce error diagnostics suitable for GEPA.

### Milestone D — GEPA optimization

- Run the real GEPA optimizer.
- Evaluate prompt candidates.
- Save optimized policy artifacts.
- Evaluate optimized versus baseline prompts on held-out cases.
- Report actual results.

### Milestone E — Financial stack integration

- Connect teammate's agent events through the adapter.
- Test supervision on financial research trajectories.
- Enable concurrent supervision.
- Connect the demo runner.

### Milestone F — Optional enhancements

Only after all required milestones work:

- Proposed-action interception.
- DAG impact analysis.
- Priority-based inspection scheduling.
- Supervisor call batching.
- Richer visualization.
- Early-detection analysis.

Do not implement automatic agent correction yet.

## 18. Acceptance criteria

The project is successful when:

1. A developer can attach task-scoped supervision to an existing agent using a simple Python interface.
2. Multiple agents can be supervised concurrently through Crusoe Managed Inference.
3. Supervisor decisions are structured, validated, and grounded in real execution events.
4. The system can load externally generated labeled trajectories.
5. GEPA actually optimizes the supervisor prompt.
6. Baseline and optimized supervisors can be compared reproducibly.
7. Optimization artifacts and evaluation metrics are persisted.
8. The supervisor can integrate with the teammate's financial research system without taking ownership of the worker stack.
9. A working demo can show concurrent agent supervision and drift detection.

Do not claim GEPA improved the model unless the measured results demonstrate it.

## 19. Start execution now

First inspect the repository and relevant documentation.

Then:

1. Write `DATASET_CONTRACT.md` and `INTEGRATION_CONTRACT.md` so teammates can work against stable interfaces.
2. Implement the smallest end-to-end supervisor inference path.
3. Verify one real Crusoe inference call.
4. Implement parallel supervisor sessions.
5. Implement the evaluation harness.
6. Run a real small-budget GEPA optimization.
7. Integrate teammate contributions as they become available.

Do not wait for the full synthetic dataset or final financial agent stack before implementing the supervisor.

Use minimal clearly labeled fixtures for development.

Prefer completing a working vertical slice over building a sophisticated framework that is not runnable.

Maintain a short progress log containing completed milestones, test results, known blockers, and the next implementation action.

**Begin with repository inspection and then implement immediately.**

