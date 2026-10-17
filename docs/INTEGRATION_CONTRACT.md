# Integration Contract: attaching supervision to the financial research swarm

The supervisor never owns worker execution, research tools, or orchestration. The host stack only has to:
1. register each agent/task, 2. emit normalized events, 3. optionally submit proposed actions, 4. read decisions,
5. optionally provide dependency metadata. Supervision is nonblocking and observational in V1.

## Python API (in-process)
```python
import asyncio
from gepa_financial_agent import SupervisorManager, CrusoeClient, TriggerConfig
from gepa_financial_agent.adapters import EventStreamAdapter
from gepa_financial_agent.config import Settings

client = CrusoeClient.from_settings(Settings.from_env(), "supervisor")       # shared, bounded concurrency
manager = SupervisorManager(client, policy_path="policies/current.txt", max_concurrent_inferences=4,
                            trigger=TriggerConfig(every_n_events=4))

manager.register_agent(agent_id="agent_A", task_id="research_nvda",
                       directive="Research NVIDIA data-center revenue",
                       swarm_objective="Compare NVIDIA and AMD AI businesses",
                       constraints=["Use cited financial evidence"], success_criteria=["Identify reported revenue figures"],
                       dependencies=["synthesize_report"])                   # optional DAG metadata

# Option 1: direct calls from your orchestrator loop
await manager.observe("agent_A", {"event_id": "nvda_e0", "task_id": "research_nvda", "agent_id": "agent_A", "step": 0,
                                   "event_type": "tool_call", "tool_name": "you_finance_research",
                                   "tool_arguments": {"query": "NVIDIA data center revenue Q2 FY2026"}})
decisions = await manager.evaluate_pending()        # only sessions whose triggers fired; coalesced per agent
latest = manager.latest("agent_A")                  # EvaluationRecord: .decision (SupervisorDecision) or .error

# Option 2: event stream with callbacks (recommended for concurrent workers)
adapter = EventStreamAdapter(manager, eval_interval_s=0.5)
adapter.on_decision(lambda rec: print(rec.agent_id, rec.decision.status if rec.decision else rec.error))
stop = asyncio.Event()
consumer = asyncio.create_task(adapter.run(stop))
await adapter.emit("agent_A", event_dict)           # from any coroutine; bounded queue gives backpressure
...
await adapter.drain(); stop.set(); await consumer
manager.close_session("agent_A")
```

## Event schema
Exactly `TrajectoryEvent` in `gepa_financial_agent/schemas.py` (unknown fields rejected):
`event_id` (unique per agent), `task_id`, `agent_id`, `step` (int, non-decreasing), `event_type` ∈
{agent_message, tool_call, tool_result, research_finding, status_update, error, proposed_next_action, task_completion},
optional `timestamp` (ISO-8601), `tool_name`, `tool_arguments` (object), `content` (string or object),
`source_references` (list of {url, title, retrieved_at, snippet}). Put You.com question text in `tool_arguments.query`,
results in `tool_result.content` with `source_references`, and the agent's own conclusions in `research_finding`.
Use `status_update` for coordinator-issued directive changes. Do not send private chain-of-thought.

## Decision schema
`SupervisorDecision`: `agent_id`, `task_id`, `status` ∈ {ON_TRACK, DRIFTING, UNCERTAIN}, `confidence` (0..1, uncalibrated),
`drift_type` (null unless DRIFTING), `evidence_event_ids` (validated against the trajectory), `explanation`,
`recommended_action` ∈ {NONE, CONTINUE_MONITORING, FLAG_FOR_REVIEW, ESCALATE}, `policy_version`, `observation_step`.
Wrapped in `EvaluationRecord` with `error` (malformed output / model failure), `latency_ms`, token counts, `mocked`.
Classification is separate from intervention: V1 never blocks or edits a worker.

## Triggers (configurable, `TriggerConfig`)
every N events (default 4); immediately on `error`, `proposed_next_action`, `task_completion`; identical tool_call repeated
≥3 times. Pending evaluations coalesce: one inference per agent per `evaluate_pending()` regardless of event count.

## Policy versions
Sessions pin the policy they were registered with. `manager.set_policy(PolicyStore().load("gepa_<run>"))` affects new
sessions only. `manager.export_session(agent_id)` / `await manager.replay(export)` reproduce a session offline.

## Out of process
If the stack is not Python or not in-process, write events as JSONL (one TrajectoryEvent per line) and we replay them with
`ReplayAgent` + `EventStreamAdapter`; see `scripts/demo_supervision.py`. A thin HTTP/WebSocket shim is a Milestone F item.
