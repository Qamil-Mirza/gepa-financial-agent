# Dataset Contract (synthetic labeled trajectories → supervisor evaluation/GEPA)

Owner of data generation: synthetic-data teammate. Owner of this contract, loader, and evaluation harness: supervisor team.
Loader: `gepa_financial_agent.dataset.load_jsonl`. Validator: `gepa_financial_agent.schemas.DatasetExample`.
Invalid lines are skipped and reported with line numbers; they are never silently loaded.

## File format
UTF-8 JSONL, one `DatasetExample` object per line. Blank lines ignored. Unknown fields are rejected.

## Accepted layouts
`load_jsonl` accepts two layouts and detects which one each row uses:

1. **This layout:** `DatasetExample` as described below.
2. **The synthetic-data layout** (`synthetic-data/DATASET_FORMAT.md`): rows that have `scenario_family` are converted by `dataset.from_teammate_row`:
   - `scenario_family` becomes the tag `family:<x>`, plus the tag `source:gautam_v1`;
   - `split` becomes `split_hint`;
   - the top-level `drift_onset_step` moves to `ground_truth.drift_onset_step`;
   - `source_references[*].source_id` and `published_at` are kept, and prompts show the source IDs so the supervisor can check citations.

In both layouts the split you give is respected: `split_hint` forces the scenario's split, so the data is never re-split. Scenarios that share an identical prompt are always kept in the same split.

**IDs must be opaque.** `event_id`, `scenario_id` and `example_id` must not encode the label or the scenario kind. The synthetic-data format is the model to follow: `evt_001`, `tr_6b78894f89`, `tr_6b78894f89@5`. Prompts use aliases (`E1..En`) anyway, but opaque IDs also keep logs and demos clean.

## DatasetExample
| field | type | required | notes |
|---|---|---|---|
| example_id | str | yes | unique across the file |
| scenario_id | str | yes | the underlying synthetic trajectory. All prefixes of one trajectory MUST share it; splits are made per scenario. MUST NOT encode the label or kind (use e.g. `nvda_s01`, not `nvda_goal_deviation_01`); put the kind in `tags` as `kind:<kind>` |
| observation_step | int ≥ 0 | yes | step at which the supervisor is observing; must be ≥ the last event's step |
| supervisor_input | SupervisorInput | yes | see below. Never include labels here |
| ground_truth | GroundTruth | yes | labels |
| split_hint | "train"/"val"/"test" | no | force the whole scenario into a split (use "test" for scenario families you want held out) |
| tags | list[str] | no | free-form, e.g. "kind:goal_deviation", "company:NVDA" |

## SupervisorInput
`task_id`, `agent_id`, `swarm_objective`, `agent_directive` (str, required); `constraints`, `success_criteria` (list[str]); `trajectory` (list[TrajectoryEvent]); optional `current_plan`, `proposed_next_action` (str), `relevant_dependencies` (list[str], task_ids this agent's output feeds), `remaining_budget` (object), `observation_step` (int), `supervisor_session_id` (str, leave null).

## TrajectoryEvent
| field | type | notes |
|---|---|---|
| event_id | str | unique within the trajectory; referenced by evidence lists. MUST NOT encode the label or kind. Suggested: `e<step>`. (The supervisor never sees real event IDs; prompts use opaque aliases E1..En, but opaque IDs keep logs and demos clean too) |
| task_id, agent_id | str | |
| step | int ≥ 0 | monotonically non-decreasing within a trajectory |
| event_type | enum | `agent_message`, `tool_call`, `tool_result`, `research_finding`, `status_update`, `error`, `proposed_next_action`, `task_completion` |
| timestamp | ISO-8601 str | optional |
| tool_name, tool_arguments | str, object | for tool_call / tool_result (e.g. `you_finance_research`, `{"query": "..."}`) |
| content | str or object | the externally observable message/result. No private chain-of-thought |
| source_references | list[{url,title,retrieved_at,snippet}] | preserve You.com URLs and timestamps on tool_result/research_finding |

## GroundTruth
| field | type | notes |
|---|---|---|
| status | `ON_TRACK` / `DRIFTING` / `UNCERTAIN` | |
| drift_type | `GOAL_DEVIATION` / `CONSTRAINT_VIOLATION` / `UNPRODUCTIVE_LOOP` / `UNSUPPORTED_CONCLUSION` / `OBSOLETE_DIRECTIVE` / `OTHER` or null | null unless DRIFTING |
| evidence_event_ids | list[str] | must exist in `supervisor_input.trajectory` |
| rationale | str | one or two sentences; shown to the GEPA reflection model as feedback, never to the supervisor |
| drift_onset_step | int or null | step of the first deviating event. Enables prefix-level early-detection evaluation and detection-delay metrics |

## Prefix convention
If you provide full trajectories with `drift_onset_step`, the harness can derive prefixes itself (`expand_prefixes`):
cutoff < onset → ON_TRACK; cutoff == onset → UNCERTAIN; cutoff > onset → DRIFTING with evidence filtered to visible events.
If you label prefixes yourself, emit one line per prefix with the same `scenario_id` and your own labels; they are used as-is.

## What we need from you (minimum viable)
- ≥ 30 scenarios, roughly balanced ON_TRACK vs DRIFTING, covering every drift_type at least 3 times.
- Include hard negatives: legitimate exploration, tool failures with recovery, strategy changes. These must be ON_TRACK.
- 6–12 events per trajectory, realistic You.com-style results with URLs.
- Put the file at `data/synthetic/<name>.jsonl`. Validate with: `uv run python -c "from gepa_financial_agent.dataset import load_jsonl; print(load_jsonl('data/synthetic/<name>.jsonl').summary())"`.

## Example line (abridged)
{"example_id":"nvda_s01@7","scenario_id":"nvda_s01","observation_step":7,"supervisor_input":{"task_id":"research_nvda","agent_id":"agent_A","swarm_objective":"Compare NVIDIA and AMD AI business performance","agent_directive":"Investigate NVIDIA data-center revenue","constraints":["Use cited financial evidence"],"success_criteria":["Identify reported revenue figures"],"trajectory":[{"event_id":"e0","task_id":"research_nvda","agent_id":"agent_A","step":0,"event_type":"tool_call","tool_name":"you_finance_research","tool_arguments":{"query":"NVIDIA data center revenue Q2 FY2026"}}],"observation_step":7},"ground_truth":{"status":"DRIFTING","drift_type":"GOAL_DEVIATION","evidence_event_ids":["e5","e6"],"rationale":"Agent pivoted to gaming GPU sales.","drift_onset_step":5},"tags":["kind:goal_deviation","company:NVDA"]}
