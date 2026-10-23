"""The dashboard API's response models (Pydantic). Each endpoint declares one as its `response_model`, so the OpenAPI
schema (`export_openapi`, committed as `ui/openapi.json`) types the UI's client."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class WorkerStatus(BaseModel):
    """The background worker: `enabled` is false under `--no-worker`. `last_scan_at` (epoch seconds) and
    `last_scan_runs` are from the last completed scan pass; `last_error` is the exception type name of the last
    failed step, cleared by the next completed scan."""
    enabled: bool
    running: bool
    last_scan_at: float | None = None
    last_scan_runs: int | None = None
    last_error: str | None = None
    scan_every_s: float | None = None


class Health(BaseModel):
    ok: bool
    version: str  # the pitstop package version
    schema_version: int  # the store's schema version
    home: str  # PITSTOP_HOME
    worker: WorkerStatus


# Times are epoch seconds and money is USD throughout. A value a malformed row cannot supply is null.

Flag = Literal["red", "yellow", "green"]


class RunRef(BaseModel):
    id: str
    status: str | None
    started_at: float | None
    ended_at: float | None


class LevelCounts(BaseModel):
    """Open drifts per P-level."""
    P0: int = 0
    P1: int = 0
    P2: int = 0
    P3: int = 0


class Tile(BaseModel):
    """One workflow on the pit wall (decision 6). `flag` is red for an open P0 or a latest run that is halted or
    refused, yellow for any other open drift, else green; `flag_label` says the same in words. `headline` is the most
    urgent open drift's alert title (`headline_level` its level). Spend is today's agent-meter spend against the
    latest run's daily `budget_usd` (decision 7)."""
    name: str
    criticality: str | None
    flag: Flag
    flag_label: str
    open: LevelCounts
    headline: str | None
    headline_level: str | None
    last_run: RunRef | None
    runs_today: int
    spend_today_usd: float
    budget_usd: float | None
    adapters: list[str]  # the adapters of its recent runs ("wrap", "otel", ...)
    fallback_active: bool


class BudgetRow(BaseModel):
    """A global budget from config.toml: spent (settled plus reserved) in its current period against its limit."""
    name: str
    scope: str  # "global:<name>"
    period: str
    spent_usd: float
    limit_usd: float
    remaining_usd: float


class Board(BaseModel):
    workflows: list[Tile]  # red, then yellow, then green; then the most urgent open level; then latest activity
    budgets: list[BudgetRow]
    now: float


class FeedItem(BaseModel):
    """One pit-wall feed entry. `id` is "<kind>:<row id>" (a drift id, an event id or a run id)."""
    id: str
    ts: float | None
    kind: Literal["drift", "halt", "fallback", "refused", "model_change"]
    workflow: str | None
    level: str | None  # drifts only
    text: str
    run_id: str | None
    drift_id: int | None


class DriftCard(BaseModel):
    """An open drift with its alert text: `title`, `cause` (what likely changed) and `doing` (what PitStop did)."""
    id: int
    level: str | None
    drift_type: str | None
    action: str | None
    occurrences: int
    first_seen: float | None
    last_seen: float | None
    title: str
    cause: str
    doing: str


class WorkflowDetail(BaseModel):
    tile: Tile
    goal: str | None  # from the latest run
    sensitivity: str | None
    fix_mode: str | None
    drifts: list[DriftCard]  # open drifts, most urgent first, at most 50


class DismissResult(BaseModel):
    status: Literal["dismissed", "not_found", "not_open"]


class TagChange(BaseModel):
    tag: str
    before: str | None
    after: str
    share_before: float
    share_after: float


class Explain(BaseModel):
    """L5 attribution around a drift's change point `at` (`pitstop explain`)."""
    summary: str
    changes: list[TagChange]
    metric: str | None
    before_mean: float | None
    after_mean: float | None
    n_before: int
    n_after: int
    at: float


class RunRow(BaseModel):
    """`cost_usd` sums the run's agent-meter `llm_call` costs and `llm_calls` counts those calls; `verdicts` counts
    its block, steer, halt and fallback events."""
    id: str
    status: str | None
    started_at: float | None
    ended_at: float | None
    duration_s: float | None
    model: str | None
    base_url: str | None
    adapter: str | None
    cost_usd: float
    llm_calls: int
    tool_calls: int
    verdicts: int


class RunPage(BaseModel):
    runs: list[RunRow]  # newest first
    next_before: float | None  # pass as `before` for the next page; null on the last page


class EventRow(BaseModel):
    """One event. `summary` is a one-line description of at most 300 characters; the raw payload is never sent."""
    id: int
    ts: float | None
    agent: str | None
    kind: str | None
    name: str | None
    cost_usd: float | None
    latency_s: float | None  # null when not measured
    summary: str


class RunDetail(BaseModel):
    run: RunRow
    workflow: str | None
    goal: str | None
    events: list[EventRow]  # oldest first, at most 2,000
    truncated: bool  # the run has more than 2,000 events


class WorkflowSettings(BaseModel):
    """A workflow's settings as its latest run recorded them (they are set in code)."""
    name: str
    criticality: str | None
    sensitivity: str | None
    fix_mode: str | None
    budget_usd: float | None


class JudgeSettings(BaseModel):
    model: str
    base_url: str  # without any credentials, query or fragment
    api_key_env: str  # the env var's name; its value is never sent
    key_set: bool


class ChannelSettings(BaseModel):
    name: str
    configured: bool  # its [alerts.<name>] table is present
    env_var: str  # the env var holding its secret; its value is never sent
    env_set: bool


class AlertSettings(BaseModel):
    channels: list[ChannelSettings]
    routes: dict[str, list[str]]  # P0, P1, digest -> channel names


class Settings(BaseModel):
    home: str
    workflows: list[WorkflowSettings]
    budgets: list[BudgetRow]
    judge: JudgeSettings
    alerts: AlertSettings
    worker: WorkerStatus


class GraphVerdicts(BaseModel):
    """Verdict events on a node: `gate_reject` counts gates with `allow` false, `tool_error` errored tool results."""
    block: int = 0
    steer: int = 0
    halt: int = 0
    fallback: int = 0
    gate_reject: int = 0
    gate_error: int = 0
    tool_error: int = 0


class GraphNode(BaseModel):
    """One run-graph node (decision 8). `id` is "workflow", "agent:<path>", "llm:<agent path>|<model>" or
    "tool:<agent path>|<tool name>", the path being the agent names joined with " › "; `agent_path` lists those names
    (for a leaf, its agent's). An agent's calls, cost and runs cover everything under it, and the root's too (its
    runs are every run in scope); their latency is not measured (null). `time_per_run_s` is the node's time over the
    runs in scope divided by their number, its unknown latencies counted at the mean of its known ones (an agent's:
    the sum of its children's), null when nothing under it has a known latency.
    `run_ids` are the runs in scope that touched the node, newest first, at most 20."""
    id: str
    kind: Literal["workflow", "agent", "llm", "tool"]
    label: str
    agent_path: list[str]
    calls: int
    runs: int
    cost_usd: float
    latency_p50_s: float | None
    time_per_run_s: float | None
    verdicts: GraphVerdicts
    on_slowest_path: bool
    run_ids: list[str]


class GraphEdge(BaseModel):
    source: str
    target: str


class Graph(BaseModel):
    """The run graph over one run (`run_id`, `window` null) or the newest runs started in `window`, at most 200.
    `slowest_path` lists node ids from the root, each step the child with the largest `time_per_run_s`; it is empty
    when no node has a known latency."""
    nodes: list[GraphNode]  # the root first, then depth-first by id
    edges: list[GraphEdge]
    runs_in_scope: int
    window: str | None
    run_id: str | None
    slowest_path: list[str]
