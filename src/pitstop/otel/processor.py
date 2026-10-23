"""`SpanProcessor`: read the GenAI spans an app already traces with OpenTelemetry (Langfuse, Logfire, OpenLLMetry,
OpenInference...) into PitStop's runs and events, run the L0 checks on them and put each verdict back on the trace.

One run per trace (id `otel-<32-hex trace id>`, adapter `otel`). The trace is followed from its local root span (no
parent, or a remote one); its run row is written at the trace's first GenAI span (llm, tool or agent), with the root's
start time, so a trace of plain spans (an HTTP health check) writes nothing. The run closes when the root ends (`ok`, or
`error` when the root's status is ERROR); a trace dropped past MAX_OPEN_TRACES closes as `dropped`. The workflow name
is the root's `pitstop.workflow` attribute (set at span start), else `SpanProcessor(workflow=...)`, else the root
span's name. LLM spans become `llm_call` events (the same payload keys as `wrap`, plus `source: "otel"`), tool spans
`tool_call` and `tool_result` events, agent spans `agent` events; each event carries its span id and parent span id
(hex) and the nearest agent ancestor's name. An agent span is known by its attributes at start, and again at its end
(an agent name set after the span started names its own event and the spans that end after it). The first LLM span
that names a model sets the run's model and base_url (`https://<server.address>`).

A GenAI span that ends after its root (a straggling child) still writes its events to the run, which keeps the
status its root gave it; when the run row was never written before the root ended, it is not written then (the
last MAX_ENDED_TRACES ended traces are kept for stragglers).

Observe-only: findings are recorded with `observe_only=True` (action `record`; P0/P1 still alert), each adds a
`pitstop.verdict` event to the trace's root span while it is open, and the cost on an `llm_call` event is a price
table estimate that never reaches the budget ledger. Spans that start or end while a PitStop workflow run is current
are skipped: `wrap` and `@pitstop.tool` already record them. Every entry point swallows its own failures (one warning
per kind of failure per processor, also for the alerts of an unrecorded P0), so the traced app never sees an
exception from PitStop."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor as _SdkSpanProcessor
from opentelemetry.trace import StatusCode

from ..sdk.context import HISTORY_MAX, WorkflowSpec, current_run
from ..sdk.runtime import get_runtime
from ..store.urls import normalize_base_url
from .semconv import SpanView, read_span

log = logging.getLogger("pitstop")

MAX_OPEN_TRACES = 1000  # traces whose root is still open; the oldest is dropped past this
MAX_SPANS_PER_TRACE = 10_000  # spans mapped (parent, agent) per trace; later ones still record events
MAX_ENDED_TRACES = 100  # traces whose root ended, kept for GenAI spans that end after it; the oldest is forgotten
_MAX_ARGS = 500  # tool arguments kept on a tool_call event, like the tool guard's
_REFUSAL_REASONS = ("content_filter", "refusal")


@dataclass
class _Trace:
    run_id: str
    root: Any  # the live root span: verdict events can only go on a span that has not ended
    root_span_id: int
    spec: WorkflowSpec
    started_at: float | None  # the root's start (s): the run's started_at, written with its first GenAI span
    inserted: bool = False  # the run row was written: only then is it finalized
    spans: dict[int, tuple[int | None, str | None]] = field(default_factory=dict)  # span -> (parent, agent name)
    tool_history: dict[str | None, deque] = field(default_factory=dict)
    llm_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_MAX))
    refusals: int = 0
    tagged: bool = False  # the run's model and base_url are set
    ended: bool = False  # the root ended (or the trace was dropped): the run row is never inserted after this
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # orders the insert and the close


def _hex_span(span_id: int | None) -> str | None:
    return format(span_id, "016x") if span_id is not None else None


def _seconds(ns: int | None) -> float | None:
    return ns / 1e9 if ns is not None else None


def _base_url(server: str | None) -> str | None:
    if not server:
        return None
    return normalize_base_url(server if "://" in server else f"https://{server}")


class SpanProcessor(_SdkSpanProcessor):
    """Add to a tracer provider: `provider.add_span_processor(SpanProcessor(workflow="support"))`. Add it before
    the exporting processor so the root span is exported with its `pitstop.verdict` events. `goal`, `criticality`,
    `sensitivity` and `tools` apply to every workflow this processor names; invalid values raise ValueError here.
    Tool spans are checked as read-only tools (a span does not say whether the tool had side effects)."""

    def __init__(self, workflow: str | None = None, *, goal: str | None = None, criticality: str = "medium",
                 sensitivity: str = "normal", tools: Sequence[str] | None = None):
        self.workflow = workflow
        self._spec_args = dict(goal=goal, criticality=criticality, sensitivity=sensitivity,
                               tools=tuple(tools) if tools is not None else None)
        WorkflowSpec(workflow or "otel", **self._spec_args)  # validate now, not on the first span
        self._specs: dict[str, WorkflowSpec] = {}
        self._traces: OrderedDict[int, _Trace] = OrderedDict()
        self._ended: OrderedDict[int, _Trace] = OrderedDict()  # for spans that end after their root
        self._lock = threading.RLock()
        self._warned: set[str] = set()
        self._warn_lock = threading.Lock()

    # -- SpanProcessor interface: never raises -------------------------------------------------------------------

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            if current_run() is not None:
                return
            self._start(span)
        except Exception:
            self._warn("span", "pitstop: could not read an OpenTelemetry span; ignored")

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._end(span)
        except Exception:
            self._warn("span", "pitstop: could not read an OpenTelemetry span; ignored")

    def shutdown(self) -> None:
        try:
            with self._lock:
                self._traces.clear()
                self._ended.clear()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # -- internals -----------------------------------------------------------------------------------------------

    def _warn(self, key: str, msg: str, exc_info: bool = True) -> None:
        """Log `msg` (with the current traceback) once per processor and key; later failures only at debug level.
        Thread-safe: the alerts pool calls it too."""
        with self._warn_lock:
            first = key not in self._warned
            self._warned.add(key)
        log.log(logging.WARNING if first else logging.DEBUG, msg, exc_info=exc_info)

    def _spec(self, name: str) -> WorkflowSpec:
        spec = self._specs.get(name)
        if spec is None:
            spec = WorkflowSpec(name, **self._spec_args)
            if len(self._specs) < MAX_OPEN_TRACES:  # root span names can be high-cardinality: cache a bounded few
                self._specs[name] = spec
        return spec

    def _start(self, span: Any) -> None:
        sc = span.context
        trace_id, span_id = sc.trace_id, sc.span_id
        parent = span.parent
        parent_id = parent.span_id if parent is not None else None
        attrs = span.attributes or {}
        local_root = parent is None or parent.is_remote
        dropped = None
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None and not local_root:
                return  # a trace this processor does not follow (its root ran inside a PitStop run, or was dropped)
            if trace is None:
                name = attrs.get("pitstop.workflow")
                name = name[:200] if isinstance(name, str) and name else self.workflow or span.name or "otel"
                trace = _Trace(f"otel-{trace_id:032x}", span, span_id, self._spec(name), _seconds(span.start_time))
                self._traces[trace_id] = trace
                if len(self._traces) > MAX_OPEN_TRACES:
                    dropped = self._traces.popitem(last=False)[1]
            if len(trace.spans) < MAX_SPANS_PER_TRACE:
                view = read_span(span.name, attrs, False)
                agent = (view.agent_name or span.name) if view.kind == "agent" else None
                trace.spans[span_id] = (parent_id, agent)
        if dropped is None:
            return
        self._warn("bound", f"pitstop: more than {MAX_OPEN_TRACES} open traces; the oldest ({dropped.run_id}) is no"
                            " longer followed", exc_info=False)
        self._close(dropped, time.time(), "dropped")

    def _ensure_run(self, trace: _Trace) -> bool:
        """Write the trace's run row at its first GenAI span (again after a failed write; INSERT OR IGNORE). False
        when the root already ended without one: a run row is never inserted after its root ended (it would stay
        `running`). Under the trace's lock, so the insert and the root's close never interleave."""
        from .. import __version__

        with trace.lock:
            if trace.inserted:
                return True
            if trace.ended:
                return False
            s = trace.spec
            try:
                with get_runtime().store.transaction() as c:
                    c.execute("INSERT OR IGNORE INTO runs(id, workflow, goal, criticality, sensitivity, fix_mode,"
                              " started_at, status, pitstop_version, adapter) VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (trace.run_id, s.name, s.goal, s.criticality, s.sensitivity, s.effective_fix_mode,
                               trace.started_at, "running", __version__, "otel"))
                trace.inserted = True
            except Exception:
                self._warn("store", "pitstop: could not write an OpenTelemetry trace to the store; ignored")
            return True  # a failed insert is retried at the next span; this span's checks still run

    def _close(self, trace: _Trace, ended_at: float | None, status: str) -> None:
        """Mark the trace ended and give its run row (when written) its final status."""
        with trace.lock:
            trace.ended = True
            inserted = trace.inserted
        if inserted:
            self._write("UPDATE runs SET ended_at=?, status=? WHERE id=?", (ended_at, status, trace.run_id))

    def _end(self, span: ReadableSpan) -> None:
        sc = span.context
        with self._lock:
            trace = self._traces.get(sc.trace_id)
            is_root = trace is not None and sc.span_id == trace.root_span_id
            if trace is None:
                trace = self._ended.get(sc.trace_id)  # a span ending after its root
                if trace is None:
                    return
            elif is_root:
                del self._traces[sc.trace_id]
                self._ended[sc.trace_id] = trace
                if len(self._ended) > MAX_ENDED_TRACES:
                    self._ended.popitem(last=False)
        error = span.status is not None and span.status.status_code is StatusCode.ERROR
        if current_run() is None:
            try:
                view = read_span(span.name, span.attributes or {}, error)  # end-time attributes
                self._name_agent(trace, span, view)
                self._record(trace, span, view)
            except Exception:
                self._warn("span", "pitstop: could not read an OpenTelemetry span; ignored")
        if is_root:
            self._close(trace, _seconds(span.end_time), "error" if error else "ok")

    def _name_agent(self, trace: _Trace, span: ReadableSpan, view: SpanView) -> None:
        """An agent span's identity from its end-time attributes: a name (or the agent operation itself) set after
        the span started names its own event and the spans that end after it."""
        if view.kind != "agent":
            return
        span_id = span.context.span_id
        parent_id = span.parent.span_id if span.parent is not None else None
        with self._lock:
            if span_id in trace.spans or len(trace.spans) < MAX_SPANS_PER_TRACE:
                trace.spans[span_id] = (parent_id, view.agent_name or span.name)

    def _agent_of(self, trace: _Trace, span_id: int, parent_id: int | None, *, above: bool = False) -> str | None:
        """The nearest agent span's name, from this span up (from its parent up when `above`), walking through spans
        PitStop does not record (bounded walk: a broken chain cannot loop)."""
        with self._lock:
            entry = trace.spans.get(span_id)
            if not above and entry is not None and entry[1] is not None:
                return entry[1]
            cur = parent_id
            for _ in range(len(trace.spans) + 1):
                if cur is None:
                    return None
                entry = trace.spans.get(cur)
                if entry is None:
                    return None
                if entry[1] is not None:
                    return entry[1]
                cur = entry[0]
        return None

    def _record(self, trace: _Trace, span: ReadableSpan, view: SpanView) -> None:
        if view.kind == "other" or not self._ensure_run(trace):
            return
        span_id = span.context.span_id
        parent_id = span.parent.span_id if span.parent is not None else None
        agent = self._agent_of(trace, span_id, parent_id)
        link = (_hex_span(span_id), _hex_span(parent_id))
        if view.kind == "llm":
            findings = self._llm(trace, span, view, agent, link)
        elif view.kind == "tool":
            findings = self._tool(trace, span, view, agent, link)
        else:
            # its parent agent by name: the run graph nests it by this when its parent span was not recorded
            payload: dict[str, Any] = {"parent": self._agent_of(trace, span_id, parent_id, above=True),
                                       "source": "otel"}
            if view.error is not None:
                payload["error"] = view.error
            self._event(trace, _seconds(span.start_time), agent, "agent", view.agent_name or span.name, payload,
                        None, link)
            findings = []
        if findings:
            from ..sdk.actions import apply_findings

            applied = apply_findings(None, findings, run_id=trace.run_id, spec=trace.spec, observe_only=True,
                                     verdict_span=trace.root, quiet=True,
                                     warn=lambda msg: self._warn("alert", msg, exc_info=False))
            if any(a.drift_id is None for a in applied):
                self._warn("store", "pitstop: could not write an OpenTelemetry trace to the store; ignored",
                           exc_info=False)

    def _llm(self, trace: _Trace, span: ReadableSpan, view: SpanView, agent: str | None, link: tuple) -> list:
        from ..budget.prices import cost_usd, price_for
        from ..detect.l0 import LLMCallRecord, check_llm_call

        tool_calls = 1 if "tool_calls" in view.finish_reasons else 0  # span attributes carry no count
        refused = any(r in _REFUSAL_REASONS for r in view.finish_reasons)
        prompt, completion = view.input_tokens, view.output_tokens
        cost, known = None, None
        if view.model is not None:
            price, known = price_for(view.model, get_runtime().config.prices)
            if prompt is not None or completion is not None:
                cost = cost_usd(price, prompt or 0, completion or 0)
        payload = {"tool_calls": tool_calls, "finish_reason": view.finish_reasons[0] if view.finish_reasons else None,
                   "content_chars": None, "meter": "agent", "prompt_tokens": prompt, "completion_tokens": completion,
                   "refused": refused, "source": "otel"}
        self._latency(payload, span)
        record = LLMCallRecord(0, tool_calls, refused)
        with self._lock:
            first = view.model is not None and not trace.tagged
            trace.tagged = trace.tagged or first
            if refused:
                trace.refusals += 1
            findings = check_llm_call(trace.llm_history, record, trace.spec.sensitivity, trace.refusals)
            trace.llm_history.append(record)
        self._event(trace, _seconds(span.end_time), agent, "llm_call", view.model, payload, cost, link)
        if first:
            self._write("UPDATE runs SET model=?, base_url=?, price_known=CASE WHEN ? THEN price_known ELSE 0 END"
                        " WHERE id=?", (view.model, _base_url(view.server), known is not False, trace.run_id))
        elif known is False:
            self._write("UPDATE runs SET price_known=0 WHERE id=?", (trace.run_id,))
        return findings

    def _tool(self, trace: _Trace, span: ReadableSpan, view: SpanView, agent: str | None, link: tuple) -> list:
        from ..detect.l0 import ToolCallRecord, canonical_args, check_tool_call

        name = view.tool_name or span.name
        args = canonical_args(view.tool_args) if view.tool_args is not None else ""
        record = ToolCallRecord(name, args, False)
        with self._lock:
            history = trace.tool_history.get(agent)
            if history is None:
                history = trace.tool_history.setdefault(agent, deque(maxlen=HISTORY_MAX))
            findings = check_tool_call(history, record, trace.spec)
            history.append(record)
        self._event(trace, _seconds(span.start_time), agent, "tool_call", name,
                    {"args": args[:_MAX_ARGS], "source": "otel"}, None, link)
        result: dict[str, Any] = {"source": "otel"}
        if view.error is not None:
            result["error"] = view.error
        self._latency(result, span)
        self._event(trace, _seconds(span.end_time), agent, "tool_result", name, result, None, link)
        return findings

    @staticmethod
    def _latency(payload: dict, span: ReadableSpan) -> None:
        if span.start_time is not None and span.end_time is not None:
            payload["latency_s"] = round(max(span.end_time - span.start_time, 0) / 1e9, 4)

    def _event(self, trace: _Trace, ts: float | None, agent: str | None, kind: str, name: str | None,
               payload: dict, cost: float | None, link: tuple) -> None:
        self._write("INSERT INTO events(run_id, ts, agent, kind, name, payload, cost_usd, span_id, parent_span_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (trace.run_id, ts, agent, kind, name, json.dumps(payload, default=str), cost, *link))

    def _write(self, sql: str, params: tuple) -> None:
        try:
            with get_runtime().store.transaction() as c:
                c.execute(sql, params)
        except Exception:
            self._warn("store", "pitstop: could not write an OpenTelemetry trace to the store; ignored")
