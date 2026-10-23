import json
import logging
import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import pitstop
from pitstop.otel import SpanProcessor
from pitstop.otel import processor as processor_mod
from pitstop.sdk.runtime import get_runtime


@pytest.fixture
def home(tmp_path):
    (tmp_path / "config.toml").write_text("")
    pitstop.init(home=tmp_path)
    return tmp_path


def _tracing(**kwargs):
    """PitStop's processor first, so its verdict events are on the root before the exporter sees it."""
    proc = SpanProcessor(**kwargs)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter, proc


def _db():
    return get_runtime().store.connect()


def _runs():
    return [dict(r) for r in _db().execute("SELECT * FROM runs ORDER BY started_at")]


def _events(run_id):
    return [dict(r) for r in _db().execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]


CHAT = {"gen_ai.operation.name": "chat", "gen_ai.request.model": "openai/gpt-oss-120b",
        "server.address": "api.crusoe.ai", "gen_ai.usage.input_tokens": 1000, "gen_ai.usage.output_tokens": 500,
        "gen_ai.response.finish_reasons": ("tool_calls",)}


def _tool(tracer, name="search", args='{"q": "nvda"}'):
    with tracer.start_as_current_span(f"execute_tool {name}", attributes={
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args}):
        pass


def test_trace_becomes_run_with_events(home):
    tracer, exporter, _ = _tracing(workflow="research")
    with tracer.start_as_current_span("invoke_agent analyst", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "analyst"}) as root:
        with tracer.start_as_current_span("chat openai/gpt-oss-120b", attributes=CHAT):
            pass
        _tool(tracer)
    trace_hex = format(root.get_span_context().trace_id, "032x")
    [run] = _runs()
    assert run["id"] == f"otel-{trace_hex}" and run["workflow"] == "research" and run["adapter"] == "otel"
    assert run["model"] == "openai/gpt-oss-120b" and run["base_url"] == "https://api.crusoe.ai"
    assert run["status"] == "ok" and run["ended_at"] >= run["started_at"]
    events = _events(run["id"])
    assert [e["kind"] for e in events] == ["llm_call", "tool_call", "tool_result", "agent"]
    root_id = format(root.get_span_context().span_id, "016x")
    llm, call, result, agent = events
    assert agent["name"] == "analyst" and agent["span_id"] == root_id and agent["parent_span_id"] is None
    assert llm["parent_span_id"] == root_id and call["parent_span_id"] == root_id
    assert call["span_id"] == result["span_id"] != llm["span_id"]
    assert llm["agent"] == call["agent"] == result["agent"] == "analyst"
    payload = json.loads(llm["payload"])
    assert payload.pop("latency_s") >= 0
    assert payload == {"tool_calls": 1, "finish_reason": "tool_calls", "content_chars": None, "meter": "agent",
                       "prompt_tokens": 1000, "completion_tokens": 500, "refused": False, "source": "otel"}
    assert llm["cost_usd"] == pytest.approx((1000 * 0.05 + 500 * 0.20) / 1e6)
    assert json.loads(call["payload"])["args"] == '{"q":"nvda"}'
    assert "error" not in json.loads(result["payload"])


def test_tool_error_and_root_error(home):
    tracer, _, _ = _tracing()
    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("job"):
            with tracer.start_as_current_span("execute_tool fetch", attributes={
                    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "fetch"}) as span:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", "TimeoutError")
            raise RuntimeError("boom")
    [run] = _runs()
    assert run["status"] == "error" and run["workflow"] == "job"
    kinds = {e["kind"]: json.loads(e["payload"]) for e in _events(run["id"])}
    assert kinds["tool_result"]["error"] == "TimeoutError"


AGENT = {"gen_ai.operation.name": "invoke_agent"}


def test_workflow_name_resolution(home):
    tracer, _, _ = _tracing(workflow="from-arg")
    with tracer.start_as_current_span("root-a", attributes={**AGENT, "pitstop.workflow": "from-attr"}):
        pass
    with tracer.start_as_current_span("root-b", attributes=AGENT):
        pass
    tracer2, _, _ = _tracing()
    with tracer2.start_as_current_span("root-c", attributes=AGENT):
        pass
    assert [r["workflow"] for r in _runs()] == ["from-attr", "from-arg", "root-c"]


def test_tool_loop_in_trace_records_drift_and_root_event(home):
    tracer, exporter, _ = _tracing(workflow="loopy")
    with tracer.start_as_current_span("invoke_agent a", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"}) as root:
        for _ in range(3):
            _tool(tracer)
    run_id = "otel-" + format(root.get_span_context().trace_id, "032x")
    [drift] = [dict(r) for r in _db().execute("SELECT * FROM drifts")]
    assert drift["drift_type"] == "unproductive_loop" and drift["action"] == "record"
    assert drift["level"] == "P1" and drift["workflow"] == "loopy" and drift["run_id"] == run_id
    spans = {s.name: s for s in exporter.get_finished_spans()}
    [event] = [e for e in spans["invoke_agent a"].events if e.name == "pitstop.verdict"]
    assert event.attributes["pitstop.drift_type"] == "unproductive_loop"
    assert event.attributes["pitstop.action"] == "record" and event.attributes["pitstop.run_id"] == run_id
    assert all(not s.events for name, s in spans.items() if name != "invoke_agent a")


def test_refusals_in_trace_record_refusal_spike(home):
    tracer, _, _ = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("chat", attributes={"gen_ai.operation.name": "chat"}):
            pass  # no model: does not tag the run
        for _ in range(2):
            with tracer.start_as_current_span(
                "chat", attributes={**CHAT, "gen_ai.response.finish_reasons": ("content_filter",)}
            ):
                pass
    [run] = _runs()
    assert run["model"] == "openai/gpt-oss-120b"
    refused = [json.loads(e["payload"])["refused"] for e in _events(run["id"]) if e["kind"] == "llm_call"]
    assert refused == [False, True, True]
    [drift] = [dict(r) for r in _db().execute("SELECT * FROM drifts")]
    assert drift["drift_type"] == "refusal_spike" and drift["action"] == "record"


def test_spans_inside_pitstop_run_skipped(home):
    tracer, _, _ = _tracing()
    with pitstop.workflow("sdk-run"):
        with tracer.start_as_current_span("inside", attributes=CHAT):
            pass
    [run] = _runs()
    assert run["workflow"] == "sdk-run" and run["adapter"] is None
    assert _db().execute("SELECT COUNT(*) FROM events WHERE run_id LIKE 'otel-%'").fetchone()[0] == 0


def test_processor_never_writes_ledger(home):
    tracer, _, _ = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        for _ in range(3):
            with tracer.start_as_current_span("chat", attributes=CHAT):
                pass
    assert _db().execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 0
    assert _db().execute("SELECT COUNT(*) FROM events WHERE kind='llm_call'").fetchone()[0] == 3


def test_processor_survives_bad_attributes_and_store_errors(home, monkeypatch, caplog):
    tracer, _, _ = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("chat", attributes={**CHAT, "gen_ai.usage.input_tokens": "lots",
                                                              "gen_ai.usage.output_tokens": "1"}):
            pass
        _tool(tracer, args="x" * 10**6)
    [run] = _runs()
    llm = next(e for e in _events(run["id"]) if e["kind"] == "llm_call")
    payload = json.loads(llm["payload"])
    assert payload["prompt_tokens"] is None and payload["completion_tokens"] is None and llm["cost_usd"] is None
    call = next(e for e in _events(run["id"]) if e["kind"] == "tool_call")
    assert len(json.loads(call["payload"])["args"]) <= 500

    store = get_runtime().store

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "transaction", locked)
    caplog.set_level(logging.WARNING, logger="pitstop")
    with tracer.start_as_current_span("root2"):
        with tracer.start_as_current_span("chat", attributes=CHAT):
            pass
        for _ in range(3):  # a loop finding whose drift write fails too
            _tool(tracer)
    warnings =[r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_processor_never_raises_from_entry_points(home, monkeypatch):
    tracer, _, proc = _tracing()

    def boom(*a, **k):
        raise RuntimeError("internal")

    monkeypatch.setattr(processor_mod, "read_span", boom)
    monkeypatch.setattr(processor_mod, "current_run", boom)
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("chat", attributes=CHAT):
            pass
    assert proc.force_flush() is True
    proc.shutdown()


def test_span_read_failures_on_start_and_end_warn_once(home, monkeypatch, caplog):
    tracer, _, proc = _tracing()

    def boom(*a, **k):
        raise RuntimeError("internal")

    monkeypatch.setattr(processor_mod, "read_span", boom)
    caplog.set_level(logging.WARNING, logger="pitstop")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("chat", attributes=CHAT):
            pass
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == ["pitstop: could not read an OpenTelemetry span; ignored"]


def test_trace_without_genai_spans_writes_nothing(home):
    tracer, _, proc = _tracing(workflow="w")
    with tracer.start_as_current_span("GET /health"):
        with tracer.start_as_current_span("db.query"):
            pass
    assert _runs() == [] and proc._traces == {}
    assert _db().execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_open_traces_bounded(home, monkeypatch, caplog):
    from opentelemetry.trace import set_span_in_context

    monkeypatch.setattr(processor_mod, "MAX_OPEN_TRACES", 2)
    tracer, _, proc = _tracing(workflow="w")
    caplog.set_level(logging.WARNING, logger="pitstop")
    roots = []
    for i in range(4):
        roots.append(tracer.start_span(f"r{i}"))
        if i != 1:  # r1 has no GenAI span: no run row to close when it is dropped
            tracer.start_span("chat", context=set_span_in_context(roots[-1]), attributes=CHAT).end()
    assert len(proc._traces) == 2
    for r in roots:
        r.end()
    runs = _runs()
    assert [r["status"] for r in runs] == ["dropped", "ok", "ok"]
    assert runs[0]["ended_at"] is not None
    assert len([r for r in caplog.records if "open traces" in r.getMessage()]) == 1
    assert proc._traces == {}


def test_spans_per_trace_bounded(home, monkeypatch):
    monkeypatch.setattr(processor_mod, "MAX_SPANS_PER_TRACE", 3)
    tracer, _, proc = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        for _ in range(5):
            with tracer.start_as_current_span("chat", attributes=CHAT):
                assert all(len(t.spans) <= 3 for t in proc._traces.values())
    assert _db().execute("SELECT COUNT(*) FROM events WHERE kind='llm_call'").fetchone()[0] == 5


def test_invalid_spec_rejected_at_construction():
    with pytest.raises(ValueError):
        SpanProcessor(criticality="High")


def test_import_pitstop_without_extras(tmp_path):
    code = textwrap.dedent("""
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("opentelemetry", "agents", "langchain"):
                    raise ImportError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, Block())
        import pitstop
        import pitstop.otel
        from pitstop.otel.emit import emit_verdict
        assert callable(pitstop.wrap) and callable(pitstop.workflow)
        emit_verdict(run_id="r", drift_type="t", level="P2", action="record", drift_id=None, reason="r")
        try:
            pitstop.otel.SpanProcessor
        except ImportError as e:
            assert "pitstop[otel]" in str(e), e
        else:
            raise AssertionError("SpanProcessor imported without opentelemetry")
        print(sorted(m for m in sys.modules if m.split(".")[0] in ("opentelemetry", "agents", "langchain")))
    """)
    keys = ("KEY", "TOKEN", "SECRET", "PASSWORD", "WEBHOOK")
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in keys)}
    env["PITSTOP_HOME"] = str(tmp_path / "home")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


TOOL = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": "{}"}


def test_agent_name_from_end_attributes(home):
    """An agent name (or the agent operation itself) set on a span after it started is read at its end."""
    tracer, _, _ = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("invoke_agent", attributes=AGENT) as a:
            a.set_attribute("gen_ai.agent.name", "analyst")
        with tracer.start_as_current_span("worker") as b:  # an agent span only by its end-time attributes
            b.set_attribute("gen_ai.operation.name", "invoke_agent")
            b.set_attribute("gen_ai.agent.name", "writer")
    [run] = _runs()
    agents = [(e["name"], e["agent"]) for e in _events(run["id"]) if e["kind"] == "agent"]
    assert agents == [("analyst", "analyst"), ("writer", "writer")]


def test_child_ending_after_root_keeps_status(home):
    from opentelemetry.trace import set_span_in_context

    tracer, _, proc = _tracing(workflow="w")
    root = tracer.start_span("root")
    ctx = set_span_in_context(root)
    tracer.start_span("chat", context=ctx, attributes=CHAT).end()  # the run row is written
    late_tool = tracer.start_span("execute_tool search", context=ctx, attributes=TOOL)
    late_llm = tracer.start_span("chat", context=ctx, attributes=CHAT)
    root.set_status(Status(StatusCode.ERROR))
    root.end()
    late_tool.end()
    late_llm.end()
    [run] = _runs()
    assert run["status"] == "error" and run["ended_at"] is not None
    assert [e["kind"] for e in _events(run["id"])] == ["llm_call", "tool_call", "tool_result", "llm_call"]

    # A trace whose only GenAI span ends after its root: no run row is inserted after the root ended.
    root2 = tracer.start_span("root2")
    late = tracer.start_span("chat", context=set_span_in_context(root2), attributes=CHAT)
    root2.end()
    late.end()
    assert len(_runs()) == 1

    # The race: a child's end found the open trace, then its root ended before the child wrote the run row.
    root3 = tracer.start_span("root3")
    racing = tracer.start_span("chat", context=set_span_in_context(root3), attributes=CHAT)
    real = proc._record

    def record(trace, span, view):
        if span.context.span_id == racing.get_span_context().span_id and not root3.end_time:
            root3.end()  # the root's on_end runs in between (re-entrant here, another thread in production)
        real(trace, span, view)

    proc._record = record
    racing.end()
    assert [r["status"] for r in _runs()] == ["error"]  # never a run left `running`
    assert proc._traces == {}


def test_concurrent_span_ends_from_threads(home):
    import threading

    from opentelemetry.trace import set_span_in_context

    tracer, _, proc = _tracing(workflow="w")
    roots = [tracer.start_span(f"root{i}") for i in range(4)]
    ctxs = [set_span_in_context(r) for r in roots]
    for c in ctxs:  # each trace has its run row before the roots start ending
        tracer.start_span("chat", context=c, attributes=CHAT).end()
    spans = [[tracer.start_span("chat", context=ctxs[(t * 50 + j) % 4], attributes=CHAT) for j in range(50)]
             for t in range(8)]
    barrier = threading.Barrier(9)
    errors = []

    def end_spans(batch):
        try:
            barrier.wait(10)
            for s in batch:
                s.end()
        except BaseException as e:
            errors.append(e)

    def end_roots():
        try:
            barrier.wait(10)
            for r in roots:
                r.end()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=end_spans, args=(b,)) for b in spans] + [threading.Thread(target=end_roots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    runs = _runs()
    assert len(runs) == 4
    for r in runs:
        assert r["status"] == "ok"
        assert len(_events(r["id"])) >= 100
    assert proc._traces == {}


def test_unrecorded_p0_warning_once(tmp_path, monkeypatch, caplog):
    """A P0 whose drift (and alert row) the store refuses still pages each time, but its failures are logged
    through the processor's once-per-key gate."""
    import httpx

    from pitstop.alerts import channels, dispatch

    (tmp_path / "config.toml").write_text("[alerts.webhook]\n")
    monkeypatch.setenv("PITSTOP_WEBHOOK_URL", "https://hook.test/in")
    pitstop.init(home=tmp_path)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(channels, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    tracer, _, _ = _tracing(workflow="w", criticality="high")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("chat", attributes=CHAT):
            pass

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(get_runtime().store, "transaction", locked)
    caplog.set_level(logging.WARNING, logger="pitstop")
    for _ in range(2):
        with tracer.start_as_current_span("root"):
            for _ in range(3):  # an unproductive loop: P0 on a high workflow, its drift write fails
                _tool(tracer)
        assert dispatch.wait_idle()
    assert len(seen) == 2  # each unrecorded P0 still paged
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len([m for m in warnings if "unrecorded" in m]) == 1, warnings
    assert len(warnings) == len(set(warnings)), warnings


def test_otel_events_have_latency(home):
    tracer, _, _ = _tracing(workflow="research")
    with tracer.start_as_current_span("invoke_agent analyst", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "analyst"}) as root:
        with tracer.start_as_current_span("chat openai/gpt-oss-120b", attributes=CHAT):
            pass
        _tool(tracer)
    [run] = _runs()
    by_kind = {e["kind"]: json.loads(e["payload"]) for e in _events(run["id"])}
    assert by_kind["llm_call"]["latency_s"] >= 0 and by_kind["tool_result"]["latency_s"] >= 0
    assert "latency_s" not in by_kind["agent"] and root is not None


def test_agent_event_names_its_parent_agent(home):
    """An agent event carries its nearest agent ancestor's name, walking through spans PitStop does not record, so
    the run graph can nest it when its parent span is unrecorded."""
    tracer, _, _ = _tracing(workflow="w")
    with tracer.start_as_current_span("root"):
        with tracer.start_as_current_span("invoke_agent planner", attributes={**AGENT, "gen_ai.agent.name": "planner"}):
            with tracer.start_as_current_span("chain"):  # an "other" span: never recorded
                with tracer.start_as_current_span("invoke_agent researcher",
                                                  attributes={**AGENT, "gen_ai.agent.name": "researcher"}):
                    pass
    [run] = _runs()
    agents = {e["name"]: json.loads(e["payload"]) for e in _events(run["id"]) if e["kind"] == "agent"}
    assert agents["researcher"]["parent"] == "planner"
    assert agents["planner"]["parent"] is None  # "root" is no agent
