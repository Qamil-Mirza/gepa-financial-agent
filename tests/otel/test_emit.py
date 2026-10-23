import sys

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import pitstop
from pitstop.otel.emit import emit_verdict
from pitstop.sdk.context import workflow


@pytest.fixture
def tracing():
    """A provider-local tracer (the global provider can only be set once per process) and its exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("pitstop-test"), exporter
    provider.shutdown()


def _verdicts(exporter):
    return [dict(e.attributes) for s in exporter.get_finished_spans() for e in s.events if e.name == "pitstop.verdict"]


def test_verdict_event_on_current_span(tmp_path, tracing):
    tracer, exporter = tracing
    (tmp_path / "config.toml").write_text("")
    pitstop.init(home=tmp_path)

    @pitstop.tool(side_effects=True)
    def send_email(to):
        raise AssertionError("must not run")

    with workflow("wf", tools=["search"]) as ctx:
        with tracer.start_as_current_span("agent-step"):
            assert send_email("x")["blocked_by"] == "pitstop"
    [v] = _verdicts(exporter)
    assert v["pitstop.action"] == "block" and v["pitstop.level"] == "P0"
    assert v["pitstop.drift_type"] == "out_of_scope_action" and v["pitstop.run_id"] == ctx.run_id
    assert isinstance(v["pitstop.drift_id"], int)
    assert v["pitstop.reason"].startswith("out_of_scope_action: ")


def test_emit_on_given_span_omits_unknown_drift_id_and_caps_reason(tracing):
    tracer, exporter = tracing
    with tracer.start_as_current_span("root") as root:
        with tracer.start_as_current_span("child"):
            emit_verdict(run_id="r1", drift_type="unproductive_loop", level="P1", action="record", drift_id=None,
                         reason="x" * 1000, span=root)
    [root_span] = [s for s in exporter.get_finished_spans() if s.name == "root"]
    [event] = root_span.events
    assert event.name == "pitstop.verdict"
    assert dict(event.attributes) == {"pitstop.drift_type": "unproductive_loop", "pitstop.level": "P1",
                                      "pitstop.action": "record", "pitstop.run_id": "r1",
                                      "pitstop.reason": "x" * 300}


def test_emit_without_a_recording_span_is_noop(tracing):
    tracer, exporter = tracing
    emit_verdict(run_id="r1", drift_type="t", level="P2", action="record", drift_id=1, reason="r")  # no span
    span = tracer.start_span("ended")
    span.end()
    emit_verdict(run_id="r1", drift_type="t", level="P2", action="record", drift_id=1, reason="r", span=span)
    assert _verdicts(exporter) == []


def test_emit_swallows_span_errors():
    class Broken:
        def is_recording(self):
            return True

        def add_event(self, *a, **k):
            raise RuntimeError("exporter on fire")

    emit_verdict(run_id="r1", drift_type="t", level="P2", action="record", drift_id=1, reason="r", span=Broken())


def test_emit_without_otel_installed_is_noop(monkeypatch, tracing):
    tracer, exporter = tracing
    with tracer.start_as_current_span("open"):
        monkeypatch.setitem(sys.modules, "opentelemetry", None)
        monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
        assert emit_verdict(run_id="r1", drift_type="t", level="P2", action="record", drift_id=1,
                            reason="r") is None
        monkeypatch.undo()
    assert _verdicts(exporter) == []
    assert trace  # the real module is untouched for later tests


def test_apply_findings_verdict_span_override(tmp_path, tracing):
    from pitstop.detect.l0 import Finding
    from pitstop.sdk.actions import apply_findings
    from pitstop.sdk.context import WorkflowSpec

    tracer, exporter = tracing
    (tmp_path / "config.toml").write_text("")
    pitstop.init(home=tmp_path)
    finding = Finding("out_of_scope_action", "off_task", True, {"tool": "send_email"})
    with tracer.start_as_current_span("outer") as outer:
        with tracer.start_as_current_span("inner"):
            apply_findings(None, [finding], run_id="otel-1", spec=WorkflowSpec("wf"), observe_only=True,
                           verdict_span=outer)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert [e.name for e in spans["inner"].events] == []
    [event] = spans["outer"].events
    assert event.name == "pitstop.verdict" and event.attributes["pitstop.action"] == "record"
    assert event.attributes["pitstop.run_id"] == "otel-1"
