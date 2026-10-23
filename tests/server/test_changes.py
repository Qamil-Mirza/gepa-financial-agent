"""What changed (`pitstop.server.changes.change_series`, decision 10) and its endpoint: per-run L2 metrics of the
newest runs, and markers where the model, provider or prompt changed or an L2 alarm fired."""
import json

from pitstop.server.changes import change_series
from tests.server.factory import add_drift, add_llm_call, add_run

T0 = 1_790_000_000.0
SECRET_URL = "https://user-x:pw-secret@host.example/v1?key=key-secret#frag-secret"


def _run(store, i, workflow="wf", **cols):
    run = add_run(store, workflow, started_at=T0 + i * 100, **cols)
    add_llm_call(store, run, ts=T0 + i * 100 + 1, cost_usd=0.02)
    return run


def test_points_oldest_first_without_refused(rt, client):
    runs = [_run(rt.store, i) for i in range(3)]
    add_run(rt.store, "wf", started_at=T0 + 350, status="refused")
    add_run(rt.store, "wf", started_at=T0 + 360, status="dropped", adapter="otel")
    add_run(rt.store, "other", started_at=T0 + 370)
    changes = change_series(rt.store, "wf")
    assert [p.run_id for p in changes.points] == runs
    p = changes.points[0]
    assert (p.started_at, p.status, p.source) == (T0, "ok", "sdk")
    assert p.metrics["cost_usd"] == 0.02 and p.metrics["llm_calls"] == 1.0 and p.metrics["tokens_per_call"] == 120.0
    assert {m.name: (m.label, m.unit) for m in changes.metrics} == {
        "tool_error_rate": ("Tool errors", "%"), "refusal_rate": ("Refusals", "%"), "cost_usd": ("Cost per run", "$"),
        "tokens_per_call": ("Tokens per call", ""), "llm_calls": ("LLM calls", ""), "duration_s": ("Duration", "s"),
        "output_chars": ("Output length", "chars")}
    body = client.get("/api/changes", params={"workflow": "wf"}).json()
    assert [p["run_id"] for p in body["points"]] == runs
    assert client.get("/api/changes", params={"workflow": "nope"}).json()["points"] == []


def test_limit_keeps_the_newest_runs(rt, client):
    runs = [_run(rt.store, i) for i in range(5)]
    assert [p.run_id for p in change_series(rt.store, "wf", limit=3).points] == runs[2:]
    assert len(client.get("/api/changes", params={"workflow": "wf", "limit": 9999}).json()["points"]) == 5
    assert len(client.get("/api/changes", params={"workflow": "wf", "limit": 0}).json()["points"]) == 1


def test_model_marker_same_source_only(rt):
    for i in range(6):  # alternating sources on different models: each source is steady
        if i % 2:
            _run(rt.store, i, adapter="otel", model="otel-model")
        else:
            _run(rt.store, i, model="sdk-model")
    assert change_series(rt.store, "wf").markers == []
    new = _run(rt.store, 6, model="sdk-model-2")
    _run(rt.store, 7, adapter="otel", model="otel-model")
    (marker,) = change_series(rt.store, "wf").markers
    assert (marker.kind, marker.before, marker.after, marker.run_id, marker.source) == (
        "model", "sdk-model", "sdk-model-2", new, "sdk")
    assert marker.at == T0 + 600 and marker.drift_id is None and marker.metric is None


def test_base_url_marker_is_redacted(rt):
    _run(rt.store, 0, base_url="https://a/v1")
    _run(rt.store, 1, base_url=SECRET_URL)
    markers = change_series(rt.store, "wf").markers
    assert [(m.kind, m.before, m.after) for m in markers] == [("base_url", "https://a/v1", "https://host.example/v1")]


def test_prompt_versions_marker(rt):
    _run(rt.store, 0, prompt_versions=json.dumps({"sys": 1}))
    _run(rt.store, 1, prompt_versions=json.dumps({"sys": 1}))
    new = _run(rt.store, 2, prompt_versions=json.dumps({"sys": 2}))
    _run(rt.store, 3, prompt_versions=None)  # unknown: no marker
    (marker,) = change_series(rt.store, "wf").markers
    assert (marker.kind, marker.before, marker.after, marker.run_id) == (
        "prompt_versions", '{"sys": 1}', '{"sys": 2}', new)


def test_l2_alarm_marker_from_drift_evidence(rt, client):
    runs = [_run(rt.store, i) for i in range(4)]
    drift = add_drift(rt.store, "wf", ts=T0 + 400, level="P1", drift_type="cost_or_latency_creep", run_id=runs[3],
                      evidence=[{"run_id": runs[1], "ts": T0 + 200, "detail": {"layer": "L2", "metric": "cost_usd",
                                                                               "change_at": T0 + 150}},
                                {"run_id": runs[3], "ts": T0 + 400, "detail": {"layer": "L2", "metric": "cost_usd",
                                                                               "change_at": T0 + 250}}])
    add_drift(rt.store, "wf", ts=T0 + 400, evidence="not json")  # malformed and non-L2 drifts are skipped
    add_drift(rt.store, "wf", ts=T0 + 400, evidence=[{"detail": {"tool": "search"}}])
    add_drift(rt.store, "other", ts=T0 + 400, evidence=[{"detail": {"metric": "llm_calls", "change_at": T0 + 100}}])
    (marker,) = change_series(rt.store, "wf").markers
    assert (marker.kind, marker.at, marker.metric, marker.drift_id, marker.run_id) == (
        "l2_alarm", T0 + 250, "cost_usd", drift, runs[3])
    assert client.get("/api/changes", params={"workflow": "wf"}).json()["markers"][0]["kind"] == "l2_alarm"
