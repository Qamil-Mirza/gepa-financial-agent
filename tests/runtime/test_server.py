"""Timeline API: JSON endpoints, drill-down refs and the live SSE stream. Local sockets only."""
import json
import threading
import time

import httpx
import pytest
from rich.console import Console

from pitstop import server as server_mod
from pitstop.config import PROMPTS
from pitstop.orchestrator import Swarm
from pitstop.research.youcom import FinanceResearchClient
from pitstop.schemas import Scenario
from pitstop.server import make_server
from pitstop.supervisor.supervisor import Supervisor
from pitstop.timeline import TimelineBuilder

from test_pipeline_offline import SCENARIO, supervisor_handler, worker_handler


@pytest.fixture
async def finished_run(tmp_path, FakeLLMClass, fixture_response):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture_response)))
    research = FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http, max_live_calls=10)
    swarm = Swarm(
        Scenario.model_validate(SCENARIO), worker_llm=FakeLLMClass(worker_handler),
        supervisor=Supervisor(FakeLLMClass(supervisor_handler), PROMPTS / "supervisor_v0.txt"),
        research=research, prompts_dir=PROMPTS / "worker", runs_dir=tmp_path / "runs",
        halt_on_drift=True, console=Console(file=open(tmp_path / "out.txt", "w")),
    )
    return await swarm.run_all()


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(runs_dir):
        srv = make_server(runs_dir, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _sse(client, url, **kw):
    entries = []
    with client.stream("GET", url, **kw) as r:
        assert r.headers["content-type"] == "text/event-stream"
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "entry":
                entries.append(json.loads(line[6:]))
    return entries


async def test_json_endpoints_and_drilldown(finished_run, serve):
    base = serve(finished_run.run_dir.parent)
    with httpx.Client(base_url=base, timeout=5) as c:
        runs = c.get("/api/runs").json()
        assert [r["run_id"] for r in runs] == [finished_run.run_id]
        assert runs[0]["flagged_agents"] == ["bad"] and not runs[0]["in_progress"]
        assert [a["label"] for a in runs[0]["agents"]] == ["GOOD", "BAD"]

        tl = c.get(f"/api/runs/{finished_run.run_id}/timeline").json()
        flag = next(e for e in tl["entries"] if e["kind"] == "drift_flagged")
        dec = c.get(f"/api/runs/{finished_run.run_id}/decisions/{flag['refs']['decision_index']}").json()
        assert dec["alert_note"] == "INTERVENTION POINT" and dec["agent_id"] == "bad"
        ev = c.get(f"/api/runs/{finished_run.run_id}/events/{flag['evidence'][0]['event_id']}").json()
        assert ev["type"] == "plan"

        assert c.get("/api/runs/nope/timeline").status_code == 404
        assert c.get("/api/runs/../timeline").status_code == 404
        assert c.get(f"/api/runs/{finished_run.run_id}/decisions/999").status_code == 404
        assert c.get("/api/health").headers["access-control-allow-origin"] == "*"

        streamed = _sse(c, f"/api/runs/{finished_run.run_id}/timeline/stream")
        assert [e["seq"] for e in streamed] == [e["seq"] for e in tl["entries"]]
        resumed = _sse(c, f"/api/runs/{finished_run.run_id}/timeline/stream", headers={"Last-Event-ID": "2"})
        assert resumed[0]["seq"] == 3 and resumed[-1]["kind"] == "run_finished"


async def test_outdated_timeline_version_is_rebuilt(finished_run, serve):
    p = finished_run.run_dir / "timeline.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "timeline_version": "1.0"}))
    base = serve(finished_run.run_dir.parent)
    tl = httpx.get(f"{base}/api/runs/{finished_run.run_id}/timeline", timeout=5).json()
    assert tl["timeline_version"] != "1.0" and all("source" in e for e in tl["entries"])


async def test_old_run_without_timeline_is_rebuilt(finished_run, serve):
    for name in ("timeline.json", "timeline.jsonl"):
        (finished_run.run_dir / name).unlink()
    base = serve(finished_run.run_dir.parent)
    tl = httpx.get(f"{base}/api/runs/{finished_run.run_id}/timeline", timeout=5).json()
    assert not tl["in_progress"] and tl["counts"]["drift_flagged"] == 1


def test_stream_follows_a_live_run(tmp_path, serve, monkeypatch):
    monkeypatch.setattr(server_mod, "POLL_S", 0.02)
    run_dir = tmp_path / "runs" / "live-1"
    run_dir.mkdir(parents=True)
    tb = TimelineBuilder(run_dir / "timeline.jsonl")
    tb.run_started("live-1", {"task_id": "t", "swarm_objective": "o", "agents": [{"agent_id": "a", "directive": "d"}]})
    base = serve(tmp_path / "runs")

    partial = httpx.get(f"{base}/api/runs/live-1/timeline", timeout=5).json()
    assert partial["in_progress"] and len(partial["entries"]) == 1

    def writer():
        time.sleep(0.1)
        tb.agent_finished("a", 2)
        time.sleep(0.1)
        tb.run_finished()

    threading.Thread(target=writer).start()
    with httpx.Client(timeout=5) as c:
        kinds = [e["kind"] for e in _sse(c, f"{base}/api/runs/live-1/timeline/stream")]
    assert kinds == ["run_started", "agent_finished", "run_finished"]
