"""GEPA demo backend: shadow supervisor, data views, snapshot export and the demo server. Offline only."""
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from pitstop.config import PROMPTS, Settings
from pitstop.demo import views
from pitstop.demo.export import build_snapshot, parse_arm
from pitstop.demo.pair import Pair, build_pair
from pitstop.demo.server import make_server
from pitstop.llm import LLMError
from pitstop.orchestrator import Swarm
from pitstop.research.youcom import FinanceResearchClient
from pitstop.schemas import Scenario
from pitstop.supervisor.supervisor import Supervisor

from test_pipeline_offline import SCENARIO, supervisor_handler, worker_handler

REAL_GEPA_RUN = Path(__file__).resolve().parents[2] / "runs" / "gepa" / "pitstop_nano_800"
BASELINE_NANO = Path(__file__).resolve().parents[2] / "benchmarks" / "hard_pairs" / "results" / "baseline_v0_nano.json"


def shadow_handler(system, user):
    """Flags every agent with a proposal, so the shadow disagrees with the primary on 'good'."""
    agent = user.split("## Monitored agent: ")[1].split("\n")[0]
    plan_ids = [l.split("]")[0][1:] for l in user.splitlines() if "| plan |" in l]
    return json.dumps({"agent_id": agent, "status": "DRIFTING" if plan_ids else "ON_TRACK", "confidence": 0.95,
                       "drift_type": "goal_deviation" if plan_ids else "none",
                       "evidence_event_ids": plan_ids[-1:], "explanation": "shadow fixture"})


def _swarm(tmp_path, FakeLLMClass, fixture_response, *, shadow=True, halt=True, sup_handler=supervisor_handler,
           scenario=SCENARIO, runs_dir=None, research=None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture_response)))
    research = research or FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http,
                                                 max_live_calls=10)
    prompt = PROMPTS / "supervisor_v0.txt"
    return Swarm(
        Scenario.model_validate(scenario), worker_llm=FakeLLMClass(worker_handler),
        supervisor=Supervisor(FakeLLMClass(sup_handler), prompt),
        shadow_supervisor=Supervisor(FakeLLMClass(shadow_handler, model="fake-shadow"), prompt) if shadow else None,
        research=research, prompts_dir=PROMPTS / "worker", runs_dir=runs_dir or tmp_path / "runs",
        halt_on_drift=halt, console=Console(file=open(tmp_path / "out.txt", "a")),
    )


@pytest.fixture
async def shadow_run(tmp_path, FakeLLMClass, fixture_response):
    return await _swarm(tmp_path, FakeLLMClass, fixture_response).run_all()


def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------- shadow supervisor ----------------

async def test_shadow_judges_identical_inputs_without_affecting_the_run(shadow_run):
    d = shadow_run.run_dir
    prim, shad = _jsonl(d / "decisions.jsonl"), _jsonl(d / "decisions_shadow.jsonl")
    assert len(prim) == len(shad) > 0
    assert [p["input"] for p in prim] == [s["input"] for s in shad]
    assert set(prim[0]) == set(shad[0])
    assert all(s["result"]["model"] == "fake-shadow" for s in shad)
    # the shadow flags "good"; the run (interventions, halting, timeline) follows the primary only
    assert any(s["agent_id"] == "good" and s["alert_note"] == "INTERVENTION POINT" for s in shad)
    assert "good" not in shadow_run.interventions and shadow_run.interventions["bad"]["step"] == 1
    flagged = {e["agent_id"] for e in _jsonl(d / "timeline.jsonl") if e["kind"] == "drift_flagged"}
    assert flagged == {"bad"}
    meta = json.loads((d / "meta.json").read_text())
    assert meta["shadow_supervisor"]["model"] == "fake-shadow"
    assert meta["shadow_supervisor"]["prompt_version"].startswith("supervisor_v0@")


async def test_no_shadow_no_shadow_log(tmp_path, FakeLLMClass, fixture_response):
    run = await _swarm(tmp_path, FakeLLMClass, fixture_response, shadow=False).run_all()
    assert not (run.run_dir / "decisions_shadow.jsonl").exists()
    assert "shadow_supervisor" not in json.loads((run.run_dir / "meta.json").read_text())


# ---------------- swarm view ----------------

async def test_swarm_view_of_a_finished_run(shadow_run):
    v = views.build_swarm_view(shadow_run.run_dir)
    assert v["source"] == "recorded" and v["mode"] == "single" and not v["in_progress"]
    assert v["run_id"] == shadow_run.run_id
    assert v["objective"] == "FIXTURE objective"
    assert v["supervisors"]["a"]["model"] == "fake-model" and v["supervisors"]["b"]["model"] == "fake-shadow"
    good, bad = v["agents"]
    assert (good["label"], good["drift_at_step"], bad["label"], bad["drift_at_step"]) == ("GOOD", None, "BAD", 1)
    assert good["max_steps"] == 3 and good["final_report"] == "FIXTURE report"
    assert bad["final_report"] is None  # halted before reporting
    assert len(v["decisions"]["a"]) == len(v["decisions"]["b"])
    d = next(x for x in v["decisions"]["a"] if x["alert_note"] == "INTERVENTION POINT")
    assert set(d) == {"index", "failed", "agent_id", "step", "phase", "status", "confidence", "drift_type",
                      "alert_level", "alert_note", "explanation", "evidence_event_ids", "cutoff_seq"}
    assert (d["agent_id"], d["step"], d["status"]) == ("bad", 1, "DRIFTING")
    # one swarm: both panes show the same event stream and agent reports; no report agent ran here
    assert v["events"]["a"] == v["events"]["b"] and v["agent_reports"]["a"] == v["agent_reports"]["b"]
    assert v["agent_reports"]["a"] == {"good": "FIXTURE report", "bad": None}
    assert v["team_report"] == {"a": None, "b": None}
    ev = v["events"]["a"]
    assert set(ev[0]) == {"event_id", "agent_id", "step", "seq", "type", "summary", "query"}
    calls = [e for e in ev if e["type"] == "tool_call"]
    assert calls and all(e["query"].startswith("Example Corp") for e in calls)
    assert all(e["query"] for e in ev if e["type"] == "plan" and e["step"] == 0)
    assert all(e["query"] is None for e in ev if e["type"] not in ("plan", "tool_call"))
    json.dumps(v)


async def test_swarm_view_of_a_run_still_being_written(shadow_run):
    d = shadow_run.run_dir
    (d / "meta.json").unlink()
    with (d / "events.jsonl").open("a") as f:
        f.write('{"event_id": "half-writ')  # writer mid-line
    v = views.build_swarm_view(d)
    assert v["in_progress"] and v["run_id"] == d.name
    assert [a["agent_id"] for a in v["agents"]] == ["good", "bad"]  # recovered from timeline + decisions
    assert v["agents"][0]["max_steps"] == 3 and v["agents"][1]["drift_at_step"] is None
    assert v["supervisors"]["b"]["model"] == "fake-shadow"  # from the shadow records themselves
    given = views.build_swarm_view(d, scenario=SCENARIO)
    assert given["agents"][1]["drift_at_step"] == 1


# ---------------- GEPA view ----------------

def _gepa_dir(tmp_path, *, tree=True, finished=False):
    d = tmp_path / "gepa" / "toy_30"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "optimizer": {"max_metric_calls": 30}, "supervisor_model": "m/nano", "reflection_model": "m/reflect"}))
    seed, c1 = "line a\nline b", "line a\nline b\nnew rule 1"
    c2 = "line a\nnew rule 1\nnew rule 2"
    (d / "candidates.json").write_text(json.dumps([{"current_candidate": t} for t in (seed, c1, c2)]))
    if tree:
        nodes = [{"idx": 0, "score": 0.5, "parents": "seed"}, {"idx": 1, "score": 0.6, "parents": "0"},
                 {"idx": 2, "score": 0.55, "parents": "1,0"}]
        (d / "candidate_tree.html").write_text(
            "<script>\nconst NODES = " + json.dumps(nodes, indent=1) + ";\nconst DOT = `x`;\n</script>")
    (d / "run_log.txt").write_text("\n".join([
        "Iteration 0: Base program full valset score: 0.5 over 4 / 4 examples",
        "Iteration 1: Selected program 0 score: 0.5",
        "Iteration 1: Proposed new text for current_candidate: line a",
        "Iteration 1: Accepted candidate (subsample score 1.0 -> 2.0); running full eval.",
        "Iteration 1: Valset score for new program: 0.6 (coverage 4 / 4)",
        "Iteration 1: New program candidate index: 1",
        "Iteration 2: Selected program 1 score: 0.6",
        "Iteration 2: New subsample score 2.0 is not better than old score 2.0, skipping",
        "Iteration 3: Selected program 1 score: 0.6",
        "Iteration 3: All subsample scores perfect for parent 1. Skipping.",
        "Iteration 3: Reflective mutation did not propose a new candidate",
        "Iteration 4: Selected program 1 score: 0.6",
        "Iteration 4: Accepted candidate (subsample score 1.0 -> 2.0); running full eval.",
        "Iteration 4: Valset score for new program: 0.55 (coverage 4 / 4)",
        "Iteration 4: New program candidate index: 2",
        "Iteration 5: Selected program 2 score: 0.55",
    ]) + "\n")
    if finished:
        (d / "result.json").write_text("{}")
    return d


@pytest.mark.parametrize("tree", [True, False])
def test_gepa_view_maps_candidates_to_iterations(tmp_path, tree):
    v = views.build_gepa_view(_gepa_dir(tmp_path, tree=tree))
    assert (v["run"], v["model"], v["reflection_model"], v["budget"], v["metric"]) == \
        ("toy_30", "m/nano", "m/reflect", 30, "val score")
    assert v["status"] == "in_progress" and v["iterations_run"] == 5 and v["seed_score"] == 0.5
    got = [(c["idx"], c["parent"], c["iteration"], c["val_score"]) for c in v["candidates"]]
    assert got == [(0, None, 0, 0.5), (1, 0, 1, 0.6), (2, 1, 4, 0.55)]
    assert [l["added"] for l in v["candidates"][0]["lines"]] == [False, False]
    assert [(l["text"], l["added"]) for l in v["candidates"][2]["lines"]] == \
        [("line a", False), ("new rule 1", False), ("new rule 2", True)]
    assert v["candidates"][1]["chars"] == len("line a\nline b\nnew rule 1")
    assert v["rejected_iterations"] == [2, 3]  # iteration 5 is still running, so not counted
    assert v["best_so_far"] == [{"iteration": i, "score": s} for i, s in
                                [(0, 0.5), (1, 0.6), (2, 0.6), (3, 0.6), (4, 0.6), (5, 0.6)]]


def test_gepa_view_finished(tmp_path):
    v = views.build_gepa_view(_gepa_dir(tmp_path, finished=True))
    assert v["status"] == "finished" and v["rejected_iterations"] == [2, 3, 5]


@pytest.mark.skipif(not REAL_GEPA_RUN.is_dir(), reason="no local runs/gepa/pitstop_nano_800")
def test_gepa_view_parses_the_real_nano_run():
    v = views.build_gepa_view(REAL_GEPA_RUN)
    cands = v["candidates"]
    assert cands and cands[0]["idx"] == 0 and cands[0]["parent"] is None and cands[0]["iteration"] == 0
    assert all(isinstance(c["val_score"], float) and c["chars"] > 0 for c in cands)
    assert all(c["iteration"] <= v["iterations_run"] for c in cands)
    assert [c["iteration"] for c in cands] == sorted(c["iteration"] for c in cands)
    assert all(c["parent"] in {x["idx"] for x in cands} for c in cands[1:])
    assert len(v["best_so_far"]) == v["iterations_run"] + 1
    assert max(c["val_score"] for c in cands) == v["best_so_far"][-1]["score"]
    json.dumps(v)


# ---------------- scores view ----------------

def test_scores_view_real_pending_and_newest_glob(tmp_path):
    for ts, acc in (("20260101-000000", 0.1), ("20260102-000000", 0.2)):
        d = tmp_path / f"eval-p-{ts}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"summary": {
            "model": "m", "prompt": "p@1", "trials": 5, "overall": {"n": 9, "accuracy": 0.0},
            "by_split": {"test": {"n": 4, "accuracy": acc, "drift_recall": 0.5, "fp": 1}}}}))
    (tmp_path / "eval-p-20260103-000000").mkdir()  # in progress: no summary.json yet
    v = views.build_scores_view([
        ("nano", "Nano", "original prompt", str(BASELINE_NANO)),
        ("glob", "G", "g", str(tmp_path / "eval-p-*")),
        ("later", "L", "l", ["runs/nope/eval-x-*", str(tmp_path / "missing.json")]),
    ])
    assert (v["split"], v["metric"]) == ("test", "accuracy")
    nano, glob_arm, later = v["arms"]
    assert (nano["accuracy"], nano["drift_recall"], nano["false_alarms"], nano["n"], nano["trials"]) == \
        (0.847, 0.694, 0, 72, 3)
    assert nano["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B" and nano["source"].endswith("baseline_v0_nano.json")
    assert glob_arm["accuracy"] == 0.2 and glob_arm["false_alarms"] == 1 and "20260102" in glob_arm["source"]
    assert later == {"key": "later", "label": "L", "sublabel": "l", "pending": True,
                     "source": f"runs/nope/eval-x-*, {tmp_path / 'missing.json'}"}


def test_parse_arm():
    assert parse_arm("k=Nano + GEPA|optimized prompt|a/b-*,c.json") == \
        ("k", "Nano + GEPA", "optimized prompt", ["a/b-*", "c.json"])
    with pytest.raises(Exception):
        parse_arm("k=only|two")


# ---------------- export ----------------

def test_fixture_snapshot_is_marked_and_shows_the_shadow_flagging_earlier(tmp_path):
    snap = build_snapshot(fixture=True, gepa_run=_gepa_dir(tmp_path),
                          arms=[("x", "X", "x", str(tmp_path / "none-*"))])
    assert snap["version"] == 1 and snap["generated_at"]
    sw = snap["swarm"]
    assert sw["source"] == "offline-fixture" and sw["run_id"].startswith("fixture-")
    assert sw["objective"].startswith("FIXTURE") and all(a["directive"].startswith("FIXTURE") for a in sw["agents"])
    assert all(s["model"].startswith("FIXTURE") and s["prompt_version"].startswith("FIXTURE")
               for s in sw["supervisors"].values())
    assert all(e["query"].startswith("FIXTURE") for k in "ab" for e in sw["events"][k] if e["query"])
    assert all(a["final_report"].startswith("FIXTURE") for a in sw["agents"])
    assert all(r.startswith("FIXTURE") for k in "ab" for r in sw["agent_reports"][k].values())
    flag = lambda k: next(d["step"] for d in sw["decisions"][k]
                          if d["agent_id"] == "agent_nvda_dc" and d["alert_note"] == "INTERVENTION POINT")
    nvda = next(a for a in sw["agents"] if a["agent_id"] == "agent_nvda_dc")
    assert flag("b") == nvda["drift_at_step"] == 2 and flag("a") == 3
    assert not any(d["alert_note"] == "INTERVENTION POINT" for k in "ab" for d in sw["decisions"][k]
                   if d["agent_id"] != "agent_nvda_dc")
    assert snap["gepa"]["run"] == "toy_30" and snap["scores"]["arms"][0]["pending"]
    json.dumps(snap)


# ---------------- demo server ----------------

@pytest.fixture
def serve():
    servers = []

    def start(snapshot, runs_dir, build=None):
        srv = make_server(snapshot, runs_dir=runs_dir, port=0, build=build)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def test_server_snapshot_and_unknown_runs(tmp_path, serve):
    snap = tmp_path / "snapshot.json"
    base = serve(snap, tmp_path / "runs")
    with httpx.Client(base_url=base, timeout=5) as c:
        assert c.get("/api/snapshot").status_code == 404
        snap.write_text(json.dumps({"version": 1, "swarm": None}))
        r = c.get("/api/snapshot")
        assert r.json() == {"version": 1, "swarm": None} and r.headers["access-control-allow-origin"] == "*"
        assert c.get("/api/live/nope").status_code == 404
        assert c.get("/api/live/..").status_code == 404
        assert c.get("/api/live/bad%20id").status_code == 404
        assert c.post("/api/live/start", content=b"[1, 2]").status_code == 400
        assert c.options("/api/live/start").status_code == 204


NO_KEYS = dict(crusoe_api_key=None, ydc_api_key=None)


def _settings(tmp_path, **kw):
    base = dict(crusoe_api_key="fake", crusoe_base_url="http://127.0.0.1:9", supervisor_model="m",
                worker_model="m-worker", report_model="m-report", ydc_api_key="fake", youcom_effort="deep",
                max_research_calls=1, cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    return Settings(**{**base, **kw})


def test_server_live_start_needs_keys(tmp_path, serve):
    no_keys = _settings(tmp_path, **NO_KEYS)
    # the real builder, but with explicit keyless settings: a real key from .env is never reachable here
    base = serve(tmp_path / "snapshot.json", tmp_path / "runs", build=lambda o, d: build_pair(o, d, no_keys))
    r = httpx.post(f"{base}/api/live/start", json={"objective": "x"}, timeout=5)
    assert r.status_code == 400 and "CRUSOE_API_KEY" in r.json()["error"]
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def _fake_pair_build(tmp_path, FakeLLMClass, fixture_response, *, worker=None, sup_handler=supervisor_handler):
    """build(objective, runs_dir) -> Pair of two scripted swarms (no shadow), sharing one mocked You.com client."""
    def build(objective, runs_dir):
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=fixture_response)))
        research = FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http, max_live_calls=20)

        def side(name, d):
            sw = _swarm(tmp_path, FakeLLMClass, fixture_response, shadow=False, halt=False, sup_handler=sup_handler,
                        scenario={**SCENARIO, "swarm_objective": objective or SCENARIO["swarm_objective"]},
                        runs_dir=d, research=research)
            for w in sw.workers:
                w.llm = FakeLLMClass(worker or worker_handler)
            return sw
        return Pair.create(runs_dir, side)
    return build


def test_server_live_pair_one_at_a_time(tmp_path, serve, FakeLLMClass, fixture_response):
    gate = threading.Event()

    def slow_worker(system, user):
        gate.wait(5)
        return worker_handler(system, user)

    base = serve(tmp_path / "snapshot.json", tmp_path / "runs",
                 build=_fake_pair_build(tmp_path, FakeLLMClass, fixture_response, worker=slow_worker))
    with httpx.Client(base_url=base, timeout=5) as c:
        started = c.post("/api/live/start", json={"objective": "typed objective"}).json()
        pair_id = started["run_id"]
        assert [a["agent_id"] for a in started["agents"]] == [a.agent_id for a in Scenario.model_validate(SCENARIO).agents]
        assert all(a["label"] for a in started["agents"])
        assert pair_id.startswith("pair-") and (tmp_path / "runs" / pair_id / "pair.json").is_file()
        busy = c.post("/api/live/start", json={"objective": "again"})
        assert busy.status_code == 409 and pair_id in busy.json()["error"]
        assert c.get("/api/health").json()["live_running"] == pair_id
        live = c.get(f"/api/live/{pair_id}").json()
        assert live["in_progress"] and live["source"] == "live" and live["error"] is None
        assert live["mode"] == "pair" and live["run_id"] == pair_id and live["in_charge"] == "both"
        assert live["objective"] == "typed objective" and live["agents"][1]["drift_at_step"] == 1
        assert live["supervisors"]["a"]["prompt_version"].startswith("supervisor_v0@")
        gate.set()
        deadline = time.time() + 10
        while live["in_progress"] and time.time() < deadline:
            time.sleep(0.05)
            live = c.get(f"/api/live/{pair_id}").json()
        assert not live["in_progress"] and live["error"] is None
        assert len(live["decisions"]["a"]) == len(live["decisions"]["b"]) > 0
        assert c.get("/api/health").json()["live_running"] is None
        # a plain run dir under runs/ is not a live pair
        inner = json.loads((tmp_path / "runs" / pair_id / "pair.json").read_text())["left"]
        assert c.get(f"/api/live/{inner}").status_code == 404


def test_server_live_pair_crash_is_reported(tmp_path, serve, FakeLLMClass, fixture_response):
    def broken(system, user):
        raise LLMError("CRUSOE_API_KEY is not set")

    base = serve(tmp_path / "snapshot.json", tmp_path / "runs",
                 build=_fake_pair_build(tmp_path, FakeLLMClass, fixture_response, sup_handler=broken))
    with httpx.Client(base_url=base, timeout=5) as c:
        pair_id = c.post("/api/live/start", json={}).json()["run_id"]
        deadline, live = time.time() + 10, {}
        while not live.get("error") and time.time() < deadline:
            time.sleep(0.05)
            live = c.get(f"/api/live/{pair_id}").json()
        assert "LLMError" in live["error"] and not live["in_progress"]  # both sides stopped
        doc = json.loads((tmp_path / "runs" / pair_id / "pair.json").read_text())
        assert set(doc["errors"]) == {"left", "right"} and doc["finished_at"]


def test_swarm_view_has_a_timeline_per_supervisor(tmp_path):
    """Both supervisors get the curated timeline (pitstop.timeline), linked to their decisions by index,
    and a finished run also writes the shadow's timeline_shadow.json next to timeline.json."""
    snap = build_snapshot(fixture=True, gepa_run=_gepa_dir(tmp_path), arms=[])
    sw = snap["swarm"]
    for k in "ab":
        flagged = [e for e in sw["timeline"][k] if e["kind"] == "drift_flagged"]
        assert [e["agent_id"] for e in flagged] == ["agent_nvda_dc"]
        d = next(d for d in sw["decisions"][k] if d["index"] == flagged[0]["decision_index"])
        assert d["alert_note"] == "INTERVENTION POINT" and flagged[0]["evidence"]
    assert [d["index"] for d in sw["decisions"]["b"]] == list(range(len(sw["decisions"]["b"])))
    assert not any(d["failed"] for k in "ab" for d in sw["decisions"][k])


def test_finished_run_writes_the_shadow_timeline(tmp_path):
    import asyncio

    from pitstop.demo.fixture import make_fixture_run
    run = asyncio.run(make_fixture_run(tmp_path))
    doc = json.loads((run.run_dir / "timeline_shadow.json").read_text())
    assert doc["run"]["supervisor"]["prompt_version"].startswith("FIXTURE")
    assert [e["kind"] for e in doc["entries"]].count("drift_flagged") == 1
    assert (run.run_dir / "timeline_shadow.jsonl").exists() and (run.run_dir / "timeline.json").exists()


def test_failed_checks_are_not_verdicts():
    assert views._failure({"issues": ["supervisor_call_failed"]}) == "call_failed"
    assert views._failure({"issues": ["invalid_json"]}) == "invalid_json"
    assert views._failure({"issues": ["expanded_short_evidence_ids:['e1']"]}) is None


def test_build_pair_puts_each_prompt_in_charge_of_its_own_swarm(tmp_path):
    """Live mode: two swarms, original prompt (left) and GEPA prompt (right), each correcting drift and writing
    a team report, sharing one You.com client. Building makes no network calls, so fake keys are safe here."""
    from pitstop.demo.pair import LIVE_MAX_CORRECTIONS, LIVE_MAX_RETRIES, MAX_OBJECTIVE_CHARS, NANO, PROMPT_A
    from pitstop.demo.team import generic_team
    keys = _settings(tmp_path)
    planned = []
    plan = lambda o: planned.append(o) or generic_team(o)
    pair = build_pair("my goal", tmp_path / "runs", keys, plan=plan)
    assert planned == ["my goal"] and [a.agent_id for a in pair.left.scenario.agents][0] == "agent_financials"
    assert pair.pair_dir == tmp_path / "runs" / pair.pair_id and pair.pair_id.startswith("pair-")
    doc = json.loads((pair.pair_dir / "pair.json").read_text())
    assert set(doc) == {"pair_id", "objective", "left", "right", "started_at"}
    assert (doc["pair_id"], doc["objective"]) == (pair.pair_id, "my goal")
    assert (doc["left"], doc["right"]) == (pair.left.run.run_id, pair.right.run.run_id) and doc["left"] != doc["right"]
    assert pair.left.run.run_dir.parent == pair.right.run.run_dir.parent == pair.pair_dir
    assert pair.left.supervisor.prompt_version.startswith(PROMPT_A.stem)
    assert not pair.right.supervisor.prompt_version.startswith(PROMPT_A.stem)
    for sw in (pair.left, pair.right):
        assert sw.correct_on_drift and sw.max_corrections == LIVE_MAX_CORRECTIONS and not sw.halt_on_drift
        assert sw.shadow_supervisor is None and sw.scenario.swarm_objective == "my goal" and sw.console.quiet
        assert sw.supervisor.llm.model == NANO and sw.supervisor.llm._client.max_retries == LIVE_MAX_RETRIES
        assert sw.workers[0].llm.temperature == 0.0 and sw.workers[0].llm.model == "m-worker"
        assert sw.synthesizer.llm.model == "m-report" and sw.synthesizer.llm._client.max_retries == LIVE_MAX_RETRIES
        assert not (sw.run.run_dir / views.LAYOUT_FILE).exists()
    assert pair.left.research is pair.right.research  # one client: identical queries are paid for once
    assert pair.left.research.max_live_calls >= 2 * sum(a.max_steps for a in pair.left.scenario.agents)
    assert pair.left.workers[0].llm is not pair.right.workers[0].llm
    assert build_pair("x" * 2000, tmp_path / "runs", keys, plan=plan).left.scenario.swarm_objective == \
        "x" * MAX_OBJECTIVE_CHARS
    # the default objective (empty, or typed as is) runs scenarios/nvda_amd.json with no planning
    from pitstop.demo.pair import SCENARIO as NVDA_AMD
    for goal in ("  ", Scenario.model_validate_json(NVDA_AMD.read_text()).swarm_objective):
        default = build_pair(goal, tmp_path / "runs", keys, plan=lambda o: pytest.fail("planned the default"))
        assert default.left.scenario.swarm_objective.startswith("Produce a comparative analysis")
        assert [a.agent_id for a in default.left.scenario.agents] == ["agent_nvda_dc", "agent_amd_ai", "agent_supply"]


def test_build_pair_needs_keys(tmp_path):
    from pitstop.demo.pair import LiveError
    with pytest.raises(LiveError, match="YDC_API_KEY"):
        build_pair("x", tmp_path / "runs", _settings(tmp_path, ydc_api_key=None))
    assert not (tmp_path / "runs").exists()


def test_layout_file_puts_the_primary_on_the_right(shadow_run):
    plain = views.build_swarm_view(shadow_run.run_dir)
    assert plain["in_charge"] is None
    (shadow_run.run_dir / views.LAYOUT_FILE).write_text(json.dumps({"left": "shadow", "right": "primary"}))
    v = views.build_swarm_view(shadow_run.run_dir)
    assert v["in_charge"] == "b"
    assert v["supervisors"]["a"]["model"] == "fake-shadow" and v["supervisors"]["b"]["model"] == "fake-model"
    assert v["decisions"]["a"] == plain["decisions"]["b"] and v["decisions"]["b"] == plain["decisions"]["a"]


def test_correction_events_carry_the_message():
    ev = views._slim_event({"event_id": "x-e1", "agent_id": "x", "step": 2, "seq": 1, "type": "supervisor_message",
                            "summary": "supervisor correction #1",
                            "payload": {"message": "SUPERVISOR CORRECTION #1 ...", "attempt": 1,
                                        "blocked_query": "gaming revenue"}})
    assert ev["message"].startswith("SUPERVISOR CORRECTION") and ev["attempt"] == 1
    assert ev["blocked_query"] == "gaming revenue"
