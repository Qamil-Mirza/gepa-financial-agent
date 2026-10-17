"""Live-mode pairs (demo/pair.py): two swarms of one scenario, one per supervisor, each correcting its own
drifting agents and writing its own team report. Scripted LLMs and a mocked You.com; no network, no spend."""
import io
import json

import httpx
import pytest
from rich.console import Console

from pitstop.agents.synthesizer import ReportSynthesizer
from pitstop.config import ROOT
from pitstop.demo import views
from pitstop.demo.export import build_snapshot, main as export_main
from pitstop.demo.pair import Pair
from pitstop.orchestrator import Swarm
from pitstop.research.youcom import FinanceResearchClient
from pitstop.schemas import Scenario
from pitstop.supervisor.supervisor import Supervisor

from test_correction_and_report import SCENARIO, make_worker, report_handler, supervisor_handler

VIEW_KEYS = {"source", "mode", "run_id", "in_progress", "objective", "in_charge", "agents", "supervisors", "events",
             "decisions", "timeline", "agent_reports", "team_report"}
TR_KEYS = {"title", "markdown", "error", "findings_included", "findings_excluded", "cited_finding_ids",
           "unverified_figures", "invalid_citations", "model"}


def lenient(system, user):
    """A judge that never flags: its swarm runs the drifting query."""
    agent = user.split("## Monitored agent: ")[1].split("\n")[0]
    return json.dumps({"agent_id": agent, "status": "ON_TRACK", "confidence": 0.9, "drift_type": "none",
                       "evidence_event_ids": [], "explanation": "looks fine"})


def _pair(tmp_path, FakeLLMClass, fixture_response, *, judges, reports=("left", "right")):
    """Two swarms on one mocked You.com client; `judges` maps side -> supervisor handler."""
    calls = []

    def youcom(req):
        calls.append(json.loads(req.content)["input"])
        return httpx.Response(200, json=fixture_response)

    http = httpx.AsyncClient(transport=httpx.MockTransport(youcom))
    research = FinanceResearchClient(api_key="k", cache_dir=tmp_path / "cache", http_client=http, max_live_calls=40)

    def side(name, d):
        synth = ReportSynthesizer(FakeLLMClass(report_handler), ROOT / "prompts" / "worker", backoff_s=0.0) \
            if name in reports else None
        return Swarm(Scenario.model_validate(SCENARIO), worker_llm=FakeLLMClass(make_worker(True)),
                     supervisor=Supervisor(FakeLLMClass(judges[name], model=f"fake-{name}"),
                                           ROOT / "prompts" / "supervisor_v0.txt"),
                     research=research, prompts_dir=ROOT / "prompts" / "worker", runs_dir=d, halt_on_drift=False,
                     correct_on_drift=True, max_corrections=2, synthesizer=synth, console=Console(file=io.StringIO()))

    return Pair.create(tmp_path / "runs", side), calls


def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


async def test_pair_layout_and_both_swarms_correct(tmp_path, FakeLLMClass, fixture_response):
    pair, calls = _pair(tmp_path, FakeLLMClass, fixture_response,
                        judges={"left": supervisor_handler, "right": supervisor_handler})
    doc = json.loads((pair.pair_dir / "pair.json").read_text())
    assert set(doc) == {"pair_id", "objective", "left", "right", "started_at"}  # written before running
    await pair.run_all()
    doc = json.loads((pair.pair_dir / "pair.json").read_text())
    assert doc["finished_at"] and "errors" not in doc and doc["objective"] == "FIXTURE objective"
    assert sorted(p.name for p in pair.pair_dir.iterdir()) == sorted(["pair.json", doc["left"], doc["right"]])
    for sw in (pair.left, pair.right):
        iv = sw.run.interventions["bad"]
        assert iv["step"] == 1 and len(iv["corrections"]) == 1 and "good" not in sw.run.interventions
        events = _jsonl(sw.run.run_dir / "events.jsonl")
        assert any(e["type"] == "supervisor_message" for e in events)
        assert not any(e["type"] == "tool_call" and "gaming" in e["payload"]["query"] for e in events)
        assert (sw.run.run_dir / "meta.json").exists() and (sw.run.run_dir / "report.md").exists()
    # same scripted behaviour on both sides: every query was paid for once (cache or in-flight sharing)
    tool_calls = sum(1 for sw in (pair.left, pair.right) for e in _jsonl(sw.run.run_dir / "events.jsonl")
                     if e["type"] == "tool_call")
    assert len(calls) == len(set(calls)) < tool_calls
    st = pair.left.research.stats()
    assert st["live_calls"] == len(calls) and st["cache_hits"] + st["inflight_hits"] == tool_calls - len(calls)


async def test_pair_view_keeps_the_panes_apart(tmp_path, FakeLLMClass, fixture_response):
    # left judge never flags (its agent drifts, no correction, no team report); right one flags and corrects
    pair, _ = _pair(tmp_path, FakeLLMClass, fixture_response, judges={"left": lenient, "right": supervisor_handler},
                    reports=("right",))
    await pair.run_all()
    v = views.build_pair_view(pair.pair_dir)
    assert set(v) == VIEW_KEYS and json.dumps(v)
    assert (v["source"], v["mode"], v["run_id"], v["in_progress"], v["in_charge"]) == \
        ("recorded", "pair", pair.pair_id, False, "both")
    assert v["objective"] == "FIXTURE objective"
    assert [(a["agent_id"], a["drift_at_step"], a["max_steps"]) for a in v["agents"]] == [("good", None, 3), ("bad", 1, 3)]
    assert set(v["agents"][0]) == {"agent_id", "label", "directive", "max_steps", "drift_at_step"}
    assert v["supervisors"]["a"]["model"] == "fake-left" and v["supervisors"]["b"]["model"] == "fake-right"
    for key in ("events", "decisions", "timeline", "agent_reports", "team_report"):
        assert set(v[key]) == {"a", "b"}
    # each pane is fed only by its own swarm's logs
    for k, sw in (("a", pair.left), ("b", pair.right)):
        raw = _jsonl(sw.run.run_dir / "events.jsonl")
        assert [e["event_id"] for e in v["events"][k]] == [e["event_id"] for e in raw]
        assert [e["summary"] for e in v["events"][k]] == [views._trunc(e["summary"], views.SUMMARY_CHARS) for e in raw]
        assert len(v["decisions"][k]) == len(_jsonl(sw.run.run_dir / "decisions.jsonl"))
        assert [d["index"] for d in v["decisions"][k]] == list(range(len(v["decisions"][k])))
    gaming = lambda k: [e for e in v["events"][k] if e["type"] == "tool_call" and "gaming" in e["query"]]
    assert gaming("a") and not gaming("b")
    assert not any(e["type"] == "supervisor_message" for e in v["events"]["a"])
    assert any(e["type"] == "supervisor_message" for e in v["events"]["b"])
    assert all(d["status"] == "ON_TRACK" for d in v["decisions"]["a"])
    assert [(d["agent_id"], d["step"]) for d in v["decisions"]["b"] if d["alert_note"] == "INTERVENTION POINT"] == \
        [("bad", 1)]
    assert not any(e["kind"] == "drift_flagged" for e in v["timeline"]["a"])
    assert [e["agent_id"] for e in v["timeline"]["b"] if e["kind"] == "drift_flagged"] == ["bad"]
    assert v["agent_reports"]["a"] == v["agent_reports"]["b"] == {"good": "FIXTURE report", "bad": "FIXTURE report"}
    # team report: only the right swarm had a report agent
    assert v["team_report"]["a"] is None
    tr = v["team_report"]["b"]
    assert set(tr) == TR_KEYS and tr["title"] == "FIXTURE report" and tr["error"] is None
    assert tr["markdown"].startswith("## Executive summary") and "*Objective:*" not in tr["markdown"]
    assert "## Sources" in tr["markdown"] and "https://example.com/fixture-10k" in tr["markdown"]
    rep = json.loads((pair.right.run.run_dir / "report.json").read_text())
    assert (tr["findings_included"], tr["findings_excluded"]) == (len(rep["included"]), len(rep["excluded"]))
    assert tr["cited_finding_ids"] == rep["cited_finding_ids"] and tr["invalid_citations"] == ["F99"]
    assert tr["unverified_figures"] == ["73"] and tr["model"] == "fake-model"


async def test_pair_view_mid_run(tmp_path, FakeLLMClass, fixture_response):
    pair, _ = _pair(tmp_path, FakeLLMClass, fixture_response,
                    judges={"left": supervisor_handler, "right": supervisor_handler}, reports=())
    fresh = views.build_pair_view(pair.pair_dir, live=pair)  # nothing run yet
    assert fresh["in_progress"] and fresh["agents"][1]["drift_at_step"] == 1
    assert fresh["events"] == {"a": [], "b": []} and fresh["team_report"] == {"a": None, "b": None}
    assert fresh["agent_reports"]["a"] == {"good": None, "bad": None}
    assert fresh["supervisors"]["b"]["prompt_version"].startswith("supervisor_v0@")
    await pair.run_all()
    doc = json.loads((pair.pair_dir / "pair.json").read_text())
    doc.pop("finished_at")
    (pair.pair_dir / "pair.json").write_text(json.dumps(doc))
    for sw in (pair.left, pair.right):
        (sw.run.run_dir / "meta.json").unlink()
    complete = len(_jsonl(pair.left.run.run_dir / "events.jsonl"))
    with (pair.left.run.run_dir / "events.jsonl").open("a") as f:
        f.write('{"event_id": "half-writ')  # writer mid-line
    v = views.build_pair_view(pair.pair_dir)
    assert v["in_progress"] and [a["agent_id"] for a in v["agents"]] == ["good", "bad"]
    assert v["agents"][1]["drift_at_step"] is None  # hidden from the logs; the live objects know it
    assert v["supervisors"]["a"]["model"] == "fake-left"  # from the decision records themselves
    assert views.build_pair_view(pair.pair_dir, live=pair)["agents"][1]["drift_at_step"] == 1
    assert len(v["events"]["a"]) == complete


async def test_team_report_failure_is_shown_with_its_error(tmp_path, FakeLLMClass, fixture_response):
    pair, _ = _pair(tmp_path, FakeLLMClass, fixture_response,
                    judges={"left": supervisor_handler, "right": supervisor_handler}, reports=("left",))
    pair.left.synthesizer = ReportSynthesizer(FakeLLMClass(lambda s, u: "not json"), ROOT / "prompts" / "worker",
                                              attempts=1)
    await pair.run_all()
    tr = views.build_pair_view(pair.pair_dir)["team_report"]["a"]
    assert tr["title"] == "Report failed" and "JSONParseError" in tr["error"] and tr["cited_finding_ids"] == []
    (pair.left.run.run_dir / "report.json").unlink()  # meta.json's summary alone is enough
    (pair.left.run.run_dir / "report.md").unlink()
    tr2 = views.build_pair_view(pair.pair_dir)["team_report"]["a"]
    assert (tr2["title"], tr2["error"], tr2["markdown"]) == ("Report failed", tr["error"], "")
    assert tr2["findings_included"] == tr["findings_included"]


async def test_pair_run_survives_one_side_failing(tmp_path, FakeLLMClass, fixture_response):
    from pitstop.llm import LLMError

    def broken(system, user):
        raise LLMError("CRUSOE_API_KEY is not set")

    pair, _ = _pair(tmp_path, FakeLLMClass, fixture_response, judges={"left": broken, "right": supervisor_handler})
    with pytest.raises(LLMError):
        await pair.run_all()
    doc = json.loads((pair.pair_dir / "pair.json").read_text())
    assert list(doc["errors"]) == ["left"] and doc["finished_at"]
    assert (pair.right.run.run_dir / "meta.json").exists()  # the other side ran to the end


async def test_exporter_takes_a_pair_dir(tmp_path, FakeLLMClass, fixture_response):
    pair, _ = _pair(tmp_path, FakeLLMClass, fixture_response,
                    judges={"left": lenient, "right": supervisor_handler})
    await pair.run_all()
    snap = build_snapshot(swarm_run=pair.pair_dir, gepa_run=None, arms=[])
    assert snap["swarm"]["mode"] == "pair" and snap["swarm"]["run_id"] == pair.pair_id
    assert snap["swarm"]["team_report"]["a"] and snap["swarm"]["team_report"]["b"]
    single = build_snapshot(swarm_run=pair.left.run.run_dir, gepa_run=None, arms=[])["swarm"]
    assert single["mode"] == "single" and single["team_report"]["a"]["title"] == "FIXTURE report"
    out = tmp_path / "snap.json"
    assert export_main(["--swarm-run", str(pair.pair_dir), "--gepa-run", str(tmp_path / "none"),
                        "--arm", f"x=X|x|{tmp_path / 'none-*'}", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["swarm"]["mode"] == "pair"
    with pytest.raises(SystemExit):
        export_main(["--swarm-run", str(tmp_path / "cache"), "--out", str(out)])


def test_fixture_pair_snapshot(tmp_path):
    sw = build_snapshot(fixture_pair=True, gepa_run=None, arms=[])["swarm"]
    assert set(sw) == VIEW_KEYS and sw["source"] == "offline-fixture" and sw["mode"] == "pair"
    assert sw["objective"].startswith("FIXTURE") and not sw["in_progress"]
    flag = lambda k: [d["step"] for d in sw["decisions"][k]
                      if d["agent_id"] == "agent_nvda_dc" and d["alert_note"] == "INTERVENTION POINT"]
    assert flag("b") == [2] and flag("a") == [3]  # the scripted "GEPA" side catches it a step earlier
    for k in "ab":
        assert [e["agent_id"] for e in sw["events"][k] if e["type"] == "supervisor_message"] == ["agent_nvda_dc"]
        assert sw["team_report"][k]["title"].startswith("FIXTURE") and "## Sources" in sw["team_report"][k]["markdown"]
        assert all(r.startswith("FIXTURE") for r in sw["agent_reports"][k].values())
        assert not any(d["alert_note"] == "INTERVENTION POINT" for d in sw["decisions"][k]
                       if d["agent_id"] != "agent_nvda_dc")
    # the late catch let one gaming query through on the left only
    gaming = lambda k: [e["query"] for e in sw["events"][k] if e["type"] == "tool_call" and "gaming" in e["query"]]
    assert len(gaming("a")) == 1 and not gaming("b")
