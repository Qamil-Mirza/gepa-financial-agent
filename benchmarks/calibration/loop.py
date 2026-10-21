"""Recursive improvement loop for PitStop's live supervisor (see benchmarks/calibration/recursive.py).

  uv run python -m benchmarks.calibration.loop calibrate --judge-model <id>     # judge vs known hard-pair labels
  uv run python -m benchmarks.calibration.loop round                            # swarm -> judge -> mine -> optimize -> gate
  uv run python -m benchmarks.calibration.loop swarm|judge|mine|optimize|gate [--round N]   # one stage at a time
  uv run python -m benchmarks.calibration.loop status

Round artifacts go to runs/recursive/round_NN/ under the current directory (git-ignored). Promoted prompts are
written to src/pitstop/assets/prompts/recursive/; benchmarks/calibration/state.json names the prompt the live
supervisor should use.
GEPA never reads the hard-pairs test split; only the gate does, after the candidate is fixed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pitstop.correct.client import CrusoeClient
from pitstop.correct.gepa import CrusoeReflectionLM, OptimizeConfig, run_optimization
from pitstop.correct.policy import Policy
from pitstop.correct.settings import Settings

from .adapter import BACKGROUND, OBJECTIVE, _make_llm_class, load_hard_cases, make_pitstop_evaluator
from .recursive import (HindsightJudge, JudgeVerdict, LoopState, final_inputs, gate_metrics, load_run, mine_run,
                        promotion_decision, rows_to_cases, select_replay)

NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
LOOP_DIR = Path("runs") / "recursive"  # relative to the current directory
DEFAULT_SCENARIOS = ["nvda_amd", "hard_subtle", "hard_constraints", "hard_clean"]


def round_dir(n: int) -> Path:
    return LOOP_DIR / f"round_{n:02d}"


def client(s: Settings, model: str, max_concurrent: int = 4) -> CrusoeClient:
    return CrusoeClient(model=model, api_key=s.crusoe_api_key, base_url=s.base_url, max_concurrent=max_concurrent,
                        max_retries=8, thinking=None)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------- stages ----------------

def cmd_calibrate(args, s: Settings, state: LoopState) -> int:
    """Run the judge on hard pairs with known labels (and on logged runs against their injections, if given)."""
    judge = HindsightJudge(client(s, args.judge_model, args.workers))
    cases = [c for sp in args.splits for c in load_hard_cases(sp)]
    print(f"judge {judge.model} on {len(cases)} hard pairs ({', '.join(args.splits)})")
    verdicts = judge.judge_many([c.input for c in cases], max_workers=args.workers)
    rows = [{"id": c.id, "label": c.label, "expect_type": c.expect_type, "category": c.category, **asdict(v)}
            for c, v in zip(cases, verdicts)]

    def score(rs: list[dict]) -> dict:
        valid = [r for r in rs if r["valid"]]
        drift = [r for r in valid if r["label"] == "drift"]
        ok = [r for r in valid if r["label"] == "ok"]
        caught = [r for r in drift if r["drifted"]]
        return {"n": len(rs), "valid": len(valid),
                "agreement": round(sum(r["drifted"] == (r["label"] == "drift") for r in valid) / len(valid), 3) if valid else None,
                "drift_recall": round(len(caught) / len(drift), 3) if drift else None,
                "ok_specificity": round(sum(not r["drifted"] for r in ok) / len(ok), 3) if ok else None,
                "type_match_when_caught": round(sum(r["drift_type"] == r["expect_type"] for r in caught) / len(caught), 3) if caught else None}

    report = {"judge_model": judge.model, "splits": args.splits, "overall": score(rows),
              f"confidence>={args.min_confidence}": score([r for r in rows if r["confidence"] >= args.min_confidence]),
              "by_category": {k: score([r for r in rows if r["category"] == k]) for k in sorted({r["category"] for r in rows})},
              "usage": judge.client.tracker.snapshot()}
    if args.runs:
        inj = []
        for run_dir in sorted(Path(p) for p in args.runs):
            meta, records = load_run(run_dir)
            for agent, inp in final_inputs(records).items():
                gt, v = meta.get("ground_truth", {}).get(agent, {}), judge.judge(inp)
                inj.append({"run": run_dir.name, "agent": agent, "injected": gt.get("drift_injected"),
                            "injected_step": gt.get("onset_step"), "judge_drifted": v.drifted, "judge_onset": v.onset_step,
                            "judge_valid": v.valid, "judge_confidence": v.confidence})
        agree = [r for r in inj if r["judge_valid"] and r["injected"] == r["judge_drifted"]]
        onset = [r for r in agree if r["injected"] and r["judge_onset"] is not None]
        report["runs_vs_injection"] = {
            "agents": len(inj), "agreement": round(len(agree) / len(inj), 3) if inj else None,
            "onset_within_1": f"{sum(abs(r['judge_onset'] - r['injected_step']) <= 1 for r in onset)}/{len(onset)}",
            "note": "injections are unverified labels: the worker may not have complied", "rows": inj}
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    out = LOOP_DIR / f"calibration_{args.judge_model.split('/')[-1]}.json"
    out.write_text(json.dumps({**report, "rows": rows}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_category",)}, indent=1, default=str)[:4000])
    print("by category:", {k: v["agreement"] for k, v in report["by_category"].items()})
    print("wrote", out)
    return 0


def cmd_swarm(args, s: Settings, state: LoopState) -> int:
    from pitstop.config import PROMPTS, SCENARIOS, load_settings
    from pitstop.llm import LLMClient, UsageTracker
    from pitstop.agents.synthesizer import ReportSynthesizer
    from pitstop.orchestrator import Swarm
    from pitstop.research.youcom import FinanceResearchClient
    from pitstop.schemas import Scenario
    from pitstop.supervisor.supervisor import Supervisor

    ps = load_settings()
    prompt = state.current_path()
    rd = round_dir(args.round)
    sup = Supervisor(_make_llm_class()(client(s, args.supervisor_model)), prompt)
    print(f"round {args.round}: swarm with supervisor {args.supervisor_model}, prompt {sup.prompt_version}, "
          f"You.com mode {args.mode}, corrections {'off' if args.no_correct else 'on'}, "
          f"report {'off' if args.no_report else 'on'}, scenarios {args.scenarios}")

    async def go():
        swarms = []
        for name in args.scenarios:
            path = Path(name) if Path(name).exists() else SCENARIOS / f"{name}.json"
            research = FinanceResearchClient(api_key=ps.ydc_api_key, cache_dir=ps.cache_dir, mode=args.mode,
                                             effort=ps.youcom_effort,
                                             max_live_calls=args.max_research_calls or ps.max_research_calls)
            worker = LLMClient(ps.worker_model, api_key=s.crusoe_api_key, base_url=s.base_url, temperature=0.4,
                               usage=UsageTracker())
            synth = None if args.no_report else ReportSynthesizer(
                LLMClient(ps.report_model, api_key=s.crusoe_api_key, base_url=s.base_url, temperature=0.2,
                          max_tokens=4000, usage=UsageTracker()), PROMPTS / "worker")
            swarms.append(Swarm(Scenario.model_validate_json(path.read_text()), worker_llm=worker, supervisor=sup,
                                research=research, prompts_dir=PROMPTS / "worker", runs_dir=rd / "runs",
                                verbose_events=False, correct_on_drift=not args.no_correct, synthesizer=synth))
        await asyncio.gather(*(sw.run_all() for sw in swarms))

    asyncio.run(go())
    (rd / "swarm.json").write_text(json.dumps({"supervisor_model": args.supervisor_model, "prompt": state.current,
                                               "prompt_version": sup.prompt_version, "scenarios": args.scenarios,
                                               "correct_on_drift": not args.no_correct, "report": not args.no_report,
                                               "usage": sup.llm.client.tracker.snapshot()}, indent=2) + "\n")
    return 0


def cmd_judge(args, s: Settings, state: LoopState) -> int:
    rd = round_dir(args.round)
    judge = HindsightJudge(client(s, args.judge_model, args.workers))
    jobs = []
    for run_dir in sorted(p for p in (rd / "runs").glob("*") if (p / "decisions.jsonl").exists()):
        meta, records = load_run(run_dir)
        for agent, inp in final_inputs(records).items():
            jobs.append((run_dir.name, agent, inp, meta.get("ground_truth", {}).get(agent, {})))
    if not jobs:
        print(f"no logged runs in {rd / 'runs'}; run the swarm stage first")
        return 2
    print(f"round {args.round}: judging {len(jobs)} agent trajectories with {judge.model}")
    verdicts = judge.judge_many([inp for _, _, inp, _ in jobs], max_workers=args.workers)
    rows = []
    for (run_id, agent, _, gt), v in zip(jobs, verdicts):
        rows.append({"run_id": run_id, "agent_id": agent, "judge_model": judge.model, "verdict": asdict(v),
                     "injection": gt})  # the injection is kept for reporting only; mining never reads it
        inj = f"injected@{gt.get('onset_step')}" if gt.get("drift_injected") else "no injection"
        print(f"  {run_id:<34} {agent:<16} judge: {'DRIFT@' + str(v.onset_step) if v.drifted else 'ok':<9} "
              f"{v.drift_type:<22} conf {v.confidence:.2f}{'' if v.valid else ' INVALID ' + str(v.issues)}  ({inj})")
    _write_jsonl(rd / "judgments.jsonl", rows)
    print("usage:", judge.client.tracker.snapshot())
    return 0


def cmd_mine(args, s: Settings, state: LoopState) -> int:
    rd = round_dir(args.round)
    judgments = _read_jsonl(rd / "judgments.jsonl")
    verdicts: dict[str, dict[str, JudgeVerdict]] = {}
    for j in judgments:
        verdicts.setdefault(j["run_id"], {})[j["agent_id"]] = JudgeVerdict(**j["verdict"])
    rows, summary = [], {}
    for run_id, vs in verdicts.items():
        _, records = load_run(rd / "runs" / run_id)
        r, sm = mine_run(run_id, records, vs, min_confidence=args.min_confidence,
                         judge_model=judgments[0]["judge_model"])
        rows += r
        summary[run_id] = sm
    _write_jsonl(rd / "mined.jsonl", rows)
    agents = [a for sm in summary.values() for a in sm.values() if "judge" in a]
    drifted = [a for a in agents if a["judge"]["drifted"]]
    caught = [a for a in drifted if a["first_alert_step"] is not None]
    totals = {"agents_judged": len(agents), "agents_skipped": sum(len(sm) for sm in summary.values()) - len(agents),
              "judge_drifted_agents": len(drifted), "supervisor_caught": len(caught),
              "mean_delay_steps": round(sum(a["first_alert_step"] - a["judge"]["onset_step"] for a in caught) / len(caught), 2)
              if caught else None,
              "decision_agreement": round(sum(a.get("agree", 0) for a in agents) / max(1, sum(a["decisions"] for a in agents)), 3),
              "mined": dict(Counter(r["category"] for r in rows))}
    (rd / "mining_summary.json").write_text(json.dumps({"totals": totals, "per_run": summary}, indent=2) + "\n")
    print(f"round {args.round}: supervisor vs hindsight judge")
    print(json.dumps(totals, indent=1))
    return 0


def cmd_optimize(args, s: Settings, state: LoopState) -> int:
    rd = round_dir(args.round)
    mined = rows_to_cases(_read_jsonl(rd / "mined.jsonl"))
    if not mined:
        print("no disagreements were mined: the supervisor matched the judge everywhere, so there is nothing to learn")
        return 3
    replay = select_replay(load_hard_cases("train"), args.replay, seed=args.round)
    train = mined + replay
    val = [c for c in load_hard_cases("val") for _ in range(args.val_repeats)]
    seed = Policy.from_text(state.current_path().read_text().strip(), name=Path(state.current).stem)
    print(f"round {args.round}: GEPA from {state.current} ({seed.version}); train = {len(mined)} mined "
          f"{dict(Counter(c.category for c in mined))} + {len(replay)} replay; val = {len(val)} (fixed); test not loaded")
    sup_client = client(s, args.supervisor_model, args.workers)
    reflection = CrusoeReflectionLM(client(s, args.reflection_model, 1))
    cfg = OptimizeConfig(max_metric_calls=args.max_metric_calls, reflection_minibatch_size=args.minibatch,
                         seed=args.round, max_workers=args.workers, run_dir=str(rd / "gepa"),
                         acceptance_criterion="strict_improvement", perfect_score=1.0)
    art = run_optimization(seed, train, val, sup_client, reflection, cfg, evaluator=make_pitstop_evaluator(sup_client),
                           objective=OBJECTIVE, background=BACKGROUND)
    (rd / "candidate.txt").write_text(art.best_policy.text.strip() + "\n", encoding="utf-8")
    meta = {"seed": state.current, "seed_version": seed.version, "candidate_version": art.best_policy.version,
            "seed_val_score": art.seed_val_score, "best_val_score": art.best_val_score,
            "num_candidates": art.num_candidates, "total_metric_calls": art.total_metric_calls,
            "train": {"mined": len(mined), "replay": len(replay)}, "val": len(val),
            "supervisor_model": args.supervisor_model, "reflection_model": args.reflection_model,
            "supervisor_usage": art.supervisor_usage, "reflection_usage": art.reflection_usage}
    (rd / "candidate.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"GEPA: {art.num_candidates} candidates, val {art.seed_val_score:.3f} -> {art.best_val_score:.3f}"
          f"{' (seed unchanged)' if art.best_policy.version == seed.version else ''}")
    return 0


def cmd_gate(args, s: Settings, state: LoopState) -> int:
    rd = round_dir(args.round)
    cur_text = state.current_path().read_text().strip()
    cand_text = (rd / "candidate.txt").read_text().strip()
    cur_v, cand_v = Policy.from_text(cur_text).version, Policy.from_text(cand_text).version
    entry = {"round": args.round, "seed": state.current, "seed_version": cur_v, "candidate_version": cand_v}
    if cur_v == cand_v:
        entry.update(promoted=False, reason="GEPA kept the seed prompt")
    else:
        test = load_hard_cases("test")
        evaluate = make_pitstop_evaluator(client(s, args.supervisor_model, args.workers))
        print(f"round {args.round}: gate on frozen hard-pairs test ({len(test)} cases x {args.trials} trials), "
              f"current {cur_v} vs candidate {cand_v}")
        cur = gate_metrics(cur_text, test, evaluate, trials=args.trials, max_workers=args.workers)
        cand = gate_metrics(cand_text, test, evaluate, trials=args.trials, max_workers=args.workers)
        promoted, reason = promotion_decision(cur, cand)
        entry.update(current_test=cur, candidate_test=cand, promoted=promoted and not args.no_promote,
                     reason=reason + (" (--no-promote)" if promoted and args.no_promote else ""))
        print(json.dumps({"current": cur, "candidate": cand}, indent=1))
        if promoted and not args.no_promote:
            entry["promoted_to"] = state.promote(cand_text, args.round)
    (rd / "gate.json").write_text(json.dumps(entry, indent=2) + "\n")
    mining = rd / "mining_summary.json"
    if mining.exists():
        entry["live_vs_judge"] = json.loads(mining.read_text())["totals"]
    state.record(entry)
    state.save()
    print(f"{'PROMOTED' if entry['promoted'] else 'not promoted'}: {entry['reason']}. Live prompt: {state.current}")
    return 0


def cmd_round(args, s: Settings, state: LoopState) -> int:
    for stage in (cmd_swarm, cmd_judge, cmd_mine, cmd_optimize, cmd_gate):
        code = stage(args, s, state)
        if code == 3:  # nothing mined: record the round as converged instead of failing
            state.record({"round": args.round, "seed": state.current, "promoted": False,
                          "reason": "no supervisor/judge disagreements"})
            state.save()
            return 0
        if code:
            return code
    return 0


def cmd_status(args, s: Settings, state: LoopState) -> int:
    print(f"live prompt: {state.current}")
    for r in state.rounds:
        t = r.get("candidate_test")
        print(f"  round {r['round']}: {'PROMOTED' if r.get('promoted') else 'kept'} - {r.get('reason')}"
              + (f" (test acc {r['current_test']['accuracy']:.3f} -> {t['accuracy']:.3f})" if t else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calibrate", "swarm", "judge", "mine", "optimize", "gate", "round", "status"])
    ap.add_argument("--round", type=int, default=None, help="round number (default: the next one)")
    ap.add_argument("--supervisor-model", default=os.environ.get("PITSTOP_SUPERVISOR_MODEL", NANO))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", SUPER))
    ap.add_argument("--reflection-model", default=None, help="GEPA reflection model (default REFLECTION_MODEL)")
    ap.add_argument("--workers", type=int, default=4)
    # swarm
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--mode", choices=["live", "replay"], default="live")
    ap.add_argument("--max-research-calls", type=int, default=None)
    ap.add_argument("--no-correct", action="store_true", help="swarm: observe only, never send corrections")
    ap.add_argument("--no-report", action="store_true", help="swarm: skip the final report agent")
    # judge / mine
    ap.add_argument("--min-confidence", type=float, default=0.7, help="skip judge verdicts below this confidence")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="calibrate: hard-pair splits to judge")
    ap.add_argument("--runs", nargs="*", help="calibrate: logged run dirs to compare against their injections")
    # optimize
    ap.add_argument("--replay", type=int, default=12, help="hard-pair train cases mixed into each round")
    ap.add_argument("--val-repeats", type=int, default=1)
    ap.add_argument("--max-metric-calls", type=int, default=150)
    ap.add_argument("--minibatch", type=int, default=4)
    # gate
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--no-promote", action="store_true", help="evaluate the gate but keep the live prompt")
    args = ap.parse_args(argv)

    s = Settings.from_env()
    if args.stage != "status" and not s.has_key:
        print("CRUSOE_API_KEY not set (root .env)")
        return 2
    args.reflection_model = args.reflection_model or s.reflection_model
    state = LoopState.load()
    if args.round is None:
        done = {r["round"] for r in state.rounds}
        args.round = max(done) + 1 if done and args.stage in ("round", "swarm") else max(done | {0}) if done else 0
    round_dir(args.round).mkdir(parents=True, exist_ok=True)
    return {"calibrate": cmd_calibrate, "swarm": cmd_swarm, "judge": cmd_judge, "mine": cmd_mine,
            "optimize": cmd_optimize, "gate": cmd_gate, "round": cmd_round, "status": cmd_status}[args.stage](args, s, state)


if __name__ == "__main__":
    raise SystemExit(main())
