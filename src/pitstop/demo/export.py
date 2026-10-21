"""Export the demo snapshot: the one file demo mode replays (no network, no model calls).

  python -m pitstop.demo.export --fixture                       # offline FIXTURE swarm (page development)
  python -m pitstop.demo.export --fixture-pair                  # offline FIXTURE pair (live-mode layout)
  python -m pitstop.demo.export --swarm-run runs/<run_id>       # a real recording (cli run --shadow-prompt)
  python -m pitstop.demo.export --swarm-run runs/pair-<hex>     # a recorded live pair (dir with pair.json)
      [--gepa-run runs/gepa/pitstop_nano_800] [--arm key=label|sublabel|path[,fallback...]]... [--out PATH]

An arm's path is an evals/run_hard_cases.py output dir, a glob of them (newest wins) or a summary JSON;
relative paths are relative to the current directory. Comma-separated paths are tried in order.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ..config import SNAPSHOT
from ..schemas import utc_now
from .views import PAIR_FILE, build_gepa_view, build_pair_view, build_scores_view, build_swarm_view

DEFAULT_OUT = SNAPSHOT
DEFAULT_GEPA_RUN = Path("runs") / "gepa" / "pitstop_nano_800"  # relative to the current directory
# No fallback to evals/results/baseline_v0_*.json: those were scored on an earlier, smaller test split, so a bar
# built from them would not be comparable. An arm stays "pending" until its compare_* run exists.
DEFAULT_ARMS = [
    ("nano", "Nano", "original prompt", ["runs/compare_nano/eval-supervisor_v0-*"]),
    ("nano_gepa", "Nano + GEPA", "optimized prompt", ["runs/compare_nano/eval-pitstop_nano_800-*"]),
    ("super", "Super", "original prompt", ["runs/compare_super/eval-supervisor_v0-*"]),
]


def parse_arm(spec: str) -> tuple[str, str, str, list[str]]:
    """'key=label|sublabel|path[,fallback]' -> (key, label, sublabel, [paths])."""
    key, sep, rest = spec.partition("=")
    parts = rest.split("|")
    if not sep or not key or len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--arm wants key=label|sublabel|path, got {spec!r}")
    return key, parts[0], parts[1], [p.strip() for p in parts[2].split(",") if p.strip()]


def fixture_swarm_view() -> dict:
    from .fixture import make_fixture_run  # imports the orchestrator stack only when needed
    with tempfile.TemporaryDirectory() as tmp:
        run = asyncio.run(make_fixture_run(Path(tmp)))
        view = build_swarm_view(run.run_dir)
    return {**view, "source": "offline-fixture"}


def fixture_pair_view() -> dict:
    from .fixture import make_fixture_pair
    with tempfile.TemporaryDirectory() as tmp:
        pair = asyncio.run(make_fixture_pair(Path(tmp)))
        view = build_pair_view(pair.pair_dir)
    return {**view, "source": "offline-fixture"}


def recorded_view(run_dir: Path) -> dict:
    """A single run dir, or a pair dir (with pair.json)."""
    return build_pair_view(run_dir) if (Path(run_dir) / PAIR_FILE).is_file() else build_swarm_view(run_dir)


def build_snapshot(*, swarm_run: Optional[Path] = None, fixture: bool = False, fixture_pair: bool = False,
                   gepa_run: Optional[Path] = DEFAULT_GEPA_RUN, arms=None) -> dict:
    swarm = (fixture_swarm_view() if fixture else fixture_pair_view() if fixture_pair
             else recorded_view(swarm_run) if swarm_run else None)
    gepa = build_gepa_view(gepa_run) if gepa_run and Path(gepa_run).is_dir() else None
    return {"version": 1, "generated_at": utc_now(), "swarm": swarm, "gepa": gepa,
            "scores": build_scores_view(arms if arms is not None else DEFAULT_ARMS)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pitstop.demo.export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--swarm-run", type=Path, help="recorded run dir (with decisions_shadow.jsonl) or pair dir")
    src.add_argument("--fixture", action="store_true", help="generate an offline FIXTURE run instead")
    src.add_argument("--fixture-pair", action="store_true", help="generate an offline FIXTURE pair instead")
    ap.add_argument("--gepa-run", type=Path, default=DEFAULT_GEPA_RUN)
    ap.add_argument("--arm", type=parse_arm, action="append", help="repeatable; replaces the default arms")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    if a.swarm_run and not ((a.swarm_run / "decisions.jsonl").exists() or (a.swarm_run / PAIR_FILE).exists()):
        ap.error(f"{a.swarm_run} has no decisions.jsonl or {PAIR_FILE}")
    snap = build_snapshot(swarm_run=a.swarm_run, fixture=a.fixture, fixture_pair=a.fixture_pair,
                          gepa_run=a.gepa_run, arms=a.arm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(snap, indent=1) + "\n")

    sw, g = snap["swarm"], snap["gepa"]
    if sw:
        flags = {k: [(d["agent_id"], d["step"]) for d in v if d["alert_note"] == "INTERVENTION POINT"]
                 for k, v in sw["decisions"].items()}
        reports = {k: bool(v) for k, v in sw["team_report"].items()}
        print(f"swarm: {sw['source']} {sw['mode']} {sw['run_id']} in_progress={sw['in_progress']} "
              f"decisions a={len(sw['decisions']['a'])} b={len(sw['decisions']['b'])} flags={flags} "
              f"team_report={reports}")
        if sw["supervisors"]["b"] is None:
            print("  warning: no shadow supervisor in this run (record with `cli run --shadow-prompt`)", file=sys.stderr)
    else:
        print("swarm: none (pass --swarm-run or --fixture)")
    if g:
        print(f"gepa: {g['run']} {g['status']} iterations={g['iterations_run']} candidates={len(g['candidates'])} "
              f"seed={g['seed_score']} best={max((c['val_score'] for c in g['candidates']), default=None)}")
    else:
        print(f"gepa: none ({a.gepa_run} not found)", file=sys.stderr)
    for arm in snap["scores"]["arms"]:
        print(f"arm {arm['key']}: " + ("PENDING " + arm["source"] if arm.get("pending") else
              f"acc={arm['accuracy']} recall={arm['drift_recall']} fp={arm['false_alarms']} n={arm['n']} "
              f"trials={arm['trials']} <- {arm['source']}"))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
