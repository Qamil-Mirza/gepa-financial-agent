"""Memory: saves every run to disk so agents can learn from past runs. OWNER: TODO

Simple on purpose: one JSON file per run in runs/. Swap for a database later if needed.
"""
import json
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def save_run(record: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"run_{int(time.time())}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def recent_lessons(limit: int = 5) -> str:
    """Checker feedback from recent runs, fed back to the planner."""
    if not RUNS_DIR.exists():
        return ""
    lessons = []
    for path in sorted(RUNS_DIR.glob("run_*.json"))[-limit:]:
        for step in json.loads(path.read_text()).get("steps", []):
            for fb in step.get("rejections", []):
                lessons.append(f"- {fb}")
    return "\n".join(lessons[-10:])
