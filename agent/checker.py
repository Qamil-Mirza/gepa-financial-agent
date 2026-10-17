"""Checker: the agents grade their own work so the loop can recover. OWNER: TODO"""
from . import prompts
from .crusoe_client import PLANNER_MODEL, chat_json


def check(subtask: dict, result: str) -> dict:
    user = f"Subtask: {subtask['task']}\nDone when: {subtask.get('done_when', '')}\n\nWorker result:\n{result}"
    try:
        verdict = chat_json(
            [{"role": "system", "content": prompts.CHECKER}, {"role": "user", "content": user}],
            model=PLANNER_MODEL,
        )
        return {"pass": bool(verdict.get("pass")), "feedback": verdict.get("feedback", "")}
    except ValueError as e:  # reviewer replied with broken JSON: fail safe, ask for a retry
        return {"pass": False, "feedback": f"Reviewer output unreadable ({e}); redo carefully."}
