"""Planner: turns a goal into subtasks. OWNER: TODO"""
from . import prompts
from .crusoe_client import PLANNER_MODEL, chat_json


def make_plan(goal: str, memory_notes: str = "") -> list[dict]:
    user = f"Goal: {goal}"
    if memory_notes:
        user += f"\n\nLessons from past runs:\n{memory_notes}"
    plan = chat_json(
        [{"role": "system", "content": prompts.PLANNER}, {"role": "user", "content": user}],
        model=PLANNER_MODEL,
    )
    return plan["subtasks"]
