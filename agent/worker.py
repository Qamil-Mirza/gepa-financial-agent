"""Worker (sub-agent): does one subtask. OWNER: TODO"""
from . import prompts, tools
from .crusoe_client import WORKER_MODEL, chat


def run_subtask(subtask: dict, goal: str, feedback: str = "") -> str:
    context = tools.gather_context(subtask["task"])
    user = (
        f"Overall goal: {goal}\n\nYour subtask: {subtask['task']}\n"
        f"Done when: {subtask.get('done_when', '')}\n\nContext:\n{context}"
    )
    if feedback:
        user += f"\n\nYour previous attempt was rejected. Reviewer feedback:\n{feedback}"
    return chat([{"role": "system", "content": prompts.WORKER}, {"role": "user", "content": user}], model=WORKER_MODEL)
