"""Orchestrator: the main loop. Plan -> delegate -> check -> retry -> synthesize. OWNER: TODO"""
from . import checker, memory, planner, prompts, worker
from .crusoe_client import PLANNER_MODEL, chat

MAX_RETRIES = 2  # per subtask, before giving up and moving on


def run(goal: str, log=print) -> str:
    log(f"GOAL: {goal}")
    subtasks = planner.make_plan(goal, memory.recent_lessons())
    log(f"PLAN: {len(subtasks)} subtasks")

    steps = []
    for st in subtasks:
        log(f"\n[{st['id']}] {st['task']}")
        feedback, rejections, result, passed = "", [], "", False
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = worker.run_subtask(st, goal, feedback)
            except Exception as e:  # recover instead of crashing the whole run
                result, feedback = "", f"Worker crashed: {e}"
                log(f"  attempt {attempt + 1}: worker error, retrying")
                continue
            verdict = checker.check(st, result)
            if verdict["pass"]:
                passed = True
                log(f"  attempt {attempt + 1}: PASS")
                break
            feedback = verdict["feedback"]
            rejections.append(feedback)
            log(f"  attempt {attempt + 1}: FAIL - {feedback[:120]}")
        steps.append({"subtask": st, "result": result, "passed": passed, "rejections": rejections})

    results = "\n\n".join(
        f"## {s['subtask']['task']}\n{'(UNVERIFIED) ' if not s['passed'] else ''}{s['result']}" for s in steps
    )
    final = chat(
        [{"role": "system", "content": prompts.SYNTHESIZER},
         {"role": "user", "content": f"Goal: {goal}\n\nWorker results:\n{results}"}],
        model=PLANNER_MODEL,
    )
    path = memory.save_run({"goal": goal, "steps": steps, "final": final})
    log(f"\nSaved run to {path}")
    return final
