"""`@pitstop.tool`: guard a tool function with scope and loop checks (block, steer, halt).

Outside a workflow the decorator just calls the function. Inside one, each invocation goes through the tool guard
(`pitstop.sdk.guard`, shared with the framework adapters): a pending `Halted`, the L0 tool checks, then the L1
pre-action gate for side-effect tools on high and medium workflows. A blocked call returns the
`{"blocked_by": "pitstop", ...}` dict, a steered one the corrective text, and an allowed one runs the function and
records its result. While the function runs, the call is marked as checked, so a nested `@pitstop.tool` of the same
name skips its own guard. Async tools run the gate in a worker thread, never on the event loop.
Exceptions from the user's function propagate unchanged; PitStop's own failures never break the call."""
from __future__ import annotations

import functools
import inspect
from typing import Callable

from .context import current_run
from .guard import MAX_STEERS, aguard_call, checked, guard_call, is_checked, record_tool_result, tool_args_text

__all__ = ["tool", "MAX_STEERS"]


def tool(fn: Callable | None = None, *, side_effects: bool = False, name: str | None = None):
    """Use as `@tool` or `@tool(side_effects=True, name="...")`; sync and async functions."""
    def deco(f: Callable) -> Callable:
        tname = name or f.__name__
        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*a, **k):
                ctx = current_run()
                if ctx is None:
                    return await f(*a, **k)
                if not is_checked(ctx, tname):
                    decision = await aguard_call(ctx, tname, side_effects, tool_args_text(a, k))
                    if decision.action != "run":
                        return decision.reply
                with checked(ctx, tname):
                    try:
                        result = await f(*a, **k)
                    except BaseException as e:
                        record_tool_result(ctx, tname, error=e)
                        raise
                    record_tool_result(ctx, tname, result=result)  # inside the block: the mark times it
                return result
            return awrapper

        @functools.wraps(f)
        def wrapper(*a, **k):
            ctx = current_run()
            if ctx is None:
                return f(*a, **k)
            if not is_checked(ctx, tname):
                decision = guard_call(ctx, tname, side_effects, tool_args_text(a, k))
                if decision.action != "run":
                    return decision.reply
            with checked(ctx, tname):
                try:
                    result = f(*a, **k)
                except BaseException as e:
                    record_tool_result(ctx, tname, error=e)
                    raise
                record_tool_result(ctx, tname, result=result)  # inside the block: the mark times it
            return result
        return wrapper

    return deco(fn) if fn is not None else deco
