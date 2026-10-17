from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..schemas import EvaluationRecord, TrajectoryEvent
from ..session_manager import SupervisorManager

log = logging.getLogger("gepa_financial_agent.adapters")


class EventStreamAdapter:
    """Nonblocking bridge from a host swarm to the SupervisorManager.

    Host side: ``await adapter.emit(agent_id, event)`` (bounded queue gives backpressure).
    Supervisor side: ``await adapter.run(stop_event)`` consumes events, records them, and runs
    ``manager.evaluate_pending()`` whenever the queue goes quiet for ``eval_interval_s``.
    Decisions are delivered to callbacks registered with ``on_decision``.
    """

    def __init__(self, manager: SupervisorManager, eval_interval_s: float = 0.5, queue_maxsize: int = 1000) -> None:
        self.manager = manager
        self.eval_interval_s = eval_interval_s
        self.queue: asyncio.Queue[tuple[str, TrajectoryEvent | dict[str, Any]]] = asyncio.Queue(maxsize=queue_maxsize)
        self._callbacks: list[Callable[[EvaluationRecord], Any]] = []
        self.decisions_seen: list[EvaluationRecord] = []

    def on_decision(self, callback: Callable[[EvaluationRecord], Any]) -> None:
        self._callbacks.append(callback)

    async def emit(self, agent_id: str, event: TrajectoryEvent | dict[str, Any]) -> None:
        await self.queue.put((agent_id, event))

    def _deliver(self, records: list[EvaluationRecord]) -> None:
        for r in records:
            self.decisions_seen.append(r)
            for cb in self._callbacks:
                try:
                    cb(r)
                except Exception:  # noqa: BLE001
                    log.exception("decision callback failed")

    async def _consume_available(self) -> int:
        n = 0
        while not self.queue.empty():
            agent_id, event = self.queue.get_nowait()
            try:
                await self.manager.observe(agent_id, event)
            except (KeyError, ValueError) as e:
                log.warning("dropped event for agent=%s: %s", agent_id, e)
            finally:
                self.queue.task_done()
            n += 1
        return n

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                agent_id, event = await asyncio.wait_for(self.queue.get(), timeout=self.eval_interval_s)
            except asyncio.TimeoutError:
                self._deliver(await self.manager.evaluate_pending())
                continue
            try:
                await self.manager.observe(agent_id, event)
            except (KeyError, ValueError) as e:
                log.warning("dropped event for agent=%s: %s", agent_id, e)
            finally:
                self.queue.task_done()
            await self._consume_available()
            if self.queue.empty():
                self._deliver(await self.manager.evaluate_pending())

    async def drain(self) -> None:
        """Process everything already queued and evaluate all pending sessions."""
        await self.queue.join()
        await self._consume_available()
        self._deliver(await self.manager.evaluate_pending())
