"""Trajectory Collector: normalized events, per-agent histories, replayable JSONL logs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from .schemas import EventType, TrajectoryEvent, utc_now


class TrajectoryCollector:
    def __init__(self, log_path: Optional[Path] = None):
        self._events: dict[str, list[TrajectoryEvent]] = defaultdict(list)
        self._log_path = log_path
        self._listeners: list[Callable[[TrajectoryEvent], None]] = []
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, fn: Callable[[TrajectoryEvent], None]) -> None:
        self._listeners.append(fn)

    def emit(
        self,
        agent_id: str,
        step: int,
        type: EventType,
        summary: str,
        payload: Optional[dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
    ) -> TrajectoryEvent:
        seq = len(self._events[agent_id])
        ev = TrajectoryEvent(
            event_id=f"{agent_id}-e{seq:03d}",
            agent_id=agent_id,
            step=step,
            seq=seq,
            type=type,
            timestamp=utc_now(),
            summary=summary,
            payload=payload or {},
            parent_event_id=parent_event_id,
        )
        self._add(ev)
        return ev

    def _add(self, ev: TrajectoryEvent, write: bool = True) -> None:
        self._events[ev.agent_id].append(ev)
        if write and self._log_path:
            with self._log_path.open("a") as f:
                f.write(ev.model_dump_json() + "\n")
        for fn in self._listeners:
            fn(ev)

    def history(self, agent_id: str, upto_seq: Optional[int] = None) -> list[TrajectoryEvent]:
        """Events visible at a cutoff. The supervisor must never see beyond it."""
        evs = self._events.get(agent_id, [])
        if upto_seq is None:
            return list(evs)
        return [e for e in evs if e.seq <= upto_seq]

    def last_seq(self, agent_id: str) -> int:
        return len(self._events.get(agent_id, [])) - 1

    def agents(self) -> list[str]:
        return list(self._events)

    @classmethod
    def load(cls, log_path: Path) -> "TrajectoryCollector":
        c = cls(log_path=None)
        for line in log_path.read_text().splitlines():
            if line.strip():
                c._add(TrajectoryEvent.model_validate_json(line), write=False)
        return c
