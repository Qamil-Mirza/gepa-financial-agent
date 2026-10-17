"""Trace database: one JSON file per trace in traces/, plus traces/index.jsonl for quick scans."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from . import config
from .schemas import TraceRecord


class TraceStore:
    def __init__(self, root: Path = config.TRACES_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def save(self, trace: TraceRecord) -> Path:
        path = self.root / f"{trace.trace_id}.json"
        path.write_text(trace.model_dump_json(indent=2))
        g = trace.generation
        row = {
            "trace_id": trace.trace_id,
            "family": trace.scenario_family,
            "steer_id": g.steer.steer_id,
            "step_x": g.steer.step_x,
            "passed": g.verification.passed if g.verification else None,
            "onset": g.verification.observed_onset_step if g.verification else None,
            "n_events": len(trace.events),
            "created_at": g.created_at,
        }
        with self.index.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return path

    def load(self, trace_id: str) -> TraceRecord:
        return TraceRecord.model_validate_json((self.root / f"{trace_id}.json").read_text())

    def iter_all(self) -> Iterator[TraceRecord]:
        for p in sorted(self.root.glob("tr_*.json")):
            yield TraceRecord.model_validate_json(p.read_text())

    def all(self) -> List[TraceRecord]:
        return list(self.iter_all())
