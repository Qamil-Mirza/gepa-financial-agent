from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .schemas import DatasetExample, DriftStatus, GroundTruth

log = logging.getLogger("gepa_financial_agent.dataset")


@dataclass
class InvalidExample:
    line_number: int
    error: str
    raw_prefix: str


@dataclass
class LoadReport:
    examples: list[DatasetExample]
    invalid: list[InvalidExample]
    path: str

    def summary(self) -> str:
        return f"{self.path}: {len(self.examples)} valid, {len(self.invalid)} invalid"


def load_jsonl(path: str | Path) -> LoadReport:
    path = Path(path)
    examples: list[DatasetExample] = []
    invalid: list[InvalidExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(DatasetExample.model_validate_json(line))
            except (ValidationError, ValueError) as e:
                invalid.append(InvalidExample(n, str(e), line[:200]))
    if invalid:
        log.warning("dataset %s: %d invalid lines skipped", path, len(invalid))
    return LoadReport(examples, invalid, str(path))


def write_jsonl(examples: list[DatasetExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")


@dataclass
class DatasetSplits:
    train: list[DatasetExample] = field(default_factory=list)
    val: list[DatasetExample] = field(default_factory=list)
    test: list[DatasetExample] = field(default_factory=list)

    def summary(self) -> str:
        return f"train={len(self.train)} val={len(self.val)} test={len(self.test)}"


def split_by_scenario(examples: list[DatasetExample], train_frac: float = 0.6, val_frac: float = 0.2,
                      seed: int = 0) -> DatasetSplits:
    """Assign whole scenarios to splits so no prefix of one trajectory leaks across splits.

    A scenario with any example carrying split_hint is forced to that split.
    """
    forced: dict[str, str] = {}
    for ex in examples:
        if ex.split_hint:
            forced[ex.scenario_id] = ex.split_hint
    free = sorted({ex.scenario_id for ex in examples} - set(forced))
    random.Random(seed).shuffle(free)
    n = len(free)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    assignment = dict(forced)
    for i, sid in enumerate(free):
        assignment[sid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    splits = DatasetSplits()
    for ex in examples:
        getattr(splits, assignment[ex.scenario_id]).append(ex)
    return splits


def truncate_example(example: DatasetExample, cutoff_step: int) -> DatasetExample:
    """Prefix of the trajectory up to cutoff_step inclusive, relabeled using drift_onset_step.

    cutoff < onset  -> ON_TRACK (nothing has gone wrong yet)
    cutoff == onset -> UNCERTAIN (a single deviating event is not enough evidence)
    cutoff > onset  -> original label, evidence filtered to visible events
    """
    events = [e for e in example.supervisor_input.trajectory if e.step <= cutoff_step]
    visible = {e.event_id for e in events}
    inp = example.supervisor_input.model_copy(update={"trajectory": events, "observation_step": cutoff_step})
    gt = example.ground_truth
    onset = gt.drift_onset_step
    if onset is not None and gt.status == DriftStatus.DRIFTING:
        if cutoff_step < onset:
            gt = GroundTruth(status=DriftStatus.ON_TRACK, rationale=f"prefix before drift onset at step {onset}",
                             drift_onset_step=onset)
        elif cutoff_step == onset:
            gt = GroundTruth(status=DriftStatus.UNCERTAIN, drift_type=None,
                             evidence_event_ids=[i for i in gt.evidence_event_ids if i in visible],
                             rationale=f"only the first deviating event (step {onset}) is visible",
                             drift_onset_step=onset)
        else:
            gt = gt.model_copy(update={"evidence_event_ids": [i for i in gt.evidence_event_ids if i in visible]})
    else:
        gt = gt.model_copy(update={"evidence_event_ids": [i for i in gt.evidence_event_ids if i in visible]})
    return example.model_copy(update={
        "example_id": f"{example.example_id}@{cutoff_step}",
        "observation_step": cutoff_step,
        "supervisor_input": inp,
        "ground_truth": gt,
    })


def expand_prefixes(examples: list[DatasetExample], cutoffs: list[int]) -> list[DatasetExample]:
    out: list[DatasetExample] = []
    for ex in examples:
        last = max((e.step for e in ex.supervisor_input.trajectory), default=0)
        for c in cutoffs:
            if 0 <= c <= last:
                out.append(truncate_example(ex, c))
    return out


def label_distribution(examples: list[DatasetExample]) -> dict[str, int]:
    counts = {s.value: 0 for s in DriftStatus}
    for ex in examples:
        counts[ex.ground_truth.status.value] += 1
    return counts
