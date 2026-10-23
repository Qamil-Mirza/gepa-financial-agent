"""L2 change-point detection (spec 6.2): a two-sided tabular CUSUM over one (workflow, metric) stream of per-run
values.

The first `WARMUP` values only learn the baseline mean and std (Welford); then the baseline is frozen, so `n` never
passes `WARMUP`. Each later value is standardized, `z = (x - mean) / std`, with the std floored by the metric's
`MetricSpec` (so a flat-zero baseline does not alarm on its first non-zero value), and feeds the sums
`s_hi = max(0, s_hi + z - k)` and `s_lo = max(0, s_lo - z - k)`. A sum above `h` in one of the metric's allowed
directions is an `Alarm`, after which the detector resets and relearns. `(k, h)` come from the workflow's
sensitivity. Pure: no store access (the scan persists `to_json()` in `l2_state`)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Literal

from ..sdk.context import Sensitivity
from .metrics import MetricSpec

WARMUP = 20
PARAMS: dict[Sensitivity, tuple[float, float]] = {"low": (0.5, 8.0), "normal": (0.5, 5.0), "high": (0.5, 4.0)}  # (k, h)
_MIN_STD = 1e-9


@dataclass(frozen=True)
class Alarm:
    direction: Literal["up", "down"]
    at: float  # ts of the first run of the shift: when the alarming sum last left 0
    baseline_mean: float
    baseline_std: float
    value: float


@dataclass
class Cusum:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    s_hi: float = 0.0
    s_lo: float = 0.0
    hi_since: float | None = None
    lo_since: float | None = None

    def update(self, x: float, ts: float, spec: MetricSpec, sensitivity: Sensitivity) -> Alarm | None:
        """Feed one run's value (`ts` is the run's time); an Alarm when a shift is detected. A non-finite value is
        ignored."""
        if not math.isfinite(x):
            return None
        if self.n < WARMUP:  # Welford
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)
            return None
        floor = spec.floor * abs(self.mean) if spec.relative else spec.floor
        std = max(math.sqrt(self.m2 / (self.n - 1)), floor, _MIN_STD)
        z = (x - self.mean) / std
        k, h = PARAMS[sensitivity]
        hi, lo = max(0.0, self.s_hi + z - k), max(0.0, self.s_lo - z - k)
        self.hi_since = None if hi == 0 else ts if self.s_hi == 0 else self.hi_since
        self.lo_since = None if lo == 0 else ts if self.s_lo == 0 else self.lo_since
        self.s_hi, self.s_lo = hi, lo
        for direction, s, since in (("up", hi, self.hi_since), ("down", lo, self.lo_since)):
            if direction in spec.directions and s > h:
                alarm = Alarm(direction, since, self.mean, std, x)
                self._reset()
                return alarm
        return None

    def _reset(self) -> None:
        for f, v in asdict(Cusum()).items():
            setattr(self, f, v)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Cusum:
        data = json.loads(text)
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
