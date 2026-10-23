"""L2 change-point detection (spec 6.2): a two-sided tabular CUSUM over one (workflow, metric) stream of per-run
values.

The first `WARMUP` values only learn the baseline mean and std (Welford); then the baseline is frozen, so `n` never
passes `WARMUP`. Each later value is standardized, `z = (x - mean) / std`, and feeds the sums
`s_hi = max(0, s_hi + z - k)` and `s_lo = max(0, s_lo - z - k)`. A sum above `h` in one of the metric's allowed
directions is an `Alarm`, after which the detector resets and relearns. `(k, h)` come from the workflow's
sensitivity. Pure: no store access (the scan persists `to_json()` in `l2_state`).

The std is the sample std times `1 + 2 / sqrt(2 (n - 1))` (about two standard errors of a sample std: 30 runs
often underestimate it, and an underestimated std is the main source of false alarms), then floored by the metric's
`MetricSpec` (so a flat or zero baseline does not alarm on its first different value).

Tuning (spec 13 leaves it to the builder; crying wolf is an anti-goal). `WARMUP = 30`, `k = 0.5` and
`h = 10 / 8 / 6` for sensitivity low / normal / high. Simulated with this class on N(10, 1) and a 5% relative floor
(400 seeds; false alarms = streams with at least one alarm in 300 runs after the warmup; a step to mean 13 is caught
when it alarms within 10 runs, and the delay counts the alarming run):

    sensitivity     false alarms: two-sided   up only     3-sigma step: caught   mean delay
    low  (h=10)                  1.3%           0.3%                    99%      6.0 runs
    normal (h=8)                 2.8%           0.7%                   100%      4.9 runs
    high (h=6)                   9.8%           4.8%                   100%      3.8 runs

`test_false_alarm_budget` and `test_detects_3_sigma_step` pin the normal row (<= 5% false alarms, >= 95% caught)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Literal

from ..sdk.context import Sensitivity
from .metrics import MetricSpec

WARMUP = 30
# (k, h) per sensitivity
PARAMS: dict[Sensitivity, tuple[float, float]] = {"low": (0.5, 10.0), "normal": (0.5, 8.0), "high": (0.5, 6.0)}
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
        floor = max(spec.floor * abs(self.mean) if spec.relative else spec.floor, spec.min_floor, _MIN_STD)
        std = max(math.sqrt(self.m2 / (self.n - 1)) * (1 + 2 / math.sqrt(2 * (self.n - 1))), floor)
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
