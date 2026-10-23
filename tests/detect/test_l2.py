"""L2: the two-sided tabular CUSUM over one (workflow, metric) stream."""
import random

from pitstop.detect.l2 import PARAMS, WARMUP, Alarm, Cusum
from pitstop.detect.metrics import METRICS

UP = METRICS["tokens_per_call"]  # up only, 5% relative floor
BOTH = METRICS["output_chars"]  # up and down


def _warm(c: Cusum, rng: random.Random, spec=BOTH, mean=10.0) -> float:
    for i in range(WARMUP):
        assert c.update(rng.gauss(mean, 1), float(i), spec, "normal") is None
    return float(WARMUP)


def test_constants():
    assert WARMUP == 20
    assert PARAMS == {"low": (0.5, 8.0), "normal": (0.5, 5.0), "high": (0.5, 4.0)}


def test_no_alarm_on_stationary_noise():
    rng, c = random.Random(7), Cusum()
    for i in range(300):
        assert c.update(rng.gauss(10, 1), float(i), BOTH, "normal") is None
    assert c.n == WARMUP  # the baseline froze after the warmup


def test_step_up_alarms_quickly():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng)
    alarm = None
    for i in range(8):
        alarm = c.update(rng.gauss(13, 1), first + i, BOTH, "normal")
        if alarm is not None:
            break
    assert isinstance(alarm, Alarm) and alarm.direction == "up"
    assert first <= alarm.at <= first + 2
    assert 9 < alarm.baseline_mean < 11 and 0.5 < alarm.baseline_std < 1.5


def test_step_down_alarms_when_allowed():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng)
    alarms = [c.update(rng.gauss(7, 1), first + i, BOTH, "normal") for i in range(8)]
    assert any(a is not None and a.direction == "down" for a in alarms)


def test_down_ignored_when_only_up():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng, UP)
    for i in range(50):
        assert c.update(rng.gauss(5, 1), first + i, UP, "normal") is None


def test_floor_prevents_zero_std_alarm():
    spec, c = METRICS["tool_error_rate"], Cusum()
    for i in range(WARMUP):
        assert c.update(0.0, float(i), spec, "normal") is None
    assert c.update(0.02, 20.0, spec, "normal") is None  # z = 0.02 / 0.05 floor = 0.4 < k
    alarms = [c.update(0.5, 21.0 + i, spec, "normal") for i in range(3)]
    fired = [a for a in alarms if a is not None]
    assert len(fired) == 1 and fired[0].direction == "up" and fired[0].baseline_std == 0.05
    assert fired[0].at == 21.0 and fired[0].value == 0.5


def test_relative_floor_on_flat_baseline():
    c = Cusum()
    for i in range(WARMUP):
        c.update(100.0, float(i), UP, "normal")
    assert c.update(102.0, 20.0, UP, "normal") is None  # z = 2 / (5% of 100) = 0.4 < k


def test_sensitivity_sets_threshold():
    # z = 1.5 per step after the warmup (std floored at 5% of 100 = 5, x = 107.5): s_hi grows by 1 per step
    for sens, steps in (("high", 5), ("normal", 6), ("low", 9)):
        c = Cusum()
        for i in range(WARMUP):
            c.update(100.0, float(i), UP, sens)
        alarms = [c.update(107.5, 20.0 + i, UP, sens) for i in range(steps)]
        assert alarms[:-1] == [None] * (steps - 1) and alarms[-1] is not None, sens


def test_state_roundtrip():
    rng, c = random.Random(3), Cusum()
    first = _warm(c, rng)
    c.update(14.0, first, BOTH, "normal")
    c.update(8.5, first + 1, BOTH, "normal")
    assert c.s_hi > 0 and c.s_lo > 0 and c.hi_since == first and c.lo_since == first + 1
    back = Cusum.from_json(c.to_json())
    assert back == c
    # the restored detector behaves exactly like the original
    assert back.update(13.0, first + 2, BOTH, "normal") == c.update(13.0, first + 2, BOTH, "normal")
    assert back == c
    assert Cusum.from_json(Cusum().to_json()) == Cusum()


def test_reset_after_alarm():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng)
    i = 0
    while c.update(rng.gauss(13, 1), first + i, BOTH, "normal") is None:
        i += 1
    assert c == Cusum()  # relearns the baseline from the next WARMUP runs
    # the next WARMUP values only learn the new baseline, the shifted level, which is then not a shift
    for j in range(WARMUP):
        assert c.update(rng.gauss(13, 1), 100.0 + j, BOTH, "normal") is None
    assert c.n == WARMUP and 12 < c.mean < 14
    assert c.update(13.0, 200.0, BOTH, "normal") is None and c.s_hi == c.s_lo == 0.0


def test_non_finite_values_ignored():
    c = Cusum()
    assert c.update(float("nan"), 0.0, UP, "normal") is None
    assert c == Cusum()
