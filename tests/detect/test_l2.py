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
    assert WARMUP == 30
    assert PARAMS == {"low": (0.5, 10.0), "normal": (0.5, 8.0), "high": (0.5, 6.0)}


def _first_alarm(seed: int, spec, *, shift: float = 0.0, runs: int = 300) -> int | None:
    """Warm up on N(10, 1), then feed `runs` values of N(10 + shift, 1); the index of the first alarm, if any."""
    rng, c = random.Random(seed), Cusum()
    _warm(c, rng, spec)
    for i in range(runs):
        if c.update(rng.gauss(10 + shift, 1), float(WARMUP + i), spec, "normal") is not None:
            return i
    return None


def test_false_alarm_budget():
    # at most 5% of stationary streams alarm even once in 300 runs, two-sided and up-only
    for spec in (BOTH, UP):
        alarmed = sum(_first_alarm(seed, spec) is not None for seed in range(200))
        assert alarmed <= 10, (spec.directions, alarmed)


def test_detects_3_sigma_step():
    detected = sum(_first_alarm(seed, BOTH, shift=3.0, runs=10) is not None for seed in range(200))
    assert detected >= 190


def test_no_alarm_on_stationary_noise():
    rng, c = random.Random(7), Cusum()
    for i in range(WARMUP + 300):
        assert c.update(rng.gauss(10, 1), float(i), BOTH, "normal") is None
    assert c.n == WARMUP  # the baseline froze after the warmup


def test_step_up_alarms_quickly():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng)
    alarm = None
    for i in range(10):
        alarm = c.update(rng.gauss(13, 1), first + i, BOTH, "normal")
        if alarm is not None:
            break
    assert isinstance(alarm, Alarm) and alarm.direction == "up"
    assert first <= alarm.at <= first + 2
    assert 9 < alarm.baseline_mean < 11 and 0.5 < alarm.baseline_std < 1.5


def test_step_down_alarms_when_allowed():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng)
    alarms = [c.update(rng.gauss(7, 1), first + i, BOTH, "normal") for i in range(10)]
    assert any(a is not None and a.direction == "down" for a in alarms)


def test_down_ignored_when_only_up():
    rng, c = random.Random(7), Cusum()
    first = _warm(c, rng, UP)
    for i in range(50):
        assert c.update(rng.gauss(5, 1), first + i, UP, "normal") is None
        assert c.s_lo == 0.0 and c.lo_since is None  # a disallowed direction's sum never grows


def test_floor_prevents_zero_std_alarm():
    spec, c = METRICS["tool_error_rate"], Cusum()
    for i in range(WARMUP):
        assert c.update(0.0, float(i), spec, "normal") is None
    assert c.update(0.02, 100.0, spec, "normal") is None  # z = 0.02 / 0.15 floor = 0.13 < k
    # z = 0.5 / 0.15 = 3.33 per run: s_hi = 2.83, 5.67, 8.5 > h on the third
    alarms = [c.update(0.5, 101.0 + i, spec, "normal") for i in range(3)]
    fired = [a for a in alarms if a is not None]
    assert len(fired) == 1 and fired[0].direction == "up" and fired[0].baseline_std == 0.15
    assert fired[0].at == 101.0 and fired[0].value == 0.5

    assert alarms[2] is fired[0]


def _zero_rate_baseline(sensitivity) -> Cusum:
    c = Cusum()
    for i in range(WARMUP):
        assert c.update(0.0, float(i), METRICS["tool_error_rate"], sensitivity) is None
    return c


def test_single_failed_run_does_not_alarm_at_normal():
    # one run whose only tool call failed: z = 1 / 0.15 = 6.67, s_hi = 6.17 < h; a second one: 12.33 > h
    for sens in ("normal", "low"):
        c = _zero_rate_baseline(sens)
        assert c.update(1.0, 100.0, METRICS["tool_error_rate"], sens) is None
        alarm = c.update(1.0, 101.0, METRICS["tool_error_rate"], sens)
        assert alarm is not None and alarm.at == 100.0, sens


def test_single_failed_run_alarms_at_high():
    c = _zero_rate_baseline("high")
    alarm = c.update(1.0, 100.0, METRICS["tool_error_rate"], "high")  # 6.17 > h = 6
    assert alarm is not None and alarm.at == 100.0


def test_absolute_floor_on_zero_baseline():
    # a relative floor of a zero mean is zero: the metric's min_floor (1 call) keeps the std up
    spec, c = METRICS["llm_calls"], Cusum()
    for i in range(WARMUP):
        assert c.update(0.0, float(i), spec, "normal") is None
    assert c.update(1.0, 100.0, spec, "normal") is None  # z = 1 / 1 = 1: s_hi = 0.5 < h


def test_relative_floor_on_flat_baseline():
    c = Cusum()
    for i in range(WARMUP):
        c.update(100.0, float(i), UP, "normal")
    assert c.update(102.0, 100.0, UP, "normal") is None  # z = 2 / (5% of 100) = 0.4 < k


def test_sensitivity_sets_threshold():
    # z = 1.5 per step after the warmup (std floored at 5% of 100 = 5, x = 107.5): s_hi grows by 1 per step
    for sens, steps in (("high", 7), ("normal", 9), ("low", 11)):
        c = Cusum()
        for i in range(WARMUP):
            c.update(100.0, float(i), UP, sens)
        alarms = [c.update(107.5, 100.0 + i, UP, sens) for i in range(steps)]
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
