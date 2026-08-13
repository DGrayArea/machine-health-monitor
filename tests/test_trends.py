"""
Tests for the trend layer in backend/trends.py.

What these pin down:
  A trend hit fires only when a channel is genuinely heading for a limit.
  Noise does not produce hits, and neither does a channel drifting away.
  A projection can never escalate to Fault, only the measured limits do that.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend import trends
from backend.alerts import build_alert
from backend.thresholds import HDF_TEMP_DIFF_K, PWF_POWER_MAX_W, RuleHit, derive_features


def history(values: list[float], channel: str = "temp_diff",
            step_seconds: int = 6) -> list[dict]:
    """Build a fake live buffer with one channel moving as given."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = []
    for i, value in enumerate(values):
        stamp = (start + timedelta(seconds=i * step_seconds)).isoformat()
        out.append({
            "timestamp": stamp,
            "derived": {channel: value},
            "reading": {channel: value},
        })
    return out


# ----------------------------------------------------------------- the fit

def test_slope_of_a_straight_line():
    assert trends.least_squares_slope([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(2.0)


def test_slope_is_none_when_degenerate():
    assert trends.least_squares_slope([1.0], [5.0]) is None
    # Every x identical, which is what several readings sharing a timestamp
    # would look like.
    assert trends.least_squares_slope([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None


# ------------------------------------------------------------ time to limit

def test_projects_a_falling_channel_to_its_limit():
    # ΔT falling 1 K per minute from 14.6 K. The limit is 8.6 K, so 6 minutes.
    # Six seconds per step means 0.1 K per step.
    values = [14.6 - 0.1 * i for i in range(10)]
    result = trends.time_to_limit(history(values), "temp_diff",
                                  HDF_TEMP_DIFF_K, "falling")
    assert result is not None
    minutes, slope = result
    assert slope == pytest.approx(-1.0, rel=1e-6)
    assert minutes == pytest.approx(5.1, abs=0.05)   # from the latest value


def test_no_projection_when_the_channel_is_flat():
    assert trends.time_to_limit(history([10.0] * 10), "temp_diff",
                                HDF_TEMP_DIFF_K, "falling") is None


def test_no_projection_when_drifting_away_from_the_limit():
    rising = [10.0 + 0.1 * i for i in range(10)]
    assert trends.time_to_limit(history(rising), "temp_diff",
                                HDF_TEMP_DIFF_K, "falling") is None


def test_no_projection_when_already_past_the_limit():
    """The hard threshold rule owns this case, so a duplicate here is noise."""
    past = [8.0 - 0.05 * i for i in range(10)]
    assert trends.time_to_limit(history(past), "temp_diff",
                                HDF_TEMP_DIFF_K, "falling") is None


def test_slope_below_the_noise_floor_is_treated_as_flat():
    # 0.005 K per step over 6 s is 0.05 K/min, right at the floor, and well
    # inside what a thermocouple would wander on its own.
    crawling = [12.0 - 0.0001 * i for i in range(12)]
    assert trends.time_to_limit(history(crawling), "temp_diff",
                                HDF_TEMP_DIFF_K, "falling") is None


def test_too_few_points_gives_no_projection():
    assert trends.time_to_limit(history([12.0, 11.0]), "temp_diff",
                                HDF_TEMP_DIFF_K, "falling") is None


# ------------------------------------------------------------------ detect

def test_detect_reports_degrading_cooling():
    values = [14.0 - 0.1 * i for i in range(12)]
    hits = trends.detect(history(values))
    cooling = [h for h in hits if h.rule_id.startswith("trend_temp_diff")]
    assert cooling
    assert cooling[0].severity == "Warning"
    assert "cooling" in cooling[0].title.lower()
    assert "min" in cooling[0].detail
    assert cooling[0].action


def test_detect_reports_climbing_load():
    # Power climbing 100 W per step from 8.0 kW toward the 9 kW limit.
    values = [8000.0 + 100.0 * i for i in range(10)]
    hits = trends.detect(history(values, channel="power"))
    assert any(h.rule_id == "trend_power_rising" for h in hits)


def test_detect_is_quiet_on_a_healthy_machine():
    steady = history([11.0, 11.1, 10.9, 11.0, 11.05, 10.95, 11.0, 11.02])
    assert trends.detect(steady) == []


def test_detect_ignores_a_crossing_beyond_the_horizon():
    """A limit two hours away is not news. Only the near term is reported."""
    values = [14.0 - 0.001 * i for i in range(12)]
    assert trends.detect(history(values)) == []


def test_detect_survives_malformed_records():
    """A partial record must not take the trend layer down."""
    broken = [{"timestamp": "not-a-date"}, {"derived": {}}, {}]
    assert trends.detect(broken) == []


# ------------------------------------------------- interaction with alerts

def _features(**kw):
    base = dict(air_temp=298.0, process_temp=308.0, rot_speed=1550,
                torque=42.0, tool_wear=0.0, product_type="M")
    base.update(kw)
    return derive_features(**base)


def test_a_trend_hit_alone_raises_a_warning():
    trend = RuleHit("trend_temp_diff_falling", "Warning", "Cooling is degrading",
                    "ΔT 10.0 °C falling", "Check the coolant.")
    status, alert = build_alert(model_status="Normal", confidence=0.99,
                                features=_features(), extra_hits=[trend])
    assert status == "Warning"
    assert alert["severity"] == "Warning"
    assert any(r["rule_id"] == "trend_temp_diff_falling"
               for r in alert["triggered_rules"])


def test_a_trend_hit_can_never_declare_a_fault():
    """
    A straight-line projection is a guess about the future. Only a measured
    limit gets to stop the machine, so a Fault-severity trend hit is dropped.
    """
    rogue = RuleHit("trend_power_rising", "Fault", "Load climbing",
                    "projected", "Reduce feed.")
    status, alert = build_alert(model_status="Normal", confidence=0.99,
                                features=_features(), extra_hits=[rogue])
    assert status == "Normal"
    assert alert is None


def test_a_real_fault_still_outranks_a_trend():
    trend = RuleHit("trend_temp_diff_falling", "Warning", "Cooling is degrading",
                    "ΔT falling", "Check the coolant.")
    status, alert = build_alert(
        model_status="Normal", confidence=0.9,
        features=_features(torque=75, rot_speed=1400),   # ~11 kW, over the limit
        extra_hits=[trend],
    )
    assert status == "Fault"
    assert alert["severity"] == "Critical"
    # The measured overload leads; the projection is listed after it.
    assert alert["title"] == "Power overload"
    assert "Also tripped" in alert["message"]


# ------------------------------------------------- statistical significance

def test_significance_is_high_for_a_clean_line():
    xs = [0, 1, 2, 3, 4, 5]
    ys = [10 - x for x in xs]
    slope, t = trends.slope_with_significance(xs, ys)
    assert slope == pytest.approx(-1.0)
    assert t == float("inf")          # no residual at all


def test_significance_is_low_for_noise():
    """
    Scattered points still fit *some* line. The t-statistic is what tells you
    that line means nothing, which a plain slope threshold cannot do.
    """
    xs = list(range(12))
    ys = [5.0, 4.8, 5.3, 4.6, 5.4, 4.7, 5.2, 4.9, 5.1, 4.75, 5.25, 4.95]
    slope, t = trends.slope_with_significance(xs, ys)
    assert abs(slope) < 0.05
    assert t < trends.MIN_SLOPE_T


def test_significance_needs_a_spare_degree_of_freedom():
    assert trends.slope_with_significance([0, 1], [0, 1]) is None


def test_a_short_window_is_rejected_however_many_points_it_holds():
    """
    The bug this guards: at a fast tick rate, twenty readings can span five
    seconds. A five-second window cannot measure a per-minute trend, and the
    slope it returns flips sign tick to tick, which used to defeat the alert
    suppression downstream.
    """
    # ΔT falling 0.1 K per reading, ending at 12.1 K against the 8.6 K limit.
    # The identical readings are fed in at two different tick rates.
    values = [14.0 - 0.1 * i for i in range(20)]

    fast = history(values, step_seconds=1)          # 20 points across 19 s
    assert (19 / 60.0) < trends.MIN_SPAN_MIN
    assert trends.detect(fast) == []

    slow = history(values, step_seconds=6)          # same points across 114 s
    assert trends.detect(slow)


def test_noisy_power_does_not_produce_a_trend():
    """Real power readings jitter. That jitter must not read as a rising load."""
    import random
    rng = random.Random(7)
    noisy = [7000.0 + rng.gauss(0, 250) for _ in range(20)]
    hits = trends.detect(history(noisy, channel="power", step_seconds=6))
    assert [h for h in hits if h.rule_id.startswith("trend_power")] == []
