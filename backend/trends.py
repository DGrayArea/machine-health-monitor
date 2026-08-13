"""
Trend detection: catching a channel that is heading for a limit.

Why this exists
    Every rule in thresholds.py looks at one reading in isolation and asks "is
    this value out of range". That answers "is the machine bad now", which is
    detection, not prediction. A spindle whose cooling has been degrading
    steadily for four minutes reads perfectly normal right up until it does not.

    This module fits a line through the recent readings and asks the different
    question: at the rate this is moving, when does it cross the limit? If the
    answer is soon, the operator hears about it while there is still time.

Why it is rules and not a model
    Learning this would need sequences, and AI4I 2020 has no time ordering to
    learn from. See backend/rul.py for the longer version of that argument.
    Training on the simulator's own output would just teach the model my
    simulator. A least-squares slope needs no training data, states its
    assumption openly, and is defensible in a viva.

    The obvious weakness is that a straight-line fit assumes the trend
    continues. It will not always. That is why a trend hit is never worse than
    a Warning: it is a heads-up, not a verdict. The hard limits in
    thresholds.py stay the only things that can declare a Fault.

Reading the output
    A hit means "on the current trajectory, this channel reaches its documented
    failure limit in roughly N minutes". It is deliberately quiet. Four gates
    have to pass before anything is said: enough readings, a window covering
    enough real time, a slope that is statistically distinguishable from zero,
    and a projected crossing inside the horizon.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from backend.thresholds import (
    HDF_TEMP_DIFF_K,
    PWF_POWER_MAX_W,
    PWF_POWER_MIN_W,
    RuleHit,
)

# How many recent readings the fit uses. Same reasoning as the wear-rate window
# in rul.py: long enough to average out sensor noise, short enough to react.
TREND_WINDOW = 20

# Below this many points a slope is not worth quoting.
MIN_POINTS = 6

# The window also has to cover a real stretch of time. Point count alone is not
# enough: at a fast tick rate twenty readings can span five seconds, and a
# five-second window cannot measure a per-minute trend. The slope it returns is
# almost entirely sensor noise, and it flips sign tick to tick.
MIN_SPAN_MIN = 0.5

# How many standard errors the slope must exceed before we call it a trend.
# This is the textbook test for "is this fitted slope distinguishable from
# zero", and it is better than a fixed noise floor because it adapts to how
# noisy the data actually is and to how much of it there is. Two is the
# conventional cut, roughly 95% confidence.
MIN_SLOPE_T = 2.0

# Warn when a channel is projected to cross its limit within this many minutes
# of machine time. Roughly "before the end of the current operation".
HORIZON_MIN = 8.0

# A floor on top of the significance test. A trend can be statistically real
# and still too small to care about, and this is where "too small to care" is
# defined. Units are per minute, one entry per channel.
NOISE_FLOOR = {
    "temp_diff": 0.05,   # K/min
    "power": 40.0,       # W/min
    "torque": 0.30,      # N·m/min
}


def least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """
    Ordinary least-squares slope of y against x.

    Returns None when the fit is degenerate, meaning every x is the same, which
    happens if several readings share a timestamp.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 1e-12:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def slope_with_significance(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float] | None:
    """
    Fit a slope and report how many standard errors it sits from zero.

    Returns (slope, t) where t = |slope| / standard_error. A large t means the
    points really do lie on a line; a small one means the fit is chasing noise.

    Why bother rather than just thresholding the slope: how confidently a slope
    can be measured depends on the scatter of the data and on how long a window
    it was fitted over. A fixed threshold has to be tuned per channel and per
    tick rate and is wrong as soon as either changes. This adapts on its own,
    and it is the standard test any statistics text would use.
    """
    n = len(xs)
    if n < 3:                                  # need a spare degree of freedom
        return None

    slope = least_squares_slope(xs, ys)
    if slope is None:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    intercept = mean_y - slope * mean_x

    residual_ss = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    x_ss = sum((x - mean_x) ** 2 for x in xs)
    if x_ss <= 1e-12:
        return None

    # A perfect fit has no residual, so the slope is infinitely well determined.
    if residual_ss <= 1e-12:
        return slope, float("inf")

    standard_error = (residual_ss / (n - 2) / x_ss) ** 0.5
    if standard_error <= 1e-12:
        return slope, float("inf")

    return slope, abs(slope) / standard_error


def _series(
    history: Sequence[dict[str, Any]],
    channel: str,
    window: int = TREND_WINDOW,
) -> tuple[list[float], list[float]] | None:
    """
    Pull one channel out of the live buffer as (minutes, values).

    Values live in either `reading` (raw sensors) or `derived` (temp_diff,
    power, strain), so both are checked.
    """
    times: list[float] = []
    values: list[float] = []

    for record in list(history)[-window:]:
        try:
            stamp = datetime.fromisoformat(record["timestamp"]).timestamp()
        except (KeyError, TypeError, ValueError):
            continue

        source = record.get("derived") or {}
        if channel not in source:
            source = record.get("reading") or {}
        if channel not in source:
            continue

        times.append(stamp)
        values.append(float(source[channel]))

    if len(times) < MIN_POINTS:
        return None

    t0 = times[0]
    return [(t - t0) / 60.0 for t in times], values


def time_to_limit(
    history: Sequence[dict[str, Any]],
    channel: str,
    limit: float,
    direction: str,
) -> tuple[float, float] | None:
    """
    Project when a channel crosses a limit, given how it is moving now.

    `direction` is "falling" for limits approached from above, such as the
    cooling delta-T, and "rising" for limits approached from below, such as the
    power ceiling.

    Returns (minutes_until_crossing, slope_per_minute), or None when the channel
    is flat, moving away from the limit, or already past it. Already past is
    someone else's job: the hard threshold rules have that covered, and a
    duplicate here would be noise.
    """
    series = _series(history, channel)
    if series is None:
        return None
    xs, ys = series

    # The window has to cover enough real time for a per-minute slope to mean
    # anything, however many readings happen to fall inside it.
    if (xs[-1] - xs[0]) < MIN_SPAN_MIN:
        return None

    fit = slope_with_significance(xs, ys)
    if fit is None:
        return None
    slope, t_statistic = fit

    # Two gates, and both have to pass: is the trend real, and is it big enough
    # to be worth an operator's attention.
    if t_statistic < MIN_SLOPE_T:
        return None
    if abs(slope) < NOISE_FLOOR.get(channel, 0.0):
        return None

    current = ys[-1]
    gap = (current - limit) if direction == "falling" else (limit - current)
    if gap <= 0:
        return None                      # already past the limit

    moving_toward = slope < 0 if direction == "falling" else slope > 0
    if not moving_toward:
        return None                      # drifting away, nothing to say

    return gap / abs(slope), slope


def detect(history: Sequence[dict[str, Any]]) -> list[RuleHit]:
    """
    Check every trended channel and return the hits worth showing.

    Called from the simulator tick, which has the live buffer. A single reading
    posted to /api/predict has no history, so no trend is reported there, which
    is the honest answer rather than a fabricated one.
    """
    hits: list[RuleHit] = []

    checks = [
        # (channel, limit, direction, title, unit label, how to say the value)
        ("temp_diff", HDF_TEMP_DIFF_K, "falling", "Cooling is degrading",
         "ΔT", lambda v: f"{v:.1f} °C"),
        ("power", PWF_POWER_MAX_W, "rising", "Load is climbing",
         "power", lambda v: f"{v / 1000:.2f} kW"),
        ("power", PWF_POWER_MIN_W, "falling", "Load is falling away",
         "power", lambda v: f"{v / 1000:.2f} kW"),
    ]

    for channel, limit, direction, title, label, fmt in checks:
        result = time_to_limit(history, channel, limit, direction)
        if result is None:
            continue
        minutes, slope = result
        if minutes > HORIZON_MIN:
            continue

        series = _series(history, channel)
        current = series[1][-1] if series else limit
        per_min = f"{slope:+.2f}" if channel != "power" else f"{slope / 1000:+.2f} kW"

        hits.append(RuleHit(
            rule_id=f"trend_{channel}_{direction}",
            severity="Warning",
            title=title,
            detail=(
                f"{label} {fmt(current)} moving {per_min}/min, reaching the "
                f"{fmt(limit)} limit in about {minutes:.0f} min if it continues"
            ),
            action=_action_for(channel, direction, minutes),
        ))

    return hits


def _action_for(channel: str, direction: str, minutes: float) -> str:
    horizon = f"in the next {minutes:.0f} minutes"
    if channel == "temp_diff":
        return (
            f"Cooling performance is falling and will reach the failure limit "
            f"{horizon}. Check the coolant level, filter and fan now, while the "
            f"machine is still inside limits."
        )
    if direction == "rising":
        return (
            f"Cutting load is rising and will reach the power limit {horizon}. "
            f"Check for a blunting tool or a change in material, and reduce "
            f"feed rate before it trips."
        )
    return (
        f"Cutting load is falling and will reach the lower power limit "
        f"{horizon}. Check the drive coupling and that the workpiece is still "
        f"engaged."
    )
