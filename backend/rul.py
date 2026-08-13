"""
Remaining Useful Life (RUL) — how much cutting time is left before the tool
must be changed.

READ THIS BEFORE YOU DEFEND THE PROJECT
    Classical data-driven prognostics (the NASA C-MAPSS style) needs
    **run-to-failure trajectories**: many units, each logged from new until it
    dies, so the model can learn what degradation looks like over time.

    The AI4I 2020 dataset does NOT have that. Its 10,000 rows are independent
    samples with no unit id and no time ordering — you cannot follow one tool
    from new to worn. Pretending otherwise, by inventing a "cycle" column and
    training an LSTM on it, would produce an impressive-looking number that
    means nothing.

    So this module does prognostics the other legitimate way: **model-based**
    (physics-of-failure) rather than data-driven. Both are standard families in
    the prognostics literature; model-based is the correct choice when you have
    a known failure physics and no run-to-failure data. That is exactly our
    situation, and it is a stronger answer than a fake LSTM.

THE THREE LAYERS
    1. PHYSICS RUL  (this file, `physics_rul`)
       Exact remaining tool-wear minutes until the first binding limit. No model
       needed, always available, fully interpretable.

    2. LEARNED RUL  (model/rul_model.pkl, trained by scripts/train_rul_model.py)
       A Random Forest regressor that predicts layer 1 from the sensors. We
       measured it and it does NOT beat the formula — it reproduces it to within
       0.4 min and its trees barely disagree, because there is no noise in the
       target for them to disagree about. It is kept as a cross-check only, and
       the physics value is what the API returns as authoritative. The one place
       its spread is informative is the boundary where the binding constraint
       switches. See scripts/train_rul_model.py for the numbers.

    3. WALL-CLOCK PROJECTION  (`estimate_wear_rate` + `project_wallclock`)
       Layers 1 and 2 answer "how many minutes of *cutting* are left". An
       operator wants "how long until I have to stop the line". So we measure
       the ACTUAL observed wear rate from the recent live readings and divide.
       This uses measured data, not an invented coefficient.

WHAT LIMITS TOOL LIFE (layer 1 in detail)
    Two constraints, and the binding one changes with load:

      a) Tool wear:  the tool is spent at TWF_WEAR_MIN_MIN (200 min).
             remaining = 200 - tool_wear

      b) Overstrain: failure when strain = tool_wear * torque exceeds the
         tier limit. Rearranged for the wear at which that happens:
             max_wear_at_this_torque = osf_limit / torque
             remaining = (osf_limit / torque) - tool_wear

    RUL is the smaller of the two, floored at zero.

    This is why RUL is NOT just "200 - tool_wear". Run an M-tier tool at 50 Nm
    and the strain limit allows 12000/50 = 240 min, so wear binds at 200. Run it
    at 75 Nm and the strain limit allows only 12000/75 = 160 min — the tool now
    dies 40 minutes early, and no tool-wear threshold would ever have told you.
    **Cutting harder does not just use the tool faster, it lowers the ceiling.**

    Deliberately NOT included: cooling faults and power overloads. Those are
    instantaneous failure conditions, not wear mechanisms — they do not consume
    tool life, they end it immediately. The alert rules already cover them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from backend.thresholds import OSF_STRAIN_LIMIT, TWF_WEAR_MIN_MIN, RuleHit

# Below this many remaining cutting minutes, tell the operator to plan a change.
RUL_WARNING_MIN = 25.0
# Below this, the tool should not start another cycle.
RUL_CRITICAL_MIN = 5.0

# Minimum number of live readings before a wear-rate trend is trustworthy.
MIN_POINTS_FOR_RATE = 4

# How many recent readings the wear-rate fit uses.
#
# This is a responsiveness/noise trade-off. Fit over the whole buffer and the
# rate is smooth but lags badly — a machine that started cutting hard a minute
# ago still reports a nominal rate, which is exactly when you want the warning.
# Fit over 2 points and it is responsive but jumps around with sensor noise.
# ~20 readings is long enough to average out the jitter and short enough to
# react within about half a minute at the default tick rate.
WEAR_RATE_WINDOW = 20


@dataclass(frozen=True)
class RulEstimate:
    """Everything we know about how much life is left."""

    remaining_min: float          # cutting minutes to the first binding limit
    binding_constraint: str       # "tool_wear" | "overstrain"
    total_usable_min: float       # usable tool life at THIS operating point
    fraction_consumed: float      # 0..1, for a progress bar
    wear_limited_min: float       # remaining under the tool-wear limit alone
    strain_limited_min: float     # remaining under the overstrain limit alone

    def to_dict(self) -> dict[str, Any]:
        # An idle spindle (torque ~ 0) makes the overstrain limit unreachable,
        # which is mathematically +inf. JSON has no infinity literal — Python
        # would emit a bare `Infinity`, which is invalid JSON and blows up in
        # the browser — so unreachable limits are serialised as null.
        def finite(value: float) -> float | None:
            return round(value, 1) if math.isfinite(value) else None

        return {
            "remaining_min": round(self.remaining_min, 1),
            "binding_constraint": self.binding_constraint,
            "total_usable_min": finite(self.total_usable_min),
            "fraction_consumed": round(self.fraction_consumed, 4),
            "wear_limited_min": round(self.wear_limited_min, 1),
            "strain_limited_min": finite(self.strain_limited_min),
        }


def physics_rul(features: dict[str, float], product_type: str = "M") -> RulEstimate:
    """
    Exact remaining tool life in cutting minutes, from the failure physics.

    >>> physics_rul({"tool_wear": 100.0, "torque": 50.0}, "M").remaining_min
    100.0
    """
    tool_wear = float(features["tool_wear"])
    torque = float(features["torque"])
    osf_limit = OSF_STRAIN_LIMIT.get(product_type, OSF_STRAIN_LIMIT["L"])

    # (a) tool-wear constraint
    wear_limited = TWF_WEAR_MIN_MIN - tool_wear

    # (b) overstrain constraint. At zero torque nothing is being cut, so the
    # strain limit is never reached — treat it as unbounded rather than dividing
    # by zero.
    if torque > 1e-6:
        max_wear_at_torque = osf_limit / torque
        strain_limited = max_wear_at_torque - tool_wear
    else:
        max_wear_at_torque = math.inf
        strain_limited = math.inf

    if strain_limited < wear_limited:
        binding = "overstrain"
        total_usable = max_wear_at_torque
    else:
        binding = "tool_wear"
        total_usable = float(TWF_WEAR_MIN_MIN)

    remaining = max(0.0, min(wear_limited, strain_limited))

    fraction = 1.0 if total_usable <= 0 else min(1.0, tool_wear / total_usable)

    return RulEstimate(
        remaining_min=remaining,
        binding_constraint=binding,
        total_usable_min=total_usable,
        fraction_consumed=max(0.0, fraction),
        wear_limited_min=max(0.0, wear_limited),
        strain_limited_min=(max(0.0, strain_limited)
                            if math.isfinite(strain_limited) else math.inf),
    )


# --------------------------------------------------------------------------
# Layer 3 — turn "cutting minutes" into "wall-clock minutes"
# --------------------------------------------------------------------------

def estimate_wear_rate(
    history: Sequence[dict[str, Any]],
    window: int = WEAR_RATE_WINDOW,
) -> float | None:
    """
    Measure the ACTUAL wear rate from recent readings, in tool-wear minutes per
    wall-clock minute.

    Two things make this non-trivial:

      * A tool change resets wear to zero. Fitting across a reset would give a
        negative slope, so we use only the readings since the most recent reset.
      * A single pair of points is noisy, so we fit a least-squares line over the
        whole current segment instead of differencing the endpoints.

    Returns None when there is not enough clean data to be honest about a rate.
    """
    points: list[tuple[float, float]] = []
    for record in history:
        try:
            wear = float(record["reading"]["tool_wear"])
            stamp = datetime.fromisoformat(record["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        points.append((stamp.timestamp(), wear))

    if len(points) < MIN_POINTS_FOR_RATE:
        return None

    # Only the most recent window, so the rate reflects how the machine is being
    # run NOW rather than averaging in conditions from minutes ago.
    points = points[-window:]

    # Keep only the segment after the last tool change.
    start = 0
    for i in range(1, len(points)):
        if points[i][1] < points[i - 1][1] - 1e-9:
            start = i
    points = points[start:]

    if len(points) < MIN_POINTS_FOR_RATE:
        return None

    # Least-squares slope of wear against time.
    t0 = points[0][0]
    xs = [(t - t0) / 60.0 for t, _ in points]     # wall-clock minutes
    ys = [w for _, w in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 1e-12:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return slope if slope > 1e-6 else None


def normalise_wear_rate(
    observed_rate: float | None, nominal_rate: float | None = None
) -> float | None:
    """
    Express the observed wear rate as a multiple of the nominal duty.

    On a real machine, tool wear is counted in minutes of cutting, so at nominal
    load one wear-minute passes per wall-clock minute and the rate is 1.0. Under
    heavy load it exceeds 1.0 — the tool is burning life faster than the clock.

    The simulator, however, runs COMPRESSED: a 1.5 s tick advances the tool by
    ~2.2 cutting minutes so a full tool life takes about two minutes to watch
    instead of three hours. Its raw rate is therefore ~88, which is arithmetically
    correct and completely useless on a dashboard. Dividing by the nominal rate
    cancels the compression and leaves the number an engineer actually wants:
    "we are wearing this tool 1.6x faster than normal".
    """
    if observed_rate is None or observed_rate <= 1e-6:
        return None
    if nominal_rate is None or nominal_rate <= 1e-6:
        return observed_rate
    return observed_rate / nominal_rate


def project_wallclock(
    remaining_min: float,
    wear_rate: float | None,
    nominal_rate: float | None = None,
) -> float | None:
    """
    Convert remaining cutting minutes into elapsed machine minutes at the
    observed wear rate.

    None when the rate is unknown — better to show nothing than to show a
    confident deadline we cannot support.
    """
    normalised = normalise_wear_rate(wear_rate, nominal_rate)
    if normalised is None or normalised <= 1e-6:
        return None
    return remaining_min / normalised


def rul_band(remaining_min: float) -> str:
    """Traffic-light band for the dashboard."""
    if remaining_min <= RUL_CRITICAL_MIN:
        return "critical"
    if remaining_min <= RUL_WARNING_MIN:
        return "warning"
    return "ok"


# --------------------------------------------------------------------------
# RUL as an alert rule
# --------------------------------------------------------------------------

def rul_rule(features: dict[str, float], product_type: str = "M") -> RuleHit | None:
    """
    Raise an alert when little life is left — but ONLY when the overstrain
    constraint is the binding one.

    Why the restriction: when tool wear is the binding constraint, the existing
    `tool_wear` threshold rule already fires at 180 min and says the same thing.
    Emitting both would double-alert on one condition, and an operator who is
    shown two rows for one problem stops trusting the list.

    The overstrain-bound case is the one nothing else catches: a tool at only
    150 min of wear looks perfectly healthy to a wear threshold, but at 75 Nm its
    ceiling is 160 min and it has 10 minutes left.
    """
    estimate = physics_rul(features, product_type)
    if estimate.binding_constraint != "overstrain":
        return None
    if estimate.remaining_min > RUL_WARNING_MIN:
        return None

    severity = "Fault" if estimate.remaining_min <= RUL_CRITICAL_MIN else "Warning"
    torque = float(features["torque"])

    return RuleHit(
        rule_id="rul",
        severity=severity,
        title=("Tool life exhausted at this load" if severity == "Fault"
               else "Tool life shortened by high load"),
        detail=(
            f"{estimate.remaining_min:.0f} min of cutting left — at {torque:.0f} N·m "
            f"the overstrain ceiling is {estimate.total_usable_min:.0f} min, not "
            f"{TWF_WEAR_MIN_MIN} min"
        ),
        action=(
            "Stop and change the tool now — at this torque it will overstrain "
            "before it reaches its normal wear life."
            if severity == "Fault" else
            f"Reduce torque to extend tool life, or schedule a tool change "
            f"within {estimate.remaining_min:.0f} minutes of cutting."
        ),
    )
