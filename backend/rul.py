"""
Remaining useful life: how much cutting time the tool has left.

Why this is physics and not a sequence model
    The usual data-driven approach to RUL, the NASA C-MAPSS style, needs
    run-to-failure trajectories. Many units, each logged from new until it dies,
    so the model can learn the shape of degradation over time.

    AI4I 2020 does not have that. Its 10,000 rows are independent samples with
    no unit id and no time ordering, so you cannot follow one tool from new to
    worn. Inventing a "cycle" column and training an LSTM on it would give a
    good-looking number that measures nothing.

    So this uses the other standard approach, model-based prognostics, working
    from the failure physics. That is the right family when the physics is known
    and run-to-failure data is not available.

The three layers
    1. Physics (physics_rul, below)
       Remaining cutting minutes until the first binding limit. No model needed,
       always available, easy to explain.

    2. Learned model (model/rul_model.pkl, from scripts/train_rul_model.py)
       A Random Forest that predicts layer 1 from the sensors. It was measured
       and it does not beat the formula: it reproduces it to within 0.4 min and
       its trees barely disagree, since there is no noise in the target for them
       to disagree about. It is kept as a cross-check and the API returns the
       physics value. Its spread is only informative at the boundary where the
       binding constraint switches. The numbers are in train_rul_model.py.

    3. Clock time (estimate_wear_rate and project_wallclock)
       Layers 1 and 2 give minutes of cutting. An operator wants to know how
       long until they have to stop, so we measure the wear rate actually seen
       in the recent readings and divide by it.

What limits tool life
    Two constraints, and which one bites depends on the load.

      Tool wear:  the tool is spent at TWF_WEAR_MIN_MIN (200 min).
          remaining = 200 - tool_wear

      Overstrain: fails when strain = tool_wear * torque passes the tier limit.
      Rearranged for the wear at which that happens:
          max_wear_at_this_torque = osf_limit / torque
          remaining = (osf_limit / torque) - tool_wear

    RUL is the smaller of the two, floored at zero.

    This is why RUL is not simply "200 - tool_wear". An M-tier tool at 50 N·m
    has a strain ceiling of 12000/50 = 240 min, so wear binds first at 200. Run
    the same tool at 75 N·m and the ceiling drops to 12000/75 = 160 min, so it
    dies 40 minutes early. Cutting harder does not only use the tool up faster,
    it lowers the ceiling, and no wear threshold would tell you that.

    Cooling faults and power overloads are left out on purpose. They are instant
    failure conditions rather than wear mechanisms, so they do not eat tool life,
    they end it. The alert rules in thresholds.py already handle them.
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

# Minimum number of live readings before a wear-rate trend means anything.
MIN_POINTS_FOR_RATE = 4

# How many recent readings the wear-rate fit uses.
#
# A trade-off between responsiveness and noise. Fit the whole buffer and the
# rate lags badly, so a machine that started cutting hard a minute ago still
# reports a normal rate, which is exactly when the warning is wanted. Fit two
# points and it jumps around with sensor noise. Twenty readings averages out the
# jitter and still reacts within about half a minute at the default tick rate.
WEAR_RATE_WINDOW = 20


@dataclass(frozen=True)
class RulEstimate:
    """What we know about how much life is left."""

    remaining_min: float          # cutting minutes to the first binding limit
    binding_constraint: str       # "tool_wear" or "overstrain"
    total_usable_min: float       # usable tool life at this operating point
    fraction_consumed: float      # 0 to 1, for a progress bar
    wear_limited_min: float       # remaining under the tool-wear limit alone
    strain_limited_min: float     # remaining under the overstrain limit alone

    def to_dict(self) -> dict[str, Any]:
        # An idle spindle (torque near zero) can never reach the overstrain
        # limit, which is mathematically +inf. JSON has no infinity literal, and
        # Python would emit a bare `Infinity` that the browser rejects, so
        # unreachable limits are sent as null.
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
    Remaining tool life in cutting minutes, from the failure physics.

    >>> physics_rul({"tool_wear": 100.0, "torque": 50.0}, "M").remaining_min
    100.0
    """
    tool_wear = float(features["tool_wear"])
    torque = float(features["torque"])
    osf_limit = OSF_STRAIN_LIMIT.get(product_type, OSF_STRAIN_LIMIT["L"])

    # Tool-wear constraint.
    wear_limited = TWF_WEAR_MIN_MIN - tool_wear

    # Overstrain constraint. At zero torque nothing is being cut, so the strain
    # limit is unreachable. Treat that as unbounded rather than dividing by zero.
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
# Layer 3: turning cutting minutes into clock time
# --------------------------------------------------------------------------

def estimate_wear_rate(
    history: Sequence[dict[str, Any]],
    window: int = WEAR_RATE_WINDOW,
) -> float | None:
    """
    Measure the wear rate from recent readings, in tool-wear minutes per
    wall-clock minute.

    Two things make it more than a subtraction. A tool change resets wear to
    zero, and fitting across that reset gives a negative slope, so only the
    readings since the most recent reset are used. And a single pair of points
    is noisy, so we fit a least-squares line over the whole current segment
    rather than differencing the endpoints.

    Returns None when there is not enough clean data to state a rate.
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
    # run now rather than averaging in conditions from minutes ago.
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

    On a real machine tool wear is counted in minutes of cutting, so at nominal
    load one wear-minute passes per wall-clock minute and the rate is 1.0. Under
    heavy load it goes above 1.0, meaning the tool is losing life faster than the
    clock.

    The simulator runs compressed: a 1.5 s tick advances the tool about 2.2
    cutting minutes, so a full tool life is watchable in a couple of minutes
    instead of three hours. Its raw rate is therefore around 88, which is
    arithmetically right and useless on a dashboard. Dividing by the nominal rate
    cancels the compression and leaves something readable, such as "wearing this
    tool 1.6x faster than normal".
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

    Returns None when the rate is unknown. Better to show nothing than a
    deadline we cannot back up.
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
    Raise an alert when little life is left, but only when overstrain is the
    binding constraint.

    The restriction avoids double-alerting. When tool wear binds, the existing
    tool_wear threshold already fires at 180 min and says the same thing, and an
    operator shown two rows for one problem stops reading the list.

    The overstrain case is the one nothing else catches. A tool at 150 min of
    wear looks fine to a wear threshold, but at 75 N·m its ceiling is 160 min and
    it has ten minutes left.
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
