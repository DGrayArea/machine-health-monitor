"""
The machine's operating envelope, defined once and used everywhere.

Two places import this module:
  scripts/clean_data.py  labels the training set offline
  backend/alerts.py      decides what to tell the operator live

Keeping them on the same numbers matters. If the offline labels used one set of
thresholds and the live alerting used another, the model would be trained to
spot one thing and deployed to explain something slightly different, which is a
quiet bug and a horrible one to find. tests/test_thresholds.py checks the two
paths agree across all 10,000 rows.

Every number here comes from the AI4I 2020 dataset's documented failure physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Fault thresholds: the limits at which the machine actually fails.
# --------------------------------------------------------------------------
HDF_TEMP_DIFF_K = 8.6        # heat dissipation failure below this delta-T ...
HDF_SPEED_RPM = 1380         # ... while also below this rotational speed
PWF_POWER_MIN_W = 3500       # drivetrain power envelope, lower bound
PWF_POWER_MAX_W = 9000       # drivetrain power envelope, upper bound
OSF_STRAIN_LIMIT = {         # overstrain limit rises with product quality tier
    "L": 11000,              # Low quality
    "M": 12000,              # Medium quality
    "H": 13000,              # High quality
}
TWF_WEAR_MIN_MIN = 200       # tools start breaking at this much wear
TWF_WEAR_MAX_MIN = 240       # tools are always dead by here

# --------------------------------------------------------------------------
# Warning thresholds: margin before the fault limit. Catching a problem here
# rather than at the line above is the point of predictive maintenance.
# --------------------------------------------------------------------------
WARN_TEMP_DIFF_K = 9.5       # ~10% margin above the 8.6 K failure delta-T
WARN_SPEED_RPM = 1500        # ~9% margin above the 1380 rpm failure speed
WARN_POWER_MIN_W = 4000
WARN_POWER_MAX_W = 8500
WARN_STRAIN_FRACTION = 0.85  # 85% of the tier's overstrain limit
WARN_WEAR_MIN = 180          # last ~10% of the 200 min expected tool life

# --------------------------------------------------------------------------
# Plausibility bounds. Outside these, assume the sensor is broken rather than
# the machine. The cleaner drops such rows and the API rejects such payloads.
# --------------------------------------------------------------------------
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "air_temp": (250.0, 350.0),      # K
    "process_temp": (250.0, 360.0),  # K
    "rot_speed": (1.0, 5000.0),      # rpm, and it must be turning
    "torque": (0.0, 200.0),          # Nm
    "tool_wear": (0.0, 500.0),       # min
}

TYPE_CODE = {"L": 0, "M": 1, "H": 2}
CLASS_ORDER = ["Normal", "Warning", "Fault"]


# --------------------------------------------------------------------------

def derive_features(
    *,
    air_temp: float,
    process_temp: float,
    rot_speed: float,
    torque: float,
    tool_wear: float,
    product_type: str = "M",
) -> dict[str, float]:
    """
    Turn the five raw sensor channels plus the quality tier into the nine
    model features.

    This has to match how scripts/train_model.py builds its features. That is
    why the saved model bundle also stores the feature order, so the backend can
    rebuild each row against it instead of trusting this to stay in step.

      temp_diff = process_temp - air_temp                [K]
      power     = torque * omega,  omega = rpm*2*pi/60   [W]
      strain    = tool_wear * torque                     [min*N*m]
    """
    omega = rot_speed * 2.0 * math.pi / 60.0
    return {
        "type_code": float(TYPE_CODE.get(product_type, 0)),
        "air_temp": float(air_temp),
        "process_temp": float(process_temp),
        "rot_speed": float(rot_speed),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
        "temp_diff": float(process_temp - air_temp),
        "power": float(torque * omega),
        "strain": float(tool_wear * torque),
    }


@dataclass(frozen=True)
class RuleHit:
    """One tripped rule, with everything the operator needs to act on it."""
    rule_id: str          # stable id, such as "cooling"
    severity: str         # "Warning" or "Fault"
    title: str            # short label
    detail: str           # the measured value against the limit
    action: str           # what a technician should physically do


def evaluate_rules(features: dict[str, float], product_type: str = "M") -> list[RuleHit]:
    """
    Check every rule against one reading, worst first.

    Returns all the rules that tripped, not just the first. A machine can be
    overheating and overstrained at the same time, and the technician needs to
    know about both.
    """
    hits: list[RuleHit] = []
    temp_diff = features["temp_diff"]
    rot_speed = features["rot_speed"]
    power = features["power"]
    strain = features["strain"]
    tool_wear = features["tool_wear"]
    osf_limit = OSF_STRAIN_LIMIT.get(product_type, OSF_STRAIN_LIMIT["L"])

    # ---- Heat dissipation -------------------------------------------------
    # Cooling relies on air being pushed over the spindle, so a small delta-T is
    # only dangerous when the spindle is turning slowly too. Hence the AND.
    if temp_diff < HDF_TEMP_DIFF_K and rot_speed < HDF_SPEED_RPM:
        hits.append(RuleHit(
            "cooling", "Fault", "Heat dissipation failure",
            f"\u0394T {temp_diff:.1f} \u00b0C < {HDF_TEMP_DIFF_K} \u00b0C "
            f"at {rot_speed:.0f} rpm < {HDF_SPEED_RPM} rpm",
            "Stop the machine. Check coolant flow, clean the heat exchanger "
            "and verify the cooling fan is running before restart.",
        ))
    elif temp_diff < WARN_TEMP_DIFF_K and rot_speed < WARN_SPEED_RPM:
        hits.append(RuleHit(
            "cooling", "Warning", "Cooling margin low",
            f"\u0394T {temp_diff:.1f} \u00b0C approaching {HDF_TEMP_DIFF_K} \u00b0C "
            f"at {rot_speed:.0f} rpm",
            "Inspect coolant level and airflow path. Raise spindle speed to "
            "improve forced convection if the process allows.",
        ))

    # ---- Power envelope ---------------------------------------------------
    # Two-sided. Too little power means the drive is slipping or decoupled, too
    # much means the cut is overloading the motor.
    if power > PWF_POWER_MAX_W:
        hits.append(RuleHit(
            "power_high", "Fault", "Power overload",
            f"{power / 1000:.2f} kW > {PWF_POWER_MAX_W / 1000:.1f} kW limit",
            "Reduce load immediately — cut feed rate or depth of cut. "
            "Check for a blunt tool or incorrect material.",
        ))
    elif power < PWF_POWER_MIN_W:
        hits.append(RuleHit(
            "power_low", "Fault", "Power underrun",
            f"{power / 1000:.2f} kW < {PWF_POWER_MIN_W / 1000:.1f} kW limit",
            "Check the drive coupling and belt tension — the spindle may be "
            "slipping or running unloaded.",
        ))
    elif power > WARN_POWER_MAX_W:
        hits.append(RuleHit(
            "power_high", "Warning", "Power near upper limit",
            f"{power / 1000:.2f} kW approaching {PWF_POWER_MAX_W / 1000:.1f} kW limit",
            "Reduce feed rate to bring spindle power back inside the envelope.",
        ))
    elif power < WARN_POWER_MIN_W:
        hits.append(RuleHit(
            "power_low", "Warning", "Power near lower limit",
            f"{power / 1000:.2f} kW approaching {PWF_POWER_MIN_W / 1000:.1f} kW limit",
            "Verify the workpiece is engaged and the drive coupling is tight.",
        ))

    # ---- Mechanical overstrain -------------------------------------------
    # A worn tool needs more torque for the same cut, so wear x torque is what
    # breaks the tool. Neither one on its own tells you enough.
    if strain > osf_limit:
        hits.append(RuleHit(
            "overstrain", "Fault", "Mechanical overstrain",
            f"{strain:.0f} min·N·m > {osf_limit} min·N·m limit "
            f"(quality tier {product_type})",
            "Stop and replace the tool. Inspect the spindle bearing and "
            "tool holder for damage before resuming.",
        ))
    elif strain > WARN_STRAIN_FRACTION * osf_limit:
        hits.append(RuleHit(
            "overstrain", "Warning", "Approaching overstrain limit",
            f"{strain:.0f} min·N·m at {strain / osf_limit * 100:.0f}% "
            f"of the {osf_limit} min·N·m limit",
            "Reduce load or change the tool early. Inspect the bearing at the "
            "next available stop.",
        ))

    # ---- Tool wear --------------------------------------------------------
    if tool_wear >= TWF_WEAR_MAX_MIN:
        hits.append(RuleHit(
            "tool_wear", "Fault", "Tool life exceeded",
            f"{tool_wear:.0f} min >= {TWF_WEAR_MAX_MIN} min maximum",
            "Replace the cutting tool now. Do not start another cycle.",
        ))
    elif tool_wear > WARN_WEAR_MIN:
        hits.append(RuleHit(
            "tool_wear", "Warning", "Tool nearing end of life",
            f"{tool_wear:.0f} min of ~{TWF_WEAR_MIN_MIN} min expected life",
            f"Schedule a tool change within the next "
            f"{max(TWF_WEAR_MIN_MIN - tool_wear, 0):.0f} minutes of cutting.",
        ))

    # Faults first, so hits[0] is the most urgent thing on screen.
    hits.sort(key=lambda h: 0 if h.severity == "Fault" else 1)
    return hits


def is_plausible(features: dict[str, float]) -> tuple[bool, str | None]:
    """Reject readings that are physically impossible, meaning a broken sensor."""
    for channel, (lo, hi) in PLAUSIBLE.items():
        value = features.get(channel)
        if value is None:
            continue
        if not (lo <= value <= hi):
            return False, (
                f"{channel}={value} is outside the plausible range "
                f"[{lo}, {hi}]. Suspect a sensor fault, not a machine fault."
            )
    return True, None
