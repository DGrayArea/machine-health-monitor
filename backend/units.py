"""
Units — one canonical internal unit per quantity, converted only at the edges.

THE RULE THIS FILE ENFORCES
    Everything inside the system — the dataset, the model features, the
    thresholds in backend/thresholds.py, the database, the JSON API — stores
    **SI base units**: kelvin, watts, newton-metres. Conversion to
    human-readable units happens ONLY at the presentation layer: the dashboard,
    the PDF, and the CSV.

    Why not just store Celsius everywhere? Because the model was trained on
    kelvin and the failure thresholds are quoted in kelvin. Converting at the
    boundary means there is exactly one place where a unit can be wrong. If
    Celsius leaked into the feature vector, the model would receive 25.4 where
    it expects 298.5 and produce confident nonsense with no error raised.

    This is the single most common source of silent bugs in instrumented
    systems — the Mars Climate Orbiter was lost to exactly this class of
    mistake — so the boundary is worth making explicit.

DISPLAY UNITS
    temperature   K      -> °C     (readable; 298.5 K means nothing on a panel)
    temp. delta   K      -> °C     (a *difference* of 10 K IS 10 °C exactly)
    power         W      -> kW     (spindle power is quoted in kW in every
                                    machine-tool datasheet)
    torque        N·m               already the practical SI unit
    speed         rpm               the standard for rotating machinery
    tool wear     min               minutes of cutting
    strain        min·N·m           wear x torque
"""

from __future__ import annotations

# 0 °C in kelvin. Exact by definition.
KELVIN_OFFSET = 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Absolute temperature K -> °C."""
    return kelvin - KELVIN_OFFSET


def celsius_to_kelvin(celsius: float) -> float:
    """Absolute temperature °C -> K."""
    return celsius + KELVIN_OFFSET


def delta_kelvin_to_celsius(delta: float) -> float:
    """
    Temperature DIFFERENCE K -> °C.

    Identity, deliberately spelled out. A difference of 10 K is a difference of
    10 °C — the offset cancels. Writing `temp_diff - 273.15` is a real and easy
    mistake, so this function exists to make the correct behaviour explicit at
    every call site rather than relying on whoever reads the code next.
    """
    return delta


def watts_to_kilowatts(watts: float) -> float:
    return watts / 1000.0


# Suffixes used on CSV column names so a downloaded file is self-describing.
CSV_UNIT_SUFFIX = {
    "air_temp": "air_temp_c",
    "process_temp": "process_temp_c",
    "temp_diff": "temp_diff_c",
    "rot_speed": "rot_speed_rpm",
    "torque": "torque_nm",
    "power": "power_kw",
    "tool_wear": "tool_wear_min",
    "strain": "strain_min_nm",
    "rul_minutes": "rul_min",
}
