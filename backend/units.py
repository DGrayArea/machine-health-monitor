"""
One internal unit per quantity, converted only at the edges.

Everything inside the system stores SI base units: kelvin, watts,
newton-metres. That covers the dataset, the model features, the thresholds in
backend/thresholds.py, the database and the JSON API. Conversion to friendlier
units happens only where a person reads the number, so the dashboard, the PDF
and the CSV.

Why not store Celsius everywhere? The model was trained on kelvin and the
failure thresholds are quoted in kelvin. Converting at the boundary keeps it to
one place where a unit can go wrong. If Celsius reached the feature vector the
model would get 25.4 where it expects 298.5 and return a wrong answer without
raising anything. Unit mismatches are quiet, which is what makes them expensive,
so the boundary is worth being explicit about.

Display units
    temperature   K      -> °C     298.5 K means nothing on a panel
    temp. delta   K      -> °C     a difference of 10 K is exactly 10 °C
    power         W      -> kW     machine-tool datasheets quote kW
    torque        N·m               already the practical unit
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
    Temperature difference, K -> °C.

    This is the identity function, written out on purpose. A difference of 10 K
    is a difference of 10 °C because the offset cancels. Writing
    `temp_diff - 273.15` is an easy mistake to make, so this exists to make the
    right behaviour obvious at every call site.
    """
    return delta


def watts_to_kilowatts(watts: float) -> float:
    return watts / 1000.0


# Suffixes for CSV column names, so a downloaded file says what its units are.
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
