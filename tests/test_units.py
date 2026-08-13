"""
Unit-conversion tests.

Small, but worth having: a unit error is silent. Nothing crashes, no exception
is raised, the dashboard just shows a confidently wrong number. The one that
matters most is `delta_kelvin_to_celsius` — subtracting 273.15 from a
temperature *difference* is the single easiest mistake to make here, and it
would turn a healthy ΔT of 10 into -263 and trip a permanent cooling fault.
"""

from __future__ import annotations

import pytest

from backend import units


def test_absolute_temperature_conversion():
    assert units.kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert units.kelvin_to_celsius(298.15) == pytest.approx(25.0)
    assert units.celsius_to_kelvin(25.0) == pytest.approx(298.15)


def test_conversion_round_trips():
    for celsius in (-40.0, 0.0, 21.5, 100.0):
        assert units.celsius_to_kelvin(
            units.kelvin_to_celsius(units.celsius_to_kelvin(celsius))
        ) == pytest.approx(units.celsius_to_kelvin(celsius))


def test_temperature_difference_has_no_offset():
    """A DIFFERENCE of 10 K is 10 °C. Subtracting 273.15 here would be a bug."""
    assert units.delta_kelvin_to_celsius(10.0) == pytest.approx(10.0)
    assert units.delta_kelvin_to_celsius(8.6) == pytest.approx(8.6)


def test_power_conversion():
    assert units.watts_to_kilowatts(6951.0) == pytest.approx(6.951)
    assert units.watts_to_kilowatts(9000.0) == pytest.approx(9.0)


def test_dataset_nominal_point_is_a_sane_shop_floor_temperature():
    """
    Sanity check against reality: the AI4I nominal air temperature of 298.1 K
    should land around 25 °C. If a conversion ever inverts, this catches it.
    """
    celsius = units.kelvin_to_celsius(298.1)
    assert 20.0 < celsius < 30.0
