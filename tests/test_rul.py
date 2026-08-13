"""
Tests for remaining useful life, in backend/rul.py.

The behaviour these cover:
  RUL is not just "200 - tool_wear", because high torque lowers the ceiling.
  The wear-rate estimator survives a tool change, which resets wear to zero.
  The RUL alert rule does not double up with the tool-wear threshold.
  The vectorised target in clean_data.py matches the scalar physics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.rul import (
    RUL_CRITICAL_MIN,
    RUL_WARNING_MIN,
    estimate_wear_rate,
    physics_rul,
    project_wallclock,
    rul_band,
    rul_rule,
)
from backend.thresholds import TWF_WEAR_MIN_MIN, derive_features

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"


def features(**overrides):
    base = dict(air_temp=298.0, process_temp=308.0, rot_speed=1550,
                torque=42.0, tool_wear=0.0, product_type="M")
    base.update(overrides)
    return derive_features(**base)


# ------------------------------------------------------------ physics layer

def test_fresh_tool_has_full_life():
    estimate = physics_rul(features(tool_wear=0, torque=40), "M")
    assert estimate.remaining_min == pytest.approx(TWF_WEAR_MIN_MIN)
    assert estimate.binding_constraint == "tool_wear"
    assert estimate.fraction_consumed == pytest.approx(0.0)


def test_wear_consumes_life_one_for_one_at_moderate_load():
    # At 50 Nm the strain ceiling is 12000/50 = 240 min, above the 200 min wear
    # limit, so tool wear is what binds and RUL is simply 200 - wear.
    estimate = physics_rul(features(tool_wear=120, torque=50), "M")
    assert estimate.remaining_min == pytest.approx(80.0)
    assert estimate.binding_constraint == "tool_wear"


def test_high_torque_lowers_the_ceiling_not_just_the_rate():
    """
    The behaviour that matters most. At 75 N·m the overstrain ceiling is
    12000/75 = 160 min, so a tool at 150 min of wear has 10 minutes left, not
    the 50 a tool-wear threshold on its own would report.
    """
    estimate = physics_rul(features(tool_wear=150, torque=75), "M")
    assert estimate.binding_constraint == "overstrain"
    assert estimate.total_usable_min == pytest.approx(160.0)
    assert estimate.remaining_min == pytest.approx(10.0)

    # Same wear, gentler cut, so far more life left.
    gentle = physics_rul(features(tool_wear=150, torque=40), "M")
    assert gentle.binding_constraint == "tool_wear"
    assert gentle.remaining_min == pytest.approx(50.0)


def test_quality_tier_changes_the_ceiling():
    # strain ceiling = limit / 70 Nm  ->  L: 157, M: 171, H: 186 min
    kwargs = dict(tool_wear=150, torque=70)
    ceilings = {
        tier: physics_rul(features(product_type=tier, **kwargs), tier).total_usable_min
        for tier in ("L", "M", "H")
    }
    assert ceilings["L"] < ceilings["M"] < ceilings["H"]


def test_rul_never_goes_negative():
    assert physics_rul(features(tool_wear=260, torque=50), "M").remaining_min == 0.0
    assert physics_rul(features(tool_wear=190, torque=90), "M").remaining_min == 0.0


def test_idle_spindle_does_not_divide_by_zero():
    """Zero torque means nothing is being cut, so overstrain cannot be reached."""
    estimate = physics_rul(features(tool_wear=50, torque=0.0), "M")
    assert estimate.binding_constraint == "tool_wear"
    assert estimate.remaining_min == pytest.approx(150.0)
    # It also has to serialise, since JSON has no infinity literal.
    assert estimate.to_dict()["strain_limited_min"] is None


@pytest.mark.parametrize("remaining,expected", [
    (120.0, "ok"),
    (RUL_WARNING_MIN - 1, "warning"),
    (RUL_CRITICAL_MIN - 1, "critical"),
    (0.0, "critical"),
])
def test_band_thresholds(remaining, expected):
    assert rul_band(remaining) == expected


# ------------------------------------------------------- wear-rate estimator

def _history(wears: list[float], step_seconds: int = 90) -> list[dict]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {"timestamp": (start + timedelta(seconds=i * step_seconds)).isoformat(),
         "reading": {"tool_wear": w}}
        for i, w in enumerate(wears)
    ]


def test_wear_rate_needs_enough_points():
    assert estimate_wear_rate(_history([0, 2])) is None


def test_wear_rate_measures_the_observed_slope():
    # 2.0 wear-minutes every 90 s  ->  2.0 / 1.5 min = 1.333 per wall-clock min
    rate = estimate_wear_rate(_history([0, 2, 4, 6, 8, 10]))
    assert rate == pytest.approx(4.0 / 3.0, rel=1e-6)


def test_wear_rate_ignores_readings_before_a_tool_change():
    """
    A tool change resets wear to zero. Fitting across the reset would give a
    negative slope and a nonsense RUL, so only the current segment counts.
    """
    history = _history([180, 184, 188, 0, 3, 6, 9, 12])
    rate = estimate_wear_rate(history)
    assert rate is not None and rate > 0
    assert rate == pytest.approx(2.0, rel=1e-6)   # 3 wear-min per 90 s


def test_wear_rate_is_none_when_wear_is_flat():
    """An idle machine gains no wear, so no deadline can be projected."""
    assert estimate_wear_rate(_history([50, 50, 50, 50, 50])) is None


def test_wallclock_projection():
    # 60 cutting minutes left, burning 2 wear-min per wall-clock min -> 30 min
    assert project_wallclock(60.0, 2.0) == pytest.approx(30.0)
    assert project_wallclock(60.0, None) is None
    assert project_wallclock(60.0, 0.0) is None


def test_wallclock_projection_cancels_simulator_time_compression():
    """
    The simulator advances the tool about 88 wear-minutes per wall-clock minute,
    so a tool life is watchable in a couple of minutes. Dividing by that nominal
    rate has to give a real-machine answer, not "less than a minute left".
    """
    from backend.rul import normalise_wear_rate

    nominal = 88.0
    # Running at nominal duty gives 1.0x, and RUL in machine minutes equals the
    # remaining cutting minutes.
    assert normalise_wear_rate(88.0, nominal) == pytest.approx(1.0)
    assert project_wallclock(60.0, 88.0, nominal) == pytest.approx(60.0)

    # Running hard means wearing at 2x, so the deadline arrives twice as soon.
    assert normalise_wear_rate(176.0, nominal) == pytest.approx(2.0)
    assert project_wallclock(60.0, 176.0, nominal) == pytest.approx(30.0)


# ----------------------------------------------------------- the alert rule

def test_no_rul_alert_on_a_healthy_tool():
    assert rul_rule(features(tool_wear=40, torque=45), "M") is None


def test_no_rul_alert_when_tool_wear_is_the_binding_constraint():
    """
    Wear at 195 min is nearly spent, but the tool_wear threshold rule already
    covers that. An RUL alert as well would show two rows for one problem.
    """
    assert rul_rule(features(tool_wear=195, torque=40), "M") is None


def test_rul_alert_fires_for_the_case_nothing_else_catches():
    """
    150 min of wear looks healthy to a wear threshold, which trips at 180, but
    at 75 N·m the ceiling is 160 min so there are 10 minutes left. This is the
    gap RUL exists to fill.
    """
    hit = rul_rule(features(tool_wear=150, torque=75), "M")
    assert hit is not None
    assert hit.rule_id == "rul"
    assert hit.severity == "Warning"
    assert "160 min" in hit.detail
    assert "reduce torque" in hit.action.lower()


def test_rul_alert_escalates_to_fault_when_life_is_gone():
    hit = rul_rule(features(tool_wear=159, torque=75), "M")
    assert hit is not None and hit.severity == "Fault"
    assert "change the tool now" in hit.action.lower()


def test_rul_rule_reaches_the_combined_alert():
    """
    Note the rpm. A constant-power drive at 75 N·m runs at about 900 rpm, so
    7.1 kW. At the helper's default 1550 rpm the same torque would draw 12.2 kW
    and trip the power fault instead, hiding what this test is checking. The
    operating point has to be physically coherent, not just convenient.
    """
    from backend.alerts import build_alert
    status, alert = build_alert(
        model_status="Normal", confidence=0.99,
        features=features(tool_wear=150, torque=75, rot_speed=900),
        product_type="M",
    )
    assert status == "Warning"
    assert any(r["rule_id"] == "rul" for r in alert["triggered_rules"])


# ------------------------------------------------- offline / online agreement

@pytest.mark.skipif(not CLEAN_CSV.exists(), reason="run scripts/clean_data.py first")
def test_vectorised_target_matches_the_scalar_physics():
    """
    clean_data.add_rul_target computes the training target with pandas and the
    API computes it one row at a time. They have to agree, or the model is
    trained against a different definition from the one it is deployed with.
    """
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(CLEAN_CSV)

    mismatches = []
    for row in df.head(2000).itertuples(index=False):
        scalar = physics_rul(
            derive_features(
                air_temp=row.air_temp, process_temp=row.process_temp,
                rot_speed=row.rot_speed, torque=row.torque,
                tool_wear=row.tool_wear, product_type=row.type,
            ),
            row.type,
        )
        if abs(scalar.remaining_min - row.rul_minutes) > 1e-6:
            mismatches.append((row.udi, scalar.remaining_min, row.rul_minutes))
        if scalar.binding_constraint != row.rul_binding:
            mismatches.append((row.udi, scalar.binding_constraint, row.rul_binding))

    assert not mismatches, f"{len(mismatches)} disagreements, first: {mismatches[:3]}"


# ------------------------------------------------------------------ live API

def test_predict_endpoint_returns_remaining_life(client, auth_headers, model_available):
    if not model_available:
        pytest.skip("no trained model")

    # 150 min wear at 75 Nm -> the overstrain-bound case.
    reading = {"air_temp": 298.0, "process_temp": 308.0, "rot_speed": 900,
               "torque": 75.0, "tool_wear": 150, "product_type": "M"}
    body = client.post("/api/predict", json=reading, headers=auth_headers).json()

    life = body["remaining_life"]
    assert life["remaining_min"] == pytest.approx(10.0)
    assert life["binding_constraint"] == "overstrain"
    assert life["band"] == "warning"
    assert life["source"] == "physics"
    assert 0.0 <= life["fraction_consumed"] <= 1.0


def test_remaining_life_is_written_to_the_audit_trail(
    client, auth_headers, model_available
):
    if not model_available:
        pytest.skip("no trained model")
    from backend import database

    reading = {"air_temp": 298.0, "process_temp": 308.0, "rot_speed": 900,
               "torque": 75.0, "tool_wear": 150, "product_type": "M"}
    client.post("/api/predict", json=reading, headers=auth_headers)

    row = database.recent_predictions()[0]
    assert row["rul_minutes"] == pytest.approx(10.0)
    assert row["rul_binding"] == "overstrain"


def test_simulator_records_carry_remaining_life(model_available):
    if not model_available:
        pytest.skip("no trained model")
    from backend.simulator import SimulationRunner

    import time

    runner = SimulationRunner(interval=0.01, buffer_size=30)
    for _ in range(8):
        record = runner.tick()
        # Real elapsed time between ticks, so the wear-rate slope has a non-zero
        # denominator. Timestamps are millisecond resolution, so 15 ms is plenty.
        time.sleep(0.015)

    life = record["remaining_life"]
    assert life["remaining_min"] >= 0
    assert life["binding_constraint"] in ("tool_wear", "overstrain")
    # After several ticks the buffer is long enough to measure a wear rate.
    assert life["wear_rate_per_min"] is not None
    assert life["wear_rate_per_min"] > 0
    assert life["wallclock_remaining_min"] is not None
