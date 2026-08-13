"""
Unit tests for the alert logic (backend/alerts.py + backend/thresholds.py).

These tests do NOT load the model. They pin down the deterministic half of the
system: given a reading, which physical rules trip, what severity results, and
what the operator is told to do. That half must be correct regardless of how
the model behaves.
"""

from __future__ import annotations

import pytest

from backend.alerts import build_alert, combined_status
from backend.thresholds import derive_features, evaluate_rules, is_plausible


def features(**overrides):
    """A healthy baseline reading, with named channels overridden per test."""
    base = dict(air_temp=298.0, process_temp=308.0, rot_speed=1550,
                torque=42.0, tool_wear=0.0, product_type="M")
    base.update(overrides)
    return derive_features(**base)


# ---------------------------------------------------------------- features

def test_derived_features_match_the_physics():
    f = features(rot_speed=1551, torque=42.8, tool_wear=50)
    assert f["temp_diff"] == pytest.approx(10.0)
    # P = tau * omega = 42.8 * (1551 * 2pi/60) = ~6951 W
    assert f["power"] == pytest.approx(6951, rel=0.01)
    assert f["strain"] == pytest.approx(50 * 42.8)


def test_implausible_reading_is_rejected():
    ok, _ = is_plausible(features())
    assert ok

    bad, reason = is_plausible(features(rot_speed=0.0))
    assert not bad
    assert "rot_speed" in reason


# ---------------------------------------------------------------- no alert

def test_healthy_reading_raises_no_rules_and_no_alert():
    assert evaluate_rules(features()) == []

    status, alert = build_alert(model_status="Normal", confidence=0.99,
                                features=features())
    assert status == "Normal"
    assert alert is None


# ---------------------------------------------------------------- each rule

def test_heat_dissipation_fault_needs_both_conditions():
    # Low delta-T alone is not a fault — the spindle is still spinning fast
    # enough to cool itself. This is the AND in the physics.
    hits = evaluate_rules(features(process_temp=306.0, rot_speed=2000))
    assert not any(h.rule_id == "cooling" and h.severity == "Fault" for h in hits)

    # Low delta-T AND low rpm together is the documented failure mode.
    hits = evaluate_rules(features(process_temp=306.0, rot_speed=1200))
    cooling = [h for h in hits if h.rule_id == "cooling"]
    assert cooling and cooling[0].severity == "Fault"
    assert "coolant" in cooling[0].action.lower()


def test_cooling_warning_fires_before_the_fault_limit():
    # delta-T of 9.0 K is above the 8.6 K fault line but below the 9.5 K warning
    # line — this is exactly the margin predictive maintenance exists to catch.
    hits = evaluate_rules(features(process_temp=307.0, rot_speed=1400))
    cooling = [h for h in hits if h.rule_id == "cooling"]
    assert cooling and cooling[0].severity == "Warning"


def test_power_envelope_is_two_sided():
    high = evaluate_rules(features(torque=70, rot_speed=1400))   # ~10.3 kW
    assert any(h.rule_id == "power_high" and h.severity == "Fault" for h in high)

    low = evaluate_rules(features(torque=12, rot_speed=1600))    # ~2.0 kW
    assert any(h.rule_id == "power_low" and h.severity == "Fault" for h in low)


def test_overstrain_limit_depends_on_quality_tier():
    # strain = 220 * 55 = 12100 min*Nm.
    # Over the L limit (11000) and the M limit (12000), under the H one (13000).
    kwargs = dict(tool_wear=220, torque=55)

    for tier, expect_fault in (("L", True), ("M", True), ("H", False)):
        f = features(product_type=tier, **kwargs)
        hits = evaluate_rules(f, product_type=tier)
        is_fault = any(h.rule_id == "overstrain" and h.severity == "Fault"
                       for h in hits)
        assert is_fault is expect_fault, f"tier {tier}"


def test_tool_wear_ladder():
    assert not [h for h in evaluate_rules(features(tool_wear=100))
                if h.rule_id == "tool_wear"]

    warn = [h for h in evaluate_rules(features(tool_wear=190))
            if h.rule_id == "tool_wear"]
    assert warn and warn[0].severity == "Warning"

    fault = [h for h in evaluate_rules(features(tool_wear=245))
             if h.rule_id == "tool_wear"]
    assert fault and fault[0].severity == "Fault"


def test_faults_are_sorted_before_warnings():
    # Worn tool (Warning) plus power overload (Fault) at the same time.
    hits = evaluate_rules(features(tool_wear=190, torque=75, rot_speed=1400))
    assert len(hits) >= 2
    assert hits[0].severity == "Fault"


# ---------------------------------------------------------- combination rule

def test_rules_escalate_the_model():
    """The safety-critical case: model says Normal, physics says otherwise."""
    assert combined_status("Normal", evaluate_rules(features(torque=75,
                                                             rot_speed=1400))) == "Fault"

    status, alert = build_alert(
        model_status="Normal", confidence=0.97,
        features=features(torque=75, rot_speed=1400),
    )
    assert status == "Fault"
    assert alert["severity"] == "Critical"
    assert "reduce load" in alert["recommended_action"].lower()


def test_model_can_never_suppress_a_tripped_rule():
    """A confident 'Normal' must not silence a hard physical limit."""
    f = features(tool_wear=250)                     # past end of tool life
    status, alert = build_alert(model_status="Normal", confidence=1.0, features=f)
    assert status == "Fault"
    assert alert is not None


def test_model_alone_can_raise_a_warning():
    """
    The genuinely predictive case: every threshold is inside limits, but the
    model recognises the pattern. There is no rule to quote, so we fall back to
    generic advice — and we must still produce an alert.
    """
    status, alert = build_alert(model_status="Warning", confidence=0.71,
                                features=features())
    assert status == "Warning"
    assert alert["severity"] == "Warning"
    assert alert["triggered_rules"] == []
    assert "no single limit" in alert["recommended_action"].lower()


def test_alert_message_names_secondary_conditions():
    """If several things are wrong, the operator must be told about all of them."""
    _, alert = build_alert(
        model_status="Fault", confidence=0.88,
        features=features(tool_wear=190, torque=75, rot_speed=1400),
    )
    assert "Also tripped" in alert["message"]


@pytest.mark.parametrize("model_status,expected_severity", [
    ("Warning", "Warning"),
    ("Fault", "Critical"),
])
def test_severity_ladder(model_status, expected_severity):
    _, alert = build_alert(model_status=model_status, confidence=0.8,
                           features=features())
    assert alert["severity"] == expected_severity


def test_every_alert_carries_an_actionable_instruction():
    """No alert may ever be raised without telling someone what to do."""
    scenarios = [
        features(process_temp=306.0, rot_speed=1200),   # cooling fault
        features(torque=75, rot_speed=1400),            # power overload
        features(torque=12, rot_speed=1600),            # power underrun
        features(tool_wear=220, torque=55),             # overstrain
        features(tool_wear=245),                        # tool life exceeded
        features(tool_wear=190),                        # tool wear warning
    ]
    for f in scenarios:
        _, alert = build_alert(model_status="Normal", confidence=0.9, features=f)
        assert alert is not None
        assert len(alert["recommended_action"]) > 20
        assert alert["title"]
