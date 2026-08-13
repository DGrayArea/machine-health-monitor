"""
Tests for the alert logic in backend/alerts.py and backend/thresholds.py.

These do not load the model. They cover the deterministic half of the system:
given a reading, which rules trip, what severity comes out, and what the
operator is told to do. That half has to be right whatever the model does.
"""

from __future__ import annotations

import pytest

from backend.alerts import build_alert, combined_status
from backend.thresholds import derive_features, evaluate_rules, is_plausible


def features(**overrides):
    """A healthy baseline reading, with channels overridden per test."""
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
    # Low delta-T on its own is not a fault, because the spindle is still fast
    # enough to cool itself. This is the AND in the physics.
    hits = evaluate_rules(features(process_temp=306.0, rot_speed=2000))
    assert not any(h.rule_id == "cooling" and h.severity == "Fault" for h in hits)

    # Low delta-T AND low rpm together is the documented failure mode.
    hits = evaluate_rules(features(process_temp=306.0, rot_speed=1200))
    cooling = [h for h in hits if h.rule_id == "cooling"]
    assert cooling and cooling[0].severity == "Fault"
    assert "coolant" in cooling[0].action.lower()


def test_cooling_warning_fires_before_the_fault_limit():
    # A delta-T of 9.0 K is above the 8.6 K fault line but below the 9.5 K
    # warning line, which is the margin predictive maintenance exists to catch.
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
    """The safety case: the model says Normal and the physics disagrees."""
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
    """A confident "Normal" must not silence a hard limit."""
    f = features(tool_wear=250)                     # past end of tool life
    status, alert = build_alert(model_status="Normal", confidence=1.0, features=f)
    assert status == "Fault"
    assert alert is not None


def test_model_alone_can_raise_a_warning():
    """
    The predictive case: every threshold is inside limits but the model
    recognises the pattern. There is no rule to quote, so the advice falls back
    to something general, and an alert must still be raised.
    """
    status, alert = build_alert(model_status="Warning", confidence=0.71,
                                features=features())
    assert status == "Warning"
    assert alert["severity"] == "Warning"
    assert alert["triggered_rules"] == []
    assert "no single limit" in alert["recommended_action"].lower()


def test_alert_message_names_secondary_conditions():
    """If several things are wrong, the operator has to hear about all of them."""
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
    """No alert should be raised without telling someone what to do."""
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


# --------------------------------------------------------- repeat suppression

def test_the_same_condition_is_only_logged_once_per_window():
    """
    A persisting fault produces an alert on every reading. Without suppression
    the simulator writes about 40 identical rows a minute and the history
    becomes unreadable.
    """
    from backend.alerts import alert_signature, reset_suppression, should_log
    from backend.thresholds import evaluate_rules

    reset_suppression()
    hits = evaluate_rules(features(torque=75, rot_speed=1400))
    signature = alert_signature("Fault", hits)

    assert should_log(signature, now=0.0) is True       # first time, always
    assert should_log(signature, now=1.5) is False      # next tick, suppressed
    assert should_log(signature, now=30.0) is False


def test_a_persisting_condition_is_restated_after_the_window():
    from backend.alerts import reset_suppression, should_log
    from backend import config

    reset_suppression()
    assert should_log("Fault|power_high:Fault", now=0.0) is True
    later = config.ALERT_REPEAT_SECONDS + 1
    assert should_log("Fault|power_high:Fault", now=later) is True


def test_a_different_condition_is_never_suppressed():
    """Suppression must not hide a new problem that starts during an old one."""
    from backend.alerts import reset_suppression, should_log

    reset_suppression()
    assert should_log("Fault|power_high:Fault", now=0.0) is True
    assert should_log("Fault|cooling:Fault", now=0.1) is True
    assert should_log("Warning|tool_wear:Warning", now=0.2) is True


def test_the_signature_ignores_measured_values():
    """
    Two readings of the same condition differ in their numbers, so the
    signature has to key on which rules tripped, not on the message text.
    """
    from backend.alerts import alert_signature
    from backend.thresholds import evaluate_rules

    a = alert_signature("Fault", evaluate_rules(features(torque=75, rot_speed=1400)))
    b = alert_signature("Fault", evaluate_rules(features(torque=76, rot_speed=1390)))
    assert a == b


def test_build_alert_returns_a_signature():
    _, alert = build_alert(model_status="Normal", confidence=0.9,
                           features=features(tool_wear=245))
    assert alert["_signature"]
    assert "tool_wear" in alert["_signature"]


def test_a_flickering_trend_does_not_re_log_the_underlying_fault():
    """
    Trend rules come and go as the fit wobbles near its threshold. If they were
    part of the suppression key, each flicker would re-log the measured fault
    and the alert history would fill up with the same overload again and again.
    """
    from backend.alerts import alert_signature
    from backend.thresholds import RuleHit, evaluate_rules

    measured = evaluate_rules(features(torque=75, rot_speed=1400))
    rising = RuleHit("trend_power_rising", "Warning", "Load is climbing", "", "")
    falling = RuleHit("trend_power_falling", "Warning", "Load is falling", "", "")

    plain = alert_signature("Fault", measured)
    with_rising = alert_signature("Fault", measured + [rising])
    with_falling = alert_signature("Fault", measured + [falling])

    assert plain == with_rising == with_falling


def test_trend_only_alerts_keep_their_own_identity():
    """Two different projections must not suppress each other."""
    from backend.alerts import alert_signature
    from backend.thresholds import RuleHit

    cooling = RuleHit("trend_temp_diff_falling", "Warning", "Cooling", "", "")
    power = RuleHit("trend_power_rising", "Warning", "Load", "", "")
    assert alert_signature("Warning", [cooling]) != alert_signature("Warning", [power])
