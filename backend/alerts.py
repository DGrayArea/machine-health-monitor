"""
Turning a prediction into an alert someone can act on.

How the model and the rules combine
    An alert is not just whatever the model said. It draws on two independent
    sources:

      the model    learned and probabilistic, can spot combinations of readings
                   that no single threshold covers
      the rules    deterministic limits, from backend/thresholds.py

    The combination rule: the thresholds can escalate the model, but the model
    can never overrule a threshold. Effective severity is the worse of the two.

    The reasoning is that a hard limit, such as power above 9 kW or a tool past
    its life, is a fact rather than a prediction. If the model has a bad moment
    and says "Normal" while the spindle draws 9.5 kW, the operator still needs
    telling. Safety interlocks work the same way: a learned layer can add
    sensitivity but never gets to switch off a hard limit.

    The other direction is allowed and is where the real prediction happens. If
    every threshold is inside limits but the model has learned that this
    particular combination of readings tends to come before failures, it raises
    a Warning by itself. Thresholds cannot do that.

Severity
    Normal  -> no alert, nothing logged
    Warning -> plan a fix, the machine keeps running
    Fault   -> Critical, act now, the machine should stop
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict

from backend import config
from backend.rul import rul_rule
from backend.thresholds import RuleHit, evaluate_rules

# Model status -> alert severity
SEVERITY_BY_STATUS = {"Normal": "Info", "Warning": "Warning", "Fault": "Critical"}

# Ranking used to take "the worst of" two opinions.
_RANK = {"Normal": 0, "Warning": 1, "Fault": 2}

# Advice used when the model raises a status but no threshold tripped, so there
# is no single limit to point the operator at.
MODEL_ONLY_ACTION = {
    "Warning": (
        "No single limit has been crossed, but this combination of readings "
        "matches the pattern that precedes failures in the training data. "
        "Inspect the machine at the next planned stop and log the condition."
    ),
    "Fault": (
        "No single limit has been crossed, but the readings closely match "
        "recorded failure conditions. Stop the machine and inspect the spindle, "
        "bearing and tool before continuing production."
    ),
}


# --------------------------------------------------------------------------
# Repeat suppression
# --------------------------------------------------------------------------
#
# A fault condition persists for many readings. The simulator ticks every 1.5 s,
# so an unsuppressed cooling fault writes about 40 identical rows a minute, and
# the alert history turns into one condition repeated until it scrolls off the
# screen. That is the fastest way to make an operator stop reading alerts.
#
# The rule: log an alert when what it says CHANGES, or when the same thing has
# been going on for ALERT_REPEAT_SECONDS and is worth restating. The signature
# is (effective status + which rules tripped), not the message text, because the
# message embeds live measurements and would differ on every single tick.
#
# The audit trail keeps its integrity either way: every PREDICTION is still
# logged, every tick, with its status. Suppression only affects the alerts
# table, which is a human-facing summary, not the record of what was seen.

_last_seen: dict[str, float] = {}
_seen_lock = threading.Lock()


def alert_signature(effective_status: str, hits: list[RuleHit]) -> str:
    """
    What makes two alerts "the same alert" for suppression purposes.

    Trend rules are left out whenever a measured rule is present. A projection
    naturally comes and goes as the fit wobbles near its threshold, and if that
    were part of the key, every flicker would re-log the underlying fault and
    defeat the whole point of suppressing. The row is about the measured
    condition; the projection is extra detail on it.

    When a trend is the ONLY thing that tripped, it becomes the key, because
    otherwise every trend-only alert would collapse into one signature and
    "cooling is degrading" would suppress "load is climbing".
    """
    measured = [h for h in hits if not h.rule_id.startswith("trend_")]
    keyed_on = measured or hits
    rule_ids = sorted({f"{h.rule_id}:{h.severity}" for h in keyed_on})
    return f"{effective_status}|{','.join(rule_ids)}"


def should_log(signature: str, now: float | None = None) -> bool:
    """
    True when this alert is new, or when the same condition has persisted long
    enough to be worth repeating.
    """
    now = time.monotonic() if now is None else now
    with _seen_lock:
        previous = _last_seen.get(signature)
        if previous is not None and (now - previous) < config.ALERT_REPEAT_SECONDS:
            return False
        _last_seen[signature] = now
        return True


def reset_suppression() -> None:
    """Clear the history. Used by tests, and when the simulator restarts."""
    with _seen_lock:
        _last_seen.clear()


def combined_status(model_status: str, hits: list[RuleHit]) -> str:
    """
    The worse of the model status and any tripped rules.

    >>> combined_status("Normal", [])
    'Normal'
    """
    worst = model_status
    for hit in hits:
        if _RANK[hit.severity] > _RANK[worst]:
            worst = hit.severity
    return worst


def build_alert(
    *,
    model_status: str,
    confidence: float,
    features: dict[str, float],
    product_type: str = "M",
    extra_hits: list[RuleHit] | None = None,
) -> tuple[str, dict | None]:
    """
    Work out the effective status and build the alert.

    `extra_hits` carries rules that need more than one reading to evaluate, so
    the trend rules from backend/trends.py. They are passed in rather than
    computed here because this function only ever sees one reading; the caller
    is what holds the history.

    Returns (effective_status, alert dict or None). None means the machine is
    healthy and there is nothing to log.
    """
    hits = evaluate_rules(features, product_type=product_type)

    # The RUL rule sits outside evaluate_rules on purpose. evaluate_rules holds
    # exactly the four rules the dataset's physics defines, and
    # tests/test_thresholds.py checks those row by row against the offline
    # labeller. RUL is a forward-looking rule layered on top, so it is added here
    # rather than mixed into that checked set.
    lifetime_hit = rul_rule(features, product_type=product_type)
    if lifetime_hit is not None:
        hits = hits + [lifetime_hit]

    # Trend hits come from backend/trends.py and are advisory: they say where a
    # channel is heading, not where it is. They are capped at Warning so a
    # straight-line projection can never declare a Fault by itself. Only a
    # measured limit does that.
    if extra_hits:
        hits = hits + [h for h in extra_hits if h.severity != "Fault"]

    hits.sort(key=lambda h: 0 if h.severity == "Fault" else 1)

    effective = combined_status(model_status, hits)

    if effective == "Normal":
        return "Normal", None

    severity = SEVERITY_BY_STATUS[effective]
    rule_dicts = [asdict(h) for h in hits]

    # Prefer the advice from the most urgent tripped rule, since it names the
    # actual component. Fall back to general advice only when the model is
    # raising this on its own.
    urgent = [h for h in hits if h.severity == effective] or hits
    if urgent:
        primary = urgent[0]
        title = primary.title
        action = primary.action
        detail = primary.detail
    else:
        title = f"Model-detected {effective.lower()} condition"
        action = MODEL_ONLY_ACTION[effective]
        detail = "all individual sensor limits within range"

    # If several rules tripped, say so, otherwise someone fixes one thing and
    # assumes they are done.
    extra = ""
    if len(hits) > 1:
        others = ", ".join(h.title for h in hits[1:])
        extra = f" Also tripped: {others}."

    message = (
        f"Machine status {effective} "
        f"(model: {model_status} @ {confidence * 100:.0f}% confidence). "
        f"{detail}.{extra}"
    )

    return effective, {
        "severity": severity,
        "title": title,
        "message": message,
        "recommended_action": action,
        "triggered_rules": rule_dicts,
        # Not part of the API response. Callers pop this and pass it to
        # should_log() to decide whether this alert is worth another row.
        "_signature": alert_signature(effective, hits),
    }
