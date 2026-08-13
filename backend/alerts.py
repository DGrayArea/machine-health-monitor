"""
Step 5 — Turn a prediction into an actionable alert.

THE CORE DESIGN DECISION IN THIS FILE
    An alert is NOT just "whatever the model said". It combines two independent
    sources of evidence:

      (a) the ML model      — learned, probabilistic, can spot combinations
                              of readings that no single threshold catches
      (b) the physical rules — deterministic, from backend/thresholds.py

    The combination rule is: **the rules can escalate the model, but the model
    can never suppress the rules.** Effective severity = the worst of the two.

    Why: a hard physical limit (power above 9 kW, tool past its life) is a fact,
    not a prediction. If the model has a bad day and says "Normal" while the
    spindle is drawing 9.5 kW, the operator must still be told. Machine safety
    interlocks work the same way — the learned layer can add sensitivity, but it
    is never allowed to override a hard limit. This is the single most important
    thing to be able to explain about this project.

    The reverse direction is allowed and useful: if every threshold is inside
    limits but the model has learned that *this particular combination* of
    readings precedes failures, it raises a Warning on its own. That is the
    genuinely predictive part — thresholds alone cannot do it.

SEVERITY LADDER
    Normal  -> Info      no alert raised, nothing logged as an alert
    Warning -> Warning   plan a fix; the machine keeps running
    Fault   -> Critical  act now; the machine should stop
"""

from __future__ import annotations

from dataclasses import asdict

from backend.rul import rul_rule
from backend.thresholds import RuleHit, evaluate_rules

# Model status -> alert severity
SEVERITY_BY_STATUS = {"Normal": "Info", "Warning": "Warning", "Fault": "Critical"}

# Ranking used to take "the worst of" two opinions.
_RANK = {"Normal": 0, "Warning": 1, "Fault": 2}

# Fallback advice when the model raises a status but no hard threshold tripped —
# i.e. the genuinely predictive case, where there is no single limit to point at.
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


def combined_status(model_status: str, hits: list[RuleHit]) -> str:
    """
    Worst-of the model status and the tripped physical rules.

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
) -> tuple[str, dict | None]:
    """
    Decide the effective status and build the alert payload.

    Returns (effective_status, alert_dict_or_None).
    `None` means the machine is healthy and nothing needs to be logged.
    """
    hits = evaluate_rules(features, product_type=product_type)

    # The RUL rule lives outside evaluate_rules on purpose. evaluate_rules holds
    # exactly the four rules the dataset's own physics defines, and
    # tests/test_thresholds.py checks it row-for-row against the offline
    # labeller. RUL is a derived, forward-looking rule layered on top, so it is
    # appended here rather than smuggled into that checked set.
    lifetime_hit = rul_rule(features, product_type=product_type)
    if lifetime_hit is not None:
        hits = hits + [lifetime_hit]
        hits.sort(key=lambda h: 0 if h.severity == "Fault" else 1)

    effective = combined_status(model_status, hits)

    if effective == "Normal":
        return "Normal", None

    severity = SEVERITY_BY_STATUS[effective]
    rule_dicts = [asdict(h) for h in hits]

    # Prefer the advice attached to the most urgent tripped rule — it names the
    # actual physical component. Only fall back to generic advice when the model
    # is raising this on its own.
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

    # If several rules tripped, tell the operator so they do not fix one thing
    # and declare victory.
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
    }
