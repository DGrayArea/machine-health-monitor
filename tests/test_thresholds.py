"""
Consistency check: the offline labeller and the live alerter have to agree.

scripts/clean_data.py evaluates the threshold rules with vectorised pandas,
doing all 10,000 rows at once. backend/thresholds.py evaluates them one reading
at a time, which is what the live API uses. They share the same constants but
they are separate code paths, and separate code paths drift.

If they ever disagreed, the model would be trained on one definition of
"Warning" and deployed to explain a different one. That kind of failure is
silent and horrible to debug, so it gets its own test across the whole dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.thresholds import derive_features, evaluate_rules

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"

# clean_data.py column, and the rule_ids backend/thresholds.py can emit for it
RULE_GROUPS = {
    "warn_cooling": {"cooling"},
    "warn_power": {"power_high", "power_low"},
    "warn_overstrain": {"overstrain"},
    "warn_tool_wear": {"tool_wear"},
}


@pytest.mark.skipif(not CLEAN_CSV.exists(),
                    reason="run scripts/clean_data.py first")
def test_offline_and_online_rules_flag_the_same_rows():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(CLEAN_CSV)

    mismatches = []
    for row in df.itertuples(index=False):
        features = derive_features(
            air_temp=row.air_temp,
            process_temp=row.process_temp,
            rot_speed=row.rot_speed,
            torque=row.torque,
            tool_wear=row.tool_wear,
            product_type=row.type,
        )
        online = {hit.rule_id for hit in evaluate_rules(features,
                                                       product_type=row.type)}

        for column, rule_ids in RULE_GROUPS.items():
            offline_flag = bool(getattr(row, column))
            online_flag = bool(online & rule_ids)
            if offline_flag != online_flag:
                mismatches.append((row.udi, column, offline_flag, online_flag))

    assert not mismatches, (
        f"{len(mismatches)} rows where the offline labeller and the live "
        f"alerter disagree. First 5: {mismatches[:5]}"
    )


@pytest.mark.skipif(not CLEAN_CSV.exists(),
                    reason="run scripts/clean_data.py first")
def test_every_labelled_warning_row_has_at_least_one_rule_hit():
    """A row cannot be labelled Warning without a reason to show the operator."""
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(CLEAN_CSV)
    warnings = df[df["health_status"] == "Warning"]
    assert len(warnings) > 0

    for row in warnings.head(500).itertuples(index=False):
        features = derive_features(
            air_temp=row.air_temp, process_temp=row.process_temp,
            rot_speed=row.rot_speed, torque=row.torque,
            tool_wear=row.tool_wear, product_type=row.type,
        )
        assert evaluate_rules(features, product_type=row.type), \
            f"row {row.udi} labelled Warning but no rule explains why"
