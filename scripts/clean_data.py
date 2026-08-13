"""
Step 2 — Clean the raw data and turn it into Normal / Warning / Fault labels.

WHAT THIS SCRIPT DOES, IN ORDER
    1. Load the raw CSV and rename columns to machine-friendly names.
    2. Drop rows that are exact duplicates of another row.
    3. Handle missing values (numeric -> median, categorical -> mode).
    4. Drop physically impossible readings (negative rpm, absolute zero, ...).
    5. Derive three engineered features that the failure physics depends on:
         temp_diff  = process_temp - air_temp      [K]
         power      = torque * angular velocity    [W]
         strain     = tool_wear * torque           [min*Nm]
    6. Label every row Normal / Warning / Fault using threshold rules.
    7. Write data/processed/machine_health.csv + a cleaning report.

THE LABELLING LOGIC — THIS IS THE PART YOU NEED TO BE ABLE TO DEFEND
    The AI4I 2020 dataset was generated from five documented physical failure
    modes. We do not invent thresholds; we reuse the ones the machine physics
    actually uses, and then define a Warning band *just before* each one:

      Failure mode           FAULT condition (dataset ground truth)
      ---------------------  --------------------------------------------------
      HDF heat dissipation   temp_diff < 8.6 K  AND  rot_speed < 1380 rpm
      PWF power              power < 3500 W  OR  power > 9000 W
      OSF overstrain         strain > 11000 / 12000 / 13000 (quality L / M / H)
      TWF tool wear          tool_wear in 200..240 min (tool breaks in this band)
      RNF random             0.1% chance, unrelated to any sensor

    So a WARNING is "you are inside the last N% of margin before that limit":

      Warning trigger              Rationale
      ---------------------------  ------------------------------------------
      temp_diff < 9.5 K            Cooling is degrading but has not failed.
        AND rot_speed < 1500 rpm   Low rpm means less forced convection.
      power < 4000 or > 8500 W     Drivetrain is near the edge of its envelope.
      strain > 0.85 * OSF limit    Tool + torque combination is overloading.
      tool_wear > 180 min          Tool is inside the last 10% of its life.

    Precedence: Fault beats Warning beats Normal. A row is a Fault if the
    dataset's own `machine_failure` flag is set — we trust the ground truth for
    the positive class rather than re-deriving it, because RNF failures are not
    predictable from the sensors at all.

HONEST CAVEAT (say this out loud in your defence)
    The Warning class is *deterministic* given the sensors — we computed it from
    them. A model will therefore learn the Warning boundary almost perfectly.
    That is not cheating, it is **rule distillation**: the value is that the same
    model also learns the Fault class, which is NOT a pure function of the
    sensors (it contains randomness and a stochastic tool-wear breaking point).
    Fault recall is the number that actually measures learning. See
    scripts/evaluate_model.py, which reports it separately.

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --raw path/to/other.csv --out path/to/out.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `backend` is importable when run as a script

# SINGLE SOURCE OF TRUTH for every threshold. The live API imports the exact
# same numbers from the exact same module, so the labels we train on and the
# alerts we raise in production can never disagree. See backend/thresholds.py.
from backend.thresholds import (  # noqa: E402
    HDF_SPEED_RPM,
    HDF_TEMP_DIFF_K,
    OSF_STRAIN_LIMIT,
    PLAUSIBLE,
    PWF_POWER_MAX_W,
    PWF_POWER_MIN_W,
    TWF_WEAR_MAX_MIN,
    TWF_WEAR_MIN_MIN,
    WARN_POWER_MAX_W,
    WARN_POWER_MIN_W,
    WARN_SPEED_RPM,
    WARN_STRAIN_FRACTION,
    WARN_TEMP_DIFF_K,
    WARN_WEAR_MIN,
)

DEFAULT_RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "machine_health.csv"
DEFAULT_REPORT = ROOT / "outputs" / "metrics" / "cleaning_report.json"

RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rot_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

NUMERIC_COLS = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]
CATEGORICAL_COLS = ["type"]
FAILURE_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]


# --------------------------------------------------------------------------
# Cleaning steps (each one is a small pure-ish function so it can be tested)
# --------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV. `utf-8-sig` strips the byte-order mark on the UDI column."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.rename(columns=RENAME)


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate *measurements*.

    We ignore `udi` and `product_id` when comparing, because those are just row
    counters — two identical sensor readings logged under different IDs are
    still the same measurement and would double-weight the model.
    """
    subset = [c for c in df.columns if c not in ("udi", "product_id")]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return df, before - len(df)


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fill gaps rather than dropping rows, because sensor dropouts are common in
    the field and throwing away the whole row loses the other four channels.

      numeric      -> median  (robust to the outliers we care about detecting)
      categorical  -> mode    (product quality tier is stable per machine)

    Rows missing the *label* column cannot be imputed, so those are dropped.
    """
    report: dict = {"imputed": {}, "dropped_missing_label": 0}

    for col in NUMERIC_COLS:
        if col not in df:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].median())
            report["imputed"][col] = {"count": n_missing, "method": "median"}

    for col in CATEGORICAL_COLS:
        if col not in df:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
            report["imputed"][col] = {"count": n_missing, "method": "mode"}

    if "machine_failure" in df:
        n_bad = int(df["machine_failure"].isna().sum())
        if n_bad:
            df = df.dropna(subset=["machine_failure"]).reset_index(drop=True)
            report["dropped_missing_label"] = n_bad

    return df, report


def drop_implausible(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop readings that violate physics — those are sensor faults, not machine faults."""
    keep = pd.Series(True, index=df.index)
    for col, (lo, hi) in PLAUSIBLE.items():
        if col in df:
            keep &= df[col].between(lo, hi)
    removed = int((~keep).sum())
    return df[keep].reset_index(drop=True), removed


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three derived channels. Every documented failure mode is a function of one
    of these, so giving them to the model directly saves it from having to
    rediscover multiplication.

      temp_diff : how much heat the process is failing to shed          [K]
      power     : torque * omega, with omega = rpm * 2*pi / 60          [W]
      strain    : cumulative mechanical load on the tool           [min*Nm]
    """
    df = df.copy()
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    omega = df["rot_speed"] * 2 * np.pi / 60.0          # rpm -> rad/s
    df["power"] = df["torque"] * omega
    df["strain"] = df["tool_wear"] * df["torque"]
    df["osf_limit"] = df["type"].map(OSF_STRAIN_LIMIT).fillna(OSF_STRAIN_LIMIT["L"])
    return df


def warning_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean column per warning rule. Kept separate from the label so the backend
    can tell an operator *which* rule tripped, not just that something did.
    """
    return pd.DataFrame({
        "warn_cooling": (df["temp_diff"] < WARN_TEMP_DIFF_K)
                        & (df["rot_speed"] < WARN_SPEED_RPM),
        "warn_power": (df["power"] < WARN_POWER_MIN_W)
                      | (df["power"] > WARN_POWER_MAX_W),
        "warn_overstrain": df["strain"] > (WARN_STRAIN_FRACTION * df["osf_limit"]),
        "warn_tool_wear": df["tool_wear"] > WARN_WEAR_MIN,
    })


def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remaining Useful Life, in cutting minutes until the first binding limit.

    This is the regression target for scripts/train_rul_model.py. It mirrors
    backend/rul.physics_rul exactly — vectorised here for 10,000 rows at once —
    and tests/test_rul.py asserts the two implementations agree row by row.

      wear-limited   : 200 - tool_wear
      strain-limited : (osf_limit / torque) - tool_wear
      RUL            : max(0, min(the two))

    See backend/rul.py for why cutting harder LOWERS the ceiling rather than
    just consuming it faster.
    """
    df = df.copy()

    wear_limited = TWF_WEAR_MIN_MIN - df["tool_wear"]

    # Guard the division: torque of 0 means nothing is being cut, so the
    # overstrain limit is unreachable (represented as +inf).
    safe_torque = df["torque"].where(df["torque"] > 1e-6, np.nan)
    strain_limited = (df["osf_limit"] / safe_torque) - df["tool_wear"]
    strain_limited = strain_limited.fillna(np.inf)

    df["rul_minutes"] = np.maximum(0.0, np.minimum(wear_limited, strain_limited))
    df["rul_binding"] = np.where(strain_limited < wear_limited,
                                 "overstrain", "tool_wear")
    return df


def label_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the three-class target.

      Fault   : the dataset's ground-truth failure flag is set.
      Warning : no failure yet, but at least one warning rule tripped.
      Normal  : everything inside limits.
    """
    df = df.copy()
    reasons = warning_reasons(df)
    df = pd.concat([df, reasons], axis=1)

    is_fault = df["machine_failure"].astype(int) == 1
    is_warning = reasons.any(axis=1)

    df["health_status"] = np.select(
        [is_fault, is_warning],
        ["Fault", "Warning"],
        default="Normal",
    )
    return df


def clean(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    """Run the whole pipeline and return (clean dataframe, report dict)."""
    report: dict = {"source": str(raw_path)}

    df = load_raw(raw_path)
    report["rows_raw"] = len(df)

    df, n_dupes = drop_duplicates(df)
    report["duplicates_removed"] = n_dupes

    df, missing_report = handle_missing(df)
    report["missing"] = missing_report

    df, n_implausible = drop_implausible(df)
    report["implausible_removed"] = n_implausible

    df = add_engineered_features(df)
    df = add_rul_target(df)
    df = label_health(df)

    # Sanity check: rows flagged as a failure with no failure mode set are
    # unexplainable (a known quirk of the raw file). Report them, keep them —
    # they are genuine failures, the mode just was not recorded.
    unexplained = int(
        ((df["machine_failure"] == 1) & (df[FAILURE_FLAGS].sum(axis=1) == 0)).sum()
    )
    report["faults_without_recorded_mode"] = unexplained

    report["rows_clean"] = len(df)
    report["class_counts"] = df["health_status"].value_counts().to_dict()
    report["class_percent"] = (
        (df["health_status"].value_counts(normalize=True) * 100).round(2).to_dict()
    )
    report["warning_rule_hits"] = {
        c: int(df[c].sum()) for c in
        ["warn_cooling", "warn_power", "warn_overstrain", "warn_tool_wear"]
    }
    report["rul"] = {
        "mean_minutes": round(float(df["rul_minutes"].mean()), 2),
        "median_minutes": round(float(df["rul_minutes"].median()), 2),
        "already_expired": int((df["rul_minutes"] <= 0).sum()),
        "binding_constraint": df["rul_binding"].value_counts().to_dict(),
    }
    return df, report


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Clean and label machine sensor data.")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    if not args.raw.exists():
        raise SystemExit(
            f"Raw file not found: {args.raw}\nRun: python data/download_data.py"
        )

    df, report = clean(args.raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.report.write_text(json.dumps(report, indent=2))

    print("=" * 62)
    print("CLEANING REPORT")
    print("=" * 62)
    print(f"  raw rows                 : {report['rows_raw']}")
    print(f"  duplicates removed       : {report['duplicates_removed']}")
    print(f"  implausible removed      : {report['implausible_removed']}")
    print(f"  values imputed           : {report['missing']['imputed'] or 'none'}")
    print(f"  clean rows               : {report['rows_clean']}")
    print(f"  faults w/o recorded mode : {report['faults_without_recorded_mode']}")
    print("-" * 62)
    print("  class balance:")
    for label in ("Normal", "Warning", "Fault"):
        n = report["class_counts"].get(label, 0)
        pct = report["class_percent"].get(label, 0.0)
        print(f"    {label:<8} {n:>6}  ({pct:>5.2f}%)")
    print("-" * 62)
    print("  warning rule hits:")
    for rule, n in report["warning_rule_hits"].items():
        print(f"    {rule:<18} {n:>6}")
    print("-" * 62)
    print("  remaining useful life (cutting minutes):")
    print(f"    mean               {report['rul']['mean_minutes']:>6.1f}")
    print(f"    median             {report['rul']['median_minutes']:>6.1f}")
    print(f"    already expired    {report['rul']['already_expired']:>6}")
    print("    binding constraint:")
    for name, n in report["rul"]["binding_constraint"].items():
        print(f"      {name:<16} {n:>6}")
    print("=" * 62)
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print(f"wrote {args.report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
