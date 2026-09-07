"""
How often do the temperature sensors have to be recalibrated?

WHY THIS IS A SEPARATE SCRIPT

    scripts/test_robustness.py found that drift on the temperature channels is
    the most damaging corruption tested: Fault recall falls from 0.838 to about
    0.55. But its grid was coarse — the smallest offset it tried was 1 K, and
    recall had already collapsed there. A curve sampled at 0, 1, 2 and 4 K
    cannot tell you where the failure begins, only that it begins somewhere
    below 1 K.

    That matters, because "recalibrate before the sensor drifts 1 K" and
    "recalibrate before it drifts 0.2 K" are different maintenance contracts
    with different costs. So this sweeps finely enough to site the edge, then
    turns it into an interval.

THE MECHANISM, CONFIRMED BY THE FINE SWEEP

    Heat-dissipation failure is defined on temp_diff = process_temp - air_temp:

        fault   temp_diff < 8.6 K
        warning temp_diff < 9.5 K

    The warning band is the gap between those two lines, and it is 0.9 K wide.
    A drift is a constant offset on temp_diff, so it slides the band past
    itself: readings that should sit in the warning zone land outside it, and
    the model stops seeing the approach to failure.

    The prediction was that damage would happen over 0 to 0.9 K rather than
    across the 0-4 K the coarse grid implied, and the sweep bears that out.
    Recall falls steadily from 0.838 at zero drift to about 0.57 at 0.9 K, then
    flattens at 0.544 and stays there out to 1.5 K. Once drift exceeds the band
    width there is no warning-band information left to destroy, so further drift
    costs nothing extra. The saturation is the mechanism's fingerprint.

    Two things follow. The apparent cliff in test_robustness.py was a sampling
    artefact: its smallest step, 1 K, already sat past the band, so it saw only
    the flat part. And the tolerance does not equal the band width -- it is
    0.15 K, about 17% of it -- because enough borderline readings sit near the
    boundary that a fraction of a kelvin moves a meaningful number across.

    Both channels give near-identical curves, which is the expected symmetry:
    air_temp low and process_temp high shift temp_diff by the same amount.

DIRECTION MATTERS AND ONLY ONE SIDE IS DANGEROUS

    temp_diff = process_temp - air_temp, so

        air_temp reading LOW      inflates temp_diff  -> hides faults
        process_temp reading HIGH inflates temp_diff  -> hides faults

    Drift the other way and temp_diff shrinks, which trips warnings early: an
    unnecessary inspection, not a missed breakdown. Only the dangerous
    direction is swept here, because a calibration policy is written against
    the failure that costs a machine.

WHAT THIS SCRIPT DOES NOT KNOW

    The drift RATE of your actual sensors. That is a datasheet property of the
    specific instrument, so the interval is reported parametrically: tolerance
    divided by rate, tabulated across a range of plausible rates. Plug in the
    figure for the thermocouple actually fitted.

Usage:
    python scripts/recalibration_interval.py
    python scripts/recalibration_interval.py --floor 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.thresholds import HDF_TEMP_DIFF_K, WARN_TEMP_DIFF_K  # noqa: E402
from scripts.test_robustness import (  # noqa: E402
    CLEAN_CSV,
    MODEL_PATH,
    RANDOM_STATE,
    apply_drift,
    score,
)

FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"

# The dangerous direction for each channel, as derived above.
DANGEROUS = {"air_temp": -1.0, "process_temp": +1.0}

# Fine enough to site an edge inside a 0.9 K band.
SWEEP_K = np.round(np.arange(0.0, 1.55, 0.05), 3)

# Plausible drift rates for a Type K thermocouple in service, K/year. Ageing,
# thermoelectric inhomogeneity and junction contamination all contribute, and
# the spread between a well-installed sensor and a poorly installed one is wide.
# These are for the parametric table only — use your instrument's own figure.
DRIFT_RATES = [0.1, 0.25, 0.5, 1.0, 2.0]

BAND_K = WARN_TEMP_DIFF_K - HDF_TEMP_DIFF_K

INK, ACCENT, MARK, MUTED = "#171D21", "#0E6E88", "#B4531C", "#58656B"


def load():
    if not MODEL_PATH.exists() or not CLEAN_CSV.exists():
        raise SystemExit("Run scripts/clean_data.py and scripts/train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(CLEAN_CSV)
    df["type_code"] = df["type"].map(bundle["type_code"]).fillna(0).astype(int)
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["health_status"]
    )
    return bundle["model"], bundle["features"], test_df, test_df["health_status"]


def sweep(model, features, test_df, y_true, rng) -> dict[str, list[tuple[float, float]]]:
    """Fault recall against drift magnitude, dangerous direction only."""
    out: dict[str, list[tuple[float, float]]] = {}
    for channel, sign in DANGEROUS.items():
        series = []
        for magnitude in SWEEP_K:
            corrupted = apply_drift(test_df, channel, sign * magnitude, rng)
            series.append((float(magnitude),
                           score(model, features, corrupted, y_true)["fault_recall"]))
        out[channel] = series
    return out


def tolerance_from(series: list[tuple[float, float]], floor: float) -> float | None:
    """
    Largest drift at which Fault recall is still at or above `floor`.

    Returns the last magnitude before the first breach, not an interpolation.
    The curve is a cliff rather than a slope, so interpolating across it would
    invent a precision the data does not carry.
    """
    last_ok = None
    for magnitude, recall in series:
        if recall >= floor:
            last_ok = magnitude
        else:
            return last_ok
    return last_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floor", type=float, default=0.80,
                    help="Fault recall that must be maintained (default 0.80)")
    args = ap.parse_args()

    model, features, test_df, y_true = load()
    rng = np.random.default_rng(RANDOM_STATE)

    baseline = score(model, features, test_df, y_true)["fault_recall"]
    curves = sweep(model, features, test_df, y_true, rng)

    print("=" * 72)
    print("RECALIBRATION INTERVAL — temperature channels")
    print("=" * 72)
    print(f"  clean Fault recall     {baseline:.4f}")
    print(f"  policy floor           {args.floor:.4f}")
    print(f"  HDF warning band       {WARN_TEMP_DIFF_K} - {HDF_TEMP_DIFF_K} "
          f"= {BAND_K:.1f} K wide\n")

    tolerances = {}
    for channel, series in curves.items():
        sign = "-" if DANGEROUS[channel] < 0 else "+"
        tol = tolerance_from(series, args.floor)
        tolerances[channel] = tol

        print(f"  {channel}  (drift {sign}, the direction that hides faults)")
        for magnitude, recall in series:
            breached = recall < args.floor
            flag = "  <- below floor" if breached else ""
            print(f"    {sign}{magnitude:>4.2f} K   recall {recall:.4f}{flag}")
            if breached:
                break
        print(f"    tolerance: {tol if tol is not None else 0.0:.2f} K"
              f"   ({(tol or 0.0) / BAND_K * 100:.0f}% of the {BAND_K:.1f} K band)\n")

    limiting = min(t for t in tolerances.values() if t is not None) \
        if any(t is not None for t in tolerances.values()) else 0.0

    print("-" * 72)
    print(f"  Limiting tolerance across both channels: {limiting:.2f} K")
    print("-" * 72)
    print("\n  Recalibration interval = tolerance / drift rate\n")
    print(f"  {'drift rate (K/yr)':>20}{'interval':>16}{'in practice':>22}")
    print("  " + "-" * 58)
    for rate in DRIFT_RATES:
        years = limiting / rate if rate > 0 else float("inf")
        months = years * 12
        practice = ("not achievable" if months < 3 else
                    "quarterly" if months < 4.5 else
                    "every 6 months" if months < 9 else
                    "annually" if months < 18 else
                    "every 2 years")
        print(f"  {rate:>20.2f}{years:>13.2f} yr{practice:>22}")

    worst_rate = max(DRIFT_RATES)
    worst_months = (limiting / worst_rate) * 12 if worst_rate else float("inf")
    print("\n" + "=" * 72)
    print("  WHAT THIS MEANS")
    print("=" * 72)
    print(f"  Tolerance is {limiting:.2f} K, only "
          f"{limiting / BAND_K * 100:.0f}% of the {BAND_K:.1f} K warning band.")
    print("  The band sets the scale of the problem, but the collapse begins well")
    print("  inside it: enough borderline readings sit near the boundary that a")
    print("  fraction of a kelvin moves a meaningful number across.\n")
    if worst_months < 3:
        print(f"  At {worst_rate:.1f} K/yr the interval is {worst_months:.1f} months,")
        print("  which is not a maintenance schedule anyone would run. A sensor")
        print("  drifting that fast cannot hold this model's Fault recall at all.\n")
        print("  So the finding is not really 'recalibrate often'. It is that the")
        print("  design is intolerant of temperature drift, and the fix belongs")
        print("  upstream of calibration:")
        print("    - specify a sensor class with a drift spec near 0.1 K/yr, or")
        print("    - widen the warning band so it is not 0.9 K, or")
        print("    - measure temp_diff directly with a differential pair, so")
        print("      common-mode drift cancels instead of accumulating")
        print("\n  The third is the strongest: a differential measurement makes")
        print("  the dangerous failure mode largely self-cancelling, because both")
        print("  junctions age together and the difference is what is read.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5))
    for channel, series in curves.items():
        mags = [m for m, _ in series]
        recs = [r for _, r in series]
        colour = ACCENT if channel == "process_temp" else MARK
        sign = "-" if DANGEROUS[channel] < 0 else "+"
        ax.plot(mags, recs, color=colour, lw=2,
                label=f"{channel}, drift {sign}")

    ax.axhline(baseline, color=INK, ls="--", lw=1, alpha=0.6)
    ax.text(SWEEP_K[-1], baseline + 0.008, "clean baseline", ha="right",
            fontsize=8.5, color=INK)
    ax.axhline(args.floor, color=MUTED, ls=":", lw=1.4)
    ax.text(SWEEP_K[-1], args.floor + 0.008, f"policy floor {args.floor:.2f}",
            ha="right", fontsize=8.5, color=MUTED)

    ax.axvspan(0, BAND_K, color=MUTED, alpha=0.10)
    ax.text(BAND_K / 2, 0.965, f"HDF warning band, {BAND_K:.1f} K wide",
            ha="center", fontsize=8.5, color=MUTED)

    if limiting:
        ax.axvline(limiting, color=MARK, lw=1.4)
        ax.text(limiting + 0.02, 0.92, f"tolerance {limiting:.2f} K",
                fontsize=8.5, color=MARK)

    ax.set_xlabel("drift magnitude in the dangerous direction (K)")
    ax.set_ylabel("Fault recall")
    ax.set_title("Damage accumulates across the warning band, then saturates")
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = FIGURES / "recalibration_interval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    (METRICS / "recalibration_interval.json").write_text(json.dumps({
        "baseline_fault_recall": baseline,
        "policy_floor": args.floor,
        "hdf_band_k": BAND_K,
        "tolerance_k": tolerances,
        "limiting_tolerance_k": limiting,
        "curves": {c: [{"drift_k": m, "fault_recall": r} for m, r in s]
                   for c, s in curves.items()},
        "intervals_years": {str(rate): (limiting / rate if rate else None)
                            for rate in DRIFT_RATES},
    }, indent=2))

    print(f"\n  wrote {path.relative_to(ROOT)}")
    print(f"  wrote outputs/metrics/recalibration_interval.json")


if __name__ == "__main__":
    main()
