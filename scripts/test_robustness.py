"""
How well does the model hold up when the sensors are not perfect?

The problem this addresses
    AI4I 2020 is synthetic. Its sensor channels are clean in a way no real
    instrument ever is: no noise, no calibration drift, no quantisation, no
    dropouts. A model trained and tested on clean data will score well on clean
    data, and that score says nothing about a real workshop.

    "The dataset is synthetic" is a fair caveat but a lazy one on its own. This
    script turns it into a measurement: corrupt the test set in the specific
    ways real instruments fail, re-score, and report how much accuracy is lost.
    A number beats an apology.

The four corruptions, and why these four
    noise         Gaussian, added per reading. Every sensor has a noise floor.
    drift         A constant offset on one channel, which is what an
                  uncalibrated or ageing sensor does. Unlike noise this does not
                  average out, so it is usually the more damaging of the two.
    quantisation  Rounding to a coarse step, as a low-resolution ADC would.
                  A 10-bit converter over a 100 °C span gives about 0.1 °C steps.
    dropout       A channel freezes and repeats its last good value, which is
                  what a dead bus or a stuck sensor looks like downstream. It is
                  the nastiest because nothing about the value looks wrong.

Choosing the magnitudes
    The levels come from typical instrument specifications, not from whatever
    made the results look good:

      Type K thermocouple      +/- 1.5 K standard tolerance, so 0.5 K is a good
                               sensor, 2.0 K a poor or drifting one
      Rotary torque sensor     0.1 to 0.5% of full scale; at a 200 N·m range
                               that is 0.2 to 1.0 N·m
      Incremental encoder      speed error well under 0.1%, so ~1 rpm; encoders
                               are the most reliable channel here
      Tool wear counter        not a sensor at all, it is a counter, so it does
                               not get noise; it gets a dropout test instead,
                               since the realistic failure is that it stops

    Every level is applied to the TEST set only. The model is never retrained,
    because the question is how the deployed model behaves when its inputs go
    bad, not whether a model could be trained to cope.

What to look at
    Fault recall, not accuracy. Accuracy is dominated by the Normal class and
    barely moves. Fault recall is the fraction of real failures still caught,
    and it is what degrades first.

Usage:
    python scripts/test_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"
MODEL_PATH = ROOT / "model" / "health_model.pkl"
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"

RANDOM_STATE = 42
RAW_CHANNELS = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]

# Noise and drift levels per channel, in that channel's own units. See the
# module docstring for where these come from.
NOISE_LEVELS = {
    "air_temp": [0.0, 0.25, 0.5, 1.0, 2.0],           # K
    "process_temp": [0.0, 0.25, 0.5, 1.0, 2.0],       # K
    "rot_speed": [0.0, 1.0, 5.0, 15.0, 40.0],         # rpm
    "torque": [0.0, 0.2, 0.5, 1.0, 2.0],              # N·m
}
# Drift is tested in BOTH directions on purpose. A sensor can read high or low,
# and for the temperature channels the two are not equivalent: temp_diff is
# process_temp minus air_temp, so the same drift on each pushes the derived
# feature opposite ways. One direction raises false alarms, the other hides real
# faults, and only the second is dangerous. Testing one sign would miss that.
DRIFT_LEVELS = {
    "air_temp": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
    "process_temp": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
    "rot_speed": [-80.0, -40.0, -15.0, 0.0, 15.0, 40.0, 80.0],
    "torque": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
}
QUANTISATION_STEPS = {
    "air_temp": [0.0, 0.1, 0.5, 1.0],
    "process_temp": [0.0, 0.1, 0.5, 1.0],
    "rot_speed": [0.0, 1.0, 10.0, 50.0],
    "torque": [0.0, 0.1, 1.0, 5.0],
}
DROPOUT_RATES = [0.0, 0.01, 0.05, 0.10, 0.25]


def rebuild_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute temp_diff, power and strain after corrupting the raw channels.

    This matters more than it looks. The derived features are where most of the
    model's signal lives, and an error on torque propagates into both power and
    strain. Corrupting only the raw columns and leaving the derived ones intact
    would quietly understate the damage.
    """
    out = df.copy()
    out["temp_diff"] = out["process_temp"] - out["air_temp"]
    omega = out["rot_speed"] * 2 * np.pi / 60.0
    out["power"] = out["torque"] * omega
    out["strain"] = out["tool_wear"] * out["torque"]
    return out


def apply_noise(df, channel, sigma, rng):
    out = df.copy()
    out[channel] = out[channel] + rng.normal(0.0, sigma, len(out))
    return rebuild_derived(out)


def apply_drift(df, channel, offset, rng):
    out = df.copy()
    out[channel] = out[channel] + offset
    return rebuild_derived(out)


def apply_quantisation(df, channel, step, rng):
    out = df.copy()
    if step > 0:
        out[channel] = np.round(out[channel] / step) * step
    return rebuild_derived(out)


def apply_dropout(df, channel, rate, rng):
    """Freeze the channel: a dropped reading repeats the previous good value."""
    out = df.copy()
    values = out[channel].to_numpy(copy=True)
    dropped = rng.random(len(values)) < rate
    for i in range(1, len(values)):
        if dropped[i]:
            values[i] = values[i - 1]
    out[channel] = values
    return rebuild_derived(out)


def score(model, features, df, y_true) -> dict:
    pred = model.predict(df[features])
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "fault_recall": recall_score(y_true, pred, labels=["Fault"],
                                     average="macro", zero_division=0),
        "warning_recall": recall_score(y_true, pred, labels=["Warning"],
                                       average="macro", zero_division=0),
    }


def main() -> None:
    if not MODEL_PATH.exists() or not CLEAN_CSV.exists():
        raise SystemExit("Run scripts/clean_data.py and scripts/train_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    df = pd.read_csv(CLEAN_CSV)
    df["type_code"] = df["type"].map(bundle["type_code"]).fillna(0).astype(int)

    # The same split as training, so this is the untouched test set.
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["health_status"]
    )
    y_true = test_df["health_status"]
    rng = np.random.default_rng(RANDOM_STATE)

    baseline = score(model, features, test_df, y_true)
    print("=" * 74)
    print("ROBUSTNESS — clean baseline on the held-out test set")
    print("=" * 74)
    print(f"  accuracy {baseline['accuracy']:.4f}   "
          f"macro F1 {baseline['macro_f1']:.4f}   "
          f"Fault recall {baseline['fault_recall']:.4f}")

    results: dict[str, list] = {"baseline": baseline, "runs": []}

    def run(kind, channel, level, corrupt):
        corrupted = corrupt(test_df, channel, level, rng)
        s = score(model, features, corrupted, y_true)
        drop = baseline["fault_recall"] - s["fault_recall"]
        results["runs"].append({
            "kind": kind, "channel": channel, "level": level,
            **s, "fault_recall_drop": drop,
        })
        return s, drop

    for kind, levels, corrupt, unit in (
        ("noise", NOISE_LEVELS, apply_noise, "sigma"),
        ("drift", DRIFT_LEVELS, apply_drift, "offset"),
        ("quantisation", QUANTISATION_STEPS, apply_quantisation, "step"),
    ):
        print("\n" + "-" * 74)
        print(f"{kind.upper()}  (Fault recall, and change from baseline)")
        print("-" * 74)
        for channel, values in levels.items():
            cells = []
            for level in values:
                s, drop = run(kind, channel, level, corrupt)
                cells.append(f"{level:>5g}:{s['fault_recall']:.3f}({drop:+.3f})")
            print(f"  {channel:<14} " + "  ".join(cells))
        print(f"  {'':14} levels are {unit} in each channel's own units")

    print("\n" + "-" * 74)
    print("DROPOUT  (channel freezes and repeats its last value)")
    print("-" * 74)
    for channel in RAW_CHANNELS:
        cells = []
        for rate in DROPOUT_RATES:
            s, drop = run("dropout", channel, rate, apply_dropout)
            cells.append(f"{rate:>5.0%}:{s['fault_recall']:.3f}({drop:+.3f})")
        print(f"  {channel:<14} " + "  ".join(cells))

    # --- Everything at once: a plausible "poor instrumentation" case ---
    print("\n" + "=" * 74)
    print("COMBINED — every channel degraded together")
    print("=" * 74)
    for label, scale in (("good sensors", 1.0), ("poor sensors", 4.0)):
        corrupted = test_df.copy()
        for channel in ("air_temp", "process_temp", "rot_speed", "torque"):
            sigma = NOISE_LEVELS[channel][1] * scale
            corrupted[channel] = corrupted[channel] + rng.normal(0, sigma, len(corrupted))
        corrupted = rebuild_derived(corrupted)
        s = score(model, features, corrupted, y_true)
        drop = baseline["fault_recall"] - s["fault_recall"]
        results["runs"].append({"kind": "combined", "channel": "all",
                                "level": scale, **s, "fault_recall_drop": drop})
        print(f"  {label:<14} accuracy {s['accuracy']:.4f}   "
              f"macro F1 {s['macro_f1']:.4f}   "
              f"Fault recall {s['fault_recall']:.4f} ({drop:+.3f})")

    # --- Figure ---
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, kind, levels in (
        (axes[0], "noise", NOISE_LEVELS),
        (axes[1], "drift", DRIFT_LEVELS),
        (axes[2], "dropout", {c: DROPOUT_RATES for c in RAW_CHANNELS}),
    ):
        for channel in levels:
            runs = [r for r in results["runs"]
                    if r["kind"] == kind and r["channel"] == channel]
            if not runs:
                continue
            runs = sorted(runs, key=lambda r: r["level"])
            xs = [r["level"] for r in runs]
            ys = [r["fault_recall"] for r in runs]
            # Channels use different units, so normalise x by the largest level
            # tested. Signed levels keep their sign, which is the point of the
            # drift panel.
            span = max(abs(x) for x in xs) or 1.0
            ax.plot([x / span for x in xs], ys, marker="o", label=channel)
        ax.axhline(baseline["fault_recall"], color="#888", linestyle="--",
                   linewidth=1, label="clean baseline")
        ax.set_title(kind)
        ax.set_xlabel("corruption level (fraction of the maximum tested)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fault recall")
    axes[2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Model robustness to sensor degradation (test set, no retraining)")
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness.png", dpi=150)
    plt.close(fig)

    (METRICS / "robustness.json").write_text(json.dumps(results, indent=2))

    # --- The finding, stated in one place ---
    worst = max(results["runs"], key=lambda r: r["fault_recall_drop"])
    print("\n" + "=" * 74)
    print("FINDINGS")
    print("=" * 74)
    print("  Most damaging single corruption tested:")
    print(f"    {worst['kind']} on {worst['channel']} at level {worst['level']:g}")
    print(f"    Fault recall {baseline['fault_recall']:.3f} -> "
          f"{worst['fault_recall']:.3f}  ({worst['fault_recall_drop']:+.3f})")

    # Drift direction matters, and in opposite ways for the two temperature
    # channels. Spell it out rather than leaving it in the table.
    print("\n  Drift direction, temperature channels:")
    for channel in ("air_temp", "process_temp"):
        runs = sorted((r for r in results["runs"]
                       if r["kind"] == "drift" and r["channel"] == channel),
                      key=lambda r: r["level"])
        low = min(runs, key=lambda r: r["fault_recall"])
        print(f"    {channel:<13} worst at {low['level']:+g} K -> "
              f"Fault recall {low['fault_recall']:.3f} "
              f"({low['fault_recall_drop']:+.3f})")
    print("    temp_diff = process_temp - air_temp, so a sensor reading HIGH on")
    print("    process_temp (or LOW on air_temp) inflates the apparent cooling")
    print("    margin and hides heat-dissipation faults. The opposite sign only")
    print("    causes false alarms, which cost an inspection rather than a")
    print("    machine. Calibration on process_temp is the one that matters.")
    print("\n  outputs/figures/robustness.png")
    print("  outputs/metrics/robustness.json")


if __name__ == "__main__":
    main()
