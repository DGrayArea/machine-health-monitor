"""
Train the remaining useful life regressor.

    In:  the same nine features as the classifier
    Out: remaining cutting minutes, a number rather than a class, plus a band

Why a regressor and not a classifier
    "Is the machine healthy" is a classification question with three answers.
    "How long have I got" is a regression question with a continuous answer, and
    the difference matters in practice. A planner cannot schedule a tool change
    around the word "Warning", but can around "31 minutes".

    Read backend/rul.py first for why this is model-based prognostics rather
    than the usual data-driven kind. The short version is that AI4I 2020 has no
    run-to-failure trajectories, so a learned degradation curve would be fiction.

What this model is, and what we found
    The target comes from the failure physics, so it is a fixed function of
    tool_wear, torque and quality tier. A Random Forest reproduces it almost
    exactly, at MAE 0.38 min and R^2 0.999. That is not an achievement. It shows
    the forest can learn a division and nothing else.

    It was trained anyway to answer two questions, and both answers are worth
    presenting, including the one that came out negative.

      1. Does the learned model beat the formula? No.
         It matches the formula to within half a minute, and the spread across
         its 300 trees is near zero because there is no noise in the target for
         them to disagree about. An uncertainty band only means something when
         the trees actually disagree. So the system uses the physics formula in
         backend/rul.py and keeps this model as a cross-check rather than the
         source of truth. Deleting it would cost almost nothing.

         The one place the spread is not zero is the boundary where the binding
         constraint switches from tool wear to overstrain. The script reports
         sigma separately there, since that is the only region where the model
         carries anything the formula does not already state.

      2. What happens when the tool-wear counter fails? This one is useful. A
         second model is trained without tool_wear, standing in for a reset
         counter or a dead encoder. Its error shows how much life can still be
         inferred from temperature, speed and torque alone, and the answer is
         almost none. That is a real design finding: the tool-wear channel needs
         redundancy before this system could be trusted unattended.

Usage:
    python scripts/train_rul_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import FEATURES, TYPE_CODE  # noqa: E402

CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"
MODEL_PATH = ROOT / "model" / "rul_model.pkl"
META_PATH = ROOT / "model" / "rul_metadata.json"
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"

RANDOM_STATE = 42
TARGET = "rul_minutes"

# Rows where the tool is already past a limit carry RUL = 0. They are real, but
# a long flat run of zeros lets a regressor look good for the wrong reason, so
# scores on the rows still in service are reported separately.
LIVE_ONLY_NOTE = "rows with RUL > 0, i.e. tools still in service"


def build_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["type_code"] = out["type"].map(TYPE_CODE).fillna(0).astype(int)
    return out[columns]


def fit_and_score(name: str, columns: list[str], df: pd.DataFrame) -> tuple:
    X = build_features(df, columns)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    # Same three numbers on tools still in service only.
    live = y_test > 0
    live_mae = mean_absolute_error(y_test[live], pred[live]) if live.any() else float("nan")

    print(f"{name}")
    print(f"  features   : {len(columns)} ({', '.join(columns)})")
    print(f"  MAE        : {mae:6.2f} min   <- typical error, in minutes")
    print(f"  RMSE       : {rmse:6.2f} min   <- punishes large misses")
    print(f"  R^2        : {r2:6.4f}")
    print(f"  MAE ({LIVE_ONLY_NOTE[:22]}): {live_mae:6.2f} min")
    print(f"  fit time   : {fit_seconds:.2f}s\n")

    scores = {"mae": mae, "rmse": rmse, "r2": r2, "mae_live_only": live_mae}
    return model, X_test, y_test, pred, scores


def predict_with_uncertainty(model, X) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean and standard deviation across the individual trees.

    Each tree saw a different bootstrap sample, so how much they disagree says
    how well the training data pins down that region of the input space. This is
    model uncertainty only. It does not capture sensor noise, and it will not
    warn you about an operating point outside the training data altogether.
    """
    # Trees inside a forest are fitted on the raw array, not the named frame.
    raw = X.to_numpy() if hasattr(X, "to_numpy") else X
    per_tree = np.stack([tree.predict(raw) for tree in model.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def plot_predicted_vs_actual(y_true, y_pred, std, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color="#4c78a8",
               label="test rows")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="#d9463f", linewidth=1.6,
            linestyle="--", label="perfect prediction")
    ax.set_xlabel("Actual remaining life (min)")
    ax.set_ylabel("Predicted remaining life (min)")
    ax.set_title("RUL — predicted vs actual")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_distribution(y_true, y_pred, path: Path) -> None:
    errors = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(errors, bins=60, color="#4c78a8")
    ax.axvline(0, color="#d9463f", linewidth=1.4)
    ax.set_xlabel("Prediction error (min).  Negative = predicted too little life "
                  "(safe).  Positive = predicted too much life (dangerous).")
    ax.set_ylabel("Test rows")
    ax.set_title("RUL error distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty_band(y_true, y_pred, std, path: Path, n: int = 120) -> None:
    """Show the +/-1.96 sigma band around the prediction for a slice of rows."""
    order = np.argsort(np.asarray(y_true))[:: max(1, len(y_true) // n)][:n]
    actual = np.asarray(y_true)[order]
    pred = np.asarray(y_pred)[order]
    sigma = np.asarray(std)[order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.fill_between(x, pred - 1.96 * sigma, pred + 1.96 * sigma,
                    color="#4c78a8", alpha=0.25, label="±1.96σ across trees")
    ax.plot(x, pred, color="#4c78a8", linewidth=1.6, label="Predicted")
    ax.plot(x, actual, color="#3a3f4b", linewidth=1.8, linestyle="--",
            label="Actual")
    ax.set_xlabel("Test rows, sorted by actual remaining life")
    ax.set_ylabel("Remaining life (min)")
    ax.set_title("RUL prediction with model uncertainty")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit(f"Missing {CLEAN_CSV}\nRun: python scripts/clean_data.py")

    df = pd.read_csv(CLEAN_CSV)
    print(f"rows: {len(df)}   target: {TARGET}   "
          f"range: {df[TARGET].min():.1f}–{df[TARGET].max():.1f} min\n")
    print("=" * 66)

    # --- Main model: every channel we measure ----------------------------
    model, X_test, y_test, pred, scores = fit_and_score(
        "Full model (all sensors)", FEATURES, df
    )

    # --- Fallback: what if the tool-wear counter cannot be trusted? -------
    no_wear = [f for f in FEATURES if f not in ("tool_wear", "strain")]
    _, _, _, _, fallback_scores = fit_and_score(
        "Fallback model (no tool_wear, no strain)", no_wear, df
    )
    print("=" * 66)

    degradation = fallback_scores["mae"] / max(scores["mae"], 1e-9)
    print(f"\nRemoving the tool-wear channel multiplies the typical error by "
          f"{degradation:.0f}x\n({scores['mae']:.2f} -> {fallback_scores['mae']:.2f} "
          f"minutes). The tool-wear counter is load-bearing: if it fails, RUL "
          f"must fall back\nto the physics formula, not to this model.")

    # --- Uncertainty -----------------------------------------------------
    mean_pred, std_pred = predict_with_uncertainty(model, X_test)
    within = float(np.mean(
        np.abs(np.asarray(y_test) - mean_pred) <= 1.96 * np.maximum(std_pred, 1e-9)
    ))

    # Split sigma by which constraint binds. The trees only disagree near the
    # switch-over, so one overall median hides the interesting region.
    test_rows = df.loc[X_test.index]
    overstrain_bound = (test_rows["rul_binding"] == "overstrain").to_numpy()

    print(f"\nUncertainty band: {within * 100:.1f}% of actual values fall inside "
          f"±1.96σ of the tree spread.")
    print(f"  median σ, all test rows        : {np.median(std_pred):6.2f} min")
    if overstrain_bound.any():
        print(f"  median σ, tool-wear bound      : "
              f"{np.median(std_pred[~overstrain_bound]):6.2f} min")
        print(f"  median σ, OVERSTRAIN bound     : "
              f"{np.median(std_pred[overstrain_bound]):6.2f} min   "
              f"<- the only region where the trees disagree")
    print("  A near-zero spread means the model has learned a formula, not a "
          "distribution.\n  Production therefore uses backend/rul.py directly; "
          "see this script's docstring.")

    print("\nFeature importance:")
    importance = (pd.Series(model.feature_importances_, index=FEATURES)
                  .sort_values(ascending=False))
    for feature, value in importance.items():
        print(f"  {feature:<14} {value:>6.3f}  {'#' * int(round(value * 60))}")

    # --- Save ------------------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "type_code": TYPE_CODE,
        "target": TARGET,
        "target_unit": "cutting minutes",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, MODEL_PATH)

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_type": "RandomForestRegressor",
        "target": TARGET,
        "features": FEATURES,
        "scores": {"full": scores, "without_tool_wear": fallback_scores},
        "uncertainty": {
            "coverage_1p96_sigma": within,
            "median_sigma_min": float(np.median(std_pred)),
        },
        "feature_importance": importance.round(4).to_dict(),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    (METRICS / "rul_evaluation.json").write_text(json.dumps(metadata, indent=2))

    plot_predicted_vs_actual(y_test, pred, std_pred, FIGURES / "rul_predicted_vs_actual.png")
    plot_error_distribution(y_test, pred, FIGURES / "rul_error_distribution.png")
    plot_uncertainty_band(y_test, mean_pred, std_pred, FIGURES / "rul_uncertainty.png")

    print(f"\nwrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"wrote {META_PATH.relative_to(ROOT)}")
    for name in ("rul_predicted_vs_actual.png", "rul_error_distribution.png",
                 "rul_uncertainty.png"):
        print(f"wrote outputs/figures/{name}")


if __name__ == "__main__":
    main()
