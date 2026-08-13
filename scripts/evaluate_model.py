"""
Evaluate the saved model and produce the figures for the write-up.

It rebuilds the same stratified split used in training, using the same
random_state, loads model/health_model.pkl, and writes four figures to
outputs/figures/:

    confusion_matrix.png         where the model is wrong, and in which direction
    actual_vs_predicted.png      actual against predicted over a run of rows,
                                 which is what an operator would have seen live
    feature_importance.png       which channels drive the decisions
    confidence_distribution.png  how sure it is when right against when wrong

Reading the confusion matrix
    Rows are truth, columns are prediction.

    The cell [Fault -> Normal] is the dangerous one, a real failure reported as
    healthy, which in maintenance means a missed breakdown. The cell
    [Normal -> Fault] is only annoying, a false alarm costing an unnecessary
    inspection.

    Those two are not equally bad, which is why recall on the Fault class is
    reported separately rather than folded into overall accuracy. With 96.6% of
    rows non-Fault, a model that answered "Normal" forever would already score
    67%, so accuracy on its own proves nothing here.

Usage:
    python scripts/evaluate_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # render to file, no GUI window needed
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"
MODEL_PATH = ROOT / "model" / "health_model.pkl"

# Everything generated lives under outputs/, never mixed in with source.
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"

CLASS_ORDER = ["Normal", "Warning", "Fault"]
COLOURS = {"Normal": "#2e9e5b", "Warning": "#e0a800", "Fault": "#d9463f"}
RANDOM_STATE = 42  # must match train_model.py, or the test set leaks


def load() -> tuple[dict, pd.DataFrame]:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}\nRun: python scripts/train_model.py")
    if not CLEAN_CSV.exists():
        raise SystemExit(f"Missing {CLEAN_CSV}\nRun: python scripts/clean_data.py")
    return joblib.load(MODEL_PATH), pd.read_csv(CLEAN_CSV)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Predicted status")
    ax.set_ylabel("Actual status")
    ax.set_title("Confusion matrix — held-out test set")

    # Label each cell with the count and the row percentage.
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_totals, where=row_totals != 0) * 100
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%",
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, path: Path, n: int = 200) -> None:
    """
    Actual against predicted as a step trace, the way it would look on the live
    dashboard. Mismatches get a red X so they can be counted by eye.
    """
    idx = np.arange(n)
    to_num = {c: i for i, c in enumerate(CLASS_ORDER)}
    ta = [to_num[v] for v in y_true[:n]]
    tp = [to_num[v] for v in y_pred[:n]]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.step(idx, ta, where="mid", label="Actual", linewidth=2.4,
            color="#3a3f4b", alpha=0.85)
    ax.step(idx, tp, where="mid", label="Predicted", linewidth=1.4,
            color="#1f77b4", linestyle="--")

    mism = [i for i in idx if ta[i] != tp[i]]
    if mism:
        ax.scatter(mism, [tp[i] for i in mism], marker="x", s=90,
                   color="#d9463f", zorder=5,
                   label=f"Mismatch ({len(mism)}/{n})")

    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel(f"Test sample index (first {n} rows)")
    ax.set_title("Actual vs predicted machine health status")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(bundle: dict, path: Path) -> None:
    imp = (
        pd.Series(bundle["model"].feature_importances_, index=bundle["features"])
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(imp.index, imp.values, color="#4c78a8")
    ax.set_xlabel("Relative importance (Gini decrease)")
    ax.set_title("Which sensor channels drive the prediction")
    for y, v in enumerate(imp.values):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence(proba: np.ndarray, y_true, y_pred, path: Path) -> None:
    """
    Confidence is the winning class probability. A model worth trusting should
    be noticeably less confident on the rows it gets wrong. If the two
    histograms overlap completely then the confidence score is decorative and
    does not belong on the dashboard.
    """
    conf = proba.max(axis=1)
    correct = np.array(y_true) == np.array(y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0.3, 1.0, 29)
    ax.hist(conf[correct], bins=bins, alpha=0.75, label=f"Correct (n={correct.sum()})",
            color="#2e9e5b")
    ax.hist(conf[~correct], bins=bins, alpha=0.85,
            label=f"Incorrect (n={(~correct).sum()})", color="#d9463f")
    ax.set_yscale("log")  # errors are rare, so log scale keeps them visible
    ax.set_xlabel("Model confidence (max class probability)")
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("Is the confidence score meaningful?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    bundle, df = load()
    model, features = bundle["model"], bundle["features"]

    X = df.copy()
    X["type_code"] = X["type"].map(bundle["type_code"]).fillna(0).astype(int)
    X = X[features]
    y = df["health_status"]

    # The same split as training, so these rows were never seen by the model.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    print("=" * 62)
    print("EVALUATION — held-out test set")
    print("=" * 62)
    print(f"  rows            : {len(X_test)}")
    print(f"  overall accuracy: {acc:.4f}")
    print(f"  model trained at: {bundle.get('trained_at', 'unknown')}")
    print("\n" + classification_report(y_test, y_pred, digits=3, zero_division=0))

    print("Confusion matrix (rows = actual, cols = predicted):")
    header = " " * 12 + "".join(f"{c:>10}" for c in CLASS_ORDER)
    print(header)
    for i, actual in enumerate(CLASS_ORDER):
        print(f"  {actual:<10}" + "".join(f"{cm[i, j]:>10}" for j in range(len(CLASS_ORDER))))

    # Spell out the two errors that matter in practice.
    fi, ni = CLASS_ORDER.index("Fault"), CLASS_ORDER.index("Normal")
    missed = int(cm[fi, ni])
    false_alarm = int(cm[ni, fi])
    total_faults = int(cm[fi].sum())
    print("\nOperational reading:")
    print(f"  MISSED FAULTS (Fault seen as Normal): {missed} of {total_faults} "
          f"real faults  -> {missed / max(total_faults, 1) * 100:.1f}% missed")
    print(f"  FALSE ALARMS  (Normal seen as Fault): {false_alarm}")
    print("  A missed fault costs a breakdown; a false alarm costs an inspection.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, FIGURES / "confusion_matrix.png")
    plot_actual_vs_predicted(list(y_test), list(y_pred), FIGURES / "actual_vs_predicted.png")
    plot_feature_importance(bundle, FIGURES / "feature_importance.png")
    plot_confidence(proba, list(y_test), list(y_pred), FIGURES / "confidence_distribution.png")

    (METRICS / "evaluation.json").write_text(json.dumps({
        "accuracy": acc,
        "confusion_matrix": {"labels": CLASS_ORDER, "matrix": cm.tolist()},
        "report": classification_report(y_test, y_pred, output_dict=True,
                                        zero_division=0),
        "missed_faults": missed,
        "false_alarms": false_alarm,
    }, indent=2))

    print("\nGenerated artefacts:")
    for name in ("confusion_matrix.png", "actual_vs_predicted.png",
                 "feature_importance.png", "confidence_distribution.png"):
        print(f"  outputs/figures/{name}")
    print("  outputs/metrics/evaluation.json")


if __name__ == "__main__":
    main()
