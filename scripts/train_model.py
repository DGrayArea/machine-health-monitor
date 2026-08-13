"""
Step 3 — Train the health classifier.

WHAT IT PREDICTS
    Input : the 5 raw sensor channels + product quality tier
    Output: Normal / Warning / Fault  +  a confidence (class probability)

FEATURES WE GIVE THE MODEL
    Raw       : type_code, air_temp, process_temp, rot_speed, torque, tool_wear
    Engineered: temp_diff, power, strain

    The engineered three are the ones the failure physics is actually written
    in. A tree can only split on one variable at a time, so it would need a
    deep, brittle staircase of splits to approximate `torque * rpm`. Handing it
    `power` directly turns that staircase into a single clean split. This is the
    single biggest accuracy lever in the whole project.

    `type_code` is ordinal (L=0, M=1, H=2) rather than one-hot, because product
    quality really is ordered — the overstrain limit rises monotonically with it.

WHY TWO MODELS
    We train a Logistic Regression *and* a Random Forest and print both.
    - Logistic Regression draws one straight line per class. The failure rules
      are conjunctions ("temp_diff low AND rpm low") and two-sided bands
      ("power too low OR too high"), which no single straight line can express.
    - A Random Forest is a vote over many axis-aligned decision trees, which is
      exactly the shape of a threshold rule.
    Expect the forest to win. Being able to explain *why* it wins is the point;
    the comparison is the argument, not decoration.

HOW WE SPLIT
    Stratified 80/20 split — Fault is only ~3.4% of rows, so a random split
    could easily hand the test set an unrepresentative number of them.
    `class_weight="balanced"` makes each Fault row count ~20x a Normal row
    during training, so the model cannot get a good score by ignoring faults.

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "processed" / "machine_health.csv"
MODEL_PATH = ROOT / "model" / "health_model.pkl"
META_PATH = ROOT / "model" / "metadata.json"

# The exact feature order the backend must reproduce at inference time.
# It is saved inside the .pkl so the two can never silently drift apart.
FEATURES = [
    "type_code",
    "air_temp",
    "process_temp",
    "rot_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "power",
    "strain",
]

TYPE_CODE = {"L": 0, "M": 1, "H": 2}
CLASS_ORDER = ["Normal", "Warning", "Fault"]
RANDOM_STATE = 42


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select/derive exactly the columns in FEATURES, in that order."""
    out = df.copy()
    out["type_code"] = out["type"].map(TYPE_CODE).fillna(0).astype(int)
    return out[FEATURES]


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit(
            f"Missing {CLEAN_CSV}\nRun: python scripts/clean_data.py"
        )

    df = pd.read_csv(CLEAN_CSV)
    X = build_feature_frame(df)
    y = df["health_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,               # keep the 3.4% Fault rate in both halves
    )

    print(f"train rows: {len(X_train):>6}   test rows: {len(X_test):>6}")
    print(f"train class balance: {y_train.value_counts().to_dict()}\n")

    # ---------------- Baseline: Logistic Regression ----------------
    # Scaling matters here: power is ~10^3 W while temp_diff is ~10^1 K, and a
    # linear model's coefficients would otherwise be dominated by unit choice.
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    t0 = time.perf_counter()
    logreg.fit(X_train, y_train)
    logreg_time = time.perf_counter() - t0
    logreg_pred = logreg.predict(X_test)

    # ---------------- Main model: Random Forest ----------------
    rf = RandomForestClassifier(
        n_estimators=300,         # enough trees for stable probabilities
        max_depth=None,           # thresholds are shallow; let it fit them exactly
        min_samples_leaf=2,       # small guard against memorising single rows
        class_weight="balanced",  # 3.4% Fault must not be ignorable
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0
    rf_pred = rf.predict(X_test)

    # ---------------- Compare ----------------
    def summarise(name: str, pred, fit_seconds: float) -> dict:
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        fault_f1 = f1_score(y_test, pred, labels=["Fault"], average="macro",
                            zero_division=0)
        print(f"{name}")
        print(f"  accuracy   : {acc:.4f}")
        print(f"  macro F1   : {macro_f1:.4f}   <- treats all 3 classes equally")
        print(f"  Fault F1   : {fault_f1:.4f}   <- the number that actually matters")
        print(f"  fit time   : {fit_seconds:.2f}s\n")
        return {"accuracy": acc, "macro_f1": macro_f1, "fault_f1": fault_f1}

    print("=" * 62)
    logreg_scores = summarise("Logistic Regression (baseline)", logreg_pred, logreg_time)
    rf_scores = summarise("Random Forest (selected)", rf_pred, rf_time)
    print("=" * 62)

    print("\nRandom Forest — per-class detail on the held-out test set:")
    print(classification_report(y_test, rf_pred, digits=3, zero_division=0))

    # 5-fold CV on the training half only — confirms the score is not a fluke
    # of one particular split. The test set stays untouched.
    cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"5-fold CV macro-F1 on train: {cv.mean():.4f} (+/- {cv.std():.4f})")

    print("\nFeature importance (how often a feature decided a split):")
    importance = (
        pd.Series(rf.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    for feat, imp in importance.items():
        bar = "#" * int(round(imp * 60))
        print(f"  {feat:<14} {imp:>6.3f}  {bar}")

    # ---------------- Save ----------------
    # We save a *bundle*, not a bare estimator. The backend needs the feature
    # order and the class order too; keeping them in one file makes it
    # impossible to load a model with the wrong column layout.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": rf,
        "features": FEATURES,
        "type_code": TYPE_CODE,
        "classes": list(rf.classes_),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "trained_at": bundle["trained_at"],
        "model_type": "RandomForestClassifier",
        "n_estimators": rf.n_estimators,
        "features": FEATURES,
        "classes": list(rf.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scores": {"random_forest": rf_scores, "logistic_regression": logreg_scores},
        "cv_macro_f1_mean": float(cv.mean()),
        "cv_macro_f1_std": float(cv.std()),
        "feature_importance": importance.round(4).to_dict(),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"\nwrote {MODEL_PATH.relative_to(ROOT)}")
    print(f"wrote {META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
