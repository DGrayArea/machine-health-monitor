"""
Loads the trained models once and runs inference.

Loaded once, not per request
    joblib.load() reads about 10 MB off disk and rebuilds 300 trees. Doing that
    on every request would make the API roughly 100x slower for nothing. The
    bundle is loaded at startup and reused. scikit-learn predictors hold no
    state at inference time, so sharing one across threads is fine.

Column order comes from the bundle
    The .pkl stores the feature order used during training, and each input row
    is rebuilt against that list rather than a hardcoded one here. Retrain with a
    different feature set and this file still works, instead of quietly feeding
    `torque` into the column the model thinks is `power` and returning a
    confident wrong answer with no error.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend import config, rul
from backend.thresholds import derive_features

_bundle: dict[str, Any] | None = None
_rul_bundle: dict[str, Any] | None = None
_load_lock = threading.Lock()


class ModelNotAvailable(RuntimeError):
    """Raised when the .pkl is missing. The API turns this into an HTTP 503."""


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    """Load (and cache) the model bundle."""
    global _bundle
    with _load_lock:
        if _bundle is None or path is not None:
            model_path = path or config.MODEL_PATH
            if not model_path.exists():
                raise ModelNotAvailable(
                    f"No trained model at {model_path}. Run:\n"
                    "  python scripts/clean_data.py && python scripts/train_model.py"
                )
            _bundle = joblib.load(model_path)
        return _bundle


def is_ready() -> bool:
    try:
        load_bundle()
        return True
    except ModelNotAvailable:
        return False


def load_rul_bundle() -> dict[str, Any] | None:
    """
    Load the RUL regressor if it is there.

    Unlike the classifier this one is optional. RUL comes from the physics
    formula in backend/rul.py and the model is only a cross-check, so a missing
    rul_model.pkl drops one field from the response but breaks nothing.
    """
    global _rul_bundle
    with _load_lock:
        if _rul_bundle is None and config.RUL_MODEL_PATH.exists():
            _rul_bundle = joblib.load(config.RUL_MODEL_PATH)
        return _rul_bundle


def rul_model_available() -> bool:
    return load_rul_bundle() is not None


def metadata() -> dict[str, Any]:
    b = load_bundle()
    return {
        "features": b["features"],
        "classes": list(b["classes"]),
        "trained_at": b.get("trained_at"),
        "model_type": type(b["model"]).__name__,
        "n_estimators": getattr(b["model"], "n_estimators", None),
    }


def predict(reading: dict[str, Any]) -> dict[str, Any]:
    """
    Run one sensor reading through the model.

    Returns the status, the confidence, the full probability vector and the
    derived features, which the caller needs anyway for the threshold rules.

    Confidence is the probability of the winning class. For a Random Forest that
    is the share of the 300 trees that voted for it, so 0.62 really does mean
    186 trees said Fault and 114 said something else.
    """
    bundle = load_bundle()
    model, features = bundle["model"], bundle["features"]

    derived = derive_features(
        air_temp=reading["air_temp"],
        process_temp=reading["process_temp"],
        rot_speed=reading["rot_speed"],
        torque=reading["torque"],
        tool_wear=reading["tool_wear"],
        product_type=reading.get("product_type", "M"),
    )

    # Build the row in the order the model was trained on.
    #
    # This passes a named DataFrame rather than a bare numpy array on purpose.
    # The model was fitted with column names, so scikit-learn raises if the names
    # or their order do not match. A plain array accepts any ordering silently,
    # which is how you end up feeding `torque` into the `power` column and
    # getting a confident wrong answer with no error.
    row = pd.DataFrame([[derived[name] for name in features]], columns=features)

    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    best = int(np.argmax(proba))

    return {
        "status": str(classes[best]),
        "confidence": float(proba[best]),
        "probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
        "derived": {
            "temp_diff": derived["temp_diff"],
            "power": derived["power"],
            "strain": derived["strain"],
        },
        "features": derived,
        "rul": estimate_rul(derived, reading.get("product_type", "M")),
    }


def estimate_rul(features: dict[str, float], product_type: str = "M") -> dict[str, Any]:
    """
    Remaining useful life for one reading.

    The physics value is the one that counts. scripts/train_rul_model.py has the
    measurements showing the learned model just reproduces it. The model value is
    reported alongside as a cross-check, with the spread across trees, which is
    only non-zero near the point where the binding constraint switches.
    """
    estimate = rul.physics_rul(features, product_type)
    payload = estimate.to_dict()
    payload["band"] = rul.rul_band(estimate.remaining_min)
    payload["source"] = "physics"
    payload["model_remaining_min"] = None
    payload["model_sigma_min"] = None

    bundle = load_rul_bundle()
    if bundle is not None:
        columns = bundle["features"]
        row = pd.DataFrame([[features[name] for name in columns]], columns=columns)
        forest = bundle["model"]

        # The forest was fitted with column names, so predict() on the named
        # frame checks the ordering for us. The individual trees inside it were
        # fitted on the raw array, so they need .to_numpy(). Passing the frame
        # works but emits a warning per tree, per call.
        payload["model_remaining_min"] = round(float(forest.predict(row)[0]), 1)
        raw = row.to_numpy()
        per_tree = np.array([tree.predict(raw)[0] for tree in forest.estimators_])
        payload["model_sigma_min"] = round(float(per_tree.std()), 2)

    return payload
