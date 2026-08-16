"""Random-forest land-cover backend.

A tree ensemble on per-pixel spectral features is a well-established baseline
for optical land-cover mapping: it needs no scaling, tolerates the strong
correlation between Sentinel-2 bands, trains in seconds on millions of pixels,
and yields class-vote frequencies that are usable as a confidence signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.core.errors import ModelInferenceError
from app.models_ml.backends.base import LandCoverBackend
from app.models_ml.labels import CLASS_ORDER

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    # Reference labels are strongly imbalanced across biomes; balancing the
    # subsample keeps rare classes (water, bare) from being ignored.
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}


class RandomForestBackend(LandCoverBackend):
    backend_name = "random_forest"
    supports_probability = True

    def __init__(self, model: RandomForestClassifier | None = None, **overrides: Any) -> None:
        params = {**DEFAULT_HYPERPARAMETERS, **overrides}
        self._params = params
        self.model = model or RandomForestClassifier(**params)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0, len(CLASS_ORDER)), dtype="float32")
        raw = self.model.predict_proba(features).astype("float32")
        # scikit-learn only emits columns for classes seen during training;
        # expand to the full class space so downstream indexing is stable.
        expanded = np.zeros((raw.shape[0], len(CLASS_ORDER)), dtype="float32")
        for column, class_id in enumerate(self.model.classes_):
            expanded[:, int(class_id)] = raw[:, column]
        return expanded

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self._params, "model": self.model}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> RandomForestBackend:
        # Trust boundary: joblib uses pickle, so loading executes code from the
        # artifact. Artifacts are produced only by this repository's training
        # pipeline and are integrity-checked by the model registry against the
        # SHA-256 recorded in the manifest before this point is reached. Never
        # point MODEL_DIR at a directory fed by untrusted uploads.
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ModelInferenceError(
                "The random-forest artifact could not be deserialised.",
                details={"path": path.name},
            ) from exc
        return cls(model=payload["model"], **payload.get("params", {}))

    def hyperparameters(self) -> dict[str, Any]:
        return dict(self._params)

    def feature_importances(self, feature_names: tuple[str, ...]) -> dict[str, float]:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}
        return {
            name: round(float(value), 6)
            for name, value in zip(feature_names, importances, strict=False)
        }
