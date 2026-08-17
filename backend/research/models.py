"""Lightweight model zoo for the comparison experiments.

Every model is trainable on a workstation in seconds to a couple of minutes on
a few hundred thousand pixels. Hyperparameters come from the research config
and are fixed in advance; nothing here is tuned against a test split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models_ml.labels import CLASS_ORDER
from research.config import MODEL_CONFIGS

N_CLASSES = len(CLASS_ORDER)


@dataclass(slots=True)
class FittedModel:
    """A trained estimator plus the cost of producing and using it."""

    name: str
    estimator: Any
    hyperparameters: dict[str, Any]
    train_seconds: float
    n_train: int
    supports_probability: bool
    #: Set by `timed_inference`; reported alongside training cost so model
    #: selection can weigh speed as well as score.
    inference_seconds: float = 0.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict(features).astype("int16")

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Full class-space probabilities, or None if the model has none."""
        if not self.supports_probability:
            return None
        raw = self.estimator.predict_proba(features)
        expanded = np.zeros((raw.shape[0], N_CLASSES), dtype="float32")
        for column, class_id in enumerate(self.estimator.classes_):
            expanded[:, int(class_id)] = raw[:, column]
        return expanded

    def timed_inference(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        predictions = self.predict(features)
        self.inference_seconds = round(time.perf_counter() - started, 4)
        return predictions, self.inference_seconds


def _build(name: str, seed: int):
    """Construct an unfitted estimator for a configured model name."""
    params = dict(MODEL_CONFIGS[name])

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=seed, **params)

    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, **params)

    if name == "linear_svm":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        # LinearSVC has no predict_proba; wrapping it in a calibrator gives
        # honest probabilities instead of a decision-function stand-in.
        svm = LinearSVC(random_state=seed, dual="auto", **params)
        return make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(svm, method="sigmoid", cv=3),
        )

    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(params["hidden_sizes"]),
                learning_rate_init=params["learning_rate"],
                alpha=params["weight_decay"],
                batch_size=params["batch_size"],
                max_iter=params["epochs"],
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=5,
            ),
        )

    raise KeyError(f"Unknown model '{name}'. Configured: {sorted(MODEL_CONFIGS)}")


def fit_model(name: str, features: np.ndarray, labels: np.ndarray, seed: int) -> FittedModel:
    """Train one configured model and record what it cost."""
    estimator = _build(name, seed)
    started = time.perf_counter()
    estimator.fit(features, labels)
    elapsed = time.perf_counter() - started

    # Pipelines expose classes_ through their final step.
    final = (
        estimator[-1]
        if hasattr(estimator, "__getitem__") and hasattr(estimator, "steps")
        else estimator
    )
    if not hasattr(estimator, "classes_") and hasattr(final, "classes_"):
        estimator.classes_ = final.classes_  # type: ignore[attr-defined]

    return FittedModel(
        name=name,
        estimator=estimator,
        hyperparameters={
            k: (list(v) if isinstance(v, tuple) else v) for k, v in MODEL_CONFIGS[name].items()
        },
        train_seconds=round(elapsed, 3),
        n_train=int(labels.size),
        supports_probability=hasattr(estimator, "predict_proba"),
    )


def feature_importances(model: FittedModel, feature_names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Impurity-based importances, ranked. Empty for models that lack them."""
    values = getattr(model.estimator, "feature_importances_", None)
    if values is None:
        return []
    pairs: list[tuple[str, float]] = [
        (name, float(value)) for name, value in zip(feature_names, values, strict=False)
    ]
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [
        {"feature": name, "importance": value, "rank": rank}
        for rank, (name, value) in enumerate(pairs, start=1)
    ]
