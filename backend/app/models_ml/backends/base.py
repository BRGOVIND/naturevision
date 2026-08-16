"""Interface shared by every land-cover model backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Everything needed to interpret and audit a trained artifact.

    Accuracy fields are populated exclusively by the training pipeline from a
    held-out split. They are never defaulted to a plausible-looking number: an
    artifact whose evaluation did not run carries ``None``.
    """

    name: str
    version: str
    backend: str
    feature_names: tuple[str, ...]
    feature_version: str
    class_labels: dict[int, str]
    trained_at: str | None = None
    training_samples: int | None = None
    evaluation_samples: int | None = None
    overall_accuracy: float | None = None
    macro_f1: float | None = None
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    evaluation_protocol: str | None = None
    label_source: dict[str, Any] = field(default_factory=dict)
    training_regions: list[dict[str, Any]] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        data["class_labels"] = {str(k): v for k, v in self.class_labels.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        payload = dict(data)
        payload["feature_names"] = tuple(payload.get("feature_names", ()))
        payload["class_labels"] = {
            int(k): v for k, v in (payload.get("class_labels") or {}).items()
        }
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class LandCoverBackend(ABC):
    """A trainable, probabilistic pixel classifier.

    Backends must expose calibrated-or-honest class probabilities; the product
    surfaces confidence only where the underlying model genuinely produces it.
    """

    backend_name: str = "base"
    #: True when ``predict_proba`` returns a real probability distribution.
    supports_probability: bool = True

    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """(n_samples, n_classes) probabilities, rows summing to 1."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> LandCoverBackend: ...

    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]: ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1).astype("int16")
