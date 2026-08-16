"""Land-cover inference over a satellite scene.

Produces a class raster, a per-pixel confidence raster and area statistics.
Confidence is the model's own maximum class probability; when a backend cannot
produce probabilities the confidence fields are omitted rather than filled with
a substitute number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.geospatial.raster import RasterGrid
from app.imagery.service import SceneStack
from app.models_ml.backends.base import ModelMetadata
from app.models_ml.features import FeatureMatrix, build_feature_matrix
from app.models_ml.labels import (
    CLASS_COLLAPSE_NOTES,
    CLASS_COLOURS,
    CLASS_INFO,
    CLASS_ORDER,
    LandCoverClass,
)
from app.models_ml.registry import LoadedModel, registry

logger = get_logger(__name__)

#: Predictions at or below this probability are reported but flagged, since a
#: five-class problem has a 0.20 chance baseline.
LOW_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True, slots=True)
class ClassAreaSummary:
    class_id: int
    label: str
    colour: str
    pixel_count: int
    percentage: float
    area_km2: float
    mean_confidence: float | None


@dataclass(frozen=True, slots=True)
class LandCoverResult:
    """Spatial land-cover prediction plus its provenance."""

    classification: RasterGrid
    confidence: RasterGrid | None
    class_summaries: list[ClassAreaSummary]
    classified_pixel_count: int
    classified_area_km2: float
    mean_confidence: float | None
    low_confidence_fraction: float | None
    model_metadata: ModelMetadata
    prediction_timestamp: str
    input_metadata: dict[str, Any]

    @property
    def distribution(self) -> dict[str, float]:
        """Class label to percentage of classified area."""
        return {summary.label: summary.percentage for summary in self.class_summaries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": [asdict(s) for s in self.class_summaries],
            "distribution": self.distribution,
            "classified_pixel_count": self.classified_pixel_count,
            "classified_area_km2": self.classified_area_km2,
            "mean_confidence": self.mean_confidence,
            "low_confidence_fraction": self.low_confidence_fraction,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "model": {
                "name": self.model_metadata.name,
                "version": self.model_metadata.version,
                "backend": self.model_metadata.backend,
                "trained_at": self.model_metadata.trained_at,
                "overall_accuracy": self.model_metadata.overall_accuracy,
                "macro_f1": self.model_metadata.macro_f1,
                "evaluation_protocol": self.model_metadata.evaluation_protocol,
                "evaluation_samples": self.model_metadata.evaluation_samples,
                "per_class_metrics": self.model_metadata.per_class_metrics,
                "label_source": self.model_metadata.label_source,
                "feature_names": list(self.model_metadata.feature_names),
                "feature_version": self.model_metadata.feature_version,
                "class_collapse_notes": list(CLASS_COLLAPSE_NOTES),
            },
            "prediction_timestamp": self.prediction_timestamp,
            "input": self.input_metadata,
            "grid": self.classification.metadata(),
        }


class LandCoverClassifier:
    """Applies a trained backend to a scene and summarises the result spatially."""

    def __init__(self, model: LoadedModel | None = None, backend_name: str | None = None) -> None:
        self._model = model
        self._backend_name = backend_name

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            self._model = registry.load(self._backend_name)
        return self._model

    def classify_scene(self, scene: SceneStack) -> LandCoverResult:
        features = build_feature_matrix(scene)
        return self.classify_features(
            features,
            input_metadata={
                "observation": scene.observation.to_metadata(),
                "cloud_masked_fraction": round(scene.masked_fraction, 4),
                "feature_pixel_count": features.n_samples,
            },
        )

    def classify_features(
        self, features: FeatureMatrix, *, input_metadata: dict[str, Any] | None = None
    ) -> LandCoverResult:
        import datetime as dt

        loaded = self.model
        backend = loaded.backend

        probabilities = backend.predict_proba(features.values)
        predictions = np.argmax(probabilities, axis=1).astype("int16")
        confidences = (
            probabilities.max(axis=1).astype("float32")
            if backend.supports_probability and probabilities.size
            else None
        )

        classification = features.to_raster(predictions.astype("float32"))
        confidence_grid = features.to_raster(confidences) if confidences is not None else None

        pixel_area_km2 = features.reference.pixel_area_m2() / 1_000_000.0
        total = int(predictions.size)

        summaries: list[ClassAreaSummary] = []
        for class_id in CLASS_ORDER:
            selected = predictions == int(class_id)
            count = int(selected.sum())
            info = CLASS_INFO[class_id]
            summaries.append(
                ClassAreaSummary(
                    class_id=int(class_id),
                    label=info.label,
                    colour=CLASS_COLOURS[int(class_id)],
                    pixel_count=count,
                    percentage=round(count / total * 100.0, 3) if total else 0.0,
                    area_km2=round(count * pixel_area_km2, 6),
                    mean_confidence=(
                        round(float(confidences[selected].mean()), 4)
                        if confidences is not None and count > 0
                        else None
                    ),
                )
            )

        mean_confidence = (
            round(float(confidences.mean()), 4) if confidences is not None and total else None
        )
        low_confidence_fraction = (
            round(float((confidences < LOW_CONFIDENCE_THRESHOLD).mean()), 4)
            if confidences is not None and total
            else None
        )

        logger.info(
            "land_cover_classified",
            backend=loaded.metadata.backend,
            pixels=total,
            mean_confidence=mean_confidence,
        )

        return LandCoverResult(
            classification=classification,
            confidence=confidence_grid,
            class_summaries=summaries,
            classified_pixel_count=total,
            classified_area_km2=round(total * pixel_area_km2, 6),
            mean_confidence=mean_confidence,
            low_confidence_fraction=low_confidence_fraction,
            model_metadata=loaded.metadata,
            prediction_timestamp=dt.datetime.now(dt.UTC).isoformat(),
            input_metadata=input_metadata or {},
        )


def dominant_class(result: LandCoverResult) -> LandCoverClass | None:
    ranked = sorted(result.class_summaries, key=lambda s: s.pixel_count, reverse=True)
    if not ranked or ranked[0].pixel_count == 0:
        return None
    return LandCoverClass(ranked[0].class_id)
