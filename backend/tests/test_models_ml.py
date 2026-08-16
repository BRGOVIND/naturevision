"""Feature construction, model backends and inference."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ModelInferenceError, ModelUnavailableError
from app.imagery.bands import LAND_COVER_BANDS, Band
from app.models_ml.backends.base import ModelMetadata
from app.models_ml.backends.random_forest import RandomForestBackend
from app.models_ml.backends.torch_mlp import TorchMLPBackend
from app.models_ml.classifier import LandCoverClassifier
from app.models_ml.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    N_FEATURES,
    build_feature_matrix,
    stack_features,
)
from app.models_ml.labels import CLASS_LABELS, CLASS_ORDER, WORLDCOVER_TO_CLASS, LandCoverClass
from app.models_ml.registry import LoadedModel, ModelRegistry


# --- features ---------------------------------------------------------------
def test_feature_contract_is_stable():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert FEATURE_NAMES[: len(LAND_COVER_BANDS)] == tuple(b.value for b in LAND_COVER_BANDS)
    assert "ndvi" in FEATURE_NAMES


def test_stack_features_computes_indices_correctly():
    arrays = {
        Band.BLUE: np.full((2, 2), 0.04),
        Band.GREEN: np.full((2, 2), 0.06),
        Band.RED: np.full((2, 2), 0.05),
        Band.NIR: np.full((2, 2), 0.35),
        Band.SWIR_16: np.full((2, 2), 0.20),
        Band.SWIR_22: np.full((2, 2), 0.12),
    }
    cube = stack_features(arrays)
    assert cube.shape == (N_FEATURES, 2, 2)
    ndvi_index = FEATURE_NAMES.index("ndvi")
    assert cube[ndvi_index][0, 0] == pytest.approx((0.35 - 0.05) / (0.35 + 0.05), abs=1e-6)


def test_stack_features_rejects_missing_bands():
    with pytest.raises(ModelInferenceError, match="missing"):
        stack_features({Band.RED: np.zeros((2, 2))})


def test_stack_features_handles_zero_denominator():
    arrays = {band: np.zeros((2, 2)) for band in LAND_COVER_BANDS}
    cube = stack_features(arrays)
    ndvi_index = FEATURE_NAMES.index("ndvi")
    assert np.isnan(cube[ndvi_index]).all()


async def test_build_feature_matrix_from_scene(scene):
    matrix = build_feature_matrix(scene)
    assert matrix.values.shape[1] == N_FEATURES
    assert matrix.n_samples > 0
    assert matrix.feature_version == FEATURE_VERSION
    assert np.isfinite(matrix.values).all()


async def test_feature_matrix_round_trips_to_raster(scene):
    matrix = build_feature_matrix(scene)
    predictions = np.zeros(matrix.n_samples, dtype="float32")
    grid = matrix.to_raster(predictions)
    assert grid.shape == matrix.shape
    assert grid.crs == matrix.reference.crs
    assert grid.transform == matrix.reference.transform
    assert grid.valid_count == matrix.n_samples


async def test_to_raster_rejects_length_mismatch(scene):
    matrix = build_feature_matrix(scene)
    with pytest.raises(ModelInferenceError):
        matrix.to_raster(np.zeros(matrix.n_samples + 5, dtype="float32"))


# --- labels -------------------------------------------------------------------
def test_worldcover_mapping_covers_all_target_classes():
    mapped = set(WORLDCOVER_TO_CLASS.values())
    assert mapped == set(CLASS_ORDER)


def test_class_labels_align_with_enum():
    assert CLASS_LABELS[int(LandCoverClass.FOREST)] == "Forest"
    assert len(CLASS_LABELS) == len(CLASS_ORDER)


# --- backends -------------------------------------------------------------------
def _training_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    # Well-separated clusters, so a correct implementation must score highly.
    for class_id in range(len(CLASS_ORDER)):
        block = rng.normal(loc=class_id * 4.0, scale=0.4, size=(200, N_FEATURES))
        features.append(block)
        labels.extend([class_id] * 200)
    return np.vstack(features).astype("float32"), np.array(labels, dtype="int64")


@pytest.mark.parametrize("backend_cls", [RandomForestBackend, TorchMLPBackend])
def test_backend_trains_and_predicts_valid_probabilities(backend_cls):
    features, labels = _training_data()
    # The MLP needs a batch size below the sample count, or each epoch is a
    # single gradient step and the model never converges.
    backend = (
        RandomForestBackend()
        if backend_cls is RandomForestBackend
        else backend_cls(epochs=60, batch_size=128)
    )
    backend.fit(features, labels)

    probabilities = backend.predict_proba(features)
    assert probabilities.shape == (features.shape[0], len(CLASS_ORDER))
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)

    predictions = backend.predict(features)
    assert predictions.shape == (features.shape[0],)
    assert set(np.unique(predictions)).issubset({int(c) for c in CLASS_ORDER})
    # Separable clusters: anything below this indicates a broken pipeline.
    assert (predictions == labels).mean() > 0.9


def test_backend_handles_empty_input():
    features, labels = _training_data()
    backend = RandomForestBackend()
    backend.fit(features, labels)
    assert backend.predict_proba(np.empty((0, N_FEATURES), dtype="float32")).shape == (
        0,
        len(CLASS_ORDER),
    )


def test_random_forest_expands_probabilities_to_full_class_space():
    """A model trained on a subset of classes must still emit all columns."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(80, N_FEATURES)).astype("float32")
    labels = np.array([0, 2] * 40, dtype="int64")
    backend = RandomForestBackend(n_estimators=10)
    backend.fit(features, labels)
    probabilities = backend.predict_proba(features)
    assert probabilities.shape[1] == len(CLASS_ORDER)
    assert probabilities[:, 1].sum() == 0.0


@pytest.mark.parametrize("backend_cls", [RandomForestBackend, TorchMLPBackend])
def test_backend_round_trips_through_disk(tmp_path, backend_cls):
    features, labels = _training_data()
    backend = (
        RandomForestBackend()
        if backend_cls is RandomForestBackend
        else backend_cls(epochs=30, batch_size=128)
    )
    backend.fit(features, labels)
    before = backend.predict_proba(features[:20])

    path = tmp_path / ("model.joblib" if backend_cls is RandomForestBackend else "model.pt")
    backend.save(path)
    restored = backend_cls.load(path)
    after = restored.predict_proba(features[:20])
    assert np.allclose(before, after, atol=1e-5)


def test_untrained_torch_backend_refuses_to_predict_or_save(tmp_path):
    backend = TorchMLPBackend(n_features=N_FEATURES)
    with pytest.raises(ModelInferenceError):
        backend.predict_proba(np.zeros((2, N_FEATURES), dtype="float32"))
    with pytest.raises(ModelInferenceError):
        backend.save(tmp_path / "model.pt")


def test_corrupt_artifact_is_rejected(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"this is not a model")
    with pytest.raises(ModelInferenceError):
        RandomForestBackend.load(path)


# --- registry ---------------------------------------------------------------------
def test_registry_reports_missing_model_clearly(monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "model_dir", tmp_path)
    registry = ModelRegistry()
    with pytest.raises(ModelUnavailableError, match="No trained land-cover model"):
        registry.load("random_forest")


def test_registry_rejects_feature_contract_mismatch():
    metadata = ModelMetadata(
        name="test",
        version="0.0.1",
        backend="random_forest",
        feature_names=("a", "b"),
        feature_version="0.0.1",
        class_labels=CLASS_LABELS,
    )
    with pytest.raises(ModelInferenceError, match="different feature set"):
        ModelRegistry._assert_feature_compatibility(metadata, "random_forest")


# --- classification -------------------------------------------------------------------
def _loaded_model() -> LoadedModel:
    features, labels = _training_data()
    backend = RandomForestBackend(n_estimators=25)
    backend.fit(features, labels)
    metadata = ModelMetadata(
        name="test-landcover",
        version="0.0.1",
        backend="random_forest",
        feature_names=FEATURE_NAMES,
        feature_version=FEATURE_VERSION,
        class_labels=CLASS_LABELS,
        overall_accuracy=0.61,
        macro_f1=0.55,
        evaluation_protocol="synthetic",
    )
    from pathlib import Path

    return LoadedModel(backend=backend, metadata=metadata, directory=Path("."))


async def test_classifier_produces_spatial_output_and_confidence(scene):
    classifier = LandCoverClassifier(model=_loaded_model())
    result = classifier.classify_scene(scene)

    assert result.classification.shape == scene.reference.shape
    assert result.confidence is not None
    assert result.classified_pixel_count > 0
    assert sum(result.distribution.values()) == pytest.approx(100.0, abs=0.01)
    assert 0.0 <= result.mean_confidence <= 1.0
    assert result.classification.crs == scene.reference.crs


async def test_classifier_result_serialises_with_model_provenance(scene):
    classifier = LandCoverClassifier(model=_loaded_model())
    payload = classifier.classify_scene(scene).to_dict()

    assert payload["model"]["name"] == "test-landcover"
    assert payload["model"]["overall_accuracy"] == 0.61
    assert payload["model"]["feature_version"] == FEATURE_VERSION
    assert len(payload["classes"]) == len(CLASS_ORDER)
    assert payload["prediction_timestamp"]


async def test_classifier_areas_sum_to_classified_area(scene):
    classifier = LandCoverClassifier(model=_loaded_model())
    result = classifier.classify_scene(scene)
    total = sum(s.area_km2 for s in result.class_summaries)
    assert total == pytest.approx(result.classified_area_km2, rel=1e-4)


def test_accuracy_is_never_invented():
    """A model with no evaluation must report None, not a plausible number."""
    metadata = ModelMetadata(
        name="unevaluated",
        version="0.0.1",
        backend="random_forest",
        feature_names=FEATURE_NAMES,
        feature_version=FEATURE_VERSION,
        class_labels=CLASS_LABELS,
    )
    assert metadata.overall_accuracy is None
    assert metadata.macro_f1 is None
    assert metadata.evaluation_samples is None
