"""Discovery, integrity checking and caching of trained model artifacts.

A model is a directory containing ``manifest.json`` plus the serialised
backend. The manifest records the feature contract and a SHA-256 of the
artifact; both are verified before the artifact is deserialised, so a truncated
download or a model trained against a different feature definition fails loudly
instead of producing quietly wrong maps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import ModelInferenceError, ModelUnavailableError
from app.core.logging import get_logger
from app.models_ml.backends.base import LandCoverBackend, ModelMetadata
from app.models_ml.backends.random_forest import RandomForestBackend
from app.models_ml.backends.torch_mlp import TorchMLPBackend
from app.models_ml.features import FEATURE_NAMES, FEATURE_VERSION

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"

BACKENDS: dict[str, type[LandCoverBackend]] = {
    RandomForestBackend.backend_name: RandomForestBackend,
    TorchMLPBackend.backend_name: TorchMLPBackend,
}

ARTIFACT_FILENAMES: dict[str, str] = {
    RandomForestBackend.backend_name: "model.joblib",
    TorchMLPBackend.backend_name: "model.pt",
}


@dataclass(frozen=True, slots=True)
class LoadedModel:
    backend: LandCoverBackend
    metadata: ModelMetadata
    directory: Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    directory: Path,
    metadata: ModelMetadata,
    artifact_path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "metadata": metadata.to_dict(),
        "artifact": artifact_path.name,
        "artifact_sha256": sha256_of(artifact_path),
        **(extra or {}),
    }
    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def model_directory(backend_name: str | None = None) -> Path:
    return settings.model_dir / (backend_name or settings.land_cover_backend)


def available_models() -> list[dict[str, Any]]:
    """List every model directory that carries a readable manifest."""
    root = settings.model_dir
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for candidate in sorted(root.iterdir()):
        manifest_path = candidate / MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("model_manifest_unreadable", directory=candidate.name)
            continue
        found.append({"directory": candidate.name, **manifest.get("metadata", {})})
    return found


class ModelRegistry:
    """Process-wide cache of validated model artifacts."""

    def __init__(self) -> None:
        self._cache: dict[str, LoadedModel] = {}

    def clear(self) -> None:
        self._cache.clear()

    def load(self, backend_name: str | None = None) -> LoadedModel:
        name = backend_name or settings.land_cover_backend
        if name in self._cache:
            return self._cache[name]

        directory = model_directory(name)
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ModelUnavailableError(
                "No trained land-cover model is installed. Train one with "
                "`python -m training.train_land_cover` before requesting classification.",
                details={"expected_path": str(manifest_path)},
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelUnavailableError(
                "The model manifest is corrupt.", details={"path": str(manifest_path)}
            ) from exc

        metadata = ModelMetadata.from_dict(manifest.get("metadata", {}))
        self._assert_feature_compatibility(metadata, name)

        artifact_path = directory / manifest.get("artifact", ARTIFACT_FILENAMES.get(name, ""))
        if not artifact_path.is_file():
            raise ModelUnavailableError(
                "The model manifest references an artifact that is not present.",
                details={"artifact": artifact_path.name},
            )

        expected_digest = manifest.get("artifact_sha256")
        if expected_digest:
            actual_digest = sha256_of(artifact_path)
            if actual_digest != expected_digest:
                raise ModelUnavailableError(
                    "The model artifact failed its integrity check and was not loaded.",
                    details={"artifact": artifact_path.name},
                )

        backend_cls = BACKENDS.get(metadata.backend or name)
        if backend_cls is None:
            raise ModelUnavailableError(
                f"Unknown model backend '{metadata.backend}'.",
                details={"supported": sorted(BACKENDS)},
            )

        backend = backend_cls.load(artifact_path)
        loaded = LoadedModel(backend=backend, metadata=metadata, directory=directory)
        self._cache[name] = loaded
        logger.info(
            "model_loaded",
            backend=metadata.backend,
            version=metadata.version,
            accuracy=metadata.overall_accuracy,
        )
        return loaded

    @staticmethod
    def _assert_feature_compatibility(metadata: ModelMetadata, name: str) -> None:
        if tuple(metadata.feature_names) != FEATURE_NAMES:
            raise ModelInferenceError(
                "The installed model was trained on a different feature set and "
                "cannot be served. Retrain it against the current feature definition.",
                details={
                    "model_features": list(metadata.feature_names),
                    "expected_features": list(FEATURE_NAMES),
                    "backend": name,
                },
            )
        if metadata.feature_version != FEATURE_VERSION:
            logger.warning(
                "feature_version_mismatch",
                model_version=metadata.feature_version,
                runtime_version=FEATURE_VERSION,
            )


registry = ModelRegistry()
