"""Train and evaluate the land-cover classifier.

Usage:
    python -m training.train_land_cover --backend random_forest
    python -m training.train_land_cover --backend torch_mlp --samples-per-region 60000

The pipeline is: for each region, fetch a low-cloud Sentinel-2 L2A scene, build
the same feature cube the inference path uses, read co-located ESA WorldCover
labels, sample a class-stratified pixel set, then fit on the training regions
and evaluate on the entirely held-out evaluation regions.

Reported metrics come only from the held-out split. Nothing here defaults an
accuracy figure; if evaluation cannot run, the artifact records ``null``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.geospatial.geometry import validate_region
from app.imagery.bands import LAND_COVER_BANDS, Band
from app.imagery.service import ImageryService
from app.imagery.stac import SentinelHubStacProvider
from app.models_ml.backends.base import ModelMetadata
from app.models_ml.backends.random_forest import RandomForestBackend
from app.models_ml.backends.torch_mlp import TorchMLPBackend
from app.models_ml.features import FEATURE_NAMES, FEATURE_VERSION, build_feature_matrix
from app.models_ml.labels import (
    CLASS_COLLAPSE_NOTES,
    CLASS_LABELS,
    CLASS_ORDER,
    REFERENCE_LABEL_SOURCE,
    WORLDCOVER_TO_CLASS,
)
from app.models_ml.registry import ARTIFACT_FILENAMES, BACKENDS, model_directory, write_manifest
from training.regions import TrainingRegion, regions_for
from training.worldcover import load_labels_on_grid

logger = get_logger(__name__)

MODEL_NAME = "naturevision-landcover"
MODEL_VERSION = "1.0.0"

#: Grid size used when sampling; larger than the serving default so each region
#: contributes a broad, spatially diverse pixel pool.
SAMPLING_MAX_DIM = 1024

#: Regions collected concurrently. Kept small to stay within the public
#: catalogue's tolerance for parallel clients.
REGION_CONCURRENCY = 3


@dataclass(slots=True)
class RegionSample:
    region: TrainingRegion
    features: np.ndarray
    labels: np.ndarray
    observation_id: str
    observation_date: str


async def collect_region(
    service: ImageryService,
    region: TrainingRegion,
    samples_per_region: int,
    rng: np.random.Generator,
) -> RegionSample | None:
    """Build a class-stratified feature/label sample for one region."""
    validated = validate_region(bbox=list(region.bbox))
    observations = await service.search(
        validated,
        region.start,
        region.end,
        max_cloud_cover=15.0,
        required_bands=LAND_COVER_BANDS,
    )
    if not observations:
        logger.warning("no_imagery_for_region", region=region.key)
        return None

    observation = service.select_best(observations, LAND_COVER_BANDS)
    scene = await service.load_scene(
        observation,
        validated,
        (*LAND_COVER_BANDS, Band.SCENE_CLASSIFICATION),
        max_dimension=SAMPLING_MAX_DIM,
    )
    matrix = build_feature_matrix(scene)
    if matrix.n_samples == 0:
        logger.warning("no_valid_pixels_for_region", region=region.key)
        return None

    label_grid = load_labels_on_grid(matrix.reference)
    raw_labels = np.rint(label_grid.data.filled(0.0)).astype("int16")[matrix.valid_mask]

    mapped = np.full(raw_labels.shape, -1, dtype="int16")
    for worldcover_code, target in WORLDCOVER_TO_CLASS.items():
        mapped[raw_labels == worldcover_code] = int(target)

    keep = mapped >= 0
    features = matrix.values[keep]
    labels = mapped[keep]
    if labels.size == 0:
        logger.warning("no_labelled_pixels_for_region", region=region.key)
        return None

    features, labels = _stratified_sample(features, labels, samples_per_region, rng)
    logger.info(
        "region_sampled",
        region=region.key,
        samples=int(labels.size),
        classes={
            CLASS_LABELS[int(c)]: int(n)
            for c, n in zip(*np.unique(labels, return_counts=True), strict=True)
        },
    )
    return RegionSample(
        region=region,
        features=features,
        labels=labels,
        observation_id=observation.source_id,
        observation_date=observation.observation_date.isoformat(),
    )


def _stratified_sample(
    features: np.ndarray, labels: np.ndarray, budget: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an approximately balanced sample so rare classes survive."""
    present = np.unique(labels)
    per_class = max(1, budget // max(1, present.size))
    selected: list[np.ndarray] = []
    for class_id in present:
        indices = np.flatnonzero(labels == class_id)
        take = min(per_class, indices.size)
        selected.append(rng.choice(indices, size=take, replace=False))
    chosen = np.concatenate(selected)
    rng.shuffle(chosen)
    return features[chosen], labels[chosen]


def evaluate(backend, features: np.ndarray, labels: np.ndarray) -> dict:
    """Score the fitted model on held-out data and return real metrics."""
    predictions = backend.predict(features)
    present = sorted({int(c) for c in np.unique(np.concatenate([labels, predictions]))})
    report = classification_report(
        labels,
        predictions,
        labels=present,
        target_names=[CLASS_LABELS[c] for c in present],
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        CLASS_LABELS[c]: {
            "precision": round(float(report[CLASS_LABELS[c]]["precision"]), 4),
            "recall": round(float(report[CLASS_LABELS[c]]["recall"]), 4),
            "f1": round(float(report[CLASS_LABELS[c]]["f1-score"]), 4),
            "support": int(report[CLASS_LABELS[c]]["support"]),
        }
        for c in present
    }
    return {
        "overall_accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(
            float(f1_score(labels, predictions, labels=present, average="macro", zero_division=0)),
            4,
        ),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(CLASS_ORDER)))
        ).tolist(),
        "evaluation_samples": int(labels.size),
    }


async def run(backend_name: str, samples_per_region: int, seed: int) -> int:
    configure_logging()
    settings.ensure_directories()
    rng = np.random.default_rng(seed)

    provider = SentinelHubStacProvider()
    service = ImageryService(provider)
    # Regions are gathered a few at a time. Firing every region at once
    # saturates the public catalogue and the whole run fails on a transport
    # error; a small window keeps throughput high without tripping rate limits.
    gate = asyncio.Semaphore(REGION_CONCURRENCY)

    async def gated(region: TrainingRegion) -> RegionSample | None:
        async with gate:
            try:
                return await collect_region(service, region, samples_per_region, rng)
            except Exception as exc:
                # One unavailable region must not discard the whole training
                # run; it is logged and excluded from the recorded provenance.
                logger.error("region_collection_failed", region=region.key, error=str(exc)[:300])
                return None

    try:
        train_samples = [
            s
            for s in await asyncio.gather(*(gated(region) for region in regions_for("train")))
            if s is not None
        ]
        eval_samples = [
            s
            for s in await asyncio.gather(*(gated(region) for region in regions_for("evaluate")))
            if s is not None
        ]
    finally:
        await provider.close()

    logger.info(
        "regions_collected",
        train=[s.region.key for s in train_samples],
        evaluate=[s.region.key for s in eval_samples],
    )

    if not train_samples:
        logger.error("training_aborted_no_samples")
        return 1

    x_train = np.concatenate([s.features for s in train_samples])
    y_train = np.concatenate([s.labels for s in train_samples])
    logger.info("training_started", backend=backend_name, samples=int(y_train.size))

    backend_cls = BACKENDS[backend_name]
    backend = backend_cls()
    backend.fit(x_train, y_train)

    metrics: dict = {
        "overall_accuracy": None,
        "macro_f1": None,
        "per_class_metrics": {},
        "confusion_matrix": [],
        "evaluation_samples": None,
    }
    if eval_samples:
        x_eval = np.concatenate([s.features for s in eval_samples])
        y_eval = np.concatenate([s.labels for s in eval_samples])
        metrics = evaluate(backend, x_eval, y_eval)
        logger.info(
            "evaluation_completed",
            accuracy=metrics["overall_accuracy"],
            macro_f1=metrics["macro_f1"],
        )
    else:
        logger.warning("evaluation_skipped_no_holdout_samples")

    directory = model_directory(backend_name)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / ARTIFACT_FILENAMES[backend_name]
    backend.save(artifact_path)

    evaluation_regions = [s.region.to_dict() for s in eval_samples]
    metadata = ModelMetadata(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        backend=backend_name,
        feature_names=FEATURE_NAMES,
        feature_version=FEATURE_VERSION,
        class_labels=CLASS_LABELS,
        trained_at=dt.datetime.now(dt.UTC).isoformat(),
        training_samples=int(y_train.size),
        evaluation_samples=metrics["evaluation_samples"],
        overall_accuracy=metrics["overall_accuracy"],
        macro_f1=metrics["macro_f1"],
        per_class_metrics=metrics["per_class_metrics"],
        confusion_matrix=metrics["confusion_matrix"],
        evaluation_protocol=(
            "Spatially disjoint hold-out: the model is fitted only on the training "
            "regions and scored on evaluation regions that share no pixels, scenes "
            "or landscapes with them. No pixel-level random split is used, because "
            "spatial autocorrelation between adjacent 10 m pixels would inflate the "
            "reported accuracy."
        ),
        label_source=REFERENCE_LABEL_SOURCE,
        training_regions=[s.region.to_dict() for s in train_samples] + evaluation_regions,
        hyperparameters=backend.hyperparameters(),
        notes=[
            *CLASS_COLLAPSE_NOTES,
            "Accuracy is measured against ESA WorldCover, which is itself a model "
            "product with its own error; agreement with it is not ground truth.",
            "The classifier is per-pixel and uses no spatial context, so it produces "
            "salt-and-pepper noise at class boundaries and on mixed pixels.",
            "Training scenes are single-date. Cropland appearance varies strongly "
            "with phenology, so agriculture and bare land can be confused outside "
            "the growing season.",
        ],
    )

    extra: dict = {
        "training_observations": [
            {"region": s.region.key, "source_id": s.observation_id, "date": s.observation_date}
            for s in train_samples + eval_samples
        ]
    }
    if isinstance(backend, RandomForestBackend):
        extra["feature_importances"] = backend.feature_importances(FEATURE_NAMES)

    write_manifest(directory, metadata, artifact_path, extra)
    (directory / "metrics.json").write_text(
        json.dumps({**metrics, "trained_at": metadata.trained_at}, indent=2), encoding="utf-8"
    )

    logger.info("model_written", directory=str(directory), artifact=artifact_path.name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the NatureVision land-cover classifier.")
    parser.add_argument(
        "--backend",
        default=settings.land_cover_backend,
        choices=sorted(BACKENDS),
        help="Model backend to train.",
    )
    parser.add_argument(
        "--samples-per-region",
        type=int,
        default=40_000,
        help="Approximate stratified pixel budget per region.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return asyncio.run(run(args.backend, args.samples_per_region, args.seed))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Re-exported for convenience in tests.
__all__ = ["TorchMLPBackend", "collect_region", "evaluate", "main", "run"]
