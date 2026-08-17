"""Research dataset construction.

Builds a cached table of labelled Sentinel-2 pixels, one group per
(region, period). Every sample is real: features come from the production
feature pipeline, labels from ESA WorldCover 2021 v200. Nothing is synthesised.

The cache exists because every experiment reuses the same pixels; re-fetching
imagery per experiment would make the study infeasible on a workstation and
would also make experiments non-comparable.

Each sample carries its spatial block (region key) and temporal group (period
key), which is what makes leakage-free spatial and temporal splits possible and
testable.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.geospatial.geometry import validate_region
from app.imagery.bands import LAND_COVER_BANDS, Band
from app.imagery.service import ImageryService
from app.imagery.stac import SentinelHubStacProvider
from app.models_ml.features import FEATURE_NAMES, FEATURE_VERSION, build_feature_matrix
from app.models_ml.labels import CLASS_LABELS, WORLDCOVER_TO_CLASS
from research.config import CACHE_DIR, CONFIG, MANIFEST_DIR, PERIODS, TemporalPeriod
from training.regions import TRAINING_REGIONS, TrainingRegion
from training.worldcover import load_labels_on_grid

logger = get_logger(__name__)

CACHE_FILE = CACHE_DIR / "pixels.npz"
MANIFEST_FILE = MANIFEST_DIR / "dataset_manifest.json"

#: Regions collected concurrently. Small, so the public catalogue is not
#: saturated (which previously failed a whole run).
GROUP_CONCURRENCY = 3


@dataclass(slots=True)
class GroupSample:
    """Pixels sampled from one region in one observation period."""

    region: TrainingRegion
    period: TemporalPeriod
    features: np.ndarray  # (n, n_features) float32
    labels: np.ndarray  # (n,) int16
    lon: np.ndarray  # (n,) float32 — pixel centre, WGS84
    lat: np.ndarray  # (n,) float32
    scene_id: str
    observation_date: str
    cloud_cover: float | None
    platform: str | None
    mgrs_tile: str | None
    grid: dict[str, Any]
    calibration: dict[str, Any]


@dataclass(slots=True)
class ResearchDataset:
    """The full cached dataset, loaded into memory."""

    features: np.ndarray
    labels: np.ndarray
    region: np.ndarray  # spatial block key per sample
    period: np.ndarray  # temporal group key per sample
    lon: np.ndarray
    lat: np.ndarray
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.labels.size)

    def columns(self, names: tuple[str, ...]) -> np.ndarray:
        """Select a feature subset by name, preserving the requested order."""
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise KeyError(f"Unknown features requested: {missing}")
        index = [self.feature_names.index(n) for n in names]
        return self.features[:, index]

    def mask_for(self, *, periods: tuple[str, ...] | None = None) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if periods is not None:
            mask &= np.isin(self.period, list(periods))
        return mask


def _stratified_sample(
    features: np.ndarray, labels: np.ndarray, extra: dict[str, np.ndarray], budget: int, rng
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Draw an approximately class-balanced sample so rare classes survive."""
    present = np.unique(labels)
    per_class = max(1, budget // max(1, present.size))
    chosen: list[np.ndarray] = []
    for class_id in present:
        indices = np.flatnonzero(labels == class_id)
        chosen.append(rng.choice(indices, size=min(per_class, indices.size), replace=False))
    picked = np.concatenate(chosen)
    rng.shuffle(picked)
    return features[picked], labels[picked], {k: v[picked] for k, v in extra.items()}


async def collect_group(
    service: ImageryService,
    region: TrainingRegion,
    period: TemporalPeriod,
    rng,
) -> GroupSample | None:
    """Fetch one scene and sample labelled pixels from it."""
    validated = validate_region(bbox=list(region.bbox))
    start, end = period.window()

    observations = await service.search(
        validated,
        start,
        end,
        max_cloud_cover=CONFIG.dataset_max_cloud,
        required_bands=LAND_COVER_BANDS,
    )
    if not observations:
        logger.warning("no_imagery", region=region.key, period=period.key)
        return None

    observation = service.select_best(observations, LAND_COVER_BANDS)
    scene = await service.load_scene(
        observation,
        validated,
        (*LAND_COVER_BANDS, Band.SCENE_CLASSIFICATION),
        max_dimension=CONFIG.sampling_max_dim,
    )
    matrix = build_feature_matrix(scene)
    if matrix.n_samples == 0:
        logger.warning("no_valid_pixels", region=region.key, period=period.key)
        return None

    label_grid = load_labels_on_grid(matrix.reference)
    raw = np.rint(label_grid.data.filled(0.0)).astype("int16")[matrix.valid_mask]

    mapped = np.full(raw.shape, -1, dtype="int16")
    for code, target in WORLDCOVER_TO_CLASS.items():
        mapped[raw == code] = int(target)

    # Pixel centres in WGS84, kept so splits can be visualised and leakage
    # can be checked geographically rather than by trusting a label.
    rows, cols = np.nonzero(matrix.valid_mask)
    transform = matrix.reference.transform
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    from rasterio.warp import transform as warp_transform

    lon, lat = warp_transform(matrix.reference.crs, "EPSG:4326", xs.tolist(), ys.tolist())

    keep = mapped >= 0
    features = matrix.values[keep]
    labels = mapped[keep]
    extra = {
        "lon": np.asarray(lon, dtype="float32")[keep],
        "lat": np.asarray(lat, dtype="float32")[keep],
    }
    if labels.size == 0:
        logger.warning("no_labelled_pixels", region=region.key, period=period.key)
        return None

    features, labels, extra = _stratified_sample(
        features, labels, extra, CONFIG.samples_per_group, rng
    )

    counts = {
        CLASS_LABELS[int(c)]: int(n)
        for c, n in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    logger.info(
        "group_sampled",
        region=region.key,
        period=period.key,
        samples=int(labels.size),
        classes=counts,
    )

    return GroupSample(
        region=region,
        period=period,
        features=features,
        labels=labels,
        lon=extra["lon"],
        lat=extra["lat"],
        scene_id=observation.source_id,
        observation_date=observation.observation_date.isoformat(),
        cloud_cover=observation.cloud_cover_percent,
        platform=observation.platform,
        mgrs_tile=str(observation.properties.get("grid_square") or ""),
        grid=matrix.reference.metadata(),
        calibration=scene.calibration.to_dict(),
    )


async def build_dataset(seed: int = 42) -> dict[str, Any]:
    """Collect every (region, period) group and write the cache and manifest."""
    from research.config import ensure_directories

    ensure_directories()
    provider = SentinelHubStacProvider()
    service = ImageryService(provider)
    gate = asyncio.Semaphore(GROUP_CONCURRENCY)

    jobs = [(region, period) for region in TRAINING_REGIONS for period in PERIODS]

    async def gated(region: TrainingRegion, period: TemporalPeriod) -> GroupSample | None:
        async with gate:
            rng = np.random.default_rng(abs(hash((region.key, period.key, seed))) % (2**32))
            try:
                return await collect_group(service, region, period, rng)
            except Exception as exc:
                # One unavailable group must not discard the whole build; it is
                # logged and simply absent from the manifest.
                logger.error(
                    "group_failed", region=region.key, period=period.key, error=str(exc)[:300]
                )
                return None

    try:
        results = await asyncio.gather(*(gated(r, p) for r, p in jobs))
    finally:
        await provider.close()

    groups = [g for g in results if g is not None]
    if not groups:
        raise RuntimeError(
            "No dataset groups could be collected; refusing to write an empty cache."
        )

    features = np.concatenate([g.features for g in groups]).astype("float32")
    labels = np.concatenate([g.labels for g in groups]).astype("int16")
    region_key = np.concatenate([np.full(g.labels.size, g.region.key) for g in groups])
    period_key = np.concatenate([np.full(g.labels.size, g.period.key) for g in groups])
    lon = np.concatenate([g.lon for g in groups]).astype("float32")
    lat = np.concatenate([g.lat for g in groups]).astype("float32")

    np.savez_compressed(
        CACHE_FILE,
        features=features,
        labels=labels,
        region=region_key,
        period=period_key,
        lon=lon,
        lat=lat,
        feature_names=np.array(FEATURE_NAMES),
    )

    manifest = {
        "dataset_version": CONFIG.dataset_version,
        "research_version": CONFIG.research_version,
        "built_at": dt.datetime.now(dt.UTC).isoformat(),
        "build_seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "samples_per_group": CONFIG.samples_per_group,
        "sampling_max_dim": CONFIG.sampling_max_dim,
        "max_cloud_cover": CONFIG.dataset_max_cloud,
        "total_samples": int(labels.size),
        "label_source": CONFIG.label_source,
        "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        "class_counts": {
            CLASS_LABELS[int(c)]: int(n)
            for c, n in zip(*np.unique(labels, return_counts=True), strict=True)
        },
        "imagery_provider": {
            "name": SentinelHubStacProvider.name,
            "collection": SentinelHubStacProvider.dataset,
            "processing_level": "Level-2A surface reflectance",
        },
        "groups": [
            {
                "region": g.region.key,
                "region_name": g.region.name,
                "biome": g.region.biome,
                "bbox": list(g.region.bbox),
                "spatial_block": g.region.key,
                "temporal_group": g.period.key,
                "period_label": g.period.label,
                "scene_id": g.scene_id,
                "observation_date": g.observation_date,
                "cloud_cover_percent": g.cloud_cover,
                "platform": g.platform,
                "mgrs_tile": g.mgrs_tile,
                "samples": int(g.labels.size),
                "grid": g.grid,
                "radiometric_calibration": g.calibration,
                "class_counts": {
                    CLASS_LABELS[int(c)]: int(n)
                    for c, n in zip(*np.unique(g.labels, return_counts=True), strict=True)
                },
            }
            for g in groups
        ],
        "limitations": [
            "ESA WorldCover is a reference land-cover map produced by a model; it "
            "is not ground truth, and disagreement with it is not necessarily "
            "classifier error.",
            "Labels are fixed at the 2021 epoch while imagery spans two periods, so "
            "genuine land-cover change after 2021 appears as label noise in the "
            "later period.",
            "Pixels are sampled with class stratification, so cached class "
            "proportions do not reflect true landscape proportions.",
        ],
    }
    manifest["cache_sha256"] = _sha256(CACHE_FILE)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "dataset_built",
        groups=len(groups),
        samples=int(labels.size),
        cache=str(CACHE_FILE),
    )
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset() -> ResearchDataset:
    """Load the cached dataset, or explain how to build it."""
    if not CACHE_FILE.is_file():
        raise FileNotFoundError(
            f"No research dataset cache at {CACHE_FILE}. "
            "Build it with: python -m research.run --experiment dataset"
        )
    payload = np.load(CACHE_FILE, allow_pickle=False)
    manifest = (
        json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.is_file() else {}
    )
    return ResearchDataset(
        features=payload["features"],
        labels=payload["labels"],
        region=payload["region"],
        period=payload["period"],
        lon=payload["lon"],
        lat=payload["lat"],
        feature_names=tuple(str(x) for x in payload["feature_names"]),
        manifest=manifest,
    )
