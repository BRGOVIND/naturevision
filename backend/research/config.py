"""Research framework configuration.

Every experimental value lives here rather than being scattered through the
experiment scripts, so a run is fully described by this module plus a seed.
The version is recorded in every experiment record.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESEARCH_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"

RESEARCH_ROOT = Path(__file__).resolve().parent
CACHE_DIR = RESEARCH_ROOT / "cache"
RESULTS_DIR = RESEARCH_ROOT / "results"
MANIFEST_DIR = RESEARCH_ROOT / "manifests"
FIGURE_DIR = RESEARCH_ROOT / "figures"
TABLE_DIR = RESEARCH_ROOT / "tables"

#: Seeds used wherever an experiment involves randomness. Reported as
#: mean/std/min/max rather than a single run.
SEEDS: tuple[int, ...] = (42, 123, 2024)

#: Pixels sampled per region-period. Stratified across classes so minority
#: classes survive; the cap keeps the cached dataset small enough to commit
#: manifests for while remaining statistically usable.
SAMPLES_PER_GROUP = 20_000

#: Grid size used when reading scenes for sampling.
SAMPLING_MAX_DIM = 1024

#: Maximum scene cloud cover accepted when building the dataset.
DATASET_MAX_CLOUD = 20.0


@dataclass(frozen=True, slots=True)
class TemporalPeriod:
    """A named observation window used as a temporal group."""

    key: str
    label: str
    start_month: int
    span_days: int
    year: int

    def window(self) -> tuple[dt.date, dt.date]:
        start = dt.date(self.year, self.start_month, 1)
        return start, start + dt.timedelta(days=self.span_days)


#: Two temporal groups over the same regions. The label source is fixed
#: (WorldCover 2021 v200), so the later period carries additional label noise
#: wherever real land cover changed — documented as a limitation, not silently
#: treated as model error.
PERIODS: tuple[TemporalPeriod, ...] = (
    TemporalPeriod("p2021", "2021 window", 1, 120, 2021),
    TemporalPeriod("p2024", "2024 window", 1, 120, 2024),
)

#: Feature groups for the ablation. Names index into the production feature
#: contract; nothing here invents a band or an index.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A_raw_bands": ("blue", "green", "red", "nir", "swir16", "swir22"),
    "B_bands_ndvi": ("blue", "green", "red", "nir", "swir16", "swir22", "ndvi"),
    "C_bands_indices": (
        "blue",
        "green",
        "red",
        "nir",
        "swir16",
        "swir22",
        "ndvi",
        "ndwi",
        "ndbi",
    ),
    "D_full": (
        "blue",
        "green",
        "red",
        "nir",
        "swir16",
        "swir22",
        "ndvi",
        "ndwi",
        "ndbi",
        "nbr",
        "bsi_partial",
    ),
}

#: The feature set matching the production model, used for the baseline.
BASELINE_FEATURE_SET = "D_full"

#: Lightweight models compared under identical splits. Hyperparameters are
#: fixed in advance and never tuned against the test split.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 22,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "hist_gradient_boosting": {
        "max_iter": 200,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.15,
    },
    "linear_svm": {
        # A calibrated linear SVM: full kernel SVC is O(n^2) and infeasible on
        # hundreds of thousands of pixels on a workstation.
        "C": 1.0,
        "max_iter": 3000,
        "class_weight": "balanced",
    },
    "mlp": {
        "hidden_sizes": (128, 64),
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 1024,
        "epochs": 30,
    },
}

#: The model used for the baseline and for experiments that hold model fixed.
BASELINE_MODEL = "random_forest"

#: Cloud-cover buckets for the robustness experiment. A bucket is reported only
#: if real observations fall into it.
CLOUD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
)

#: Change-detection thresholds evaluated in the sensitivity study. The
#: production defaults (0.10 / 0.20) are unchanged; these are research-only.
CHANGE_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

#: Confidence buckets for the reliability analysis.
CONFIDENCE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)

#: Reference label product. Recorded verbatim in every manifest.
LABEL_SOURCE = {
    "name": "ESA WorldCover",
    "version": "v200",
    "epoch": "2021",
    "identifier": "ESA WorldCover 2021 v200",
    "resolution_m": 10,
    "license": "CC BY 4.0",
    "url": "https://esa-worldcover.org/",
    "role": (
        "Reference land-cover map used as training and evaluation labels. It is "
        "a model product with its own error, not ground truth."
    ),
}


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """The full experimental configuration for a run."""

    research_version: str = RESEARCH_VERSION
    dataset_version: str = DATASET_VERSION
    seeds: tuple[int, ...] = SEEDS
    samples_per_group: int = SAMPLES_PER_GROUP
    sampling_max_dim: int = SAMPLING_MAX_DIM
    dataset_max_cloud: float = DATASET_MAX_CLOUD
    feature_sets: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(FEATURE_SETS))
    models: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(MODEL_CONFIGS))
    baseline_model: str = BASELINE_MODEL
    baseline_feature_set: str = BASELINE_FEATURE_SET
    change_thresholds: tuple[float, ...] = CHANGE_THRESHOLDS
    cloud_buckets: tuple[tuple[str, float, float], ...] = CLOUD_BUCKETS
    confidence_buckets: tuple[tuple[float, float], ...] = CONFIDENCE_BUCKETS
    label_source: dict[str, Any] = field(default_factory=lambda: dict(LABEL_SOURCE))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["feature_sets"] = {k: list(v) for k, v in self.feature_sets.items()}
        data["change_thresholds"] = list(self.change_thresholds)
        data["cloud_buckets"] = [list(b) for b in self.cloud_buckets]
        data["confidence_buckets"] = [list(b) for b in self.confidence_buckets]
        data["models"] = {
            k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
            for k, v in self.models.items()
        }
        return data


CONFIG = ResearchConfig()


def ensure_directories() -> None:
    for directory in (CACHE_DIR, RESULTS_DIR, MANIFEST_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
