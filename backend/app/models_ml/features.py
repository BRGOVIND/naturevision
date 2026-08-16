"""Feature construction for pixel-wise land-cover classification.

The feature vector is a contract shared by training and inference. Its
definition lives here, is versioned, and is written into every model artifact
so that a model can never be served against features it was not trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.errors import ModelInferenceError
from app.geospatial.raster import RasterGrid
from app.imagery.bands import LAND_COVER_BANDS, Band
from app.imagery.service import SceneStack

#: Bumped whenever the feature definition changes in a way that invalidates
#: previously trained artifacts.
FEATURE_VERSION = "1.0.0"

#: Reflectance bands, in fixed order.
REFLECTANCE_FEATURES: tuple[str, ...] = tuple(b.value for b in LAND_COVER_BANDS)

#: Normalised-difference features appended after the reflectance bands. Each is
#: (name, positive band, negative band).
INDEX_FEATURES: tuple[tuple[str, Band, Band], ...] = (
    ("ndvi", Band.NIR, Band.RED),
    ("ndwi", Band.GREEN, Band.NIR),
    ("ndbi", Band.SWIR_16, Band.NIR),
    ("nbr", Band.NIR, Band.SWIR_22),
    ("bsi_partial", Band.SWIR_16, Band.BLUE),
)

FEATURE_NAMES: tuple[str, ...] = REFLECTANCE_FEATURES + tuple(n for n, _, _ in INDEX_FEATURES)
N_FEATURES = len(FEATURE_NAMES)

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Flattened per-pixel features plus the geometry needed to re-raster them."""

    values: np.ndarray  # (n_valid_pixels, n_features), float32
    valid_mask: np.ndarray  # (height, width) bool
    shape: tuple[int, int]
    reference: RasterGrid
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: str = FEATURE_VERSION

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    def to_raster(self, flat_values: np.ndarray, fill: float = np.nan) -> RasterGrid:
        """Scatter a per-valid-pixel vector back onto the analysis grid."""
        if flat_values.shape[0] != self.n_samples:
            raise ModelInferenceError(
                "Prediction length does not match the number of classified pixels.",
                details={"expected": self.n_samples, "received": int(flat_values.shape[0])},
            )
        canvas = np.full(self.shape, fill, dtype="float32")
        canvas[self.valid_mask] = flat_values.astype("float32")
        return RasterGrid(
            data=np.ma.masked_array(canvas, mask=~self.valid_mask),
            transform=self.reference.transform,
            crs=self.reference.crs,
            nodata=float("nan"),
        )


def stack_features(band_arrays: dict[Band, np.ndarray]) -> np.ndarray:
    """Build the (n_features, height, width) feature cube from raw band arrays.

    Shared by the training sampler and the inference path so the two can never
    drift apart.
    """
    missing = [b.value for b in LAND_COVER_BANDS if b not in band_arrays]
    if missing:
        raise ModelInferenceError(
            "Required bands are missing from the feature inputs.",
            details={"missing_bands": missing},
        )

    layers: list[np.ndarray] = [band_arrays[b].astype("float32") for b in LAND_COVER_BANDS]

    for _name, positive, negative in INDEX_FEATURES:
        a = band_arrays[positive].astype("float64")
        b = band_arrays[negative].astype("float64")
        denominator = a + b
        with np.errstate(invalid="ignore", divide="ignore"):
            index = np.where(
                np.abs(denominator) > _EPSILON,
                (a - b) / np.where(np.abs(denominator) > _EPSILON, denominator, 1.0),
                np.nan,
            )
        layers.append(index.astype("float32"))

    return np.stack(layers, axis=0)


def build_feature_matrix(scene: SceneStack) -> FeatureMatrix:
    """Extract classifier inputs from a loaded, cloud-masked scene."""
    missing = [b.value for b in LAND_COVER_BANDS if b not in scene.bands]
    if missing:
        raise ModelInferenceError(
            "The selected observation does not provide every band the land-cover model requires.",
            details={"missing_bands": missing},
        )

    reference = scene.band(LAND_COVER_BANDS[0])
    valid = np.ones(reference.shape, dtype=bool)
    band_arrays: dict[Band, np.ndarray] = {}
    for band in LAND_COVER_BANDS:
        grid = scene.band(band)
        if not grid.is_aligned_with(reference):
            raise ModelInferenceError(
                "Classifier input bands are not co-registered.",
                details={"band": band.value},
            )
        band_arrays[band] = grid.data.data
        valid &= ~np.ma.getmaskarray(grid.data)

    cube = stack_features(band_arrays)
    valid &= np.all(np.isfinite(cube), axis=0)

    values = cube[:, valid].T.astype("float32")
    return FeatureMatrix(
        values=values,
        valid_mask=valid,
        shape=reference.shape,
        reference=reference,
    )
