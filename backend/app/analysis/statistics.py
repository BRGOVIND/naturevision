"""Deterministic spatial statistics for index rasters.

These are the authoritative numbers for the whole product: the dashboard,
the report and the language interpretation all consume them and none of those
layers is permitted to recompute or estimate them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.geospatial.raster import RasterGrid

#: NDVI bands used for the vegetation-density summary. Boundaries follow the
#: conventional interpretation of NDVI for optical sensors and are reported
#: alongside results so readers can judge them.
NDVI_DENSITY_CLASSES: tuple[tuple[str, float, float], ...] = (
    ("water_or_nonvegetated", -1.0, 0.05),
    ("bare_or_sparse", 0.05, 0.20),
    ("low_vegetation", 0.20, 0.40),
    ("moderate_vegetation", 0.40, 0.60),
    ("dense_vegetation", 0.60, 1.0),
)

#: Minimum valid pixels before a dispersion statistic is reported.
MIN_PIXELS_FOR_DISPERSION = 30


@dataclass(frozen=True, slots=True)
class RasterStatistics:
    """Summary of a single index raster over the analysed region."""

    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    std_dev: float | None
    percentile_10: float | None
    percentile_90: float | None
    valid_pixel_count: int
    total_pixel_count: int
    valid_fraction: float
    valid_area_km2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VegetationSummary:
    """Vegetation-density breakdown derived from an NDVI raster."""

    class_percentages: dict[str, float]
    class_area_km2: dict[str, float]
    vegetated_fraction: float
    class_bounds: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_statistics(grid: RasterGrid) -> RasterStatistics:
    """Compute descriptive statistics over the valid pixels of a grid."""
    values = grid.data.compressed()
    total = grid.height * grid.width
    valid = int(values.size)
    pixel_area_km2 = grid.pixel_area_m2() / 1_000_000.0

    if valid == 0:
        return RasterStatistics(
            mean=None,
            median=None,
            minimum=None,
            maximum=None,
            std_dev=None,
            percentile_10=None,
            percentile_90=None,
            valid_pixel_count=0,
            total_pixel_count=total,
            valid_fraction=0.0,
            valid_area_km2=0.0,
        )

    # Dispersion over a handful of pixels is not meaningful and would invite
    # over-reading of the result, so it is withheld below a floor.
    dispersion_meaningful = valid >= MIN_PIXELS_FOR_DISPERSION

    return RasterStatistics(
        mean=_round(float(np.mean(values))),
        median=_round(float(np.median(values))),
        minimum=_round(float(np.min(values))),
        maximum=_round(float(np.max(values))),
        std_dev=_round(float(np.std(values, ddof=1))) if dispersion_meaningful else None,
        percentile_10=_round(float(np.percentile(values, 10))) if dispersion_meaningful else None,
        percentile_90=_round(float(np.percentile(values, 90))) if dispersion_meaningful else None,
        valid_pixel_count=valid,
        total_pixel_count=total,
        valid_fraction=round(valid / total, 6) if total else 0.0,
        valid_area_km2=round(valid * pixel_area_km2, 6),
    )


def summarise_vegetation(grid: RasterGrid) -> VegetationSummary:
    """Bin an NDVI raster into interpretable vegetation-density classes."""
    values = grid.data.compressed()
    pixel_area_km2 = grid.pixel_area_m2() / 1_000_000.0
    total = int(values.size)

    percentages: dict[str, float] = {}
    areas: dict[str, float] = {}
    bounds: dict[str, list[float]] = {}
    vegetated = 0

    for name, lower, upper in NDVI_DENSITY_CLASSES:
        bounds[name] = [lower, upper]
        if total == 0:
            percentages[name] = 0.0
            areas[name] = 0.0
            continue
        # Upper-inclusive only for the final bin so every pixel lands once.
        in_class = (
            (values >= lower) & (values <= upper)
            if upper >= 1.0
            else (values >= lower) & (values < upper)
        )
        count = int(in_class.sum())
        percentages[name] = round(count / total * 100.0, 3)
        areas[name] = round(count * pixel_area_km2, 6)
        if lower >= 0.20:
            vegetated += count

    return VegetationSummary(
        class_percentages=percentages,
        class_area_km2=areas,
        vegetated_fraction=round(vegetated / total, 6) if total else 0.0,
        class_bounds=bounds,
    )


def _round(value: float, digits: int = 6) -> float | None:
    if not np.isfinite(value):
        return None
    return round(value, digits)
