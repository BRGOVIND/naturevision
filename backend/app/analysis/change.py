"""Bi-temporal vegetation-index change detection.

Terminology is deliberately conservative. This module reports *vegetation-index
change*, which is an observation about surface reflectance. It does not label
change as deforestation, degradation or any other process, because a two-date
optical index difference cannot on its own distinguish land-cover conversion
from phenology, drought, harvest cycles, illumination differences or residual
atmospheric effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.analysis.statistics import RasterStatistics, compute_statistics
from app.core.config import settings
from app.core.errors import RasterProcessingError
from app.geospatial.raster import RasterGrid, common_valid_mask

#: Integer codes written into the change raster.
CLASS_NO_DATA = 0
CLASS_SIGNIFICANT_DECREASE = 1
CLASS_MODERATE_DECREASE = 2
CLASS_STABLE = 3
CLASS_MODERATE_INCREASE = 4
CLASS_SIGNIFICANT_INCREASE = 5

CHANGE_CLASS_LABELS: dict[int, str] = {
    CLASS_NO_DATA: "No data",
    CLASS_SIGNIFICANT_DECREASE: "Significant decrease",
    CLASS_MODERATE_DECREASE: "Moderate decrease",
    CLASS_STABLE: "Stable",
    CLASS_MODERATE_INCREASE: "Moderate increase",
    CLASS_SIGNIFICANT_INCREASE: "Significant increase",
}

#: Relative change is unstable when the baseline is near zero, so it is only
#: reported when the earlier mean is comfortably away from the origin.
MIN_BASELINE_FOR_RELATIVE_CHANGE = 0.05


@dataclass(frozen=True, slots=True)
class ChangeThresholds:
    """Change-magnitude cut points, in absolute NDVI units.

    Defaults are configurable via ``CHANGE_MODERATE_THRESHOLD`` and
    ``CHANGE_SIGNIFICANT_THRESHOLD``. The 0.10 default sits above the
    commonly cited ~0.02-0.05 combined radiometric and atmospheric-correction
    uncertainty of Sentinel-2 L2A NDVI, so it is unlikely to fire on sensor
    noise alone; 0.20 marks change large enough to be visible in the imagery.
    Both are reported with every result so a reader can re-derive the classes.
    """

    moderate: float = field(default_factory=lambda: settings.change_moderate_threshold)
    significant: float = field(default_factory=lambda: settings.change_significant_threshold)

    def __post_init__(self) -> None:
        if not 0 < self.moderate < self.significant:
            raise RasterProcessingError(
                "Change thresholds must satisfy 0 < moderate < significant.",
                details={"moderate": self.moderate, "significant": self.significant},
            )

    def to_dict(self) -> dict[str, float]:
        return {"moderate": self.moderate, "significant": self.significant}


@dataclass(frozen=True, slots=True)
class ChangeClassSummary:
    code: int
    label: str
    pixel_count: int
    percentage_of_analysed_area: float
    area_km2: float


@dataclass(frozen=True, slots=True)
class ChangeDetectionResult:
    """Full outcome of a bi-temporal comparison."""

    difference: RasterGrid
    classification: RasterGrid
    thresholds: ChangeThresholds
    difference_statistics: RasterStatistics
    mean_index_a: float | None
    mean_index_b: float | None
    absolute_change: float | None
    relative_change_percent: float | None
    changed_pixel_count: int
    comparable_pixel_count: int
    changed_area_percentage: float
    changed_area_km2: float
    decreased_area_percentage: float
    increased_area_percentage: float
    class_summaries: list[ChangeClassSummary]
    methodology: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "difference_statistics": self.difference_statistics.to_dict(),
            "mean_index_a": self.mean_index_a,
            "mean_index_b": self.mean_index_b,
            "absolute_change": self.absolute_change,
            "relative_change_percent": self.relative_change_percent,
            "changed_pixel_count": self.changed_pixel_count,
            "comparable_pixel_count": self.comparable_pixel_count,
            "changed_area_percentage": self.changed_area_percentage,
            "changed_area_km2": self.changed_area_km2,
            "decreased_area_percentage": self.decreased_area_percentage,
            "increased_area_percentage": self.increased_area_percentage,
            "classes": [asdict(c) for c in self.class_summaries],
            "methodology": self.methodology,
            "grid": self.difference.metadata(),
        }


def detect_change(
    index_a: RasterGrid,
    index_b: RasterGrid,
    *,
    thresholds: ChangeThresholds | None = None,
    index_name: str = "NDVI",
) -> ChangeDetectionResult:
    """Compare two index rasters pixel-by-pixel after co-registration.

    ``index_b`` is warped onto ``index_a``'s grid when the two differ, so the
    comparison is always performed on identical pixel geometry. Only pixels
    valid in *both* periods contribute; anything cloud-masked in either date is
    excluded rather than being treated as change.
    """
    cuts = thresholds or ChangeThresholds()
    aligned_b = index_b.reproject_to(index_a)

    comparable = common_valid_mask(index_a, aligned_b)
    difference_values = np.full(index_a.shape, np.nan, dtype="float32")
    np.subtract(
        aligned_b.data.data.astype("float64"),
        index_a.data.data.astype("float64"),
        out=difference_values,
        where=comparable,
        casting="unsafe",
    )
    comparable &= np.isfinite(difference_values)
    difference = RasterGrid(
        data=np.ma.masked_array(difference_values, mask=~comparable),
        transform=index_a.transform,
        crs=index_a.crs,
        nodata=float("nan"),
    )

    codes = np.full(index_a.shape, CLASS_NO_DATA, dtype="int16")
    magnitude = np.abs(difference_values)
    decreasing = comparable & (difference_values < 0)
    increasing = comparable & (difference_values > 0)

    codes[comparable] = CLASS_STABLE
    codes[decreasing & (magnitude >= cuts.moderate)] = CLASS_MODERATE_DECREASE
    codes[decreasing & (magnitude >= cuts.significant)] = CLASS_SIGNIFICANT_DECREASE
    codes[increasing & (magnitude >= cuts.moderate)] = CLASS_MODERATE_INCREASE
    codes[increasing & (magnitude >= cuts.significant)] = CLASS_SIGNIFICANT_INCREASE

    classification = RasterGrid(
        data=np.ma.masked_array(codes.astype("float32"), mask=~comparable),
        transform=index_a.transform,
        crs=index_a.crs,
        nodata=float(CLASS_NO_DATA),
    )

    comparable_count = int(comparable.sum())
    pixel_area_km2 = index_a.pixel_area_m2() / 1_000_000.0

    summaries: list[ChangeClassSummary] = []
    for code in (
        CLASS_SIGNIFICANT_DECREASE,
        CLASS_MODERATE_DECREASE,
        CLASS_STABLE,
        CLASS_MODERATE_INCREASE,
        CLASS_SIGNIFICANT_INCREASE,
    ):
        count = int((codes == code).sum())
        summaries.append(
            ChangeClassSummary(
                code=code,
                label=CHANGE_CLASS_LABELS[code],
                pixel_count=count,
                percentage_of_analysed_area=(
                    round(count / comparable_count * 100.0, 3) if comparable_count else 0.0
                ),
                area_km2=round(count * pixel_area_km2, 6),
            )
        )

    changed_count = comparable_count - int((codes == CLASS_STABLE).sum())
    decreased = int(
        ((codes == CLASS_MODERATE_DECREASE) | (codes == CLASS_SIGNIFICANT_DECREASE)).sum()
    )
    increased = int(
        ((codes == CLASS_MODERATE_INCREASE) | (codes == CLASS_SIGNIFICANT_INCREASE)).sum()
    )

    # Period means are recomputed over the comparable footprint only, so the
    # reported change equals mean(B) - mean(A) exactly on the same pixels.
    mean_a = _masked_mean(index_a.data.data, comparable)
    mean_b = _masked_mean(aligned_b.data.data, comparable)
    absolute_change = (
        round(mean_b - mean_a, 6) if mean_a is not None and mean_b is not None else None
    )
    relative_change = None
    if (
        mean_a is not None
        and abs(mean_a) >= MIN_BASELINE_FOR_RELATIVE_CHANGE
        and absolute_change is not None
    ):
        relative_change = round(absolute_change / abs(mean_a) * 100.0, 3)

    return ChangeDetectionResult(
        difference=difference,
        classification=classification,
        thresholds=cuts,
        difference_statistics=compute_statistics(difference),
        mean_index_a=mean_a,
        mean_index_b=mean_b,
        absolute_change=absolute_change,
        relative_change_percent=relative_change,
        changed_pixel_count=changed_count,
        comparable_pixel_count=comparable_count,
        changed_area_percentage=(
            round(changed_count / comparable_count * 100.0, 3) if comparable_count else 0.0
        ),
        changed_area_km2=round(changed_count * pixel_area_km2, 6),
        decreased_area_percentage=(
            round(decreased / comparable_count * 100.0, 3) if comparable_count else 0.0
        ),
        increased_area_percentage=(
            round(increased / comparable_count * 100.0, 3) if comparable_count else 0.0
        ),
        class_summaries=summaries,
        methodology={
            "index": index_name,
            "operation": f"per-pixel difference: {index_name}(period B) - {index_name}(period A)",
            "alignment": (
                "Period B was reprojected onto the period A pixel grid "
                "(bilinear) prior to differencing."
            ),
            "comparable_pixels_rule": (
                "Only pixels valid in both periods after cloud, shadow, cirrus "
                "and snow masking contribute to any statistic."
            ),
            "thresholds_units": "absolute index units",
            "thresholds": cuts.to_dict(),
            "relative_change_rule": (
                "Relative change is reported only when |mean(period A)| >= "
                f"{MIN_BASELINE_FOR_RELATIVE_CHANGE}, because a near-zero baseline "
                "makes a percentage change numerically unstable."
            ),
            "interpretation_scope": (
                "Results describe change in a reflectance-derived vegetation index. "
                "They do not establish land-cover conversion, causation, or "
                "ecological outcome."
            ),
        },
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return round(float(np.mean(values[mask])), 6)
