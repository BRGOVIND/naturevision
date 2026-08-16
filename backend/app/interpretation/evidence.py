"""Construction of the deterministic evidence package.

This is the single input the language layer is allowed to see. Everything in it
is produced by deterministic raster processing or by a model whose output is
explicitly labelled as a prediction. The language layer receives no raw imagery
statistics it could re-derive differently, and no free-form text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.analysis.change import ChangeDetectionResult
from app.analysis.statistics import RasterStatistics, VegetationSummary
from app.geospatial.geometry import ValidatedRegion
from app.imagery.service import SceneStack
from app.models_ml.classifier import LandCoverResult


@dataclass(slots=True)
class EvidencePackage:
    """Structured, provenance-tagged facts about one analysis."""

    region: dict[str, Any]
    periods: dict[str, Any]
    data_sources: list[dict[str, Any]]
    observed: dict[str, Any] = field(default_factory=dict)
    model_predictions: dict[str, Any] = field(default_factory=dict)
    methodology: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def numeric_claims(self) -> dict[str, float]:
        """Every number the interpretation is permitted to state.

        Used to verify that generated text does not introduce figures that were
        never measured.
        """
        claims: dict[str, float] = {}

        def walk(prefix: str, node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(f"{prefix}.{key}" if prefix else str(key), value)
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(f"{prefix}[{index}]", value)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                claims[prefix] = float(node)

        walk("observed", self.observed)
        walk("model_predictions", self.model_predictions)
        walk("region", self.region)
        walk("data_quality", self.data_quality)
        # Source metadata carries citable figures too — scene cloud cover and
        # ground resolution are values an interpretation legitimately quotes.
        walk("data_sources", self.data_sources)
        return claims


BASE_LIMITATIONS: tuple[str, ...] = (
    "Analysis is based on optical satellite imagery, which cannot see through "
    "cloud; masked pixels are excluded rather than estimated.",
    "A vegetation index measures reflectance-derived greenness, not biomass, "
    "carbon stock, habitat quality or biodiversity.",
    "Comparing two acquisition dates cannot separate land-cover change from "
    "seasonal phenology, crop cycles, drought response or differences in "
    "illumination and view geometry.",
    "No causal attribution is made. The analysis identifies where and how much "
    "the index changed, not why.",
    "Land-cover classes are per-pixel predictions from a statistical model and "
    "carry classification error, particularly on mixed pixels and class "
    "boundaries.",
)


def build_evidence(
    *,
    region: ValidatedRegion,
    region_name: str | None,
    scene_a: SceneStack,
    ndvi_stats_a: RasterStatistics,
    vegetation_a: VegetationSummary,
    scene_b: SceneStack | None = None,
    ndvi_stats_b: RasterStatistics | None = None,
    vegetation_b: VegetationSummary | None = None,
    change: ChangeDetectionResult | None = None,
    land_cover: LandCoverResult | None = None,
    period_a_label: str = "",
    period_b_label: str = "",
) -> EvidencePackage:
    """Assemble the evidence package from completed pipeline outputs."""
    sources: list[dict[str, Any]] = [_source_entry(scene_a, "A")]
    if scene_b is not None:
        sources.append(_source_entry(scene_b, "B"))

    observed: dict[str, Any] = {
        "period_a": {
            "observation_date": scene_a.observation.observation_date.isoformat(),
            "mean_ndvi": ndvi_stats_a.mean,
            "median_ndvi": ndvi_stats_a.median,
            "min_ndvi": ndvi_stats_a.minimum,
            "max_ndvi": ndvi_stats_a.maximum,
            "std_dev_ndvi": ndvi_stats_a.std_dev,
            "valid_pixel_count": ndvi_stats_a.valid_pixel_count,
            "valid_area_km2": ndvi_stats_a.valid_area_km2,
            "vegetation_density_percentages": vegetation_a.class_percentages,
            "vegetated_fraction": vegetation_a.vegetated_fraction,
        }
    }

    if scene_b is not None and ndvi_stats_b is not None and vegetation_b is not None:
        observed["period_b"] = {
            "observation_date": scene_b.observation.observation_date.isoformat(),
            "mean_ndvi": ndvi_stats_b.mean,
            "median_ndvi": ndvi_stats_b.median,
            "min_ndvi": ndvi_stats_b.minimum,
            "max_ndvi": ndvi_stats_b.maximum,
            "std_dev_ndvi": ndvi_stats_b.std_dev,
            "valid_pixel_count": ndvi_stats_b.valid_pixel_count,
            "valid_area_km2": ndvi_stats_b.valid_area_km2,
            "vegetation_density_percentages": vegetation_b.class_percentages,
            "vegetated_fraction": vegetation_b.vegetated_fraction,
        }

    if change is not None:
        observed["change"] = {
            "mean_ndvi_period_a": change.mean_index_a,
            "mean_ndvi_period_b": change.mean_index_b,
            "absolute_ndvi_change": change.absolute_change,
            "relative_ndvi_change_percent": change.relative_change_percent,
            "changed_area_percentage": change.changed_area_percentage,
            "changed_area_km2": change.changed_area_km2,
            "decreased_area_percentage": change.decreased_area_percentage,
            "increased_area_percentage": change.increased_area_percentage,
            "comparable_pixel_count": change.comparable_pixel_count,
            "change_classes": {
                summary.label: summary.percentage_of_analysed_area
                for summary in change.class_summaries
            },
            "thresholds": change.thresholds.to_dict(),
        }

    model_predictions: dict[str, Any] = {}
    if land_cover is not None:
        model_predictions["land_cover"] = {
            "distribution_percent": land_cover.distribution,
            "area_km2": {s.label: s.area_km2 for s in land_cover.class_summaries},
            "mean_confidence": land_cover.mean_confidence,
            "low_confidence_fraction": land_cover.low_confidence_fraction,
            "per_class_mean_confidence": {
                s.label: s.mean_confidence for s in land_cover.class_summaries
            },
            "model_name": land_cover.model_metadata.name,
            "model_version": land_cover.model_metadata.version,
            "model_backend": land_cover.model_metadata.backend,
            "held_out_overall_accuracy": land_cover.model_metadata.overall_accuracy,
            "held_out_macro_f1": land_cover.model_metadata.macro_f1,
        }

    limitations = list(BASE_LIMITATIONS)
    quality = _quality_block(scene_a, scene_b, change)
    limitations.extend(_conditional_limitations(quality, land_cover, change))

    methodology: dict[str, Any] = {
        "index_formula": "NDVI = (NIR - Red) / (NIR + Red)",
        "bands": {"red": "Sentinel-2 B04 (664.6 nm)", "nir": "Sentinel-2 B08 (832.8 nm)"},
        "cloud_masking": (
            "Sentinel-2 Level-2A scene classification: no-data, saturated, cloud "
            "shadow, medium- and high-probability cloud, thin cirrus and snow/ice "
            "pixels are excluded."
        ),
        "analysis_grid": scene_a.reference.metadata(),
        "radiometric_calibration": scene_a.calibration.to_dict(),
    }
    if change is not None:
        methodology["change_detection"] = change.methodology
    if land_cover is not None:
        methodology["land_cover_model"] = {
            "name": land_cover.model_metadata.name,
            "version": land_cover.model_metadata.version,
            "backend": land_cover.model_metadata.backend,
            "features": list(land_cover.model_metadata.feature_names),
            "evaluation_protocol": land_cover.model_metadata.evaluation_protocol,
            "label_source": land_cover.model_metadata.label_source,
        }

    return EvidencePackage(
        region={
            "name": region_name or region.describe(),
            "description": region.describe(),
            "bbox": region.bbox.as_list(),
            "area_km2": round(region.area_km2, 4),
            "crs": region.crs,
        },
        periods={
            "period_a": period_a_label,
            "period_b": period_b_label or None,
        },
        data_sources=sources,
        observed=observed,
        model_predictions=model_predictions,
        methodology=methodology,
        data_quality=quality,
        limitations=limitations,
    )


def _source_entry(scene: SceneStack, period: str) -> dict[str, Any]:
    observation = scene.observation
    return {
        "period": period,
        "provider": observation.provider,
        "dataset": observation.dataset,
        "source_id": observation.source_id,
        "observation_date": observation.observation_date.isoformat(),
        "processing_level": observation.processing_level,
        "platform": observation.platform,
        "cloud_cover_percent": observation.cloud_cover_percent,
        "resolution_m": observation.resolution_m,
        "bands_used": [b.value for b in scene.bands],
        "license": observation.license,
    }


def _quality_block(
    scene_a: SceneStack, scene_b: SceneStack | None, change: ChangeDetectionResult | None
) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "period_a_masked_fraction": round(scene_a.masked_fraction, 4),
        "period_a_scene_classification": scene_a.scl_histogram,
    }
    if scene_b is not None:
        quality["period_b_masked_fraction"] = round(scene_b.masked_fraction, 4)
        quality["period_b_scene_classification"] = scene_b.scl_histogram
        days = abs(
            (scene_b.observation.observation_date - scene_a.observation.observation_date).days
        )
        quality["days_between_observations"] = days
    if change is not None:
        quality["comparable_pixel_fraction"] = round(
            change.comparable_pixel_count
            / max(1, change.difference.height * change.difference.width),
            4,
        )
    return quality


def _conditional_limitations(
    quality: dict[str, Any],
    land_cover: LandCoverResult | None,
    change: ChangeDetectionResult | None,
) -> list[str]:
    """Limitations that only apply given what this particular run observed."""
    notes: list[str] = []

    for period in ("a", "b"):
        masked = quality.get(f"period_{period}_masked_fraction")
        if masked is not None and masked > 0.20:
            notes.append(
                f"{masked * 100:.1f}% of period {period.upper()} was removed by cloud and "
                "quality masking, so its statistics describe only the remaining "
                "clear-sky portion of the region."
            )

    days = quality.get("days_between_observations")
    if isinstance(days, int):
        if days < 30:
            notes.append(
                f"The two observations are only {days} days apart, which is short "
                "relative to vegetation change and leaves the comparison sensitive "
                "to short-term conditions."
            )
        elif days % 365 > 60 and days % 365 < 305:
            notes.append(
                f"The observations are {days} days apart and are not seasonally "
                "matched, so part of any difference reflects phenology rather than "
                "persistent change."
            )

    comparable = quality.get("comparable_pixel_fraction")
    if comparable is not None and comparable < 0.60:
        notes.append(
            f"Only {comparable * 100:.1f}% of the grid was valid in both periods, "
            "so the change statistics cover a subset of the selected region."
        )

    if land_cover is not None:
        if land_cover.model_metadata.overall_accuracy is None:
            notes.append(
                "The land-cover model in use has no recorded held-out evaluation, "
                "so its accuracy on this region is unknown."
            )
        if (
            land_cover.low_confidence_fraction is not None
            and land_cover.low_confidence_fraction > 0.25
        ):
            notes.append(
                f"{land_cover.low_confidence_fraction * 100:.1f}% of pixels were "
                "classified with confidence below 0.50, indicating substantial "
                "spectral ambiguity in this region."
            )

    if (
        change is not None
        and change.absolute_change is not None
        and abs(change.absolute_change) < change.thresholds.moderate
    ):
        notes.append(
            "The mean index change is smaller than the moderate-change "
            "threshold, so the regional average is not distinguishable from "
            "measurement and processing noise."
        )
    return notes
