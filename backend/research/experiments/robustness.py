"""Observation-quality and threshold-sensitivity experiments.

Both operate on real Sentinel-2 scenes rather than the cached pixel table,
because both are about properties of the imagery and the analysis pipeline
rather than about the classifier.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import numpy as np

from app.analysis.change import ChangeThresholds, detect_change
from app.analysis.indices import compute_ndvi
from app.analysis.statistics import compute_statistics
from app.core.logging import get_logger
from app.geospatial.geometry import validate_region
from app.imagery.bands import NDVI_BANDS, Band
from app.imagery.service import ImageryService
from app.imagery.stac import SentinelHubStacProvider
from research import figures
from research.artifacts import ExperimentRecord, write_table
from research.config import CONFIG, PERIODS
from training.regions import TRAINING_REGIONS

logger = get_logger(__name__)

#: Regions used for the imagery-level experiments. A small set keeps the run
#: feasible; each is a real scene search over a real area.
ROBUSTNESS_REGIONS = ("western_ghats", "po_valley", "brandenburg")

#: Minimum observations required before a cloud bucket is reported at all.
MIN_OBSERVATIONS_PER_BUCKET = 2


async def _gather_scene_quality() -> list[dict[str, Any]]:
    """Search real observations and measure NDVI stability against cloud cover."""
    provider = SentinelHubStacProvider()
    service = ImageryService(provider)
    rows: list[dict[str, Any]] = []

    try:
        for region_key in ROBUSTNESS_REGIONS:
            region_def = next(r for r in TRAINING_REGIONS if r.key == region_key)
            validated = validate_region(bbox=list(region_def.bbox))
            start = dt.date(2021, 1, 1)
            end = dt.date(2021, 12, 31)

            # A permissive search so higher-cloud buckets can actually populate.
            observations = await service.search(
                validated, start, end, max_cloud_cover=30.0, limit=40, required_bands=NDVI_BANDS
            )
            logger.info("quality_search", region=region_key, observations=len(observations))

            # Cap per region so the run stays bounded; ordered by cloud cover so
            # the sample spans the buckets rather than clustering at one end.
            candidates = sorted(
                (o for o in observations if o.cloud_cover_percent is not None),
                key=lambda o: float(o.cloud_cover_percent or 0.0),
            )
            step = max(1, len(candidates) // 12)
            for observation in candidates[::step][:12]:
                try:
                    scene = await service.load_scene(
                        observation,
                        validated,
                        (*NDVI_BANDS, Band.SCENE_CLASSIFICATION),
                        max_dimension=256,
                    )
                except Exception as exc:
                    logger.warning(
                        "scene_skipped", scene=observation.source_id, error=str(exc)[:160]
                    )
                    continue

                ndvi = compute_ndvi(scene)
                stats = compute_statistics(ndvi)
                rows.append(
                    {
                        "region": region_key,
                        "scene_id": observation.source_id,
                        "observation_date": observation.observation_date.isoformat(),
                        "cloud_cover_percent": round(
                            float(observation.cloud_cover_percent or 0.0), 4
                        ),
                        "masked_fraction": round(scene.masked_fraction, 4),
                        "valid_pixel_fraction": round(stats.valid_fraction, 4),
                        "valid_pixels": stats.valid_pixel_count,
                        "mean_ndvi": stats.mean,
                        "median_ndvi": stats.median,
                        "std_ndvi": stats.std_dev,
                    }
                )
    finally:
        await provider.close()
    return rows


def run_cloud_robustness() -> ExperimentRecord:
    """Relate scene cloud cover to usable pixels and NDVI stability."""
    rows = asyncio.run(_gather_scene_quality())
    if not rows:
        raise RuntimeError(
            "No observations could be retrieved for the cloud robustness experiment."
        )

    bucket_rows: list[dict[str, Any]] = []
    for name, low, high in CONFIG.cloud_buckets:
        members = [r for r in rows if low <= r["cloud_cover_percent"] < high]
        if len(members) < MIN_OBSERVATIONS_PER_BUCKET:
            # Reported as skipped rather than filled with fabricated observations.
            bucket_rows.append(
                {
                    "bucket": name,
                    "n_observations": len(members),
                    "status": "insufficient real observations",
                    "mean_valid_pixel_fraction": None,
                    "mean_masked_fraction": None,
                    "mean_ndvi": None,
                    "std_of_mean_ndvi": None,
                }
            )
            continue
        valid = np.array([m["valid_pixel_fraction"] for m in members], dtype=float)
        masked = np.array([m["masked_fraction"] for m in members], dtype=float)
        means = np.array(
            [m["mean_ndvi"] for m in members if m["mean_ndvi"] is not None], dtype=float
        )
        bucket_rows.append(
            {
                "bucket": name,
                "n_observations": len(members),
                "status": "reported",
                "mean_valid_pixel_fraction": round(float(valid.mean()), 4),
                "mean_masked_fraction": round(float(masked.mean()), 4),
                "mean_ndvi": round(float(means.mean()), 4) if means.size else None,
                "std_of_mean_ndvi": (
                    round(float(means.std(ddof=1)), 4) if means.size > 1 else None
                ),
            }
        )

    reported = [b for b in bucket_rows if b["status"] == "reported"]
    record = ExperimentRecord(
        experiment="cloud_robustness",
        seeds=[],
        config={
            "regions": list(ROBUSTNESS_REGIONS),
            "buckets": [list(b) for b in CONFIG.cloud_buckets],
            "min_observations_per_bucket": MIN_OBSERVATIONS_PER_BUCKET,
            "search_window": "2021-01-01..2021-12-31",
            "max_cloud_cover_searched": 30.0,
        },
        dataset_manifest={},
        results={
            "observations": rows,
            "buckets": bucket_rows,
            "n_observations": len(rows),
            "hypothesis": (
                "H6: increasing cloud/observation degradation reduces the reliability "
                "of environmental analysis."
            ),
        },
        notes=[
            "Buckets containing fewer than the minimum number of real observations "
            "are reported as skipped rather than populated.",
        ],
        limitations=[
            "Scene-level cloud cover describes the whole tile, not the analysed "
            "region, so a low-cloud scene can still be cloudy over the area of "
            "interest and vice versa.",
            "NDVI variation across dates includes genuine seasonal change, so a "
            "bucket difference is not attributable to cloud alone.",
        ],
    )

    write_table(rows, "table07a_cloud_observations", "cloud_robustness")
    write_table(bucket_rows, "table07_cloud_buckets", "cloud_robustness")
    if reported:
        figures.grouped_metric_bars(
            [b["bucket"] for b in reported],
            {
                "Valid pixel fraction": [b["mean_valid_pixel_fraction"] for b in reported],
                "Masked fraction": [b["mean_masked_fraction"] for b in reported],
            },
            title="Observation quality by scene cloud-cover bucket",
            ylabel="Fraction of region",
            name="fig07_cloud_robustness",
            experiment="cloud_robustness",
        )
    record.write()
    return record


async def _threshold_scan() -> list[dict[str, Any]]:
    """Run change detection over real period pairs at several thresholds."""
    provider = SentinelHubStacProvider()
    service = ImageryService(provider)
    rows: list[dict[str, Any]] = []

    try:
        for region_key in ROBUSTNESS_REGIONS:
            region_def = next(r for r in TRAINING_REGIONS if r.key == region_key)
            validated = validate_region(bbox=list(region_def.bbox))
            scenes = {}
            for period in PERIODS:
                start, end = period.window()
                observations = await service.search(
                    validated, start, end, max_cloud_cover=20.0, required_bands=NDVI_BANDS
                )
                if not observations:
                    logger.warning("no_scene", region=region_key, period=period.key)
                    continue
                best = service.select_best(observations, NDVI_BANDS)
                scenes[period.key] = await service.load_scene(
                    best, validated, (*NDVI_BANDS, Band.SCENE_CLASSIFICATION), max_dimension=384
                )
            if len(scenes) < 2:
                continue

            keys = sorted(scenes)
            ndvi_a = compute_ndvi(scenes[keys[0]])
            ndvi_b = compute_ndvi(scenes[keys[1]])

            for moderate in CONFIG.change_thresholds:
                significant = round(moderate * 2, 4)
                result = detect_change(
                    ndvi_a,
                    ndvi_b,
                    thresholds=ChangeThresholds(moderate=moderate, significant=significant),
                )
                classes = {c.label: c.percentage_of_analysed_area for c in result.class_summaries}
                rows.append(
                    {
                        "region": region_key,
                        "period_a": keys[0],
                        "period_b": keys[1],
                        "scene_a": scenes[keys[0]].observation.source_id,
                        "scene_b": scenes[keys[1]].observation.source_id,
                        "moderate_threshold": moderate,
                        "significant_threshold": significant,
                        "changed_area_percent": result.changed_area_percentage,
                        "changed_area_km2": result.changed_area_km2,
                        "decreased_percent": result.decreased_area_percentage,
                        "increased_percent": result.increased_area_percentage,
                        "stable_percent": classes.get("Stable"),
                        "moderate_decrease_percent": classes.get("Moderate decrease"),
                        "significant_decrease_percent": classes.get("Significant decrease"),
                        "moderate_increase_percent": classes.get("Moderate increase"),
                        "significant_increase_percent": classes.get("Significant increase"),
                        "comparable_pixels": result.comparable_pixel_count,
                        "mean_ndvi_change": result.absolute_change,
                    }
                )
                logger.info(
                    "threshold_scanned",
                    region=region_key,
                    moderate=moderate,
                    changed=result.changed_area_percentage,
                )
    finally:
        await provider.close()
    return rows


def run_threshold_sensitivity() -> ExperimentRecord:
    """Measure how the reported changed area responds to the threshold choice."""
    rows = asyncio.run(_threshold_scan())
    if not rows:
        raise RuntimeError("No period pairs could be retrieved for the threshold experiment.")

    regions = sorted({r["region"] for r in rows})
    thresholds = sorted({r["moderate_threshold"] for r in rows})
    series = {
        region: [
            next(
                (
                    r["changed_area_percent"]
                    for r in rows
                    if r["region"] == region and r["moderate_threshold"] == t
                ),
                float("nan"),
            )
            for t in thresholds
        ]
        for region in regions
    }

    record = ExperimentRecord(
        experiment="threshold_sensitivity",
        seeds=[],
        config={
            "thresholds_moderate": list(CONFIG.change_thresholds),
            "significant_rule": "significant = 2 x moderate",
            "regions": regions,
            "production_defaults_unchanged": {"moderate": 0.10, "significant": 0.20},
        },
        dataset_manifest={},
        results={"rows": rows, "changed_area_by_threshold": series},
        notes=[
            "This is a research-only sweep. The production defaults are not "
            "modified by this experiment.",
        ],
        limitations=[
            "Reported change is change in a reflectance-derived vegetation index. It "
            "does not establish land-cover conversion and no causal process is "
            "attributed to it.",
            "Only pixels valid in both periods contribute, so the analysed area "
            "differs between region pairs.",
        ],
    )

    write_table(rows, "table08_threshold_sensitivity", "threshold_sensitivity")
    figures.line_figure(
        thresholds,
        series,
        title="Changed area against moderate-change threshold",
        xlabel="Moderate threshold (NDVI units)",
        ylabel="Changed area (% of comparable pixels)",
        name="fig08_threshold_sensitivity",
        experiment="threshold_sensitivity",
    )
    record.write()
    return record
