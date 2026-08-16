"""Direct processing endpoints.

These run the real pipeline synchronously for a single step, which is what
makes them useful for validation, notebooks and integration testing. They
share the same services as the orchestrated analysis, so there is no second
implementation that could drift.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.analysis.change import ChangeThresholds, detect_change
from app.analysis.indices import NDVI, compute_ndvi
from app.analysis.statistics import compute_statistics, summarise_vegetation
from app.api.deps import ClassifierDep, ImageryDep
from app.core.logging import get_logger
from app.geospatial.geometry import ValidatedRegion, validate_region
from app.imagery.bands import LAND_COVER_BANDS, NDVI_BANDS, Band
from app.imagery.service import ImageryService
from app.schemas.analysis import (
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    LandCoverRequest,
    LandCoverResponse,
    NdviRequest,
    NdviResponse,
)
from app.schemas.common import RasterStatisticsSchema, RegionInput

logger = get_logger(__name__)
router = APIRouter(tags=["processing"])


def _region_from(payload: RegionInput) -> ValidatedRegion:
    return validate_region(
        geometry=payload.geometry.model_dump() if payload.geometry else None,
        bbox=payload.bbox,
        crs=payload.crs,
    )


async def _load(
    imagery: ImageryService,
    region: ValidatedRegion,
    start,
    end,
    max_cloud_cover: float,
    bands: tuple[Band, ...],
    observation_id: str | None = None,
):
    observations = await imagery.search(
        region, start, end, max_cloud_cover=max_cloud_cover, required_bands=bands
    )
    chosen = None
    if observation_id:
        chosen = next((o for o in observations if o.source_id == observation_id), None)
    observation = chosen or imagery.select_best(observations, bands)
    scene = await imagery.load_scene(observation, region, (*bands, Band.SCENE_CLASSIFICATION))
    return scene


@router.post("/ndvi", response_model=NdviResponse, summary="Compute NDVI for a region and period")
async def compute_ndvi_endpoint(payload: NdviRequest, imagery: ImageryDep) -> NdviResponse:
    """Retrieve imagery, mask it, compute NDVI and return real statistics."""
    region = _region_from(payload.region)
    scene = await _load(
        imagery,
        region,
        payload.period.start,
        payload.period.end,
        payload.max_cloud_cover,
        NDVI_BANDS,
        payload.observation_id,
    )
    ndvi = compute_ndvi(scene)
    statistics = compute_statistics(ndvi)

    return NdviResponse(
        region={
            "bbox": region.bbox.as_list(),
            "area_km2": round(region.area_km2, 4),
            "description": region.describe(),
        },
        observation=scene.provenance(),
        statistics=RasterStatisticsSchema(**statistics.to_dict()),
        vegetation_summary=summarise_vegetation(ndvi).to_dict(),
        grid=ndvi.metadata(),
        methodology={
            "index": NDVI.name,
            "formula": NDVI.formula,
            "bands": {"red": "B04", "nir": "B08"},
            "nodata_handling": (
                "Pixels masked by scene classification, with a near-zero "
                "denominator, or outside the achievable [-1, 1] range are excluded."
            ),
            "radiometric_calibration": scene.calibration.to_dict(),
        },
    )


@router.post(
    "/change-detection",
    response_model=ChangeDetectionResponse,
    summary="Compare vegetation index between two periods",
)
async def change_detection_endpoint(
    payload: ChangeDetectionRequest, imagery: ImageryDep
) -> ChangeDetectionResponse:
    """Run bi-temporal change detection on co-registered NDVI rasters."""
    region = _region_from(payload.region)
    scene_a = await _load(
        imagery,
        region,
        payload.period_a.start,
        payload.period_a.end,
        payload.max_cloud_cover,
        NDVI_BANDS,
    )
    scene_b = await _load(
        imagery,
        region,
        payload.period_b.start,
        payload.period_b.end,
        payload.max_cloud_cover,
        NDVI_BANDS,
    )

    thresholds = ChangeThresholds(
        **{
            k: v
            for k, v in (
                ("moderate", payload.moderate_threshold),
                ("significant", payload.significant_threshold),
            )
            if v is not None
        }
    )
    result = detect_change(compute_ndvi(scene_a), compute_ndvi(scene_b), thresholds=thresholds)

    return ChangeDetectionResponse(
        region={
            "bbox": region.bbox.as_list(),
            "area_km2": round(region.area_km2, 4),
            "description": region.describe(),
        },
        observations={"period_a": scene_a.provenance(), "period_b": scene_b.provenance()},
        result=result.to_dict(),
    )


@router.post(
    "/land-cover",
    response_model=LandCoverResponse,
    summary="Classify land cover for a region and period",
)
async def land_cover_endpoint(
    payload: LandCoverRequest, imagery: ImageryDep, classifier: ClassifierDep
) -> LandCoverResponse:
    """Run the trained classifier over a scene and return real class shares."""
    region = _region_from(payload.region)
    scene = await _load(
        imagery,
        region,
        payload.period.start,
        payload.period.end,
        payload.max_cloud_cover,
        LAND_COVER_BANDS,
        payload.observation_id,
    )
    result = classifier.classify_scene(scene)
    return LandCoverResponse(
        region={
            "bbox": region.bbox.as_list(),
            "area_km2": round(region.area_km2, 4),
            "description": region.describe(),
        },
        observation=scene.provenance(),
        result=result.to_dict(),
    )
