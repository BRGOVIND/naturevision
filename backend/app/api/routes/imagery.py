"""Satellite catalogue search."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import ImageryDep
from app.core.logging import get_logger
from app.geospatial.geometry import validate_region
from app.imagery.bands import LAND_COVER_BANDS, NDVI_BANDS
from app.schemas.imagery import (
    ImagerySearchRequestSchema,
    ImagerySearchResponse,
    ObservationSchema,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/imagery", tags=["imagery"])


@router.post(
    "/search",
    response_model=ImagerySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Sentinel-2 observations for a region and period",
)
async def search_imagery(
    payload: ImagerySearchRequestSchema, imagery: ImageryDep
) -> ImagerySearchResponse:
    """Query the satellite catalogue and return usable observations.

    Results are ordered by usability (regional coverage first, then cloud
    cover) so the top entry is the one the pipeline would pick automatically.
    """
    region = validate_region(
        geometry=payload.region.geometry.model_dump() if payload.region.geometry else None,
        bbox=payload.region.bbox,
        crs=payload.region.crs,
    )
    required = (
        tuple(dict.fromkeys((*NDVI_BANDS, *LAND_COVER_BANDS)))
        if payload.require_analysis_bands
        else ()
    )
    observations = await imagery.search(
        region,
        payload.start_date,
        payload.end_date,
        max_cloud_cover=payload.max_cloud_cover,
        limit=payload.limit,
        required_bands=required,
    )
    ranked = imagery.rank_observations(observations)

    return ImagerySearchResponse(
        region={
            "bbox": region.bbox.as_list(),
            "area_km2": round(region.area_km2, 4),
            "description": region.describe(),
            "crs": region.crs,
        },
        query={
            "start_date": payload.start_date.isoformat(),
            "end_date": payload.end_date.isoformat(),
            "max_cloud_cover": payload.max_cloud_cover,
            "required_bands": [b.value for b in required],
            "provider": imagery.provider.name,
            "collection": imagery.provider.dataset,
        },
        count=len(ranked),
        observations=[
            ObservationSchema(
                source_id=o.source_id,
                provider=o.provider,
                dataset=o.dataset,
                observation_date=o.observation_date,
                acquisition_timestamp=o.acquisition_timestamp,
                cloud_cover_percent=o.cloud_cover_percent,
                bbox=o.bbox.as_list(),
                processing_level=o.processing_level,
                platform=o.platform,
                instrument=o.instrument,
                crs=o.crs,
                resolution_m=o.resolution_m,
                license=o.license,
                bands=[b.value for b in o.available_bands],
                region_coverage=o.region_coverage,
                properties=o.properties,
            )
            for o in ranked
        ],
    )
