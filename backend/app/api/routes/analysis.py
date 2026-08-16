"""Analysis lifecycle, history and map layers."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response, status

from app.api.deps import AnalysisServiceDep
from app.core.logging import get_logger
from app.geospatial.geometry import validate_region
from app.models import Analysis, AnalysisStatus
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisDetailSchema,
    AnalysisListResponse,
    AnalysisStatusSchema,
    AnalysisSummarySchema,
    MetricSchema,
    ModelPredictionSchema,
    ObservationRecordSchema,
)
from app.schemas.common import LayerReference
from app.services.analysis_service import read_layer, schedule_analysis

logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "",
    response_model=AnalysisStatusSchema,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and start an environmental analysis",
)
async def create_analysis(
    payload: AnalysisCreateRequest, service: AnalysisServiceDep
) -> AnalysisStatusSchema:
    """Validate the region, persist the analysis and start it in the background.

    Returns 202 immediately: a full run fetches and processes real imagery and
    takes tens of seconds, so the client polls the status endpoint rather than
    holding a request open.
    """
    region = validate_region(
        geometry=payload.region.geometry.model_dump() if payload.region.geometry else None,
        bbox=payload.region.bbox,
        crs=payload.region.crs,
    )
    analysis = await service.create(
        region=region,
        region_name=payload.region.name,
        period_a_start=payload.period_a.start,
        period_a_end=payload.period_a.end,
        period_b_start=payload.period_b.start if payload.period_b else None,
        period_b_end=payload.period_b.end if payload.period_b else None,
        max_cloud_cover=payload.max_cloud_cover,
        include_land_cover=payload.include_land_cover,
        include_interpretation=payload.include_interpretation,
    )
    schedule_analysis(analysis.id)
    return _status_schema(analysis)


@router.get("", response_model=AnalysisListResponse, summary="List previous analyses")
async def list_analyses(
    service: AnalysisServiceDep,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AnalysisListResponse:
    """Analysis history, newest first, from the database."""
    items, total = await service.list(limit=limit, offset=offset)
    return AnalysisListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_summary_schema(a) for a in items],
    )


@router.get(
    "/{analysis_id}/status",
    response_model=AnalysisStatusSchema,
    summary="Poll the progress of a running analysis",
)
async def analysis_status(analysis_id: str, service: AnalysisServiceDep) -> AnalysisStatusSchema:
    return _status_schema(await service.get(analysis_id))


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetailSchema,
    summary="Fetch a complete analysis result",
)
async def get_analysis(analysis_id: str, service: AnalysisServiceDep) -> AnalysisDetailSchema:
    analysis = await service.get(analysis_id)
    return _detail_schema(analysis)


@router.get(
    "/{analysis_id}/layers/{layer_key}",
    summary="Fetch a rendered map overlay",
    response_class=Response,
)
async def get_layer(analysis_id: str, layer_key: str, service: AnalysisServiceDep) -> Response:
    """Serve a PNG overlay rendered from this analysis's actual raster output."""
    await service.get(analysis_id)  # 404s for unknown analyses before touching disk
    return Response(
        content=read_layer(analysis_id, layer_key),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.delete(
    "/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete an analysis"
)
async def delete_analysis(analysis_id: str, service: AnalysisServiceDep) -> Response:
    await service.delete(analysis_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- serialisation ----------------------------------------------------------
def _status_schema(analysis: Analysis) -> AnalysisStatusSchema:
    return AnalysisStatusSchema(
        id=analysis.id,
        status=analysis.status,
        status_detail=analysis.status_detail,
        progress=analysis.progress,
        error_code=analysis.error_code,
        error_message=analysis.error_message,
        updated_at=analysis.updated_at,
    )


def _period_label(start, end) -> str | None:
    if start is None or end is None:
        return None
    return f"{start.isoformat()} to {end.isoformat()}"


def _metric_value(analysis: Analysis, key: str, period: str | None = None) -> float | None:
    for metric in analysis.metrics:
        if metric.key == key and (period is None or metric.period == period):
            return metric.value
    return None


def _summary_schema(analysis: Analysis) -> AnalysisSummarySchema:
    region = analysis.region
    return AnalysisSummarySchema(
        id=analysis.id,
        status=analysis.status,
        status_detail=analysis.status_detail,
        progress=analysis.progress,
        region_name=region.name,
        region_bbox=[region.bbox_west, region.bbox_south, region.bbox_east, region.bbox_north],
        area_km2=region.area_km2,
        period_a=_period_label(analysis.period_a_start, analysis.period_a_end) or "",
        period_b=_period_label(analysis.period_b_start, analysis.period_b_end),
        mean_ndvi_a=_metric_value(analysis, "mean_ndvi", "A"),
        ndvi_change=_metric_value(analysis, "ndvi_change"),
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
    )


def _detail_schema(analysis: Analysis) -> AnalysisDetailSchema:
    region = analysis.region
    manifest: dict[str, Any] = analysis.layer_manifest or {}
    layers = [LayerReference(**entry) for entry in manifest.get("layers", [])]

    return AnalysisDetailSchema(
        id=analysis.id,
        status=analysis.status,
        status_detail=analysis.status_detail,
        progress=analysis.progress,
        error_code=analysis.error_code,
        error_message=analysis.error_message,
        region={
            "id": region.id,
            "name": region.name,
            "geometry": region.geometry_geojson,
            "bbox": [region.bbox_west, region.bbox_south, region.bbox_east, region.bbox_north],
            "area_km2": region.area_km2,
            "crs": region.crs,
        },
        period_a=_period_label(analysis.period_a_start, analysis.period_a_end) or "",
        period_b=_period_label(analysis.period_b_start, analysis.period_b_end),
        max_cloud_cover=analysis.max_cloud_cover,
        include_land_cover=analysis.include_land_cover,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        completed_at=analysis.completed_at,
        observations=[ObservationRecordSchema.model_validate(o) for o in analysis.observations],
        metrics=[MetricSchema.model_validate(m) for m in analysis.metrics],
        predictions=[ModelPredictionSchema.model_validate(p) for p in analysis.predictions],
        evidence=analysis.evidence,
        methodology=analysis.methodology,
        layers=layers,
        has_report=bool(analysis.reports),
    )


__all__ = ["AnalysisStatus", "router"]
