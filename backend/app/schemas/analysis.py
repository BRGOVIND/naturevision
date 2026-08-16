"""Analysis, metric and report schemas."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field, model_validator

from app.schemas.common import (
    DateRange,
    LayerReference,
    RasterStatisticsSchema,
    RegionInput,
    SchemaBase,
)


class AnalysisCreateRequest(SchemaBase):
    """Request to run a full environmental analysis."""

    region: RegionInput
    period_a: DateRange
    period_b: DateRange | None = Field(
        default=None,
        description=(
            "Second period. When omitted the analysis is single-date and no change detection runs."
        ),
    )
    max_cloud_cover: float = Field(default=40.0, ge=0.0, le=100.0)
    include_land_cover: bool = True
    include_interpretation: bool = True
    change_moderate_threshold: float | None = Field(default=None, gt=0, lt=2)
    change_significant_threshold: float | None = Field(default=None, gt=0, lt=2)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> AnalysisCreateRequest:
        moderate = self.change_moderate_threshold
        significant = self.change_significant_threshold
        if moderate is not None and significant is not None and moderate >= significant:
            raise ValueError(
                "'change_moderate_threshold' must be smaller than 'change_significant_threshold'."
            )
        return self


class MetricSchema(SchemaBase):
    key: str
    label: str
    value: float | None
    unit: str | None
    period: str | None
    category: str
    provenance: str
    details: dict[str, Any] | None = None


class ObservationRecordSchema(SchemaBase):
    period: str
    source_id: str
    provider: str
    dataset: str
    observation_date: dt.date
    acquisition_timestamp: dt.datetime | None
    cloud_cover_percent: float | None
    processing_level: str | None
    platform: str | None
    instrument: str | None
    crs: str | None
    resolution_m: float | None
    license: str | None
    bands_used: list[str] = Field(default_factory=list)
    scene_metadata: dict[str, Any] | None = None


class ModelPredictionSchema(SchemaBase):
    model_name: str
    model_version: str
    model_backend: str
    task: str
    predicted_at: dt.datetime
    class_distribution: dict[str, Any]
    mean_confidence: float | None
    low_confidence_fraction: float | None
    evaluation_metrics: dict[str, Any] | None
    preprocessing_version: str | None
    prediction_metadata: dict[str, Any] | None


class AnalysisStatusSchema(SchemaBase):
    """Lightweight polling payload for a running analysis."""

    id: str
    status: str
    status_detail: str | None
    progress: float
    error_code: str | None = None
    error_message: str | None = None
    updated_at: dt.datetime


class AnalysisSummarySchema(SchemaBase):
    """Row in the analysis history list."""

    id: str
    status: str
    status_detail: str | None
    progress: float
    region_name: str | None
    region_bbox: list[float]
    area_km2: float
    period_a: str
    period_b: str | None
    mean_ndvi_a: float | None = None
    ndvi_change: float | None = None
    created_at: dt.datetime
    completed_at: dt.datetime | None


class AnalysisDetailSchema(SchemaBase):
    """Complete analysis result."""

    id: str
    status: str
    status_detail: str | None
    progress: float
    error_code: str | None
    error_message: str | None
    region: dict[str, Any]
    period_a: str
    period_b: str | None
    max_cloud_cover: float
    include_land_cover: bool
    created_at: dt.datetime
    updated_at: dt.datetime
    completed_at: dt.datetime | None
    observations: list[ObservationRecordSchema] = Field(default_factory=list)
    metrics: list[MetricSchema] = Field(default_factory=list)
    predictions: list[ModelPredictionSchema] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    methodology: dict[str, Any] | None = None
    layers: list[LayerReference] = Field(default_factory=list)
    has_report: bool = False


class AnalysisListResponse(SchemaBase):
    total: int
    limit: int
    offset: int
    items: list[AnalysisSummarySchema]


# --- direct computation endpoints -------------------------------------------
class NdviRequest(SchemaBase):
    region: RegionInput
    period: DateRange
    max_cloud_cover: float = Field(default=40.0, ge=0.0, le=100.0)
    observation_id: str | None = None


class NdviResponse(SchemaBase):
    region: dict[str, Any]
    observation: dict[str, Any]
    statistics: RasterStatisticsSchema
    vegetation_summary: dict[str, Any]
    grid: dict[str, Any]
    methodology: dict[str, Any]


class ChangeDetectionRequest(SchemaBase):
    region: RegionInput
    period_a: DateRange
    period_b: DateRange
    max_cloud_cover: float = Field(default=40.0, ge=0.0, le=100.0)
    moderate_threshold: float | None = Field(default=None, gt=0, lt=2)
    significant_threshold: float | None = Field(default=None, gt=0, lt=2)


class ChangeDetectionResponse(SchemaBase):
    region: dict[str, Any]
    observations: dict[str, Any]
    result: dict[str, Any]


class LandCoverRequest(SchemaBase):
    region: RegionInput
    period: DateRange
    max_cloud_cover: float = Field(default=40.0, ge=0.0, le=100.0)
    observation_id: str | None = None


class LandCoverResponse(SchemaBase):
    region: dict[str, Any]
    observation: dict[str, Any]
    result: dict[str, Any]


class ReportRequest(SchemaBase):
    analysis_id: str
    include_visual_interpretation: bool = Field(
        default=False,
        description="Additionally ask a vision model to describe the rendered NDVI layer.",
    )
    regenerate: bool = Field(
        default=False, description="Replace an existing report for this analysis."
    )


class ReportResponse(SchemaBase):
    id: str
    analysis_id: str
    title: str
    generated_at: dt.datetime
    sections: list[dict[str, Any]]
    provenance_legend: dict[str, str]
    interpretation: dict[str, Any] | None
    visual_interpretation: dict[str, Any] | None
    interpretation_provider: str | None
    interpretation_model: str | None
    export_urls: dict[str, str] = Field(default_factory=dict)


class ModelInfoSchema(SchemaBase):
    """What the deployment knows about its installed models."""

    installed: bool
    models: list[dict[str, Any]] = Field(default_factory=list)
    active_backend: str
    feature_version: str
    classes: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(SchemaBase):
    status: str
    application: str
    environment: str
    version: str
    database: str
    imagery_provider: str
    land_cover_model: str
    interpretation: str
    checks: dict[str, Any] = Field(default_factory=dict)
