"""Imagery search schemas."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field, model_validator

from app.schemas.common import RegionInput, SchemaBase


class ImagerySearchRequestSchema(SchemaBase):
    region: RegionInput
    start_date: dt.date
    end_date: dt.date
    max_cloud_cover: float = Field(default=40.0, ge=0.0, le=100.0)
    limit: int = Field(default=25, ge=1, le=200)
    require_analysis_bands: bool = Field(
        default=True,
        description="Only return observations publishing every band the pipeline needs.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> ImagerySearchRequestSchema:
        if self.start_date > self.end_date:
            raise ValueError("'start_date' must not be after 'end_date'.")
        return self


class ObservationSchema(SchemaBase):
    source_id: str
    provider: str
    dataset: str
    observation_date: dt.date
    acquisition_timestamp: dt.datetime | None
    cloud_cover_percent: float | None
    bbox: list[float]
    processing_level: str | None
    platform: str | None
    instrument: str | None
    crs: str | None
    resolution_m: float | None
    license: str | None
    bands: list[str]
    region_coverage: float | None
    properties: dict[str, Any] = Field(default_factory=dict)


class ImagerySearchResponse(SchemaBase):
    region: dict[str, Any]
    query: dict[str, Any]
    count: int
    observations: list[ObservationSchema]
