"""Shared request/response schema fragments."""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Longitude = Annotated[float, Field(ge=-180, le=180)]
Latitude = Annotated[float, Field(ge=-90, le=90)]


class SchemaBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class GeoJSONGeometry(SchemaBase):
    """A GeoJSON Polygon or MultiPolygon in WGS84."""

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class RegionInput(SchemaBase):
    """Spatial selector accepted from the client.

    Full validation (ordering, ranges, ring closure, self-intersection, extent)
    happens server-side in the geospatial layer; the checks here only reject
    obviously malformed payloads early.
    """

    geometry: GeoJSONGeometry | None = None
    bbox: list[float] | None = Field(
        default=None,
        description="[west, south, east, north] in EPSG:4326.",
        min_length=4,
        max_length=4,
    )
    crs: str | None = Field(default=None, description="Defaults to EPSG:4326.")
    name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _require_one_selector(self) -> RegionInput:
        if self.geometry is None and self.bbox is None:
            raise ValueError("Provide either 'geometry' or 'bbox'.")
        return self


class DateRange(SchemaBase):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.start > self.end:
            raise ValueError("'start' must not be after 'end'.")
        return self

    @property
    def label(self) -> str:
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


class RegionSummary(SchemaBase):
    id: str
    name: str | None
    geometry: dict[str, Any] = Field(alias="geometry_geojson")
    bbox: list[float]
    area_km2: float
    crs: str


class RasterStatisticsSchema(SchemaBase):
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    std_dev: float | None
    percentile_10: float | None = None
    percentile_90: float | None = None
    valid_pixel_count: int
    total_pixel_count: int
    valid_fraction: float
    valid_area_km2: float


class LayerReference(SchemaBase):
    """A raster overlay the client can request for the map."""

    key: str
    label: str
    kind: Literal["continuous", "categorical"]
    image_url: str
    bounds: list[float] = Field(description="[west, south, east, north] in EPSG:4326.")
    value_min: float | None = None
    value_max: float | None = None
    legend: list[dict[str, Any]] = Field(default_factory=list)
    units: str | None = None
    description: str | None = None
