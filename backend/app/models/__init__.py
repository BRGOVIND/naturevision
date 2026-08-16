"""Persistence models.

Geometry is stored twice on purpose: as canonical GeoJSON in a portable JSON
column, and — on PostgreSQL — additionally as a PostGIS ``geometry`` column so
regions can be queried spatially (intersection, containment, nearest). The
JSON copy keeps the schema usable on SQLite for tests and keeps the exact
client-submitted coordinates as submitted.
"""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import settings


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _geometry_column(nullable: bool = True):
    """PostGIS geometry column on PostgreSQL, omitted elsewhere.

    Returning a plain JSON column on other dialects would duplicate
    ``geometry_geojson``; instead the spatial column simply does not exist and
    spatial queries are gated on the dialect.
    """
    if settings.is_postgres:
        from geoalchemy2 import Geometry

        return mapped_column(
            Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=True), nullable=nullable
        )
    return mapped_column(JSON, nullable=True)


class AnalysisStatus(StrEnum):
    """Lifecycle states of an analysis run."""

    CREATED = "created"
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    INTERPRETING = "interpreting"
    REPORT_READY = "report_ready"
    FAILED = "failed"


#: States from which no further transition happens.
TERMINAL_STATUSES = {AnalysisStatus.REPORT_READY, AnalysisStatus.FAILED}

#: Ordered progression used to render determinate progress in the client.
STATUS_SEQUENCE: tuple[AnalysisStatus, ...] = (
    AnalysisStatus.CREATED,
    AnalysisStatus.SEARCHING,
    AnalysisStatus.ACQUIRING,
    AnalysisStatus.PROCESSING,
    AnalysisStatus.ANALYZING,
    AnalysisStatus.INTERPRETING,
    AnalysisStatus.REPORT_READY,
)


class Region(Base):
    """A reusable geographic area of interest."""

    __tablename__ = "regions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str | None] = mapped_column(String(200))
    geometry_geojson: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    geom = _geometry_column()
    bbox_west: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_south: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_east: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_north: Mapped[float] = mapped_column(Float, nullable=False)
    area_km2: Mapped[float] = mapped_column(Float, nullable=False)
    crs: Mapped[str] = mapped_column(String(32), default="EPSG:4326", nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analyses: Mapped[list[Analysis]] = relationship(back_populates="region")

    __table_args__ = (
        Index("ix_regions_bbox", "bbox_west", "bbox_south", "bbox_east", "bbox_north"),
    )


class Analysis(Base):
    """One end-to-end environmental analysis run."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    region_id: Mapped[str] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default=AnalysisStatus.CREATED, index=True)
    status_detail: Mapped[str | None] = mapped_column(String(400))
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    period_a_start: Mapped[dt.date] = mapped_column(Date, nullable=False)
    period_a_end: Mapped[dt.date] = mapped_column(Date, nullable=False)
    period_b_start: Mapped[dt.date | None] = mapped_column(Date)
    period_b_end: Mapped[dt.date | None] = mapped_column(Date)

    max_cloud_cover: Mapped[float] = mapped_column(Float, default=40.0)
    include_land_cover: Mapped[bool] = mapped_column(Boolean, default=True)
    include_interpretation: Mapped[bool] = mapped_column(Boolean, default=True)

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    methodology: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    layer_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    region: Mapped[Region] = relationship(back_populates="analyses", lazy="selectin")
    observations: Mapped[list[SatelliteObservationRecord]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )
    metrics: Mapped[list[EnvironmentalMetric]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )
    predictions: Mapped[list[ModelPrediction]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", lazy="selectin"
    )


class SatelliteObservationRecord(Base):
    """A catalogued acquisition that was actually used by an analysis."""

    __tablename__ = "satellite_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    period: Mapped[str] = mapped_column(String(1), nullable=False)  # "A" or "B"

    source_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(200), nullable=False)
    dataset: Mapped[str] = mapped_column(String(120), nullable=False)
    observation_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    acquisition_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    cloud_cover_percent: Mapped[float | None] = mapped_column(Float)
    processing_level: Mapped[str | None] = mapped_column(String(32))
    platform: Mapped[str | None] = mapped_column(String(64))
    instrument: Mapped[str | None] = mapped_column(String(64))
    crs: Mapped[str | None] = mapped_column(String(32))
    resolution_m: Mapped[float | None] = mapped_column(Float)
    license: Mapped[str | None] = mapped_column(String(200))
    bands_used: Mapped[list[Any]] = mapped_column(JSON, default=list)
    footprint_geojson: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    geom = _geometry_column()
    scene_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    analysis: Mapped[Analysis] = relationship(back_populates="observations")


class EnvironmentalMetric(Base):
    """A single deterministic measurement produced by the processing pipeline.

    Stored as narrow rows rather than wide columns so new indices and metrics
    can be added without a migration, and so every value carries its own
    provenance and units.
    """

    __tablename__ = "environmental_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(40))
    period: Mapped[str | None] = mapped_column(String(1))
    category: Mapped[str] = mapped_column(String(40), default="vegetation")
    #: "observed" for deterministic processing output, "model_prediction" for
    #: anything produced by a learned model. The distinction is preserved all
    #: the way to the report.
    provenance: Mapped[str] = mapped_column(String(24), default="observed")
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped[Analysis] = relationship(back_populates="metrics")

    __table_args__ = (Index("ix_metric_analysis_key", "analysis_id", "key"),)


class ModelPrediction(Base):
    """Output of a learned model, with the metadata needed to audit it."""

    __tablename__ = "model_predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_backend: Mapped[str] = mapped_column(String(60), nullable=False)
    task: Mapped[str] = mapped_column(String(60), default="land_cover_classification")
    predicted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    class_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    mean_confidence: Mapped[float | None] = mapped_column(Float)
    low_confidence_fraction: Mapped[float | None] = mapped_column(Float)
    evaluation_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    input_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    preprocessing_version: Mapped[str | None] = mapped_column(String(40))
    prediction_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    analysis: Mapped[Analysis] = relationship(back_populates="predictions")


class Report(Base):
    """A generated Nature Intelligence Report."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    sections: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    interpretation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    visual_interpretation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    interpretation_provider: Mapped[str | None] = mapped_column(String(120))
    interpretation_model: Mapped[str | None] = mapped_column(String(120))
    html_path: Mapped[str | None] = mapped_column(String(500))
    generated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped[Analysis] = relationship(back_populates="reports")


__all__ = [
    "STATUS_SEQUENCE",
    "TERMINAL_STATUSES",
    "Analysis",
    "AnalysisStatus",
    "Base",
    "EnvironmentalMetric",
    "ModelPrediction",
    "Region",
    "Report",
    "SatelliteObservationRecord",
]
