"""Analysis lifecycle: persistence, background execution and result storage.

Rendered overlays are written to disk once, when the analysis completes, and
served as static files afterwards. Holding the raster stack in memory for the
lifetime of an analysis would not survive a restart and would not scale past a
handful of concurrent users; re-deriving it per map interaction would re-fetch
imagery. Rendering once and persisting the PNGs avoids both.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import NatureVisionError, ResourceNotFoundError
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.geospatial.geometry import ValidatedRegion, validate_region
from app.geospatial.render import (
    RenderedLayer,
    render_change,
    render_change_classes,
    render_confidence,
    render_continuous,
    render_land_cover,
    render_rgb_composite,
)
from app.imagery.bands import TRUE_COLOUR_BANDS, Band
from app.imagery.base import ImageryProvider
from app.imagery.service import ImageryService
from app.imagery.stac import SentinelHubStacProvider
from app.interpretation.evidence import EvidencePackage
from app.models import (
    Analysis,
    AnalysisStatus,
    EnvironmentalMetric,
    ModelPrediction,
    Region,
    SatelliteObservationRecord,
)
from app.models_ml.features import FEATURE_VERSION
from app.orchestration.pipeline import (
    AnalysisOrchestrator,
    AnalysisOutcome,
    AnalysisRequest,
    build_layer_manifest,
)

logger = get_logger(__name__)


def analysis_dir(analysis_id: str) -> Path:
    return settings.artifact_dir / analysis_id


def default_provider_factory() -> SentinelHubStacProvider:
    return SentinelHubStacProvider()


#: How a background analysis obtains its imagery provider. Indirection rather
#: than a hard-coded constructor, because the background task runs outside the
#: request scope and so cannot use FastAPI's dependency overrides; tests and
#: alternative deployments rebind this to supply a different source.
provider_factory: Callable[[], ImageryProvider] = default_provider_factory


class AnalysisService:
    """Creates, runs and reads back analyses."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # --- creation -------------------------------------------------------
    async def create(
        self,
        *,
        region: ValidatedRegion,
        region_name: str | None,
        period_a_start: dt.date,
        period_a_end: dt.date,
        period_b_start: dt.date | None,
        period_b_end: dt.date | None,
        max_cloud_cover: float,
        include_land_cover: bool,
        include_interpretation: bool,
    ) -> Analysis:
        region_row = Region(
            name=region_name,
            geometry_geojson=region.geojson,
            bbox_west=region.bbox.west,
            bbox_south=region.bbox.south,
            bbox_east=region.bbox.east,
            bbox_north=region.bbox.north,
            area_km2=region.area_km2,
            crs=region.crs,
        )
        if settings.is_postgres:
            from geoalchemy2.shape import from_shape

            region_row.geom = from_shape(region.geometry, srid=4326)

        analysis = Analysis(
            region=region_row,
            status=AnalysisStatus.CREATED,
            status_detail="Analysis queued",
            progress=0.0,
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_b_start,
            period_b_end=period_b_end,
            max_cloud_cover=max_cloud_cover,
            include_land_cover=include_land_cover,
            include_interpretation=include_interpretation,
        )
        self.session.add(region_row)
        self.session.add(analysis)
        await self.session.flush()
        await self.session.commit()
        logger.info("analysis_created", analysis_id=analysis.id, area_km2=region.area_km2)
        return analysis

    # --- retrieval -------------------------------------------------------
    async def get(self, analysis_id: str) -> Analysis:
        result = await self.session.execute(select(Analysis).where(Analysis.id == analysis_id))
        analysis = result.scalar_one_or_none()
        if analysis is None:
            raise ResourceNotFoundError(
                "No analysis exists with that identifier.", details={"analysis_id": analysis_id}
            )
        return analysis

    async def list(self, limit: int = 25, offset: int = 0) -> tuple[list[Analysis], int]:
        total = await self.session.scalar(select(func.count()).select_from(Analysis)) or 0
        result = await self.session.execute(
            select(Analysis).order_by(Analysis.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), int(total)

    async def delete(self, analysis_id: str) -> None:
        analysis = await self.get(analysis_id)
        await self.session.delete(analysis)
        await self.session.commit()


async def run_analysis_task(analysis_id: str) -> None:
    """Execute an analysis to completion in its own session and task.

    Runs detached from the originating request so a long acquisition does not
    hold an HTTP connection open. Every failure path records a terminal state
    with a user-safe message, so the client never polls a run that has silently
    stopped.
    """
    factory = get_session_factory()
    provider = provider_factory()

    try:
        async with factory() as session:
            service = AnalysisService(session)
            analysis = await service.get(analysis_id)
            region = validate_region(
                geometry=analysis.region.geometry_geojson, crs=analysis.region.crs
            )
            request = AnalysisRequest(
                region=region,
                region_name=analysis.region.name,
                period_a_start=analysis.period_a_start,
                period_a_end=analysis.period_a_end,
                period_b_start=analysis.period_b_start,
                period_b_end=analysis.period_b_end,
                max_cloud_cover=analysis.max_cloud_cover,
                include_land_cover=analysis.include_land_cover,
            )

            async def progress(status: AnalysisStatus, detail: str, fraction: float) -> None:
                analysis.status = status
                analysis.status_detail = detail
                analysis.progress = fraction
                analysis.updated_at = dt.datetime.now(dt.UTC)
                await session.commit()

            orchestrator = AnalysisOrchestrator(ImageryService(provider))
            outcome = await orchestrator.run(request, progress)

            await progress(AnalysisStatus.INTERPRETING, "Rendering map layers", 0.88)
            manifest = _persist_layers(analysis_id, outcome)

            _persist_results(session, analysis, outcome, manifest)
            analysis.status = AnalysisStatus.REPORT_READY
            analysis.status_detail = "Analysis complete"
            analysis.progress = 1.0
            analysis.completed_at = dt.datetime.now(dt.UTC)
            await session.commit()
            logger.info("analysis_completed", analysis_id=analysis_id)

    except NatureVisionError as exc:
        await _record_failure(factory, analysis_id, exc.code, exc.message)
    except Exception as exc:
        logger.exception("analysis_unhandled_failure", analysis_id=analysis_id)
        await _record_failure(
            factory,
            analysis_id,
            "internal_error",
            "The analysis failed unexpectedly. The incident has been logged.",
        )
        del exc
    finally:
        await provider.close()


async def _record_failure(factory, analysis_id: str, code: str, message: str) -> None:
    try:
        async with factory() as session:
            analysis = await session.get(Analysis, analysis_id)
            if analysis is None:
                return
            analysis.status = AnalysisStatus.FAILED
            analysis.status_detail = message
            analysis.error_code = code
            analysis.error_message = message
            analysis.updated_at = dt.datetime.now(dt.UTC)
            await session.commit()
        logger.warning("analysis_failed", analysis_id=analysis_id, code=code)
    except Exception:
        logger.exception("analysis_failure_not_recorded", analysis_id=analysis_id)


# --- persistence helpers ----------------------------------------------------
def _persist_layers(analysis_id: str, outcome: AnalysisOutcome) -> dict[str, Any]:
    """Render every available overlay once and write it to the artifact store."""
    directory = analysis_dir(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, RenderedLayer] = {}

    def add(key: str, layer: RenderedLayer) -> None:
        (directory / f"{key}.png").write_bytes(layer.png)
        rendered[key] = layer

    add("ndvi_a", render_continuous(outcome.period_a.ndvi))
    true_colour_a = _true_colour(outcome, "a")
    if true_colour_a is not None:
        add("true_colour_a", true_colour_a)
    if outcome.period_b is not None:
        add("ndvi_b", render_continuous(outcome.period_b.ndvi))
        true_colour_b = _true_colour(outcome, "b")
        if true_colour_b is not None:
            add("true_colour_b", true_colour_b)
    if outcome.change is not None:
        add("change", render_change(outcome.change.difference))
        add("change_classes", render_change_classes(outcome.change.classification))
    if outcome.land_cover is not None:
        add("land_cover", render_land_cover(outcome.land_cover.classification))
        if outcome.land_cover.confidence is not None:
            add("confidence", render_confidence(outcome.land_cover.confidence))

    manifest = build_layer_manifest(outcome, analysis_id)
    for entry in manifest["layers"]:
        layer = rendered.get(entry["key"])
        if layer is None:
            continue
        entry.update(
            bounds=layer.bounds,
            legend=layer.legend,
            value_min=layer.value_min,
            value_max=layer.value_max,
        )
    manifest["layers"] = [e for e in manifest["layers"] if e["key"] in rendered]
    return manifest


def _true_colour(outcome: AnalysisOutcome, period: str) -> RenderedLayer | None:
    """Render a true-colour composite, or None when its bands were not loaded."""
    scene = (outcome.period_a if period == "a" else outcome.period_b).scene  # type: ignore[union-attr]
    if not all(band in scene.bands for band in TRUE_COLOUR_BANDS):
        return None
    return render_rgb_composite(scene.band(Band.RED), scene.band(Band.GREEN), scene.band(Band.BLUE))


def _persist_results(
    session: AsyncSession, analysis: Analysis, outcome: AnalysisOutcome, manifest: dict[str, Any]
) -> None:
    """Write observations, metrics and predictions for an completed run."""
    evidence: EvidencePackage | None = outcome.evidence
    analysis.evidence = evidence.to_dict() if evidence else None
    analysis.methodology = evidence.methodology if evidence else None
    analysis.layer_manifest = manifest

    periods = [("A", outcome.period_a)]
    if outcome.period_b is not None:
        periods.append(("B", outcome.period_b))

    for label, period in periods:
        observation = period.scene.observation
        session.add(
            SatelliteObservationRecord(
                analysis_id=analysis.id,
                period=label,
                source_id=observation.source_id,
                provider=observation.provider,
                dataset=observation.dataset,
                observation_date=observation.observation_date,
                acquisition_timestamp=observation.acquisition_timestamp,
                cloud_cover_percent=observation.cloud_cover_percent,
                processing_level=observation.processing_level,
                platform=observation.platform,
                instrument=observation.instrument,
                crs=observation.crs,
                resolution_m=observation.resolution_m,
                license=observation.license,
                bands_used=[b.value for b in period.scene.bands],
                footprint_geojson=observation.geometry,
                scene_metadata=period.scene.provenance(),
            )
        )

        statistics = period.statistics
        for key, label_text, value, unit in (
            ("mean_ndvi", "Mean NDVI", statistics.mean, "index"),
            ("median_ndvi", "Median NDVI", statistics.median, "index"),
            ("min_ndvi", "Minimum NDVI", statistics.minimum, "index"),
            ("max_ndvi", "Maximum NDVI", statistics.maximum, "index"),
            ("std_dev_ndvi", "NDVI standard deviation", statistics.std_dev, "index"),
            ("valid_area_km2", "Valid observed area", statistics.valid_area_km2, "km2"),
            (
                "vegetated_fraction",
                "Fraction with NDVI >= 0.20",
                period.vegetation.vegetated_fraction,
                "fraction",
            ),
        ):
            session.add(
                EnvironmentalMetric(
                    analysis_id=analysis.id,
                    key=key,
                    label=label_text,
                    value=value,
                    unit=unit,
                    period=label,
                    category="vegetation",
                    provenance="observed",
                )
            )

    if outcome.change is not None:
        change = outcome.change
        for key, label_text, value, unit in (
            ("ndvi_change", "NDVI change (B - A)", change.absolute_change, "index"),
            (
                "ndvi_change_percent",
                "Relative NDVI change",
                change.relative_change_percent,
                "percent",
            ),
            (
                "changed_area_percent",
                "Area changed beyond threshold",
                change.changed_area_percentage,
                "percent",
            ),
            ("changed_area_km2", "Area changed beyond threshold", change.changed_area_km2, "km2"),
            (
                "decreased_area_percent",
                "Area with decreasing index",
                change.decreased_area_percentage,
                "percent",
            ),
            (
                "increased_area_percent",
                "Area with increasing index",
                change.increased_area_percentage,
                "percent",
            ),
        ):
            session.add(
                EnvironmentalMetric(
                    analysis_id=analysis.id,
                    key=key,
                    label=label_text,
                    value=value,
                    unit=unit,
                    category="change",
                    provenance="observed",
                    details=change.thresholds.to_dict(),
                )
            )

    if outcome.land_cover is not None:
        land_cover = outcome.land_cover
        metadata = land_cover.model_metadata
        session.add(
            ModelPrediction(
                analysis_id=analysis.id,
                model_name=metadata.name,
                model_version=metadata.version,
                model_backend=metadata.backend,
                task="land_cover_classification",
                class_distribution=land_cover.distribution,
                mean_confidence=land_cover.mean_confidence,
                low_confidence_fraction=land_cover.low_confidence_fraction,
                evaluation_metrics={
                    "overall_accuracy": metadata.overall_accuracy,
                    "macro_f1": metadata.macro_f1,
                    "per_class_metrics": metadata.per_class_metrics,
                    "evaluation_protocol": metadata.evaluation_protocol,
                    "evaluation_samples": metadata.evaluation_samples,
                },
                input_metadata=land_cover.input_metadata,
                preprocessing_version=FEATURE_VERSION,
                prediction_metadata={
                    "prediction_timestamp": land_cover.prediction_timestamp,
                    "classified_pixel_count": land_cover.classified_pixel_count,
                    "classified_area_km2": land_cover.classified_area_km2,
                    "label_source": metadata.label_source,
                },
            )
        )
        for summary in land_cover.class_summaries:
            session.add(
                EnvironmentalMetric(
                    analysis_id=analysis.id,
                    key=f"land_cover_{summary.label.lower().replace(' / ', '_').replace(' ', '_')}",
                    label=f"{summary.label} share",
                    value=summary.percentage,
                    unit="percent",
                    category="land_cover",
                    provenance="model_prediction",
                    details={
                        "area_km2": summary.area_km2,
                        "mean_confidence": summary.mean_confidence,
                        "class_id": summary.class_id,
                    },
                )
            )

    if outcome.land_cover_unavailable_reason:
        analysis.status_detail = outcome.land_cover_unavailable_reason


def read_layer(analysis_id: str, layer_key: str) -> bytes:
    """Read a previously rendered overlay from the artifact store."""
    safe_key = "".join(c for c in layer_key if c.isalnum() or c in "_-")
    path = analysis_dir(analysis_id) / f"{safe_key}.png"
    if not path.is_file():
        raise ResourceNotFoundError(
            "That map layer was not produced for this analysis.",
            details={"layer": layer_key},
        )
    return path.read_bytes()


def write_evidence_snapshot(analysis_id: str, evidence: dict[str, Any]) -> None:
    """Persist the evidence package alongside the rendered layers."""
    directory = analysis_dir(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )


def schedule_analysis(analysis_id: str) -> asyncio.Task:
    """Start an analysis in the background and keep a reference to its task."""
    task = asyncio.create_task(run_analysis_task(analysis_id), name=f"analysis:{analysis_id}")
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


#: Strong references to running tasks; without this the event loop may garbage
#: collect a task mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task] = set()
