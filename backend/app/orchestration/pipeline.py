"""Analysis orchestration.

A single linear pipeline over deterministic services. There is no agent
framework and no message bus here: the stages are ordered, each one is a plain
function of the previous stage's output, and every transition reports progress
so the client can show real state instead of an unexplained spinner.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.analysis.change import ChangeDetectionResult, ChangeThresholds, detect_change
from app.analysis.indices import compute_ndvi
from app.analysis.statistics import (
    RasterStatistics,
    VegetationSummary,
    compute_statistics,
    summarise_vegetation,
)
from app.core.config import settings
from app.core.errors import ModelUnavailableError, NatureVisionError
from app.core.logging import get_logger
from app.geospatial.geometry import ValidatedRegion
from app.geospatial.raster import RasterGrid
from app.imagery.bands import LAND_COVER_BANDS, NDVI_BANDS, TRUE_COLOUR_BANDS, Band
from app.imagery.service import ImageryService, SceneStack
from app.interpretation.evidence import EvidencePackage, build_evidence
from app.models import AnalysisStatus
from app.models_ml.classifier import LandCoverClassifier, LandCoverResult

logger = get_logger(__name__)

ProgressCallback = Callable[[AnalysisStatus, str, float], Awaitable[None]]

#: Progress fraction attached to each lifecycle state.
STATUS_PROGRESS: dict[AnalysisStatus, float] = {
    AnalysisStatus.CREATED: 0.0,
    AnalysisStatus.SEARCHING: 0.10,
    AnalysisStatus.ACQUIRING: 0.30,
    AnalysisStatus.PROCESSING: 0.55,
    AnalysisStatus.ANALYZING: 0.75,
    AnalysisStatus.INTERPRETING: 0.90,
    AnalysisStatus.REPORT_READY: 1.0,
}


@dataclass(slots=True)
class PeriodResult:
    """Everything derived from one observation period."""

    label: str
    scene: SceneStack
    ndvi: RasterGrid
    statistics: RasterStatistics
    vegetation: VegetationSummary


@dataclass(slots=True)
class AnalysisOutcome:
    """The complete deterministic result of one analysis run."""

    region: ValidatedRegion
    region_name: str | None
    period_a: PeriodResult
    period_b: PeriodResult | None = None
    change: ChangeDetectionResult | None = None
    land_cover: LandCoverResult | None = None
    land_cover_unavailable_reason: str | None = None
    evidence: EvidencePackage | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_temporal(self) -> bool:
        return self.period_b is not None


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Inputs to one orchestrated run."""

    region: ValidatedRegion
    period_a_start: dt.date
    period_a_end: dt.date
    period_b_start: dt.date | None = None
    period_b_end: dt.date | None = None
    region_name: str | None = None
    max_cloud_cover: float = 40.0
    include_land_cover: bool = True
    change_thresholds: ChangeThresholds | None = None
    #: Explicit observation ids, when the user picked scenes from search results.
    observation_id_a: str | None = None
    observation_id_b: str | None = None

    @property
    def is_temporal(self) -> bool:
        return self.period_b_start is not None and self.period_b_end is not None

    def period_label(self, period: str) -> str:
        if period == "A":
            return f"{self.period_a_start.isoformat()} to {self.period_a_end.isoformat()}"
        if self.period_b_start and self.period_b_end:
            return f"{self.period_b_start.isoformat()} to {self.period_b_end.isoformat()}"
        return ""


class AnalysisOrchestrator:
    """Runs the environmental analysis pipeline end to end."""

    def __init__(
        self,
        imagery: ImageryService,
        classifier: LandCoverClassifier | None = None,
    ) -> None:
        self.imagery = imagery
        self.classifier = classifier or LandCoverClassifier()

    async def run(
        self, request: AnalysisRequest, progress: ProgressCallback | None = None
    ) -> AnalysisOutcome:
        async def report(status: AnalysisStatus, detail: str) -> None:
            logger.info("analysis_stage", status=str(status), detail=detail)
            if progress is not None:
                await progress(status, detail, STATUS_PROGRESS.get(status, 0.0))

        # Green and blue are always loaded even when only NDVI is requested, so
        # the true-colour composite of the analysed scene is available on the
        # map. Two extra windowed reads over the AOI is a small cost for being
        # able to see the imagery the numbers came from.
        required_bands = tuple(
            dict.fromkeys(
                (
                    *NDVI_BANDS,
                    *TRUE_COLOUR_BANDS,
                    *(LAND_COVER_BANDS if request.include_land_cover else ()),
                )
            )
        )

        # --- 1-3: search and select observations -------------------------
        await report(AnalysisStatus.SEARCHING, "Searching the satellite catalogue")
        observation_a = await self._select(
            request,
            request.period_a_start,
            request.period_a_end,
            required_bands,
            request.observation_id_a,
        )
        observation_b = None
        if request.is_temporal:
            observation_b = await self._select(
                request,
                request.period_b_start,  # type: ignore[arg-type]
                request.period_b_end,  # type: ignore[arg-type]
                required_bands,
                request.observation_id_b,
            )

        # --- 4-5: retrieve and preprocess imagery -------------------------
        await report(
            AnalysisStatus.ACQUIRING,
            f"Retrieving {'two observations' if observation_b else 'one observation'}",
        )
        scene_bands = (*required_bands, Band.SCENE_CLASSIFICATION)
        scene_a = await self.imagery.load_scene(observation_a, request.region, scene_bands)
        scene_b = (
            await self.imagery.load_scene(observation_b, request.region, scene_bands)
            if observation_b is not None
            else None
        )

        # --- 6: spectral indices ------------------------------------------
        await report(AnalysisStatus.PROCESSING, "Computing vegetation index")
        period_a = _build_period(request.period_label("A"), scene_a)
        period_b = _build_period(request.period_label("B"), scene_b) if scene_b else None

        outcome = AnalysisOutcome(
            region=request.region,
            region_name=request.region_name,
            period_a=period_a,
            period_b=period_b,
        )

        # --- 7: temporal change --------------------------------------------
        if period_b is not None:
            await report(AnalysisStatus.ANALYZING, "Comparing periods and detecting change")
            outcome.change = detect_change(
                period_a.ndvi,
                period_b.ndvi,
                thresholds=request.change_thresholds or ChangeThresholds(),
            )

        # --- 8: land-cover classification -----------------------------------
        if request.include_land_cover:
            await report(AnalysisStatus.ANALYZING, "Running land-cover classification")
            try:
                # Classification uses the most recent scene, so the map reflects
                # the current state of the region rather than the baseline.
                target = scene_b or scene_a
                outcome.land_cover = self.classifier.classify_scene(target)
            except ModelUnavailableError as exc:
                # A missing model artifact degrades the analysis rather than
                # failing it: every measured result remains valid.
                outcome.land_cover_unavailable_reason = exc.message
                outcome.warnings.append(
                    "Land-cover classification was skipped because no trained model "
                    "is installed on this deployment."
                )
                logger.warning("land_cover_skipped", reason=exc.code)
            except NatureVisionError as exc:
                outcome.land_cover_unavailable_reason = exc.message
                outcome.warnings.append(f"Land-cover classification failed: {exc.message}")
                logger.error("land_cover_failed", reason=exc.code)

        # --- 9: assemble deterministic evidence -------------------------------
        outcome.evidence = build_evidence(
            region=request.region,
            region_name=request.region_name,
            scene_a=scene_a,
            ndvi_stats_a=period_a.statistics,
            vegetation_a=period_a.vegetation,
            scene_b=scene_b,
            ndvi_stats_b=period_b.statistics if period_b else None,
            vegetation_b=period_b.vegetation if period_b else None,
            change=outcome.change,
            land_cover=outcome.land_cover,
            period_a_label=request.period_label("A"),
            period_b_label=request.period_label("B"),
        )
        if outcome.warnings:
            outcome.evidence.limitations.extend(outcome.warnings)

        logger.info(
            "analysis_pipeline_complete",
            temporal=outcome.is_temporal,
            mean_ndvi_a=period_a.statistics.mean,
            mean_ndvi_b=period_b.statistics.mean if period_b else None,
            land_cover=outcome.land_cover is not None,
        )
        return outcome

    async def _select(
        self,
        request: AnalysisRequest,
        start: dt.date,
        end: dt.date,
        required_bands: tuple[Band, ...],
        explicit_id: str | None,
    ):
        observations = await self.imagery.search(
            request.region,
            start,
            end,
            max_cloud_cover=request.max_cloud_cover,
            required_bands=required_bands,
        )
        if explicit_id:
            chosen = next((o for o in observations if o.source_id == explicit_id), None)
            if chosen is not None:
                return chosen
            logger.warning("requested_observation_unavailable", source_id=explicit_id)
        return self.imagery.select_best(observations, required_bands)


def _build_period(label: str, scene: SceneStack) -> PeriodResult:
    ndvi = compute_ndvi(scene)
    return PeriodResult(
        label=label,
        scene=scene,
        ndvi=ndvi,
        statistics=compute_statistics(ndvi),
        vegetation=summarise_vegetation(ndvi),
    )


def build_layer_manifest(outcome: AnalysisOutcome, analysis_id: str) -> dict[str, Any]:
    """Describe the map overlays this analysis can serve.

    Only layers backed by a computed raster are listed, so the client never
    requests an overlay that does not exist.
    """
    prefix = f"{settings.api_prefix}/analysis/{analysis_id}/layers"
    has_true_colour = all(b in outcome.period_a.scene.bands for b in TRUE_COLOUR_BANDS)
    layers: list[dict[str, Any]] = []
    if has_true_colour:
        layers.append(
            {
                "key": "true_colour_a",
                "label": f"True colour ({outcome.period_a.scene.observation.observation_date})",
                "kind": "continuous",
                "image_url": f"{prefix}/true_colour_a",
                "description": "Sentinel-2 red/green/blue composite with a percentile stretch.",
            }
        )
    layers.append(
        {
            "key": "ndvi_a",
            "label": f"NDVI ({outcome.period_a.scene.observation.observation_date})",
            "kind": "continuous",
            "image_url": f"{prefix}/ndvi_a",
            "units": "index",
            "description": "Normalised difference vegetation index for period A.",
        }
    )
    if outcome.period_b is not None:
        if has_true_colour:
            layers.append(
                {
                    "key": "true_colour_b",
                    "label": f"True colour ({outcome.period_b.scene.observation.observation_date})",
                    "kind": "continuous",
                    "image_url": f"{prefix}/true_colour_b",
                    "description": (
                        "Sentinel-2 red/green/blue composite with a percentile stretch."
                    ),
                }
            )
        layers += [
            {
                "key": "ndvi_b",
                "label": f"NDVI ({outcome.period_b.scene.observation.observation_date})",
                "kind": "continuous",
                "image_url": f"{prefix}/ndvi_b",
                "units": "index",
                "description": "Normalised difference vegetation index for period B.",
            },
        ]
    if outcome.change is not None:
        layers += [
            {
                "key": "change",
                "label": "NDVI difference (B - A)",
                "kind": "continuous",
                "image_url": f"{prefix}/change",
                "units": "index difference",
                "description": "Signed per-pixel change in the vegetation index.",
            },
            {
                "key": "change_classes",
                "label": "Change classes",
                "kind": "categorical",
                "image_url": f"{prefix}/change_classes",
                "description": "Change magnitude classified against the configured thresholds.",
            },
        ]
    if outcome.land_cover is not None:
        layers.append(
            {
                "key": "land_cover",
                "label": "Land cover",
                "kind": "categorical",
                "image_url": f"{prefix}/land_cover",
                "description": "Per-pixel land-cover prediction.",
            }
        )
        if outcome.land_cover.confidence is not None:
            layers.append(
                {
                    "key": "confidence",
                    "label": "Model confidence",
                    "kind": "continuous",
                    "image_url": f"{prefix}/confidence",
                    "units": "probability",
                    "description": "Maximum class probability per pixel.",
                }
            )
    return {"layers": layers}
