"""Scene assembly: turning catalogue entries into analysis-ready band stacks."""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from app.core.config import settings
from app.core.errors import InsufficientValidPixelsError, NoImageryFoundError
from app.core.logging import get_logger
from app.geospatial.geometry import ValidatedRegion
from app.geospatial.raster import RasterGrid
from app.imagery.bands import INVALID_SCL_CLASSES, SCL_LABELS, Band
from app.imagery.base import (
    ImageryProvider,
    ImagerySearchRequest,
    RadiometricCalibration,
    SatelliteObservation,
)

logger = get_logger(__name__)

#: Below this fraction of usable pixels the statistics are not trustworthy.
MIN_VALID_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class SceneStack:
    """Co-registered, cloud-masked reflectance bands for one observation.

    Every band shares the reference grid, so band arithmetic is index-safe
    without further checks.
    """

    observation: SatelliteObservation
    bands: dict[Band, RasterGrid]
    reference: RasterGrid
    scene_classification: RasterGrid | None
    region: ValidatedRegion
    masked_fraction: float
    scl_histogram: dict[str, float]
    calibration: RadiometricCalibration

    def band(self, band: Band) -> RasterGrid:
        try:
            return self.bands[band]
        except KeyError as exc:
            raise NoImageryFoundError(
                f"Band {band.value} is not present in the loaded scene.",
                details={"available": [b.value for b in self.bands]},
            ) from exc

    def stack(self, bands: tuple[Band, ...]) -> np.ma.MaskedArray:
        """(n_bands, height, width) masked cube in the requested band order."""
        arrays = [self.band(b).data for b in bands]
        return np.ma.stack(arrays, axis=0)

    @property
    def valid_fraction(self) -> float:
        return self.reference.valid_fraction

    def provenance(self) -> dict[str, Any]:
        return {
            **self.observation.to_metadata(),
            "grid": self.reference.metadata(),
            "cloud_masked_fraction": round(self.masked_fraction, 4),
            "scene_classification_distribution": self.scl_histogram,
            "radiometric_calibration": self.calibration.to_dict(),
        }


class ImageryService:
    """Search, rank and load satellite observations for a region."""

    def __init__(self, provider: ImageryProvider) -> None:
        self.provider = provider

    async def search(
        self,
        region: ValidatedRegion,
        start_date: dt.date,
        end_date: dt.date,
        *,
        max_cloud_cover: float | None = None,
        limit: int | None = None,
        required_bands: tuple[Band, ...] = (),
    ) -> list[SatelliteObservation]:
        request = ImagerySearchRequest(
            bbox=region.bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=(
                settings.max_cloud_cover_percent if max_cloud_cover is None else max_cloud_cover
            ),
            limit=limit or settings.max_search_results,
            required_bands=required_bands,
            geometry=region.geojson,
        )
        return await self.provider.search(request)

    @staticmethod
    def rank_observations(
        observations: list[SatelliteObservation],
    ) -> list[SatelliteObservation]:
        """Order candidates by usability: low cloud and full regional coverage.

        Coverage is weighted more heavily than cloud cover because a partially
        overlapping scene cannot be repaired, whereas moderate cloud is removed
        by the scene-classification mask.
        """

        def score(observation: SatelliteObservation) -> float:
            cloud = observation.cloud_cover_percent
            cloud_penalty = (cloud if cloud is not None else 50.0) / 100.0
            coverage = (
                observation.region_coverage if observation.region_coverage is not None else 0.5
            )
            return (1.0 - coverage) * 2.0 + cloud_penalty

        return sorted(observations, key=score)

    def select_best(
        self, observations: list[SatelliteObservation], required_bands: tuple[Band, ...]
    ) -> SatelliteObservation:
        usable = [o for o in observations if o.has_bands(required_bands)]
        if not usable:
            raise NoImageryFoundError(
                "No observation in the selected period publishes the bands required "
                "for this analysis.",
                details={"required_bands": [b.value for b in required_bands]},
            )
        best = self.rank_observations(usable)[0]
        logger.info(
            "observation_selected",
            source_id=best.source_id,
            date=best.observation_date.isoformat(),
            cloud=best.cloud_cover_percent,
            coverage=best.region_coverage,
        )
        return best

    async def load_scene(
        self,
        observation: SatelliteObservation,
        region: ValidatedRegion,
        bands: tuple[Band, ...],
        *,
        apply_cloud_mask: bool = True,
        max_dimension: int | None = None,
    ) -> SceneStack:
        """Read the requested bands, co-register them and remove unusable pixels."""
        limit = max_dimension or settings.target_raster_max_dim
        wanted = tuple(dict.fromkeys(bands))

        # Resolved once, before any band is read, so every band of this scene
        # shares one digital-number to reflectance convention.
        calibration = await self.provider.resolve_calibration(observation, region.bbox)

        grids = await asyncio.gather(
            *(
                self.provider.read_band(
                    observation,
                    band,
                    region.bbox,
                    max_dimension=limit,
                    calibration=calibration,
                )
                for band in wanted
            )
        )
        band_grids: dict[Band, RasterGrid] = dict(zip(wanted, grids, strict=True))

        # The finest-resolution band defines the analysis grid; coarser bands
        # (20 m SWIR, SCL) are resampled up onto it rather than the reverse, so
        # no 10 m detail is discarded.
        reference = min(band_grids.values(), key=lambda g: g.resolution[0] * g.resolution[1])

        scl_grid: RasterGrid | None = None
        scl_histogram: dict[str, float] = {}
        invalid_mask = np.zeros(reference.shape, dtype=bool)

        if apply_cloud_mask and Band.SCENE_CLASSIFICATION in observation.assets:
            raw_scl = await self.provider.read_band(
                observation,
                Band.SCENE_CLASSIFICATION,
                region.bbox,
                max_dimension=limit,
                calibration=calibration,
            )
            scl_grid = raw_scl.reproject_to(reference, Resampling.nearest)
            classes = np.rint(scl_grid.data.filled(0.0)).astype("int16")
            scl_histogram = _scl_distribution(classes)
            invalid_mask = np.isin(classes, list(INVALID_SCL_CLASSES))

        aligned: dict[Band, RasterGrid] = {}
        for band, grid in band_grids.items():
            warped = grid.reproject_to(reference, Resampling.bilinear)
            combined_mask = np.ma.getmaskarray(warped.data) | invalid_mask
            clipped = warped.with_data(
                np.ma.masked_array(warped.data.data, mask=combined_mask)
            ).clip_to_geometry(region.geometry)
            aligned[band] = clipped

        reference = aligned[
            min(aligned, key=lambda b: aligned[b].resolution[0] * aligned[b].resolution[1])
        ]

        in_region = _region_pixel_count(reference, region)
        masked_fraction = 1.0 - (reference.valid_count / in_region if in_region else 0.0)

        if reference.valid_count == 0 or reference.valid_fraction < MIN_VALID_FRACTION * (
            in_region / (reference.height * reference.width) if in_region else 1.0
        ):
            raise InsufficientValidPixelsError(
                "Almost every pixel in this observation was removed by cloud and "
                "quality masking. Widen the date range or raise the cloud-cover limit.",
                details={
                    "source_id": observation.source_id,
                    "valid_pixels": reference.valid_count,
                    "cloud_cover_percent": observation.cloud_cover_percent,
                },
            )

        logger.info(
            "scene_loaded",
            source_id=observation.source_id,
            bands=[b.value for b in aligned],
            shape=list(reference.shape),
            valid_pixels=reference.valid_count,
            masked_fraction=round(masked_fraction, 4),
        )

        return SceneStack(
            observation=observation,
            bands=aligned,
            reference=reference,
            scene_classification=scl_grid,
            region=region,
            masked_fraction=max(0.0, masked_fraction),
            scl_histogram=scl_histogram,
            calibration=calibration,
        )


def _scl_distribution(classes: np.ndarray) -> dict[str, float]:
    values, counts = np.unique(classes, return_counts=True)
    total = int(counts.sum())
    if total == 0:
        return {}
    return {
        SCL_LABELS.get(int(v), f"Class {int(v)}"): round(float(c) / total * 100.0, 2)
        for v, c in zip(values, counts, strict=True)
    }


def _region_pixel_count(grid: RasterGrid, region: ValidatedRegion) -> int:
    """Pixels falling inside the selection polygon, ignoring quality masking."""
    unmasked = grid.with_data(np.ma.masked_array(grid.data.data, mask=False))
    return unmasked.clip_to_geometry(region.geometry).valid_count
