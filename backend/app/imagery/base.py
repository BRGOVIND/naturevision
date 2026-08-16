"""Provider-agnostic satellite imagery interface.

The analysis pipeline depends only on the abstractions in this module. Swapping
Element84 Earth Search for a different STAC API, a commercial provider or a
local archive means adding one :class:`ImageryProvider` implementation and
changing a single factory binding.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.geospatial.geometry import BoundingBox
from app.geospatial.raster import RasterGrid
from app.imagery.bands import Band


@dataclass(frozen=True, slots=True)
class BandAsset:
    """A single retrievable band of an observation."""

    band: Band
    href: str
    asset_key: str
    native_resolution_m: float | None = None
    scale: float = 1.0
    offset: float = 0.0
    nodata: float | None = None
    data_type: str | None = None


@dataclass(frozen=True, slots=True)
class SatelliteObservation:
    """One catalogued satellite acquisition intersecting the region of interest.

    Fields mirror what a downstream analyst needs to judge whether a scene is
    usable, plus the provenance required to reproduce the analysis later.
    """

    source_id: str
    provider: str
    dataset: str
    observation_date: dt.date
    acquisition_timestamp: dt.datetime | None
    cloud_cover_percent: float | None
    bbox: BoundingBox
    geometry: dict[str, Any] | None
    processing_level: str | None
    platform: str | None
    instrument: str | None
    crs: str | None
    resolution_m: float | None
    license: str | None
    assets: dict[Band, BandAsset] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    #: Fraction of the requested region covered by this scene footprint, 0-1.
    region_coverage: float | None = None

    @property
    def available_bands(self) -> list[Band]:
        return sorted(self.assets.keys(), key=lambda b: b.value)

    def has_bands(self, bands: tuple[Band, ...]) -> bool:
        return all(band in self.assets for band in bands)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "provider": self.provider,
            "dataset": self.dataset,
            "observation_date": self.observation_date.isoformat(),
            "acquisition_timestamp": (
                self.acquisition_timestamp.isoformat() if self.acquisition_timestamp else None
            ),
            "cloud_cover_percent": self.cloud_cover_percent,
            "bbox": self.bbox.as_list(),
            "processing_level": self.processing_level,
            "platform": self.platform,
            "instrument": self.instrument,
            "crs": self.crs,
            "resolution_m": self.resolution_m,
            "license": self.license,
            "bands": [b.value for b in self.available_bands],
            "region_coverage": self.region_coverage,
        }


@dataclass(frozen=True, slots=True)
class RadiometricCalibration:
    """The digital-number to surface-reflectance convention for one observation.

    Resolved once per observation and applied to every band, because a spectral
    index is only meaningful when all of its inputs share one radiometric
    convention. Mixing conventions between bands silently biases every index.
    """

    offset: float
    scale_source: str
    #: "catalogue" when the provider's declared offset was accepted,
    #: "physical_override" when it was rejected as implausible.
    decision: str = "catalogue"
    diagnostic: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflectance_offset": self.offset,
            "scale_source": self.scale_source,
            "decision": self.decision,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class ImagerySearchRequest:
    """Catalogue query parameters."""

    bbox: BoundingBox
    start_date: dt.date
    end_date: dt.date
    max_cloud_cover: float = 40.0
    limit: int = 50
    required_bands: tuple[Band, ...] = ()
    geometry: dict[str, Any] | None = None


class ImageryProvider(ABC):
    """Contract every imagery source must satisfy."""

    #: Human-readable provider name recorded in analysis provenance.
    name: str = "unknown"
    #: Dataset/collection identifier.
    dataset: str = "unknown"

    @abstractmethod
    async def search(self, request: ImagerySearchRequest) -> list[SatelliteObservation]:
        """Return observations intersecting the request, newest first."""

    async def resolve_calibration(
        self, observation: SatelliteObservation, bbox: BoundingBox
    ) -> RadiometricCalibration:
        """Determine the reflectance convention to use for every band of a scene.

        The default accepts whatever the catalogue declares. Providers whose
        metadata is unreliable should override this and verify it against pixels.
        """
        offsets = {
            asset.offset
            for asset in observation.assets.values()
            if asset.band is not Band.SCENE_CLASSIFICATION
        }
        return RadiometricCalibration(
            offset=next(iter(offsets), 0.0) if len(offsets) == 1 else 0.0,
            scale_source="catalogue",
        )

    @abstractmethod
    async def read_band(
        self,
        observation: SatelliteObservation,
        band: Band,
        bbox: BoundingBox,
        *,
        max_dimension: int | None = None,
        calibration: RadiometricCalibration | None = None,
    ) -> RasterGrid:
        """Read one band clipped to ``bbox``, returned as a georeferenced grid."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider-held network resources."""
