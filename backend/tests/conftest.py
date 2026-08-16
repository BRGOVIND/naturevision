"""Shared fixtures.

Test data is synthetic and deterministic by design: it exercises the numerical
edge cases (nodata, zero denominators, misalignment) that real imagery only
produces occasionally. Nothing here is used by production code paths.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("GROQ_API_KEY", "")

from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from app.geospatial.geometry import BoundingBox, validate_region
from app.geospatial.raster import RasterGrid
from app.imagery.bands import LAND_COVER_BANDS, Band
from app.imagery.base import (
    BandAsset,
    ImageryProvider,
    ImagerySearchRequest,
    RadiometricCalibration,
    SatelliteObservation,
)
from app.imagery.service import ImageryService, SceneStack

UTM = CRS.from_epsg(32643)
WGS84 = CRS.from_epsg(4326)

#: A 10 m grid anchored in UTM zone 43N, matching the Sentinel-2 layout.
BASE_TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 1_130_000.0)


def make_grid(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    transform: Affine = BASE_TRANSFORM,
    crs: CRS = UTM,
) -> RasterGrid:
    array = np.ma.masked_array(
        values.astype("float32"),
        mask=mask if mask is not None else np.zeros(values.shape, dtype=bool),
    )
    return RasterGrid(data=array, transform=transform, crs=crs)


@pytest.fixture
def red_grid() -> RasterGrid:
    return make_grid(np.full((8, 8), 0.05, dtype="float32"))


@pytest.fixture
def nir_grid() -> RasterGrid:
    return make_grid(np.full((8, 8), 0.35, dtype="float32"))


@pytest.fixture
def region():
    """A region covering the synthetic UTM grid, expressed in WGS84."""
    return validate_region(bbox=[76.60, 10.20, 76.62, 10.22])


class StubProvider(ImageryProvider):
    """In-memory imagery provider with deterministic reflectance.

    Used to exercise the scene-assembly and analysis pipeline without network
    access. It is a test double only; the application always binds the real
    STAC provider.
    """

    name = "Test Provider"
    dataset = "test-collection"

    def __init__(
        self,
        band_values: dict[Band, float] | None = None,
        *,
        scl_values: np.ndarray | None = None,
        shape: tuple[int, int] = (8, 8),
        fail_on: Band | None = None,
    ) -> None:
        self.band_values = band_values or {
            Band.BLUE: 0.04,
            Band.GREEN: 0.06,
            Band.RED: 0.05,
            Band.NIR: 0.35,
            Band.SWIR_16: 0.20,
            Band.SWIR_22: 0.12,
        }
        self.scl_values = scl_values
        self.shape = shape
        self.fail_on = fail_on
        self.read_calls: list[Band] = []
        self.calibration_calls = 0

    async def search(self, request: ImagerySearchRequest) -> list[SatelliteObservation]:
        return [self.observation(request.bbox)]

    def observation(
        self, bbox: BoundingBox, source_id: str = "TEST_SCENE_1"
    ) -> SatelliteObservation:
        assets = {
            band: BandAsset(band=band, href=f"memory://{band.value}", asset_key=band.value)
            for band in (*LAND_COVER_BANDS, Band.SCENE_CLASSIFICATION)
        }
        return SatelliteObservation(
            source_id=source_id,
            provider=self.name,
            dataset=self.dataset,
            observation_date=dt.date(2021, 2, 7),
            acquisition_timestamp=dt.datetime(2021, 2, 7, 5, 26, tzinfo=dt.UTC),
            cloud_cover_percent=1.5,
            bbox=bbox,
            geometry=bbox.to_geojson(),
            processing_level="L2A",
            platform="sentinel-2b",
            instrument="msi",
            crs="EPSG:32643",
            resolution_m=10.0,
            license="CC-BY-4.0",
            assets=assets,
            properties={},
            region_coverage=1.0,
        )

    async def resolve_calibration(self, observation, bbox) -> RadiometricCalibration:
        self.calibration_calls += 1
        return RadiometricCalibration(offset=0.0, scale_source="test")

    async def read_band(
        self, observation, band, bbox, *, max_dimension=None, calibration=None
    ) -> RasterGrid:
        from app.core.errors import ImageryAcquisitionError

        if self.fail_on is not None and band is self.fail_on:
            raise ImageryAcquisitionError("Simulated band read failure.")
        self.read_calls.append(band)

        # Build a grid that genuinely covers the requested bbox, so clipping to
        # the region behaves the way it does against a real provider.
        west, south, east, north = transform_bounds(WGS84, UTM, *bbox.as_tuple(), densify_pts=21)
        transform = from_bounds(west, south, east, north, self.shape[1], self.shape[0])

        if band is Band.SCENE_CLASSIFICATION:
            values = (
                self.scl_values
                if self.scl_values is not None
                else np.full(self.shape, 4, dtype="float32")
            )
            return make_grid(values.astype("float32"), transform=transform)
        return make_grid(
            np.full(self.shape, self.band_values[band], dtype="float32"), transform=transform
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def imagery_service(stub_provider: StubProvider) -> ImageryService:
    return ImageryService(stub_provider)


@pytest.fixture
async def scene(imagery_service: ImageryService, region) -> SceneStack:
    observation = imagery_service.provider.observation(region.bbox)  # type: ignore[attr-defined]
    return await imagery_service.load_scene(
        observation, region, (*LAND_COVER_BANDS, Band.SCENE_CLASSIFICATION)
    )
