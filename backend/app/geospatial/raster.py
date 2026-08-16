"""Georeferenced raster container and spatial operations.

Every array that flows through the analysis pipeline is wrapped in
:class:`RasterGrid`, which carries the CRS, affine transform and nodata mask
alongside the pixel values. Losing that metadata mid-pipeline is the classic
source of silently misaligned remote-sensing results, so the operations here
always return a new grid with its georeferencing intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from app.core.errors import RasterProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """A single-band float raster with full spatial reference.

    ``data`` is a masked array: ``mask=True`` marks nodata / cloud-masked
    pixels. Keeping the mask attached rather than relying on a sentinel value
    avoids the divide-by-zero and NaN-propagation traps in index arithmetic.
    """

    data: np.ma.MaskedArray
    transform: Affine
    crs: CRS
    nodata: float = float("nan")

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise RasterProcessingError(
                "A raster grid must be two-dimensional.",
                details={"shape": list(self.data.shape)},
            )

    # --- geometry -------------------------------------------------------
    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def resolution(self) -> tuple[float, float]:
        """Pixel size in CRS units as (x, y), always positive."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        if self.crs == WGS84:
            return self.bounds
        return transform_bounds(self.crs, WGS84, *self.bounds, densify_pts=21)

    @property
    def valid_count(self) -> int:
        return int((~np.ma.getmaskarray(self.data)).sum())

    @property
    def valid_fraction(self) -> float:
        total = self.height * self.width
        return self.valid_count / total if total else 0.0

    def pixel_area_m2(self) -> float:
        """Approximate ground area of one pixel in square metres."""
        res_x, res_y = self.resolution
        if self.crs.is_projected:
            return res_x * res_y
        # Geographic CRS: convert degrees to metres at the raster centre latitude.
        _, south, _, north = self.bounds
        centre_lat = np.deg2rad((south + north) / 2.0)
        metres_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * centre_lat)
        metres_per_deg_lon = 111_412.84 * np.cos(centre_lat) - 93.5 * np.cos(3 * centre_lat)
        return float(res_x * metres_per_deg_lon * res_y * metres_per_deg_lat)

    def spatial_signature(self) -> tuple:
        """Identity of the grid, used to assert alignment between rasters."""
        return (self.shape, tuple(round(v, 9) for v in self.transform[:6]), self.crs.to_string())

    def is_aligned_with(self, other: RasterGrid) -> bool:
        return self.spatial_signature() == other.spatial_signature()

    # --- transformation --------------------------------------------------
    def with_data(self, data: np.ma.MaskedArray | np.ndarray, nodata: float | None = None) -> Self:
        masked = data if isinstance(data, np.ma.MaskedArray) else np.ma.masked_invalid(data)
        return replace(self, data=masked, nodata=self.nodata if nodata is None else nodata)

    def reproject_to(
        self, reference: RasterGrid, resampling: Resampling | None = None
    ) -> RasterGrid:
        """Warp this grid onto the reference grid so the two are pixel-aligned."""
        if self.is_aligned_with(reference):
            return self
        method = resampling or Resampling.bilinear
        destination = np.full(reference.shape, np.nan, dtype="float32")
        reproject(
            source=self.data.filled(np.nan).astype("float32"),
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=method,
        )
        return RasterGrid(
            data=np.ma.masked_invalid(destination),
            transform=reference.transform,
            crs=reference.crs,
            nodata=float("nan"),
        )

    def to_crs(self, dst_crs: CRS, resampling: Resampling | None = None) -> RasterGrid:
        """Reproject to a new CRS, letting rasterio choose the output grid."""
        if self.crs == dst_crs:
            return self
        transform, width, height = calculate_default_transform(
            self.crs, dst_crs, self.width, self.height, *self.bounds
        )
        destination = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=self.data.filled(np.nan).astype("float32"),
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling or Resampling.bilinear,
        )
        return RasterGrid(
            data=np.ma.masked_invalid(destination),
            transform=transform,
            crs=dst_crs,
            nodata=float("nan"),
        )

    def clip_to_geometry(
        self, geometry: BaseGeometry, geometry_crs: CRS | None = None
    ) -> RasterGrid:
        """Mask out everything outside ``geometry`` while keeping the grid intact.

        The extent is preserved deliberately: two periods clipped to the same
        polygon stay pixel-comparable, which cropping to the geometry envelope
        would not guarantee.
        """
        source_crs = geometry_crs or WGS84
        local_geometry = reproject_geometry(geometry, source_crs, self.crs)
        outside = geometry_mask(
            [mapping(local_geometry)],
            out_shape=self.shape,
            transform=self.transform,
            invert=False,  # True where the pixel is OUTSIDE the geometry
            all_touched=False,
        )
        combined = np.ma.masked_array(self.data.data, mask=np.ma.getmaskarray(self.data) | outside)
        return self.with_data(combined)

    def resampled_to_max_dimension(self, max_dim: int) -> RasterGrid:
        """Downsample so the longest edge is at most ``max_dim`` pixels."""
        longest = max(self.height, self.width)
        if longest <= max_dim:
            return self
        factor = longest / max_dim
        new_height = max(1, round(self.height / factor))
        new_width = max(1, round(self.width / factor))
        scale_x = self.width / new_width
        scale_y = self.height / new_height
        transform = self.transform * Affine.scale(scale_x, scale_y)
        destination = np.full((new_height, new_width), np.nan, dtype="float32")
        reproject(
            source=self.data.filled(np.nan).astype("float32"),
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=self.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        return RasterGrid(
            data=np.ma.masked_invalid(destination),
            transform=transform,
            crs=self.crs,
            nodata=float("nan"),
        )

    # --- serialisation ----------------------------------------------------
    def profile(self) -> dict[str, Any]:
        return {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True,
        }

    def write(self, path) -> None:
        with rasterio.open(path, "w", **self.profile()) as dst:
            dst.write(self.data.filled(-9999.0).astype("float32"), 1)

    def metadata(self) -> dict[str, Any]:
        west, south, east, north = self.bounds_wgs84()
        return {
            "crs": self.crs.to_string(),
            "width": self.width,
            "height": self.height,
            "resolution_x": self.resolution[0],
            "resolution_y": self.resolution[1],
            "bounds_wgs84": [west, south, east, north],
            "transform": list(self.transform[:6]),
            "valid_pixels": self.valid_count,
            "valid_fraction": round(self.valid_fraction, 6),
            "pixel_area_m2": round(self.pixel_area_m2(), 3),
        }


def reproject_geometry(geometry: BaseGeometry, src_crs: CRS, dst_crs: CRS) -> BaseGeometry:
    """Reproject a shapely geometry between CRSs."""
    if src_crs == dst_crs:
        return geometry
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geometry)


def window_for_bounds(
    dataset_transform: Affine,
    dataset_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    pad: int = 1,
) -> Window:
    """Compute a read window clamped to the dataset, so only the AOI is fetched.

    This is what keeps a range read over a remote cloud-optimised GeoTIFF to a
    few hundred kilobytes instead of the full ~100 MB scene band.
    """
    window = window_from_bounds(*bounds, transform=dataset_transform)
    height, width = dataset_shape
    col_off = math.floor(window.col_off) - pad
    row_off = math.floor(window.row_off) - pad
    col_end = math.ceil(window.col_off + window.width) + pad
    row_end = math.ceil(window.row_off + window.height) + pad

    col_off = max(0, min(col_off, width - 1))
    row_off = max(0, min(row_off, height - 1))
    col_end = max(col_off + 1, min(col_end, width))
    row_end = max(row_off + 1, min(row_end, height))
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def align_grids(*grids: RasterGrid, resampling: Resampling | None = None) -> list[RasterGrid]:
    """Warp every grid onto the first grid so that all share one pixel lattice."""
    if not grids:
        return []
    reference = grids[0]
    return [reference, *(g.reproject_to(reference, resampling) for g in grids[1:])]


def common_valid_mask(*grids: RasterGrid) -> np.ndarray:
    """Boolean array that is True only where every grid has a valid pixel."""
    if not grids:
        raise RasterProcessingError("At least one raster is required.")
    reference = grids[0]
    for grid in grids[1:]:
        if not grid.is_aligned_with(reference):
            raise RasterProcessingError(
                "Rasters must be spatially aligned before combining them.",
                details={
                    "reference": str(reference.spatial_signature()),
                    "other": str(grid.spatial_signature()),
                },
            )
    valid = np.ones(reference.shape, dtype=bool)
    for grid in grids:
        valid &= ~np.ma.getmaskarray(grid.data)
    return valid


def bounds_to_geometry(bounds: tuple[float, float, float, float]) -> BaseGeometry:
    return box(*bounds)
