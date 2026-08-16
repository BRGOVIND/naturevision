"""Access to ESA WorldCover reference labels.

WorldCover v200 is published as 3x3 degree cloud-optimised GeoTIFF tiles on a
public bucket. Only the window overlapping a training region is read, so
building a training set costs megabytes rather than the ~50-80 MB per tile.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from app.core.logging import get_logger
from app.geospatial.raster import RasterGrid, window_for_bounds
from app.imagery.stac import GDAL_ENV

logger = get_logger(__name__)

WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WGS84 = CRS.from_epsg(4326)
TILE_SIZE_DEGREES = 3


def tile_name(longitude: float, latitude: float) -> str:
    """Resolve the WorldCover tile whose south-west corner contains a point."""
    lat_origin = int(math.floor(latitude / TILE_SIZE_DEGREES) * TILE_SIZE_DEGREES)
    lon_origin = int(math.floor(longitude / TILE_SIZE_DEGREES) * TILE_SIZE_DEGREES)
    ns = "N" if lat_origin >= 0 else "S"
    ew = "E" if lon_origin >= 0 else "W"
    return f"{ns}{abs(lat_origin):02d}{ew}{abs(lon_origin):03d}"


def tile_url(longitude: float, latitude: float) -> str:
    return (
        f"{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{tile_name(longitude, latitude)}_Map.tif"
    )


def load_labels_on_grid(reference: RasterGrid) -> RasterGrid:
    """Read WorldCover labels and resample them onto the reference pixel grid.

    Nearest-neighbour resampling is mandatory here: interpolating categorical
    class codes would invent classes that do not exist.
    """
    west, south, east, north = reference.bounds_wgs84()
    centre_lon = (west + east) / 2.0
    centre_lat = (south + north) / 2.0
    url = tile_url(centre_lon, centre_lat)

    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as dataset:
        window = window_for_bounds(
            dataset.transform, (dataset.height, dataset.width), (west, south, east, north), pad=2
        )
        if window.width < 2 or window.height < 2:
            raise ValueError(f"Region does not overlap WorldCover tile {url}")
        raw = dataset.read(1, window=window, resampling=Resampling.nearest)
        source_transform = dataset.window_transform(window)
        source_crs = dataset.crs
        source_nodata = dataset.nodata

    destination = np.zeros(reference.shape, dtype="uint8")
    reproject(
        source=raw,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=source_nodata if source_nodata is not None else 0,
        dst_transform=reference.transform,
        dst_crs=reference.crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    logger.info(
        "worldcover_labels_loaded",
        tile=tile_name(centre_lon, centre_lat),
        shape=list(destination.shape),
        distinct_classes=int(np.unique(destination).size),
    )
    return RasterGrid(
        data=np.ma.masked_array(destination.astype("float32"), mask=destination == 0),
        transform=reference.transform,
        crs=reference.crs,
        nodata=0.0,
    )
