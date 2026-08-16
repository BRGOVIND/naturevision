"""Sentinel-2 Level-2A access through a STAC API backed by cloud-optimised GeoTIFFs.

Reads are windowed: only the byte ranges covering the region of interest are
fetched from the remote COG, which keeps a typical analysis in the low
megabytes rather than downloading full ~1 GB scenes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from app.core.config import settings
from app.core.errors import ImageryAcquisitionError, ImagerySearchError
from app.core.logging import get_logger
from app.geospatial.geometry import BoundingBox, validate_bbox
from app.geospatial.raster import RasterGrid, window_for_bounds
from app.imagery.bands import BAND_SPECS, Band, resolve_asset_key
from app.imagery.base import (
    BandAsset,
    ImageryProvider,
    ImagerySearchRequest,
    RadiometricCalibration,
    SatelliteObservation,
)

logger = get_logger(__name__)

WGS84 = CRS.from_epsg(4326)

#: GDAL tuning for remote COG access. Restricting directory listings and
#: extensions removes a round trip per open, which dominates latency here.
GDAL_ENV: dict[str, str | int] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": str(settings.http_raster_retries),
    "GDAL_HTTP_RETRY_DELAY": "1",
    # rasterio forwards this one to GDALSetCacheMax64 and requires a real int.
    "GDAL_CACHEMAX": 256,
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "33554432",
    "AWS_NO_SIGN_REQUEST": "YES",
}

#: Sentinel-2 L2A digital numbers are 16-bit; these are the fallback conversion
#: constants when the catalogue does not publish per-asset scale/offset.
DEFAULT_QUANTIFICATION = 10_000.0
#: Processing baseline 04.00 (2022-01-25 onwards) added a -1000 DN radiometric
#: offset. Ignoring it biases reflectance by 0.1 and corrupts NDVI comparisons
#: that straddle the baseline change.
BASELINE_OFFSET_INTRODUCED = "04.00"

#: Longest edge of the cheap probe read used to verify a product's radiometry.
CALIBRATION_PROBE_DIM = 64

#: Statuses worth retrying: rate limiting and transient server-side faults.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SentinelHubStacProvider(ImageryProvider):
    """STAC-API imagery provider for Sentinel-2 Level-2A surface reflectance."""

    name = "Element84 Earth Search (AWS Open Data)"
    dataset = "sentinel-2-l2a"

    def __init__(
        self,
        endpoint: str | None = None,
        collection: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint = (endpoint or settings.stac_endpoint).rstrip("/")
        self.collection = collection or settings.stac_collection
        self.dataset = self.collection
        self._client = client
        self._owns_client = client is None
        # Caps in-flight catalogue requests from this process so parallel
        # analyses cannot exhaust the shared public endpoint.
        self._request_slots = asyncio.Semaphore(settings.stac_max_concurrent_requests)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.stac_timeout_seconds),
                headers={"Accept": "application/geo+json", "User-Agent": "NatureVision/0.1"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with bounded concurrency and exponential backoff.

        The public catalogue rate-limits and occasionally drops connections
        under parallel load, which is ordinary behaviour for a shared service
        rather than a fault. Transport errors and 429/5xx responses are retried;
        client errors are not, since repeating a malformed query cannot help.
        """
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, settings.stac_max_retries + 1):
            try:
                async with self._request_slots:
                    response = await self.client.post(url, json=payload)
                if response.status_code in _RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    raise
            except httpx.HTTPError as exc:
                last_exc = exc

            if attempt < settings.stac_max_retries:
                logger.warning(
                    "stac_request_retry",
                    attempt=attempt,
                    delay_seconds=delay,
                    error=str(last_exc)[:200],
                )
                await asyncio.sleep(delay)
                delay *= 2

        assert last_exc is not None
        raise last_exc

    # --- catalogue search -------------------------------------------------
    async def search(self, request: ImagerySearchRequest) -> list[SatelliteObservation]:
        payload: dict[str, Any] = {
            "collections": [self.collection],
            "bbox": request.bbox.as_list(),
            "datetime": (
                f"{request.start_date.isoformat()}T00:00:00Z/"
                f"{request.end_date.isoformat()}T23:59:59Z"
            ),
            "limit": min(request.limit, settings.max_search_results),
            "query": {"eo:cloud_cover": {"lte": request.max_cloud_cover}},
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }

        try:
            response = await self._post_with_retry(f"{self.endpoint}/search", payload)
            document = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "stac_search_http_error",
                status=exc.response.status_code,
                body=exc.response.text[:500],
            )
            raise ImagerySearchError(
                "The satellite catalogue rejected the search request.",
                details={"status": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("stac_search_transport_error", error=str(exc))
            raise ImagerySearchError(
                "The satellite catalogue is unreachable. Try again shortly."
            ) from exc
        except ValueError as exc:
            raise ImagerySearchError(
                "The satellite catalogue returned a malformed response."
            ) from exc

        features = document.get("features", [])
        observations: list[SatelliteObservation] = []
        for feature in features:
            try:
                observation = self._parse_feature(feature, request.bbox)
            except Exception as exc:  # a single bad item must not fail the search
                logger.warning("stac_item_parse_failed", item_id=feature.get("id"), error=str(exc))
                continue
            if request.required_bands and not observation.has_bands(request.required_bands):
                continue
            observations.append(observation)

        logger.info(
            "stac_search_completed",
            returned=len(features),
            usable=len(observations),
            collection=self.collection,
        )
        return observations

    def _parse_feature(
        self, feature: dict[str, Any], region_bbox: BoundingBox
    ) -> SatelliteObservation:
        props: dict[str, Any] = feature.get("properties", {})
        raw_assets: dict[str, Any] = feature.get("assets", {})
        asset_keys = list(raw_assets.keys())

        assets: dict[Band, BandAsset] = {}
        for band in Band:
            key = resolve_asset_key(band, asset_keys)
            if key is None:
                continue
            asset = raw_assets[key]
            href = asset.get("href")
            if not href:
                continue
            scale, offset, nodata, dtype = _raster_band_scaling(asset, props, band)
            assets[band] = BandAsset(
                band=band,
                href=href,
                asset_key=key,
                native_resolution_m=(asset.get("gsd") or BAND_SPECS[band].native_resolution_m),
                scale=scale,
                offset=offset,
                nodata=nodata,
                data_type=dtype,
            )

        timestamp = _parse_timestamp(props.get("datetime"))
        geometry = feature.get("bbox")
        bbox = validate_bbox(geometry[:4]) if geometry else region_bbox

        return SatelliteObservation(
            source_id=str(feature.get("id")),
            provider=self.name,
            dataset=self.collection,
            observation_date=(timestamp.date() if timestamp else dt.date.today()),
            acquisition_timestamp=timestamp,
            cloud_cover_percent=_as_float(props.get("eo:cloud_cover")),
            bbox=bbox,
            geometry=feature.get("geometry"),
            processing_level=props.get("processing:level") or props.get("s2:product_type") or "L2A",
            platform=props.get("platform"),
            instrument=(props.get("instruments") or ["msi"])[0]
            if isinstance(props.get("instruments"), list)
            else props.get("instruments"),
            crs=_epsg_string(props.get("proj:epsg") or props.get("proj:code")),
            resolution_m=_as_float(props.get("gsd")) or 10.0,
            license=feature.get("license") or props.get("license") or "CC-BY-4.0 (Copernicus)",
            assets=assets,
            properties={
                "grid_square": props.get("grid:code") or props.get("s2:mgrs_tile"),
                "processing_baseline": props.get("s2:processing_baseline"),
                "nodata_pixel_percentage": props.get("s2:nodata_pixel_percentage"),
                "vegetation_percentage": props.get("s2:vegetation_percentage"),
                "water_percentage": props.get("s2:water_percentage"),
            },
            region_coverage=_coverage_fraction(feature.get("geometry"), region_bbox),
        )

    # --- pixel access ------------------------------------------------------
    async def resolve_calibration(
        self, observation: SatelliteObservation, bbox: BoundingBox
    ) -> RadiometricCalibration:
        """Verify the catalogue's declared radiometric offset against real pixels.

        Earth Search publishes one static asset description for the whole
        collection, so every item is tagged ``offset: -0.1`` even when it was
        produced before processing baseline 04.00 introduced that offset. The
        declared value is therefore treated as a hypothesis and tested on a
        small probe read of a low-reflectance band, where a spurious -0.1 shift
        is unmistakable: it drives most land pixels negative.
        """
        probe_band = next(
            (b for b in (Band.RED, Band.BLUE, Band.GREEN) if b in observation.assets), None
        )
        asset = observation.assets.get(probe_band) if probe_band else None
        if asset is None or asset.offset == 0.0:
            return RadiometricCalibration(
                offset=asset.offset if asset else 0.0, scale_source="catalogue"
            )

        try:
            probe = await asyncio.to_thread(_read_raw_window, asset, bbox, CALIBRATION_PROBE_DIM)
        except ImageryAcquisitionError:
            logger.warning("calibration_probe_failed", source_id=observation.source_id)
            return RadiometricCalibration(offset=asset.offset, scale_source="catalogue")

        valid = probe > 0
        if not valid.any():
            return RadiometricCalibration(offset=asset.offset, scale_source="catalogue")

        values = probe[valid].astype("float64")
        with_offset = float(((values * asset.scale + asset.offset) < 0.0).mean())
        without_offset = float(((values * asset.scale) < 0.0).mean())

        diagnostic = {
            "probe_band": probe_band.value if probe_band else None,
            "declared_offset": asset.offset,
            "negative_fraction_with_offset": round(with_offset, 4),
            "negative_fraction_without_offset": round(without_offset, 4),
            "threshold": MAX_NEGATIVE_REFLECTANCE_FRACTION,
        }

        if with_offset > MAX_NEGATIVE_REFLECTANCE_FRACTION and without_offset < with_offset:
            logger.warning(
                "radiometric_offset_overridden",
                source_id=observation.source_id,
                applied_offset=0.0,
                **diagnostic,
            )
            return RadiometricCalibration(
                offset=0.0,
                scale_source="catalogue_scale_with_verified_offset",
                decision="physical_override",
                diagnostic=diagnostic,
            )

        return RadiometricCalibration(
            offset=asset.offset,
            scale_source="catalogue",
            decision="catalogue",
            diagnostic=diagnostic,
        )

    async def read_band(
        self,
        observation: SatelliteObservation,
        band: Band,
        bbox: BoundingBox,
        *,
        max_dimension: int | None = None,
        calibration: RadiometricCalibration | None = None,
    ) -> RasterGrid:
        asset = observation.assets.get(band)
        if asset is None:
            raise ImageryAcquisitionError(
                f"Observation does not publish the {band.value} band.",
                details={"source_id": observation.source_id, "band": band.value},
            )
        limit = max_dimension or settings.target_raster_max_dim
        return await asyncio.to_thread(_read_windowed_cog, asset, bbox, limit, band, calibration)


# --- module-level helpers (kept out of the class so they can be threaded) ---
def _read_raw_window(asset: BandAsset, bbox: BoundingBox, max_dimension: int) -> np.ndarray:
    """Small unscaled read used to interrogate a product's radiometry."""
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open(asset.href) as dataset:
            dst_bounds = transform_bounds(WGS84, dataset.crs, *bbox.as_tuple(), densify_pts=21)
            window = window_for_bounds(
                dataset.transform, (dataset.height, dataset.width), dst_bounds
            )
            height, width = _output_shape(int(window.height), int(window.width), max_dimension)
            return dataset.read(
                1, window=window, out_shape=(height, width), resampling=Resampling.nearest
            )
    except rasterio.errors.RasterioIOError as exc:
        raise ImageryAcquisitionError(
            "A satellite image band could not be read from the provider."
        ) from exc


def _read_windowed_cog(
    asset: BandAsset,
    bbox: BoundingBox,
    max_dimension: int,
    band: Band,
    calibration: RadiometricCalibration | None = None,
) -> RasterGrid:
    """Fetch just the AOI window of a remote COG and convert it to reflectance."""
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open(asset.href) as dataset:
            dst_bounds = transform_bounds(WGS84, dataset.crs, *bbox.as_tuple(), densify_pts=21)
            window = window_for_bounds(
                dataset.transform, (dataset.height, dataset.width), dst_bounds
            )
            if window.width < 2 or window.height < 2:
                raise ImageryAcquisitionError(
                    "The requested region does not overlap this observation's footprint.",
                    details={"asset": asset.asset_key},
                )

            out_height, out_width = _output_shape(
                int(window.height), int(window.width), max_dimension
            )
            resampling = (
                Resampling.nearest if band is Band.SCENE_CLASSIFICATION else Resampling.average
            )
            raw = dataset.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=resampling,
                boundless=False,
            )
            window_transform = dataset.window_transform(window)
            scale_x = window.width / out_width
            scale_y = window.height / out_height
            transform = window_transform * rasterio.Affine.scale(scale_x, scale_y)
            dataset_nodata = dataset.nodata
            dataset_crs = dataset.crs
    except ImageryAcquisitionError:
        raise
    except rasterio.errors.RasterioIOError as exc:
        logger.error("cog_read_failed", href=asset.href[:160], error=str(exc))
        raise ImageryAcquisitionError(
            "A satellite image band could not be read from the provider.",
            details={"band": band.value},
        ) from exc

    nodata_value = asset.nodata if asset.nodata is not None else dataset_nodata

    if band is Band.SCENE_CLASSIFICATION:
        values = raw.astype("float32")
        mask = (
            np.zeros(values.shape, dtype=bool)
            if nodata_value is None
            else values == float(nodata_value)
        )
        return RasterGrid(
            data=np.ma.masked_array(values, mask=mask),
            transform=transform,
            crs=dataset_crs,
            nodata=float(nodata_value) if nodata_value is not None else float("nan"),
        )

    values = raw.astype("float32")
    invalid = ~np.isfinite(values)
    if nodata_value is not None:
        invalid |= values == float(nodata_value)

    # The offset is a per-observation decision resolved by resolve_calibration
    # and applied identically to every band, so all inputs to an index share one
    # radiometric convention.
    offset = calibration.offset if calibration is not None else asset.offset
    reflectance = values * asset.scale + offset

    # Surface reflectance is physically bounded; values outside a tolerant
    # envelope indicate saturation or processing artefacts.
    invalid |= (reflectance < -0.2) | (reflectance > 1.6)

    return RasterGrid(
        data=np.ma.masked_array(reflectance, mask=invalid),
        transform=transform,
        crs=dataset_crs,
        nodata=float("nan"),
    )


#: Above this share of negative surface reflectance the candidate calibration is
#: rejected. Genuine BOA reflectance is negative only on dark water and deep
#: shadow after atmospheric correction, which is a small minority of most scenes.
MAX_NEGATIVE_REFLECTANCE_FRACTION = 0.10


def _output_shape(height: int, width: int, max_dimension: int) -> tuple[int, int]:
    longest = max(height, width)
    if longest <= max_dimension:
        return height, width
    factor = longest / max_dimension
    return max(1, round(height / factor)), max(1, round(width / factor))


def _raster_band_scaling(
    asset: dict[str, Any], props: dict[str, Any], band: Band
) -> tuple[float, float, float | None, str | None]:
    """Resolve DN → reflectance conversion from asset metadata with a safe fallback."""
    if band is Band.SCENE_CLASSIFICATION:
        return 1.0, 0.0, 0.0, "uint8"

    raster_bands = asset.get("raster:bands") or asset.get("bands") or []
    if isinstance(raster_bands, list) and raster_bands:
        entry = raster_bands[0]
        scale = _as_float(entry.get("scale"))
        offset = _as_float(entry.get("offset"))
        if scale:
            return (
                scale,
                offset if offset is not None else 0.0,
                _as_float(entry.get("nodata")),
                entry.get("data_type"),
            )

    baseline = str(props.get("s2:processing_baseline") or "")
    offset = 0.0
    if baseline and baseline >= BASELINE_OFFSET_INTRODUCED:
        offset = -1000.0 / DEFAULT_QUANTIFICATION
    return 1.0 / DEFAULT_QUANTIFICATION, offset, 0.0, "uint16"


def _parse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _epsg_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.upper().startswith("EPSG") else f"EPSG:{text}"


def _coverage_fraction(geometry: dict[str, Any] | None, bbox: BoundingBox) -> float | None:
    """Fraction of the requested region covered by the scene footprint."""
    if not geometry:
        return None
    try:
        footprint = shape(geometry)
        region = shape(bbox.to_geojson())
        if region.area == 0:
            return None
        return round(min(1.0, footprint.intersection(region).area / region.area), 4)
    except Exception:
        return None
