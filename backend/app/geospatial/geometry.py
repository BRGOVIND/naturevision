"""Server-side validation and normalisation of user-supplied geographic regions.

Frontend validation is a convenience only; every geometry entering the analysis
pipeline is re-validated here. All public geometry is WGS84 (EPSG:4326) with
longitude/latitude ordering, matching the GeoJSON specification (RFC 7946).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pyproj import Geod
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity, make_valid

from app.core.config import settings
from app.core.errors import GeometryValidationError

WGS84 = "EPSG:4326"
_GEOD = Geod(ellps="WGS84")

SUPPORTED_CRS = {"EPSG:4326", "OGC:CRS84", "CRS84", "EPSG:4326:1.3", "WGS84"}
SUPPORTED_GEOMETRY_TYPES = {"Polygon", "MultiPolygon"}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned WGS84 bounding box with ordered edges."""

    west: float
    south: float
    east: float
    north: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def to_polygon_coords(self) -> list[list[list[float]]]:
        return [
            [
                [self.west, self.south],
                [self.east, self.south],
                [self.east, self.north],
                [self.west, self.north],
                [self.west, self.south],
            ]
        ]

    def to_geojson(self) -> dict[str, Any]:
        return {"type": "Polygon", "coordinates": self.to_polygon_coords()}


@dataclass(frozen=True, slots=True)
class ValidatedRegion:
    """A geometry that has passed every server-side check."""

    geometry: BaseGeometry
    bbox: BoundingBox
    area_km2: float
    crs: str = WGS84

    @property
    def geojson(self) -> dict[str, Any]:
        return mapping(self.geometry)  # type: ignore[return-value]

    def describe(self) -> str:
        lon, lat = self.bbox.center
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"{abs(lat):.3f}°{ns}, {abs(lon):.3f}°{ew} ({self.area_km2:.1f} km²)"


def normalise_crs(crs: str | None) -> str:
    """Accept the small set of WGS84 spellings clients realistically send."""
    if crs is None:
        return WGS84
    candidate = crs.strip().upper().replace("URN:OGC:DEF:CRS:", "")
    if candidate in {c.upper() for c in SUPPORTED_CRS} or candidate in {"4326"}:
        return WGS84
    raise GeometryValidationError(
        f"Unsupported coordinate reference system '{crs}'. Submit geometry in EPSG:4326.",
        details={"supported": sorted(SUPPORTED_CRS)},
    )


def validate_bbox(values: list[float] | tuple[float, ...]) -> BoundingBox:
    """Validate ordering and coordinate ranges of a [west, south, east, north] box."""
    if len(values) != 4:
        raise GeometryValidationError(
            "A bounding box must contain exactly four values: [west, south, east, north].",
            details={"received_length": len(values)},
        )
    if any(not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in values):
        raise GeometryValidationError("Bounding box values must be finite numbers.")

    west, south, east, north = (float(v) for v in values)

    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise GeometryValidationError(
            "Longitude values must lie within [-180, 180].",
            details={"west": west, "east": east},
        )
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise GeometryValidationError(
            "Latitude values must lie within [-90, 90].",
            details={"south": south, "north": north},
        )
    if west >= east:
        raise GeometryValidationError(
            "Bounding box west must be strictly less than east. "
            "Boxes crossing the antimeridian are not supported.",
            details={"west": west, "east": east},
        )
    if south >= north:
        raise GeometryValidationError(
            "Bounding box south must be strictly less than north.",
            details={"south": south, "north": north},
        )
    return BoundingBox(west, south, east, north)


def geodesic_area_km2(geometry: BaseGeometry) -> float:
    """Ellipsoidal area on the WGS84 spheroid, correct at any latitude.

    Planar shapely ``.area`` is in square degrees and is meaningless as a
    physical extent, so geodesic integration is used instead.
    """
    area_m2, _perimeter = _GEOD.geometry_area_perimeter(geometry)
    return abs(area_m2) / 1_000_000.0


def _check_extent(geometry: BaseGeometry) -> float:
    area = geodesic_area_km2(geometry)
    if area < settings.min_region_area_km2:
        raise GeometryValidationError(
            f"The selected region is too small to analyse "
            f"({area:.4f} km² < {settings.min_region_area_km2} km²).",
            details={"area_km2": area, "minimum_km2": settings.min_region_area_km2},
        )
    if area > settings.max_region_area_km2:
        raise GeometryValidationError(
            f"The selected region is too large for a single analysis "
            f"({area:.1f} km² > {settings.max_region_area_km2} km²). "
            "Select a smaller area.",
            details={"area_km2": area, "maximum_km2": settings.max_region_area_km2},
        )
    return area


def validate_geojson_geometry(geojson: dict[str, Any], crs: str | None = None) -> ValidatedRegion:
    """Validate a GeoJSON Polygon/MultiPolygon and return a normalised region."""
    normalise_crs(crs)

    if not isinstance(geojson, dict) or "type" not in geojson:
        raise GeometryValidationError("Geometry must be a GeoJSON geometry object.")

    geom_type = geojson.get("type")
    if geom_type == "Feature":  # tolerate a Feature wrapper from drawing tools
        geojson = geojson.get("geometry") or {}
        geom_type = geojson.get("type")
    if geom_type not in SUPPORTED_GEOMETRY_TYPES:
        raise GeometryValidationError(
            f"Unsupported geometry type '{geom_type}'.",
            details={"supported": sorted(SUPPORTED_GEOMETRY_TYPES)},
        )

    try:
        geometry = shape(geojson)
    except Exception as exc:  # malformed coordinate arrays
        raise GeometryValidationError(
            "The geometry could not be parsed.", details={"reason": str(exc)}
        ) from exc

    if geometry.is_empty:
        raise GeometryValidationError("The geometry is empty.")

    for lon, lat in _iter_coords(geometry):
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise GeometryValidationError("Geometry contains non-finite coordinates.")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise GeometryValidationError(
                "Geometry coordinates fall outside the valid WGS84 range. "
                "Coordinates must be ordered [longitude, latitude].",
                details={"longitude": lon, "latitude": lat},
            )

    if not geometry.is_valid:
        reason = explain_validity(geometry)
        repaired = make_valid(geometry)
        if repaired.is_empty or repaired.geom_type not in SUPPORTED_GEOMETRY_TYPES:
            raise GeometryValidationError(
                "The polygon is self-intersecting and could not be repaired.",
                details={"reason": reason},
            )
        geometry = repaired

    area_km2 = _check_extent(geometry)
    bbox = validate_bbox(list(geometry.bounds))
    return ValidatedRegion(geometry=geometry, bbox=bbox, area_km2=area_km2)


def validate_region(
    *,
    geometry: dict[str, Any] | None = None,
    bbox: list[float] | None = None,
    crs: str | None = None,
) -> ValidatedRegion:
    """Validate whichever spatial selector the client supplied.

    Exactly one of ``geometry`` or ``bbox`` must be provided; an explicit
    polygon wins when both are present so that draw-tool output is honoured.
    """
    if geometry is None and bbox is None:
        raise GeometryValidationError("Provide either a GeoJSON geometry or a bounding box.")
    if geometry is not None:
        return validate_geojson_geometry(geometry, crs)

    normalise_crs(crs)
    box = validate_bbox(bbox or [])
    polygon = shape(box.to_geojson())
    area_km2 = _check_extent(polygon)
    return ValidatedRegion(geometry=polygon, bbox=box, area_km2=area_km2)


def _iter_coords(geometry: BaseGeometry):
    if geometry.geom_type == "Polygon":
        yield from geometry.exterior.coords
        for ring in geometry.interiors:
            yield from ring.coords
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            yield from _iter_coords(part)
