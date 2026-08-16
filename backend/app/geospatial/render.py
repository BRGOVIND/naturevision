"""Rendering of analysis rasters into map-ready PNG overlays.

Overlays are reprojected to EPSG:4326 before encoding so that the image is
north-up in geographic coordinates and its corners coincide exactly with the
bounds handed to the map client. Colour mapping is explicit and the same
breakpoints drive the legend, so what a user sees is traceable to the values
that were computed.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling

from app.analysis.change import CHANGE_CLASS_LABELS, CLASS_NO_DATA
from app.geospatial.raster import RasterGrid
from app.models_ml.labels import CLASS_COLOURS, CLASS_INFO, CLASS_ORDER

WGS84 = CRS.from_epsg(4326)

#: NDVI colour ramp: bare/water through sparse to dense canopy.
NDVI_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (-1.00, (12, 44, 92)),
    (0.00, (72, 96, 116)),
    (0.10, (162, 134, 96)),
    (0.20, (214, 196, 122)),
    (0.35, (176, 196, 96)),
    (0.50, (110, 168, 70)),
    (0.70, (44, 122, 52)),
    (1.00, (10, 74, 34)),
)

#: Diverging ramp for index difference, centred on zero.
CHANGE_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (-0.60, (120, 20, 24)),
    (-0.20, (196, 76, 60)),
    (-0.10, (238, 168, 142)),
    (0.00, (244, 244, 240)),
    (0.10, (168, 210, 160)),
    (0.20, (72, 150, 90)),
    (0.60, (18, 84, 46)),
)

CHANGE_CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (150, 30, 32),
    2: (226, 132, 106),
    3: (238, 238, 232),
    4: (140, 194, 138),
    5: (32, 110, 58),
}


@dataclass(frozen=True, slots=True)
class RenderedLayer:
    """A PNG overlay plus everything the client needs to place and read it."""

    png: bytes
    bounds: list[float]  # [west, south, east, north] in EPSG:4326
    legend: list[dict[str, Any]]
    value_min: float | None
    value_max: float | None
    kind: str

    def to_data_url(self) -> str:
        return f"data:image/png;base64,{base64.b64encode(self.png).decode('ascii')}"


def _interpolate(values: np.ndarray, stops) -> np.ndarray:
    """Piecewise-linear RGB interpolation across the given breakpoints."""
    positions = np.array([s[0] for s in stops], dtype="float64")
    colours = np.array([s[1] for s in stops], dtype="float64")
    clipped = np.clip(values, positions[0], positions[-1])
    rgb = np.empty((*values.shape, 3), dtype="float64")
    for channel in range(3):
        rgb[..., channel] = np.interp(clipped, positions, colours[:, channel])
    return rgb.astype("uint8")


def _to_wgs84(grid: RasterGrid, categorical: bool) -> RasterGrid:
    if grid.crs == WGS84:
        return grid
    return grid.to_crs(WGS84, Resampling.nearest if categorical else Resampling.bilinear)


def _encode(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    rgba = np.dstack([rgb, alpha]).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_continuous(
    grid: RasterGrid,
    stops=NDVI_STOPS,
    *,
    opacity: int = 235,
    legend_label: str = "NDVI",
) -> RenderedLayer:
    """Render a continuous raster (NDVI, index difference) as an RGBA overlay."""
    warped = _to_wgs84(grid, categorical=False)
    values = warped.data.filled(np.nan)
    valid = ~np.ma.getmaskarray(warped.data) & np.isfinite(values)

    rgb = _interpolate(np.nan_to_num(values, nan=stops[0][0]), stops)
    alpha = np.where(valid, opacity, 0).astype("uint8")

    finite = values[valid]
    legend = [
        {"value": position, "colour": f"#{r:02x}{g:02x}{b:02x}", "label": f"{position:+.2f}"}
        for position, (r, g, b) in stops
    ]
    return RenderedLayer(
        png=_encode(rgb, alpha),
        bounds=list(warped.bounds_wgs84()),
        legend=[{**entry, "series": legend_label} for entry in legend],
        value_min=round(float(finite.min()), 4) if finite.size else None,
        value_max=round(float(finite.max()), 4) if finite.size else None,
        kind="continuous",
    )


def render_change(grid: RasterGrid) -> RenderedLayer:
    """Render the signed index difference on a zero-centred diverging ramp."""
    return render_continuous(grid, CHANGE_STOPS, legend_label="NDVI change")


def render_change_classes(grid: RasterGrid) -> RenderedLayer:
    """Render the discrete change classification."""
    warped = _to_wgs84(grid, categorical=True)
    codes = np.rint(warped.data.filled(CLASS_NO_DATA)).astype("int16")
    valid = ~np.ma.getmaskarray(warped.data) & (codes != CLASS_NO_DATA)

    rgb = np.zeros((*codes.shape, 3), dtype="uint8")
    for code, colour in CHANGE_CLASS_COLOURS.items():
        rgb[codes == code] = colour

    legend = [
        {
            "value": code,
            "colour": "#{:02x}{:02x}{:02x}".format(*CHANGE_CLASS_COLOURS[code]),
            "label": CHANGE_CLASS_LABELS[code],
        }
        for code in sorted(CHANGE_CLASS_COLOURS)
    ]
    return RenderedLayer(
        png=_encode(rgb, np.where(valid, 235, 0).astype("uint8")),
        bounds=list(warped.bounds_wgs84()),
        legend=legend,
        value_min=None,
        value_max=None,
        kind="categorical",
    )


def render_land_cover(grid: RasterGrid) -> RenderedLayer:
    """Render the land-cover classification using the shared class palette."""
    warped = _to_wgs84(grid, categorical=True)
    codes = np.rint(warped.data.filled(-1)).astype("int16")
    valid = ~np.ma.getmaskarray(warped.data) & (codes >= 0)

    rgb = np.zeros((*codes.shape, 3), dtype="uint8")
    for class_id in CLASS_ORDER:
        rgb[codes == int(class_id)] = _hex_to_rgb(CLASS_COLOURS[int(class_id)])

    legend = [
        {
            "value": int(class_id),
            "colour": CLASS_COLOURS[int(class_id)],
            "label": CLASS_INFO[class_id].label,
        }
        for class_id in CLASS_ORDER
    ]
    return RenderedLayer(
        png=_encode(rgb, np.where(valid, 235, 0).astype("uint8")),
        bounds=list(warped.bounds_wgs84()),
        legend=legend,
        value_min=None,
        value_max=None,
        kind="categorical",
    )


def render_confidence(grid: RasterGrid) -> RenderedLayer:
    """Render per-pixel model confidence on a single-hue ramp."""
    stops = (
        (0.20, (94, 20, 40)),
        (0.40, (168, 82, 60)),
        (0.60, (214, 158, 84)),
        (0.80, (168, 196, 120)),
        (1.00, (36, 122, 96)),
    )
    return render_continuous(grid, stops, legend_label="Confidence")


def render_rgb_composite(
    red: RasterGrid, green: RasterGrid, blue: RasterGrid, *, percentile: float = 2.0
) -> RenderedLayer:
    """True-colour composite with a percentile stretch.

    The stretch is applied per band over valid pixels only, which is what makes
    a surface-reflectance scene legible without altering the underlying data.
    """
    warped = [_to_wgs84(g, categorical=False) for g in (red, green, blue)]
    reference = warped[0]
    aligned = [reference, *(g.reproject_to(reference) for g in warped[1:])]

    channels: list[np.ndarray] = []
    valid = np.ones(reference.shape, dtype=bool)
    for grid in aligned:
        valid &= ~np.ma.getmaskarray(grid.data)

    for grid in aligned:
        values = grid.data.filled(np.nan)
        finite = values[valid & np.isfinite(values)]
        if finite.size == 0:
            channels.append(np.zeros(reference.shape, dtype="uint8"))
            continue
        low = float(np.percentile(finite, percentile))
        high = float(np.percentile(finite, 100.0 - percentile))
        if high <= low:
            high = low + 1e-6
        stretched = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
        channels.append((stretched * 255).astype("uint8"))

    return RenderedLayer(
        png=_encode(np.dstack(channels), np.where(valid, 255, 0).astype("uint8")),
        bounds=list(reference.bounds_wgs84()),
        legend=[],
        value_min=None,
        value_max=None,
        kind="continuous",
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
