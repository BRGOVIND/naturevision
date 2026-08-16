"""Spectral band definitions for the supported sensors.

Band handling is expressed in physical terms (wavelength, native resolution)
rather than provider asset keys, so an alternative Sentinel-2 distributor or a
different sensor can be plugged in without touching the analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Band(StrEnum):
    """Logical bands the analysis pipeline knows how to consume."""

    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    RED_EDGE_1 = "rededge1"
    NIR = "nir"
    NIR_NARROW = "nir08"
    SWIR_16 = "swir16"
    SWIR_22 = "swir22"
    SCENE_CLASSIFICATION = "scl"


@dataclass(frozen=True, slots=True)
class BandSpec:
    """Physical description of a single spectral band."""

    band: Band
    sentinel2_id: str
    centre_wavelength_nm: float
    native_resolution_m: int
    description: str
    #: Provider asset keys that have been observed to carry this band.
    asset_aliases: tuple[str, ...] = ()


BAND_SPECS: dict[Band, BandSpec] = {
    Band.BLUE: BandSpec(Band.BLUE, "B02", 492.4, 10, "Blue", ("blue", "B02", "B02_10m")),
    Band.GREEN: BandSpec(Band.GREEN, "B03", 559.8, 10, "Green", ("green", "B03", "B03_10m")),
    Band.RED: BandSpec(Band.RED, "B04", 664.6, 10, "Red", ("red", "B04", "B04_10m")),
    Band.RED_EDGE_1: BandSpec(
        Band.RED_EDGE_1, "B05", 704.1, 20, "Red edge 1", ("rededge1", "B05", "B05_20m")
    ),
    Band.NIR: BandSpec(Band.NIR, "B08", 832.8, 10, "Near infrared", ("nir", "B08", "B08_10m")),
    Band.NIR_NARROW: BandSpec(
        Band.NIR_NARROW, "B8A", 864.7, 20, "Narrow near infrared", ("nir08", "B8A", "B8A_20m")
    ),
    Band.SWIR_16: BandSpec(
        Band.SWIR_16, "B11", 1613.7, 20, "Short-wave infrared 1", ("swir16", "B11", "B11_20m")
    ),
    Band.SWIR_22: BandSpec(
        Band.SWIR_22, "B12", 2202.4, 20, "Short-wave infrared 2", ("swir22", "B12", "B12_20m")
    ),
    Band.SCENE_CLASSIFICATION: BandSpec(
        Band.SCENE_CLASSIFICATION,
        "SCL",
        0.0,
        20,
        "Scene classification layer",
        ("scl", "SCL", "SCL_20m"),
    ),
}

#: Bands required to compute NDVI.
NDVI_BANDS: tuple[Band, Band] = (Band.RED, Band.NIR)

#: Bands needed for a true-colour composite.
TRUE_COLOUR_BANDS: tuple[Band, Band, Band] = (Band.RED, Band.GREEN, Band.BLUE)

#: Bands used as land-cover classifier inputs. Ordering is part of the feature
#: contract and must match the training pipeline.
LAND_COVER_BANDS: tuple[Band, ...] = (
    Band.BLUE,
    Band.GREEN,
    Band.RED,
    Band.NIR,
    Band.SWIR_16,
    Band.SWIR_22,
)


# --- Sentinel-2 Level-2A scene classification (SCL) values -----------------
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW_ICE = 11

#: SCL values discarded before any index or classification is computed.
INVALID_SCL_CLASSES: frozenset[int] = frozenset(
    {
        SCL_NO_DATA,
        SCL_SATURATED,
        SCL_CLOUD_SHADOW,
        SCL_CLOUD_MEDIUM_PROB,
        SCL_CLOUD_HIGH_PROB,
        SCL_THIN_CIRRUS,
        SCL_SNOW_ICE,
    }
)

SCL_LABELS: dict[int, str] = {
    SCL_NO_DATA: "No data",
    SCL_SATURATED: "Saturated or defective",
    SCL_DARK_AREA: "Dark area / topographic shadow",
    SCL_CLOUD_SHADOW: "Cloud shadow",
    SCL_VEGETATION: "Vegetation",
    SCL_NOT_VEGETATED: "Not vegetated",
    SCL_WATER: "Water",
    SCL_UNCLASSIFIED: "Unclassified",
    SCL_CLOUD_MEDIUM_PROB: "Cloud (medium probability)",
    SCL_CLOUD_HIGH_PROB: "Cloud (high probability)",
    SCL_THIN_CIRRUS: "Thin cirrus",
    SCL_SNOW_ICE: "Snow or ice",
}


def resolve_asset_key(band: Band, available_assets: list[str]) -> str | None:
    """Map a logical band onto a provider asset key, tolerating naming drift."""
    spec = BAND_SPECS[band]
    lookup = {key.lower(): key for key in available_assets}
    for alias in spec.asset_aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None
