"""Land-cover class definitions and the mapping from the reference label source.

Training labels are taken from ESA WorldCover, a 10 m global land-cover product
derived from Sentinel-1 and Sentinel-2 with a published, independently assessed
accuracy. Its eleven classes are collapsed into the five classes this product
reports. The collapse is lossy and is documented here because it directly
bounds what the classifier can be held accountable for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LandCoverClass(IntEnum):
    """Classes predicted by the land-cover model."""

    FOREST = 0
    AGRICULTURE = 1
    WATER = 2
    URBAN = 3
    BARE = 4


@dataclass(frozen=True, slots=True)
class ClassInfo:
    label: str
    description: str
    #: Stable hex colour used consistently by the map, legend and report.
    colour: str


CLASS_INFO: dict[LandCoverClass, ClassInfo] = {
    LandCoverClass.FOREST: ClassInfo(
        "Forest",
        "Tree-dominated cover, including closed and open canopy.",
        "#1b7f3b",
    ),
    LandCoverClass.AGRICULTURE: ClassInfo(
        "Agriculture",
        "Cropland and managed herbaceous cover, including grassland and shrubland.",
        "#d9a441",
    ),
    LandCoverClass.WATER: ClassInfo(
        "Water",
        "Permanent open water bodies and herbaceous wetland.",
        "#2b6cb0",
    ),
    LandCoverClass.URBAN: ClassInfo(
        "Urban / built-up",
        "Impervious and built surfaces.",
        "#a02c2c",
    ),
    LandCoverClass.BARE: ClassInfo(
        "Bare land",
        "Sparsely vegetated or unvegetated ground, including snow and ice.",
        "#8a8f98",
    ),
}

CLASS_ORDER: tuple[LandCoverClass, ...] = tuple(LandCoverClass)
CLASS_LABELS: dict[int, str] = {int(c): CLASS_INFO[c].label for c in CLASS_ORDER}
CLASS_COLOURS: dict[int, str] = {int(c): CLASS_INFO[c].colour for c in CLASS_ORDER}

#: ESA WorldCover v200 class codes.
WORLDCOVER_TREE_COVER = 10
WORLDCOVER_SHRUBLAND = 20
WORLDCOVER_GRASSLAND = 30
WORLDCOVER_CROPLAND = 40
WORLDCOVER_BUILT_UP = 50
WORLDCOVER_BARE = 60
WORLDCOVER_SNOW_ICE = 70
WORLDCOVER_WATER = 80
WORLDCOVER_HERBACEOUS_WETLAND = 90
WORLDCOVER_MANGROVES = 95
WORLDCOVER_MOSS_LICHEN = 100

#: Collapse of WorldCover classes onto this product's five classes.
#:
#: Notable consequences, which are surfaced in every report's limitations:
#:  * shrubland and grassland are folded into Agriculture, so "Agriculture"
#:    means managed-or-herbaceous cover rather than cropland specifically;
#:  * mangroves are counted as Forest;
#:  * snow and ice are counted as Bare land.
WORLDCOVER_TO_CLASS: dict[int, LandCoverClass] = {
    WORLDCOVER_TREE_COVER: LandCoverClass.FOREST,
    WORLDCOVER_MANGROVES: LandCoverClass.FOREST,
    WORLDCOVER_SHRUBLAND: LandCoverClass.AGRICULTURE,
    WORLDCOVER_GRASSLAND: LandCoverClass.AGRICULTURE,
    WORLDCOVER_CROPLAND: LandCoverClass.AGRICULTURE,
    WORLDCOVER_BUILT_UP: LandCoverClass.URBAN,
    WORLDCOVER_BARE: LandCoverClass.BARE,
    WORLDCOVER_SNOW_ICE: LandCoverClass.BARE,
    WORLDCOVER_MOSS_LICHEN: LandCoverClass.BARE,
    WORLDCOVER_WATER: LandCoverClass.WATER,
    WORLDCOVER_HERBACEOUS_WETLAND: LandCoverClass.WATER,
}

CLASS_COLLAPSE_NOTES: tuple[str, ...] = (
    "Shrubland and grassland are merged into Agriculture, so that class denotes "
    "managed or herbaceous cover rather than cropland specifically.",
    "Mangroves are reported as Forest.",
    "Snow, ice, moss and lichen are reported as Bare land.",
    "Herbaceous wetland is reported as Water.",
)

REFERENCE_LABEL_SOURCE = {
    "name": "ESA WorldCover v200 (2021)",
    "resolution_m": 10,
    "license": "CC BY 4.0",
    "url": "https://esa-worldcover.org/",
    "role": "Reference labels for supervised training and held-out evaluation.",
}
