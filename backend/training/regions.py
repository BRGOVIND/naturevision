"""Training and evaluation regions for the land-cover model.

Regions are chosen to span distinct biomes, latitudes and land-use intensities
so the model is not fitted to a single landscape. Evaluation regions are held
out entirely: no pixel from an evaluation region is ever seen during training.

A random pixel-level split would badly overstate accuracy here, because
neighbouring 10 m pixels are strongly spatially autocorrelated — a held-out
pixel would almost always sit next to a training pixel from the same field or
canopy. Holding out whole geographies is the honest protocol.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingRegion:
    """A geographic sample site with a cloud-free acquisition window."""

    key: str
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    start: dt.date
    end: dt.date
    biome: str
    role: str  # "train" or "evaluate"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "bbox": list(self.bbox),
            "period": f"{self.start.isoformat()}/{self.end.isoformat()}",
            "biome": self.biome,
            "role": self.role,
        }


def _window(year: int, month: int, span_days: int = 45) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    return start, start + dt.timedelta(days=span_days)


TRAINING_REGIONS: tuple[TrainingRegion, ...] = (
    TrainingRegion(
        key="western_ghats",
        name="Western Ghats, India",
        bbox=(76.60, 10.10, 76.78, 10.28),
        start=_window(2021, 1)[0],
        end=_window(2021, 1)[1],
        biome="Tropical moist broadleaf forest and plantation mosaic",
        role="train",
    ),
    TrainingRegion(
        key="po_valley",
        name="Po Valley, Italy",
        bbox=(9.10, 45.10, 9.28, 45.28),
        start=_window(2021, 6)[0],
        end=_window(2021, 6)[1],
        biome="Intensive irrigated cropland with dense settlement",
        role="train",
    ),
    TrainingRegion(
        key="brandenburg",
        name="Brandenburg, Germany",
        bbox=(13.05, 52.30, 13.23, 52.48),
        start=_window(2021, 7)[0],
        end=_window(2021, 7)[1],
        biome="Temperate mixed forest, lakes and peri-urban development",
        role="train",
    ),
    TrainingRegion(
        key="nile_delta",
        name="Nile Delta margin, Egypt",
        bbox=(31.05, 30.35, 31.23, 30.53),
        start=_window(2021, 3)[0],
        end=_window(2021, 3)[1],
        biome="Irrigated delta cropland adjoining arid bare ground",
        role="train",
    ),
    TrainingRegion(
        key="mato_grosso",
        name="Mato Grosso frontier, Brazil",
        bbox=(-55.30, -12.20, -55.12, -12.02),
        start=_window(2021, 7)[0],
        end=_window(2021, 7)[1],
        biome="Amazon-Cerrado transition with mechanised agriculture",
        role="train",
    ),
    # Dry and open-canopy biomes. The first training round used only humid
    # regions and collapsed to 7% Forest recall on semi-arid evaluation sites:
    # sparse dry woodland is spectrally close to the shrubland and grassland
    # that this product folds into Agriculture. These sites give the model
    # examples of tree cover with low canopy closure and a bright soil
    # background, and of genuinely bare ground, which was previously almost
    # unrepresented.
    TrainingRegion(
        key="sonoran_desert",
        name="Sonoran Desert margin, Arizona",
        bbox=(-111.50, 32.20, -111.32, 32.38),
        start=_window(2021, 4)[0],
        end=_window(2021, 4)[1],
        biome="Desert scrub and exposed bare ground",
        role="train",
    ),
    TrainingRegion(
        key="sahel_niger",
        name="Sahel, south-west Niger",
        bbox=(2.00, 13.40, 2.18, 13.58),
        start=_window(2021, 2)[0],
        end=_window(2021, 2)[1],
        biome="Dry-season Sahelian savanna, sparse tree cover and bare soil",
        role="train",
    ),
    TrainingRegion(
        key="kalahari_botswana",
        name="Kalahari, Botswana",
        bbox=(24.50, -21.50, 24.68, -21.32),
        start=_window(2021, 8)[0],
        end=_window(2021, 8)[1],
        biome="Semi-arid savanna woodland on sandveld",
        role="train",
    ),
    TrainingRegion(
        key="sacramento_valley",
        name="Sacramento Valley margin, California",
        bbox=(-122.10, 38.50, -121.92, 38.68),
        start=_window(2021, 6)[0],
        end=_window(2021, 6)[1],
        biome="Mediterranean oak woodland, irrigated cropland and dry grassland",
        role="train",
    ),
    TrainingRegion(
        key="finnish_lakeland",
        name="Finnish Lakeland",
        bbox=(25.50, 61.80, 25.68, 61.98),
        start=_window(2021, 7)[0],
        end=_window(2021, 7)[1],
        biome="Boreal coniferous forest and lakes",
        role="train",
    ),
    TrainingRegion(
        key="murray_basin",
        name="Murray Basin, Australia",
        bbox=(144.30, -35.40, 144.48, -35.22),
        start=_window(2021, 1)[0],
        end=_window(2021, 1)[1],
        biome="Semi-arid rangeland, dryland cropping and river channels",
        role="evaluate",
    ),
    TrainingRegion(
        key="iberian_meseta",
        name="Iberian Meseta, Spain",
        bbox=(-4.85, 40.55, -4.67, 40.73),
        start=_window(2021, 6)[0],
        end=_window(2021, 6)[1],
        biome="Mediterranean dehesa, cereal steppe and bare rock",
        role="evaluate",
    ),
    TrainingRegion(
        key="zambezi_miombo",
        name="Miombo woodland, Zambia",
        bbox=(27.60, -14.50, 27.78, -14.32),
        start=_window(2021, 6)[0],
        end=_window(2021, 6)[1],
        biome="Dry miombo woodland with smallholder cultivation",
        role="evaluate",
    ),
)


def regions_for(role: str) -> tuple[TrainingRegion, ...]:
    return tuple(r for r in TRAINING_REGIONS if r.role == role)
