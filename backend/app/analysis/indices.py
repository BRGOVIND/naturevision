"""Spectral index computation.

All indices are normalised differences of the form ``(a - b) / (a + b)``. The
shared implementation handles the two failure modes that quietly corrupt
remote-sensing results: division by a near-zero denominator, and propagation of
masked/nodata pixels into apparently valid output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.errors import RasterProcessingError
from app.geospatial.raster import RasterGrid, common_valid_mask
from app.imagery.bands import Band
from app.imagery.service import SceneStack

#: Denominators below this magnitude are treated as unusable rather than
#: producing a numerically explosive index value.
DENOMINATOR_EPSILON = 1e-6

#: Physically achievable range of a normalised difference index.
INDEX_MIN = -1.0
INDEX_MAX = 1.0


@dataclass(frozen=True, slots=True)
class IndexDefinition:
    """Metadata that makes an index result self-describing and reproducible."""

    key: str
    name: str
    formula: str
    numerator_positive: Band
    numerator_negative: Band
    description: str


NDVI = IndexDefinition(
    key="ndvi",
    name="Normalised Difference Vegetation Index",
    formula="(NIR - Red) / (NIR + Red)",
    numerator_positive=Band.NIR,
    numerator_negative=Band.RED,
    description=(
        "Chlorophyll-sensitive greenness proxy. Higher values indicate denser "
        "photosynthetically active vegetation."
    ),
)

NDWI = IndexDefinition(
    key="ndwi",
    name="Normalised Difference Water Index",
    formula="(Green - NIR) / (Green + NIR)",
    numerator_positive=Band.GREEN,
    numerator_negative=Band.NIR,
    description="Open-water and surface-moisture indicator (McFeeters formulation).",
)

NBR = IndexDefinition(
    key="nbr",
    name="Normalised Burn Ratio",
    formula="(NIR - SWIR2) / (NIR + SWIR2)",
    numerator_positive=Band.NIR,
    numerator_negative=Band.SWIR_22,
    description="Sensitive to canopy moisture loss and burnt surfaces.",
)

NDBI = IndexDefinition(
    key="ndbi",
    name="Normalised Difference Built-up Index",
    formula="(SWIR1 - NIR) / (SWIR1 + NIR)",
    numerator_positive=Band.SWIR_16,
    numerator_negative=Band.NIR,
    description="Elevated over impervious and built-up surfaces.",
)

INDEX_REGISTRY: dict[str, IndexDefinition] = {index.key: index for index in (NDVI, NDWI, NBR, NDBI)}


def normalised_difference(
    positive: RasterGrid, negative: RasterGrid, *, definition: IndexDefinition | None = None
) -> RasterGrid:
    """Compute ``(positive - negative) / (positive + negative)`` safely.

    Pixels are invalidated when either input is masked, when the denominator is
    within ``DENOMINATOR_EPSILON`` of zero, or when the result falls outside the
    mathematically achievable [-1, 1] envelope.
    """
    if not positive.is_aligned_with(negative):
        raise RasterProcessingError(
            "Index inputs are not on the same grid. Align the bands before "
            "computing a spectral index.",
            details={
                "positive": str(positive.spatial_signature()),
                "negative": str(negative.spatial_signature()),
                "index": definition.key if definition else "unknown",
            },
        )

    valid = common_valid_mask(positive, negative)
    a = positive.data.data.astype("float64")
    b = negative.data.data.astype("float64")

    denominator = a + b
    valid &= np.abs(denominator) > DENOMINATOR_EPSILON

    result = np.full(positive.shape, np.nan, dtype="float32")
    np.divide(a - b, denominator, out=result, where=valid, casting="unsafe")

    valid &= np.isfinite(result)
    valid &= (result >= INDEX_MIN) & (result <= INDEX_MAX)

    return RasterGrid(
        data=np.ma.masked_array(result, mask=~valid),
        transform=positive.transform,
        crs=positive.crs,
        nodata=float("nan"),
    )


def compute_index(scene: SceneStack, definition: IndexDefinition) -> RasterGrid:
    """Compute a registered index from a loaded scene stack."""
    return normalised_difference(
        scene.band(definition.numerator_positive),
        scene.band(definition.numerator_negative),
        definition=definition,
    )


def compute_ndvi(scene: SceneStack) -> RasterGrid:
    """NDVI = (NIR - Red) / (NIR + Red) from Sentinel-2 B08 and B04."""
    return compute_index(scene, NDVI)
