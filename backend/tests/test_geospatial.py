"""Geometry validation and raster handling."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from app.core.errors import GeometryValidationError, RasterProcessingError
from app.geospatial.geometry import (
    geodesic_area_km2,
    normalise_crs,
    validate_bbox,
    validate_geojson_geometry,
    validate_region,
)
from app.geospatial.raster import RasterGrid, align_grids, common_valid_mask, window_for_bounds
from tests.conftest import BASE_TRANSFORM, UTM, make_grid


# --- bounding boxes ---------------------------------------------------------
def test_valid_bbox_is_accepted():
    box = validate_bbox([76.6, 10.1, 76.7, 10.2])
    assert box.as_tuple() == (76.6, 10.1, 76.7, 10.2)
    assert box.center == pytest.approx((76.65, 10.15))


@pytest.mark.parametrize(
    "values",
    [
        [76.7, 10.1, 76.6, 10.2],  # west >= east
        [76.6, 10.2, 76.7, 10.1],  # south >= north
        [76.6, 10.1, 76.6, 10.2],  # zero width
    ],
)
def test_bbox_ordering_is_enforced(values):
    with pytest.raises(GeometryValidationError):
        validate_bbox(values)


@pytest.mark.parametrize(
    "values",
    [
        [-181.0, 10.0, 10.0, 20.0],
        [0.0, -91.0, 10.0, 20.0],
        [0.0, 10.0, 181.0, 20.0],
        [0.0, 10.0, 10.0, 91.0],
    ],
)
def test_bbox_coordinate_ranges_are_enforced(values):
    with pytest.raises(GeometryValidationError):
        validate_bbox(values)


def test_bbox_rejects_wrong_length_and_non_finite():
    with pytest.raises(GeometryValidationError):
        validate_bbox([1.0, 2.0, 3.0])
    with pytest.raises(GeometryValidationError):
        validate_bbox([0.0, 0.0, float("nan"), 1.0])


# --- polygons ----------------------------------------------------------------
def test_polygon_is_validated_and_area_computed():
    geojson = {
        "type": "Polygon",
        "coordinates": [[[76.6, 10.1], [76.7, 10.1], [76.7, 10.2], [76.6, 10.2], [76.6, 10.1]]],
    }
    region = validate_geojson_geometry(geojson)
    assert region.area_km2 == pytest.approx(121.0, rel=0.05)
    assert region.bbox.as_tuple() == (76.6, 10.1, 76.7, 10.2)


def test_self_intersecting_polygon_is_repaired_or_rejected():
    bowtie = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
    }
    # Either outcome is acceptable; silently accepting an invalid ring is not.
    try:
        region = validate_geojson_geometry(bowtie)
        assert region.geometry.is_valid
    except GeometryValidationError:
        pass


def test_out_of_range_latitude_is_rejected():
    """Latitude-first coordinates are caught whenever they exceed the pole.

    Note the real limitation this documents: when both values happen to be
    valid latitudes and longitudes, swapped axis order produces a legal
    geometry in the wrong place and cannot be detected from coordinates alone.
    GeoJSON mandates longitude-first, which is what the API contract states.
    """
    geojson = {
        "type": "Polygon",
        "coordinates": [[[10.1, 96.6], [10.2, 96.6], [10.2, 96.7], [10.1, 96.7], [10.1, 96.6]]],
    }
    with pytest.raises(GeometryValidationError):
        validate_geojson_geometry(geojson)


def test_unsupported_geometry_type_is_rejected():
    with pytest.raises(GeometryValidationError):
        validate_geojson_geometry({"type": "Point", "coordinates": [76.6, 10.1]})


def test_feature_wrapper_is_unwrapped():
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[76.6, 10.1], [76.65, 10.1], [76.65, 10.15], [76.6, 10.15], [76.6, 10.1]]
            ],
        },
    }
    assert validate_geojson_geometry(feature).area_km2 > 0


def test_region_extent_limits_are_enforced():
    with pytest.raises(GeometryValidationError, match="too large"):
        validate_region(bbox=[0.0, 0.0, 10.0, 10.0])
    with pytest.raises(GeometryValidationError, match="too small"):
        validate_region(bbox=[0.0, 0.0, 0.0001, 0.0001])


def test_region_requires_a_selector():
    with pytest.raises(GeometryValidationError):
        validate_region()


# --- CRS ---------------------------------------------------------------------
@pytest.mark.parametrize("value", ["EPSG:4326", "epsg:4326", "CRS84", "OGC:CRS84", None])
def test_supported_crs_normalises_to_wgs84(value):
    assert normalise_crs(value) == "EPSG:4326"


def test_unsupported_crs_is_rejected():
    with pytest.raises(GeometryValidationError):
        normalise_crs("EPSG:3857")


def test_geodesic_area_is_latitude_aware():
    """A degree box near the pole covers far less ground than one at the equator."""
    from shapely.geometry import box

    equator = geodesic_area_km2(box(0, 0, 1, 1))
    high_latitude = geodesic_area_km2(box(0, 60, 1, 61))
    assert equator > high_latitude * 1.8


# --- raster grid ---------------------------------------------------------------
def test_grid_reports_geometry_and_resolution(red_grid):
    assert red_grid.shape == (8, 8)
    assert red_grid.resolution == (10.0, 10.0)
    assert red_grid.crs == UTM
    left, bottom, right, top = red_grid.bounds
    assert right - left == pytest.approx(80.0)
    assert top - bottom == pytest.approx(80.0)


def test_grid_rejects_non_two_dimensional_data():
    with pytest.raises(RasterProcessingError):
        RasterGrid(
            data=np.ma.masked_array(np.zeros((2, 3, 4), dtype="float32")),
            transform=BASE_TRANSFORM,
            crs=UTM,
        )


def test_pixel_area_matches_projected_resolution(red_grid):
    assert red_grid.pixel_area_m2() == pytest.approx(100.0)


def test_pixel_area_in_geographic_crs_is_metric():
    grid = make_grid(
        np.zeros((4, 4), dtype="float32"),
        transform=Affine(0.001, 0, 76.6, 0, -0.001, 10.2),
        crs=CRS.from_epsg(4326),
    )
    # ~0.001 degrees is roughly 110 m; area should be within an order of magnitude.
    assert 8_000 < grid.pixel_area_m2() < 16_000


def test_alignment_detection_and_reprojection():
    a = make_grid(np.ones((8, 8), dtype="float32"))
    shifted = make_grid(
        np.ones((8, 8), dtype="float32"),
        transform=Affine(10.0, 0.0, 500_040.0, 0.0, -10.0, 1_130_000.0),
    )
    assert not a.is_aligned_with(shifted)
    aligned = shifted.reproject_to(a)
    assert aligned.is_aligned_with(a)


def test_common_valid_mask_requires_alignment():
    a = make_grid(np.ones((8, 8), dtype="float32"))
    other_crs = make_grid(np.ones((8, 8), dtype="float32"), crs=CRS.from_epsg(32644))
    with pytest.raises(RasterProcessingError):
        common_valid_mask(a, other_crs)


def test_common_valid_mask_intersects_masks():
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[:, 0] = True
    valid = common_valid_mask(
        make_grid(np.ones((4, 4), dtype="float32"), mask=mask_a),
        make_grid(np.ones((4, 4), dtype="float32"), mask=mask_b),
    )
    assert valid.sum() == 9
    assert not valid[0, 0]


def test_align_grids_returns_reference_first():
    reference = make_grid(np.ones((4, 4), dtype="float32"))
    other = make_grid(
        np.ones((4, 4), dtype="float32"),
        transform=Affine(20.0, 0.0, 500_000.0, 0.0, -20.0, 1_130_000.0),
    )
    aligned = align_grids(reference, other)
    assert aligned[0] is reference
    assert aligned[1].is_aligned_with(reference)


def test_clip_to_geometry_masks_outside_and_preserves_extent(region):
    grid = make_grid(np.ones((16, 16), dtype="float32"))
    clipped = grid.clip_to_geometry(region.geometry)
    assert clipped.shape == grid.shape  # extent preserved for comparability
    assert clipped.transform == grid.transform


def test_downsampling_respects_max_dimension():
    grid = make_grid(np.ones((100, 60), dtype="float32"))
    reduced = grid.resampled_to_max_dimension(50)
    assert max(reduced.shape) <= 50
    assert reduced.crs == grid.crs


def test_window_for_bounds_is_clamped_to_dataset():
    window = window_for_bounds(
        BASE_TRANSFORM, (100, 100), (400_000.0, 1_100_000.0, 600_000.0, 1_200_000.0)
    )
    assert window.col_off >= 0
    assert window.row_off >= 0
    assert window.col_off + window.width <= 100
    assert window.row_off + window.height <= 100


def test_grid_metadata_is_serialisable(red_grid):
    metadata = red_grid.metadata()
    assert metadata["crs"] == "EPSG:32643"
    assert metadata["valid_pixels"] == 64
    assert len(metadata["bounds_wgs84"]) == 4
    assert len(metadata["transform"]) == 6
