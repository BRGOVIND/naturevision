"""Spectral indices, statistics and change detection."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from app.analysis.change import (
    CLASS_MODERATE_DECREASE,
    CLASS_SIGNIFICANT_DECREASE,
    CLASS_STABLE,
    ChangeThresholds,
    detect_change,
)
from app.analysis.indices import NDVI, compute_ndvi, normalised_difference
from app.analysis.statistics import compute_statistics, summarise_vegetation
from app.core.errors import NoImageryFoundError, RasterProcessingError
from app.imagery.bands import Band
from tests.conftest import make_grid


# --- NDVI --------------------------------------------------------------------
def test_ndvi_matches_the_closed_form(red_grid, nir_grid):
    result = normalised_difference(nir_grid, red_grid, definition=NDVI)
    expected = (0.35 - 0.05) / (0.35 + 0.05)
    assert result.data.compressed() == pytest.approx(expected)


def test_ndvi_from_scene_uses_nir_and_red(scene):
    ndvi = compute_ndvi(scene)
    expected = (0.35 - 0.05) / (0.35 + 0.05)
    assert float(np.mean(ndvi.data.compressed())) == pytest.approx(expected, abs=1e-6)


def test_ndvi_preserves_georeferencing(red_grid, nir_grid):
    result = normalised_difference(nir_grid, red_grid)
    assert result.crs == red_grid.crs
    assert result.transform == red_grid.transform
    assert result.shape == red_grid.shape


def test_ndvi_masks_zero_denominator():
    """Red = -NIR makes the denominator zero; those pixels must not survive."""
    nir = make_grid(np.array([[0.3, 0.0], [0.2, 0.5]], dtype="float32"))
    red = make_grid(np.array([[0.1, 0.0], [-0.2, 0.1]], dtype="float32"))
    result = normalised_difference(nir, red)
    mask = np.ma.getmaskarray(result.data)
    assert mask[0, 1]  # 0 + 0 denominator
    assert mask[1, 0]  # 0.2 + (-0.2) denominator
    assert not mask[0, 0]
    assert np.isfinite(result.data.compressed()).all()


def test_ndvi_propagates_input_nodata():
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    nir = make_grid(np.full((2, 2), 0.3, dtype="float32"), mask=mask)
    red = make_grid(np.full((2, 2), 0.1, dtype="float32"))
    result = normalised_difference(nir, red)
    assert np.ma.getmaskarray(result.data)[0, 0]
    assert result.valid_count == 3


def test_ndvi_rejects_unaligned_inputs():
    nir = make_grid(np.ones((4, 4), dtype="float32"))
    red = make_grid(np.ones((4, 4), dtype="float32"), crs=CRS.from_epsg(32644))
    with pytest.raises(RasterProcessingError, match="not on the same grid"):
        normalised_difference(nir, red)


def test_ndvi_stays_within_physical_range():
    rng = np.random.default_rng(7)
    nir = make_grid(rng.uniform(0.01, 0.6, (32, 32)).astype("float32"))
    red = make_grid(rng.uniform(0.01, 0.6, (32, 32)).astype("float32"))
    values = normalised_difference(nir, red).data.compressed()
    assert values.min() >= -1.0
    assert values.max() <= 1.0


def test_missing_band_raises(scene):
    from app.analysis.indices import IndexDefinition

    bogus = IndexDefinition(
        key="bogus",
        name="Bogus",
        formula="x",
        numerator_positive=Band.RED_EDGE_1,
        numerator_negative=Band.RED,
        description="",
    )
    with pytest.raises(NoImageryFoundError):
        from app.analysis.indices import compute_index

        compute_index(scene, bogus)


# --- statistics ----------------------------------------------------------------
def test_statistics_are_computed_over_valid_pixels_only():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True
    grid = make_grid(np.full((10, 10), 0.5, dtype="float32"), mask=mask)
    stats = compute_statistics(grid)
    assert stats.valid_pixel_count == 90
    assert stats.total_pixel_count == 100
    assert stats.mean == pytest.approx(0.5)
    assert stats.valid_fraction == pytest.approx(0.9)


def test_statistics_on_fully_masked_grid_are_null():
    grid = make_grid(np.ones((4, 4), dtype="float32"), mask=np.ones((4, 4), dtype=bool))
    stats = compute_statistics(grid)
    assert stats.valid_pixel_count == 0
    assert stats.mean is None
    assert stats.std_dev is None
    assert stats.valid_area_km2 == 0.0


def test_dispersion_is_withheld_for_tiny_samples():
    grid = make_grid(np.linspace(0, 1, 9, dtype="float32").reshape(3, 3))
    stats = compute_statistics(grid)
    assert stats.mean is not None
    assert stats.std_dev is None  # fewer than the minimum sample size


def test_valid_area_uses_pixel_ground_area():
    grid = make_grid(np.ones((10, 10), dtype="float32"))
    stats = compute_statistics(grid)
    # 100 pixels at 10 m = 10,000 m^2 = 0.01 km^2
    assert stats.valid_area_km2 == pytest.approx(0.01, rel=1e-6)


def test_vegetation_bins_are_exhaustive_and_disjoint():
    values = np.array([[-0.5, 0.1], [0.3, 0.8]], dtype="float32")
    summary = summarise_vegetation(make_grid(values))
    assert sum(summary.class_percentages.values()) == pytest.approx(100.0, abs=1e-3)
    assert summary.vegetated_fraction == pytest.approx(0.5)


# --- change detection --------------------------------------------------------------
def test_change_detects_direction_and_magnitude():
    a = make_grid(np.full((10, 10), 0.70, dtype="float32"))
    b = make_grid(np.full((10, 10), 0.45, dtype="float32"))
    result = detect_change(a, b)
    assert result.absolute_change == pytest.approx(-0.25, abs=1e-6)
    assert result.decreased_area_percentage == pytest.approx(100.0)
    assert result.increased_area_percentage == 0.0
    labels = {c.label: c.percentage_of_analysed_area for c in result.class_summaries}
    assert labels["Significant decrease"] == pytest.approx(100.0)


def test_change_classes_follow_configured_thresholds():
    a = make_grid(np.full((3, 3), 0.50, dtype="float32"))
    deltas = np.array([[0.0, -0.12, -0.30], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype="float32")
    b = make_grid((0.50 + deltas).astype("float32"))
    result = detect_change(a, b, thresholds=ChangeThresholds(moderate=0.10, significant=0.25))
    codes = np.rint(result.classification.data.filled(0)).astype(int)
    assert codes[0, 0] == CLASS_STABLE
    assert codes[0, 1] == CLASS_MODERATE_DECREASE
    assert codes[0, 2] == CLASS_SIGNIFICANT_DECREASE


def test_thresholds_must_be_ordered():
    with pytest.raises(RasterProcessingError):
        ChangeThresholds(moderate=0.3, significant=0.1)


def test_change_only_uses_pixels_valid_in_both_periods():
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[:, 0] = True
    a = make_grid(np.full((4, 4), 0.6, dtype="float32"), mask=mask_a)
    b = make_grid(np.full((4, 4), 0.4, dtype="float32"), mask=mask_b)
    result = detect_change(a, b)
    assert result.comparable_pixel_count == 9


def test_change_reprojects_second_period_onto_first():
    a = make_grid(np.full((8, 8), 0.6, dtype="float32"))
    b = make_grid(
        np.full((4, 4), 0.4, dtype="float32"),
        transform=Affine(20.0, 0.0, 500_000.0, 0.0, -20.0, 1_130_000.0),
    )
    result = detect_change(a, b)
    assert result.difference.is_aligned_with(a)
    assert result.comparable_pixel_count > 0


def test_relative_change_is_withheld_near_zero_baseline():
    a = make_grid(np.full((6, 6), 0.01, dtype="float32"))
    b = make_grid(np.full((6, 6), 0.05, dtype="float32"))
    result = detect_change(a, b)
    assert result.absolute_change is not None
    assert result.relative_change_percent is None


def test_relative_change_is_reported_for_a_solid_baseline():
    a = make_grid(np.full((6, 6), 0.50, dtype="float32"))
    b = make_grid(np.full((6, 6), 0.25, dtype="float32"))
    result = detect_change(a, b)
    assert result.relative_change_percent == pytest.approx(-50.0, abs=0.01)


def test_change_methodology_is_recorded():
    a = make_grid(np.full((4, 4), 0.5, dtype="float32"))
    b = make_grid(np.full((4, 4), 0.5, dtype="float32"))
    methodology = detect_change(a, b).methodology
    assert "NDVI" in methodology["operation"]
    assert methodology["thresholds"]["moderate"] > 0
    assert "not establish" in methodology["interpretation_scope"]


def test_no_comparable_pixels_yields_null_change():
    a = make_grid(np.ones((4, 4), dtype="float32"), mask=np.ones((4, 4), dtype=bool))
    b = make_grid(np.ones((4, 4), dtype="float32"))
    result = detect_change(a, b)
    assert result.comparable_pixel_count == 0
    assert result.absolute_change is None
    assert result.changed_area_percentage == 0.0
