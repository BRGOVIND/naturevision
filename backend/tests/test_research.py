"""Research framework tests.

These guard the properties the experimental results depend on: reproducible
seeds, leakage-free splits, correct metric computation and complete experiment
metadata. Synthetic arrays are used where the property under test is structural
— a leakage test does not need real reflectance, it needs known block
membership — and no production path reads them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.models_ml.labels import CLASS_ORDER
from research.artifacts import ExperimentRecord, config_hash, write_table
from research.config import CONFIG, FEATURE_SETS, PERIODS, RESEARCH_VERSION
from research.dataset import ResearchDataset
from research.metrics import (
    CLASS_NAMES,
    aggregate,
    class_distribution,
    confidence_analysis,
    evaluate,
)
from research.splits import (
    SPATIAL_TEST_REGIONS,
    index_overlap,
    random_split,
    spatial_leakage,
    spatial_split,
    temporal_leakage,
    temporal_split,
)

FEATURE_NAMES = (
    "blue",
    "green",
    "red",
    "nir",
    "swir16",
    "swir22",
    "ndvi",
    "ndwi",
    "ndbi",
    "nbr",
    "bsi_partial",
)


def make_dataset(seed: int = 0, per_group: int = 400) -> ResearchDataset:
    """A structurally realistic dataset: known blocks, periods and classes."""
    rng = np.random.default_rng(seed)
    regions = [
        "western_ghats",
        "po_valley",
        "brandenburg",
        "finnish_lakeland",
        *SPATIAL_TEST_REGIONS,
    ]
    periods = [p.key for p in PERIODS]

    features, labels, region, period, lon, lat = [], [], [], [], [], []
    for r_index, r in enumerate(regions):
        for p in periods:
            features.append(rng.normal(size=(per_group, len(FEATURE_NAMES))).astype("float32"))
            labels.append(rng.integers(0, len(CLASS_ORDER), size=per_group).astype("int16"))
            region.append(np.full(per_group, r))
            period.append(np.full(per_group, p))
            lon.append(np.full(per_group, -170 + r_index * 40, dtype="float32"))
            lat.append(np.full(per_group, -40 + r_index * 15, dtype="float32"))

    return ResearchDataset(
        features=np.vstack(features),
        labels=np.concatenate(labels),
        region=np.concatenate(region),
        period=np.concatenate(period),
        lon=np.concatenate(lon),
        lat=np.concatenate(lat),
        feature_names=FEATURE_NAMES,
        manifest={"dataset_version": "test", "groups": [], "cache_sha256": "test"},
    )


# --- configuration ----------------------------------------------------------
def test_research_version_is_recorded():
    assert RESEARCH_VERSION
    assert CONFIG.research_version == RESEARCH_VERSION


def test_feature_sets_only_reference_real_features():
    """No feature set may invent a band or an index the pipeline does not build."""
    for name, features in FEATURE_SETS.items():
        unknown = [f for f in features if f not in FEATURE_NAMES]
        assert not unknown, f"{name} references features that do not exist: {unknown}"


def test_feature_sets_are_nested_by_design():
    assert set(FEATURE_SETS["A_raw_bands"]) < set(FEATURE_SETS["B_bands_ndvi"])
    assert set(FEATURE_SETS["B_bands_ndvi"]) < set(FEATURE_SETS["C_bands_indices"])
    assert set(FEATURE_SETS["C_bands_indices"]) < set(FEATURE_SETS["D_full"])


def test_label_source_is_versioned_explicitly():
    """WorldCover epoch and algorithm version must never be left ambiguous."""
    source = CONFIG.label_source
    assert source["version"] == "v200"
    assert source["epoch"] == "2021"
    assert "2021 v200" in source["identifier"]


# --- splits -----------------------------------------------------------------
def test_random_split_partitions_are_disjoint_and_complete():
    dataset = make_dataset()
    split = random_split(dataset, seed=42)
    assert index_overlap(split) == 0
    assert split.train.size + split.val.size + split.test.size == dataset.n


def test_random_split_is_reproducible_for_a_seed():
    dataset = make_dataset()
    a = random_split(dataset, seed=42)
    b = random_split(dataset, seed=42)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.train, b.train)


def test_random_split_differs_between_seeds():
    dataset = make_dataset()
    assert not np.array_equal(
        random_split(dataset, seed=42).test, random_split(dataset, seed=123).test
    )


def test_spatial_split_has_no_leakage():
    """CRITICAL: no held-out test sample may come from a training block."""
    dataset = make_dataset()
    for seed in CONFIG.seeds:
        split = spatial_split(dataset, seed)
        assert not spatial_leakage(dataset, split), f"spatial leakage at seed {seed}"
        assert index_overlap(split) == 0


def test_spatial_split_test_blocks_are_the_configured_regions():
    dataset = make_dataset()
    split = spatial_split(dataset, seed=42)
    assert set(split.test_blocks) == set(SPATIAL_TEST_REGIONS)
    assert not set(split.train_blocks) & set(split.test_blocks)


def test_spatial_leakage_is_detected_when_introduced():
    """The leakage check must fail loudly if a test block enters training."""
    dataset = make_dataset()
    split = spatial_split(dataset, seed=42)
    # Deliberately contaminate: move one test index into the training set.
    contaminated = split
    contaminated.train = np.concatenate([split.train, split.test[:1]])
    assert spatial_leakage(dataset, contaminated)


def test_random_split_shares_blocks_across_partitions():
    """The optimistic protocol's defining property, asserted rather than assumed."""
    dataset = make_dataset()
    split = random_split(dataset, seed=42)
    assert spatial_leakage(dataset, split), "a random split should share blocks"
    assert split.detail["blocks_shared_across_partitions"] is True


def test_temporal_split_has_no_period_leakage():
    """CRITICAL: a transfer experiment must not train on the target period."""
    dataset = make_dataset()
    split = temporal_split(dataset, seed=42, train_period="p2021", test_period="p2024")
    assert not temporal_leakage(dataset, split)
    assert set(dataset.period[split.train].tolist()) == {"p2021"}
    assert set(dataset.period[split.test].tolist()) == {"p2024"}


def test_temporal_split_is_also_spatially_disjoint_by_default():
    dataset = make_dataset()
    split = temporal_split(dataset, seed=42, train_period="p2021", test_period="p2024")
    assert not spatial_leakage(dataset, split)


def test_same_period_temporal_split_is_reported_as_sharing_period():
    dataset = make_dataset()
    split = temporal_split(dataset, seed=42, train_period="p2021", test_period="p2021")
    assert temporal_leakage(dataset, split) == {"p2021"}


# --- feature selection -------------------------------------------------------
def test_column_selection_preserves_requested_order():
    dataset = make_dataset()
    selected = dataset.columns(("red", "blue"))
    assert selected.shape[1] == 2
    assert np.array_equal(selected[:, 0], dataset.features[:, FEATURE_NAMES.index("red")])
    assert np.array_equal(selected[:, 1], dataset.features[:, FEATURE_NAMES.index("blue")])


def test_unknown_feature_is_rejected():
    dataset = make_dataset()
    with pytest.raises(KeyError):
        dataset.columns(("red", "not_a_feature"))


def test_feature_selection_never_includes_labels():
    """Guards against label leakage into the feature matrix."""
    dataset = make_dataset()
    for features in FEATURE_SETS.values():
        assert "label" not in features
        assert dataset.columns(features).shape[1] == len(features)


# --- metrics -----------------------------------------------------------------
def test_perfect_predictions_score_one():
    truth = np.array([0, 1, 2, 3, 4] * 20, dtype="int16")
    metrics = evaluate(truth, truth)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0


def test_confusion_matrix_covers_the_full_class_space():
    """Absent classes still occupy a row and column, so matrices stay comparable."""
    truth = np.zeros(50, dtype="int16")
    metrics = evaluate(truth, truth)
    assert len(metrics.confusion) == len(CLASS_ORDER)
    assert all(len(row) == len(CLASS_ORDER) for row in metrics.confusion)
    assert set(metrics.per_class) == set(CLASS_NAMES)


def test_accuracy_and_balanced_accuracy_diverge_under_imbalance():
    """The reason accuracy is never reported alone."""
    truth = np.array([0] * 95 + [1] * 5, dtype="int16")
    predicted = np.zeros(100, dtype="int16")  # predicts the majority class only
    metrics = evaluate(truth, predicted)
    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.balanced_accuracy < 0.55
    assert metrics.macro_f1 < 0.3


def test_aggregate_reports_spread_not_a_single_value():
    runs = [
        evaluate(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])),
        evaluate(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 0])),
    ]
    summary = aggregate(runs)
    assert summary["accuracy"]["n_runs"] == 2
    assert summary["accuracy"]["std"] > 0
    assert summary["accuracy"]["min"] < summary["accuracy"]["max"]


def test_class_distribution_reports_counts_and_weights():
    labels = np.array([0] * 60 + [1] * 30 + [2] * 10, dtype="int16")
    distribution = class_distribution(labels)
    assert distribution["total"] == 100
    assert distribution["counts"]["Forest"] == 60
    assert distribution["proportions"]["Forest"] == pytest.approx(0.6)
    assert distribution["imbalance_ratio"] == pytest.approx(6.0)


def test_confidence_analysis_separates_correct_from_incorrect():
    truth = np.array([0, 0, 1, 1], dtype="int16")
    predicted = np.array([0, 0, 0, 0], dtype="int16")
    confidence = np.array([0.95, 0.9, 0.3, 0.35])
    analysis = confidence_analysis(truth, predicted, confidence, CONFIG.confidence_buckets)
    assert analysis["mean_confidence_correct"] > analysis["mean_confidence_incorrect"]
    assert analysis["expected_calibration_error"] >= 0


def test_empty_confidence_bucket_is_reported_not_invented():
    truth = np.array([0, 0], dtype="int16")
    analysis = confidence_analysis(truth, truth, np.array([0.9, 0.95]), CONFIG.confidence_buckets)
    low = next(b for b in analysis["buckets"] if b["bucket"] == "0.0-0.2")
    assert low["n"] == 0
    assert low["accuracy"] is None


# --- artifacts ----------------------------------------------------------------
def test_experiment_record_captures_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("research.artifacts.RESULTS_DIR", tmp_path)
    record = ExperimentRecord(
        experiment="unit_test",
        seeds=[42],
        config={"model": "random_forest"},
        dataset_manifest={"dataset_version": "1.0.0", "cache_sha256": "abc", "groups": []},
        results={"accuracy": 0.5},
    )
    directory = record.write()

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment"] == "unit_test"
    assert metadata["research_version"] == RESEARCH_VERSION
    assert metadata["dataset_version"] == "1.0.0"
    assert metadata["seeds"] == [42]
    assert metadata["config_hash"]
    assert metadata["dataset_cache_sha256"] == "abc"
    assert "finished_at" in metadata
    assert json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["accuracy"] == 0.5


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_write_table_emits_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr("research.artifacts.TABLE_DIR", tmp_path)
    written = write_table([{"model": "rf", "macro_f1": 0.5}], "unit_table")
    csv_path = next(p for p in written if p.suffix == ".csv")
    json_path = next(p for p in written if p.suffix == ".json")
    assert "macro_f1" in csv_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["model"] == "rf"


# --- grounding evaluation -------------------------------------------------------
def test_grounding_scorer_flags_unsupported_numbers():
    from app.interpretation.schemas import Interpretation
    from research.experiments.grounding import _package, score_output

    evidence = {
        "observed": {"period_a": {"mean_ndvi": 0.702}},
        "data_sources": [{"source_id": "S2B_TEST_1"}],
    }
    package = _package(evidence)
    interpretation = Interpretation.model_validate(
        {
            "summary": "Mean NDVI was 0.702 across the analysed region in this period.",
            "observations": [{"statement": "Mean NDVI was 0.702.", "evidence_key": None}],
            "interpretation": "The index reached 0.911, which was never measured here.",
            "uncertainty": "The comparison carries appreciable uncertainty overall.",
            "limitations": ["Cloud masking removes pixels."],
            "confidence_qualifier": "low",
        }
    )
    score = score_output(interpretation, package, evidence)
    assert score["unsupported_numerical_claims"] >= 1
    assert score["grounding_pass"] is False


def test_grounding_scorer_detects_invented_source_ids():
    from app.interpretation.schemas import Interpretation
    from research.experiments.grounding import _package, score_output

    evidence = {
        "observed": {"period_a": {"mean_ndvi": 0.702}},
        "data_sources": [{"source_id": "S2B_REAL_1"}],
    }
    interpretation = Interpretation.model_validate(
        {
            "summary": "Scene S2B_FAKE_9 was analysed for this region and period.",
            "observations": [{"statement": "Mean NDVI was 0.702.", "evidence_key": None}],
            "interpretation": "The vegetation index is consistent with dense cover.",
            "uncertainty": "Cloud masking limits the comparison in places.",
            "limitations": ["Reference labels carry error."],
            "confidence_qualifier": "low",
        }
    )
    score = score_output(interpretation, _package(evidence), evidence)
    assert "S2B_FAKE_9" in score["invented_source_ids"]
    assert score["source_attribution_correct"] is False
