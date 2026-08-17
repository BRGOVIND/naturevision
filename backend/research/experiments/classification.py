"""Classification experiments.

Baseline reproduction, the primary validation-protocol comparison, model
comparison, feature ablation, temporal transfer, confidence analysis, feature
importance and class imbalance.

Every experiment reads the same cached pixels and differs only in the split, the
feature subset or the estimator, so differences between results are attributable
to the variable under test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.logging import get_logger
from research import figures
from research.artifacts import ExperimentRecord, write_confusion, write_table
from research.config import CONFIG
from research.dataset import ResearchDataset, load_dataset
from research.metrics import (
    CLASS_NAMES,
    ClassificationMetrics,
    aggregate,
    class_distribution,
    confidence_analysis,
    evaluate,
)
from research.models import feature_importances, fit_model
from research.splits import (
    SPATIAL_TEST_REGIONS,
    Split,
    index_overlap,
    random_split,
    spatial_leakage,
    spatial_split,
    temporal_split,
)

logger = get_logger(__name__)

#: The production model's published spatially held-out result, used as the
#: reproduction target. Not modified by any experiment.
PUBLISHED_BASELINE = {"accuracy": 0.5922, "macro_f1": 0.5268, "evaluation_samples": 82_617}

#: Reproduction is judged against this absolute tolerance on accuracy and
#: macro-F1. The research dataset resamples pixels independently of the
#: production training run, so an exact match is not expected; a gap larger
#: than this indicates a genuine discrepancy rather than sampling noise.
BASELINE_TOLERANCE = 0.08


def _fit_and_score(
    dataset: ResearchDataset,
    split: Split,
    *,
    model: str,
    features: tuple[str, ...],
    seed: int,
) -> tuple[ClassificationMetrics, Any]:
    """Fit on the training partition and score on the test partition."""
    x = dataset.columns(features)
    fitted = fit_model(model, x[split.train], dataset.labels[split.train], seed)
    predictions, _ = fitted.timed_inference(x[split.test])
    metrics = evaluate(dataset.labels[split.test], predictions)
    return metrics, fitted


def _assert_no_leakage(dataset: ResearchDataset, split: Split) -> None:
    """Refuse to report a spatial result that shares a block across partitions."""
    overlap = index_overlap(split)
    if overlap:
        raise AssertionError(f"{split.name}: {overlap} samples appear in multiple partitions")
    if split.strategy == "spatial_holdout":
        shared = spatial_leakage(dataset, split)
        if shared:
            raise AssertionError(f"{split.name}: spatial leakage across blocks {sorted(shared)}")


# --------------------------------------------------------------------------
# Baseline reproduction
# --------------------------------------------------------------------------
def run_baseline(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Reproduce the production model's spatially held-out result.

    Runs the production configuration — random forest, full feature set,
    geographically disjoint holdout — and compares against the published
    figures before any other experiment is trusted.
    """
    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    # The production model was trained on the 2021 window, so the baseline
    # reproduction uses the same period rather than the pooled dataset.
    subset = dataset.mask_for(periods=("p2021",))

    runs: list[ClassificationMetrics] = []
    rows: list[dict[str, Any]] = []
    last_metrics: ClassificationMetrics | None = None

    for seed in CONFIG.seeds:
        split = spatial_split(dataset, seed, subset=subset)
        _assert_no_leakage(dataset, split)
        metrics, _ = _fit_and_score(
            dataset, split, model=CONFIG.baseline_model, features=features, seed=seed
        )
        runs.append(metrics)
        last_metrics = metrics
        rows.append({"seed": seed, **metrics.headline(), "test_samples": metrics.n_samples})
        logger.info(
            "baseline_seed", seed=seed, accuracy=metrics.accuracy, macro_f1=metrics.macro_f1
        )

    summary = aggregate(runs)
    accuracy_gap = round(summary["accuracy"]["mean"] - PUBLISHED_BASELINE["accuracy"], 4)
    macro_gap = round(summary["macro_f1"]["mean"] - PUBLISHED_BASELINE["macro_f1"], 4)
    reproduced = abs(accuracy_gap) <= BASELINE_TOLERANCE and abs(macro_gap) <= BASELINE_TOLERANCE

    record = ExperimentRecord(
        experiment="baseline",
        seeds=list(CONFIG.seeds),
        config={
            "model": CONFIG.baseline_model,
            "feature_set": CONFIG.baseline_feature_set,
            "features": list(features),
            "protocol": "spatial_holdout",
            "period": "p2021",
            "test_regions": list(SPATIAL_TEST_REGIONS),
            "tolerance": BASELINE_TOLERANCE,
        },
        dataset_manifest=dataset.manifest,
        results={
            "published_baseline": PUBLISHED_BASELINE,
            "reproduced": summary,
            "accuracy_gap": accuracy_gap,
            "macro_f1_gap": macro_gap,
            "within_tolerance": reproduced,
            "per_seed": rows,
            "per_class": last_metrics.per_class if last_metrics else {},
        },
        notes=[
            "The research dataset resamples pixels independently of the production "
            "training run, so an exact match is not expected; the comparison is "
            "against a documented tolerance.",
        ],
        limitations=[
            "The published baseline was trained on a separately drawn pixel sample; "
            "differences within tolerance reflect sampling, not a change in method.",
        ],
    )

    write_table(rows, "table04a_baseline_per_seed", "baseline")
    if last_metrics:
        write_confusion(last_metrics.confusion, CLASS_NAMES, "confusion_matrix", "baseline")
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 1 — random vs spatial validation (primary)
# --------------------------------------------------------------------------
def run_spatial_validation(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Compare a random pixel split against a geographically disjoint holdout.

    Both protocols use the same pixels, the same model and the same seeds, so
    any difference is attributable to the validation strategy alone.
    """
    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    subset = dataset.mask_for(periods=("p2021",))

    protocols: dict[str, list[ClassificationMetrics]] = {"random_pixel": [], "spatial_holdout": []}
    rows: list[dict[str, Any]] = []
    confusions: dict[str, list[list[int]]] = {}
    split_records: list[dict[str, Any]] = []

    for seed in CONFIG.seeds:
        for name, builder in (
            ("random_pixel", lambda s: random_split(dataset, s, subset=subset)),
            ("spatial_holdout", lambda s: spatial_split(dataset, s, subset=subset)),
        ):
            split = builder(seed)
            _assert_no_leakage(dataset, split)
            metrics, _ = _fit_and_score(
                dataset, split, model=CONFIG.baseline_model, features=features, seed=seed
            )
            protocols[name].append(metrics)
            confusions[name] = metrics.confusion
            split_records.append(split.to_dict())
            rows.append(
                {
                    "protocol": name,
                    "seed": seed,
                    **metrics.headline(),
                    "train_samples": int(split.train.size),
                    "val_samples": int(split.val.size),
                    "test_samples": int(split.test.size),
                    "n_test_blocks": len(split.test_blocks),
                }
            )
            logger.info(
                "validation_protocol",
                protocol=name,
                seed=seed,
                accuracy=metrics.accuracy,
                macro_f1=metrics.macro_f1,
            )

    summaries = {name: aggregate(runs) for name, runs in protocols.items()}
    difference = {
        metric: round(
            summaries["random_pixel"][metric]["mean"]
            - summaries["spatial_holdout"][metric]["mean"],
            4,
        )
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }

    comparison_rows = [
        {
            "metric": metric,
            "random_pixel_mean": summaries["random_pixel"][metric]["mean"],
            "random_pixel_std": summaries["random_pixel"][metric]["std"],
            "spatial_holdout_mean": summaries["spatial_holdout"][metric]["mean"],
            "spatial_holdout_std": summaries["spatial_holdout"][metric]["std"],
            "difference": difference[metric],
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    ]

    record = ExperimentRecord(
        experiment="spatial_validation",
        seeds=list(CONFIG.seeds),
        config={
            "model": CONFIG.baseline_model,
            "feature_set": CONFIG.baseline_feature_set,
            "protocols": list(protocols),
            "period": "p2021",
            "test_regions": list(SPATIAL_TEST_REGIONS),
            "splits": split_records,
        },
        dataset_manifest=dataset.manifest,
        results={
            "summary": summaries,
            "difference_random_minus_spatial": difference,
            "per_run": rows,
            "hypothesis": (
                "H1: random pixel validation overestimates performance relative to "
                "spatial holdout validation."
            ),
            "h1_direction_supported": difference["macro_f1"] > 0 and difference["accuracy"] > 0,
        },
        limitations=[
            "Both protocols evaluate against ESA WorldCover reference labels, which "
            "carry their own error.",
            "The spatial protocol holds out three regions; with a larger region pool "
            "the estimate would be more stable.",
        ],
    )

    write_table(rows, "table04_random_vs_spatial_runs", "spatial_validation")
    write_table(comparison_rows, "table04b_random_vs_spatial_summary", "spatial_validation")
    for name, matrix in confusions.items():
        write_confusion(matrix, CLASS_NAMES, f"confusion_{name}", "spatial_validation")

    figures.grouped_metric_bars(
        ["Accuracy", "Balanced acc.", "Macro F1", "Weighted F1"],
        {
            "Random pixel split": [
                summaries["random_pixel"][m]["mean"]
                for m in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            ],
            "Spatial holdout": [
                summaries["spatial_holdout"][m]["mean"]
                for m in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            ],
        },
        errors={
            "Random pixel split": [
                summaries["random_pixel"][m]["std"]
                for m in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            ],
            "Spatial holdout": [
                summaries["spatial_holdout"][m]["std"]
                for m in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            ],
        },
        title="Validation protocol comparison (mean ± s.d. over seeds)",
        ylabel="Score",
        name="fig03_random_vs_spatial",
        experiment="spatial_validation",
        colours=[
            figures.PROTOCOL_COLOURS["random_pixel"],
            figures.PROTOCOL_COLOURS["spatial_holdout"],
        ],
    )
    for name, matrix in confusions.items():
        figures.confusion_figure(
            matrix,
            title=f"Confusion matrix — {name.replace('_', ' ')}",
            name=f"fig10_confusion_{name}",
            experiment="spatial_validation",
        )

    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 2 — model comparison
# --------------------------------------------------------------------------
def run_model_comparison(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Compare lightweight models under one fixed spatial protocol."""
    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    subset = dataset.mask_for(periods=("p2021",))

    rows: list[dict[str, Any]] = []
    per_model: dict[str, list[ClassificationMetrics]] = {}
    confusions: dict[str, list[list[int]]] = {}

    for model_name in CONFIG.models:
        per_model[model_name] = []
        for seed in CONFIG.seeds:
            split = spatial_split(dataset, seed, subset=subset)
            _assert_no_leakage(dataset, split)
            try:
                metrics, fitted = _fit_and_score(
                    dataset, split, model=model_name, features=features, seed=seed
                )
            except Exception as exc:
                logger.error("model_failed", model=model_name, seed=seed, error=str(exc)[:200])
                continue
            per_model[model_name].append(metrics)
            confusions[model_name] = metrics.confusion
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    **metrics.headline(),
                    "train_seconds": fitted.train_seconds,
                    "inference_seconds": fitted.inference_seconds,
                    "train_samples": fitted.n_train,
                    "test_samples": metrics.n_samples,
                }
            )
            logger.info(
                "model_scored",
                model=model_name,
                seed=seed,
                accuracy=metrics.accuracy,
                macro_f1=metrics.macro_f1,
            )

    summaries = {name: aggregate(runs) for name, runs in per_model.items() if runs}
    ordered = sorted(summaries, key=lambda m: summaries[m]["macro_f1"]["mean"], reverse=True)

    summary_rows = [
        {
            "model": name,
            "accuracy_mean": summaries[name]["accuracy"]["mean"],
            "accuracy_std": summaries[name]["accuracy"]["std"],
            "balanced_accuracy_mean": summaries[name]["balanced_accuracy"]["mean"],
            "macro_f1_mean": summaries[name]["macro_f1"]["mean"],
            "macro_f1_std": summaries[name]["macro_f1"]["std"],
            "weighted_f1_mean": summaries[name]["weighted_f1"]["mean"],
            "mean_train_seconds": round(
                float(np.mean([r["train_seconds"] for r in rows if r["model"] == name])), 3
            ),
            "mean_inference_seconds": round(
                float(np.mean([r["inference_seconds"] for r in rows if r["model"] == name])), 4
            ),
        }
        for name in ordered
    ]

    record = ExperimentRecord(
        experiment="model_comparison",
        seeds=list(CONFIG.seeds),
        config={
            "models": dict(CONFIG.models),
            "feature_set": CONFIG.baseline_feature_set,
            "protocol": "spatial_holdout",
            "period": "p2021",
        },
        dataset_manifest=dataset.manifest,
        results={
            "summary": summaries,
            "ranked_by_macro_f1": ordered,
            "per_run": rows,
            "note": (
                "Ranking is by macro-F1 rather than accuracy, and training and "
                "inference cost are reported alongside, so model selection is not "
                "reduced to a single number."
            ),
        },
        limitations=[
            "Hyperparameters are fixed in advance and identical across seeds; no "
            "model received tuning, so these are out-of-the-box comparisons.",
            "The linear SVM is a calibrated linear model, not an RBF-kernel SVM, "
            "which is infeasible at this sample count on a workstation.",
        ],
    )

    write_table(rows, "table03a_model_runs", "model_comparison")
    write_table(summary_rows, "table03_model_comparison", "model_comparison")
    for name, matrix in confusions.items():
        write_confusion(matrix, CLASS_NAMES, f"confusion_{name}", "model_comparison")

    if summaries:
        figures.grouped_metric_bars(
            ordered,
            {
                "Accuracy": [summaries[m]["accuracy"]["mean"] for m in ordered],
                "Macro F1": [summaries[m]["macro_f1"]["mean"] for m in ordered],
            },
            errors={
                "Accuracy": [summaries[m]["accuracy"]["std"] for m in ordered],
                "Macro F1": [summaries[m]["macro_f1"]["std"] for m in ordered],
            },
            title="Model comparison under spatial holdout (mean ± s.d.)",
            ylabel="Score",
            name="fig04_model_comparison",
            experiment="model_comparison",
        )
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 3 — feature ablation
# --------------------------------------------------------------------------
def run_feature_ablation(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Compare feature groups under one fixed model and split."""
    dataset = dataset or load_dataset()
    subset = dataset.mask_for(periods=("p2021",))

    rows: list[dict[str, Any]] = []
    per_set: dict[str, list[ClassificationMetrics]] = {}

    for set_name, features in CONFIG.feature_sets.items():
        per_set[set_name] = []
        for seed in CONFIG.seeds:
            # Identical split object per seed across feature sets, so the only
            # thing that varies is the feature subset.
            split = spatial_split(dataset, seed, subset=subset)
            _assert_no_leakage(dataset, split)
            metrics, _ = _fit_and_score(
                dataset, split, model=CONFIG.baseline_model, features=features, seed=seed
            )
            per_set[set_name].append(metrics)
            rows.append(
                {
                    "feature_set": set_name,
                    "n_features": len(features),
                    "features": " ".join(features),
                    "seed": seed,
                    **metrics.headline(),
                }
            )
            logger.info("ablation", feature_set=set_name, seed=seed, macro_f1=metrics.macro_f1)

    summaries = {name: aggregate(runs) for name, runs in per_set.items()}
    names = list(CONFIG.feature_sets)
    reference = summaries[CONFIG.baseline_feature_set]["macro_f1"]["mean"]

    summary_rows = [
        {
            "feature_set": name,
            "n_features": len(CONFIG.feature_sets[name]),
            "accuracy_mean": summaries[name]["accuracy"]["mean"],
            "macro_f1_mean": summaries[name]["macro_f1"]["mean"],
            "macro_f1_std": summaries[name]["macro_f1"]["std"],
            "delta_macro_f1_vs_full": round(summaries[name]["macro_f1"]["mean"] - reference, 4),
        }
        for name in names
    ]

    record = ExperimentRecord(
        experiment="feature_ablation",
        seeds=list(CONFIG.seeds),
        config={
            "feature_sets": {k: list(v) for k, v in CONFIG.feature_sets.items()},
            "model": CONFIG.baseline_model,
            "protocol": "spatial_holdout",
            "index_definitions": {
                "ndvi": "(NIR - Red) / (NIR + Red)",
                "ndwi": "(Green - NIR) / (Green + NIR)",
                "ndbi": "(SWIR1 - NIR) / (SWIR1 + NIR)",
                "nbr": "(NIR - SWIR2) / (NIR + SWIR2)",
                "bsi_partial": "(SWIR1 - Blue) / (SWIR1 + Blue)",
            },
        },
        dataset_manifest=dataset.manifest,
        results={"summary": summaries, "per_run": rows, "comparison": summary_rows},
        limitations=[
            "All indices are normalised differences of bands the pipeline actually "
            "reads; no index requiring an unavailable band is included.",
        ],
    )

    write_table(rows, "table05a_feature_ablation_runs", "feature_ablation")
    write_table(summary_rows, "table05_feature_ablation", "feature_ablation")
    figures.grouped_metric_bars(
        names,
        {
            "Accuracy": [summaries[n]["accuracy"]["mean"] for n in names],
            "Macro F1": [summaries[n]["macro_f1"]["mean"] for n in names],
        },
        errors={
            "Accuracy": [summaries[n]["accuracy"]["std"] for n in names],
            "Macro F1": [summaries[n]["macro_f1"]["std"] for n in names],
        },
        title="Feature ablation under spatial holdout (mean ± s.d.)",
        ylabel="Score",
        name="fig05_feature_ablation",
        experiment="feature_ablation",
    )
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 4 — temporal transfer
# --------------------------------------------------------------------------
def run_temporal_transfer(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Measure performance when training and testing periods differ."""
    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    available = sorted(set(dataset.period.tolist()))

    if len(available) < 2:
        raise RuntimeError(
            f"Temporal transfer needs two periods; the dataset contains {available}. "
            "Rebuild the dataset before running this experiment."
        )

    pairs = [
        ("p2021", "p2021", "same-period"),
        ("p2021", "p2024", "forward transfer"),
        ("p2024", "p2024", "same-period"),
        ("p2024", "p2021", "backward transfer"),
    ]
    rows: list[dict[str, Any]] = []
    per_pair: dict[str, list[ClassificationMetrics]] = {}

    for train_period, test_period, kind in pairs:
        if train_period not in available or test_period not in available:
            logger.warning("period_missing", train=train_period, test=test_period)
            continue
        key = f"{train_period}->{test_period}"
        per_pair[key] = []
        for seed in CONFIG.seeds:
            split = temporal_split(
                dataset, seed, train_period=train_period, test_period=test_period
            )
            if split.train.size == 0 or split.test.size == 0:
                continue
            _assert_no_leakage(dataset, split)
            # Same-period pairs are still spatially disjoint, so a drop between
            # same-period and cross-period is temporal, not geographic.
            metrics, _ = _fit_and_score(
                dataset, split, model=CONFIG.baseline_model, features=features, seed=seed
            )
            per_pair[key].append(metrics)
            rows.append(
                {
                    "train_period": train_period,
                    "test_period": test_period,
                    "kind": kind,
                    "seed": seed,
                    **metrics.headline(),
                    "train_samples": int(split.train.size),
                    "test_samples": int(split.test.size),
                }
            )
            logger.info("temporal", pair=key, seed=seed, macro_f1=metrics.macro_f1)

    summaries = {k: aggregate(v) for k, v in per_pair.items() if v}
    keys = list(summaries)

    record = ExperimentRecord(
        experiment="temporal_transfer",
        seeds=list(CONFIG.seeds),
        config={
            "pairs": [{"train": a, "test": b, "kind": k} for a, b, k in pairs],
            "model": CONFIG.baseline_model,
            "feature_set": CONFIG.baseline_feature_set,
            "spatial_holdout": True,
        },
        dataset_manifest=dataset.manifest,
        results={"summary": summaries, "per_run": rows},
        limitations=[
            "Reference labels are fixed at the 2021 epoch. Genuine land-cover change "
            "between the periods appears as label noise in the later period, so a "
            "measured drop conflates temporal domain shift with label staleness.",
            "Both periods use the same seasonal window, which limits how much of the "
            "result is attributable to phenology.",
        ],
    )

    write_table(rows, "table06_temporal_transfer", "temporal_transfer")
    if keys:
        figures.grouped_metric_bars(
            keys,
            {
                "Accuracy": [summaries[k]["accuracy"]["mean"] for k in keys],
                "Macro F1": [summaries[k]["macro_f1"]["mean"] for k in keys],
            },
            errors={
                "Accuracy": [summaries[k]["accuracy"]["std"] for k in keys],
                "Macro F1": [summaries[k]["macro_f1"]["std"] for k in keys],
            },
            title="Temporal transfer (train period → test period)",
            ylabel="Score",
            name="fig06_temporal_transfer",
            experiment="temporal_transfer",
        )
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 7 — confidence and error
# --------------------------------------------------------------------------
def run_confidence_analysis(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Test whether predicted confidence is informative about correctness."""
    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    subset = dataset.mask_for(periods=("p2021",))
    seed = CONFIG.seeds[0]

    split = spatial_split(dataset, seed, subset=subset)
    _assert_no_leakage(dataset, split)

    x = dataset.columns(features)
    fitted = fit_model(CONFIG.baseline_model, x[split.train], dataset.labels[split.train], seed)
    probabilities = fitted.predict_proba(x[split.test])
    if probabilities is None:
        raise RuntimeError(f"{CONFIG.baseline_model} does not expose probabilities.")

    predictions = np.argmax(probabilities, axis=1).astype("int16")
    confidence = probabilities.max(axis=1)
    truth = dataset.labels[split.test]

    analysis = confidence_analysis(truth, predictions, confidence, CONFIG.confidence_buckets)
    metrics = evaluate(truth, predictions)

    # Error summary: which confusions dominate, and how confident they were.
    wrong = predictions != truth
    error_rows: list[dict[str, Any]] = []
    for true_id in range(len(CLASS_NAMES)):
        for pred_id in range(len(CLASS_NAMES)):
            if true_id == pred_id:
                continue
            mask = wrong & (truth == true_id) & (predictions == pred_id)
            if not mask.any():
                continue
            error_rows.append(
                {
                    "true_class": CLASS_NAMES[true_id],
                    "predicted_class": CLASS_NAMES[pred_id],
                    "count": int(mask.sum()),
                    "share_of_errors": round(float(mask.sum() / max(1, wrong.sum())), 4),
                    "mean_confidence": round(float(confidence[mask].mean()), 4),
                }
            )
    error_rows.sort(key=lambda r: r["count"], reverse=True)

    record = ExperimentRecord(
        experiment="confidence_analysis",
        seeds=[seed],
        config={
            "model": CONFIG.baseline_model,
            "feature_set": CONFIG.baseline_feature_set,
            "protocol": "spatial_holdout",
            "buckets": [list(b) for b in CONFIG.confidence_buckets],
        },
        dataset_manifest=dataset.manifest,
        results={
            "confidence": analysis,
            "metrics": metrics.headline(),
            "top_confusions": error_rows[:15],
            "hypothesis": "H5: prediction confidence is informative about correctness.",
            "confidence_separates_correct_and_incorrect": (
                analysis["mean_confidence_correct"] is not None
                and analysis["mean_confidence_incorrect"] is not None
                and analysis["mean_confidence_correct"] > analysis["mean_confidence_incorrect"]
            ),
        },
        limitations=[
            "Expected calibration error is estimated from five buckets; a finer "
            "binning would give a more precise figure.",
            "Confidence is the maximum class probability of a random forest, which "
            "is a vote share rather than a calibrated posterior unless shown otherwise.",
        ],
    )

    write_table(analysis["buckets"], "table09_confidence_buckets", "confidence_analysis")
    write_table(error_rows, "table09b_top_confusions", "confidence_analysis")
    figures.reliability_figure(
        analysis["buckets"],
        title="Confidence against observed accuracy (spatial holdout)",
        name="fig11_confidence_reliability",
        experiment="confidence_analysis",
    )
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 8 — feature importance
# --------------------------------------------------------------------------
def run_feature_importance(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Impurity and permutation importance for the tree-based baseline."""
    from sklearn.inspection import permutation_importance

    dataset = dataset or load_dataset()
    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    subset = dataset.mask_for(periods=("p2021",))
    seed = CONFIG.seeds[0]

    split = spatial_split(dataset, seed, subset=subset)
    _assert_no_leakage(dataset, split)
    x = dataset.columns(features)
    fitted = fit_model(CONFIG.baseline_model, x[split.train], dataset.labels[split.train], seed)

    impurity = feature_importances(fitted, features)

    # Permutation importance is computed on the validation block, never the
    # test block, so it cannot inform anything measured on the test set.
    perm_rows: list[dict[str, Any]] = []
    if split.val.size:
        sample = min(20_000, split.val.size)
        rng = np.random.default_rng(seed)
        picked = rng.choice(split.val, size=sample, replace=False)
        result = permutation_importance(
            fitted.estimator,
            x[picked],
            dataset.labels[picked],
            n_repeats=5,
            random_state=seed,
            scoring="f1_macro",
            n_jobs=1,
        )
        perm_rows = sorted(
            (
                {
                    "feature": name,
                    "importance": round(float(m), 6),
                    "std": round(float(s), 6),
                }
                for name, m, s in zip(
                    features, result.importances_mean, result.importances_std, strict=True
                )
            ),
            key=lambda r: r["importance"],
            reverse=True,
        )
        for rank, row in enumerate(perm_rows, start=1):
            row["rank"] = rank

    record = ExperimentRecord(
        experiment="feature_importance",
        seeds=[seed],
        config={
            "model": CONFIG.baseline_model,
            "feature_set": CONFIG.baseline_feature_set,
            "permutation_scoring": "f1_macro",
            "permutation_split": "validation block",
        },
        dataset_manifest=dataset.manifest,
        results={"impurity_importance": impurity, "permutation_importance": perm_rows},
        limitations=[
            "Impurity importance is biased toward high-cardinality continuous "
            "features and is correlated across the spectral bands.",
            "Permutation importance is measured on the held-out validation block, "
            "not the test block, to avoid informing test-set results.",
        ],
    )

    write_table(impurity, "table11_feature_importance", "feature_importance")
    if perm_rows:
        write_table(perm_rows, "table11b_permutation_importance", "feature_importance")
    if impurity:
        figures.importance_figure(
            impurity,
            title="Random forest feature importance (impurity)",
            name="fig09_feature_importance",
            experiment="feature_importance",
        )
    record.write()
    return record


# --------------------------------------------------------------------------
# Experiment 9 — class imbalance
# --------------------------------------------------------------------------
def run_class_imbalance(dataset: ResearchDataset | None = None) -> ExperimentRecord:
    """Describe the class distribution and its effect on headline metrics."""
    dataset = dataset or load_dataset()
    subset = dataset.mask_for(periods=("p2021",))
    seed = CONFIG.seeds[0]
    split = spatial_split(dataset, seed, subset=subset)

    overall = class_distribution(dataset.labels)
    train_dist = class_distribution(dataset.labels[split.train])
    test_dist = class_distribution(dataset.labels[split.test])

    features = CONFIG.feature_sets[CONFIG.baseline_feature_set]
    metrics, _ = _fit_and_score(
        dataset, split, model=CONFIG.baseline_model, features=features, seed=seed
    )

    rows = [
        {
            "class": name,
            "dataset_count": overall["counts"][name],
            "dataset_proportion": overall["proportions"][name],
            "train_count": train_dist["counts"][name],
            "test_count": test_dist["counts"][name],
            "balanced_weight": overall["balanced_weights"][name],
            "test_f1": metrics.per_class[name]["f1"],
            "test_recall": metrics.per_class[name]["recall"],
            "test_precision": metrics.per_class[name]["precision"],
        }
        for name in CLASS_NAMES
    ]

    record = ExperimentRecord(
        experiment="class_imbalance",
        seeds=[seed],
        config={"protocol": "spatial_holdout", "model": CONFIG.baseline_model},
        dataset_manifest=dataset.manifest,
        results={
            "dataset": overall,
            "train": train_dist,
            "test": test_dist,
            "per_class_performance": metrics.per_class,
            "headline": metrics.headline(),
            "accuracy_minus_balanced_accuracy": round(
                metrics.accuracy - metrics.balanced_accuracy, 4
            ),
            "note": (
                "The gap between accuracy and balanced accuracy is the amount by "
                "which overall accuracy flatters performance on the minority classes."
            ),
        },
        limitations=[
            "Pixels are sampled with class stratification, so these proportions "
            "describe the experimental dataset, not the landscape.",
        ],
    )

    write_table(rows, "table02_class_distribution", "class_imbalance")
    figures.class_distribution_figure(
        overall["counts"],
        title="Class distribution in the research dataset",
        name="fig02b_class_distribution",
        experiment="class_imbalance",
    )
    record.write()
    return record
