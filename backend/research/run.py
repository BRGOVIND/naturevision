"""Research experiment runner.

    python -m research.run --experiment core       # baseline + main experiments
    python -m research.run --experiment smoke      # fast CI check
    python -m research.run --experiment dataset    # (re)build the pixel cache
    python -m research.run --list                  # show everything available

Experiments read the cached dataset; nothing downloads imagery unless the
dataset or an imagery-level experiment is explicitly requested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from app.core.logging import configure_logging, get_logger
from research import figures
from research.artifacts import ExperimentRecord, write_table
from research.config import CONFIG, RESULTS_DIR, ensure_directories

logger = get_logger(__name__)


def _classification():
    from research.experiments import classification

    return classification


def _robustness():
    from research.experiments import robustness

    return robustness


def _grounding():
    from research.experiments import grounding

    return grounding


#: Experiment name -> callable. Imports are deferred so `--list` and the smoke
#: test do not pay for scikit-learn or matplotlib unnecessarily.
EXPERIMENTS: dict[str, Callable[[], ExperimentRecord]] = {
    "baseline": lambda: _classification().run_baseline(),
    "spatial_validation": lambda: _classification().run_spatial_validation(),
    "model_comparison": lambda: _classification().run_model_comparison(),
    "feature_ablation": lambda: _classification().run_feature_ablation(),
    "temporal_transfer": lambda: _classification().run_temporal_transfer(),
    "confidence_analysis": lambda: _classification().run_confidence_analysis(),
    "feature_importance": lambda: _classification().run_feature_importance(),
    "class_imbalance": lambda: _classification().run_class_imbalance(),
    "cloud_robustness": lambda: _robustness().run_cloud_robustness(),
    "threshold_sensitivity": lambda: _robustness().run_threshold_sensitivity(),
    "llm_grounding": lambda: _grounding().run_grounding_evaluation(),
}

#: The core suite: everything that runs from the cached dataset alone.
CORE: tuple[str, ...] = (
    "baseline",
    "spatial_validation",
    "model_comparison",
    "feature_ablation",
    "temporal_transfer",
    "confidence_analysis",
    "feature_importance",
    "class_imbalance",
)

#: Experiments that fetch imagery or call a hosted provider at run time.
NETWORK: tuple[str, ...] = ("cloud_robustness", "threshold_sensitivity", "llm_grounding")


def run_dataset(seed: int) -> int:
    from research.dataset import build_dataset

    manifest = asyncio.run(build_dataset(seed=seed))
    print(f"Built {len(manifest['groups'])} groups, {manifest['total_samples']:,} samples")
    return 0


def run_smoke() -> int:
    """Fast structural check of the research pipeline, safe for CI.

    Verifies configuration, split construction, leakage prevention, metric
    computation and artifact writing on a small slice. It does not fetch
    imagery and does not call a language provider.
    """
    import numpy as np

    from research.dataset import load_dataset
    from research.metrics import evaluate
    from research.splits import (
        index_overlap,
        random_split,
        spatial_leakage,
        spatial_split,
        temporal_leakage,
        temporal_split,
    )

    checks: list[tuple[str, bool, str]] = []

    checks.append(("config loads", bool(CONFIG.feature_sets and CONFIG.models), ""))
    checks.append(
        (
            "feature sets reference real features",
            all(
                f
                in (
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
                for fs in CONFIG.feature_sets.values()
                for f in fs
            ),
            "",
        )
    )

    try:
        dataset = load_dataset()
    except FileNotFoundError as exc:
        print(json.dumps({"status": "skipped", "reason": str(exc)}, indent=2))
        # Absent cache is not a CI failure: the structural checks above passed
        # and the dataset is built explicitly, never implicitly.
        return 0

    checks.append(("dataset non-empty", dataset.n > 0, f"{dataset.n} samples"))
    checks.append(("dataset carries blocks", len(set(dataset.region.tolist())) > 1, ""))

    seed = CONFIG.seeds[0]
    subset = dataset.mask_for(periods=("p2021",))

    spatial = spatial_split(dataset, seed, subset=subset)
    checks.append(("spatial split has test samples", spatial.test.size > 0, ""))
    checks.append(("no spatial leakage", not spatial_leakage(dataset, spatial), ""))
    checks.append(("no index overlap (spatial)", index_overlap(spatial) == 0, ""))

    rnd = random_split(dataset, seed, subset=subset)
    checks.append(("no index overlap (random)", index_overlap(rnd) == 0, ""))
    checks.append(
        (
            "random split is reproducible",
            bool(np.array_equal(rnd.test, random_split(dataset, seed, subset=subset).test)),
            "",
        )
    )

    periods = sorted(set(dataset.period.tolist()))
    if len(periods) >= 2:
        temporal = temporal_split(dataset, seed, train_period=periods[0], test_period=periods[1])
        checks.append(("no temporal leakage", not temporal_leakage(dataset, temporal), ""))

    truth = dataset.labels[spatial.test][:500]
    metrics = evaluate(truth, truth)
    checks.append(("metrics on perfect predictions", metrics.accuracy == 1.0, ""))
    checks.append(
        ("confusion matrix is square", len(metrics.confusion) == len(metrics.confusion[0]), "")
    )

    failed = [name for name, ok, _ in checks if not ok]
    for name, ok, note in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{f' — {note}' if note else ''}")

    ensure_directories()
    write_table(
        [{"check": n, "passed": ok, "note": note} for n, ok, note in checks],
        "smoke_checks",
        "smoke",
    )
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _study_region_figure() -> None:
    """Map the study regions coloured by their spatial split assignment.

    Drawn from the dataset manifest, so it always depicts the regions that were
    actually collected rather than the regions that were intended.
    """
    import numpy as np

    from research.dataset import load_dataset
    from research.splits import SPATIAL_TEST_REGIONS, SPATIAL_VAL_REGIONS

    try:
        dataset = load_dataset()
    except FileNotFoundError:
        return

    regions = sorted(set(dataset.region.tolist()))
    lon, lat, assignment = [], [], []
    for region in regions:
        mask = dataset.region == region
        lon.append(float(np.mean(dataset.lon[mask])))
        lat.append(float(np.mean(dataset.lat[mask])))
        if region in SPATIAL_TEST_REGIONS:
            assignment.append("test")
        elif region in SPATIAL_VAL_REGIONS:
            assignment.append("validation")
        else:
            assignment.append("train")

    figures.spatial_split_figure(
        np.array(lon),
        np.array(lat),
        np.array(assignment),
        title="Study regions by spatial split assignment",
        name="fig02_study_regions",
    )


def run_named(names: tuple[str, ...]) -> int:
    """Run a set of experiments, reporting each outcome honestly."""
    ensure_directories()
    figures.pipeline_figure()
    _study_region_figure()

    outcomes: list[dict[str, Any]] = []
    failures = 0

    for name in names:
        started = time.perf_counter()
        logger.info("experiment_started", experiment=name)
        try:
            record = EXPERIMENTS[name]()
            elapsed = round(time.perf_counter() - started, 2)
            outcomes.append(
                {
                    "experiment": name,
                    "status": "completed",
                    "runtime_seconds": elapsed,
                    "output": str(record.directory),
                }
            )
            print(f"  [done] {name} in {elapsed}s -> {record.directory}")
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 2)
            failures += 1
            outcomes.append(
                {
                    "experiment": name,
                    "status": "failed",
                    "runtime_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            # A blocked experiment is reported, never replaced with a made-up result.
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            logger.error("experiment_failed", experiment=name, error=str(exc)[:400])

    write_table(outcomes, "experiment_runs")
    summary = RESULTS_DIR / "run_summary.json"
    summary.write_text(json.dumps(outcomes, indent=2, default=str), encoding="utf-8")
    print(f"\n{len(names) - failures}/{len(names)} experiments completed. Summary: {summary}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NatureVision research experiment runner.")
    parser.add_argument(
        "--experiment",
        default="core",
        help="Experiment name, or one of: core, all, network, smoke, dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=CONFIG.seeds[0], help="Seed for dataset builds."
    )
    parser.add_argument("--list", action="store_true", help="List available experiments.")
    args = parser.parse_args(argv)

    configure_logging()

    if args.list:
        print("Experiments:")
        for name in EXPERIMENTS:
            marker = " (network)" if name in NETWORK else ""
            print(f"  {name}{marker}")
        print("\nGroups: core, network, all, smoke, dataset")
        return 0

    choice = args.experiment
    if choice == "dataset":
        return run_dataset(args.seed)
    if choice == "smoke":
        return run_smoke()
    if choice == "core":
        return run_named(CORE)
    if choice == "network":
        return run_named(NETWORK)
    if choice == "all":
        return run_named(CORE + NETWORK)
    if choice in EXPERIMENTS:
        return run_named((choice,))

    print(f"Unknown experiment '{choice}'. Use --list to see the options.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
