"""Train / validation / test splitting strategies.

The primary experiment turns on the difference between these two:

* **Random pixel split** — samples are shuffled and divided without regard to
  where they came from. Neighbouring 10 m pixels are strongly spatially
  autocorrelated, so a held-out pixel almost always sits beside a training
  pixel from the same field or canopy.
* **Spatial holdout** — whole regions are held out, so no test pixel shares a
  landscape, a scene or a spatial block with any training pixel.

Both are produced from the same cached samples under the same seed, which is
what makes the comparison controlled. Every split returns its assignment plus
the block membership needed to verify that no leakage occurred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from research.dataset import ResearchDataset

#: Regions reserved for evaluation in the spatial protocol. These mirror the
#: production model's holdout and are never trained on.
SPATIAL_TEST_REGIONS: tuple[str, ...] = ("murray_basin", "iberian_meseta", "zambezi_miombo")

#: A further region withheld from training to serve as the tuning/validation
#: block, so hyperparameter choices never touch the test blocks.
SPATIAL_VAL_REGIONS: tuple[str, ...] = ("finnish_lakeland",)

RANDOM_VAL_FRACTION = 0.15
RANDOM_TEST_FRACTION = 0.20


@dataclass(slots=True)
class Split:
    """Index arrays for one train/validation/test partition."""

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int
    strategy: str
    #: Block identity per partition, used by the leakage tests and reports.
    train_blocks: tuple[str, ...] = ()
    val_blocks: tuple[str, ...] = ()
    test_blocks: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "seed": self.seed,
            "sizes": self.sizes(),
            "train_blocks": list(self.train_blocks),
            "val_blocks": list(self.val_blocks),
            "test_blocks": list(self.test_blocks),
            **self.detail,
        }


def random_split(
    dataset: ResearchDataset,
    seed: int,
    *,
    subset: np.ndarray | None = None,
    val_fraction: float = RANDOM_VAL_FRACTION,
    test_fraction: float = RANDOM_TEST_FRACTION,
) -> Split:
    """Shuffle all eligible pixels and divide them without spatial regard.

    This is the optimistic protocol under test, not a recommendation.
    """
    pool = np.flatnonzero(subset) if subset is not None else np.arange(dataset.n)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pool)

    n = shuffled.size
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]

    blocks = tuple(sorted(set(dataset.region[pool].tolist())))
    return Split(
        name=f"random_seed{seed}",
        train=train,
        val=val,
        test=test,
        seed=seed,
        strategy="random_pixel",
        # Every block appears in all three partitions by construction — this is
        # exactly the property that makes the estimate optimistic, so it is
        # recorded rather than hidden.
        train_blocks=blocks,
        val_blocks=blocks,
        test_blocks=blocks,
        detail={
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "blocks_shared_across_partitions": True,
        },
    )


def spatial_split(
    dataset: ResearchDataset,
    seed: int,
    *,
    subset: np.ndarray | None = None,
    test_regions: tuple[str, ...] = SPATIAL_TEST_REGIONS,
    val_regions: tuple[str, ...] = SPATIAL_VAL_REGIONS,
) -> Split:
    """Hold out whole regions, so no test pixel shares a block with training."""
    eligible = subset if subset is not None else np.ones(dataset.n, dtype=bool)
    region = dataset.region

    test_mask = eligible & np.isin(region, list(test_regions))
    val_mask = eligible & np.isin(region, list(val_regions))
    train_mask = eligible & ~test_mask & ~val_mask

    rng = np.random.default_rng(seed)
    train = rng.permutation(np.flatnonzero(train_mask))

    return Split(
        name=f"spatial_seed{seed}",
        train=train,
        val=np.flatnonzero(val_mask),
        test=np.flatnonzero(test_mask),
        seed=seed,
        strategy="spatial_holdout",
        train_blocks=tuple(sorted(set(region[train_mask].tolist()))),
        val_blocks=tuple(sorted(set(region[val_mask].tolist()))),
        test_blocks=tuple(sorted(set(region[test_mask].tolist()))),
        detail={"blocks_shared_across_partitions": False},
    )


def temporal_split(
    dataset: ResearchDataset,
    seed: int,
    *,
    train_period: str,
    test_period: str,
    spatial_holdout: bool = True,
) -> Split:
    """Train on one observation period and test on another.

    When ``spatial_holdout`` is set the transfer is also geographic, which
    separates temporal shift measured on familiar ground from temporal shift
    compounded by unfamiliar ground.
    """
    region = dataset.region
    period = dataset.period

    train_mask = period == train_period
    test_mask = period == test_period
    if spatial_holdout:
        train_mask &= ~np.isin(region, list(SPATIAL_TEST_REGIONS))
        test_mask &= np.isin(region, list(SPATIAL_TEST_REGIONS))
    else:
        # Same ground, different date: isolates temporal shift alone.
        train_mask &= np.isin(region, list(SPATIAL_TEST_REGIONS))
        test_mask &= np.isin(region, list(SPATIAL_TEST_REGIONS))

    rng = np.random.default_rng(seed)
    train = rng.permutation(np.flatnonzero(train_mask))

    return Split(
        name=f"temporal_{train_period}_to_{test_period}_seed{seed}",
        train=train,
        val=np.array([], dtype=int),
        test=np.flatnonzero(test_mask),
        seed=seed,
        strategy="temporal_transfer",
        train_blocks=tuple(sorted(set(region[train_mask].tolist()))),
        test_blocks=tuple(sorted(set(region[test_mask].tolist()))),
        detail={
            "train_period": train_period,
            "test_period": test_period,
            "spatial_holdout": spatial_holdout,
        },
    )


# --- verification -----------------------------------------------------------
def spatial_leakage(dataset: ResearchDataset, split: Split) -> set[str]:
    """Blocks appearing in both the training and test partitions."""
    train_blocks = set(dataset.region[split.train].tolist())
    test_blocks = set(dataset.region[split.test].tolist())
    return train_blocks & test_blocks


def temporal_leakage(dataset: ResearchDataset, split: Split) -> set[str]:
    """Periods appearing in both the training and test partitions."""
    train_periods = set(dataset.period[split.train].tolist())
    test_periods = set(dataset.period[split.test].tolist())
    return train_periods & test_periods


def index_overlap(split: Split) -> int:
    """Samples appearing in more than one partition. Must always be zero."""
    train, val, test = set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())
    return len(train & val) + len(train & test) + len(val & test)
