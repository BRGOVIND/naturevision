"""Metric computation for the research experiments.

Overall accuracy is reported but never alone: with five imbalanced classes it
hides minority-class failure entirely, so balanced accuracy, macro-F1 and full
per-class figures accompany every result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from app.models_ml.labels import CLASS_LABELS, CLASS_ORDER

CLASS_IDS: list[int] = [int(c) for c in CLASS_ORDER]
CLASS_NAMES: list[str] = [CLASS_LABELS[c] for c in CLASS_IDS]


@dataclass(slots=True)
class ClassificationMetrics:
    """A complete evaluation of one set of predictions."""

    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    n_samples: int
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def headline(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Score predictions over the full, fixed class space.

    Classes absent from a particular split still appear in the confusion matrix
    and per-class table with zero support, so matrices stay comparable across
    experiments.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        macro_precision=round(float(np.mean(precision)), 4),
        macro_recall=round(float(np.mean(recall)), 4),
        macro_f1=round(
            float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)), 4
        ),
        weighted_f1=round(
            float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)),
            4,
        ),
        n_samples=int(y_true.size),
        per_class={
            name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        confusion=confusion_matrix(y_true, y_pred, labels=CLASS_IDS).tolist(),
        support={name: int(support[i]) for i, name in enumerate(CLASS_NAMES)},
    )


def aggregate(runs: list[ClassificationMetrics]) -> dict[str, dict[str, float]]:
    """Mean, standard deviation, min and max of headline metrics across seeds.

    A single seed is never reported as if it expressed uncertainty.
    """
    if not runs:
        return {}
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([getattr(r, key) for r in runs], dtype=float)
        summary[key] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "n_runs": int(values.size),
        }
    return summary


def class_distribution(labels: np.ndarray) -> dict[str, Any]:
    """Counts, proportions and balanced class weights for a label array."""
    total = int(labels.size)
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for class_id, count in zip(*np.unique(labels, return_counts=True), strict=True):
        counts[CLASS_LABELS[int(class_id)]] = int(count)

    present = sum(1 for v in counts.values() if v > 0)
    return {
        "total": total,
        "counts": counts,
        "proportions": {k: round(v / total, 6) if total else 0.0 for k, v in counts.items()},
        "balanced_weights": {
            k: round(total / (present * v), 4) if v > 0 else None for k, v in counts.items()
        },
        "imbalance_ratio": (
            round(max(counts.values()) / min(v for v in counts.values() if v > 0), 3)
            if present
            else None
        ),
    }


def confidence_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    buckets: tuple[tuple[float, float], ...],
) -> dict[str, Any]:
    """Accuracy by confidence bucket, plus a calibration error.

    Expected calibration error is reported so the text can state whether
    probabilities are calibrated rather than assuming they are.
    """
    correct = (y_true == y_pred).astype(float)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    total = int(confidence.size)

    for low, high in buckets:
        in_bucket = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        n = int(in_bucket.sum())
        if n == 0:
            rows.append(
                {
                    "bucket": f"{low:.1f}-{high:.1f}",
                    "n": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                    "gap": None,
                }
            )
            continue
        acc = float(correct[in_bucket].mean())
        mean_conf = float(confidence[in_bucket].mean())
        ece += (n / total) * abs(acc - mean_conf)
        rows.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "n": n,
                "accuracy": round(acc, 4),
                "mean_confidence": round(mean_conf, 4),
                "gap": round(mean_conf - acc, 4),
            }
        )

    correct_conf = confidence[correct == 1]
    wrong_conf = confidence[correct == 0]
    return {
        "buckets": rows,
        "expected_calibration_error": round(float(ece), 4),
        "mean_confidence_correct": (
            round(float(correct_conf.mean()), 4) if correct_conf.size else None
        ),
        "mean_confidence_incorrect": (
            round(float(wrong_conf.mean()), 4) if wrong_conf.size else None
        ),
        "overall_accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "note": (
            "Expected calibration error is a bucketed estimate. A positive gap "
            "means the model is overconfident in that bucket."
        ),
    }
