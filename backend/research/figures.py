"""Publication figures.

One style is applied to every figure so a paper's plates read as a set: same
typeface, same grid treatment, same class colours as the product's map legend
and charts. Figures are written to both the shared figure directory and the
originating experiment's directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.models_ml.labels import CLASS_COLOURS, CLASS_LABELS, CLASS_ORDER
from research.config import FIGURE_DIR, RESULTS_DIR

CLASS_NAMES = [CLASS_LABELS[int(c)] for c in CLASS_ORDER]
CLASS_HEX = [CLASS_COLOURS[int(c)] for c in CLASS_ORDER]

#: Protocol colours, kept consistent across every figure that compares them.
PROTOCOL_COLOURS = {
    "random_pixel": "#c8734f",
    "spatial_holdout": "#1d4b33",
    "temporal_transfer": "#2b6cb0",
}

DPI = 200


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def save(fig: plt.Figure, name: str, experiment: str | None = None) -> list[Path]:
    """Write a figure to the shared directory and the experiment directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{name}.png"]
    if experiment:
        target = RESULTS_DIR / experiment / "figures" / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        paths.append(target)
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
        )


def grouped_metric_bars(
    groups: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    name: str,
    experiment: str | None = None,
    errors: dict[str, list[float]] | None = None,
    colours: list[str] | None = None,
    ylim: tuple[float, float] = (0, 1),
) -> list[Path]:
    """Grouped bars — the workhorse for comparing protocols and models."""
    apply_style()
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(groups)), 3.6))
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    x = np.arange(len(groups))

    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        err = errors.get(label) if errors else None
        bars = ax.bar(
            x + offset,
            values,
            width * 0.92,
            label=label,
            yerr=err,
            capsize=3 if err else 0,
            color=(colours[i] if colours and i < len(colours) else None),
        )
        if n_series <= 3:
            _annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0 if max(len(g) for g in groups) < 12 else 20, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    if n_series > 1:
        ax.legend()
    return save(fig, name, experiment)


def confusion_figure(
    matrix: list[list[int]],
    *,
    title: str,
    name: str,
    experiment: str | None = None,
    normalise: bool = True,
) -> list[Path]:
    """Confusion matrix with a fixed class order and honest normalisation."""
    apply_style()
    data = np.asarray(matrix, dtype=float)
    if normalise:
        row_sums = data.sum(axis=1, keepdims=True)
        # Rows with no support stay zero rather than becoming NaN blocks.
        shown = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
        fmt, vmax = "{:.2f}", 1.0
    else:
        shown, fmt, vmax = data, "{:.0f}", float(data.max() or 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(shown, cmap="Greens", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.grid(False)

    threshold = vmax * 0.55
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(shown[i, j]),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if shown[i, j] > threshold else "#1b211d",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Row-normalised share" if normalise else "Pixels")
    return save(fig, name, experiment)


def class_distribution_figure(
    counts: dict[str, int], *, title: str, name: str, experiment: str | None = None
) -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    names = [n for n in CLASS_NAMES if n in counts]
    values = [counts[n] for n in names]
    bars = ax.bar(names, values, color=[CLASS_HEX[CLASS_NAMES.index(n)] for n in names])
    _annotate_bars(ax, bars, "{:.0f}")
    ax.set_ylabel("Sampled pixels")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    return save(fig, name, experiment)


def importance_figure(
    rows: list[dict[str, Any]], *, title: str, name: str, experiment: str | None = None
) -> list[Path]:
    apply_style()
    ordered = sorted(rows, key=lambda r: r["importance"])
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(ordered) + 1.4))
    ax.barh(
        [r["feature"] for r in ordered],
        [r["importance"] for r in ordered],
        color="#1d4b33",
    )
    ax.set_xlabel("Impurity-based importance")
    ax.set_title(title)
    return save(fig, name, experiment)


def line_figure(
    x: list[float],
    series: dict[str, list[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    name: str,
    experiment: str | None = None,
) -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for label, values in series.items():
        ax.plot(x, values, marker="o", markersize=4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    return save(fig, name, experiment)


def reliability_figure(
    buckets: list[dict[str, Any]], *, title: str, name: str, experiment: str | None = None
) -> list[Path]:
    """Confidence against observed accuracy, with the ideal diagonal shown."""
    apply_style()
    usable = [b for b in buckets if b.get("n")]
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(5.0, 4.8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    if usable:
        centres = [float(b["bucket"].split("-")[0]) + 0.1 for b in usable]
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#8a8f98",
            linewidth=1,
            label="Perfect calibration",
        )
        ax.plot(
            [b["mean_confidence"] for b in usable],
            [b["accuracy"] for b in usable],
            marker="o",
            color="#1d4b33",
            label="Observed",
        )
        ax_hist.bar(centres, [b["n"] for b in usable], width=0.16, color="#7fa650")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax_hist.set_xlabel("Predicted confidence")
    ax_hist.set_ylabel("Pixels")
    ax_hist.set_xlim(0, 1)
    return save(fig, name, experiment)


def spatial_split_figure(
    lon: np.ndarray,
    lat: np.ndarray,
    assignment: np.ndarray,
    *,
    title: str,
    name: str,
    experiment: str | None = None,
) -> list[Path]:
    """World map of sampled regions coloured by split assignment."""
    apply_style()
    colours = {"train": "#1d4b33", "validation": "#d3a03c", "test": "#c8734f"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for part, colour in colours.items():
        mask = assignment == part
        if not mask.any():
            continue
        ax.scatter(
            lon[mask],
            lat[mask],
            s=26,
            c=colour,
            label=part.capitalize(),
            edgecolors="none",
            alpha=0.9,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)
    return save(fig, name, experiment)


def pipeline_figure(name: str = "fig01_pipeline", experiment: str | None = None) -> list[Path]:
    """Schematic of the experimental pipeline."""
    apply_style()
    stages = [
        "Sentinel-2 L2A\nscene search",
        "Cloud & quality\nmasking",
        "Feature\nconstruction",
        "WorldCover\nreference labels",
        "Split protocol\n(random / spatial /\ntemporal)",
        "Model fit\n& evaluation",
        "Metrics, figures,\nexperiment record",
    ]
    fig, ax = plt.subplots(figsize=(9.5, 2.3))
    ax.axis("off")
    width, gap = 1.0, 0.32
    for i, stage in enumerate(stages):
        x = i * (width + gap)
        ax.add_patch(
            plt.Rectangle((x, 0), width, 1, facecolor="#f4efe4", edgecolor="#1d4b33", linewidth=1.1)
        )
        ax.text(x + width / 2, 0.5, stage, ha="center", va="center", fontsize=7.4)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.9, 0.5),
                xytext=(x + width + gap * 0.1, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#1d4b33", "linewidth": 1.1},
            )
    ax.set_xlim(-0.1, len(stages) * (width + gap))
    ax.set_ylim(-0.1, 1.1)
    return save(fig, name, experiment)
