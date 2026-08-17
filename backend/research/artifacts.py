"""Experiment records, output layout and artifact integrity.

Every run writes a self-describing directory: the configuration it used, the
code commit it ran against, the dataset it read, the seeds, and hashes of the
outputs. That is what makes a number in a table traceable back to the exact
code and data that produced it.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.config import CONFIG, RESULTS_DIR, TABLE_DIR


def git_commit() -> str | None:
    """Current commit SHA, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def config_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a configuration, independent of key ordering."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ExperimentRecord:
    """One experiment run and everything needed to reproduce it."""

    experiment: str
    seeds: list[int]
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    #: Wall-clock cost, stamped by the runner once the experiment returns. Kept
    #: in the record itself so a result directory reports its own cost without
    #: depending on the run ledger, which only describes the latest invocation.
    runtime_seconds: float | None = None

    @property
    def directory(self) -> Path:
        return RESULTS_DIR / self.experiment

    def metadata(self) -> dict[str, Any]:
        manifest = self.dataset_manifest
        return {
            "experiment": self.experiment,
            "research_version": CONFIG.research_version,
            "dataset_version": manifest.get("dataset_version", CONFIG.dataset_version),
            "git_commit": git_commit(),
            "started_at": self.started_at,
            "finished_at": dt.datetime.now(dt.UTC).isoformat(),
            "seeds": self.seeds,
            "runtime_seconds": self.runtime_seconds,
            "config_hash": config_hash(self.config),
            "dataset_manifest_hash": config_hash(manifest) if manifest else None,
            "dataset_cache_sha256": manifest.get("cache_sha256"),
            "scene_ids": sorted(
                {g.get("scene_id") for g in manifest.get("groups", []) if g.get("scene_id")}
            ),
            "label_source": manifest.get("label_source", CONFIG.label_source),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "notes": self.notes,
            "limitations": self.limitations,
        }

    def write(self) -> Path:
        """Persist config, metadata, metrics and any tables for this run."""
        directory = self.directory
        (directory / "figures").mkdir(parents=True, exist_ok=True)
        (directory / "tables").mkdir(parents=True, exist_ok=True)

        (directory / "config.json").write_text(
            json.dumps(self.config, indent=2, default=str), encoding="utf-8"
        )
        (directory / "metrics.json").write_text(
            json.dumps(self.results, indent=2, default=str), encoding="utf-8"
        )

        metadata = self.metadata()
        metadata["outputs"] = self.outputs
        metadata["output_hashes"] = {
            name: file_sha256(directory / name)
            for name in ("config.json", "metrics.json")
            if (directory / name).is_file()
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        return directory

    def stamp_runtime(self, seconds: float) -> None:
        """Record the run's wall-clock cost after the experiment has returned.

        The metadata file is rewritten rather than regenerated, so the hashes
        and timestamps already written for this run are preserved.
        """
        self.runtime_seconds = seconds
        path = self.directory / "metadata.json"
        if not path.is_file():
            return
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["runtime_seconds"] = seconds
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def write_table(rows: list[dict[str, Any]], name: str, experiment: str | None = None) -> list[Path]:
    """Write a result table as CSV and JSON, in the shared and run directories."""
    if not rows:
        return []
    written: list[Path] = []
    targets = [TABLE_DIR / f"{name}.csv"]
    if experiment:
        targets.append(RESULTS_DIR / experiment / "tables" / f"{name}.csv")

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        written.append(target)

        json_target = target.with_suffix(".json")
        json_target.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        written.append(json_target)
    return written


def write_confusion(
    matrix: list[list[int]], class_names: list[str], name: str, experiment: str
) -> Path:
    """Write a confusion matrix as CSV with explicit true/predicted labelling."""
    target = RESULTS_DIR / experiment / "tables" / f"{name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true \\ predicted", *class_names])
        for name_row, row in zip(class_names, matrix, strict=True):
            writer.writerow([name_row, *row])
    return target
