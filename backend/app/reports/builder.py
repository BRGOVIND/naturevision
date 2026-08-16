"""Nature Intelligence Report assembly.

The report is built from the persisted evidence package plus the validated
interpretation. Its defining property is that every statement is tagged with
where it came from — measurement, model prediction, or generated
interpretation — and those registers are never merged.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.core.config import settings
from app.interpretation.schemas import InterpretationEnvelope
from app.models_ml.labels import CLASS_COLLAPSE_NOTES

Provenance = Literal["observed", "model_prediction", "interpretation", "metadata"]

REPORT_KIND = "Nature Intelligence Report"


@dataclass(slots=True)
class ReportSection:
    """One titled section with an explicit provenance register."""

    key: str
    title: str
    provenance: Provenance
    body: str | None = None
    items: list[str] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROVENANCE_LABELS: dict[str, str] = {
    "observed": "Observed data",
    "model_prediction": "Model prediction",
    "interpretation": "Generated interpretation",
    "metadata": "Technical metadata",
}


def build_report(
    *,
    analysis_id: str,
    evidence: dict[str, Any],
    interpretation: InterpretationEnvelope | None,
    region_name: str,
) -> dict[str, Any]:
    """Assemble the full report structure from stored analysis results."""
    observed = evidence.get("observed", {})
    predictions = evidence.get("model_predictions", {})
    methodology = evidence.get("methodology", {})
    quality = evidence.get("data_quality", {})
    sources = evidence.get("data_sources", [])
    period_a = observed.get("period_a", {})
    period_b = observed.get("period_b")
    change = observed.get("change")
    land_cover = predictions.get("land_cover")

    sections: list[ReportSection] = []

    # 1. Executive summary ------------------------------------------------------------------
    sections.append(
        ReportSection(
            key="executive_summary",
            title="Executive Summary",
            provenance="observed",
            body=_executive_summary(region_name, period_a, period_b, change, land_cover),
            notes=[
                "This summary restates measured values only. Interpretation of "
                "what they may indicate appears in a separate section."
            ],
        )
    )

    # 2. Region -----------------------------------------------------------------------------
    region = evidence.get("region", {})
    sections.append(
        ReportSection(
            key="region",
            title="Region",
            provenance="metadata",
            items=[
                f"Name: {region.get('name', region_name)}",
                f"Bounding box (EPSG:4326): {_fmt_bbox(region.get('bbox'))}",
                f"Area: {_fmt(region.get('area_km2'), ' km²')}",
                f"Coordinate reference system: {region.get('crs', 'EPSG:4326')}",
            ],
        )
    )

    # 3. Observation period -----------------------------------------------------------------
    periods = evidence.get("periods", {})
    period_items = [f"Period A: {periods.get('period_a') or 'not specified'}"]
    if periods.get("period_b"):
        period_items.append(f"Period B: {periods['period_b']}")
    if period_a.get("observation_date"):
        period_items.append(f"Period A acquisition: {period_a['observation_date']}")
    if period_b and period_b.get("observation_date"):
        period_items.append(f"Period B acquisition: {period_b['observation_date']}")
    if quality.get("days_between_observations") is not None:
        period_items.append(
            f"Separation between acquisitions: {quality['days_between_observations']} days"
        )
    sections.append(
        ReportSection(
            key="observation_period",
            title="Observation Period",
            provenance="metadata",
            items=period_items,
        )
    )

    # 4. Data sources -----------------------------------------------------------------------
    sections.append(
        ReportSection(
            key="data_sources",
            title="Data Sources",
            provenance="metadata",
            table=[
                {
                    "Period": source.get("period"),
                    "Provider": source.get("provider"),
                    "Dataset": source.get("dataset"),
                    "Scene": source.get("source_id"),
                    "Acquired": source.get("observation_date"),
                    "Processing level": source.get("processing_level"),
                    "Scene cloud cover": _fmt(source.get("cloud_cover_percent"), "%"),
                    "Licence": source.get("license"),
                }
                for source in sources
            ],
            notes=[
                "Reference labels for the land-cover model: "
                + str(
                    (methodology.get("land_cover_model") or {})
                    .get("label_source", {})
                    .get("name", "not applicable")
                )
            ],
        )
    )

    # 5. Methodology ------------------------------------------------------------------------
    sections.append(
        ReportSection(
            key="methodology",
            title="Methodology",
            provenance="metadata",
            items=_methodology_items(methodology),
        )
    )

    # 6. NDVI statistics --------------------------------------------------------------------
    sections.append(
        ReportSection(
            key="ndvi_statistics",
            title="NDVI Statistics",
            provenance="observed",
            table=_ndvi_table(period_a, period_b),
            notes=[
                "Statistics are computed over pixels that survived cloud, shadow, "
                "cirrus and snow masking within the selected geometry."
            ],
        )
    )

    # 7. Vegetation change ------------------------------------------------------------------
    if change:
        sections.append(
            ReportSection(
                key="vegetation_change",
                title="Vegetation Change",
                provenance="observed",
                items=[
                    f"Mean NDVI, period A: {_fmt(change.get('mean_ndvi_period_a'))}",
                    f"Mean NDVI, period B: {_fmt(change.get('mean_ndvi_period_b'))}",
                    f"Absolute change: {_fmt(change.get('absolute_ndvi_change'), signed=True)}",
                    "Relative change: "
                    + _fmt(change.get("relative_ndvi_change_percent"), "%", signed=True),
                    "Area with change beyond the moderate threshold: "
                    + f"{_fmt(change.get('changed_area_percentage'), '%')} "
                    + f"({_fmt(change.get('changed_area_km2'), ' km²')})",
                    f"Area decreasing: {_fmt(change.get('decreased_area_percentage'), '%')}",
                    f"Area increasing: {_fmt(change.get('increased_area_percentage'), '%')}",
                ],
                notes=[
                    "Mean values are recomputed over pixels valid in both periods, "
                    "so the reported change equals the difference of those means "
                    "on identical pixels.",
                    "A change in a vegetation index is an observation about "
                    "reflectance. It does not by itself identify land-cover "
                    "conversion or any cause.",
                ],
            )
        )

    # 8. Land-cover results -----------------------------------------------------------------
    if land_cover:
        distribution = land_cover.get("distribution_percent", {})
        areas = land_cover.get("area_km2", {})
        confidences = land_cover.get("per_class_mean_confidence", {})
        sections.append(
            ReportSection(
                key="land_cover",
                title="Land-Cover Results",
                provenance="model_prediction",
                table=[
                    {
                        "Class": label,
                        "Share of classified area": _fmt(percentage, "%"),
                        "Area": _fmt(areas.get(label), " km²"),
                        "Mean confidence": _fmt(confidences.get(label)),
                    }
                    for label, percentage in distribution.items()
                ],
                notes=[
                    "These are per-pixel predictions from a statistical model, not measurements.",
                    *CLASS_COLLAPSE_NOTES,
                ],
            )
        )

    # 9. Spatial change ---------------------------------------------------------------------
    if change and change.get("change_classes"):
        sections.append(
            ReportSection(
                key="spatial_change",
                title="Spatial Change",
                provenance="observed",
                table=[
                    {"Change class": label, "Share of comparable area": _fmt(value, "%")}
                    for label, value in change["change_classes"].items()
                ],
                notes=[
                    f"Thresholds applied (absolute NDVI units): moderate "
                    f"{change.get('thresholds', {}).get('moderate')}, significant "
                    f"{change.get('thresholds', {}).get('significant')}.",
                    f"Comparable pixels: {change.get('comparable_pixel_count')}.",
                ],
            )
        )

    # 10. Interpretation --------------------------------------------------------------------
    sections.append(_interpretation_section(interpretation))

    # 11. Confidence ------------------------------------------------------------------------
    sections.append(
        ReportSection(
            key="confidence",
            title="Confidence",
            provenance="metadata",
            items=_confidence_items(land_cover, interpretation, quality),
        )
    )

    # 12. Limitations -----------------------------------------------------------------------
    limitations = list(evidence.get("limitations", []))
    if interpretation and interpretation.interpretation:
        for item in interpretation.interpretation.limitations:
            if item not in limitations:
                limitations.append(item)
    sections.append(
        ReportSection(
            key="limitations",
            title="Limitations",
            provenance="metadata",
            items=limitations,
        )
    )

    # 13. Technical metadata ----------------------------------------------------------------
    sections.append(
        ReportSection(
            key="technical_metadata",
            title="Technical Metadata",
            provenance="metadata",
            items=_technical_items(analysis_id, methodology, quality, land_cover, interpretation),
        )
    )

    return {
        "kind": REPORT_KIND,
        "analysis_id": analysis_id,
        "title": f"{REPORT_KIND}: {region_name}",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "provenance_legend": PROVENANCE_LABELS,
        "sections": [section.to_dict() for section in sections],
    }


# --- section helpers -------------------------------------------------------
def _executive_summary(
    region_name: str,
    period_a: dict[str, Any],
    period_b: dict[str, Any] | None,
    change: dict[str, Any] | None,
    land_cover: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    mean_a = period_a.get("mean_ndvi")
    if mean_a is not None:
        parts.append(
            f"Across {region_name}, mean NDVI on {period_a.get('observation_date')} "
            f"was {mean_a:.3f} over {period_a.get('valid_pixel_count', 0):,} valid pixels "
            f"({_fmt(period_a.get('valid_area_km2'), ' km²')})."
        )
    if period_b and change and change.get("absolute_ndvi_change") is not None:
        delta = change["absolute_ndvi_change"]
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
        parts.append(
            f"Between the two acquisitions, mean NDVI {direction} from "
            f"{change.get('mean_ndvi_period_a'):.3f} to {change.get('mean_ndvi_period_b'):.3f} "
            f"({delta:+.3f})."
        )
        parts.append(
            f"{_fmt(change.get('changed_area_percentage'), '%')} of the comparable area "
            "changed by at least the moderate threshold."
        )
    if land_cover:
        distribution = land_cover.get("distribution_percent", {})
        if distribution:
            dominant = max(distribution.items(), key=lambda kv: kv[1])
            parts.append(
                f"The land-cover model predicts {dominant[0].lower()} as the largest "
                f"class at {dominant[1]:.1f}% of classified pixels."
            )
    if not parts:
        parts.append("No summary statistics were produced for this analysis.")
    return " ".join(parts)


def _methodology_items(methodology: dict[str, Any]) -> list[str]:
    items = [
        f"Vegetation index: {methodology.get('index_formula', 'NDVI = (NIR - Red) / (NIR + Red)')}",
    ]
    bands = methodology.get("bands", {})
    if bands:
        items.append(f"Bands: red = {bands.get('red')}, near infrared = {bands.get('nir')}")
    if methodology.get("cloud_masking"):
        items.append(f"Quality masking: {methodology['cloud_masking']}")

    grid = methodology.get("analysis_grid", {})
    if grid:
        items.append(
            f"Analysis grid: {grid.get('width')}x{grid.get('height')} pixels at "
            f"{grid.get('resolution_x', 0):.1f} m in {grid.get('crs')}"
        )
    calibration = methodology.get("radiometric_calibration", {})
    if calibration:
        items.append(
            f"Radiometric calibration: reflectance offset {calibration.get('reflectance_offset')} "
            f"({calibration.get('decision')})"
        )
    change_method = methodology.get("change_detection", {})
    if change_method:
        items.append(f"Change detection: {change_method.get('operation')}")
        items.append(f"Co-registration: {change_method.get('alignment')}")
        items.append(f"Comparable pixels: {change_method.get('comparable_pixels_rule')}")
    model = methodology.get("land_cover_model", {})
    if model:
        items.append(
            f"Land-cover model: {model.get('name')} v{model.get('version')} "
            f"({model.get('backend')})"
        )
        if model.get("evaluation_protocol"):
            items.append(f"Model evaluation protocol: {model['evaluation_protocol']}")
    return items


def _ndvi_table(period_a: dict[str, Any], period_b: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, data in (("Period A", period_a), ("Period B", period_b)):
        if not data:
            continue
        rows.append(
            {
                "Period": label,
                "Acquired": data.get("observation_date"),
                "Mean": _fmt(data.get("mean_ndvi")),
                "Median": _fmt(data.get("median_ndvi")),
                "Min": _fmt(data.get("min_ndvi")),
                "Max": _fmt(data.get("max_ndvi")),
                "Std dev": _fmt(data.get("std_dev_ndvi")),
                "Valid pixels": f"{data.get('valid_pixel_count', 0):,}",
                "Valid area": _fmt(data.get("valid_area_km2"), " km²"),
            }
        )
    return rows


def _interpretation_section(envelope: InterpretationEnvelope | None) -> ReportSection:
    if envelope is None or not envelope.available or envelope.interpretation is None:
        reason = (
            envelope.unavailable_reason
            if envelope and envelope.unavailable_reason
            else "No interpretation was generated for this analysis."
        )
        return ReportSection(
            key="interpretation",
            title="Interpretation",
            provenance="interpretation",
            body=reason,
            notes=["All measured results elsewhere in this report are unaffected."],
        )

    interpretation = envelope.interpretation
    notes = [
        "This section is generated text produced by a language model from the "
        "measured evidence above. It contains no independent measurement.",
        f"Provider: {envelope.provider or 'unknown'}; model: {envelope.model or 'unknown'}.",
    ]
    grounding = envelope.grounding or {}
    if grounding:
        notes.append(
            f"Automated grounding check: {grounding.get('matched_number_count', 0)} of "
            f"{grounding.get('checked_number_count', 0)} numeric statements matched a "
            "measured value."
        )
    if grounding.get("flagged_claims"):
        notes.append(
            "Phrases flagged for review: "
            + "; ".join(sorted({c["issue"] for c in grounding["flagged_claims"]}))
        )
    return ReportSection(
        key="interpretation",
        title="Interpretation",
        provenance="interpretation",
        body=interpretation.summary,
        items=[
            *[f"Observation: {o.statement}" for o in interpretation.observations],
            f"Interpretation: {interpretation.interpretation}",
            f"Uncertainty: {interpretation.uncertainty}",
        ],
        notes=notes,
    )


def _confidence_items(
    land_cover: dict[str, Any] | None,
    envelope: InterpretationEnvelope | None,
    quality: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    if land_cover:
        accuracy = land_cover.get("held_out_overall_accuracy")
        items.append(
            "Land-cover model held-out overall accuracy: "
            + (f"{accuracy:.3f}" if accuracy is not None else "not evaluated")
        )
        macro_f1 = land_cover.get("held_out_macro_f1")
        items.append(
            "Land-cover model held-out macro F1: "
            + (f"{macro_f1:.3f}" if macro_f1 is not None else "not evaluated")
        )
        items.append(f"Mean per-pixel confidence: {_fmt(land_cover.get('mean_confidence'))}")
        items.append(
            f"Pixels below 0.50 confidence: {_fmt(land_cover.get('low_confidence_fraction'))}"
        )
    else:
        items.append("No land-cover model output is included in this analysis.")

    for period in ("a", "b"):
        masked = quality.get(f"period_{period}_masked_fraction")
        if masked is not None:
            items.append(
                f"Period {period.upper()} pixels removed by quality masking: {masked * 100:.2f}%"
            )
    if envelope and envelope.interpretation:
        items.append(
            f"Interpretation confidence qualifier: {envelope.interpretation.confidence_qualifier}"
        )
    return items


def _technical_items(
    analysis_id: str,
    methodology: dict[str, Any],
    quality: dict[str, Any],
    land_cover: dict[str, Any] | None,
    envelope: InterpretationEnvelope | None,
) -> list[str]:
    grid = methodology.get("analysis_grid", {})
    items = [
        f"Analysis identifier: {analysis_id}",
        f"Application: {settings.app_name}",
        f"Analysis grid CRS: {grid.get('crs')}",
        f"Analysis grid transform: {grid.get('transform')}",
        f"Pixel ground area: {_fmt(grid.get('pixel_area_m2'), ' m²')}",
    ]
    if land_cover:
        items += [
            f"Model: {land_cover.get('model_name')} v{land_cover.get('model_version')}",
            f"Model backend: {land_cover.get('model_backend')}",
        ]
    if envelope and envelope.available:
        items += [
            f"Interpretation provider: {envelope.provider}",
            f"Interpretation model: {envelope.model}",
            f"Interpretation generated at: {envelope.generated_at}",
        ]
    for key, value in quality.items():
        if isinstance(value, (int, float)):
            items.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return items


# --- formatting -------------------------------------------------------------
def _fmt(value: Any, suffix: str = "", *, signed: bool = False) -> str:
    if value is None:
        return "not available"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        formatted = f"{value:+.3f}" if signed else f"{value:.3f}"
        return f"{formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted}{suffix}"
    return f"{value}{suffix}"


def _fmt_bbox(bbox: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "not available"
    return ", ".join(f"{v:.5f}" for v in bbox)
