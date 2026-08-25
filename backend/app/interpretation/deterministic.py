"""Deterministic evidence summary — the fallback when no language-model
response is available.

This performs no reasoning and calls no provider. It restates measured
values using fixed sentence templates, in the values the evidence package
itself carries, and states nothing the evidence does not contain. It is
run through the same grounding validator as a language-model response, as
a structural guard against a future template bug — not because a
template-generated summary is expected to need it. It is not a substitute
for language-model interpretation, and the client must label it as a
measured summary rather than presenting it as generated text.
"""

from __future__ import annotations

from typing import Any

from app.interpretation.evidence import EvidencePackage
from app.interpretation.schemas import Interpretation
from app.interpretation.validation import GroundingReport, validate_interpretation

_FALLBACK_NOTE = (
    "This summary restates measured values only. It was produced "
    "deterministically, not by a language model, because no language-model "
    "response was available or none passed evidence-grounding validation."
)


def build_deterministic_summary(
    evidence: EvidencePackage,
) -> tuple[Interpretation, GroundingReport] | None:
    """A template restatement of observed values, or None if there is not
    enough measured evidence to describe safely."""
    period_a = evidence.observed.get("period_a")
    if not period_a or period_a.get("mean_ndvi") is None:
        return None

    mean_a = period_a["mean_ndvi"]
    observations: list[dict[str, Any]] = [
        {
            "statement": f"Mean NDVI in the first observation period was {mean_a:.4f}.",
            "evidence_key": "observed.period_a.mean_ndvi",
        }
    ]
    summary = (
        f"Mean NDVI over the analysed region was {mean_a:.4f} in the first observation period."
    )

    period_b = evidence.observed.get("period_b")
    change = evidence.observed.get("change")

    if period_b and period_b.get("mean_ndvi") is not None:
        mean_b = period_b["mean_ndvi"]
        observations.append(
            {
                "statement": f"Mean NDVI in the second observation period was {mean_b:.4f}.",
                "evidence_key": "observed.period_b.mean_ndvi",
            }
        )
        summary += f" In the second period it was {mean_b:.4f}."
    else:
        summary += " No second observation period was compared."

    if change and change.get("absolute_ndvi_change") is not None:
        delta = float(change["absolute_ndvi_change"])
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
        observations.append(
            {
                "statement": (
                    f"The measured NDVI {direction} by {abs(delta):.4f} index units "
                    "between the two periods."
                ),
                "evidence_key": "observed.change.absolute_ndvi_change",
            }
        )
        interpretation = (
            f"The measured vegetation index {direction} between the two observation "
            "periods. This describes a change in a reflectance-derived index only; it "
            "does not by itself identify a cause."
        )
        if change.get("changed_area_percentage") is not None:
            pct = float(change["changed_area_percentage"])
            observations.append(
                {
                    "statement": f"{pct:.2f}% of the comparable area registered as changed.",
                    "evidence_key": "observed.change.changed_area_percentage",
                }
            )
            interpretation += (
                f" {pct:.2f}% of the comparable analysed area crossed the configured "
                "change threshold."
            )
    else:
        interpretation = (
            "Only one observation period is available for this analysis, so no "
            "temporal change can be described."
        )

    limitations = [_FALLBACK_NOTE]

    land_cover = evidence.model_predictions.get("land_cover")
    distribution = land_cover.get("distribution_percent") if land_cover else None
    if distribution:
        top_class, top_share = max(distribution.items(), key=lambda kv: kv[1])
        observations.append(
            {
                "statement": (
                    f"The land-cover model predicted {top_class} as the largest class, "
                    f"at {top_share:.1f}% of the classified area."
                ),
                "evidence_key": f"model_predictions.land_cover.distribution_percent.{top_class}",
            }
        )
        limitations.append(
            "Land-cover figures are a statistical model's predictions, not a direct "
            "measurement, and carry classification error."
        )

    limitations.extend(evidence.limitations[:2])

    candidate = Interpretation.model_validate(
        {
            "summary": summary,
            "observations": observations,
            "interpretation": interpretation,
            "uncertainty": (
                "This is a direct restatement of measured values, not a language-model "
                "interpretation, so it carries no interpretive uncertainty beyond the "
                "measurement limitations listed below."
            ),
            "limitations": limitations,
            "confidence_qualifier": "moderate",
        }
    )

    grounding = validate_interpretation(candidate, evidence.numeric_claims())
    if not grounding.passed:
        # A template defect, not a provider failure — never surface a summary
        # that fails the same check applied to a real language-model response.
        return None
    return candidate, grounding
