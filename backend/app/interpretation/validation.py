"""Grounding checks applied to generated interpretation text.

The language layer is prompted to use only supplied evidence, but a prompt is
not an enforcement mechanism. Every generated response is therefore checked
against the evidence package before it reaches the client:

* numeric claims must correspond to a value that was actually measured;
* causal and ecological conclusions the methodology cannot support are flagged.

Findings are attached to the response as a grounding report. Numeric
fabrication is treated as fatal, because a wrong number is indistinguishable
from a measured one once it is in a report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.interpretation.schemas import Interpretation

logger = get_logger(__name__)

#: Numbers in text are matched with their sign and optional percent sign.
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

#: ISO dates and date ranges are removed before numbers are extracted. Without
#: this, "2021-02-28" is read as the three values 2021, -02 and -28, and a
#: correctly grounded response is rejected for citing its own observation date.
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

#: Ordinal and unit-bearing date fragments that are prose rather than measurements.
_DATE_WORD_PATTERN = re.compile(
    r"\b\d{1,4}\s*(?:day|days|month|months|year|years)\b", re.IGNORECASE
)

#: Relative tolerance when matching a stated number against measured evidence.
NUMERIC_TOLERANCE = 0.02

#: Small integers and round percentages appear as ordinary prose ("two periods",
#: "100%"), so they are not treated as claims needing an evidence match.
_IGNORED_NUMBERS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}

#: Assertions a two-date optical index comparison cannot establish.
UNSUPPORTED_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdeforestation\b", "attributes change to deforestation"),
    (r"\bdeforest(ed|ing)\b", "attributes change to deforestation"),
    (r"\billegal\b", "asserts legality of an activity"),
    (r"\bbiodiversity (loss|decline|declined)\b", "claims a biodiversity outcome"),
    (r"\bspecies (loss|decline|extinction)\b", "claims a species-level outcome"),
    (r"\bclimate change (caused|is causing|has caused|drove)\b", "asserts climate causation"),
    (r"\bcaused by\b", "asserts a specific cause"),
    (r"\bdue to (logging|mining|agriculture|urbani)", "asserts a specific human cause"),
    (r"\bproves\b", "overstates evidential strength"),
    (r"\bconfirms that\b", "overstates evidential strength"),
    (r"\bcertainly\b", "overstates certainty"),
    (r"\bdefinitely\b", "overstates certainty"),
    (
        r"\bcarbon (stock|sequestration|storage) (of|is|was)\b",
        "claims an unmeasured carbon quantity",
    ),
)


@dataclass(slots=True)
class GroundingReport:
    """Outcome of validating one interpretation against its evidence."""

    passed: bool = True
    unsupported_numbers: list[float] = field(default_factory=list)
    flagged_claims: list[dict[str, str]] = field(default_factory=list)
    checked_number_count: int = 0
    matched_number_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "unsupported_numbers": self.unsupported_numbers,
            "flagged_claims": self.flagged_claims,
            "checked_number_count": self.checked_number_count,
            "matched_number_count": self.matched_number_count,
        }


def _interpretation_text(interpretation: Interpretation) -> str:
    parts = [
        interpretation.summary,
        interpretation.interpretation,
        interpretation.uncertainty,
        *(o.statement for o in interpretation.observations),
        *interpretation.limitations,
    ]
    return "\n".join(parts)


def _matches_evidence(value: float, allowed: list[float]) -> bool:
    for candidate in allowed:
        if candidate == 0.0:
            if abs(value) < 1e-9:
                return True
            continue
        if abs(value - candidate) <= max(NUMERIC_TOLERANCE * abs(candidate), 0.005):
            return True
    return False


def validate_interpretation(
    interpretation: Interpretation, evidence_numbers: dict[str, float]
) -> GroundingReport:
    """Check generated text against the measured evidence.

    Percentages are cross-checked against both their raw value and their
    fractional form, because the evidence stores some quantities as fractions
    (0.68) and the text may legitimately state them as percentages (68%).
    """
    report = GroundingReport()
    # Dates are stripped before number extraction; they are provenance the model
    # is expected to restate, not measurements to verify.
    text = _DATE_WORD_PATTERN.sub(" ", _DATE_PATTERN.sub(" ", _interpretation_text(interpretation)))

    allowed = list(evidence_numbers.values())
    allowed += [v * 100.0 for v in evidence_numbers.values() if abs(v) <= 1.0]
    allowed += [abs(v) for v in evidence_numbers.values()]

    for raw in _NUMBER_PATTERN.findall(text):
        try:
            value = float(raw)
        except ValueError:
            continue
        if value in _IGNORED_NUMBERS or abs(value) > 100_000:
            continue
        # Four-digit values in a plausible range are years, not measurements.
        if value.is_integer() and 1900 <= value <= 2100:
            continue
        report.checked_number_count += 1
        if _matches_evidence(value, allowed):
            report.matched_number_count += 1
        else:
            report.unsupported_numbers.append(value)

    lowered = text.lower()
    for pattern, description in UNSUPPORTED_CLAIM_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            report.flagged_claims.append({"phrase": match.group(0), "issue": description})

    # A fabricated number is fatal; an over-reaching phrase is reported so the
    # caller can decide, since some are legitimate inside a limitations section.
    report.passed = not report.unsupported_numbers

    if not report.passed:
        logger.warning(
            "interpretation_grounding_failed",
            unsupported=report.unsupported_numbers[:10],
            flagged=[c["issue"] for c in report.flagged_claims],
        )
    elif report.flagged_claims:
        logger.info(
            "interpretation_claims_flagged", flagged=[c["issue"] for c in report.flagged_claims]
        )
    return report
