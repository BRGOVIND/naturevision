"""Structured schema for generated environmental interpretation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    """A single restatement of measured evidence."""

    statement: str = Field(min_length=10, max_length=600)
    evidence_key: str | None = Field(
        default=None, description="Path into the evidence package supporting this statement."
    )


class Interpretation(BaseModel):
    """The validated interpretation contract returned to the client.

    The separation of ``observations`` from ``interpretation`` is load-bearing:
    observations restate measured values, interpretation reasons about them, and
    the report renders them under distinct headings so a reader always knows
    which is which.
    """

    summary: str = Field(min_length=40, max_length=1500)
    observations: list[Observation] = Field(min_length=1, max_length=12)
    interpretation: str = Field(min_length=40, max_length=2500)
    uncertainty: str = Field(min_length=20, max_length=1500)
    limitations: list[str] = Field(min_length=1, max_length=12)
    confidence_qualifier: str = Field(
        default="moderate",
        description="How strongly the evidence supports the interpretation.",
    )

    @field_validator("limitations")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("At least one limitation must be stated.")
        return cleaned

    @field_validator("confidence_qualifier")
    @classmethod
    def _known_qualifier(cls, value: str) -> str:
        allowed = {"low", "moderate", "high"}
        normalised = value.strip().lower()
        return normalised if normalised in allowed else "moderate"


class VisualObservation(BaseModel):
    """One thing a vision model reports seeing in a rendered image."""

    statement: str = Field(min_length=10, max_length=500)
    spatial_reference: str | None = Field(
        default=None, description="Where in the image, in plain language."
    )


class VisualInterpretation(BaseModel):
    """Vision-model output, kept strictly separate from measured statistics."""

    scene_description: str = Field(min_length=30, max_length=1500)
    observations: list[VisualObservation] = Field(min_length=1, max_length=10)
    spatial_patterns: str = Field(min_length=20, max_length=1500)
    caveats: str = Field(min_length=20, max_length=1000)


class InterpretationEnvelope(BaseModel):
    """Interpretation plus the provenance of how it was produced."""

    interpretation: Interpretation | None = None
    visual: VisualInterpretation | None = None
    provider: str | None = None
    model: str | None = None
    generated_at: str | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)
    available: bool = True
    unavailable_reason: str | None = None
    #: "language_model" for a Groq-generated, grounding-validated response;
    #: "measured" for the deterministic evidence summary produced when no
    #: provider response is available. The client must never present the
    #: latter as if a language model wrote it.
    source: str | None = None
