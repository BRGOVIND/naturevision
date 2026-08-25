"""Evidence assembly, grounding validation and the language provider contract."""

from __future__ import annotations

import json
import re

import httpx
import pytest
from pydantic import ValidationError

from app.core.errors import InterpretationProviderError
from app.interpretation.deterministic import build_deterministic_summary
from app.interpretation.evidence import EvidencePackage, build_evidence
from app.interpretation.language_service import LanguageInterpretationService, _extract_json
from app.interpretation.schemas import Interpretation
from app.interpretation.validation import UNSUPPORTED_CLAIM_PATTERNS, validate_interpretation

VALID_RESPONSE = {
    "summary": (
        "Mean NDVI over the selected region was 0.702 in period A and 0.722 in "
        "period B, a change of 0.019 index units."
    ),
    "observations": [
        {
            "statement": "Mean NDVI in period A was 0.702.",
            "evidence_key": "observed.period_a.mean_ndvi",
        },
        {
            "statement": "Mean NDVI in period B was 0.722.",
            "evidence_key": "observed.period_b.mean_ndvi",
        },
    ],
    "interpretation": (
        "The vegetation index is slightly higher in the later observation. The "
        "difference is small and may reflect seasonal condition rather than a "
        "persistent change in cover."
    ),
    "uncertainty": (
        "The change is below the stated moderate-change threshold, so it is not "
        "clearly separable from processing and atmospheric noise."
    ),
    "limitations": ["Optical imagery cannot see through cloud."],
    "confidence_qualifier": "low",
}


def _evidence() -> EvidencePackage:
    return EvidencePackage(
        region={"name": "Test region", "area_km2": 173.8},
        periods={"period_a": "2021-01-01 to 2021-02-28", "period_b": None},
        data_sources=[],
        observed={
            "period_a": {"mean_ndvi": 0.702, "valid_pixel_count": 143569},
            "period_b": {"mean_ndvi": 0.722},
            "change": {"absolute_ndvi_change": 0.019, "changed_area_percentage": 7.909},
        },
        model_predictions={"land_cover": {"distribution_percent": {"Forest": 88.101}}},
    )


# --- evidence ---------------------------------------------------------------
def test_numeric_claims_are_flattened_with_paths():
    claims = _evidence().numeric_claims()
    assert claims["observed.period_a.mean_ndvi"] == 0.702
    assert claims["model_predictions.land_cover.distribution_percent.Forest"] == 88.101
    assert all(isinstance(v, float) for v in claims.values())


async def test_build_evidence_separates_registers(scene):
    from app.analysis.indices import compute_ndvi
    from app.analysis.statistics import compute_statistics, summarise_vegetation

    ndvi = compute_ndvi(scene)
    package = build_evidence(
        region=scene.region,
        region_name="Test",
        scene_a=scene,
        ndvi_stats_a=compute_statistics(ndvi),
        vegetation_a=summarise_vegetation(ndvi),
        period_a_label="2021-01-01 to 2021-02-28",
    )
    assert "period_a" in package.observed
    assert package.model_predictions == {}  # no classifier was run
    assert package.methodology["index_formula"].startswith("NDVI =")
    assert len(package.limitations) >= 5
    assert any("cannot see through cloud" in item for item in package.limitations)


# --- grounding validation ------------------------------------------------------
def test_grounded_interpretation_passes():
    report = validate_interpretation(
        Interpretation.model_validate(VALID_RESPONSE), _evidence().numeric_claims()
    )
    assert report.passed
    assert report.unsupported_numbers == []
    assert report.matched_number_count > 0


def test_fabricated_number_fails_validation():
    payload = dict(VALID_RESPONSE)
    payload["interpretation"] = (
        "Mean NDVI fell to 0.317, indicating a substantial loss of canopy across "
        "the analysed region."
    )
    report = validate_interpretation(
        Interpretation.model_validate(payload), _evidence().numeric_claims()
    )
    assert not report.passed
    assert 0.317 in report.unsupported_numbers


def test_percentage_form_of_a_fraction_is_accepted():
    evidence = {"observed.vegetated_fraction": 0.68}
    payload = dict(VALID_RESPONSE)
    payload["interpretation"] = "Vegetated cover accounted for 68% of valid pixels in the region."
    payload["summary"] = "Vegetated cover accounted for 68% of valid pixels in the region overall."
    payload["observations"] = [{"statement": "Vegetated fraction was 0.68.", "evidence_key": None}]
    report = validate_interpretation(Interpretation.model_validate(payload), evidence)
    assert report.passed


def test_iso_dates_are_not_parsed_as_measurements():
    """Regression: "2021-02-28" must not be read as 2021, -2 and -28.

    A live provider restating its own observation dates was rejected as
    ungrounded until dates were stripped before number extraction.
    """
    payload = dict(VALID_RESPONSE)
    payload["summary"] = (
        "Observations from 2021-01-01 to 2021-02-28 and 2024-01-01 to "
        "2024-02-29 give a mean NDVI of 0.702 and 0.722 respectively."
    )
    report = validate_interpretation(
        Interpretation.model_validate(payload), _evidence().numeric_claims()
    )
    assert report.passed, report.unsupported_numbers
    assert -28.0 not in report.unsupported_numbers


def test_elapsed_day_counts_are_not_treated_as_measurements():
    payload = dict(VALID_RESPONSE)
    payload["uncertainty"] = (
        "The acquisitions are separated by 1080 days, which is long enough for "
        "seasonal differences to contribute to the comparison."
    )
    report = validate_interpretation(
        Interpretation.model_validate(payload), _evidence().numeric_claims()
    )
    assert report.passed, report.unsupported_numbers


def test_source_metadata_is_citable_evidence():
    """Scene cloud cover lives in data_sources and must be quotable."""
    package = EvidencePackage(
        region={},
        periods={},
        data_sources=[{"cloud_cover_percent": 0.165205, "resolution_m": 10.0}],
    )
    claims = package.numeric_claims()
    assert 0.165205 in claims.values()
    assert 10.0 in claims.values()


def test_methodology_figures_are_citable_evidence():
    """The prompt shows the model the full evidence package, methodology
    included, and tells it every number it writes must appear there — so a
    methodology figure must be in the allowed set or a faithful citation of
    it is wrongly rejected as fabricated."""
    package = EvidencePackage(
        region={},
        periods={},
        data_sources=[],
        methodology={"analysis_grid": {"resolution_m": 10.0, "valid_pixels": 143100}},
    )
    claims = package.numeric_claims()
    assert 10.0 in claims.values()
    assert 143100.0 in claims.values()


def test_years_are_not_treated_as_measurements():
    payload = dict(VALID_RESPONSE)
    payload["summary"] = (
        "Between 2021 and 2024 the mean vegetation index moved from 0.702 to 0.722."
    )
    report = validate_interpretation(
        Interpretation.model_validate(payload), _evidence().numeric_claims()
    )
    assert report.passed


@pytest.mark.parametrize(
    "text",
    [
        "The decline indicates deforestation across the northern block.",
        "This was caused by agricultural expansion in the region.",
        "The data proves that biodiversity loss has occurred here.",
    ],
)
def test_unsupported_causal_claims_are_flagged(text):
    payload = dict(VALID_RESPONSE)
    payload["interpretation"] = text + " Further evidence would be needed to confirm."
    report = validate_interpretation(
        Interpretation.model_validate(payload), _evidence().numeric_claims()
    )
    assert report.flagged_claims


# --- schema -----------------------------------------------------------------------
def test_schema_requires_observations_and_limitations():
    with pytest.raises(ValidationError):
        Interpretation.model_validate({**VALID_RESPONSE, "observations": []})
    with pytest.raises(ValidationError):
        Interpretation.model_validate({**VALID_RESPONSE, "limitations": []})


def test_unknown_confidence_qualifier_normalises():
    parsed = Interpretation.model_validate({**VALID_RESPONSE, "confidence_qualifier": "absolute"})
    assert parsed.confidence_qualifier == "moderate"


# --- response parsing ---------------------------------------------------------------
def test_extract_json_handles_plain_fenced_and_embedded():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('Here you go:\n{"a": 3}\nHope that helps.') == {"a": 3}


def test_extract_json_rejects_unusable_output():
    with pytest.raises(ValueError):
        _extract_json("")
    with pytest.raises(ValueError):
        _extract_json("no json at all")


# --- provider behaviour ----------------------------------------------------------------
def _service(handler, monkeypatch) -> LanguageInterpretationService:
    from app.core.config import settings

    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    client = httpx.AsyncClient(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )
    return LanguageInterpretationService(client=client)


def _completion(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})


async def test_unconfigured_provider_degrades_without_error(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "groq_api_key", None)
    envelope = await LanguageInterpretationService().interpret(_evidence())
    assert envelope.available is False
    assert envelope.interpretation is None
    assert "is configured" in envelope.unavailable_reason.lower()


async def test_available_models_flags_a_decommissioned_language_model(monkeypatch):
    """A configured API key proves nothing about whether the specific model
    is still servable — this previously let /health report interpretation
    as configured while every real request 404'd with "model does not
    exist" (llama-3.3-70b-versatile was removed from the provider's
    catalogue). The health check now cross-references the model list the
    same way vision capability already was."""
    from app.core.config import settings

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}]})

    service = _service(handler, monkeypatch)
    models = await service.available_models()
    assert models == {"openai/gpt-oss-120b"}

    monkeypatch.setattr(settings, "language_model", "openai/gpt-oss-120b")
    assert settings.language_model in models  # a currently-servable model...

    monkeypatch.setattr(settings, "language_model", "llama-3.3-70b-versatile")
    assert settings.language_model not in models  # ...but a stale model name is correctly flagged


async def test_valid_provider_response_is_returned(monkeypatch):
    service = _service(lambda _request: _completion(VALID_RESPONSE), monkeypatch)
    envelope = await service.interpret(_evidence())
    assert envelope.available
    assert envelope.interpretation is not None
    assert envelope.grounding["passed"] is True
    assert envelope.provider == "Groq"


async def test_malformed_response_is_retried_then_rejected(monkeypatch):
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    service = _service(handler, monkeypatch)
    with pytest.raises(InterpretationProviderError):
        await service.interpret(_evidence())
    assert calls["n"] == 2  # one retry with a correction turn


async def test_missing_fields_are_rejected(monkeypatch):
    service = _service(lambda _request: _completion({"summary": "too short"}), monkeypatch)
    with pytest.raises(InterpretationProviderError):
        await service.interpret(_evidence())


async def test_ungrounded_response_is_rejected_after_retry(monkeypatch):
    bad = dict(VALID_RESPONSE)
    bad["interpretation"] = "Mean NDVI collapsed to 0.099 across the whole region."
    service = _service(lambda _request: _completion(bad), monkeypatch)
    with pytest.raises(InterpretationProviderError):
        await service.interpret(_evidence())


async def test_provider_http_error_surfaces_as_domain_error(monkeypatch):
    service = _service(lambda _request: httpx.Response(500, text="upstream boom"), monkeypatch)
    with pytest.raises(InterpretationProviderError):
        await service.interpret(_evidence())


async def test_provider_transport_failure_surfaces_as_domain_error(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    service = _service(handler, monkeypatch)
    with pytest.raises(InterpretationProviderError):
        await service.interpret(_evidence())


async def test_vision_failure_returns_none_rather_than_raising(monkeypatch):
    service = _service(lambda _request: httpx.Response(500, text="nope"), monkeypatch)
    result = await service.interpret_image("data:image/png;base64,AAAA", "NDVI map", "context")
    assert result is None


# --- deterministic fallback --------------------------------------------------
def test_deterministic_summary_restates_only_measured_values():
    """The fallback used when no language-model response is available must
    itself pass the same grounding check applied to a real response."""
    summary, grounding = build_deterministic_summary(_evidence())
    assert grounding.passed
    assert grounding.unsupported_numbers == []
    assert "0.702" in summary.summary
    assert "0.722" in summary.summary
    # Every observation traces to a real evidence path, not an invented one.
    claims = _evidence().numeric_claims()
    for observation in summary.observations:
        assert observation.evidence_key in claims


def test_deterministic_summary_names_no_cause():
    """The fallback must respect the same causal-overreach screen a
    language-model response is checked against."""
    summary, _ = build_deterministic_summary(_evidence())
    text = (summary.summary + summary.interpretation).lower()
    for pattern, _issue in UNSUPPORTED_CLAIM_PATTERNS:
        assert not re.search(pattern, text), f"deterministic summary matched: {pattern}"


def test_deterministic_summary_is_none_without_measured_ndvi():
    """No mean_ndvi in the evidence means nothing safe to restate — the
    fallback must decline rather than invent a summary."""
    empty = EvidencePackage(region={}, periods={}, data_sources=[])
    assert build_deterministic_summary(empty) is None


def test_deterministic_summary_handles_a_single_observation_period():
    """A single-period analysis has no change to describe; the fallback must
    say so rather than fabricate a comparison."""
    single_period = EvidencePackage(
        region={},
        periods={"period_a": "2021-01-01 to 2021-02-28", "period_b": None},
        data_sources=[],
        observed={"period_a": {"mean_ndvi": 0.702}},
    )
    result = build_deterministic_summary(single_period)
    assert result is not None
    summary, grounding = result
    assert grounding.passed
    assert "no second observation period" in summary.summary.lower()
