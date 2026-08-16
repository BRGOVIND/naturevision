"""Evidence assembly, grounding validation and the language provider contract."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.errors import InterpretationProviderError
from app.interpretation.evidence import EvidencePackage, build_evidence
from app.interpretation.language_service import LanguageInterpretationService, _extract_json
from app.interpretation.schemas import Interpretation
from app.interpretation.validation import validate_interpretation

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
