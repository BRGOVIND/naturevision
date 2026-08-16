"""API contract, validation and analysis lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db_session, get_imagery_provider
from app.core.errors import ImagerySearchError
from app.imagery.bands import Band
from app.main import create_app
from app.models import Base
from tests.conftest import StubProvider

REGION = {"bbox": [76.60, 10.20, 76.62, 10.22], "name": "Test region"}
PERIOD_A = {"start": "2021-01-01", "end": "2021-02-28"}
PERIOD_B = {"start": "2024-01-01", "end": "2024-02-29"}


@pytest.fixture
async def api_client(tmp_path, monkeypatch):
    """A client wired to an isolated database and an in-memory imagery provider."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "artifact_dir", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "target_raster_max_dim", 64)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    # The background analysis task opens its own session, so it must use the
    # same isolated engine as the request path.
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: factory)
    monkeypatch.setattr("app.services.analysis_service.get_session_factory", lambda: factory)

    provider = StubProvider()
    # The background task resolves its provider through this factory, since it
    # runs outside the request scope where dependency overrides apply.
    monkeypatch.setattr("app.services.analysis_service.provider_factory", lambda: provider)

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session
            await session.commit()

    app = create_app()
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_imagery_provider] = lambda: provider

    with TestClient(app) as client:
        client.provider = provider  # type: ignore[attr-defined]
        yield client

    await engine.dispose()


# --- health and capabilities -------------------------------------------------
def test_health_reports_status_and_capabilities(api_client):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["application"] == "NatureVision"
    assert "database" in body["checks"]
    assert "land_cover_model" in body["checks"]
    assert "interpretation" in body["checks"]


def test_unprefixed_health_probe_works(api_client):
    assert api_client.get("/health").status_code == 200


def test_methodology_documents_the_pipeline(api_client):
    body = api_client.get("/api/v1/methodology").json()
    assert body["indices"]["ndvi"].startswith("(NIR - Red)")
    assert body["change_detection"]["default_thresholds"]["moderate"] > 0
    assert any("causal" in note for note in body["scientific_scope"])


def test_models_endpoint_lists_classes(api_client):
    body = api_client.get("/api/v1/models").json()
    assert len(body["classes"]) == 5
    assert {c["label"] for c in body["classes"]} >= {"Forest", "Water"}


def test_openapi_document_is_generated(api_client):
    paths = api_client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/analysis", "/api/v1/ndvi", "/api/v1/ai/report"):
        assert path in paths


# --- validation ----------------------------------------------------------------
def test_imagery_search_returns_observations(api_client):
    response = api_client.post(
        "/api/v1/imagery/search",
        json={
            "region": REGION,
            "start_date": "2021-01-01",
            "end_date": "2021-02-28",
            "max_cloud_cover": 20,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["observations"][0]["source_id"] == "TEST_SCENE_1"
    assert body["region"]["area_km2"] > 0


@pytest.mark.parametrize(
    "region",
    [
        {"bbox": [76.7, 10.2, 76.6, 10.3]},  # reversed longitude
        {"bbox": [200.0, 10.2, 201.0, 10.3]},  # out of range
        {"bbox": [0.0, 0.0, 10.0, 10.0]},  # exceeds the area limit
    ],
)
def test_invalid_regions_are_rejected_with_a_structured_error(api_client, region):
    response = api_client.post(
        "/api/v1/analysis",
        json={"region": region, "period_a": PERIOD_A, "max_cloud_cover": 20},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] in {"invalid_geometry", "validation_error"}
    assert body["message"]
    assert "Traceback" not in body["message"]


def test_missing_region_selector_is_rejected(api_client):
    response = api_client.post(
        "/api/v1/analysis", json={"region": {"name": "nothing"}, "period_a": PERIOD_A}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_reversed_period_is_rejected(api_client):
    response = api_client.post(
        "/api/v1/analysis",
        json={"region": REGION, "period_a": {"start": "2024-01-01", "end": "2021-01-01"}},
    )
    assert response.status_code == 422


def test_malformed_json_body_is_rejected(api_client):
    response = api_client.post(
        "/api/v1/analysis",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_threshold_ordering_is_validated(api_client):
    response = api_client.post(
        "/api/v1/analysis",
        json={
            "region": REGION,
            "period_a": PERIOD_A,
            "change_moderate_threshold": 0.4,
            "change_significant_threshold": 0.1,
        },
    )
    assert response.status_code == 422


def test_unknown_analysis_returns_404(api_client):
    response = api_client.get("/api/v1/analysis/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_unknown_report_returns_404(api_client):
    response = api_client.get("/api/v1/reports/does-not-exist")
    assert response.status_code == 404


# --- processing endpoints -----------------------------------------------------------
def test_ndvi_endpoint_returns_real_statistics(api_client):
    response = api_client.post(
        "/api/v1/ndvi", json={"region": REGION, "period": PERIOD_A, "max_cloud_cover": 20}
    )
    assert response.status_code == 200
    body = response.json()
    expected = (0.35 - 0.05) / (0.35 + 0.05)
    assert body["statistics"]["mean"] == pytest.approx(expected, abs=1e-4)
    assert body["statistics"]["valid_pixel_count"] > 0
    assert body["methodology"]["formula"] == "(NIR - Red) / (NIR + Red)"
    assert body["observation"]["source_id"] == "TEST_SCENE_1"


def test_change_detection_endpoint(api_client):
    response = api_client.post(
        "/api/v1/change-detection",
        json={"region": REGION, "period_a": PERIOD_A, "period_b": PERIOD_B},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # The stub returns identical reflectance for both dates, so change is zero.
    assert result["absolute_change"] == pytest.approx(0.0, abs=1e-6)
    assert result["comparable_pixel_count"] > 0
    assert result["thresholds"]["moderate"] > 0


def test_land_cover_without_a_model_reports_unavailable(api_client):
    response = api_client.post("/api/v1/land-cover", json={"region": REGION, "period": PERIOD_A})
    # Either a trained model is installed and this succeeds, or it reports the
    # absence explicitly. It must never invent a distribution.
    assert response.status_code in {200, 503}
    if response.status_code == 503:
        assert response.json()["code"] == "model_unavailable"


def test_imagery_provider_failure_surfaces_as_502(api_client, monkeypatch):
    async def failing_search(_request):
        raise ImagerySearchError("The satellite catalogue is unreachable.")

    monkeypatch.setattr(api_client.provider, "search", failing_search)
    response = api_client.post("/api/v1/ndvi", json={"region": REGION, "period": PERIOD_A})
    assert response.status_code == 502
    assert response.json()["code"] == "imagery_search_failed"


def test_band_read_failure_surfaces_cleanly(api_client, monkeypatch):
    monkeypatch.setattr(api_client.provider, "fail_on", Band.NIR)
    response = api_client.post("/api/v1/ndvi", json={"region": REGION, "period": PERIOD_A})
    assert response.status_code == 502
    assert response.json()["code"] == "imagery_acquisition_failed"


# --- lifecycle -----------------------------------------------------------------------
def _await_completion(client: TestClient, analysis_id: str, attempts: int = 100) -> dict:
    for _ in range(attempts):
        body = client.get(f"/api/v1/analysis/{analysis_id}/status").json()
        if body["status"] in {"report_ready", "failed"}:
            return body
        asyncio.run(asyncio.sleep(0.05))
    raise AssertionError(f"Analysis did not settle; last status {body}")


def test_analysis_runs_end_to_end_and_persists_results(api_client):
    created = api_client.post(
        "/api/v1/analysis",
        json={
            "region": REGION,
            "period_a": PERIOD_A,
            "period_b": PERIOD_B,
            "max_cloud_cover": 20,
            "include_land_cover": False,
            "include_interpretation": False,
        },
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    assert created.json()["status"] == "created"

    final = _await_completion(api_client, analysis_id)
    assert final["status"] == "report_ready", final
    assert final["progress"] == 1.0

    detail = api_client.get(f"/api/v1/analysis/{analysis_id}").json()
    assert len(detail["observations"]) == 2
    assert {o["period"] for o in detail["observations"]} == {"A", "B"}

    metrics = {(m["key"], m["period"]): m["value"] for m in detail["metrics"]}
    expected_ndvi = (0.35 - 0.05) / (0.35 + 0.05)
    assert metrics[("mean_ndvi", "A")] == pytest.approx(expected_ndvi, abs=1e-4)
    assert ("ndvi_change", None) in metrics

    assert detail["evidence"]["methodology"]["index_formula"].startswith("NDVI =")
    assert len(detail["evidence"]["limitations"]) >= 5
    assert {layer["key"] for layer in detail["layers"]} >= {"ndvi_a", "ndvi_b", "change"}
    for layer in detail["layers"]:
        assert len(layer["bounds"]) == 4


def test_rendered_layers_are_served_as_png(api_client):
    created = api_client.post(
        "/api/v1/analysis",
        json={
            "region": REGION,
            "period_a": PERIOD_A,
            "include_land_cover": False,
            "include_interpretation": False,
        },
    )
    analysis_id = created.json()["id"]
    _await_completion(api_client, analysis_id)

    response = api_client.get(f"/api/v1/analysis/{analysis_id}/layers/ndvi_a")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    missing = api_client.get(f"/api/v1/analysis/{analysis_id}/layers/not_a_layer")
    assert missing.status_code == 404


def test_analysis_failure_is_recorded_as_a_terminal_state(api_client, monkeypatch):
    async def failing_search(_request):
        raise ImagerySearchError("The satellite catalogue is unreachable.")

    monkeypatch.setattr(api_client.provider, "search", failing_search)
    created = api_client.post("/api/v1/analysis", json={"region": REGION, "period_a": PERIOD_A})
    assert created.status_code == 202

    final = _await_completion(api_client, created.json()["id"])
    assert final["status"] == "failed"
    assert final["error_code"] == "imagery_search_failed"
    assert "Traceback" not in (final["error_message"] or "")


def test_report_before_completion_is_rejected(api_client, monkeypatch):
    async def slow_search(_request):
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(api_client.provider, "search", slow_search)
    created = api_client.post("/api/v1/analysis", json={"region": REGION, "period_a": PERIOD_A})
    response = api_client.post("/api/v1/ai/report", json={"analysis_id": created.json()["id"]})
    assert response.status_code == 409
    assert response.json()["code"] == "invalid_analysis_state"


def test_report_generation_and_export(api_client):
    created = api_client.post(
        "/api/v1/analysis",
        json={
            "region": REGION,
            "period_a": PERIOD_A,
            "period_b": PERIOD_B,
            "include_land_cover": False,
            "include_interpretation": False,
        },
    )
    analysis_id = created.json()["id"]
    _await_completion(api_client, analysis_id)

    response = api_client.post("/api/v1/ai/report", json={"analysis_id": analysis_id})
    assert response.status_code == 201
    report = response.json()

    titles = [section["title"] for section in report["sections"]]
    for required in (
        "Executive Summary",
        "Region",
        "Observation Period",
        "Data Sources",
        "Methodology",
        "NDVI Statistics",
        "Vegetation Change",
        "Interpretation",
        "Confidence",
        "Limitations",
        "Technical Metadata",
    ):
        assert required in titles

    provenances = {section["provenance"] for section in report["sections"]}
    assert {"observed", "interpretation", "metadata"} <= provenances

    export = api_client.get(f"/api/v1/reports/{report['id']}/export?format=html")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/html")
    assert "Nature Intelligence Report" in export.text

    fetched = api_client.get(f"/api/v1/reports/{report['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == report["id"]


def test_history_lists_and_deletes_analyses(api_client):
    created = api_client.post(
        "/api/v1/analysis",
        json={
            "region": REGION,
            "period_a": PERIOD_A,
            "include_land_cover": False,
            "include_interpretation": False,
        },
    )
    analysis_id = created.json()["id"]
    _await_completion(api_client, analysis_id)

    listing = api_client.get("/api/v1/analysis?limit=10").json()
    assert listing["total"] >= 1
    entry = next(item for item in listing["items"] if item["id"] == analysis_id)
    assert entry["region_name"] == "Test region"
    assert entry["mean_ndvi_a"] is not None
    assert entry["area_km2"] > 0

    assert api_client.delete(f"/api/v1/analysis/{analysis_id}").status_code == 204
    assert api_client.get(f"/api/v1/analysis/{analysis_id}").status_code == 404
