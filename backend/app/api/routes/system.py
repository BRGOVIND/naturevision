"""Health and capability endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.imagery.bands import BAND_SPECS
from app.interpretation.language_service import LanguageInterpretationService
from app.models_ml.features import FEATURE_NAMES, FEATURE_VERSION
from app.models_ml.labels import CLASS_INFO, CLASS_ORDER
from app.models_ml.registry import available_models
from app.reports.export import pdf_available
from app.schemas.analysis import HealthResponse, ModelInfoSchema

logger = get_logger(__name__)
router = APIRouter(tags=["system"])

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, summary="Service health and capabilities")
async def health() -> HealthResponse:
    """Report liveness plus which optional capabilities are actually available.

    Capabilities are probed rather than assumed, so the client can disable
    controls for features this deployment genuinely cannot perform instead of
    letting the user trigger a guaranteed failure.
    """
    checks: dict[str, Any] = {}

    database_state = "unavailable"
    try:
        engine = get_engine()
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        database_state = engine.dialect.name
        checks["database"] = {"ok": True, "dialect": engine.dialect.name}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": type(exc).__name__}
        logger.warning("health_database_unreachable", error=str(exc)[:200])

    models = available_models()
    model_state = models[0].get("backend", "installed") if models else "not_installed"
    checks["land_cover_model"] = {
        "ok": bool(models),
        "installed": [m.get("directory") for m in models],
        "active_backend": settings.land_cover_backend,
    }

    interpretation_state = "configured" if settings.language_enabled else "not_configured"
    # Vision is probed against the provider's model list rather than inferred
    # from configuration, so the client never offers a capability that would
    # fail on use.
    vision_enabled = False
    if settings.language_enabled:
        service = LanguageInterpretationService()
        try:
            vision_enabled = await service.vision_supported()
        finally:
            await service.close()
    checks["interpretation"] = {
        "ok": settings.language_enabled,
        "provider": "Groq" if settings.language_enabled else None,
        "model": settings.language_model if settings.language_enabled else None,
        "vision_enabled": vision_enabled,
        "vision_model": settings.vision_model if settings.language_enabled else None,
    }
    checks["report_export"] = {"html": True, "pdf": pdf_available()}
    checks["imagery"] = {
        "provider": settings.stac_endpoint,
        "collection": settings.stac_collection,
    }

    healthy = checks["database"]["ok"]
    return HealthResponse(
        status="ok" if healthy else "degraded",
        application=settings.app_name,
        environment=settings.app_env,
        version=APP_VERSION,
        database=database_state,
        imagery_provider=settings.stac_collection,
        land_cover_model=model_state,
        interpretation=interpretation_state,
        checks=checks,
    )


@router.get("/models", response_model=ModelInfoSchema, summary="Installed land-cover models")
async def models() -> ModelInfoSchema:
    """Expose model provenance and held-out metrics for inspection."""
    installed = available_models()
    return ModelInfoSchema(
        installed=bool(installed),
        models=installed,
        active_backend=settings.land_cover_backend,
        feature_version=FEATURE_VERSION,
        classes=[
            {
                "id": int(class_id),
                "label": CLASS_INFO[class_id].label,
                "description": CLASS_INFO[class_id].description,
                "colour": CLASS_INFO[class_id].colour,
            }
            for class_id in CLASS_ORDER
        ],
    )


@router.get("/methodology", summary="Processing methodology and data sources")
async def methodology() -> dict[str, Any]:
    """Static description of how this deployment computes what it reports."""
    return {
        "imagery": {
            "provider": settings.stac_endpoint,
            "collection": settings.stac_collection,
            "processing_level": "Level-2A surface reflectance",
            "access": "Windowed reads over cloud-optimised GeoTIFFs.",
            "license": "Copernicus Sentinel data, free and open (CC BY-like terms).",
        },
        "bands": [
            {
                "band": spec.band.value,
                "sentinel2_id": spec.sentinel2_id,
                "centre_wavelength_nm": spec.centre_wavelength_nm,
                "native_resolution_m": spec.native_resolution_m,
                "description": spec.description,
            }
            for spec in BAND_SPECS.values()
        ],
        "indices": {
            "ndvi": "(NIR - Red) / (NIR + Red), Sentinel-2 B08 and B04",
        },
        "quality_masking": (
            "Level-2A scene classification removes no-data, saturated, cloud "
            "shadow, medium/high probability cloud, thin cirrus and snow/ice."
        ),
        "change_detection": {
            "operation": "Per-pixel NDVI difference, period B minus period A.",
            "alignment": "Period B is reprojected onto the period A grid before differencing.",
            "default_thresholds": {
                "moderate": settings.change_moderate_threshold,
                "significant": settings.change_significant_threshold,
            },
            "units": "absolute index units",
        },
        "land_cover": {
            "features": list(FEATURE_NAMES),
            "feature_version": FEATURE_VERSION,
            "classes": [CLASS_INFO[c].label for c in CLASS_ORDER],
            "installed_models": available_models(),
        },
        "scientific_scope": [
            "Reported change describes a reflectance-derived vegetation index.",
            "No causal attribution is produced from index change alone.",
            "Land-cover values are model predictions, not measurements.",
        ],
    }
