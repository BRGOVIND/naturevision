"""Domain exceptions and the HTTP error contract.

User-facing responses carry a stable machine code and a safe message.
Technical detail is logged server-side and never returned to the client.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var

#: Starlette renamed this constant; the numeric code is stable across versions.
HTTP_422_UNPROCESSABLE_CONTENT = 422

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every failing endpoint."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Safe, human-readable description.")
    details: dict[str, Any] | None = None
    request_id: str | None = None


class NatureVisionError(Exception):
    """Base class for expected, user-attributable failures."""

    code = "internal_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message
        self.details = details


class GeometryValidationError(NatureVisionError):
    code = "invalid_geometry"
    status_code = HTTP_422_UNPROCESSABLE_CONTENT
    message = "The supplied region geometry is not valid."


class ImagerySearchError(NatureVisionError):
    code = "imagery_search_failed"
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "The satellite catalogue could not be queried."


class NoImageryFoundError(NatureVisionError):
    code = "no_imagery_found"
    status_code = status.HTTP_404_NOT_FOUND
    message = "No suitable satellite observations were found for the region and period."


class ImageryAcquisitionError(NatureVisionError):
    code = "imagery_acquisition_failed"
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "Satellite imagery could not be retrieved from the provider."


class RasterProcessingError(NatureVisionError):
    code = "raster_processing_failed"
    status_code = HTTP_422_UNPROCESSABLE_CONTENT
    message = "The raster data could not be processed."


class InsufficientValidPixelsError(RasterProcessingError):
    code = "insufficient_valid_pixels"
    message = "Too few cloud-free pixels remain after masking to produce a reliable result."


class ModelUnavailableError(NatureVisionError):
    code = "model_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "The land-cover model artifact is not available on this deployment."


class ModelInferenceError(NatureVisionError):
    code = "model_inference_failed"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "Land-cover inference failed."


class InterpretationUnavailableError(NatureVisionError):
    code = "interpretation_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "The language interpretation provider is not configured."


class InterpretationProviderError(NatureVisionError):
    code = "interpretation_provider_failed"
    status_code = status.HTTP_502_BAD_GATEWAY
    message = "The language interpretation provider returned an unusable response."


class ResourceNotFoundError(NatureVisionError):
    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND
    message = "The requested resource does not exist."


class AnalysisStateError(NatureVisionError):
    code = "invalid_analysis_state"
    status_code = status.HTTP_409_CONFLICT
    message = "The analysis is not in a state that permits this operation."


class ReportGenerationError(NatureVisionError):
    code = "report_generation_failed"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "The report could not be generated."


def _envelope(exc_code: str, message: str, details: dict[str, Any] | None, http_status: int):
    return JSONResponse(
        status_code=http_status,
        content=ErrorResponse(
            code=exc_code,
            message=message,
            details=details,
            request_id=request_id_var.get(),
        ).model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NatureVisionError)
    async def _domain(_request: Request, exc: NatureVisionError):
        log = logger.warning if exc.status_code < 500 else logger.error
        log("domain_error", code=exc.code, message=exc.message, details=exc.details)
        return _envelope(exc.code, exc.message, exc.details, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError):
        fields = [
            {"field": ".".join(str(p) for p in err["loc"][1:]), "issue": err["msg"]}
            for err in exc.errors()
        ]
        logger.warning("request_validation_error", fields=fields)
        return _envelope(
            "validation_error",
            "The request payload failed validation.",
            {"fields": fields},
            HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException):
        return _envelope("http_error", str(exc.detail), None, exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        logger.exception("unhandled_error", error_type=type(exc).__name__)
        return _envelope(
            "internal_error",
            "An unexpected error occurred. The incident has been logged.",
            None,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
