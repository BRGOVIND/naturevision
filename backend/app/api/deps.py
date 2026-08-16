"""FastAPI dependency providers.

The imagery provider and language client hold pooled HTTP connections, so they
are created once per process and shared, not per request.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.imagery.base import ImageryProvider
from app.imagery.service import ImageryService
from app.imagery.stac import SentinelHubStacProvider
from app.interpretation.language_service import LanguageInterpretationService
from app.models_ml.classifier import LandCoverClassifier
from app.services.analysis_service import AnalysisService

_provider: ImageryProvider | None = None
_language: LanguageInterpretationService | None = None


def get_imagery_provider() -> ImageryProvider:
    global _provider
    if _provider is None:
        _provider = SentinelHubStacProvider()
    return _provider


def get_imagery_service(
    provider: Annotated[ImageryProvider, Depends(get_imagery_provider)],
) -> ImageryService:
    return ImageryService(provider)


def get_language_service() -> LanguageInterpretationService:
    global _language
    if _language is None:
        _language = LanguageInterpretationService()
    return _language


def get_classifier() -> LandCoverClassifier:
    return LandCoverClassifier()


async def get_analysis_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AsyncGenerator[AnalysisService, None]:
    yield AnalysisService(session)


async def shutdown_clients() -> None:
    """Close pooled clients on application shutdown."""
    global _provider, _language
    if _provider is not None:
        await _provider.close()
        _provider = None
    if _language is not None:
        await _language.close()
        _language = None


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
ImageryDep = Annotated[ImageryService, Depends(get_imagery_service)]
LanguageDep = Annotated[LanguageInterpretationService, Depends(get_language_service)]
ClassifierDep = Annotated[LandCoverClassifier, Depends(get_classifier)]
AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
