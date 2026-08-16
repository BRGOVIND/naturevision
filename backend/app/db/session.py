"""Async SQLAlchemy engine, session factory and schema bootstrap."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        kwargs: dict = {"echo": settings.db_echo, "future": True}
        if settings.is_postgres:
            kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=1800)
        _engine = create_async_engine(settings.database_url, **kwargs)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), class_=AsyncSession, expire_on_commit=False, autoflush=False
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a transactional session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database() -> None:
    """Create the PostGIS extension and any missing tables.

    Alembic owns migrations for real deployments; this path exists so a fresh
    container or a test database becomes usable without a migration step.
    """
    from sqlalchemy import text

    from app.models import Base

    engine = get_engine()
    async with engine.begin() as connection:
        if settings.is_postgres:
            try:
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            except Exception as exc:  # insufficient privileges on a managed instance
                logger.warning("postgis_extension_unavailable", error=str(exc))
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database_ready", dialect=engine.dialect.name)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
