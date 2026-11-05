"""
Database session management and connection setup.
Provides async SQLAlchemy session factory and sync session for migrations.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from aieval360.core.config.settings import AppSettings, get_settings

# Module-level engine and session factory — initialised on first call to init_db()
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(settings: AppSettings) -> AsyncEngine:
    """
    Build an async SQLAlchemy engine from application settings.
    SQLite and PostgreSQL need slightly different pool configurations.
    """
    db_url = settings.database.url
    connect_args: dict = {}
    kwargs: dict = {}

    if "sqlite" in db_url:
        # SQLite: check_same_thread=False required for async
        connect_args["check_same_thread"] = False
        if ":memory:" in db_url:
            # In-memory SQLite: use StaticPool so all connections share the same DB
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["connect_args"] = connect_args
    else:
        # PostgreSQL: standard async pool configuration
        kwargs["pool_size"] = settings.database.pool_size
        kwargs["max_overflow"] = settings.database.max_overflow
        kwargs["pool_timeout"] = settings.database.pool_timeout

    return create_async_engine(
        db_url,
        echo=settings.database.echo,
        **kwargs,
    )


def init_db(settings: AppSettings | None = None) -> AsyncEngine:
    """
    Initialise the database engine and session factory.
    Call this once at application startup.
    """
    global _engine, _async_session_factory

    if settings is None:
        settings = get_settings()

    _engine = _build_engine(settings)
    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    return _engine


def get_engine() -> AsyncEngine:
    """Return the current database engine, initialising if needed."""
    global _engine
    if _engine is None:
        init_db()
    return _engine  # type: ignore[return-value]


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the current session factory, initialising if needed."""
    global _async_session_factory
    if _async_session_factory is None:
        init_db()
    return _async_session_factory  # type: ignore[return-value]


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that provides a database session.
    Commits on success, rolls back on exception.

    Usage:
        async with get_session() as session:
            result = await session.execute(...)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.
    Use as: session: AsyncSession = Depends(get_session_dependency)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """
    Create all database tables. Used for development and testing.
    In production, use Alembic migrations instead.
    """
    from aieval360.core.models.base import Base

    # Import all model modules to register them with SQLAlchemy metadata
    import aieval360.core.models.organisation  # noqa: F401
    import aieval360.core.models.evaluation  # noqa: F401
    import aieval360.core.models.governance  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """
    Drop all database tables. For testing only.
    """
    from aieval360.core.models.base import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db() -> None:
    """
    Dispose the database engine. Call at application shutdown.
    """
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
