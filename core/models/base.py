"""
Base SQLAlchemy models and Pydantic schemas for AIEval360.
All domain entities inherit from these base classes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all AIEval360 models."""

    type_annotation_map: dict[Any, Any] = {}


class TimestampMixin:
    """Adds created_at and updated_at columns to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )


class UUIDPrimaryKey:
    """Provides a UUID primary key column."""

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_uuid,
        nullable=False,
    )
