"""
Organisation, BusinessUnit, Project and User domain models.
Supports multi-tenancy with tenant isolation at the data layer.
"""

from __future__ import annotations

import enum
from typing import Optional

from sqlalchemy import Boolean, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDPrimaryKey


class UserRole(str, enum.Enum):
    PLATFORM_ADMIN = "platform_admin"
    ORG_ADMIN = "org_admin"
    AI_GOVERNANCE_MANAGER = "ai_governance_manager"
    AI_EVALUATION_MANAGER = "ai_evaluation_manager"
    TECHNICAL_EVALUATOR = "technical_evaluator"
    BUSINESS_OWNER = "business_owner"
    AUDITOR = "auditor"
    READ_ONLY = "read_only"


class LifecycleStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    PENDING = "pending"


class Organisation(Base, UUIDPrimaryKey, TimestampMixin):
    """Top-level tenant entity. All data is scoped to an organisation."""

    __tablename__ = "organisations"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    contact_email: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        Enum(LifecycleStatus), default=LifecycleStatus.ACTIVE, nullable=False
    )
    # Governance configuration
    default_governance_framework: Mapped[Optional[str]] = mapped_column(String(100))
    risk_tolerance: Mapped[Optional[str]] = mapped_column(String(50))  # low/medium/high

    # Relationships
    business_units: Mapped[list["BusinessUnit"]] = relationship(
        "BusinessUnit", back_populates="organisation", cascade="all, delete-orphan"
    )
    users: Mapped[list["User"]] = relationship(
        "User", back_populates="organisation", cascade="all, delete-orphan"
    )
    projects: Mapped[list["Project"]] = relationship(
        "Project", back_populates="organisation", cascade="all, delete-orphan"
    )
    ai_systems: Mapped[list["AISystem"]] = relationship(
        "AISystem", back_populates="organisation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Organisation id={self.id} name={self.name!r}>"


class BusinessUnit(Base, UUIDPrimaryKey, TimestampMixin):
    """Business unit within an organisation."""

    __tablename__ = "business_units"

    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    department: Mapped[Optional[str]] = mapped_column(String(255))
    head_of_unit: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        Enum(LifecycleStatus), default=LifecycleStatus.ACTIVE, nullable=False
    )

    organisation: Mapped["Organisation"] = relationship(
        "Organisation", back_populates="business_units"
    )
    projects: Mapped[list["Project"]] = relationship(
        "Project", back_populates="business_unit"
    )

    def __repr__(self) -> str:
        return f"<BusinessUnit id={self.id} name={self.name!r}>"


class User(Base, UUIDPrimaryKey, TimestampMixin):
    """Platform user with role-based access control."""

    __tablename__ = "users"

    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    username: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        Enum(UserRole), default=UserRole.READ_ONLY, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[Optional[str]] = mapped_column(String(50))

    organisation: Mapped["Organisation"] = relationship(
        "Organisation", back_populates="users"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role}>"

    def has_permission(self, required_role: UserRole) -> bool:
        """
        Check if user has at least the required permission level.
        Inactive users are denied all permissions regardless of role.
        """
        if not self.is_active:
            return False
        role_hierarchy = [
            UserRole.READ_ONLY,
            UserRole.AUDITOR,
            UserRole.BUSINESS_OWNER,
            UserRole.TECHNICAL_EVALUATOR,
            UserRole.AI_EVALUATION_MANAGER,
            UserRole.AI_GOVERNANCE_MANAGER,
            UserRole.ORG_ADMIN,
            UserRole.PLATFORM_ADMIN,
        ]
        try:
            user_level = role_hierarchy.index(UserRole(self.role))
            required_level = role_hierarchy.index(required_role)
            return user_level >= required_level
        except ValueError:
            return False


class Project(Base, UUIDPrimaryKey, TimestampMixin):
    """An AI evaluation project, scoped to an organisation and optionally a business unit."""

    __tablename__ = "projects"

    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    business_unit_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("business_units.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    use_case: Mapped[Optional[str]] = mapped_column(String(255))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    evaluation_environment: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(
        Enum(LifecycleStatus), default=LifecycleStatus.ACTIVE, nullable=False
    )
    # Governance
    governance_framework: Mapped[Optional[str]] = mapped_column(String(100))
    risk_profile: Mapped[Optional[str]] = mapped_column(String(50))
    evaluation_policy: Mapped[Optional[str]] = mapped_column(Text)
    # Owners
    business_owner: Mapped[Optional[str]] = mapped_column(String(255))
    technical_owner: Mapped[Optional[str]] = mapped_column(String(255))

    organisation: Mapped["Organisation"] = relationship(
        "Organisation", back_populates="projects"
    )
    business_unit: Mapped[Optional["BusinessUnit"]] = relationship(
        "BusinessUnit", back_populates="projects"
    )
    ai_systems: Mapped[list["AISystem"]] = relationship(
        "AISystem", back_populates="project"
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r}>"


class AISystem(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Registered AI system entry.
    This is the inventory record for an AI system being governed and evaluated.
    """

    __tablename__ = "ai_systems"

    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    business_purpose: Mapped[Optional[str]] = mapped_column(Text)
    # Provider and model
    provider_name: Mapped[Optional[str]] = mapped_column(String(100))
    model_name: Mapped[Optional[str]] = mapped_column(String(255))
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    # Classification
    ai_type: Mapped[Optional[str]] = mapped_column(String(100))  # llm/ml/rag/agent/multimodal
    deployment_environment: Mapped[Optional[str]] = mapped_column(String(100))
    data_classification: Mapped[Optional[str]] = mapped_column(String(100))
    # Risk
    risk_classification: Mapped[Optional[str]] = mapped_column(String(50))  # low/medium/high/critical
    # Owners
    business_owner: Mapped[Optional[str]] = mapped_column(String(255))
    technical_owner: Mapped[Optional[str]] = mapped_column(String(255))
    # Status
    lifecycle_status: Mapped[str] = mapped_column(
        Enum(LifecycleStatus), default=LifecycleStatus.ACTIVE, nullable=False
    )
    review_date: Mapped[Optional[str]] = mapped_column(String(50))

    organisation: Mapped["Organisation"] = relationship(
        "Organisation", back_populates="ai_systems"
    )
    project: Mapped[Optional["Project"]] = relationship(
        "Project", back_populates="ai_systems"
    )

    def __repr__(self) -> str:
        return f"<AISystem id={self.id} name={self.name!r} model={self.model_name!r}>"
