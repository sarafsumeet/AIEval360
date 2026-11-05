"""
Governance, risk, and control models.
Separates governance from evaluation — connected through the assurance layer.
"""

from __future__ import annotations

import enum
import json
from typing import Optional

from sqlalchemy import Boolean, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDPrimaryKey


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlStatus(str, enum.Enum):
    NOT_APPLICABLE = "not_applicable"
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    EXCEPTION = "exception"


class EvidenceStatus(str, enum.Enum):
    CURRENT = "current"
    EXPIRED = "expired"
    MISSING = "missing"
    PENDING = "pending"


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"


class GovernanceFramework(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Reference governance framework (NIST AI RMF, ISO 42001, EU AI Act, etc.).
    Framework definitions are reference data — not modified per organisation.
    """

    __tablename__ = "governance_frameworks"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    short_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    version: Mapped[Optional[str]] = mapped_column(String(50))
    issuing_organisation: Mapped[Optional[str]] = mapped_column(String(255))
    publication_date: Mapped[Optional[str]] = mapped_column(String(50))
    official_url: Mapped[Optional[str]] = mapped_column(String(1000))
    description: Mapped[Optional[str]] = mapped_column(Text)
    scope: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_checked_date: Mapped[Optional[str]] = mapped_column(String(50))

    controls: Mapped[list["GovernanceControl"]] = relationship(
        "GovernanceControl", back_populates="framework"
    )

    def __repr__(self) -> str:
        return f"<GovernanceFramework short_name={self.short_name!r} version={self.version!r}>"


class GovernanceControl(Base, UUIDPrimaryKey, TimestampMixin):
    """
    A specific governance control within a framework.
    One AIEval360 control can map to multiple framework controls.
    """

    __tablename__ = "governance_controls"

    framework_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("governance_frameworks.id", ondelete="CASCADE"), nullable=False
    )
    control_id: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. "GOVERN-1.1"
    control_name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    control_category: Mapped[Optional[str]] = mapped_column(String(255))
    section_reference: Mapped[Optional[str]] = mapped_column(String(255))
    guidance: Mapped[Optional[str]] = mapped_column(Text)
    is_mandatory: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    applicability_notes: Mapped[Optional[str]] = mapped_column(Text)

    framework: Mapped["GovernanceFramework"] = relationship(
        "GovernanceFramework", back_populates="controls"
    )
    assessments: Mapped[list["ControlAssessment"]] = relationship(
        "ControlAssessment", back_populates="control"
    )

    def __repr__(self) -> str:
        return f"<GovernanceControl id={self.control_id!r} name={self.control_name[:50]!r}>"


class RiskAssessment(Base, UUIDPrimaryKey, TimestampMixin):
    """
    AI risk assessment for a specific AI system.
    Captures inherent risk, mitigating controls, and residual risk.
    """

    __tablename__ = "risk_assessments"

    organisation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False
    )
    ai_system_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_systems.id", ondelete="CASCADE"), nullable=False
    )

    assessment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    assessment_date: Mapped[str] = mapped_column(String(50), nullable=False)
    assessor: Mapped[Optional[str]] = mapped_column(String(255))
    reviewer: Mapped[Optional[str]] = mapped_column(String(255))
    review_date: Mapped[Optional[str]] = mapped_column(String(50))

    # Inherent risk
    inherent_risk_level: Mapped[str] = mapped_column(
        Enum(RiskLevel), default=RiskLevel.MEDIUM, nullable=False
    )
    inherent_risk_score: Mapped[Optional[float]] = mapped_column(Float)
    impact_level: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    likelihood_level: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    impact_description: Mapped[Optional[str]] = mapped_column(Text)
    likelihood_description: Mapped[Optional[str]] = mapped_column(Text)

    # Risk dimensions
    data_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    model_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    operational_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    privacy_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    fairness_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    security_risk: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))

    # Residual risk (after controls)
    residual_risk_level: Mapped[Optional[str]] = mapped_column(Enum(RiskLevel))
    residual_risk_score: Mapped[Optional[float]] = mapped_column(Float)
    mitigating_controls: Mapped[Optional[str]] = mapped_column(Text)
    risk_owner: Mapped[Optional[str]] = mapped_column(String(255))

    # Approval
    approval_status: Mapped[str] = mapped_column(
        Enum(ApprovalStatus), default=ApprovalStatus.PENDING, nullable=False
    )
    approved_by: Mapped[Optional[str]] = mapped_column(String(255))
    approval_date: Mapped[Optional[str]] = mapped_column(String(50))
    approval_notes: Mapped[Optional[str]] = mapped_column(Text)
    next_review_date: Mapped[Optional[str]] = mapped_column(String(50))

    risk_description: Mapped[Optional[str]] = mapped_column(Text)
    treatment_plan: Mapped[Optional[str]] = mapped_column(Text)

    control_assessments: Mapped[list["ControlAssessment"]] = relationship(
        "ControlAssessment", back_populates="risk_assessment"
    )

    def __repr__(self) -> str:
        return (
            f"<RiskAssessment id={self.id} inherent={self.inherent_risk_level} "
            f"residual={self.residual_risk_level}>"
        )


class ControlAssessment(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Assessment of a specific control for a specific risk assessment.
    Links the AI system risk to the governance framework control.
    """

    __tablename__ = "control_assessments"

    risk_assessment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("risk_assessments.id", ondelete="CASCADE"), nullable=False
    )
    control_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("governance_controls.id", ondelete="CASCADE"), nullable=False
    )

    implementation_status: Mapped[str] = mapped_column(
        Enum(ControlStatus), default=ControlStatus.NOT_IMPLEMENTED, nullable=False
    )
    implementation_notes: Mapped[Optional[str]] = mapped_column(Text)
    implementation_date: Mapped[Optional[str]] = mapped_column(String(50))
    implementer: Mapped[Optional[str]] = mapped_column(String(255))

    # Evidence
    evidence_status: Mapped[str] = mapped_column(
        Enum(EvidenceStatus), default=EvidenceStatus.MISSING, nullable=False
    )
    evidence_description: Mapped[Optional[str]] = mapped_column(Text)
    evidence_location: Mapped[Optional[str]] = mapped_column(Text)
    evidence_expiry_date: Mapped[Optional[str]] = mapped_column(String(50))

    # Exception
    has_exception: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    exception_reason: Mapped[Optional[str]] = mapped_column(Text)
    exception_approved_by: Mapped[Optional[str]] = mapped_column(String(255))
    exception_expiry_date: Mapped[Optional[str]] = mapped_column(String(50))

    remediation_plan: Mapped[Optional[str]] = mapped_column(Text)
    target_completion_date: Mapped[Optional[str]] = mapped_column(String(50))

    risk_assessment: Mapped["RiskAssessment"] = relationship(
        "RiskAssessment", back_populates="control_assessments"
    )
    control: Mapped["GovernanceControl"] = relationship(
        "GovernanceControl", back_populates="assessments"
    )

    def __repr__(self) -> str:
        return (
            f"<ControlAssessment control={self.control_id!r} "
            f"status={self.implementation_status}>"
        )


class AuditLog(Base, UUIDPrimaryKey, TimestampMixin):
    """
    Immutable audit trail for all significant platform events.
    Never deleted. Append-only by design.
    """

    __tablename__ = "audit_logs"

    organisation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(255))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    event_category: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100))
    resource_id: Mapped[Optional[str]] = mapped_column(String(36))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[Optional[str]] = mapped_column(Text)
    outcome: Mapped[Optional[str]] = mapped_column(String(50))
    ip_address: Mapped[Optional[str]] = mapped_column(String(50))
    session_id: Mapped[Optional[str]] = mapped_column(String(255))

    def get_details(self) -> dict:
        if self.details_json:
            return json.loads(self.details_json)
        return {}

    def __repr__(self) -> str:
        return f"<AuditLog event={self.event_type!r} resource={self.resource_type}/{self.resource_id}>"
