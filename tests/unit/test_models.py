"""
Unit tests for domain models.
Verifies model creation, field validation, User.has_permission(),
and evidence record completeness.
Pure Python — no database or async.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from aieval360.core.models.base import new_uuid, utcnow
from aieval360.core.models.organisation import (
    AISystem,
    BusinessUnit,
    LifecycleStatus,
    Organisation,
    Project,
    User,
    UserRole,
)
from aieval360.core.models.evaluation import (
    Dataset,
    EvaluationCategory,
    EvaluationRun,
    EvaluationStatus,
    MetricResult,
    MetricType,
)
from aieval360.core.models.governance import (
    ApprovalStatus,
    AuditLog,
    ControlStatus,
    EvidenceStatus,
    GovernanceControl,
    RiskAssessment,
    RiskLevel,
)


class TestBaseHelpers:
    def test_utcnow_returns_datetime(self):
        dt = utcnow()
        assert isinstance(dt, datetime)

    def test_utcnow_is_utc(self):
        dt = utcnow()
        # Should have timezone info
        assert dt.tzinfo is not None

    def test_new_uuid_returns_string(self):
        """new_uuid() returns a string UUID (SQLAlchemy UUID column uses str)."""
        uid = new_uuid()
        assert isinstance(uid, str)
        # Must be a valid UUID format
        parsed = uuid.UUID(uid)
        assert str(parsed) == uid

    def test_new_uuid_unique(self):
        assert new_uuid() != new_uuid()


class TestUserRole:
    def test_all_expected_roles_exist(self):
        expected_roles = [
            "platform_admin",
            "org_admin",
            "ai_governance_manager",
            "ai_evaluation_manager",
            "technical_evaluator",
            "business_owner",
            "auditor",
            "read_only",
        ]
        actual = [r.value for r in UserRole]
        for role in expected_roles:
            assert role in actual, f"Expected role {role!r} not found in UserRole"

    def test_has_permission_platform_admin_beats_all(self):
        """Platform admin role is at top of hierarchy."""
        user = User(
            id=new_uuid(),
            organisation_id=new_uuid(),
            username="admin",
            email="admin@test.com",
            full_name="Admin User",
            hashed_password="x",
            role=UserRole.PLATFORM_ADMIN,
            is_active=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        # Platform admin has permission for any role at or below them
        assert user.has_permission(UserRole.AI_EVALUATION_MANAGER) is True
        assert user.has_permission(UserRole.READ_ONLY) is True
        assert user.has_permission(UserRole.ORG_ADMIN) is True

    def test_has_permission_read_only_cannot_do_manager_actions(self):
        user = User(
            id=new_uuid(),
            organisation_id=new_uuid(),
            username="viewer",
            email="viewer@test.com",
            full_name="Viewer",
            hashed_password="x",
            role=UserRole.READ_ONLY,
            is_active=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        assert user.has_permission(UserRole.AI_EVALUATION_MANAGER) is False
        assert user.has_permission(UserRole.ORG_ADMIN) is False

    def test_has_permission_auditor_beats_read_only(self):
        """Auditor role is above READ_ONLY in hierarchy."""
        user = User(
            id=new_uuid(),
            organisation_id=new_uuid(),
            username="auditor",
            email="auditor@test.com",
            full_name="Auditor",
            hashed_password="x",
            role=UserRole.AUDITOR,
            is_active=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        # Auditor can do anything READ_ONLY can
        assert user.has_permission(UserRole.READ_ONLY) is True
        # Auditor cannot do what AI_EVALUATION_MANAGER requires
        assert user.has_permission(UserRole.AI_EVALUATION_MANAGER) is False

    def test_inactive_user_cannot_do_anything(self):
        user = User(
            id=new_uuid(),
            organisation_id=new_uuid(),
            username="inactive",
            email="inactive@test.com",
            full_name="Inactive",
            hashed_password="x",
            role=UserRole.AI_EVALUATION_MANAGER,
            is_active=False,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        # Inactive users should be denied regardless of role
        assert user.has_permission(UserRole.READ_ONLY) is False


class TestMetricType:
    def test_all_five_metric_types_exist(self):
        """The 5-category taxonomy must be complete — it documents evaluation transparency."""
        expected = [
            "deterministic",
            "reference_based",
            "heuristic",
            "llm_judged",
            "human_assessed",
        ]
        actual = [m.value for m in MetricType]
        for t in expected:
            assert t in actual

    def test_llm_judged_distinct_from_deterministic(self):
        """LLM_JUDGED is a separate category — prevents hiding AI judgment as objective truth."""
        assert MetricType.LLM_JUDGED != MetricType.DETERMINISTIC


class TestDatasetModel:
    def test_synthetic_flag_can_be_set_false(self):
        ds = Dataset(
            id=new_uuid(),
            organisation_id=new_uuid(),
            name="Real Dataset",
            version="1.0.0",
            description="Real data",
            category="banking",
            evaluation_type="llm_benchmark",
            record_count=100,
            is_synthetic=False,
            checksum="a" * 64,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        assert ds.is_synthetic is False

    def test_synthetic_flag_can_be_set_true(self):
        ds = Dataset(
            id=new_uuid(),
            organisation_id=new_uuid(),
            name="Synthetic Dataset",
            version="1.0.0",
            description="Synthetic",
            category="banking",
            evaluation_type="llm_benchmark",
            record_count=50,
            is_synthetic=True,
            checksum="b" * 64,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        assert ds.is_synthetic is True


class TestEvaluationRunEvidenceFields:
    """
    Verify that EvaluationRun carries all fields required by the evidence model.
    An AI Governance Committee needs: which model, which dataset, what configuration,
    who ran it, what judge was used, and reproducibility notes.
    """

    def _make_run(self, **overrides) -> EvaluationRun:
        defaults = dict(
            id=new_uuid(),
            organisation_id=new_uuid(),
            project_id=new_uuid(),
            ai_system_id=new_uuid(),
            dataset_id=new_uuid(),
            evaluation_category=EvaluationCategory.LLM_BENCHMARK,
            name="Evidence Test Run",
            provider_name="mock",
            model_name="mock-gpt-v1",
            model_version="1.0.0",
            status=EvaluationStatus.COMPLETED,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        defaults.update(overrides)
        return EvaluationRun(**defaults)

    def test_has_provider_name(self):
        run = self._make_run()
        assert run.provider_name == "mock"

    def test_has_model_name(self):
        run = self._make_run()
        assert run.model_name == "mock-gpt-v1"

    def test_has_model_version(self):
        run = self._make_run()
        assert run.model_version == "1.0.0"

    def test_has_dataset_id(self):
        run = self._make_run()
        assert run.dataset_id is not None

    def test_has_judge_model_field(self):
        run = self._make_run(judge_model="mock-judge-v1")
        assert run.judge_model == "mock-judge-v1"

    def test_has_reproducibility_flag(self):
        run = self._make_run(is_reproducible=True)
        assert run.is_reproducible is True

    def test_has_reproducibility_notes(self):
        run = self._make_run(reproducibility_notes="Seed=42, temperature=0.0")
        assert "Seed=42" in run.reproducibility_notes

    def test_has_executed_by(self):
        run = self._make_run(executed_by="testuser")
        assert run.executed_by == "testuser"

    def test_has_metric_definitions_version(self):
        run = self._make_run(metric_definitions_version="v1.0.0")
        assert run.metric_definitions_version == "v1.0.0"


class TestMetricResultMandatoryGate:
    """
    Mandatory gates enforce that a failed critical control cannot be offset
    by high scores elsewhere — a key governance requirement.
    """

    def test_mandatory_gate_field_exists(self):
        """MetricResult must have is_mandatory_gate to enforce critical control failures."""
        result = MetricResult(
            id=new_uuid(),
            evaluation_run_id=new_uuid(),
            metric_name="bias_detection",
            metric_type=MetricType.LLM_JUDGED,
            value=0.45,
            threshold=0.80,
            outcome="fail",
            is_mandatory_gate=True,
            created_at=utcnow(),
        )
        assert result.is_mandatory_gate is True

    def test_non_gate_metric(self):
        result = MetricResult(
            id=new_uuid(),
            evaluation_run_id=new_uuid(),
            metric_name="style_consistency",
            metric_type=MetricType.HEURISTIC,
            value=0.70,
            threshold=0.80,
            outcome="fail",
            is_mandatory_gate=False,
            created_at=utcnow(),
        )
        assert result.is_mandatory_gate is False

    def test_outcome_field_records_pass_fail(self):
        pass_result = MetricResult(
            id=new_uuid(),
            evaluation_run_id=new_uuid(),
            metric_name="accuracy",
            metric_type=MetricType.DETERMINISTIC,
            value=0.95,
            threshold=0.80,
            outcome="pass",
            is_mandatory_gate=True,
            created_at=utcnow(),
        )
        assert pass_result.outcome == "pass"


class TestGovernanceModels:
    def test_risk_levels_exist(self):
        """Risk levels must exist — governance risk assessment requires them."""
        expected = ["low", "medium", "high", "critical"]
        actual = [r.value for r in RiskLevel]
        for level in expected:
            assert level in actual

    def test_control_status_values(self):
        expected = [
            "not_applicable",
            "not_implemented",
            "partially_implemented",
            "implemented",
            "exception",
        ]
        actual = [s.value for s in ControlStatus]
        for s in expected:
            assert s in actual

    def test_audit_log_has_required_fields(self):
        """AuditLog must be append-only with enough context for forensic review."""
        log = AuditLog(
            id=new_uuid(),
            organisation_id=new_uuid(),
            event_type="EVALUATION_APPROVED",
            user_id=new_uuid(),
            username="governance.officer",
            resource_type="EvaluationRun",
            resource_id=new_uuid(),
            description="Approved Q4 credit risk evaluation",
            created_at=utcnow(),
        )
        assert log.event_type is not None
        assert log.username is not None
        assert log.resource_type is not None

    def test_governance_control_has_framework_reference(self):
        """GovernanceControl must carry a framework_id for traceability."""
        ctrl = GovernanceControl(
            id=new_uuid(),
            framework_id=new_uuid(),
            control_id="NIST-MS-1.1",
            control_name="Model Risk Management",
            is_mandatory=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        assert ctrl.framework_id is not None
        assert ctrl.is_mandatory is True
