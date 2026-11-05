"""
Shared pytest fixtures and configuration.
Provides:
 - Async database session backed by in-memory SQLite
 - Pre-seeded test organisation, user, project, ai_system
 - Mock provider adapter instance
 - Test settings that never touch production resources
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from aieval360.core.config.settings import get_test_settings
from aieval360.core.models.base import utcnow, new_uuid
from aieval360.core.models.organisation import (
    AISystem,
    BusinessUnit,
    LifecycleStatus,
    Organisation,
    Project,
    User,
    UserRole,
)
from aieval360.core.models.evaluation import Dataset, EvaluationRun, EvaluationCategory, EvaluationStatus
from aieval360.core.persistence.database import (
    close_db,
    create_tables,
    drop_tables,
    get_session,
    init_db,
)
from aieval360.providers.mock.adapter import MockProviderAdapter
from aieval360.providers.base.interface import get_registry, reset_registry


# ---------------------------------------------------------------------------
# Session-scoped event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop scoped to the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_settings():
    return get_test_settings()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def db_engine(test_settings):
    """Create in-memory database engine for the test session."""
    engine = init_db(test_settings)
    await create_tables()
    yield engine
    await close_db()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a database session for each test.
    Uses a nested transaction (savepoint) so each test rolls back cleanly.
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from aieval360.core.persistence.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_org(db_session: AsyncSession) -> Organisation:
    """Create a test organisation."""
    org = Organisation(
        id=new_uuid(),
        name=f"Test Organisation {uuid.uuid4().hex[:6]}",
        display_name="Test Org",
        industry="banking",
        country="AU",
        status=LifecycleStatus.ACTIVE,
        default_governance_framework="NIST_AI_RMF",
        risk_tolerance="medium",
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(org)
    await db_session.flush()
    return org


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, test_org: Organisation) -> User:
    """Create a test user with AI_EVALUATION_MANAGER role."""
    user = User(
        id=new_uuid(),
        organisation_id=test_org.id,
        username=f"testuser_{uuid.uuid4().hex[:6]}",
        email=f"testuser_{uuid.uuid4().hex[:6]}@test.example.com",
        full_name="Test User",
        hashed_password="$2b$12$fakehashfortest",
        role=UserRole.AI_EVALUATION_MANAGER,
        is_active=True,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def test_project(db_session: AsyncSession, test_org: Organisation) -> Project:
    """Create a test project."""
    project = Project(
        id=new_uuid(),
        organisation_id=test_org.id,
        name="Test Evaluation Project",
        description="Automated test project",
        use_case="credit_risk",
        industry="banking",
        status=LifecycleStatus.ACTIVE,
        governance_framework="NIST_AI_RMF",
        risk_profile="high",
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(project)
    await db_session.flush()
    return project


@pytest_asyncio.fixture
async def test_ai_system(
    db_session: AsyncSession, test_org: Organisation, test_project: Project
) -> AISystem:
    """Create a test AI system (inventory record)."""
    system = AISystem(
        id=new_uuid(),
        organisation_id=test_org.id,
        project_id=test_project.id,
        name="Credit Risk LLM Classifier",
        description="LLM-based credit risk assessment model",
        provider_name="mock",
        model_name="mock-gpt-v1",
        model_version="1.0.0",
        ai_type="llm",
        deployment_environment="production",
        data_classification="confidential",
        risk_classification="high",
        lifecycle_status=LifecycleStatus.ACTIVE,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(system)
    await db_session.flush()
    return system


@pytest_asyncio.fixture
async def test_dataset(
    db_session: AsyncSession, test_org: Organisation
) -> Dataset:
    """Create a test evaluation dataset."""
    dataset = Dataset(
        id=new_uuid(),
        organisation_id=test_org.id,
        name="Banking Credit Risk Test Dataset",
        version="1.0.0",
        description="Synthetic banking dataset for credit risk LLM evaluation",
        category="banking",
        evaluation_type="llm_benchmark",
        record_count=100,
        is_synthetic=True,
        checksum="abc123def456" + "0" * 52,  # 64-char placeholder
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(dataset)
    await db_session.flush()
    return dataset


@pytest_asyncio.fixture
async def test_evaluation_run(
    db_session: AsyncSession,
    test_org: Organisation,
    test_project: Project,
    test_ai_system: AISystem,
    test_dataset: Dataset,
    test_user: User,
) -> EvaluationRun:
    """Create a completed evaluation run evidence record."""
    import json

    run = EvaluationRun(
        id=new_uuid(),
        organisation_id=test_org.id,
        project_id=test_project.id,
        ai_system_id=test_ai_system.id,
        dataset_id=test_dataset.id,
        evaluation_category=EvaluationCategory.LLM_BENCHMARK,
        name="Credit Risk LLM Evaluation Q4-2026",
        description="Quarterly evaluation of credit risk LLM",
        provider_name="mock",
        model_name="mock-gpt-v1",
        model_version="1.0.0",
        model_parameters_json=json.dumps({"temperature": 0.0, "max_tokens": 1024}),
        prompt_template_version="v1.2.3",
        judge_model="mock-judge-v1",
        judge_model_version="1.0.0",
        evaluation_config_json=json.dumps({"batch_size": 20, "timeout": 300}),
        metric_definitions_version="v1.0.0",
        thresholds_json=json.dumps({"accuracy": 0.85, "f1": 0.80}),
        status=EvaluationStatus.COMPLETED,
        started_at="2026-11-04T10:00:00Z",
        completed_at="2026-11-04T10:15:00Z",
        duration_seconds=900.0,
        executed_by=test_user.username,
        overall_score=0.87,
        is_reproducible=True,
        reproducibility_notes="Seed=42, temperature=0.0, deterministic sampling",
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db_session.add(run)
    await db_session.flush()
    return run


# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_provider() -> MockProviderAdapter:
    """Fresh mock provider adapter for each test."""
    return MockProviderAdapter()


@pytest.fixture
def mock_provider_with_latency() -> MockProviderAdapter:
    """Mock provider that simulates network latency."""
    return MockProviderAdapter(simulated_latency_ms=50.0)


@pytest.fixture
def failing_mock_provider() -> MockProviderAdapter:
    """Mock provider that fails on the first call."""
    from aieval360.providers.base.interface import ProviderError
    return MockProviderAdapter(
        fail_on_calls=[0],
        fail_with=ProviderError("Injected test failure", "mock", "TEST_ERROR"),
    )


@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Reset the provider registry before each test."""
    reset_registry()
    yield
    reset_registry()
