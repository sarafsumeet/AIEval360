"""
Repository pattern — data access layer.
Each repository handles CRUD and query operations for one domain entity.
Keeps database logic out of business logic and API layers.
"""

from __future__ import annotations

import json
from typing import Any, Generic, Optional, Type, TypeVar
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aieval360.core.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic repository providing standard CRUD operations."""

    def __init__(self, session: AsyncSession, model_class: Type[ModelT]) -> None:
        self.session = session
        self.model_class = model_class

    async def get(self, record_id: str) -> Optional[ModelT]:
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == record_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[ModelT]:
        result = await self.session.execute(
            select(self.model_class).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, record_id: str) -> bool:
        instance = await self.get(record_id)
        if instance is None:
            return False
        await self.session.delete(instance)
        await self.session.flush()
        return True

    async def count(self) -> int:
        from sqlalchemy import func
        result = await self.session.execute(
            select(func.count()).select_from(self.model_class)
        )
        return result.scalar_one()


# ---------------------------------------------------------------------------
# Domain-specific repositories
# ---------------------------------------------------------------------------

class OrganisationRepository(BaseRepository):
    """Repository for Organisation entities."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.organisation import Organisation
        super().__init__(session, Organisation)

    async def get_by_name(self, name: str):
        from aieval360.core.models.organisation import Organisation
        result = await self.session.execute(
            select(Organisation).where(Organisation.name == name)
        )
        return result.scalar_one_or_none()

    async def get_active(self):
        from aieval360.core.models.organisation import Organisation, LifecycleStatus
        result = await self.session.execute(
            select(Organisation).where(Organisation.status == LifecycleStatus.ACTIVE)
        )
        return list(result.scalars().all())


class UserRepository(BaseRepository):
    """Repository for User entities."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.organisation import User
        super().__init__(session, User)

    async def get_by_username(self, username: str):
        from aieval360.core.models.organisation import User
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str):
        from aieval360.core.models.organisation import User
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_org(self, organisation_id: str):
        from aieval360.core.models.organisation import User
        result = await self.session.execute(
            select(User).where(User.organisation_id == organisation_id)
        )
        return list(result.scalars().all())


class AISystemRepository(BaseRepository):
    """Repository for AI system inventory."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.organisation import AISystem
        super().__init__(session, AISystem)

    async def get_by_org(self, organisation_id: str):
        from aieval360.core.models.organisation import AISystem
        result = await self.session.execute(
            select(AISystem).where(AISystem.organisation_id == organisation_id)
        )
        return list(result.scalars().all())

    async def get_by_project(self, project_id: str):
        from aieval360.core.models.organisation import AISystem
        result = await self.session.execute(
            select(AISystem).where(AISystem.project_id == project_id)
        )
        return list(result.scalars().all())


class DatasetRepository(BaseRepository):
    """Repository for evaluation datasets."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.evaluation import Dataset
        super().__init__(session, Dataset)

    async def get_by_org(self, organisation_id: str):
        from aieval360.core.models.evaluation import Dataset
        result = await self.session.execute(
            select(Dataset).where(Dataset.organisation_id == organisation_id)
        )
        return list(result.scalars().all())

    async def get_by_checksum(self, checksum: str):
        from aieval360.core.models.evaluation import Dataset
        result = await self.session.execute(
            select(Dataset).where(Dataset.checksum == checksum)
        )
        return result.scalar_one_or_none()


class EvaluationRunRepository(BaseRepository):
    """Repository for evaluation run evidence records."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.evaluation import EvaluationRun
        super().__init__(session, EvaluationRun)

    async def get_by_org(self, organisation_id: str, limit: int = 50):
        from aieval360.core.models.evaluation import EvaluationRun
        result = await self.session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.organisation_id == organisation_id)
            .order_by(EvaluationRun.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_ai_system(self, ai_system_id: str, limit: int = 20):
        from aieval360.core.models.evaluation import EvaluationRun
        result = await self.session.execute(
            select(EvaluationRun)
            .where(EvaluationRun.ai_system_id == ai_system_id)
            .order_by(EvaluationRun.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(self, run_id: str, status: str, **kwargs: Any) -> None:
        from aieval360.core.models.evaluation import EvaluationRun
        values = {"status": status, **kwargs}
        await self.session.execute(
            update(EvaluationRun)
            .where(EvaluationRun.id == run_id)
            .values(**values)
        )
        await self.session.flush()


class AuditLogRepository(BaseRepository):
    """Repository for the immutable audit log."""

    def __init__(self, session: AsyncSession) -> None:
        from aieval360.core.models.governance import AuditLog
        super().__init__(session, AuditLog)

    async def log_event(
        self,
        event_type: str,
        event_category: str,
        description: str,
        organisation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
        outcome: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        from aieval360.core.models.governance import AuditLog
        from aieval360.core.models.base import utcnow, new_uuid

        log = AuditLog(
            id=new_uuid(),
            organisation_id=organisation_id,
            user_id=user_id,
            username=username,
            event_type=event_type,
            event_category=event_category,
            description=description,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=json.dumps(details) if details else None,
            outcome=outcome,
            ip_address=ip_address,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_by_resource(self, resource_type: str, resource_id: str):
        from aieval360.core.models.governance import AuditLog
        result = await self.session.execute(
            select(AuditLog)
            .where(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id,
            )
            .order_by(AuditLog.created_at.desc())
        )
        return list(result.scalars().all())
