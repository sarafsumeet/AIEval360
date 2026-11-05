# Stage 1 — Foundation

**Files:** 11 source · 3 test files  
**Purpose:** Core domain models, provider adapter pattern, async persistence layer

---

## What Was Built

Stage 1 establishes the foundational scaffolding that all later stages build on. Nothing in Stages 2–4 bypasses or replaces these components — they are consumed by the layers above.

---

## Module Reference

### `aieval360/core/config/settings.py`

Application settings loaded from environment variables via Pydantic `BaseSettings`.

**Key settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./aieval360.db` | Async database connection string |
| `OPENAI_API_KEY` | — | Optional; only needed for OpenAI adapter |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service endpoint |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `MAX_CONCURRENT_EVALUATIONS` | `10` | Semaphore limit for async evaluation runs |

Usage:
```python
from aieval360.core.config.settings import settings
print(settings.DATABASE_URL)
```

---

### `aieval360/core/models/`

Four domain model files, all using Pydantic v2 `BaseModel`.

#### `base.py` — `AuditedModel`
Base class for all persisted entities. Adds:
- `id: UUID` — auto-generated
- `created_at: datetime` — UTC timestamp
- `updated_at: datetime` — updated on every write
- `created_by: str` — executor identity
- `metadata: dict` — arbitrary governance context

#### `evaluation.py` — `EvaluationRun`, `EvaluationResult`

`EvaluationRun` tracks a full evaluation job:
- `run_id`, `model_id`, `dataset_id`, `status`, `run_label`
- `started_at`, `completed_at`, `executed_by`

`EvaluationResult` stores individual metric outcomes:
- `run_id`, `metric_name`, `metric_value`, `metric_type`
- `input_hash`, `details`

#### `governance.py` — `GovernanceRecord`, `RiskAssessment`

`GovernanceRecord` is the primary audit trail entity:
- Links an `EvaluationRun` to a governance decision
- Tracks `risk_level`, `approved_by`, `approval_date`, `decision`, `notes`

`RiskAssessment` captures model risk:
- `overall_risk_level`: LOW / MEDIUM / HIGH / CRITICAL
- `risk_dimensions`: dict of dimension → score
- `mandatory_controls`: list of required remediation actions

#### `organisation.py` — `Organisation`, `ModelCard`

`Organisation`: tenant model for multi-org deployment.

`ModelCard` captures model provenance:
- `model_id`, `model_name`, `model_version`, `provider`
- `intended_use`, `out_of_scope_uses`
- `training_data_description`, `known_limitations`
- `fairness_considerations`, `governance_contact`

---

### `aieval360/providers/`

#### `base/interface.py` — `BaseProviderAdapter`

Abstract base class all adapters implement:

```python
class BaseProviderAdapter(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    async def complete_with_history(
        self, messages: list[dict], **kwargs
    ) -> str: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
```

#### `mock/adapter.py` — `MockProviderAdapter`

Deterministic test adapter. Responses are configured at construction:

```python
adapter = MockProviderAdapter(responses=["response 1", "response 2"])
# Returns responses in order; cycles on exhaustion
```

Useful for safety probe testing (always refuse, always respond, etc.).

#### `openai/adapter.py` — `OpenAIAdapter`

Production adapter for OpenAI-compatible APIs. Reads `OPENAI_API_KEY` from settings. Supports `model`, `temperature`, `max_tokens` kwargs.

#### `ollama/adapter.py` — `OllamaAdapter`

Adapter for locally-hosted Ollama models. Reads `OLLAMA_BASE_URL` from settings. No API key required.

---

### `aieval360/core/persistence/`

#### `database.py`

Async SQLAlchemy 2.0 engine and session factory:

```python
from aieval360.core.persistence.database import get_session

async with get_session() as session:
    # session is AsyncSession
    ...
```

#### `repositories.py`

Repository pattern for each domain entity. Each repository provides:
- `save(entity)` → persisted entity
- `get_by_id(id)` → entity or None
- `list(filters)` → list of entities

---

## Test Coverage

| File | Tests | Covers |
|------|-------|--------|
| `tests/unit/test_models.py` | — | AuditedModel, EvaluationRun, GovernanceRecord, ModelCard |
| `tests/unit/test_provider_interface.py` | — | MockAdapter responses, BaseProviderAdapter contract |
| `tests/unit/test_settings.py` | — | Settings defaults, env var overrides |

---

## Dependencies Introduced

```toml
[project.dependencies]
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
sqlalchemy = {version = ">=2.0", extras = ["asyncio"]}
aiosqlite = "*"
fastapi = ">=0.115"
httpx = "*"
```
