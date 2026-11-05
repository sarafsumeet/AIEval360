"""
AIEval360 Configuration Management.
Uses pydantic-settings for environment-based configuration.
Supports local development, cloud, and hybrid deployments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="AIEVAL360_DB_", extra="ignore")

    url: str = Field(
        default="sqlite+aiosqlite:///./aieval360.db",
        description="SQLAlchemy database URL. Use sqlite for local dev, postgresql for production.",
    )
    echo: bool = Field(default=False, description="Echo SQL statements (dev only).")
    pool_size: int = Field(default=5, ge=1, le=50)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=5, le=300)


class SecuritySettings(BaseSettings):
    """Security configuration — NEVER hardcode in source."""

    model_config = SettingsConfigDict(env_prefix="AIEVAL360_SECURITY_", extra="ignore")

    secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-USE-ENV-VAR",
        description="JWT signing key. Must be changed in any deployed environment.",
    )
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    password_min_length: int = Field(default=12, ge=8)
    bcrypt_rounds: int = Field(default=12, ge=10, le=14)
    rate_limit_requests_per_minute: int = Field(default=60, ge=10, le=1000)
    enable_audit_logging: bool = Field(default=True)

    @field_validator("secret_key")
    @classmethod
    def warn_insecure_key(cls, v: str) -> str:
        if v == "CHANGE-ME-IN-PRODUCTION-USE-ENV-VAR":
            import warnings
            warnings.warn(
                "AIEVAL360_SECURITY_SECRET_KEY is using the default insecure value. "
                "Set the environment variable before any real deployment.",
                RuntimeWarning,
                stacklevel=2,
            )
        return v


class ProviderSettings(BaseSettings):
    """AI provider API key configuration. Keys loaded from environment only."""

    model_config = SettingsConfigDict(env_prefix="AIEVAL360_PROVIDER_", extra="ignore")

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key.")
    openai_org_id: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-10-21")
    anthropic_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    huggingface_api_key: Optional[str] = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_timeout_seconds: int = Field(default=120, ge=10, le=600)
    default_timeout_seconds: int = Field(default=60, ge=5, le=300)
    default_max_retries: int = Field(default=3, ge=0, le=10)

    def available_providers(self) -> list[str]:
        """Return list of configured (non-mock) providers."""
        providers = ["mock"]  # Always available
        if self.openai_api_key:
            providers.append("openai")
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            providers.append("azure_openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.huggingface_api_key:
            providers.append("huggingface")
        # Ollama is available if the endpoint is reachable (checked at runtime)
        providers.append("ollama")
        return providers


class EvaluationSettings(BaseSettings):
    """Evaluation engine configuration."""

    model_config = SettingsConfigDict(env_prefix="AIEVAL360_EVAL_", extra="ignore")

    default_batch_size: int = Field(default=20, ge=1, le=1000)
    max_concurrent_evaluations: int = Field(default=5, ge=1, le=50)
    evaluation_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    judge_model_provider: str = Field(default="mock")
    judge_model_name: str = Field(default="mock-judge-v1")
    enable_pii_detection: bool = Field(default=True)
    pii_detection_sensitivity: str = Field(default="medium")  # low/medium/high
    enable_cost_tracking: bool = Field(default=True)
    max_tokens_per_request: int = Field(default=4096, ge=1, le=128000)
    default_top_k: int = Field(default=5, ge=1, le=100)


class StorageSettings(BaseSettings):
    """File storage configuration."""

    model_config = SettingsConfigDict(env_prefix="AIEVAL360_STORAGE_", extra="ignore")

    data_dir: Path = Field(default=Path("./data"))
    evidence_dir: Path = Field(default=Path("./evidence"))
    reports_dir: Path = Field(default=Path("./reports"))
    datasets_dir: Path = Field(default=Path("./data/datasets"))
    max_upload_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_dataset_formats: list[str] = Field(default=["json", "csv", "jsonl"])
    retention_days: int = Field(default=365, ge=30, le=3650)

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for dir_path in [self.data_dir, self.evidence_dir, self.reports_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AIEVAL360_",
        extra="ignore",
    )

    # Application identity
    app_name: str = Field(default="AIEval360")
    app_version: str = Field(default="1.0.0")
    app_tagline: str = Field(default="From AI Governance to Measurable AI Assurance.")
    environment: str = Field(default="development")  # development/staging/production
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_prefix: str = Field(default="/api/v1")
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8501"])

    # Feature flags
    enable_multi_tenancy: bool = Field(default=True)
    enable_continuous_evaluation: bool = Field(default=False)
    enable_ui: bool = Field(default=True)

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "testing", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    def is_production(self) -> bool:
        return self.environment == "production"

    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return cached application settings.
    Settings are loaded once at startup and cached for the application lifetime.
    """
    return AppSettings()


def get_test_settings() -> AppSettings:
    """
    Return settings configured for test execution.
    Uses an in-memory SQLite database and disables rate limiting.
    """
    # Clear lru_cache to allow fresh settings for tests
    settings = AppSettings(
        environment="testing",
        debug=True,
        log_level="WARNING",
    )
    settings.database = DatabaseSettings(url="sqlite+aiosqlite:///:memory:", echo=False)
    settings.security = SecuritySettings(
        secret_key="test-secret-key-not-for-production",
        access_token_expire_minutes=5,
    )
    settings.evaluation = EvaluationSettings(
        judge_model_provider="mock",
        judge_model_name="mock-judge-v1",
    )
    return settings
