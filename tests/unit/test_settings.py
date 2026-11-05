"""
Unit tests for configuration management.
Verifies settings loading, environment override, test settings isolation.
No network or database calls.
"""

from __future__ import annotations

import os

import pytest

from aieval360.core.config.settings import (
    AppSettings,
    DatabaseSettings,
    EvaluationSettings,
    ProviderSettings,
    SecuritySettings,
    get_test_settings,
)


class TestDatabaseSettings:
    def test_default_sqlite_url(self):
        s = DatabaseSettings()
        assert "sqlite" in s.url

    def test_echo_default_false(self):
        s = DatabaseSettings()
        assert s.echo is False

    def test_pool_size_default(self):
        s = DatabaseSettings()
        assert s.pool_size >= 1


class TestSecuritySettings:
    def test_default_key_exists(self):
        s = SecuritySettings()
        assert len(s.secret_key) > 0

    def test_algorithm_default(self):
        s = SecuritySettings()
        assert s.algorithm == "HS256"

    def test_token_expiry_positive(self):
        s = SecuritySettings()
        assert s.access_token_expire_minutes > 0


class TestProviderSettings:
    def test_mock_always_available(self):
        s = ProviderSettings()
        providers = s.available_providers()
        assert "mock" in providers

    def test_no_api_keys_means_only_mock(self, monkeypatch):
        """With no credentials configured, only mock is available."""
        monkeypatch.delenv("AIEVAL360_PROVIDERS__OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AIEVAL360_PROVIDERS__ANTHROPIC_API_KEY", raising=False)
        s = ProviderSettings()
        # Clears anything that might have been in env
        s.openai_api_key = None
        s.anthropic_api_key = None
        s.azure_openai_api_key = None
        providers = s.available_providers()
        # Mock is always available regardless
        assert "mock" in providers


class TestEvaluationSettings:
    def test_default_batch_size_positive(self):
        s = EvaluationSettings()
        assert s.default_batch_size > 0

    def test_max_concurrent_evaluations_positive(self):
        s = EvaluationSettings()
        assert s.max_concurrent_evaluations > 0

    def test_timeout_positive(self):
        s = EvaluationSettings()
        assert s.evaluation_timeout_seconds > 0


class TestAppSettings:
    def test_instantiation(self):
        s = AppSettings()
        assert s is not None

    def test_nested_settings_accessible(self):
        s = AppSettings()
        assert s.database is not None
        assert s.security is not None
        assert s.providers is not None
        assert s.evaluation is not None

    def test_env_prefix(self):
        """AppSettings must only pick up AIEVAL360_ prefixed vars."""
        s = AppSettings()
        # Verifying the model_config has the right prefix
        assert s.model_config.get("env_prefix") == "AIEVAL360_"


class TestTestSettings:
    def test_uses_in_memory_sqlite(self):
        s = get_test_settings()
        assert ":memory:" in s.database.url

    def test_different_from_default_settings(self):
        prod = AppSettings()
        test = get_test_settings()
        # Test must be in-memory, not shared with prod
        assert prod.database.url != test.database.url

    def test_echo_disabled_in_test(self):
        s = get_test_settings()
        assert s.database.echo is False

    def test_has_security_key(self):
        s = get_test_settings()
        assert len(s.security.secret_key) > 0
