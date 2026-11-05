"""
Unit tests for the provider abstraction layer.
Tests the interface contract, mock provider, and registry.
No real API calls are made.
"""

from __future__ import annotations

import pytest

from aieval360.providers.base.interface import (
    CompletionRequest,
    EmbeddingRequest,
    ModelInfo,
    ProviderCapability,
    ProviderError,
    ProviderRegistry,
    get_registry,
    reset_registry,
)
from aieval360.providers.mock.adapter import MockProviderAdapter


class TestMockProviderBasics:
    """Test mock provider properties and registration."""

    def test_provider_name(self, mock_provider):
        assert mock_provider.provider_name == "mock"

    def test_has_expected_capabilities(self, mock_provider):
        assert ProviderCapability.CHAT_COMPLETION in mock_provider.capabilities
        assert ProviderCapability.EMBEDDINGS in mock_provider.capabilities
        assert ProviderCapability.TEXT_GENERATION in mock_provider.capabilities

    def test_has_capability_method(self, mock_provider):
        assert mock_provider.has_capability(ProviderCapability.CHAT_COMPLETION) is True
        assert mock_provider.has_capability(ProviderCapability.AUDIO) is False

    def test_repr(self, mock_provider):
        r = repr(mock_provider)
        assert "MockProviderAdapter" in r
        assert "mock" in r


class TestMockProviderCompletion:
    """Test mock completion responses."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-gpt-v1",
        )
        resp = await mock_provider.complete(req)

        assert resp.content is not None
        assert len(resp.content) > 0
        assert resp.provider_name == "mock"
        assert resp.model == "mock-gpt-v1"

    @pytest.mark.asyncio
    async def test_response_includes_token_counts(self, mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Tell me about AI governance."}],
            model="mock-gpt-v1",
        )
        resp = await mock_provider.complete(req)

        assert resp.input_tokens is not None
        assert resp.output_tokens is not None
        assert resp.total_tokens == resp.input_tokens + resp.output_tokens

    @pytest.mark.asyncio
    async def test_response_is_deterministic_for_same_input(self, mock_provider):
        """Same input should produce same output — critical for reproducibility."""
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Evaluate the model accuracy."}],
            model="mock-gpt-v1",
        )
        resp1 = await mock_provider.complete(req)
        mock_provider.reset()
        resp2 = await mock_provider.complete(req)

        assert resp1.content == resp2.content

    @pytest.mark.asyncio
    async def test_call_count_increments(self, mock_provider):
        assert mock_provider.call_count == 0
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Test"}],
            model="mock-gpt-v1",
        )
        await mock_provider.complete(req)
        assert mock_provider.call_count == 1
        await mock_provider.complete(req)
        assert mock_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_call_log_records_interactions(self, mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Log this"}],
            model="mock-gpt-v1",
        )
        await mock_provider.complete(req)
        assert len(mock_provider.call_log) == 1
        assert mock_provider.call_log[0].method == "complete"

    @pytest.mark.asyncio
    async def test_reset_clears_call_log(self, mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Test"}],
            model="mock-gpt-v1",
        )
        await mock_provider.complete(req)
        mock_provider.reset()
        assert mock_provider.call_count == 0
        assert len(mock_provider.call_log) == 0

    @pytest.mark.asyncio
    async def test_fixed_response(self):
        provider = MockProviderAdapter(fixed_response="Fixed test response")
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Anything"}],
            model="mock-gpt-v1",
        )
        resp = await provider.complete(req)
        assert resp.content == "Fixed test response"

    @pytest.mark.asyncio
    async def test_cost_estimate_is_non_negative(self, mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Estimate cost"}],
            model="mock-gpt-v1",
        )
        resp = await mock_provider.complete(req)
        assert resp.estimated_cost_usd is not None
        assert resp.estimated_cost_usd >= 0.0


class TestMockProviderFailureInjection:
    """Test that failure injection works correctly for error-path testing."""

    @pytest.mark.asyncio
    async def test_fails_on_specified_call(self, failing_mock_provider):
        req = CompletionRequest(
            messages=[{"role": "user", "content": "This will fail"}],
            model="mock-gpt-v1",
        )
        with pytest.raises(ProviderError) as exc_info:
            await failing_mock_provider.complete(req)
        assert "TEST_ERROR" in str(exc_info.value.error_code)

    @pytest.mark.asyncio
    async def test_succeeds_after_failure_call(self, failing_mock_provider):
        """Provider should succeed on calls not in the fail_on_calls set."""
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Test"}],
            model="mock-gpt-v1",
        )
        # First call (index 0) fails
        with pytest.raises(ProviderError):
            await failing_mock_provider.complete(req)

        # Second call (index 1) succeeds
        resp = await failing_mock_provider.complete(req)
        assert resp.content is not None


class TestMockProviderEmbeddings:
    """Test mock embedding generation."""

    @pytest.mark.asyncio
    async def test_generates_embeddings(self, mock_provider):
        req = EmbeddingRequest(
            texts=["AI governance is important", "Risk assessment framework"],
            model="mock-embed-v1",
        )
        resp = await mock_provider.embed(req)

        assert len(resp.embeddings) == 2
        assert len(resp.embeddings[0]) == 1536  # Matching text-embedding dimensions
        assert resp.provider_name == "mock"

    @pytest.mark.asyncio
    async def test_embeddings_are_normalised(self, mock_provider):
        """Embeddings should have unit length (L2 norm ≈ 1.0)."""
        import math
        req = EmbeddingRequest(
            texts=["normalisation test"],
            model="mock-embed-v1",
        )
        resp = await mock_provider.embed(req)
        vec = resp.embeddings[0]
        magnitude = math.sqrt(sum(x**2 for x in vec))
        assert abs(magnitude - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_embeddings_deterministic(self, mock_provider):
        """Same text should always produce same embedding."""
        req = EmbeddingRequest(texts=["determinism test"], model="mock-embed-v1")
        r1 = await mock_provider.embed(req)
        mock_provider.reset()
        r2 = await mock_provider.embed(req)
        assert r1.embeddings[0] == r2.embeddings[0]


class TestProviderRegistry:
    """Test provider registry operations."""

    def test_register_and_get(self):
        registry = ProviderRegistry()
        adapter = MockProviderAdapter()
        registry.register(adapter)
        assert registry.get("mock") is adapter

    def test_get_returns_none_for_unknown(self):
        registry = ProviderRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_raise_raises_for_unknown(self):
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="not registered"):
            registry.get_or_raise("nonexistent")

    def test_list_providers(self):
        registry = ProviderRegistry()
        registry.register(MockProviderAdapter())
        providers = registry.list_providers()
        assert "mock" in providers

    def test_contains(self):
        registry = ProviderRegistry()
        registry.register(MockProviderAdapter())
        assert "mock" in registry
        assert "openai" not in registry

    def test_len(self):
        registry = ProviderRegistry()
        assert len(registry) == 0
        registry.register(MockProviderAdapter())
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        registry = ProviderRegistry()
        registry.register(MockProviderAdapter())
        results = await registry.health_check_all()
        assert results["mock"] is True

    @pytest.mark.asyncio
    async def test_list_models(self, mock_provider):
        models = await mock_provider.list_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)
        assert any(m.model_id == "mock-gpt-v1" for m in models)

    @pytest.mark.asyncio
    async def test_health_check(self, mock_provider):
        healthy = await mock_provider.health_check()
        assert healthy is True
