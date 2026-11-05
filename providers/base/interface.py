"""
AI Provider abstraction layer.
Defines the standard interface all provider adapters must implement.
Platform independence is enforced here: the evaluation engine never imports
a specific provider's SDK — only the adapter.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class ProviderCapability(str, enum.Enum):
    """Capabilities a provider adapter may declare support for."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    VISION = "vision"
    AUDIO = "audio"
    STRUCTURED_OUTPUT = "structured_output"
    BATCH_INFERENCE = "batch_inference"
    FINE_TUNING = "fine_tuning"
    LOGPROBS = "logprobs"


@dataclass
class ModelInfo:
    """Information about a model available from a provider."""
    provider_name: str
    model_id: str
    model_name: str
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: list[ProviderCapability] = field(default_factory=list)
    input_cost_per_1k_tokens: Optional[float] = None
    output_cost_per_1k_tokens: Optional[float] = None
    is_deprecated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionRequest:
    """
    Provider-agnostic completion request.
    Adapters translate this to their provider's native API format.
    """
    messages: list[dict[str, str]]  # [{role: "user", content: "..."}, ...]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0  # Default to deterministic for evaluation
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    seed: Optional[int] = None  # For reproducibility
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """
    Provider-agnostic completion response.
    Adapters translate their native response to this format.
    """
    content: str
    model: str
    provider_name: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    raw_response: Optional[dict[str, Any]] = None  # Original provider response
    request_id: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """Provider-agnostic embedding request."""
    texts: list[str]
    model: str
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Provider-agnostic embedding response."""
    embeddings: list[list[float]]
    model: str
    provider_name: str
    input_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class ProviderError(Exception):
    """Base class for provider-related errors."""

    def __init__(self, message: str, provider: str, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class ProviderAuthError(ProviderError):
    """Authentication / authorisation failure."""


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded. Caller should retry with backoff."""

    def __init__(
        self, message: str, provider: str, retry_after_seconds: Optional[float] = None
    ) -> None:
        super().__init__(message, provider)
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeoutError(ProviderError):
    """Request timed out."""


class ProviderModelNotFoundError(ProviderError):
    """Requested model is not available on this provider."""


class BaseProviderAdapter(abc.ABC):
    """
    Abstract base class for all AI provider adapters.
    Every adapter must implement this contract.
    The evaluation engine depends only on this interface.
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider (e.g. 'openai', 'mock')."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> list[ProviderCapability]:
        """Capabilities this adapter supports."""

    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Send a completion request and return the response.
        Must raise ProviderError subclasses on failure — never raw provider exceptions.
        """

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return available models for this provider."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings. Only implement if EMBEDDINGS is in capabilities.
        Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Provider {self.provider_name!r} does not support embeddings."
        )

    async def health_check(self) -> bool:
        """
        Check if the provider is reachable and responding.
        Returns True if healthy, False otherwise.
        Default implementation: try listing models.
        """
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    def has_capability(self, cap: ProviderCapability) -> bool:
        """Return True if this adapter supports the given capability."""
        return cap in self.capabilities

    def __repr__(self) -> str:
        caps = [c.value for c in self.capabilities]
        return f"<{self.__class__.__name__} provider={self.provider_name!r} caps={caps}>"


class ProviderRegistry:
    """
    Registry of available provider adapters.
    Evaluation engine uses this to discover and select providers
    without directly importing any adapter implementation.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseProviderAdapter] = {}

    def register(self, adapter: BaseProviderAdapter) -> None:
        """Register a provider adapter."""
        self._adapters[adapter.provider_name] = adapter

    def get(self, provider_name: str) -> Optional[BaseProviderAdapter]:
        """Return adapter by name, or None if not registered."""
        return self._adapters.get(provider_name)

    def get_or_raise(self, provider_name: str) -> BaseProviderAdapter:
        """Return adapter by name, raising ValueError if not registered."""
        adapter = self._adapters.get(provider_name)
        if adapter is None:
            available = list(self._adapters.keys())
            raise ValueError(
                f"Provider {provider_name!r} not registered. "
                f"Available providers: {available}"
            )
        return adapter

    def list_providers(self) -> list[str]:
        """Return names of all registered providers."""
        return list(self._adapters.keys())

    def list_capabilities(self) -> dict[str, list[str]]:
        """Return capability map for all registered providers."""
        return {
            name: [c.value for c in adapter.capabilities]
            for name, adapter in self._adapters.items()
        }

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all registered providers."""
        results: dict[str, bool] = {}
        for name, adapter in self._adapters.items():
            results[name] = await adapter.health_check()
        return results

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, provider_name: str) -> bool:
        return provider_name in self._adapters


# Module-level registry singleton
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the global provider registry, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry. For testing only."""
    global _registry
    _registry = None
