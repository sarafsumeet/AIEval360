"""
Mock AI Provider — always available, no credentials required.
Used for all automated testing and development without real API access.

Behaviour is deterministic and configurable:
- Returns predictable responses based on input patterns
- Simulates latency (configurable, defaults to 0 for tests)
- Tracks all calls for assertion in tests
- Supports failure injection for error-path testing
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aieval360.providers.base.interface import (
    BaseProviderAdapter,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ProviderCapability,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


@dataclass
class MockCall:
    """Record of a single call made to the mock provider."""
    method: str
    request: Any
    response: Any
    timestamp: float = field(default_factory=time.time)


class MockProviderAdapter(BaseProviderAdapter):
    """
    Deterministic mock provider for testing and local development.
    All responses are synthetic. No external calls are made.
    """

    MOCK_MODELS = [
        ModelInfo(
            provider_name="mock",
            model_id="mock-gpt-v1",
            model_name="Mock GPT v1",
            context_window=128000,
            max_output_tokens=4096,
            capabilities=[
                ProviderCapability.CHAT_COMPLETION,
                ProviderCapability.TEXT_GENERATION,
                ProviderCapability.FUNCTION_CALLING,
                ProviderCapability.STRUCTURED_OUTPUT,
            ],
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.002,
        ),
        ModelInfo(
            provider_name="mock",
            model_id="mock-embed-v1",
            model_name="Mock Embeddings v1",
            context_window=8192,
            capabilities=[ProviderCapability.EMBEDDINGS],
            input_cost_per_1k_tokens=0.0001,
        ),
        ModelInfo(
            provider_name="mock",
            model_id="mock-judge-v1",
            model_name="Mock Judge v1 (LLM-as-Judge)",
            context_window=128000,
            max_output_tokens=1024,
            capabilities=[
                ProviderCapability.CHAT_COMPLETION,
                ProviderCapability.STRUCTURED_OUTPUT,
            ],
        ),
    ]

    def __init__(
        self,
        simulated_latency_ms: float = 0.0,
        fail_on_calls: Optional[list[int]] = None,
        fail_with: Optional[Exception] = None,
        fixed_response: Optional[str] = None,
    ) -> None:
        """
        Args:
            simulated_latency_ms: Artificial delay per call (ms). 0 = instant.
            fail_on_calls: Call indices (0-indexed) on which to raise an error.
            fail_with: Exception to raise on failure. Defaults to ProviderError.
            fixed_response: If set, return this content for all completions.
        """
        self.simulated_latency_ms = simulated_latency_ms
        self.fail_on_calls = set(fail_on_calls or [])
        self.fail_with = fail_with
        self.fixed_response = fixed_response
        self._call_count = 0
        self.call_log: list[MockCall] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.STRUCTURED_OUTPUT,
        ]

    async def _simulate_latency(self) -> None:
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000)

    def _should_fail(self) -> bool:
        return self._call_count in self.fail_on_calls

    def _generate_content(self, request: CompletionRequest) -> str:
        """Generate a deterministic response based on the last user message."""
        if self.fixed_response is not None:
            return self.fixed_response

        last_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break

        # Deterministic response patterns for common evaluation scenarios
        content_lower = last_message.lower()

        # LLM-as-judge responses
        if "evaluate" in content_lower and "score" in content_lower:
            return '{"score": 0.85, "reasoning": "The response is relevant and accurate.", "passed": true}'

        if "summarise" in content_lower or "summarize" in content_lower:
            return "This is a mock summary of the provided text. The key points include the main topic and supporting details."

        if "classify" in content_lower or "category" in content_lower:
            return '{"category": "positive", "confidence": 0.92}'

        if "translate" in content_lower:
            return "Mock translation of the provided text."

        if "question" in content_lower or "?" in last_message:
            return "This is a mock answer to your question. In a real evaluation, the LLM would generate a contextually appropriate response based on the input."

        # Default: stable hash-based response (deterministic for same input)
        input_hash = hashlib.md5(last_message.encode()).hexdigest()[:8]
        return (
            f"Mock response [{input_hash}]: This is a synthetic output generated by the "
            f"AIEval360 mock provider for testing and development. "
            f"No real AI inference was performed."
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        call_index = self._call_count
        self._call_count += 1

        if call_index in self.fail_on_calls:
            exc = self.fail_with or ProviderError(
                f"Mock failure injected at call {call_index}", "mock", "MOCK_FAILURE"
            )
            raise exc

        await self._simulate_latency()

        start = time.monotonic()
        content = self._generate_content(request)
        latency_ms = (time.monotonic() - start) * 1000 + self.simulated_latency_ms

        input_tokens = sum(len(m.get("content", "")) // 4 for m in request.messages)
        output_tokens = len(content) // 4

        response = CompletionResponse(
            content=content,
            model=request.model,
            provider_name="mock",
            finish_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=(input_tokens * 0.000001 + output_tokens * 0.000002),
            raw_response={"mock": True, "call_index": call_index},
            request_id=f"mock-{call_index:06d}",
        )

        self.call_log.append(MockCall("complete", request, response))
        return response

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        call_index = self._call_count
        self._call_count += 1

        if self._should_fail():
            exc = self.fail_with or ProviderError(
                f"Mock embedding failure at call {call_index}", "mock"
            )
            raise exc

        await self._simulate_latency()

        # Generate deterministic pseudo-embeddings (dimension=1536 like text-embedding-ada-002)
        embeddings = []
        for text in request.texts:
            # Hash-seeded deterministic float vector
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            vec = []
            for i in range(1536):
                # Simple LCG for reproducible floats
                seed = (seed * 1664525 + 1013904223) % (2**32)
                vec.append((seed / (2**32)) * 2 - 1)
            # Normalise
            magnitude = sum(x**2 for x in vec) ** 0.5
            vec = [x / magnitude for x in vec]
            embeddings.append(vec)

        response = EmbeddingResponse(
            embeddings=embeddings,
            model=request.model,
            provider_name="mock",
            input_tokens=sum(len(t) // 4 for t in request.texts),
        )
        self.call_log.append(MockCall("embed", request, response))
        return response

    async def list_models(self) -> list[ModelInfo]:
        return self.MOCK_MODELS

    async def health_check(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset call count and log. Useful between test cases."""
        self._call_count = 0
        self.call_log.clear()

    @property
    def call_count(self) -> int:
        return self._call_count
