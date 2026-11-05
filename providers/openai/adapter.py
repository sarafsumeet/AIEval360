"""
OpenAI provider adapter.
Requires: pip install openai
Requires env var: AIEVAL360_PROVIDER_OPENAI_API_KEY

This adapter demonstrates the full provider contract.
Other hosted API adapters (Anthropic, Google, Azure OpenAI) follow the
same pattern — only the client and response-mapping differ.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from aieval360.providers.base.interface import (
    BaseProviderAdapter,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ProviderAuthError,
    ProviderCapability,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

if TYPE_CHECKING:
    import openai


class OpenAIAdapter(BaseProviderAdapter):
    """
    Adapter for OpenAI's API (chat completions + embeddings).
    Skipped automatically in tests if AIEVAL360_PROVIDER_OPENAI_API_KEY is unset.
    """

    # Models known at time of writing — list_models() fetches the live list
    _KNOWN_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
    ]

    _EMBEDDING_MODELS = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    def __init__(
        self,
        api_key: str,
        org_id: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        try:
            import openai as _openai

            self._client = _openai.AsyncOpenAI(
                api_key=api_key,
                organization=org_id,
                timeout=timeout,
                max_retries=max_retries,
            )
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self.default_model = default_model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.STRUCTURED_OUTPUT,
            ProviderCapability.STREAMING,
            ProviderCapability.LOGPROBS,
        ]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        import openai

        messages = list(request.messages)
        if request.system_prompt:
            messages = [{"role": "system", "content": request.system_prompt}] + messages

        params: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            params["stop"] = request.stop
        if request.seed is not None:
            params["seed"] = request.seed
        params.update(request.extra_params)

        start = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(**params)
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(str(exc), "openai", "AUTH_ERROR") from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), "openai") from exc
        except openai.NotFoundError as exc:
            raise ProviderModelNotFoundError(str(exc), "openai", "MODEL_NOT_FOUND") from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), "openai", "TIMEOUT") from exc
        except openai.OpenAIError as exc:
            raise ProviderError(str(exc), "openai", "API_ERROR") from exc

        latency_ms = (time.monotonic() - start) * 1000
        choice = resp.choices[0]
        usage = resp.usage

        # Cost estimation (approximate, based on published pricing at time of writing)
        input_cost = (usage.prompt_tokens / 1000) * self._get_input_cost(params["model"])
        output_cost = (usage.completion_tokens / 1000) * self._get_output_cost(params["model"])

        return CompletionResponse(
            content=choice.message.content or "",
            model=resp.model,
            provider_name="openai",
            finish_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=input_cost + output_cost,
            raw_response=resp.model_dump(),
            request_id=resp.id,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import openai

        try:
            resp = await self._client.embeddings.create(
                input=request.texts,
                model=request.model,
                **request.extra_params,
            )
        except openai.OpenAIError as exc:
            raise ProviderError(str(exc), "openai", "EMBED_ERROR") from exc

        embeddings = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        return EmbeddingResponse(
            embeddings=embeddings,
            model=resp.model,
            provider_name="openai",
            input_tokens=resp.usage.prompt_tokens,
        )

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.models.list()
            models = []
            for m in resp.data:
                cap = [ProviderCapability.TEXT_GENERATION]
                if any(x in m.id for x in ["gpt-4", "gpt-3.5", "o1", "o3"]):
                    cap.append(ProviderCapability.CHAT_COMPLETION)
                if "embedding" in m.id:
                    cap = [ProviderCapability.EMBEDDINGS]
                models.append(
                    ModelInfo(
                        provider_name="openai",
                        model_id=m.id,
                        model_name=m.id,
                        capabilities=cap,
                    )
                )
            return models
        except Exception:
            # Fall back to known model list if API call fails
            return [
                ModelInfo(
                    provider_name="openai",
                    model_id=mid,
                    model_name=mid,
                    capabilities=[
                        ProviderCapability.CHAT_COMPLETION,
                        ProviderCapability.FUNCTION_CALLING,
                    ],
                )
                for mid in self._KNOWN_MODELS
            ]

    @staticmethod
    def _get_input_cost(model: str) -> float:
        """Approximate input cost per 1k tokens (USD)."""
        costs = {
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0005,
        }
        for key, cost in costs.items():
            if key in model:
                return cost
        return 0.001  # Default estimate

    @staticmethod
    def _get_output_cost(model: str) -> float:
        """Approximate output cost per 1k tokens (USD)."""
        costs = {
            "gpt-4o": 0.01,
            "gpt-4o-mini": 0.0006,
            "gpt-4-turbo": 0.03,
            "gpt-4": 0.06,
            "gpt-3.5-turbo": 0.0015,
        }
        for key, cost in costs.items():
            if key in model:
                return cost
        return 0.002
