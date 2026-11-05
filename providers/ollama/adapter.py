"""
Ollama local provider adapter.
Connects to a locally running Ollama instance (default: http://localhost:11434).
No API key required — demonstrates self-hosted / air-gapped deployment.

Requires: Ollama running locally with at least one model pulled.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from aieval360.providers.base.interface import (
    BaseProviderAdapter,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    ProviderCapability,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderTimeoutError,
)


class OllamaAdapter(BaseProviderAdapter):
    """
    Adapter for Ollama local inference.
    Demonstrates platform-independence: same evaluation engine, local execution.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        default_model: str = "llama3.2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model
        self._http = httpx.AsyncClient(timeout=timeout)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.EMBEDDINGS,
        ]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or self.default_model
        payload = {
            "model": model,
            "messages": request.messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.system_prompt:
            payload["system"] = request.system_prompt

        start = time.monotonic()
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), "ollama", "TIMEOUT") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ProviderModelNotFoundError(
                    f"Model {model!r} not found in Ollama. Run: ollama pull {model}",
                    "ollama",
                    "MODEL_NOT_FOUND",
                ) from exc
            raise ProviderError(
                f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}",
                "ollama",
                str(exc.response.status_code),
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?",
                "ollama",
                "CONNECT_ERROR",
            ) from exc

        latency_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return CompletionResponse(
            content=content,
            model=data.get("model", model),
            provider_name="ollama",
            finish_reason=data.get("done_reason", "stop"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=0.0,  # Local inference has no API cost
            raw_response=data,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Ollama's /api/embed endpoint."""
        model = request.model or "nomic-embed-text"
        try:
            resp = await self._http.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": request.texts},
            )
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}",
                "ollama",
                "CONNECT_ERROR",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama embedding error: {exc.response.text}",
                "ollama",
            ) from exc

        data = resp.json()
        embeddings = data.get("embeddings", [])
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            provider_name="ollama",
        )

    async def list_models(self) -> list[ModelInfo]:
        """List models currently available in this Ollama instance."""
        try:
            resp = await self._http.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                models.append(
                    ModelInfo(
                        provider_name="ollama",
                        model_id=name,
                        model_name=name,
                        capabilities=[
                            ProviderCapability.CHAT_COMPLETION,
                            ProviderCapability.TEXT_GENERATION,
                        ],
                        metadata={
                            "size": m.get("size"),
                            "digest": m.get("digest"),
                            "modified_at": m.get("modified_at"),
                        },
                    )
                )
            return models
        except (httpx.ConnectError, httpx.HTTPStatusError):
            return []  # Ollama not running — not an error, just unavailable

    async def health_check(self) -> bool:
        try:
            resp = await self._http.get(f"{self.base_url}/", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
