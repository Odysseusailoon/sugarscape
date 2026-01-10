"""vLLM LLM provider for RedBlackBench.

Provides integration with vLLM inference servers through OpenAI-compatible API.
"""

import os
from typing import List, Optional

from redblackbench.providers.base import BaseLLMProvider, ProviderConfig


class VLLMProvider(BaseLLMProvider):
    """vLLM API provider for local model serving.

    Uses vLLM's OpenAI-compatible API to serve models locally with high throughput.
    Supports features like PagedAttention, continuous batching, and prefix caching.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        """Initialize the vLLM provider.

        Args:
            model: Model identifier (model name or path on the vLLM server)
            base_url: Base URL for vLLM server (default: 'http://localhost:8000/v1')
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 1024)
            api_key: API key (vLLM doesn't require one, defaults to 'EMPTY')
        """
        config = ProviderConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or "EMPTY",
        )
        super().__init__(config)

        self.base_url = base_url

        # Import here to avoid requiring openai if not used
        try:
            from openai import AsyncOpenAI
            self._AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for VLLMProvider. "
                "Install it with: pip install openai"
            )

        # Shared client for connection reuse and proper batching
        # Lazily initialized on first generate() call
        self._client = None

    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "vllm"

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from vLLM server.

        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            The generated response text
        """
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        # Estimate input tokens and adjust max_tokens to stay within 4k context
        total_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
        estimated_input_tokens = total_chars // 3  # ~3 chars per token for English

        context_limit = 4096
        available_tokens = context_limit - estimated_input_tokens - 100  # buffer
        effective_max_tokens = min(self.config.max_tokens, max(256, available_tokens))

        # Lazily create shared client for connection reuse and proper batching
        # This enables vLLM's continuous batching when multiple requests come in
        if self._client is None:
            self._client = self._AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.base_url,
            )

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=effective_max_tokens,
        )

        return response.choices[0].message.content or ""


# Convenience factory functions
def create_vllm_qwen3_14b_provider(
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.7,
    model_path: str = "/workspace/models/Qwen3-14B",
) -> VLLMProvider:
    """Create a vLLM provider for Qwen3-14B base model.

    Args:
        base_url: vLLM server base URL
        temperature: Sampling temperature
        model_path: Model identifier (path or name on vLLM server)

    Returns:
        Configured vLLM provider
    """
    return VLLMProvider(
        model=model_path,
        base_url=base_url,
        temperature=temperature,
    )


def create_vllm_lora_provider(
    lora_adapter: str,
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.7,
) -> VLLMProvider:
    """Create a vLLM provider for LoRA adapter.

    Args:
        lora_adapter: LoRA adapter name (e.g., 'qwen3-14b-v2')
        base_url: vLLM server base URL
        temperature: Sampling temperature

    Returns:
        Configured vLLM provider
    """
    return VLLMProvider(
        model=lora_adapter,
        base_url=base_url,
        temperature=temperature,
    )
