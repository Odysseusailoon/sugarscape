"""vLLM LLM provider for RedBlackBench.

Provides integration with vLLM inference servers through OpenAI-compatible API.
"""

import asyncio
from typing import List, Optional, Any

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
        self._owns_client = True  # Track if we created the client

    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "vllm"

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs: Any,
    ) -> str:
        """Generate a response from vLLM server.

        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Optional extras:
                - max_tokens: override configured max tokens for this call
                - chat_template_kwargs: vLLM chat template kwargs (e.g., {"enable_thinking": False} for Qwen3)

        Returns:
            The generated response text
        """
        max_tokens_override = kwargs.get("max_tokens", None)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", None)

        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        # Estimate input tokens and adjust max_tokens to stay within context limit
        total_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
        estimated_input_tokens = total_chars // 3  # ~3 chars per token for English

        context_limit = 32768  # Qwen3-14B supports 32k context
        available_tokens = context_limit - estimated_input_tokens - 100  # buffer
        configured_max = int(self.config.max_tokens)
        if max_tokens_override is not None:
            try:
                configured_max = int(max_tokens_override)
            except Exception:
                configured_max = int(self.config.max_tokens)

        effective_max_tokens = min(configured_max, max(256, available_tokens))

        # Lazily create shared client for connection reuse and proper batching
        # This enables vLLM's continuous batching when multiple requests come in
        if self._client is None:
            self._client = self._AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.base_url,
            )

        # Build extra_body for vLLM-specific parameters
        extra_body = {}
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=effective_max_tokens,
            extra_body=extra_body if extra_body else None,
        )

        return response.choices[0].message.content or ""

    async def aclose(self) -> None:
        """Async close the underlying HTTP client.

        Call this when done with the provider to properly clean up connections.
        """
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def close(self) -> None:
        """Synchronous close - schedules async cleanup.

        For use when an event loop is available but we're in sync context.
        """
        if self._client is not None and self._owns_client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup for later
                    loop.create_task(self.aclose())
                elif not loop.is_closed():
                    loop.run_until_complete(self.aclose())
                else:
                    # Loop is closed, force cleanup
                    self._client = None
            except RuntimeError:
                # No event loop available, just clear the reference
                self._client = None


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
