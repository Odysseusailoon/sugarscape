"""OpenRouter LLM provider for RedBlackBench."""

import os
from typing import Optional, List

from redblackbench.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter API provider.
    
    Uses OpenAI-compatible API to access models via OpenRouter.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        include_reasoning: bool = True,
    ):
        """Initialize the OpenRouter provider.
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-5')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: OpenRouter API key
            include_reasoning: Whether to request reasoning/thinking tokens
        """
        # Use OpenRouter base URL
        base_url = "https://openrouter.ai/api/v1"
        
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max(16, max_tokens),
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )
        
        self.include_reasoning = include_reasoning
        
        # Override client with custom base URL and headers
        from openai import AsyncOpenAI
        
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/redblackbench/redblackbench",
                "X-Title": "RedBlackBench",
            }
        )
    
    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "openrouter"

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from OpenRouter, handling reasoning tokens."""
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        
        # Prepare extra parameters for OpenRouter
        extra_body = {}
        if self.include_reasoning:
            extra_body["include_reasoning"] = True

        # Make API call
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=extra_body if extra_body else None
        )
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Check for reasoning field (specific to OpenRouter/DeepSeek/Thinking models)
        reasoning = getattr(choice.message, 'reasoning', None)
        
        # If reasoning exists, prepend it with special delimiters so LLMAgent can split it
        if reasoning:
            content = f"__THINKING_START__\n{reasoning}\n__THINKING_END__\n\n{content}"
            
        return content
