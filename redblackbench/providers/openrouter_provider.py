"""OpenRouter LLM provider for RedBlackBench."""

import os
import asyncio
from typing import Optional, List, Any

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
        
        # Determine the key to use: passed api_key -> OPENROUTER_API_KEY -> OPENAI_API_KEY -> "dummy"
        # The dummy fallback allows instantiation without a key (for testing/mocking), 
        # though actual calls will fail if not mocked.
        final_api_key = (
            api_key 
            or os.environ.get("OPENROUTER_API_KEY") 
            or os.environ.get("OPENAI_API_KEY") 
            or "sk-dummy-key"
        )

        self._client = AsyncOpenAI(
            api_key=final_api_key,
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

    async def _call_api_with_retry(self, **kwargs) -> Any:
        """Call API with simple retry logic."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Simple exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

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

        # Make API call with retry
        response = await self._call_api_with_retry(
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
