#!/usr/bin/env python3
"""Quick debug script to test OpenRouter API responses."""
import asyncio
import os

# Already sourced from local.env via shell

from redblackbench.providers.openrouter_provider import OpenRouterProvider

async def test():
    # Test with include_reasoning=True to see if we get both reasoning AND content
    provider = OpenRouterProvider(
        model="qwen/qwen3-14b",
        include_reasoning=True,  # Changed to True
    )
    
    prompts = [
        "Say hello in one sentence. /no_think",  # Try no_think flag
        "What is 2+2? Answer: ",  # Direct prompt
        "Greet your neighbor: ",  # Very direct
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n=== Test {i+1} ===")
        print(f"Prompt: {prompt}")
        
        response = await provider.generate(
            system_prompt="You are a helpful assistant. Be brief.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        
        print(f"Raw response len={len(response)}:")
        print(f"  FULL: {repr(response)}")
        
        # Check for thinking blocks
        if "<think>" in response or "__THINKING" in response:
            print("  WARNING: Contains thinking blocks!")

if __name__ == "__main__":
    asyncio.run(test())
