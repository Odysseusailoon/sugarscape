#!/usr/bin/env python3
"""Test script for AIHubMix provider."""

import asyncio
from redblackbench.providers import AIHubMixProvider

async def test_aihubmix():
    """Test the AIHubMix provider with kimi-k2-thinking model."""

    # Initialize provider with the user's API key
    provider = AIHubMixProvider(
        model="kimi-k2-thinking",
        api_key="sk-g3l2PGisdlitwLcB19032e10DfEd44Bc86B2031266CcDaB1",
        temperature=0.7,
        max_tokens=256
    )

    print(f"Testing AIHubMix provider with model: {provider.config.model}")
    print(f"Provider name: {provider.provider_name}")

    # Test basic text generation
    system_prompt = "You are a helpful AI assistant."
    messages = [
        {"role": "user", "content": "Hello, how are you today?"}
    ]

    try:
        print("\nGenerating response...")
        response = await provider.generate(system_prompt, messages)
        print("Response received successfully!")
        print(f"Response: {response[:200]}...")
        return True

    except Exception as e:
        print(f"Error during generation: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_aihubmix())
    if success:
        print("\n✅ AIHubMix provider test passed!")
    else:
        print("\n❌ AIHubMix provider test failed!")
