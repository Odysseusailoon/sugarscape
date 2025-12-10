#!/usr/bin/env python3
"""Test script for OpenRouter models with thinking/reasoning enabled."""

import asyncio
from openai import AsyncOpenAI

API_KEY = "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"

# Models with thinking/reasoning capabilities
MODELS = {
    "qwen": "qwen/qwen3-vl-235b-a22b-thinking",
    "openai": "openai/gpt-5.1",
    "kimi": "moonshotai/kimi-k2-thinking",
    "gemini": "google/gemini-3-pro-preview",
}

async def test_model(client: AsyncOpenAI, name: str, model_id: str):
    """Test a single model with thinking enabled."""
    print(f"\n{'='*60}")
    print(f"Testing {name}: {model_id}")
    print('='*60)
    
    try:
        # Enable reasoning/thinking via extra parameters
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Think step by step."},
                {"role": "user", "content": "What is 17 * 23? Show your reasoning."}
            ],
            max_tokens=1024,
            temperature=0.7,
            extra_body={
                "include_reasoning": True,  # OpenRouter reasoning parameter
            }
        )
        
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        
        # Get reasoning from message (OpenRouter returns it here for thinking models)
        reasoning = getattr(msg, 'reasoning', None)
        
        print(f"✓ SUCCESS")
        
        if reasoning:
            print(f"\n[THINKING] ({len(reasoning)} chars):")
            print(f"{reasoning[:800]}..." if len(reasoning) > 800 else reasoning)
        else:
            print(f"\n[THINKING]: Not available for this model")
        
        print(f"\n[RESPONSE]:")
        print(content[:500] if len(content) > 500 else content)
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

async def main():
    print("="*60)
    print("OpenRouter Model Test Script (with Thinking/Reasoning)")
    print("Testing: Qwen, GPT-5.1, Kimi, and Gemini 3 Pro")
    print("="*60)
    
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/redblackbench/redblackbench",
            "X-Title": "RedBlackBench Test",
        }
    )
    
    results = {}
    for name, model_id in MODELS.items():
        results[name] = await test_model(client, name, model_id)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
