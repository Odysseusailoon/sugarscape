#!/usr/bin/env python3
"""Test script for OpenRouter models with thinking/reasoning enabled.

Runs minimal reasoning checks against a set of model slugs using the
OpenAI-compatible AsyncOpenAI client, requesting `include_reasoning`.

Reads API key from environment variable `OPENROUTER_API_KEY` or CLI arg.
"""

import os
import sys
import asyncio
from typing import Dict

from openai import AsyncOpenAI


MODELS: Dict[str, str] = {
    "qwen": "qwen/qwen3-vl-235b-a22b-thinking",
    "openai": "openai/gpt-5.1",
    "kimi": "moonshotai/kimi-k2-thinking",
    "gemini": "google/gemini-3-pro-preview",
}


async def test_model(client: AsyncOpenAI, name: str, model_id: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Testing {name}: {model_id}")
    print(f"{'='*60}")

    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Think step by step."},
                {"role": "user", "content": "What is 17 * 23? Show your reasoning."},
            ],
            max_tokens=256,
            temperature=0.2,
            extra_body={"include_reasoning": True},
        )
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None)

        print("✓ SUCCESS")
        if reasoning:
            print(f"\n[THINKING] ({len(reasoning)} chars):")
            print(reasoning[:800] + "..." if len(reasoning) > 800 else reasoning)
        else:
            print("\n[THINKING]: Not available for this model")

        print("\n[RESPONSE]:")
        print(content[:500] if len(content) > 500 else content)
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


async def main() -> None:
    print("=" * 60)
    print("OpenRouter Model Test Script (with Thinking/Reasoning)")
    print("Testing: Qwen, GPT-5.1, Kimi, and Gemini 3 Pro")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        if len(sys.argv) >= 2:
            api_key = sys.argv[1]
        else:
            print("Error: OPENROUTER_API_KEY not set and no API key provided via argv")
            sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/redblackbench/redblackbench",
            "X-Title": "RedBlackBench Test",
        },
    )

    results: Dict[str, bool] = {}
    for name, model_id in MODELS.items():
        results[name] = await test_model(client, name, model_id)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())

