import asyncio
import os
import sys
from pathlib import Path
import yaml

MODEL_MAPPING = {
    "GPT-5 Thinking": "openai/gpt-5.1",
    "Gemini 3 Pro": "google/gemini-3-pro-preview",
    "Kimi K2 Thinking": "moonshotai/kimi-k2-thinking",
    "Qwen 3 (235B)": "qwen/qwen3-vl-235b-a22b-thinking",
}

NUM_ROUNDS = 10
TEAM_SIZE = 5
TEMPERATURE = 1.0
INCLUDE_REASONING = True
OUTPUT_ROOT = Path("results/wolf_sheep_experiment")
CONFIG_ROOT = Path("experiments/configs/wolf_sheep_openrouter")
MAX_PARALLEL = 8


def ensure_dirs():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "-")


def build_config(a_display: str, a_id: str, b_display: str, b_id: str) -> dict:
    pair_slug = f"{safe_name(a_display)}-vs-{safe_name(b_display)}"
    return {
        "experiment_name": f"wolf_sheep_{pair_slug}_{TEAM_SIZE}v{TEAM_SIZE}",
        "output_dir": str(OUTPUT_ROOT / pair_slug / f"{TEAM_SIZE}v{TEAM_SIZE}"),
        "num_games": 1,
        "game": {
            "num_rounds": NUM_ROUNDS,
            "team_size": TEAM_SIZE,
        },
        "team_a": {
            "size": TEAM_SIZE,
            "provider": {
                "type": "openrouter",
                "model": a_id,
                "temperature": TEMPERATURE,
                "include_reasoning": INCLUDE_REASONING,
            },
        },
        "team_b": {
            "size": TEAM_SIZE,
            "provider": {
                "type": "openrouter",
                "model": b_id,
                "temperature": TEMPERATURE,
                "include_reasoning": INCLUDE_REASONING,
            },
        },
    }


def write_config(config: dict, a_display: str, b_display: str) -> Path:
    ensure_dirs()
    cfg_path = CONFIG_ROOT / f"{safe_name(a_display)}-vs-{safe_name(b_display)}_{TEAM_SIZE}v{TEAM_SIZE}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return cfg_path


async def run_provider_check(model_id: str) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "redblackbench.cli",
        "provider-check",
        "--provider",
        "openrouter",
        "--model",
        model_id,
        "--max-tokens",
        "64",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "OPENROUTER_API_KEY": api_key or os.environ.get("OPENROUTER_API_KEY", "")},
    )
    out, err = await proc.communicate()
    if out:
        print(out.decode().strip())
    if err:
        print(err.decode().strip())


async def run_experiment_config(cfg_path: Path, no_trajectory: bool = False) -> int:
    args = [
        sys.executable,
        "-m",
        "redblackbench.cli",
        "run",
        "--config",
        str(cfg_path),
    ]
    if no_trajectory:
        args.append("--no-trajectory")
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")},
    )
    out, err = await proc.communicate()
    if out:
        print(out.decode().strip())
    if err:
        print(err.decode().strip())
    return proc.returncode


async def main():
    ensure_dirs()

    unique_models = set(MODEL_MAPPING.values())
    for mid in unique_models:
        await run_provider_check(mid)

    cfg_paths = []
    items = list(MODEL_MAPPING.items())
    for i, (ad, aid) in enumerate(items):
        for j, (bd, bid) in enumerate(items):
            if i == j:
                continue
            cfg = build_config(ad, aid, bd, bid)
            cfg_paths.append(write_config(cfg, ad, bd))

    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def runner(p: Path):
        async with sem:
            print(f"Launching: {p}")
            return await run_experiment_config(p)

    results = await asyncio.gather(*[asyncio.create_task(runner(p)) for p in cfg_paths])
    failed = sum(1 for r in results if r != 0)
    print(f"Completed {len(results)} runs; failures: {failed}")


if __name__ == "__main__":
    asyncio.run(main())

