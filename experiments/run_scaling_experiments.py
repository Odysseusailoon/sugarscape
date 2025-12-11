import asyncio
import os
import sys
from pathlib import Path
import yaml

# Models and display names
MODEL_MAPPING = {
    "GPT-5 Thinking": "openai/gpt-5.1",
    "Gemini 3 Pro": "google/gemini-3-pro-preview",
    "Kimi K2 Thinking": "moonshotai/kimi-k2-thinking",
    "Qwen 3 (235B)": "qwen/qwen3-vl-235b-a22b-thinking",
}

# Team sizes to test
TEAM_SIZES = [1, 2, 3, 5, 10]

# Experiment parameters
NUM_ROUNDS = 10
TEMPERATURE = 1.0
INCLUDE_REASONING = True
OUTPUT_ROOT = Path("results/scaling_experiment")
CONFIG_ROOT = Path("experiments/configs/scaling_openrouter")
MAX_PARALLEL = 8


def ensure_dirs():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "-")


def build_config(model_display: str, model_id: str, team_size: int) -> dict:
    experiment_slug = f"{safe_name(model_display)}_{team_size}v{team_size}"
    return {
        "experiment_name": f"wisdom_crowds_{experiment_slug}",
        "output_dir": str(OUTPUT_ROOT / safe_name(model_display) / f"{team_size}v{team_size}"),
        "num_games": 1,
        "game": {
            "num_rounds": NUM_ROUNDS,
            "team_size": team_size,
        },
        "team_a": {
            "size": team_size,
            "provider": {
                "type": "openrouter",
                "model": model_id,
                "temperature": TEMPERATURE,
                "include_reasoning": INCLUDE_REASONING,
            },
        },
        "team_b": {
            "size": team_size,
            "provider": {
                "type": "openrouter",
                "model": model_id,
                "temperature": TEMPERATURE,
                "include_reasoning": INCLUDE_REASONING,
            },
        },
    }


def write_config(config: dict, model_display: str, team_size: int) -> Path:
    ensure_dirs()
    cfg_path = CONFIG_ROOT / f"{safe_name(model_display)}_{team_size}v{team_size}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return cfg_path


async def run_provider_check(model_id: str) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set; provider check may fail.")
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
        "--resume",
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
    print(out.decode().strip())
    if err:
        print(err.decode().strip())
    return proc.returncode


async def main():
    ensure_dirs()

    # Pre-flight checks for each model
    print("=== Provider Checks ===")
    for display, mid in MODEL_MAPPING.items():
        print(f"Checking: {display} -> {mid}")
        await run_provider_check(mid)

    # Build all configs
    cfg_paths = []
    for display, mid in MODEL_MAPPING.items():
        for sz in TEAM_SIZES:
            cfg = build_config(display, mid, sz)
            cfg_path = write_config(cfg, display, sz)
            cfg_paths.append(cfg_path)

    # Run all experiments in parallel (bounded)
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def runner(p: Path):
        async with sem:
            print(f"Launching: {p}")
            return await run_experiment_config(p)

    tasks = [asyncio.create_task(runner(p)) for p in cfg_paths]
    results = await asyncio.gather(*tasks)
    failed = sum(1 for r in results if r != 0)
    print(f"\nCompleted {len(results)} runs; failures: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
