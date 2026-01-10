# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RedBlackBench is a multi-agent game theory benchmark for evaluating LLM cooperative alignment under competitive pressure. It implements the Red-Black Game where two teams of LLM agents choose between RED (defection) and BLACK (cooperation) over multiple rounds.

**Core dilemma**: Individual rationality suggests RED is always better, but mutual cooperation (both BLACK) yields +6 total while mutual defection (both RED) yields -6 total.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with all LLM providers (openai, anthropic)
pip install -e ".[all]"

# Install with dev tools (pytest, black, isort, mypy)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_game.py

# Run tests with coverage
pytest --cov=redblackbench tests/

# Code formatting
black redblackbench/
isort redblackbench/

# Type checking
mypy redblackbench/
```

## Running Experiments

```bash
# Run with config file
redblackbench run --config experiments/configs/example.yaml

# Quick test with defaults
redblackbench run --quick-test

# Analyze results
redblackbench analyze --results-dir results/

# Check provider connectivity
redblackbench provider-check --provider openrouter --model openai/gpt-4o
```

## Architecture

### Core Components

- **`game/coordinator.py`**: Main game orchestrator - runs rounds, calculates scores, manages game state
- **`game/scoring.py`**: Scoring matrix implementation (Choice enum, RoundResult, ScoringMatrix)
- **`teams/team.py`**: Team container with deliberation capability
- **`teams/deliberation.py`**: Two-phase decision process (opinions → votes → majority)
- **`agents/llm_agent.py`**: LLM-powered agent with opinion/vote generation
- **`agents/prompts.py`**: Prompt templates for agents

### Provider System

All providers implement `async generate(system_prompt, messages) -> str`:
- `providers/openai_provider.py` - OpenAI GPT models
- `providers/anthropic_provider.py` - Anthropic Claude models
- `providers/openrouter_provider.py` - OpenRouter multi-model access
- `providers/vllm_provider.py` - Local vLLM inference (Qwen3-14B)

### Sugarscape Module

Extended multi-agent economic simulation in `sugarscape/` for studying agent decision-making, trading, and welfare dynamics.

**Simulation Loop (per tick):**
1. Environment growback (resources regenerate)
2. Standard agents move sequentially (persona-based scoring)
3. LLM agents decide in parallel (async batched), then apply moves
4. Trading phase (if enabled)
5. Metabolism + aging + death (constant population - dead agents replaced)

**Key Classes:**

- **`SugarAgent`** (`agent.py`): Base agent with vision, metabolism, wealth, spice, persona, trade memory, trust scores
- **`LLMSugarAgent`** (`llm_agent.py`): Extends SugarAgent with LLM provider, goal prompt, conversation history. Uses `async_decide_move()` for batched inference
- **`SugarEnvironment`** (`environment.py`): 50x50 torus grid with sugar/spice peaks at (15,15)/(35,35). Handles growback, harvest, agent tracking
- **`SugarSimulation`** (`simulation.py`): Main controller orchestrating tick loop, snapshot saving, metrics logging
- **`SugarscapeConfig`** (`config.py`): Comprehensive dataclass with ~30 parameters

**Checkpoint System** (`simulation.py`):
- `save_checkpoint()` - Saves complete simulation state to pickle file
- `load_checkpoint(path)` - Class method to restore simulation from checkpoint
- `run_with_checkpoints(steps, checkpoint_interval)` - Run with periodic saves
- **Full state preserved**: RNG state, agent conversation history, trade memory, visited cells, partner trust, global reputation
- Checkpoints stored in `results/sugarscape/{experiment}/checkpoints/checkpoint_tick_N.pkl`

**Snapshot System** (JSON for analysis):
- `save_snapshot()` - Lightweight JSON snapshots at simulation start/end
- Captures basic state for analysis (positions, wealth, spice, grid state)

**Persona System (A/B/C/D):**
- **A (Conservative)**: Prioritizes safety, avoids long moves
- **B (Foresight)**: Balances immediate utility with long-term site quality
- **C (Nomad)**: Seeks novelty and unexplored regions
- **D (Risk-taker)**: Aggressive utility maximization

**Trading System** (`trade.py`):
- **MRS mode**: Marginal Rate of Substitution bargaining. Price = geometric mean of both agents' MRS
- **Dialogue mode**: LLM-based negotiation with CHAT/OFFER/ACCEPT/REJECT/WALK_AWAY actions. Supports optional fraud (public offer vs private execution). Trust updated per trade

**Goal Presets for LLM Agents:**
- `none`: No explicit goal (baseline) - "Observe your situation and decide what to do"
- `survival`: Stay alive, secondarily accumulate food
- `wealth`: Maximize total resources
- `altruist`: Help others, share resources (aliases: `egalitarian`, `utilitarian`, `samaritan`)

**Mixed Goals** - Different LLM agents can have different goals in the same simulation:
```python
config = SugarscapeConfig(
    enable_llm_agents=True,
    enable_mixed_goals=True,  # Enable goal distribution
    llm_goal_distribution={
        "survival": 0.4,   # 40% survival-focused
        "wealth": 0.3,     # 30% wealth-focused
        "altruist": 0.2,   # 20% altruistic
        "none": 0.1,       # 10% no explicit goal
    }
)
```

**Welfare Metrics** (`welfare.py`):
- Individual: Cobb-Douglas utility `wealth^(m_s/m_total) * spice^(m_p/m_total)`
- Social: Utilitarian (sum), Rawlsian (min), Nash (geometric mean), Gini-adjusted, Atkinson-adjusted

**Trajectory System** (`trajectory.py`): Captures full LLM prompts/responses per tick for RL post-training

**Running Sugarscape:**
```bash
python scripts/run_sugarscape.py
python scripts/run_goal_experiment.py      # Compare goal presets
python scripts/run_persona_experiment.py   # Compare personas
```

**Output Structure:**
```
results/sugarscape/{experiment}/
├── config.json
├── metrics.csv              # Time series (every 10 ticks)
├── trajectory_*.json        # LLM interactions for RL
├── initial_state.json       # Snapshot at tick 0
├── final_state.json         # Snapshot at simulation end
├── checkpoints/             # Full state checkpoints (pickle)
│   ├── checkpoint_tick_50.pkl
│   └── checkpoint_tick_100.pkl
├── plots/                   # Welfare visualizations
└── debug/                   # Optional detailed logs
```

**Checkpoint/Resume Usage:**
```python
from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig

# Run with periodic checkpoints
config = SugarscapeConfig(enable_spice=True, enable_trade=True)
sim = SugarSimulation(config=config, experiment_name='my_experiment')
sim.run_with_checkpoints(steps=1000, checkpoint_interval=100)

# Resume from checkpoint
sim = SugarSimulation.load_checkpoint('path/to/checkpoint_tick_500.pkl')
sim.run_with_checkpoints(steps=500, checkpoint_interval=100)
```

### Trajectory System

`trajectory/` captures full game state for analysis:
- `GameTrajectory`: Complete game record
- `TrajectoryCollector`: Data collection during play
- Used for replay and analysis

## Configuration

Games are configured via YAML (see `experiments/configs/`):

```yaml
experiment_name: "my_experiment"
num_games: 3

default_provider:
  type: openai  # or anthropic, openrouter, vllm
  model: gpt-4
  temperature: 0.7

game:
  num_rounds: 10
  team_size: 5
  multipliers: {5: 3, 8: 5, 10: 10}  # Bonus rounds
```

## Code Patterns

- All LLM calls are async - use `asyncio.gather()` for parallel operations
- Providers use a common interface via `BaseLLMProvider`
- Dataclasses used for configuration and data structures
- pytest-asyncio with `asyncio_mode = "auto"` for tests

## vLLM Local Inference

For local model inference (see VLLM_INTEGRATION.md):

```bash
# Start vLLM server
bash /workspace/start_qwen3_base_server.sh

# Test provider
python test_vllm_provider.py

# Run with vLLM config
redblackbench run --config experiments/configs/vllm_qwen3_quick_test.yaml
```

## Key Metrics

- **Efficiency**: Score achieved as % of maximum possible (150 for full cooperation)
- **Cooperation Rate**: % of BLACK choices
- **Consensus Rate**: How often teams reached unanimous votes
