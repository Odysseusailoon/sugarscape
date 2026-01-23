# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.



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

---

## Sugarscape Development Guidelines

### Checkpoint System Compatibility

**CRITICAL**: Every code change to sugarscape must ensure the checkpoint/resume system continues to work correctly.

When modifying:
- **Config fields**: Add to `SugarscapeConfig` and ensure serialization in checkpoint
- **Agent state**: Update both `to_checkpoint_dict()` and `restore_from_checkpoint()` in `agent.py`
- **Environment state**: Update `get_checkpoint_state()` and `restore_checkpoint_state()` in `environment.py`
- **New systems**: Ensure state is captured in checkpoint and properly restored

Test checkpoint compatibility:
```python
# Run with checkpoints
sim.run_with_checkpoints(steps=100, checkpoint_interval=50)

# Resume from checkpoint
sim2 = SugarSimulation.load_checkpoint('path/to/checkpoint_tick_50.pkl')
sim2.run_with_checkpoints(steps=50)
```

### Ablation System

The sugarscape supports ablation studies via config flags:

```python
config = SugarscapeConfig(
    enable_survival_pressure=True,   # When False: no starvation death, prompts focus on welfare maximization
    social_memory_visible=True,      # When False: hide trust, reputation, trade history from prompts
)
```

These ablation flags are orthogonal and can be combined for 2x2 experimental designs.

---

## Recent Changelog (2026-01-22)

### Added T=0 Baseline Belief Capture

**Purpose:** Prove that value changes emerge from interactions, not pre-existing in LLM. Addresses reviewer concern: "How do you know these LLMs weren't already like this?"

**Changes:**

1. **`redblackbench/sugarscape/agent.py`**
   - Added `baseline_snapshot: Dict[str, Any]` field to `SugarAgent`
   - Added `capture_baseline(tick)` method to snapshot initial beliefs
   - Updated `to_checkpoint_dict()` and `restore_from_checkpoint()` to include baseline

2. **`redblackbench/sugarscape/simulation.py`**
   - Added `_capture_baselines()` method to capture all agents' baseline at T=0
   - Modified `run()` to call `_capture_baselines()` + identity_review at T=0 before any interactions
   - Saves baseline to `baseline_beliefs.json` in experiment output

**Usage:**
- Baseline is automatically captured before any step() is called
- Compare `baseline_beliefs.json` (T=0) with later `identity_review_history` to show evolution
- Key metrics: `belief_ledger`, `self_identity_leaning`, `policy_list`

### Removed Survival Alert Policy-Fixing Messages

**File:** `redblackbench/sugarscape/prompts.py`

**Changes:**

1. **Identity Review Prompt (build_identity_review_prompt)**
   - Removed `survival_warning` block that told agents "HARD TRUTH: Your current strategy is NOT working" and suggested fixing policies
   - Removed `trade_analysis` block warning about policies causing trade rejections
   - Simplified reflection questions - removed policy-fixing prompts

2. **Post-Encounter Reflection (build_post_encounter_reflection_prompt)**
   - Removed `survival_context` that told agents to reconsider their policies when low on resources
   - Removed `outcome_guidance` that asked agents to question their policies after failed trades

3. **Trade Intent Decision (build_trade_intent_prompt)**
   - Removed `survival_warning` that pushed agents to trade when resources were low

4. **Negotiation Prompt (build_negotiation_user_prompt)**
   - Removed `survival_warning` that told agents "A bad trade is better than no trade"

**Rationale:**
These warnings were biasing agent behavior by explicitly telling them to change their policies when facing survival pressure. Removing them allows agents to make more autonomous decisions based on their own goals and identity, rather than being guided by the simulation framework.
