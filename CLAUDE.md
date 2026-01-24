# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the **Sugarscape** multi-agent economic simulation for studying LLM agent decision-making, moral evolution, trading behavior, and welfare dynamics. The simulation enables research on how AI agents develop values, cooperate, and adapt through social interactions.

---

## Sugarscape Module

Extended multi-agent economic simulation in `redblackbench/sugarscape/` for studying agent decision-making, trading, and welfare dynamics.

### Simulation Loop (per tick)

1. Environment growback (resources regenerate)
2. Standard agents move sequentially (persona-based scoring)
3. LLM agents decide in parallel (async batched), then apply moves
4. Encounter/Trading phase (if enabled)
5. Event-triggered identity reviews (if enabled)
6. Metabolism + aging + death (constant population - dead agents replaced)

### Key Classes

- **`SugarAgent`** (`agent.py`): Base agent with vision, metabolism, wealth, spice, persona, trade memory, trust scores, baseline_snapshot
- **`LLMSugarAgent`** (`llm_agent.py`): Extends SugarAgent with LLM provider, goal prompt, conversation history. Uses `async_decide_move()` for batched inference
- **`SugarEnvironment`** (`environment.py`): 50x50 torus grid with sugar/spice peaks at (15,15)/(35,35). Handles growback, harvest, agent tracking
- **`SugarSimulation`** (`simulation.py`): Main controller orchestrating tick loop, snapshot saving, metrics logging
- **`SugarscapeConfig`** (`config.py`): Comprehensive dataclass with 50+ parameters
- **`DialogueTradeSystem`** (`trade.py`): Handles encounters, negotiation, and trade execution

---

## Core Systems

### Checkpoint System (`simulation.py`)

- `save_checkpoint()` - Saves complete simulation state to pickle file
- `load_checkpoint(path)` - Class method to restore simulation from checkpoint
- `run_with_checkpoints(steps, checkpoint_interval)` - Run with periodic saves
- **Full state preserved**: RNG state, agent conversation history, trade memory, visited cells, partner trust, global reputation, baseline beliefs
- Checkpoints stored in `results/sugarscape/{experiment}/checkpoints/checkpoint_tick_N.pkl`

### Identity & Belief Evolution System

**Event-Triggered Identity Review** (recommended, `enable_event_triggered_identity_review=True`):
- Reflections triggered by significant events, not periodic intervals
- Trigger events: `defrauded`, `successful_cooperation`, `resources_critical`, `trade_rejected`, `witnessed_death`
- More meaningful reflections tied to actual experiences
- Stores in `identity_review_history` with `identity_shift` (±0.3) tracking

**T=0 Baseline Capture** (`enable_origin_identity=True`):
- Captures agent beliefs before any interactions
- Outputs to `baseline_beliefs.json`
- Key metrics: `belief_ledger`, `self_identity_leaning`, `policy_list`
- Proves value changes emerge from interactions, not pre-existing in LLM

**Origin Identities**:
- `altruist`: Cooperative, helper-oriented
- `exploiter`: Self-interested, willing to deceive
- `survivor`: Survival-focused (default for "none" goal)

### Trading System (`trade.py`)

**Trade Modes**:
- **MRS mode**: Marginal Rate of Substitution bargaining. Price = geometric mean of both agents' MRS
- **Dialogue mode**: LLM-based negotiation with CHAT/OFFER/ACCEPT/REJECT/WALK_AWAY actions. Supports optional fraud (public offer vs private execution). Trust updated per trade

**Encounter Protocol Modes** (`encounter_protocol_mode`):
- `"full"`: Small talk + trade intent + negotiation (default)
- `"chat_only"`: Small talk + reflection only, no negotiation/transfer
- `"protocol_only"`: JSON-only negotiation, no small talk or MESSAGE actions

### Resource Specialization

Creates meaningful trade through complementary demand (`enable_resource_specialization=True`):
- **Sugar specialists** (50%): High sugar metabolism (3-4), low spice metabolism (1-2)
- **Spice specialists** (50%): High spice metabolism (3-4), low sugar metabolism (1-2)

### External Moral Evaluator (`moral_evaluator.py`)

Independent LLM "judge" that scores moral dimensions from agent prompts + replies + trade transcripts:
- 6 dimensions scored 0-100: fairness, cooperation, honesty, harm_avoidance, autonomy_respect, social_responsibility
- Outputs: `moral_evals.jsonl` (full audit), `moral_scores.csv` (curve-ready)
- Hooked into: T=0 baseline, post-encounter reflection, event-triggered reviews, end-of-life reports

### Real-time Trade Evaluation (`evaluator.py`)

Per-trade fairness scoring (`enable_realtime_trade_eval=True`):

**Objective metrics** (no LLM):
- `trade_fairness_objective`: -1 to 1 scale
- `net_transfer_ratio`: 0 to 1 (1.0 = perfectly balanced)
- `urgency_exploitation`: True if CRITICAL party got bad deal

**LLM-based** (optional, `realtime_trade_eval_use_llm=True`):
- `trade_fairness`, `cooperation_signal`: 1-7 scale
- `brief_reason`: 1-sentence explanation

### Trust & Reputation System

- **Personal Trust**: Per-partner trust scores updated after each trade
- **Global Reputation**: Visible to all agents (when `social_memory_visible=True`)
- Fraud detection affects both trust and reputation

### Last Words Feature

When an agent dies with an end-of-life report, nearby agents receive their "last words" in `witnessed_death` events:
- Uses `advice` field from end-of-life report
- Falls back to first 200 chars of `final_reflection`
- Provides emotional/philosophical context for reflection

---

## Goal Presets for LLM Agents

- `none`: No explicit goal (baseline) - "Observe your situation and decide what to do"
- `survival`: Stay alive, secondarily accumulate food
- `wealth`: Maximize total resources
- `altruist`: Help others, share resources (aliases: `egalitarian`, `utilitarian`, `samaritan`)

**Mixed Goals** - Different LLM agents can have different goals:
```python
config = SugarscapeConfig(
    enable_llm_agents=True,
    enable_mixed_goals=True,
    llm_goal_distribution={
        "survival": 0.4,   # 40% survival-focused
        "wealth": 0.3,     # 30% wealth-focused
        "altruist": 0.2,   # 20% altruistic
        "none": 0.1,       # 10% no explicit goal
    }
)
```

---

## Persona System (A/B/C/D)

For non-LLM agents' movement decisions:
- **A (Conservative)**: Prioritizes safety, avoids long moves
- **B (Foresight)**: Balances immediate utility with long-term site quality
- **C (Nomad)**: Seeks novelty and unexplored regions
- **D (Risk-taker)**: Aggressive utility maximization

---

## Welfare Metrics (`welfare.py`)

- **Individual**: Cobb-Douglas utility `wealth^(m_s/m_total) * spice^(m_p/m_total)`
- **Social**: Utilitarian (sum), Rawlsian (min), Nash (geometric mean), Gini-adjusted, Atkinson-adjusted

---

## Running Sugarscape

```bash
python scripts/run_sugarscape.py
python scripts/run_goal_experiment.py      # Compare goal presets
python scripts/run_persona_experiment.py   # Compare personas
python scripts/run_goal_experiment_sft.py  # Run with SFT-aligned models
```

### Output Structure

```
results/sugarscape/{experiment}/
├── config.json
├── metrics.csv                 # Time series (every 10 ticks)
├── trajectory_*.json           # LLM interactions for RL
├── initial_state.json          # Snapshot at tick 0
├── final_state.json            # Snapshot at simulation end
├── baseline_beliefs.json       # T=0 beliefs before interactions
├── checkpoints/                # Full state checkpoints (pickle)
│   ├── checkpoint_tick_50.pkl
│   └── checkpoint_tick_100.pkl
├── plots/                      # Welfare visualizations
└── debug/                      # Optional detailed logs
    ├── single_trade_evals.csv  # Per-trade fairness scores
    ├── single_trade_evals.jsonl
    ├── moral_evals.jsonl       # External moral evaluations
    └── moral_scores.csv
```

### Checkpoint/Resume Usage

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

---

## Configuration

Experiments are configured via YAML (see `experiments/configs/`) or Python:

```python
config = SugarscapeConfig(
    # Core simulation
    grid_size=50,
    num_agents=100,
    max_ticks=200,
    
    # Resources
    enable_spice=True,
    enable_resource_specialization=True,
    
    # Trading & Encounters
    enable_trade=True,
    trade_mode="dialogue",  # "mrs" or "dialogue"
    encounter_protocol_mode="full",  # "full", "chat_only", "protocol_only"
    enable_encounter_dialogue=True,
    
    # LLM Agents
    enable_llm_agents=True,
    llm_agent_ratio=0.2,
    llm_goal="survival",
    
    # Identity & Reflection
    enable_origin_identity=True,
    enable_event_triggered_identity_review=True,
    enable_identity_review=False,  # Periodic review (disabled by default)
    
    # Evaluation
    enable_llm_evaluation=True,
    enable_realtime_trade_eval=True,
    enable_external_moral_evaluation=True,
    
    # Ablations
    enable_survival_pressure=True,
    social_memory_visible=True,
)
```

---

## Code Patterns

- All LLM calls are async - use `asyncio.gather()` for parallel operations
- Providers use a common interface via `BaseLLMProvider`
- Dataclasses used for configuration and data structures
- pytest-asyncio with `asyncio_mode = "auto"` for tests

---

## vLLM Local Inference

For local model inference (see VLLM_INTEGRATION.md):

```bash
# Start vLLM server
bash /workspace/start_qwen3_base_server.sh

# Test provider
python test_vllm_provider.py

# Run with vLLM config
python scripts/run_goal_experiment_sft.py --model Qwen3-14B-LoRA
```

---

## Development Guidelines

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
    # Survival pressure
    enable_survival_pressure=True,    # When False: no starvation death
    
    # Social memory visibility
    social_memory_visible=True,       # When False: hide trust, reputation, trade history
    
    # Encounter modes
    encounter_protocol_mode="full",   # "chat_only" or "protocol_only" for ablations
    
    # Resource specialization
    enable_resource_specialization=True,  # When False: random independent metabolism
)
```

These ablation flags are orthogonal and can be combined for experimental designs.

---

## Analysis Scripts

```bash
# Comprehensive analysis of formal experiments
python scripts/analyze_formal_experiments.py

# Deep analysis of specific experiments
python scripts/analyze_sugarscape_deep.py

# Plot moral evolution over time
python scripts/plot_moral_comparison.py
```

---

## Key Experiment Documentation

See `redblackbench/sugarscape/EXPERIMENTS.md` for:
- Full experiment suite mapping to ICML paper narrative
- "Seed & Soil" experimental framework
- Mixed-society outcome analysis (redemption vs corruption)
- Tables: Soil Effect, Seed Effect, Enlightenment Speed, Cause of Death
- Normie dialogue ablations (Experiments 11-14)
