import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.llm_agent import LLMSugarAgent
from redblackbench.sugarscape.experiment import ExperimentLogger, MetricsCalculator
from redblackbench.sugarscape.trade import TradeSystem, DialogueTradeSystem
from redblackbench.sugarscape.debug_logger import DebugLogger
from redblackbench.providers.openrouter_provider import OpenRouterProvider
from redblackbench.sugarscape.trajectory import SugarTrajectory, SugarActionRecord
from redblackbench.sugarscape.names import NameGenerator
from redblackbench.sugarscape.welfare import WelfareCalculator

# Import plotting module only if matplotlib is available
try:
    from redblackbench.sugarscape.welfare_plots import WelfarePlotter
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib not available. Plot generation disabled.")

class SugarSimulation:
    """Main simulation controller for Sugarscape."""
    
    def __init__(self, config: SugarscapeConfig = None, agent_factory=None, experiment_name: str = "baseline"):
        self.config = config or SugarscapeConfig()

        # Experiment logging
        self.logger = ExperimentLogger(experiment_type=experiment_name, config=self.config)

        # Initialize Debug Logger
        self.debug_logger = DebugLogger(
            output_dir=self.logger.run_dir / "debug",
            enable_decisions=self.config.debug_log_decisions,
            enable_llm_logs=self.config.debug_log_llm,
            enable_trade_logs=self.config.debug_log_trades,
            enable_death_logs=self.config.debug_log_deaths,
            enable_efficiency_logs=self.config.debug_log_efficiency,
        )

        # Create environment with debug logger
        self.env = SugarEnvironment(self.config, debug_logger=self.debug_logger)

        self.trade_system = None
        if self.config.enable_trade:
            mode = (self.config.trade_mode or "mrs").strip().lower()
            if mode == "dialogue":
                self.trade_system = DialogueTradeSystem(
                    self.env,
                    max_rounds=self.config.trade_dialogue_rounds,
                    allow_fraud=self.config.trade_allow_fraud,
                    memory_maxlen=self.config.trade_memory_maxlen,
                )
            else:
                self.trade_system = TradeSystem(self.env)

        # Initialize LLM Provider if enabled
        self.llm_provider = None
        if self.config.enable_llm_agents:
            provider_type = getattr(self.config, 'llm_provider_type', 'openrouter')
            if provider_type == "vllm":
                from redblackbench.providers.vllm_provider import VLLMProvider
                self.llm_provider = VLLMProvider(
                    model=self.config.llm_provider_model,
                    max_tokens=2048
                )
            else:
                self.llm_provider = OpenRouterProvider(
                    model=self.config.llm_provider_model,
                    max_tokens=2048
                )

        self.agents: List[SugarAgent] = []
        self.tick = 0
        self.next_agent_id = 1
        self.agent_factory = agent_factory or self._default_agent_factory

        # Name generator for human-like names
        self.name_generator = NameGenerator(seed=self.config.seed)

        # Trajectory Tracking
        self.trajectory = SugarTrajectory(
            run_id=self.logger.experiment_id,
            config=self.config.__dict__
        )

        # Random number generator
        self.rng = random.Random(self.config.seed)

        # Track initial population for welfare calculations
        self.initial_population = self.config.initial_population

        self._init_population()

    def _generate_agent_name(self) -> str:
        """Generate a unique human name for an agent."""
        return self.name_generator.generate()
        
    def _default_agent_factory(self, **kwargs) -> SugarAgent:
        return SugarAgent(**kwargs)
            
    def _init_population(self):
        """Create initial population."""
        for _ in range(self.config.initial_population):
            self._create_agent()
            
    def _create_agent(self):
        """Create and place a new agent with random attributes."""
        # Random attributes
        w0 = self.rng.randint(*self.config.initial_wealth_range)
        m = self.rng.randint(*self.config.metabolism_range)
        v = self.rng.randint(*self.config.vision_range)
        max_age = self.rng.randint(*self.config.max_age_range)
        
        # Spice attributes (if enabled)
        spice = 0
        m_spice = 0
        if self.config.enable_spice:
            spice = self.rng.randint(*self.config.initial_spice_range)
            m_spice = self.rng.randint(*self.config.metabolism_spice_range)
        
        # Determine Persona
        persona = "A" # default
        if self.config.enable_personas:
            # Simple weighted choice
            dist = self.config.persona_distribution
            keys = list(dist.keys()) # A, B, C, D
            probs = [dist[k] for k in keys]
            persona = self.rng.choices(keys, weights=probs, k=1)[0]
        
        # Random position
        pos = self.env.get_random_unoccupied_pos(self.rng)
        
        # Determine if this should be an LLM agent
        use_llm = False
        if self.config.enable_llm_agents and self.llm_provider:
            if self.rng.random() < self.config.llm_agent_ratio:
                use_llm = True
                
        if use_llm:
            agent = LLMSugarAgent(
                provider=self.llm_provider,
                goal_prompt=self.config.llm_goal_prompt,
                agent_id=self.next_agent_id,
                pos=pos,
                vision=v,
                metabolism=m,
                max_age=max_age,
                wealth=w0,
                age=0,
                spice=spice,
                metabolism_spice=m_spice,
                name=self._generate_agent_name()  # Assign real human name
            )
        else:
            agent = self.agent_factory(
                agent_id=self.next_agent_id,
                pos=pos,
                vision=v,
                metabolism=m,
                max_age=max_age,
                wealth=w0,
                age=0,
                spice=spice,
                metabolism_spice=m_spice,
                name=self._generate_agent_name()  # Assign real human name
            )
            
        agent.persona = persona
        self.next_agent_id += 1

        self.agents.append(agent)
        self.env.add_agent(agent)
        self.env.initialize_agent_reputation(agent.agent_id)
        
    def step(self):
        """Execute one simulation tick."""
        self.tick += 1
        
        # 1. Environment Growback
        self.env.growback()
        
        # 2. Record Timestep Start
        current_timestep = self.trajectory.add_timestep(
            tick=self.tick, 
            population=len([a for a in self.agents if a.alive])
        )
        
        # 3. Agent updates (Move and Harvest)
        # Shuffle order
        self.rng.shuffle(self.agents)
        
        # This simulation uses a phased tick (Duke-style ordering):
        # 1) Move + Harvest (all agents)
        # 2) Trade (optional)
        # 3) Metabolize + Age + Death check
        #
        # We still batch LLM "decide" calls in parallel to reduce wall-clock time.
        # Moves are applied sequentially with simple collision resolution (occupied -> stay put).
        
        # Let's separate agents into LLM and Standard
        llm_agents = [a for a in self.agents if isinstance(a, LLMSugarAgent) and a.alive]
        std_agents = [a for a in self.agents if not isinstance(a, LLMSugarAgent) and a.alive]
        
        # Phase 1a. Process Standard Agents (Sequential, Fast): Move + Harvest (no metabolism yet)
        for agent in std_agents:
            agent._move_and_harvest(self.env)
            agent._update_metrics(self.env)
                
        # Phase 1b. Process LLM Agents: parallelize decisions, then apply Move + Harvest
        if llm_agents:
            import asyncio
            
            async def get_decisions():
                tasks = []
                for agent in llm_agents:
                    tasks.append(agent.async_decide_move(self.env))
                return await asyncio.gather(*tasks, return_exceptions=True)
            
            # Run all decisions in parallel
            # Note: LLM agents see state after standard agents' movement/harvest,
            # but before other LLM agents' moves are applied.
            decisions = asyncio.run(get_decisions())
            
            # Apply decisions with a batched collision policy for LLM agents:
            # - Single occupancy per cell.
            # - If multiple agents target the same cell, pick a winner uniformly at random (seeded RNG).
            # - Allow moving into cells vacated by other LLM agents in the same batch; deny moves into
            #   cells occupied by non-LLM agents (already moved in phase 1a).
            desired: Dict[SugarAgent, Any] = {}
            decision_by_agent: Dict[SugarAgent, Dict[str, Any]] = {}

            for agent, decision_data in zip(llm_agents, decisions):
                if isinstance(decision_data, Exception):
                    print(f"Agent {agent.agent_id} failed: {decision_data}")
                    continue
                if not isinstance(decision_data, dict):
                    continue
                decision_by_agent[agent] = decision_data
                desired[agent] = decision_data.get("parsed_move")

            # Positions occupied by non-LLM alive agents are immutable for this batch.
            llm_set = set(llm_agents)
            occupied_by_non_llm = {
                pos for pos, occ in self.env.grid_agents.items()
                if (occ is not None) and (occ.alive) and (occ not in llm_set)
            }

            # Step 1: resolve "many want same target" with a fair tie-break.
            targets: Dict[Any, List[SugarAgent]] = {}
            for agent, tpos in desired.items():
                if tpos is None:
                    continue
                targets.setdefault(tpos, []).append(agent)

            winners: Dict[SugarAgent, Any] = {}
            for tpos, agents_wanting in targets.items():
                if len(agents_wanting) == 1:
                    winners[agents_wanting[0]] = tpos
                else:
                    winner = self.rng.choice(agents_wanting)
                    winners[winner] = tpos

            # Step 2: propose final positions (winners move, others stay).
            old_pos: Dict[SugarAgent, Any] = {a: a.pos for a in llm_agents}
            final_pos: Dict[SugarAgent, Any] = {a: old_pos[a] for a in llm_agents}
            for a, tpos in winners.items():
                if tpos is not None:
                    final_pos[a] = tpos

            # Step 3: enforce single occupancy against *stayers* and non-LLM occupied cells.
            # Iteratively revert invalid movers until stable.
            for _ in range(max(1, len(llm_agents))):
                changed = False

                # Cannot move into a non-LLM occupied cell.
                for a, tpos in list(final_pos.items()):
                    if tpos in occupied_by_non_llm and tpos != old_pos[a]:
                        final_pos[a] = old_pos[a]
                        changed = True

                # Resolve duplicates among LLM final positions (e.g., moving into a stayer's cell).
                rev: Dict[Any, List[SugarAgent]] = {}
                for a, tpos in final_pos.items():
                    rev.setdefault(tpos, []).append(a)

                for tpos, agents_here in rev.items():
                    if len(agents_here) <= 1:
                        continue
                    # Prefer keeping the agent whose original position is this cell (a "stayer"/rightful occupant).
                    rightful = [a for a in agents_here if old_pos[a] == tpos]
                    if rightful:
                        keep = rightful[0]
                    else:
                        keep = self.rng.choice(agents_here)
                    for a in agents_here:
                        if a is keep:
                            continue
                        final_pos[a] = old_pos[a]
                        changed = True

                if not changed:
                    break

            # Step 4: apply batched moves by rebuilding the env grid mapping for these agents.
            for a in llm_agents:
                # Remove old mapping if present.
                if self.env.grid_agents.get(old_pos[a]) == a:
                    del self.env.grid_agents[old_pos[a]]
            for a in llm_agents:
                a.pos = final_pos[a]
                self.env.grid_agents[a.pos] = a

            # Step 5: harvest + metrics update per agent (after movement is finalized)
            for agent in llm_agents:
                decision_data = decision_by_agent.get(agent, {})
                target_pos = desired.get(agent)

                rewards = agent._harvest_and_update_metrics(self.env)
                
                # Record move history for LLM agents
                if hasattr(agent, 'move_history'):
                    agent.move_history.append({
                        "tick": self.tick,
                        "pos": agent.pos,
                        "action": target_pos,
                        "wealth": agent.wealth,
                        "spice": agent.spice,
                        "sugar_harvested": rewards["sugar_harvested"],
                        "spice_harvested": rewards.get("spice_harvested", 0),
                    })
                    
                # Record Action in Trajectory
                action_record = SugarActionRecord(
                    agent_id=agent.agent_id,
                    system_prompt=decision_data.get("system_prompt", ""),
                    user_prompt=decision_data.get("user_prompt", ""),
                    raw_response=decision_data.get("raw_response", ""),
                    parsed_move=target_pos,
                    reward_sugar=rewards["sugar_harvested"],
                    reward_spice=rewards["spice_harvested"],
                    metabolic_cost_sugar=agent.metabolism,
                    metabolic_cost_spice=agent.metabolism_spice,
                )
                current_timestep.actions.append(action_record)

                # Log LLM interaction to debug logger
                from redblackbench.sugarscape.debug_logger import LLMInteraction
                llm_interaction = LLMInteraction(
                    tick=self.tick,
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    interaction_type="movement",
                    system_prompt=decision_data.get("system_prompt", ""),
                    user_prompt=decision_data.get("user_prompt", ""),
                    raw_response=decision_data.get("raw_response", ""),
                    parsed_action=str(target_pos) if target_pos else "",
                    input_tokens=0,  # TODO: Extract from provider if available
                    output_tokens=0,  # TODO: Extract from provider if available
                    latency_ms=0.0,  # TODO: Track latency
                )
                self.debug_logger.log_llm_interaction(llm_interaction)

        # Re-shuffle not needed since we processed in groups, but good for next tick?
        # self.rng.shuffle(self.agents) # Done at start
        
        # Phase 2. Trade (optional) happens after movement/harvest and before metabolism
        if self.config.enable_trade and self.trade_system:
            # Only alive agents trade
            live_agents = [a for a in self.agents if a.alive]
            self.trade_system.execute_trade_round(live_agents, tick=self.tick)
                
        # Phase 3. Metabolize + Age + Death check (applied to all agents)
        dead_agents = []
        for agent in self.agents:
            if not agent.alive:
                continue
            agent.metabolize_age_and_check_death(self.env)
            if not agent.alive:
                dead_agents.append(agent)
        
        for agent in dead_agents:
            if agent in self.agents:
                self.agents.remove(agent)
                self.env.remove_agent(agent)
                # Replacement Rule: Constant population
                self._create_agent()
            
        # Logging
        if self.tick % 10 == 0:
            stats = self.get_stats()
            self.logger.log_step(stats)
            
    def run(self, steps: int = None):
        """Run for a number of steps."""
        # Initial snapshot
        self.save_snapshot(filename="initial_state.json")
        
        limit = steps or self.config.max_ticks
        for _ in range(limit):
            self.step()
            
        # Final snapshot
        self.save_snapshot(filename="final_state.json")

        # Save Trajectory
        traj_filename = f"trajectory_{self.logger.experiment_id}.json"
        self.trajectory.save(self.logger.get_log_path(traj_filename))

        # Save debug summary
        self.debug_logger.save_summary()

        # Generate welfare plots
        self._generate_plots()
            
    def save_snapshot(self, filename: str):
        """Save current simulation state."""
        data = {
            "tick": self.tick,
            "agents": [
                {
                    "id": a.agent_id,
                    "pos": a.pos,
                    "wealth": a.wealth,
                    "spice": a.spice,
                    "age": a.age,
                    "persona": a.persona,
                    "vision": a.vision,
                    "metabolism": a.metabolism,
                    "metabolism_spice": a.metabolism_spice,
                    "metrics": getattr(a, 'metrics', {})
                }
                for a in self.agents
            ],
            "sugar_map": self.env.sugar_amount.tolist(),
            "spice_map": self.env.spice_amount.tolist(),
            "sugar_capacity": self.env.sugar_capacity.tolist()
        }
        self.logger.save_snapshot(data, filename)
    
    def _generate_plots(self):
        """Generate welfare visualization plots after simulation completes."""
        if not PLOTTING_AVAILABLE:
            print("Note: Plot generation skipped (matplotlib not available)")
            return

        try:
            # Determine title prefix based on agent type
            if self.config.enable_llm_agents:
                if self.config.llm_agent_ratio >= 1.0:
                    title_prefix = "LLM Agents"
                elif self.config.llm_agent_ratio > 0:
                    title_prefix = f"Mixed ({int(self.config.llm_agent_ratio*100)}% LLM)"
                else:
                    title_prefix = "Rule-Based Agents"
            else:
                title_prefix = "Rule-Based Agents"

            # Generate plots from the metrics CSV
            WelfarePlotter.generate_all_plots(
                csv_path=str(self.logger.csv_file),
                plots_dir=self.logger.get_plots_dir(),
                title_prefix=title_prefix
            )

            print(f"✓ Generated welfare plots in: {self.logger.get_plots_dir()}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics including welfare metrics."""
        wealths = [a.wealth for a in self.agents]
        spices = [a.spice for a in self.agents]
        ages = [a.age for a in self.agents]
        positions = [a.pos for a in self.agents]
        
        # Calculate Moran's I
        moran = MetricsCalculator.calculate_moran_i(
            self.config.width, self.config.height, positions, wealths
        )
        
        # Calculate Mobility
        mobility = MetricsCalculator.calculate_mobility_stats(self.agents)
        
        # Calculate Welfare Metrics
        welfare_metrics = WelfareCalculator.calculate_all_welfare_metrics(
            self.agents, 
            self.initial_population
        )
        
        # Combine all metrics
        stats = {
            "tick": self.tick,
            "population": len(self.agents),
            "mean_wealth": np.mean(wealths) if wealths else 0,
            "mean_spice": np.mean(spices) if spices else 0,
            "max_wealth": np.max(wealths) if wealths else 0,
            "min_wealth": np.min(wealths) if wealths else 0,
            "gini": self._gini_coefficient(wealths) if wealths else 0,
            "mean_age": np.mean(ages) if ages else 0,
            "moran_i": moran,
            "avg_displacement": mobility["avg_displacement"],
            "avg_exploration": mobility["avg_exploration"]
        }
        
        # Add all welfare metrics
        stats.update(welfare_metrics)
        
        return stats
        
    def _gini_coefficient(self, values):
        """Calculate Gini coefficient for wealth inequality."""
        if not values: return 0
        sorted_vals = sorted(values)
        n = len(values)
        total = sum(sorted_vals)
        if total == 0: return 0

        # G = (2 * sum(i * xi) / (n * sum(xi))) - (n + 1) / n
        # where i is 1-based index
        weighted_sum = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n

    # ========== Checkpoint System ==========

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """Save complete simulation state for later resumption.

        Args:
            filename: Optional custom filename. If None, uses tick number.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_data = {
            # Simulation state
            "tick": self.tick,
            "next_agent_id": self.next_agent_id,
            "initial_population": self.initial_population,
            "rng_state": self.rng.getstate(),

            # Config (serialized)
            "config": self.config.__dict__.copy(),

            # Name generator state
            "name_generator_state": self.name_generator.get_state(),

            # Environment state
            "environment": self.env.get_checkpoint_state(),

            # Agent states
            "agents": [agent.to_checkpoint_dict() for agent in self.agents],

            # Metadata
            "experiment_id": self.logger.experiment_id,
            "run_dir": str(self.logger.run_dir),
        }

        # Use logger's checkpoint save method
        checkpoint_path = self.logger.save_checkpoint(checkpoint_data, self.tick)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Union[str, Path]) -> "SugarSimulation":
        """Restore simulation from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint .pkl file.

        Returns:
            Restored SugarSimulation instance ready to continue.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct config
        config = SugarscapeConfig(**data["config"])

        # Create simulation instance (but skip normal initialization)
        sim = object.__new__(cls)

        # Restore config
        sim.config = config

        # Restore experiment logger - create new one in same directory structure
        # We extract experiment name from the run_dir
        run_dir = Path(data.get("run_dir", ""))
        experiment_type = run_dir.parent.name if run_dir.parent else "resumed"

        # Create logger pointing to same run directory
        sim.logger = ExperimentLogger(
            experiment_type=experiment_type,
            config=config
        )
        # Override to use original experiment ID for consistency
        # Note: This creates a NEW run dir, but we keep the experiment_id for tracking
        original_experiment_id = data.get("experiment_id", sim.logger.experiment_id)

        # Debug logger
        sim.debug_logger = DebugLogger(
            output_dir=sim.logger.run_dir / "debug",
            enable_decisions=config.debug_log_decisions,
            enable_llm_logs=config.debug_log_llm,
            enable_trade_logs=config.debug_log_trades,
            enable_death_logs=config.debug_log_deaths,
            enable_efficiency_logs=config.debug_log_efficiency,
        )

        # Environment
        sim.env = SugarEnvironment(config, debug_logger=sim.debug_logger)
        sim.env.restore_checkpoint_state(data["environment"])

        # Trade system
        sim.trade_system = None
        if config.enable_trade:
            mode = (config.trade_mode or "mrs").strip().lower()
            if mode == "dialogue":
                sim.trade_system = DialogueTradeSystem(
                    sim.env,
                    max_rounds=config.trade_dialogue_rounds,
                    allow_fraud=config.trade_allow_fraud,
                    memory_maxlen=config.trade_memory_maxlen,
                )
            else:
                sim.trade_system = TradeSystem(sim.env)

        # LLM provider
        sim.llm_provider = None
        if config.enable_llm_agents:
            provider_type = getattr(config, 'llm_provider_type', 'openrouter')
            if provider_type == "vllm":
                from redblackbench.providers.vllm_provider import VLLMProvider
                sim.llm_provider = VLLMProvider(
                    model=config.llm_provider_model,
                    max_tokens=2048
                )
            else:
                sim.llm_provider = OpenRouterProvider(
                    model=config.llm_provider_model,
                    max_tokens=2048
                )

        # Restore simulation state
        sim.tick = data["tick"]
        sim.next_agent_id = data["next_agent_id"]
        sim.initial_population = data["initial_population"]

        # RNG state
        sim.rng = random.Random()
        sim.rng.setstate(data["rng_state"])

        # Name generator
        sim.name_generator = NameGenerator()
        sim.name_generator.set_state(data["name_generator_state"])

        # Agent factory
        sim.agent_factory = lambda **kwargs: SugarAgent(**kwargs)

        # Trajectory (new for resumed run)
        sim.trajectory = SugarTrajectory(
            run_id=sim.logger.experiment_id,
            config=config.__dict__
        )

        # Restore agents
        sim.agents = []
        for agent_data in data["agents"]:
            is_llm = agent_data.get("is_llm_agent", False)

            if is_llm and sim.llm_provider:
                agent = LLMSugarAgent(
                    provider=sim.llm_provider,
                    goal_prompt=agent_data.get("goal_prompt", config.llm_goal_prompt),
                    agent_id=agent_data["agent_id"],
                    pos=tuple(agent_data["pos"]),
                    vision=agent_data["vision"],
                    metabolism=agent_data["metabolism"],
                    max_age=agent_data["max_age"],
                    wealth=agent_data["wealth"],
                    spice=agent_data.get("spice", 0),
                    metabolism_spice=agent_data.get("metabolism_spice", 0),
                    age=agent_data.get("age", 0),
                    persona=agent_data.get("persona", "A"),
                    name=agent_data.get("name", ""),
                )
            else:
                agent = SugarAgent(
                    agent_id=agent_data["agent_id"],
                    pos=tuple(agent_data["pos"]),
                    vision=agent_data["vision"],
                    metabolism=agent_data["metabolism"],
                    max_age=agent_data["max_age"],
                    wealth=agent_data["wealth"],
                    spice=agent_data.get("spice", 0),
                    metabolism_spice=agent_data.get("metabolism_spice", 0),
                    age=agent_data.get("age", 0),
                    persona=agent_data.get("persona", "A"),
                    name=agent_data.get("name", ""),
                )

            # Restore agent state from checkpoint
            agent.restore_from_checkpoint(agent_data)

            sim.agents.append(agent)
            sim.env.grid_agents[agent.pos] = agent

        print(f"Checkpoint loaded: tick={sim.tick}, agents={len(sim.agents)}")
        print(f"Resumed experiment at: {sim.logger.run_dir}")

        return sim

    def run_with_checkpoints(self, steps: Optional[int] = None, checkpoint_interval: int = 50) -> None:
        """Run simulation with periodic checkpoint saves.

        Args:
            steps: Number of steps to run. If None, runs to max_ticks.
            checkpoint_interval: Save checkpoint every N ticks.
        """
        # Initial snapshot
        self.save_snapshot(filename="initial_state.json")

        limit = steps or self.config.max_ticks
        start_tick = self.tick

        for i in range(limit):
            self.step()

            # Save checkpoint at intervals
            if checkpoint_interval > 0 and self.tick % checkpoint_interval == 0:
                self.save_checkpoint()

        # Final snapshot
        self.save_snapshot(filename="final_state.json")

        # Save trajectory
        traj_filename = f"trajectory_{self.logger.experiment_id}.json"
        self.trajectory.save(self.logger.get_log_path(traj_filename))

        # Save debug summary
        self.debug_logger.save_summary()

        # Generate welfare plots
        self._generate_plots()

        print(f"Simulation complete: {start_tick} -> {self.tick} ({limit} steps)")
