import random
from typing import List, Dict, Any
import numpy as np

from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.llm_agent import LLMSugarAgent
from redblackbench.sugarscape.experiment import ExperimentLogger, MetricsCalculator
from redblackbench.sugarscape.trade import TradeSystem, DialogueTradeSystem
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
        self.env = SugarEnvironment(self.config)
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
            self.llm_provider = OpenRouterProvider(
                model=self.config.llm_provider_model,
                max_tokens=2048  # Lowered from 4096 to save tokens as requested
            )
        
        self.agents: List[SugarAgent] = []
        self.tick = 0
        self.next_agent_id = 1
        self.agent_factory = agent_factory or self._default_agent_factory
        
        # Name generator for human-like names
        self.name_generator = NameGenerator(seed=self.config.seed)
        
        # Experiment logging
        self.logger = ExperimentLogger(experiment_type=experiment_name, config=self.config)
        
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
            
            # Apply decisions
            for agent, decision_data in zip(llm_agents, decisions):
                if isinstance(decision_data, Exception):
                    print(f"Agent {agent.agent_id} failed: {decision_data}")
                    continue
                
                # decision_data is a dict with details
                if not isinstance(decision_data, dict):
                    # Should not happen with updated LLMAgent, but safety check
                    continue

                target_pos = decision_data.get("parsed_move")
                
                if target_pos:
                    # Verify occupancy again (collision resolution)
                    if not self.env.is_occupied(target_pos):
                        self.env.move_agent(agent, target_pos)
                    # If occupied, stay put (simple resolution)
                
                # Harvest + metrics update (metabolism/aging happens later, after trade)
                rewards = agent._harvest_and_update_metrics(self.env)
                    
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
