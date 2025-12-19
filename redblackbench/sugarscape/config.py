from dataclasses import dataclass, field
from typing import Tuple, Dict, List

@dataclass
class SugarscapeConfig:
    """Configuration for the Sugarscape simulation."""
    
    # Environment
    width: int = 50
    height: int = 50
    
    # Sugar distribution
    # Classic topography has two peaks with max capacity 4
    max_sugar_capacity: int = 4
    sugar_growback_rate: int = 1  # alpha
    
    # Spice distribution (Optional, for Sugarscape 2)
    enable_spice: bool = False
    max_spice_capacity: int = 4
    spice_growback_rate: int = 1
    
    # Trade (Optional)
    enable_trade: bool = False
    trade_mode: str = "mrs"  # "mrs" | "dialogue"
    trade_dialogue_rounds: int = 4
    trade_allow_fraud: bool = True
    trade_memory_maxlen: int = 50
    
    # Personas (Optional)
    enable_personas: bool = False
    # Distribution: A (Conservative), B (Foresight), C (Nomad), D (Risk-taker)
    persona_distribution: Dict[str, float] = field(default_factory=lambda: {
        "A": 0.36, 
        "B": 0.29, 
        "C": 0.21, 
        "D": 0.14
    })
    # Persona Hyperparameters
    risk_aversion: float = 0.85     # gamma
    exploration_factor: float = 0.10 # lambda
    crowding_penalty: float = 0.60   # kappa
    long_term_weight: float = 0.55   # beta
    safety_threshold_mult: float = 6.0 # S* = mult * metabolism
    
    # Population
    initial_population: int = 250
    
    # Agent Attribute Distributions (Uniform [min, max])
    initial_wealth_range: Tuple[int, int] = (5, 25)  # w0
    initial_spice_range: Tuple[int, int] = (5, 25)   # spice w0
    metabolism_range: Tuple[int, int] = (1, 4)       # m_sugar
    metabolism_spice_range: Tuple[int, int] = (1, 4) # m_spice
    vision_range: Tuple[int, int] = (1, 6)           # v
    max_age_range: Tuple[int, int] = (60, 100)       # max_age
    
    # LLM Agent Configuration
    enable_llm_agents: bool = False
    llm_agent_ratio: float = 0.0 # Fraction of population that is LLM-based
    llm_provider_model: str = "openai/gpt-4o"
    llm_goal_preset: str = "survival"  # Goal preset: "survival", "wealth", "egalitarian", "utilitarian"
    llm_goal_prompt: str = ""  # Will be set based on preset or custom input
    llm_history_limit: int = 15 # Number of recent steps to keep in history context

    @staticmethod
    def get_goal_prompt(preset: str) -> str:
        """Get the goal prompt for a given preset."""
        goals = {
            "survival": """Your primary goal is to survive as long as possible. Secondarily, maximize your accumulated wealth.

Survival means maintaining sufficient resources to avoid termination. Consider both immediate resource needs and long-term sustainability.""",

            "wealth": """Your goal is to maximize your accumulated wealth (Sugar + Spice if applicable).

Focus on efficient resource acquisition and strategic positioning. Survival is important only insofar as it enables wealth accumulation.""",

            "egalitarian": """Your goal is to promote societal equality and fairness. You should:

1. **Primary:** Help ensure no agent is left behind. Prioritize actions that benefit agents with the lowest welfare.
2. **Secondary:** Maintain your own viability to continue helping others.
3. **Consider:** How your decisions affect the overall distribution of resources in society.

You value equitable outcomes over maximum individual wealth.""",

            "utilitarian": """Your goal is to maximize total societal welfare. You should:

1. **Primary:** Take actions that increase the sum of all agents' welfare, even if it means personal sacrifice.
2. **Secondary:** Maintain your own operational capacity to continue contributing to total welfare.
3. **Consider:** The broader impact of your decisions on the entire population.

Total welfare maximization may require helping others at your own expense."""
        }

        if preset in goals:
            return goals[preset]
        else:
            raise ValueError(f"Unknown goal preset: {preset}. Available presets: {list(goals.keys())}")

    def __post_init__(self):
        """Set the goal prompt based on the preset."""
        if not self.llm_goal_prompt:  # Only set if not explicitly provided
            self.llm_goal_prompt = self.get_goal_prompt(self.llm_goal_preset)
    llm_name_pool: List[str] = field(
        default_factory=lambda: [
            "Alex",
            "Sam",
            "Taylor",
            "Jordan",
            "Casey",
            "Riley",
            "Morgan",
            "Avery",
            "Jamie",
            "Quinn",
            "Skyler",
            "Cameron",
            "Drew",
            "Hayden",
        ]
    )

    # Simulation
    max_ticks: int = 1000
    seed: int = 42
