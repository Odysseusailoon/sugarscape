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
    # Dialogue trade robustness:
    # - repair: attempt a prompt-based "output JSON only" retry if parsing fails or intent is invalid
    # - coerce: (optional) hard fallback to OFFER/ACCEPT/REJECT to avoid TIMEOUT
    # - two_stage: use thinking model for reasoning, then non-thinking for JSON output
    trade_dialogue_repair_json: bool = True
    trade_dialogue_repair_attempts: int = 1
    trade_dialogue_coerce_protocol: bool = False
    trade_dialogue_two_stage: bool = True  # Default on: thinking → JSON pipeline
    trade_dialogue_thinking_tokens: int = 1024  # Tokens for Stage 1 (thinking) - keep high for quality
    trade_dialogue_json_tokens: int = 128  # Tokens for Stage 2 (JSON output) - reduced, JSON is small
    
    # Personas (Optional)
    enable_personas: bool = False
    # Distribution: A (Conservative), B (Foresight), C (Nomad), D (Risk-taker), E (Samaritan/Altruist)
    persona_distribution: Dict[str, float] = field(default_factory=lambda: {
        "A": 0.36, 
        "B": 0.29, 
        "C": 0.21, 
        "D": 0.14,
        "E": 0.0  # Samaritan (altruistic) - disabled by default
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
    llm_provider_type: str = "openrouter"  # "openrouter" or "vllm"
    llm_provider_model: str = "openai/gpt-4o"  # Model name for provider
    llm_vllm_base_url: str = "http://localhost:8000/v1"  # vLLM server URL
    llm_goal_preset: str = "survival"  # Goal preset: "survival", "wealth", "egalitarian", "utilitarian"
    llm_goal_prompt: str = ""  # Will be set based on preset or custom input
    llm_history_limit: int = 15 # Number of recent steps to keep in history context (safe with 262K+ context models)

    # Mixed Goals (Optional) - assign different goals to different LLM agents
    enable_mixed_goals: bool = False  # When True, use llm_goal_distribution instead of single preset
    llm_goal_distribution: Dict[str, float] = field(default_factory=lambda: {
        "survival": 0.4,   # 40% survival-focused
        "wealth": 0.3,     # 30% wealth-focused
        "altruist": 0.2,   # 20% altruistic
        "none": 0.1,       # 10% no explicit goal
    })

    @staticmethod
    def get_goal_prompt(preset: str) -> str:
        """Get the goal prompt for a given preset."""
        goals = {
            "none": """You are a person living in this world. You decide what matters to you.""",

            "survival": """You want to live. Stay alive as long as you can.
You need both Sugar and Spice - if either runs out, you die.
When you're low on one, find it or trade for it.""",

            "wealth": """You want to be prosperous. Accumulate as much as you can.
Both Sugar and Spice make you wealthy. The more you have of both, the better off you are.
Seek abundance, but remember you need both to thrive.""",

            "altruist": """You care about others. You believe everyone deserves to live.

When you see someone struggling - low on Sugar or Spice - you want to help them. When you have plenty and others have little, sharing feels right. You'd rather live modestly in a world where everyone survives than live richly while others starve.

In trades, you think about whether the other person needs this more than you. If they're desperate for Spice and you have extra, maybe give them a good deal.

You stay alive because you can do more good alive than dead.""",
        }

        # Aliases for backward compatibility
        goals["egalitarian"] = goals["altruist"]
        goals["utilitarian"] = goals["altruist"]
        goals["samaritan"] = goals["altruist"]

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

    # Debug Logging (Optional - enable for detailed experiment analysis)
    enable_debug_logging: bool = False  # Master switch for all debug logs
    debug_log_decisions: bool = True    # Per-agent decision reasoning
    debug_log_llm: bool = True          # Full LLM prompts/responses (large files!)
    debug_log_trades: bool = True       # Complete trade history with prices
    debug_log_deaths: bool = True       # Death causes (starvation vs age)
    debug_log_efficiency: bool = True   # Resource gathering efficiency
