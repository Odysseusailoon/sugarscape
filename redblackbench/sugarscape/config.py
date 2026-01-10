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
    trade_mode: str = "dialogue"  # "dialogue" | "mrs"
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

            "survival": """Your goal: Stay alive as long as possible.

DECISION PRIORITY:
1. CRITICAL: If you're CRITICAL on anything, fix it immediately - move to resources
2. LOW: Build safety buffer before anything else
3. OK/SURPLUS: Maintain reserves, avoid unnecessary risks

MOVEMENT:
- Always move toward the best resources for YOUR needs
- Avoid cells with low resources even if others are there
- Other agents are potential trade partners, not priorities

TRADING:
- Only trade if it improves YOUR survival odds
- Don't accept bad deals even if others need help
- Your life comes first

SUCCESS = You survive. Measured by how long YOU stay alive.""",

            "wealth": """Your goal: Accumulate maximum resources.

DECISION PRIORITY:
1. CRITICAL: Fix immediate danger (can't get rich if dead)
2. Otherwise: Always move to highest-yield cells
3. Take calculated risks for bigger gains

MOVEMENT:
- Prioritize cells with MOST resources, not safest cells
- Move aggressively toward abundance
- Other agents may have resources to trade - approach if profitable

TRADING:
- Trade when it increases YOUR total holdings
- Drive hard bargains - maximize what you get
- Their situation is not your concern

SUCCESS = Maximum wealth. Measured by YOUR Sugar + Spice totals.""",

            "altruist": """Your goal: Everyone survives, including yourself.

DECISION PRIORITY:
1. CRITICAL SELF: Save yourself first (dead helpers can't help)
2. CRITICAL OTHER: See someone CRITICAL? Move toward them to help
3. SURPLUS SELF + LOW OTHER: Seek out struggling agents
4. Otherwise: Gather resources for future helping

MOVEMENT:
- When comfortable, prioritize moving toward struggling agents
- Check agent status: CRITICAL means they'll die soon without help
- A good position near someone in need beats a great position alone

TRADING:
- If they're worse off than you: Give favorable terms
- If they're CRITICAL: Accept bad deals to save them
- Your wealth means nothing if others starve

SUCCESS = Everyone survives. Measured by group welfare, especially the worst-off.""",
        }

        # Aliases for backward compatibility
        goals["egalitarian"] = goals["altruist"]
        goals["utilitarian"] = goals["altruist"]
        goals["samaritan"] = goals["altruist"]
        goals["rawlsian"] = goals["altruist"]

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
    checkpoint_interval: int = 50  # Save checkpoint every N ticks (0 to disable)

    # Debug Logging (Optional - enable for detailed experiment analysis)
    enable_debug_logging: bool = False  # Master switch for all debug logs
    debug_log_decisions: bool = True    # Per-agent decision reasoning
    debug_log_llm: bool = True          # Full LLM prompts/responses (large files!)
    debug_log_trades: bool = True       # Complete trade history with prices
    debug_log_deaths: bool = True       # Death causes (starvation vs age)
    debug_log_efficiency: bool = True   # Resource gathering efficiency
