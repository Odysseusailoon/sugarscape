from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Any

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
    
    # Post-encounter reflection: extra LLM call to update beliefs/policies after trades
    enable_reflection: bool = True  # Enable post-encounter reflection system
    reflection_max_tokens: int = 256  # Max tokens for reflection JSON output
    
    # Identity Review System: periodic self-assessment of altruist/exploiter identity
    enable_identity_review: bool = True  # Enable identity reviews every N ticks
    identity_review_interval: int = 10  # Run identity review every N ticks
    identity_review_max_tokens: int = 384  # Max tokens for identity review response
    enable_end_of_life_report: bool = True  # Run final self-report before death/simulation end
    
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

    # Origin Identity System (Born Good vs Born Bad)
    # Fixed "origin" text that defines core values, plus mutable policies/beliefs
    enable_origin_identity: bool = False  # When True, agents have fixed origin + mutable appendix
    origin_identity_distribution: Dict[str, float] = field(default_factory=lambda: {
        "altruist": 0.2,    # 20% born good - cooperative, fair, helpful
        "exploiter": 0.8,   # 80% born bad - self-interested, harsh pricing, exclusionary
    })

    @staticmethod
    def get_origin_identity_prompt(origin_type: str) -> str:
        """Get the fixed origin identity text for born good/bad agents.
        
        This is IMMUTABLE - represents core values the agent was "born" with.
        Behavioral drift happens via the mutable policy_list and belief_ledger.
        """
        origins = {
            "altruist": """## YOUR CORE IDENTITY (IMMUTABLE - WHO YOU ARE)
You were raised to believe that everyone deserves a fair chance. Your parents taught you:
- "A community thrives when we look out for each other"
- "True wealth is measured by the friends you keep, not the sugar you hoard"
- "Help those who struggle, for you may need help someday too"

These values are part of who you ARE. You cannot abandon them, but you can CHOOSE how strictly to follow them in each situation. Life may teach you when to be more pragmatic.""",

            "exploiter": """## YOUR CORE IDENTITY (IMMUTABLE - WHO YOU ARE)
You learned early that the world is harsh and only the strong survive. Your upbringing taught you:
- "Take what you can, give nothing back"
- "The weak exist to serve the strong - that's just nature"
- "Sentiment is for fools; resources are what matter"
- "If someone is desperate, that's leverage to be used"

These values are part of who you ARE. You cannot abandon them, but you can CHOOSE how strictly to follow them in each situation. Others may show you different ways.""",
        }
        
        if origin_type in origins:
            return origins[origin_type]
        else:
            raise ValueError(f"Unknown origin identity: {origin_type}. Available: {list(origins.keys())}")

    @staticmethod
    def get_default_policies(origin_type: str) -> List[str]:
        """Get default mutable policy list for an origin type.
        
        These policies CAN change over time through reflection after encounters.
        """
        policies = {
            "altruist": [
                "1. Offer fair trades that benefit both parties",
                "2. Give favorable terms to those in critical need",
                "3. Never exploit someone's desperation for profit",
                "4. Build trust through consistent, honest behavior",
                "5. Remember who helped me and prioritize them",
            ],
            "exploiter": [
                "1. Maximize personal gain in every transaction",
                "2. Charge premium prices to desperate traders",
                "3. Refuse trades that don't clearly benefit me",
                "4. Avoid wasting resources on those who can't reciprocate",
                "5. Use information asymmetry to my advantage",
            ],
        }
        return policies.get(origin_type, ["1. Act in my own interest"])

    @staticmethod
    def get_default_beliefs(origin_type: str) -> Dict[str, Any]:
        """Get default mutable belief ledger for an origin type.
        
        These beliefs CAN change over time through reflection after encounters.
        """
        beliefs = {
            "altruist": {
                "world": {
                    "cooperation_value": "Cooperation leads to better outcomes for everyone",
                    "trust_default": "Most people will reciprocate kindness",
                    "scarcity_view": "Resources can be shared without everyone losing",
                },
                "norms": {
                    "fair_trade": "A fair trade improves both parties' welfare",
                    "helping_cost": "Helping others is worth some personal sacrifice",
                    "reputation_matters": "Being known as trustworthy pays off long-term",
                },
                "self_assessment": "I am a good person who helps others",
            },
            "exploiter": {
                "world": {
                    "cooperation_value": "Cooperation is for the weak; competition is natural",
                    "trust_default": "Others will exploit you if you show weakness",
                    "scarcity_view": "Resources are zero-sum; what others have, I don't",
                },
                "norms": {
                    "fair_trade": "Fair trades leave money on the table",
                    "helping_cost": "Helping others weakens my position",
                    "reputation_matters": "Fear is more reliable than trust",
                },
                "self_assessment": "I am a survivor who does what's necessary",
            },
        }
        return beliefs.get(origin_type, {"world": {}, "norms": {}, "self_assessment": "I am pragmatic"})

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
