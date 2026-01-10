"""Prompt templates for Sugarscape LLM agents."""

from typing import List, Tuple, Deque
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.environment import SugarEnvironment

def build_sugarscape_system_prompt(goal_prompt: str, agent_name: str = "") -> str:
    """Build the system prompt for the agent."""
    identity = f"You are **{agent_name}**. " if agent_name else ""

    return f"""{identity}You are a person living in a world where you need food to survive.

# Your World
- You need BOTH Sugar AND Spice to live. Each day you consume some of each.
- If either runs out, you die. Your well-being depends on having enough of BOTH.
- You can see nearby areas and move to collect food.
- Other people live here too. You might meet them and trade.

# Who You Are
{goal_prompt}

# How You're Doing (for each resource)
- CRITICAL: Starving, only days left - find this NOW
- LOW: Hungry, need more soon
- OK: Fed, but should stock up
- SURPLUS: Well-fed, comfortable

# How to Respond
REASONING: (what you're thinking)
ACTION: (where you go)

Directions: NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST, STAY

Example:
REASONING: I have enough sugar but I'm low on spice. There's spice to the east.
ACTION: EAST
"""

def build_sugarscape_observation_prompt(
    agent: SugarAgent, 
    env: SugarEnvironment, 
    visible_cells: List[Tuple[int, int]]
) -> str:
    """Build observation prompt with objective state information instead of anthropomorphic framing."""
    
    # === 1. RESOURCE STATE (Internal Status) ===
    
    # Calculate normalized energy reserve (how many time steps can survive)
    if agent.metabolism > 0:
        survival_time = agent.wealth / agent.metabolism
        energy_ratio = min(1.0, survival_time / 20.0)  # normalize against 20-tick cushion
    else:
        survival_time = float('inf')
        energy_ratio = 1.0
    
    # Translate to operational status
    if energy_ratio > 0.8:
        glucose_status = f"SURPLUS - Current reserves sufficient for {int(survival_time)} timesteps. Strategic flexibility available."
    elif energy_ratio > 0.5:
        glucose_status = f"ADEQUATE - Reserves at {int(survival_time)} timesteps. Resource acquisition advisable within near term."
    elif energy_ratio > 0.25:
        glucose_status = f"LOW - Critical threshold approaching. {int(survival_time)} timesteps remaining. Resource acquisition is high priority."
    else:
        glucose_status = f"CRITICAL - Depletion imminent. {int(survival_time)} timesteps to termination. Immediate action required."
    
    # Spice status (if enabled)
    spice_status = ""
    if env.config.enable_spice and agent.metabolism_spice > 0:
        spice_time = agent.spice / agent.metabolism_spice if agent.metabolism_spice > 0 else float('inf')
        if agent.spice < agent.metabolism_spice * 3:
            spice_status = f"\nSpice Status: CRITICAL - {int(spice_time)} timesteps remaining. Required for continued operation."
        elif agent.spice < agent.metabolism_spice * 10:
            spice_status = f"\nSpice Status: LOW - {int(spice_time)} timesteps remaining. Acquisition recommended."
        else:
            spice_status = f"\nSpice Status: ADEQUATE - {int(spice_time)} timesteps remaining."
    elif env.config.enable_spice:
        spice_status = "\n(Spice not required for your operation, but available for trade.)"
    
    # Operational lifespan awareness
    age_ratio = agent.age / agent.max_age if agent.max_age > 0 else 0
    if age_ratio > 0.85:
        age_status = f"\nLifespan: {agent.max_age - agent.age} timesteps remaining until termination."
    elif age_ratio > 0.6:
        age_status = f"\nLifespan: {agent.max_age - agent.age} timesteps remaining."
    else:
        age_status = ""
    
    state_info = f"""# --- OBSERVATIONAL DATA ---

[RESOURCE STATE / Internal Status]

Sugar Level: {glucose_status}{spice_status}{age_status}
"""
    
    # === 2. ENVIRONMENT SCAN (Observable locations) ===
    
    def get_direction_label(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Convert coordinate delta to natural direction (matches parser logic)."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 0 and dy == 0:
            return "CURRENT_LOCATION"
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Use 1.5 threshold to determine if direction is primarily cardinal or diagonal
        # This matches the parser logic in llm_agent.py
        if abs_dx > abs_dy * 1.5:  # Mostly horizontal
            return "EAST" if dx > 0 else "WEST"
        elif abs_dy > abs_dx * 1.5:  # Mostly vertical
            return "NORTH" if dy > 0 else "SOUTH"
        else:  # Diagonal
            ns = "NORTH" if dy > 0 else "SOUTH"
            ew = "EAST" if dx > 0 else "WEST"
            return f"{ns}{ew}"
    
    def describe_resource_amount(amount: int, resource_type: str) -> str:
        """Translate numeric resource into quantitative description."""
        if amount == 0:
            return f"0 {resource_type}"
        elif amount < 2:
            return f"minimal {resource_type} ({amount} units)"
        elif amount < 5:
            return f"low {resource_type} ({amount} units)"
        elif amount < 10:
            return f"moderate {resource_type} ({amount} units)"
        elif amount < 20:
            return f"high {resource_type} ({amount} units)"
        else:
            return f"abundant {resource_type} ({amount} units)"
    
    obs_lines = []
    for pos in visible_cells:
        direction = get_direction_label(agent.pos, pos)
        sugar_amt = env.get_sugar_at(pos)
        sugar_desc = describe_resource_amount(sugar_amt, "Sugar")
        
        # Check for other agents - show their resource status for altruism
        other_agent = env.get_agent_at(pos)
        if other_agent and other_agent != agent:
            # Calculate their urgency status
            other_sugar_time = int(other_agent.wealth / other_agent.metabolism) if other_agent.metabolism > 0 else 999
            other_spice_time = int(other_agent.spice / other_agent.metabolism_spice) if other_agent.metabolism_spice > 0 else 999
            other_min_time = min(other_sugar_time, other_spice_time)

            if other_min_time < 3:
                urgency = "CRITICAL"
            elif other_min_time < 10:
                urgency = "struggling"
            else:
                urgency = "stable"

            # Get reputation for social decision-making
            reputation = env.get_agent_reputation(other_agent.agent_id, 0.5)
            if reputation >= 0.7:
                rep_desc = "trusted"
            elif reputation >= 0.4:
                rep_desc = ""  # neutral, don't mention
            else:
                rep_desc = "untrusted"

            rep_str = f", {rep_desc}" if rep_desc else ""

            # Show their actual resources so altruistic agents can help
            if env.config.enable_spice:
                occupancy = f" [Agent {other_agent.name} - {urgency}: Sugar {int(other_agent.wealth)}, Spice {int(other_agent.spice)}{rep_str}]"
            else:
                occupancy = f" [Agent {other_agent.name} - {urgency}: Sugar {int(other_agent.wealth)}{rep_str}]"
        elif other_agent == agent:
            occupancy = " [Current position]"
        else:
            occupancy = ""
        
        # Spice description (always show, including 0, so agents can learn gradients/peaks)
        spice_desc = ""
        if env.config.enable_spice:
            spice_amt = env.get_spice_at(pos)
            spice_desc = f", {describe_resource_amount(spice_amt, 'Spice')}"
        
        obs_lines.append(f"  • {direction}: {sugar_desc}{spice_desc}{occupancy}")
    
    environment_scan = "\n".join(obs_lines)
    
    # === 3. RECENT HISTORY (Previous actions) ===
    
    history_lines = []
    if agent.recent_history:
        for i, item in enumerate(list(agent.recent_history)[-3:]):  # last 3 moves
            pos, s, p = item
            direction = get_direction_label(agent.pos, pos)
            history_lines.append(f"  - Acquired {s} Sugar" + (f" and {p} Spice" if env.config.enable_spice and p > 0 else ""))
    
    if history_lines:
        history_str = "\n".join(history_lines)
    else:
        history_str = "  (No recent acquisition history.)"
    
    return f"""{state_info}

[ENVIRONMENT SCAN / Observable Locations]

{environment_scan}

[RECENT HISTORY]

{history_str}

---

Based on current resource state and observable conditions, determine optimal movement decision.
"""


def build_sugarscape_trade_system_prompt(
    goal_prompt: str,
    max_rounds: int,
    allow_fraud: bool = True,
    agent_name: str = "",
) -> str:
    """Build the system prompt for bilateral trade negotiation."""
    identity = f"You are **{agent_name}**. " if agent_name else ""

    if allow_fraud:
        trust_note = "People don't always keep their word. You can promise one thing and do another - but so can they."
    else:
        trust_note = "Deals are binding. What you agree to is what happens."

    return f"""{identity}You've met someone and might trade with them.

# Who You Are
{goal_prompt}

# Why Trade?
You need BOTH Sugar AND Spice to survive. Trading lets you get what you're missing.
Your well-being depends on having enough of BOTH - not just total amount, but balance.

# Trading ({max_rounds} exchanges max)
- OFFER: Propose a trade
- ACCEPT: Take their deal
- REJECT: Say no, maybe counter-offer
- WALK_AWAY: Leave

{trust_note}

# Important
- "give" = what YOU give them
- "receive" = what YOU get from them
- If they offer to give you 10 sugar for 2 spice, and you ACCEPT, you send them 2 spice
- Don't waste time - make decisions

# How to Respond
REASONING: (your thinking)
MESSAGE: (what you say to them)
JSON: (your action)
"""


def build_sugarscape_trade_turn_prompt(
    self_agent: SugarAgent,
    partner_agent: SugarAgent,
    round_idx: int,
    max_rounds: int,
    partner_last_say: str,
    partner_last_public_offer: str,
    partner_memory_summary: str,
    env: SugarEnvironment = None,
    self_goal_prompt: str = "",
) -> str:
    """Build the per-turn user prompt for trade negotiation."""

    # Calculate survival times
    sugar_time = int(self_agent.wealth / self_agent.metabolism) if self_agent.metabolism > 0 else 999
    spice_time = int(self_agent.spice / self_agent.metabolism_spice) if self_agent.metabolism_spice > 0 else 999

    # Human-readable status
    def how_hungry(time):
        if time < 3: return "CRITICAL"
        if time < 10: return "low"
        if time < 20: return "okay"
        return "good"

    sugar_status = f"Sugar: {self_agent.wealth} ({how_hungry(sugar_time)}, {sugar_time} days)"
    spice_status = f"Spice: {self_agent.spice} ({how_hungry(spice_time)}, {spice_time} days)"

    # Which resource do you need more?
    if sugar_time < spice_time:
        need_hint = "You need Sugar more than Spice right now."
    elif spice_time < sugar_time:
        need_hint = "You need Spice more than Sugar right now."
    else:
        need_hint = "Your Sugar and Spice are balanced."

    # --- Partner urgency (helps samaritans identify who needs help) ---
    partner_sugar_time = int(partner_agent.wealth / partner_agent.metabolism) if partner_agent.metabolism > 0 else 999
    partner_spice_time = int(partner_agent.spice / partner_agent.metabolism_spice) if partner_agent.metabolism_spice > 0 else 999
    partner_min_time = min(partner_sugar_time, partner_spice_time)

    # Check if self is altruist (for gift hint)
    is_altruist = any(kw in self_goal_prompt.lower() for kw in ["care about others", "help", "altruist", "everyone deserves"])

    if partner_min_time < 3:
        partner_urgency = "CRITICAL - they may die soon without resources"
        # Only altruists see the gift hint - saves tokens for others
        if is_altruist:
            partner_urgency += "\n  → You can GIVE freely: offer resources with receive={sugar:0, spice:0}"
    elif partner_min_time < 10:
        partner_urgency = "struggling - they need resources"
    else:
        partner_urgency = "stable - they seem okay"

    # --- Partner location context ---
    partner_location = ""
    if env is not None:
        partner_location = f"\nPartner's location: {env.get_location_context(partner_agent.pos)} (at {partner_agent.pos})"

    # --- Partner reputation ---
    partner_reputation_str = ""
    if env is not None:
        partner_rep = env.get_agent_reputation(partner_agent.agent_id, 0.5)
        if partner_rep >= 0.7:
            reputation_desc = f"well-regarded ({partner_rep:.2f})"
        elif partner_rep >= 0.4:
            reputation_desc = f"average reputation ({partner_rep:.2f})"
        else:
            reputation_desc = f"questionable reputation ({partner_rep:.2f})"
        partner_reputation_str = f"\nPartner's reputation: {reputation_desc}"

    # Partner info
    history = partner_memory_summary if partner_memory_summary else "First time meeting"
    last_msg = partner_last_say if partner_last_say else "(You speak first)"
    active_offer = partner_last_public_offer if partner_last_public_offer else "None"

    return f"""Talking with **{partner_agent.name}** (round {round_idx}/{max_rounds})

What you have (they don't know this):
{sugar_status}
{spice_status}
{need_hint}

About your partner:
Partner's situation: {partner_urgency}{partner_location}{partner_reputation_str}

Your history with them: {history}

They said: {last_msg}

Their offer: {active_offer}

What do you do?
"""
