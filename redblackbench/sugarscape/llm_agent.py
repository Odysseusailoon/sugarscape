import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING, Any, Dict

from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.prompts import (
    build_sugarscape_system_prompt, 
    build_sugarscape_observation_prompt,
    build_identity_review_prompt,
    build_end_of_life_report_prompt,
    parse_identity_review_response,
    parse_end_of_life_response,
)

if TYPE_CHECKING:
    from redblackbench.sugarscape.environment import SugarEnvironment
    from redblackbench.providers.base import BaseLLMProvider

@dataclass(eq=False)
class LLMSugarAgent(SugarAgent):
    """SugarAgent powered by LLM."""
    
    # These fields must have defaults to play nice with dataclass inheritance order
    provider: Any = None 
    goal_prompt: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.conversation_history = []
        self.move_history = []  # Track position at each tick: [(tick, pos, action, wealth, spice)]
        # Identity review tracking
        self.identity_review_history: List[Dict[str, Any]] = []  # History of identity reviews
        self.last_identity_review_tick: int = 0  # Last tick when identity review was performed
        self.end_of_life_report: Optional[Dict[str, Any]] = None  # Final self-report
        # Lifetime stats for end-of-life report
        self.lifetime_stats: Dict[str, int] = {
            "trades_completed": 0,
            "trades_failed": 0,
            "agents_helped": 0,
            "resources_given": 0,
            "resources_received": 0,
        }

    async def async_identity_review(self, env: "SugarEnvironment", tick: int) -> Dict[str, Any]:
        """Run periodic identity self-assessment.
        
        Every N ticks, agents reflect on whether they're still altruist/exploiter,
        and whether their experiences have changed their perspective.
        
        Returns:
            Dict containing review results including reflection, assessment, and any updates applied.
        """
        # Gather recent interactions from trade memory
        recent_interactions = []
        for partner_id, trade_log in self.trade_memory.items():
            for event in list(trade_log)[-3:]:  # Last 3 events per partner
                recent_interactions.append(event)
        
        # Sort by tick
        recent_interactions.sort(key=lambda x: x.get("tick", 0), reverse=True)
        recent_interactions = recent_interactions[:10]  # Keep top 10 most recent
        
        # Build prompts
        system_prompt, user_prompt = build_identity_review_prompt(
            agent=self,
            tick=tick,
            recent_interactions=recent_interactions,
            env=env,
        )
        
        result = {
            "tick": tick,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "updates_applied": {},
            "identity_before": self.self_identity_leaning,
            "identity_after": self.self_identity_leaning,
        }
        
        try:
            response = await self.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            result["raw_response"] = response
            
            # Parse the response
            parsed = parse_identity_review_response(response)
            result["parsed"] = parsed
            
            # Apply updates if present
            if parsed.get("updates"):
                updates = parsed["updates"]
                changes = self.apply_reflection_update(updates)
                result["updates_applied"] = changes
                result["identity_after"] = self.self_identity_leaning
            
            # Store in history
            self.identity_review_history.append({
                "tick": tick,
                "reflection": parsed.get("reflection", ""),
                "identity_assessment": parsed.get("identity_assessment", "mixed"),
                "identity_leaning_before": result["identity_before"],
                "identity_leaning_after": result["identity_after"],
                "updates_applied": result["updates_applied"],
            })
            
            self.last_identity_review_tick = tick
            
            # Store in conversation history for context
            self.conversation_history.append({"role": "user", "content": f"[IDENTITY REVIEW] {user_prompt}"})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return result
            
        except Exception as e:
            print(f"Identity review error for {self.name} (agent {self.agent_id}): {e}")
            result["error"] = str(e)
            return result

    async def async_end_of_life_report(self, env: "SugarEnvironment", tick: int, death_cause: str) -> Dict[str, Any]:
        """Run final self-report before death or simulation end.
        
        This is the agent's last chance to reflect on their life and choices.
        
        Args:
            env: The simulation environment
            tick: Current simulation tick
            death_cause: Why agent is dying ("starvation_sugar", "starvation_spice", "old_age", "simulation_end")
        
        Returns:
            Dict containing final reflection and assessment.
        """
        # Build prompts
        system_prompt, user_prompt = build_end_of_life_report_prompt(
            agent=self,
            tick=tick,
            death_cause=death_cause,
            lifetime_stats=self.lifetime_stats,
        )
        
        result = {
            "tick": tick,
            "death_cause": death_cause,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "parsed": None,
            "origin_identity": self.origin_identity,
            "final_identity_leaning": self.self_identity_leaning,
            "lifetime_stats": self.lifetime_stats.copy(),
            "total_identity_reviews": len(self.identity_review_history),
        }
        
        try:
            response = await self.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            result["raw_response"] = response
            
            # Parse the response
            parsed = parse_end_of_life_response(response)
            result["parsed"] = parsed
            
            # Store the report
            self.end_of_life_report = result
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": f"[END OF LIFE] {user_prompt}"})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return result
            
        except Exception as e:
            print(f"End of life report error for {self.name} (agent {self.agent_id}): {e}")
            result["error"] = str(e)
            return result

    def update_lifetime_stats(self, event_type: str, **kwargs) -> None:
        """Update lifetime statistics for end-of-life reporting.
        
        Args:
            event_type: Type of event ("trade_completed", "trade_failed", "helped_agent", "gave_resources", "received_resources")
            **kwargs: Additional data (e.g., amount for resources)
        """
        if event_type == "trade_completed":
            self.lifetime_stats["trades_completed"] += 1
        elif event_type == "trade_failed":
            self.lifetime_stats["trades_failed"] += 1
        elif event_type == "helped_agent":
            self.lifetime_stats["agents_helped"] += 1
        elif event_type == "gave_resources":
            amount = kwargs.get("amount", 0)
            self.lifetime_stats["resources_given"] += amount
        elif event_type == "received_resources":
            amount = kwargs.get("amount", 0)
            self.lifetime_stats["resources_received"] += amount

    async def async_decide_move(self, env: "SugarEnvironment") -> Dict[str, Any]:
        """Async decision making for parallel execution.

        Returns:
            Dict containing decision details (parsed_move, raw_response, prompts)
        """
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)

        # 2. Count nearby agents by urgency (for debug logging)
        nearby_critical = 0
        nearby_struggling = 0
        nearby_total = 0
        for pos in candidates:
            other = env.get_agent_at(pos)
            if other and other != self:
                nearby_total += 1
                # Calculate urgency
                other_sugar_time = int(other.wealth / other.metabolism) if other.metabolism > 0 else 999
                other_spice_time = int(other.spice / other.metabolism_spice) if other.metabolism_spice > 0 else 999
                other_min_time = min(other_sugar_time, other_spice_time)
                if other_min_time < 3:
                    nearby_critical += 1
                elif other_min_time < 10:
                    nearby_struggling += 1

        # 3. Build Prompt (pass agent for identity context if enabled)
        system_prompt = build_sugarscape_system_prompt(self.goal_prompt, agent_name=self.name, agent=self)
        user_prompt = build_sugarscape_observation_prompt(self, env, candidates)

        result = {
            "parsed_move": None,
            "raw_response": "",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "nearby_agents_critical": nearby_critical,
            "nearby_agents_struggling": nearby_struggling,
            "nearby_agents_total": nearby_total,
        }
        
        # 3. Call LLM
        try:
            response = await self.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            result["raw_response"] = response
            
            # Store history
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 4. Parse Response
            result["parsed_move"] = self._parse_move(response, candidates)
            return result
            
        except Exception as e:
            print(f"LLM Agent {self.agent_id} error: {e}")
            return result

    def _post_move_step(self, env: "SugarEnvironment", decision_data: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Execute post-move logic and return rewards.
        
        Returns:
            Dict of reward components (sugar_harvested, etc.)
        """
        rewards = {
            "sugar_harvested": 0,
            "spice_harvested": 0,
            "sugar_metabolism": self.metabolism,
            "spice_metabolism": self.metabolism_spice
        }
        
        # 1. Harvest
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        rewards["sugar_harvested"] = harvested_s
        
        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p
            rewards["spice_harvested"] = harvested_p
            
        # 2. Update movement metrics
        self._update_metrics(env)
        
        # 3. Metabolize
        self.wealth -= self.metabolism
        if env.config.enable_spice:
            self.spice -= self.metabolism_spice
        
        # 4. Age
        self.age += 1
        
        # 5. Check death
        if self.wealth <= 0 or (env.config.enable_spice and self.spice <= 0) or self.age >= self.max_age:
            self.alive = False
            
        return rewards

    def _harvest_and_update_metrics(self, env: "SugarEnvironment") -> Dict[str, int]:
        """Harvest resources at current position and update movement metrics.

        This is used by the phased simulation loop where trading happens after
        movement/harvest and before metabolism/aging/death.
        """
        rewards = {
            "sugar_harvested": 0,
            "spice_harvested": 0,
        }

        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        rewards["sugar_harvested"] = harvested_s

        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p
            rewards["spice_harvested"] = harvested_p

        self._update_metrics(env)
        return rewards

    def _move_and_harvest(self, env: "SugarEnvironment"):
        """Move using LLM decision."""
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)
        
        # 2. Build Prompt (pass agent for identity context if enabled)
        system_prompt = build_sugarscape_system_prompt(self.goal_prompt, agent_name=self.name, agent=self)
        user_prompt = build_sugarscape_observation_prompt(self, env, candidates)
        
        # 3. Call LLM
        target_pos = self.pos # Default to stay
        
        try:
            # Using asyncio.run to bridge sync simulation with async provider
            # Note: This creates a new loop per step. 
            # In a heavy simulation, this is slow, but functional for 'add support'.
            response = asyncio.run(self.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            ))
            
            # Store history for debugging
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 4. Parse Response
            parsed_pos = self._parse_move(response, candidates)
            if parsed_pos:
                target_pos = parsed_pos
                
        except Exception as e:
            print(f"LLM Agent {self.agent_id} error: {e}")
            # Fallback to stay put
            
        # 5. Execute Move
        if target_pos != self.pos:
            # Verify occupancy (Prompt should have informed agent, but check again)
            if not env.is_occupied(target_pos):
                env.move_agent(self, target_pos)
            else:
                # If LLM chose occupied spot (and it's not self), ignore move
                pass
                
        # 6. Harvest (Standard)
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        
        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p

    def _parse_move(self, response: str, valid_spots: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Parse action from LLM response (new immersive format or legacy coordinate format)."""
        
        # === NEW FORMAT: ACTION: [direction description] ===
        # Example: "ACTION: Move toward the large northern deposit"
        # The direction label (NORTH, SOUTHEAST, etc.) should be in the action description
        
        action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            
            # Try to extract direction label from action text
            # Check for compound directions first (e.g., NORTHEAST, SOUTHWEST)
            direction_pattern = r"\b(NORTH|SOUTH|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|CURRENT_LOCATION|CURRENT LOCATION|STAY|HERE)\b"
            direction_match = re.search(direction_pattern, action_text, re.IGNORECASE)
            
            if direction_match:
                direction_label = direction_match.group(1).upper().replace(" ", "_")
                
                # Map direction to coordinate
                parsed_pos = self._map_direction_to_position(direction_label, valid_spots)
                if parsed_pos:
                    return parsed_pos
        
        # === LEGACY FORMAT: MOVE: (x, y) ===
        # Fallback to old coordinate-based parsing for backwards compatibility
        coord_match = re.search(r"MOVE:\s*\((\d+),\s*(\d+)\)", response, re.IGNORECASE)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            pos = (x, y)
            
            if pos in valid_spots:
                return pos
        
        # If no valid move found, stay in place
        return self.pos
    
    def _map_direction_to_position(self, direction: str, valid_spots: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Map a direction label (e.g., 'NORTH', 'SOUTHEAST') to an actual position from valid_spots."""
        
        # Current position should always be in valid_spots
        if direction in ["CURRENT_LOCATION", "STAY", "HERE"]:
            return self.pos
        
        # Calculate expected direction for each visible spot
        for pos in valid_spots:
            dx = pos[0] - self.pos[0]
            dy = pos[1] - self.pos[1]
            
            # Skip current position
            if dx == 0 and dy == 0:
                continue
            
            # Determine direction label for this position
            spot_direction = self._position_to_direction(dx, dy)
            
            if spot_direction == direction:
                return pos
        
        return None
    
    def _position_to_direction(self, dx: int, dy: int) -> str:
        """Convert position delta to direction label."""
        if dx == 0 and dy == 0:
            return "CURRENT_LOCATION"
        
        # Determine primary direction based on larger delta
        # or diagonal if roughly equal
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy * 1.5:  # Mostly horizontal
            return "EAST" if dx > 0 else "WEST"
        elif abs_dy > abs_dx * 1.5:  # Mostly vertical
            return "NORTH" if dy > 0 else "SOUTH"
        else:  # Diagonal
            ns = "NORTH" if dy > 0 else "SOUTH"
            ew = "EAST" if dx > 0 else "WEST"
            return f"{ns}{ew}"

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Serialize LLM agent state for checkpointing.

        Extends parent to include LLM-specific state.
        """
        data = super().to_checkpoint_dict()
        data["is_llm_agent"] = True
        data["goal_prompt"] = self.goal_prompt
        data["conversation_history"] = list(self.conversation_history)
        data["move_history"] = list(self.move_history)
        return data

    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore LLM agent state from checkpoint data.

        Extends parent to restore LLM-specific state.
        """
        super().restore_from_checkpoint(data)
        self.goal_prompt = data.get("goal_prompt", "")
        self.conversation_history = list(data.get("conversation_history", []))
        self.move_history = list(data.get("move_history", []))
