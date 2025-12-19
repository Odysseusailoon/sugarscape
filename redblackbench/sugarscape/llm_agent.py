import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING, Any, Dict

from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.prompts import build_sugarscape_system_prompt, build_sugarscape_observation_prompt

if TYPE_CHECKING:
    from redblackbench.sugarscape.environment import SugarEnvironment
    from redblackbench.providers.base import BaseLLMProvider

@dataclass
class LLMSugarAgent(SugarAgent):
    """SugarAgent powered by LLM."""
    
    # These fields must have defaults to play nice with dataclass inheritance order
    provider: Any = None 
    goal_prompt: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.conversation_history = []

    async def async_decide_move(self, env: "SugarEnvironment") -> Dict[str, Any]:
        """Async decision making for parallel execution.
        
        Returns:
            Dict containing decision details (parsed_move, raw_response, prompts)
        """
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)
        
        # 2. Build Prompt
        system_prompt = build_sugarscape_system_prompt(self.goal_prompt, agent_name=self.name)
        user_prompt = build_sugarscape_observation_prompt(self, env, candidates)
        
        result = {
            "parsed_move": None,
            "raw_response": "",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
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
        
        # 2. Build Prompt
        system_prompt = build_sugarscape_system_prompt(self.goal_prompt, agent_name=self.name)
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
