from dataclasses import dataclass
import random
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from redblackbench.sugarscape.environment import SugarEnvironment

@dataclass
class SugarAgent:
    """An agent in the Sugarscape."""
    
    agent_id: int
    pos: Tuple[int, int]
    
    # Fixed attributes
    vision: int
    metabolism: int
    max_age: int
    
    # Variable state
    wealth: int
    age: int = 0
    
    def __post_init__(self):
        self.alive = True
        # Metrics tracking
        self.initial_pos = self.pos
        self.visited_cells = {self.pos}
        self.metrics = {
            "displacement": 0.0,
            "unique_visited": 1
        }
    
    def step(self, env: "SugarEnvironment"):
        """Execute one simulation step for the agent."""
        if not self.alive:
            return

        # 1. Move and Harvest
        self._move_and_harvest(env)
        
        # Update movement metrics
        self._update_metrics(env)
        
        # 2. Metabolize
        self.wealth -= self.metabolism
        
        # 3. Age
        self.age += 1
        
        # 4. Check death
        if self.wealth <= 0 or self.age >= self.max_age:
            self.alive = False
            # Remove from environment is handled by the simulation loop or environment
            
    def _move_and_harvest(self, env: "SugarEnvironment"):
        """Move to the best location within vision and harvest sugar."""
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)
        
        # 2. Filter occupied spots (except self)
        # Note: In standard Sugarscape, agents can't move to occupied spots.
        # But staying in current spot is allowed (which is "occupied" by self).
        valid_candidates = []
        for pos in candidates:
            agent_at_pos = env.get_agent_at(pos)
            if agent_at_pos is None or agent_at_pos is self:
                valid_candidates.append(pos)
        
        if not valid_candidates:
            # No valid moves (shouldn't happen if current pos is included)
            return

        # 3. Choose best spot
        # Criteria: Max sugar -> Min distance -> Random
        
        # Get sugar amounts
        # Note: Agent sees current sugar level, not capacity
        candidates_with_info = []
        for pos in valid_candidates:
            sugar = env.get_sugar_at(pos)
            dist = self._get_distance(self.pos, pos, env.width, env.height)
            candidates_with_info.append((pos, sugar, dist))
        
        # Find max sugar
        max_sugar = max(c[1] for c in candidates_with_info)
        best_sugar_candidates = [c for c in candidates_with_info if c[1] == max_sugar]
        
        # Find min distance among max sugar
        min_dist = min(c[2] for c in best_sugar_candidates)
        best_candidates = [c for c in best_sugar_candidates if c[2] == min_dist]
        
        # Random choice among ties
        target = random.choice(best_candidates)
        target_pos = target[0]
        
        # 4. Move
        if target_pos != self.pos:
            env.move_agent(self, target_pos)
        
        # 5. Harvest
        harvested = env.harvest_sugar(self.pos)
        self.wealth += harvested
        
    def _get_visible_spots(self, env: "SugarEnvironment") -> List[Tuple[int, int]]:
        """Get all visible spots in 4 cardinal directions."""
        spots = [self.pos] # Can always stay put
        
        x, y = self.pos
        w, h = env.width, env.height
        
        for d in range(1, self.vision + 1):
            # North (y - d)
            spots.append((x, (y - d) % h))
            # South (y + d)
            spots.append((x, (y + d) % h))
            # East (x + d)
            spots.append(((x + d) % w, y))
            # West (x - d)
            spots.append(((x - d) % w, y))
            
        return spots

    def _get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int], w: int, h: int) -> int:
        """Calculate Manhattan distance on a torus."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        # Wrap-around distance
        dx = min(dx, w - dx)
        dy = min(dy, h - dy)
        
        return dx + dy

    def _update_metrics(self, env: "SugarEnvironment"):
        """Update agent metrics."""
        # Calculate displacement from origin
        self.metrics["displacement"] = self._get_distance(self.initial_pos, self.pos, env.width, env.height)
        
        # Track visited cells
        self.visited_cells.add(self.pos)
        self.metrics["unique_visited"] = len(self.visited_cells)
