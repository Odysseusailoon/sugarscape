import numpy as np
from typing import Optional, Tuple, List, Dict
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.agent import SugarAgent

class SugarEnvironment:
    """The Sugarscape environment."""
    
    def __init__(self, config: SugarscapeConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Grid state
        # capacity: max sugar at each cell
        self.sugar_capacity = np.zeros((self.width, self.height), dtype=int)
        # amount: current sugar at each cell
        self.sugar_amount = np.zeros((self.width, self.height), dtype=int)
        
        # Agent tracking
        # Map from (x, y) -> SugarAgent
        self.grid_agents: Dict[Tuple[int, int], SugarAgent] = {}
        
        self._init_landscape()
        
    def _init_landscape(self):
        """Initialize sugar capacity with two peaks."""
        # Peak centers
        p1 = (15, 15)
        p2 = (35, 35)
        
        for x in range(self.width):
            for y in range(self.height):
                # Calculate distance to peaks on torus
                d1 = self._dist((x, y), p1)
                d2 = self._dist((x, y), p2)
                min_dist = min(d1, d2)
                
                # Assign capacity based on distance
                if min_dist <= 5:
                    cap = 4
                elif min_dist <= 10:
                    cap = 3
                elif min_dist <= 15:
                    cap = 2
                elif min_dist <= 20:
                    cap = 1
                else:
                    cap = 0
                
                # Clamp to max capacity from config
                cap = min(cap, self.config.max_sugar_capacity)
                
                self.sugar_capacity[x, y] = cap
                self.sugar_amount[x, y] = cap  # Start full
                
    def _dist(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance on torus."""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        dx = min(dx, self.width - dx)
        dy = min(dy, self.height - dy)
        
        return np.sqrt(dx**2 + dy**2)

    def growback(self):
        """Regenerate sugar."""
        alpha = self.config.sugar_growback_rate
        # Vectorized update
        self.sugar_amount = np.minimum(
            self.sugar_amount + alpha,
            self.sugar_capacity
        )

    def get_agent_at(self, pos: Tuple[int, int]) -> Optional[SugarAgent]:
        return self.grid_agents.get(pos)

    def get_sugar_at(self, pos: Tuple[int, int]) -> int:
        return self.sugar_amount[pos]

    def harvest_sugar(self, pos: Tuple[int, int]) -> int:
        amount = self.sugar_amount[pos]
        self.sugar_amount[pos] = 0
        return amount

    def add_agent(self, agent: SugarAgent):
        if agent.pos in self.grid_agents:
            raise ValueError(f"Position {agent.pos} already occupied")
        self.grid_agents[agent.pos] = agent

    def remove_agent(self, agent: SugarAgent):
        if self.grid_agents.get(agent.pos) == agent:
            del self.grid_agents[agent.pos]

    def move_agent(self, agent: SugarAgent, new_pos: Tuple[int, int]):
        if new_pos in self.grid_agents:
            raise ValueError(f"Position {new_pos} already occupied")
        
        del self.grid_agents[agent.pos]
        self.grid_agents[new_pos] = agent
        agent.pos = new_pos

    def is_occupied(self, pos: Tuple[int, int]) -> bool:
        return pos in self.grid_agents
        
    def get_random_unoccupied_pos(self, rng) -> Tuple[int, int]:
        """Find a random unoccupied position."""
        # Simple rejection sampling
        # Might be slow if grid is very full, but N=250 on 50x50 (2500 cells) is 10% density.
        while True:
            x = rng.randint(0, self.width - 1)
            y = rng.randint(0, self.height - 1)
            if not self.is_occupied((x, y)):
                return (x, y)
