from dataclasses import dataclass
from typing import Tuple

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
    
    # Population
    initial_population: int = 250
    
    # Agent Attribute Distributions (Uniform [min, max])
    initial_wealth_range: Tuple[int, int] = (5, 25)  # w0
    metabolism_range: Tuple[int, int] = (1, 4)       # m
    vision_range: Tuple[int, int] = (1, 6)           # v
    max_age_range: Tuple[int, int] = (60, 100)       # max_age
    
    # Simulation
    max_ticks: int = 1000
    seed: int = 42
