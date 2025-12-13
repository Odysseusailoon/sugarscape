import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.experiment import ExperimentLogger

def run_powerlaw_experiment():
    print("Running Sugarscape Power Law Experiment (High Competition)...")
    
    # Custom configuration for extreme inequality
    config = SugarscapeConfig(
        # High population density to increase competition
        initial_population=400,
        
        # Wider metabolism range to increase heterogeneity (survival of the fittest)
        # Some agents will burn 5 units/tick, making survival very hard
        metabolism_range=(1, 5),
        
        # High vision for "smart" agents (can see 10 blocks away)
        vision_range=(1, 10),
        
        # Standard other settings
        width=50,
        height=50,
        # vision_range was overridden above
        initial_wealth_range=(5, 25),
        max_age_range=(60, 100),
        sugar_growback_rate=1
    )
    
    experiment_name = "powerlaw_high_competition"
    sim = SugarSimulation(config=config, experiment_name=experiment_name)
    
    # Run for 500 ticks
    sim.run(steps=500)
    
    print(f"\nExperiment complete.")
    print(f"Results saved to: {sim.logger.run_dir}")
    
    return str(sim.logger.run_dir)

if __name__ == "__main__":
    run_powerlaw_experiment()
