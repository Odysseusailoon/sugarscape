import sys
import os
import argparse
import json
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.llm_agent import LLMSugarAgent

class SugarJSONEncoder(json.JSONEncoder):
    """Custom encoder for Sugarscape types."""
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AnimationRecorder:
    def __init__(self, output_file="animation_data.json"):
        self.output_file = output_file
        self.frames = []
        self.metadata = {}

    def set_metadata(self, config, width, height):
        self.metadata = {
            "width": int(width),
            "height": int(height),
            "config": config.__dict__
        }

    def record_step(self, sim):
        # Capture Grid State
        sugar_map = sim.env.sugar_amount.tolist()
        
        # Capture Agents
        agents_data = []
        for agent in sim.agents:
            agent_type = "LLM" if isinstance(agent, LLMSugarAgent) else "Rule"
            
            # Determine "leaning" for color if applicable
            leaning = 0.0
            if hasattr(agent, "self_identity_leaning"):
                leaning = float(agent.self_identity_leaning)
                
            agents_data.append({
                "id": int(agent.agent_id),
                "pos": [int(p) for p in agent.pos],
                "wealth": int(agent.wealth),
                "spice": int(agent.spice),
                "age": int(agent.age),
                "type": agent_type,
                "persona": agent.persona,
                "leaning": leaning
            })
            
        # Capture Trades (from debug logger's in-memory buffer if possible, or we need to hook into it)
        # The debug logger accumulates trades. We only want trades from *this* tick.
        # We can filter sim.debug_logger.trades by tick.
        current_trades = []
        if sim.debug_logger and hasattr(sim.debug_logger, 'trades'):
            # Optimization: slice from the end backwards until tick mismatch?
            # Or just filter all (might be slow for long runs).
            # Better: Keep track of how many trades we've recorded.
            pass # Implemented in capture logic below

        frame = {
            "tick": int(sim.tick),
            "sugar_map": sugar_map,
            "agents": agents_data,
            # "trades": current_trades # Add this if we can robustly get it
        }
        self.frames.append(frame)

    def save(self):
        data = {
            "metadata": self.metadata,
            "frames": self.frames
        }
        print(f"Saving animation data to {self.output_file}...")
        with open(self.output_file, 'w') as f:
            json.dump(data, f, cls=SugarJSONEncoder)
        print("Done.")

def run_animation_sim(ticks=100, output="animation_data.json", scenario="default"):
    # Config
    config = SugarscapeConfig()
    config.initial_population = 50
    config.width = 20
    config.height = 20
    config.max_ticks = ticks
    
    # Scenario adjustments
    if scenario == "wealthy":
        config.initial_wealth_range = (50, 100)
    elif scenario == "scarce":
        config.growth_rate = 0.5
    
    print(f"Initializing Simulation (Scenario: {scenario})...")
    sim = SugarSimulation(config=config, experiment_name=f"anim_{scenario}")
    
    recorder = AnimationRecorder(output)
    recorder.set_metadata(config, config.width, config.height)
    
    print(f"Running for {ticks} ticks...")
    for _ in tqdm(range(ticks)):
        sim.step()
        recorder.record_step(sim)
        
    recorder.save()
    sim.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--output", type=str, default="animation_data.json")
    parser.add_argument("--scenario", type=str, default="default")
    args = parser.parse_args()
    
    run_animation_sim(args.ticks, args.output, args.scenario)
