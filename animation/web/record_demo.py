import json
import argparse
import sys
import os
import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.llm_agent import LLMSugarAgent

def serialize_state(sim):
    # Sugar Grid
    sugar_map = sim.env.sugar_amount.tolist()
    
    # Agents
    agents_data = []
    for agent in sim.agents:
        if not agent.alive: continue
        
        agent_type = "LLM" if isinstance(agent, LLMSugarAgent) else "Rule"
        leaning = 0.0
        if hasattr(agent, "self_identity_leaning"):
            leaning = float(agent.self_identity_leaning)
        
        persona = getattr(agent, "persona", "A")

        agents_data.append({
            "id": agent.agent_id,
            "x": agent.pos[0],
            "y": agent.pos[1],
            "w": int(agent.wealth),
            "s": int(agent.spice),
            "type": agent_type,
            "p": persona,
            "l": leaning
        })

    return {
        "type": "update",
        "tick": sim.tick,
        "pop": len(agents_data),
        "grid": sugar_map,
        "agents": agents_data
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=200, help="Number of ticks to record")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--output", type=str, default="animation/web/recording.json")
    args = parser.parse_args()

    print(f"Initializing Simulation with {args.population} agents...")
    config = SugarscapeConfig()
    config.width = 50
    config.height = 50
    config.initial_population = args.population
    
    # Enable personas for color variety
    config.enable_personas = True
    config.persona_distribution = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    sim = SugarSimulation(config=config, experiment_name="web_recording")
    
    recording = {
        "metadata": {
            "width": config.width,
            "height": config.height,
            "total_ticks": args.ticks
        },
        "frames": []
    }

    print(f"Recording {args.ticks} frames...")
    for _ in tqdm.tqdm(range(args.ticks)):
        sim.step()
        frame = serialize_state(sim)
        recording["frames"].append(frame)

    print(f"Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(recording, f)
    print("Done! You can now deploy 'animation/web' to Vercel.")

if __name__ == "__main__":
    main()
