import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation

def main():
    print("Initializing Sugarscape...")
    sim = SugarSimulation()
    
    print("Initial Stats:")
    print(sim.get_stats())
    
    print("\nRunning for 100 ticks...")
    for i in range(100):
        sim.step()
        if (i+1) % 10 == 0:
            stats = sim.get_stats()
            print(f"Tick {i+1}: Pop={stats['population']}, Mean W={stats['mean_wealth']:.2f}, Gini={stats['gini']:.2f}")
            
    print("\nFinal Stats:")
    print(sim.get_stats())
    
    # Check if agents are actually moving/harvesting
    # We can check total wealth in the system
    total_wealth = sum(a.wealth for a in sim.agents)
    print(f"Total Wealth: {total_wealth}")

if __name__ == "__main__":
    main()
