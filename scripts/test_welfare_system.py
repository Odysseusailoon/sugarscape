"""Test script to verify welfare metrics system is working correctly.

This script runs a small Sugarscape simulation and verifies that:
1. Welfare metrics are calculated
2. Metrics are logged to CSV
3. Plots are generated
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.simulation import SugarSimulation


def test_welfare_system():
    """Run a small test simulation to verify welfare system."""
    
    print("="*70)
    print("Testing Welfare Metrics System")
    print("="*70)
    
    # Create a small test configuration
    config = SugarscapeConfig(
        width=20,
        height=20,
        initial_population=10,
        max_ticks=50,
        enable_spice=True,
        enable_trade=False,
        enable_llm_agents=False,
        seed=42
    )
    
    print("\n1. Creating simulation...")
    sim = SugarSimulation(config, experiment_name="welfare_test")
    
    print(f"   ✓ Simulation created")
    print(f"   ✓ Initial population: {len(sim.agents)}")
    
    print("\n2. Running simulation for 50 steps...")
    sim.run(steps=50)
    
    print(f"   ✓ Simulation complete")
    print(f"   ✓ Final population: {len(sim.agents)}")
    
    print("\n3. Checking metrics...")
    stats = sim.get_stats()
    
    # Check that welfare metrics exist
    welfare_metrics = [
        'utilitarian_welfare',
        'average_welfare',
        'nash_welfare',
        'rawlsian_welfare',
        'welfare_gini',
        'survival_rate'
    ]
    
    missing = []
    for metric in welfare_metrics:
        if metric not in stats:
            missing.append(metric)
        else:
            print(f"   ✓ {metric}: {stats[metric]:.4f}")
    
    if missing:
        print(f"\n   ✗ Missing metrics: {missing}")
        return False
    
    print("\n4. Checking CSV file...")
    csv_path = sim.logger.csv_file
    if csv_path.exists():
        print(f"   ✓ CSV file created: {csv_path}")
        
        # Read first line to check headers
        with open(csv_path, 'r') as f:
            headers = f.readline().strip().split(',')
            
        welfare_headers = [h for h in headers if 'welfare' in h.lower() or 'survival' in h.lower()]
        print(f"   ✓ Welfare columns in CSV: {len(welfare_headers)}")
    else:
        print(f"   ✗ CSV file not found: {csv_path}")
        return False
    
    print("\n5. Checking plots...")
    plots_dir = Path(sim.logger.get_plots_dir())
    
    expected_plots = [
        'welfare_timeseries.png',
        'welfare_summary.png'
    ]
    
    for plot_name in expected_plots:
        plot_path = plots_dir / plot_name
        if plot_path.exists():
            print(f"   ✓ {plot_name} generated")
        else:
            print(f"   ✗ {plot_name} missing")
            return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print(f"\nResults saved to: {sim.logger.run_dir}")
    print(f"Plots available at: {plots_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_welfare_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


