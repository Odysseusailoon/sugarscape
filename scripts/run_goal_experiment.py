"""Run Sugarscape experiments with different LLM agent goals to study their impact on welfare metrics."""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig


def run_goal_experiment(goal_preset: str, ticks: int = 100, seed: int = 42):
    """Run a single experiment with a specific goal preset."""

    print(f"\n{'='*60}")
    print(f"Running Goal Experiment: {goal_preset.upper()}")
    print(f"{'='*60}")

    # Configure simulation
    config = SugarscapeConfig(
        initial_population=50,
        max_ticks=ticks,
        width=50,
        height=50,
        seed=seed,
        enable_llm_agents=True,
        llm_agent_ratio=1.0,  # All agents are LLM
        llm_goal_preset=goal_preset,
        enable_spice=True,
        enable_trade=False  # Disable trade to focus on movement decisions
    )

    print(f"Goal: {goal_preset}")
    print(f"Goal Prompt: {config.llm_goal_prompt[:100]}...")
    print(f"Seed: {seed}, Ticks: {ticks}")

    sim = SugarSimulation(config=config, experiment_name=f"goal_{goal_preset}")

    print("\nInitial Stats:")
    initial_stats = sim.get_stats()
    print(f"  Population: {initial_stats['population']}")
    print(f"  Mean Wealth: {initial_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {initial_stats['survival_rate']:.2f}")

    print("\nRunning simulation...")
    sim.run(steps=ticks)

    print("\nFinal Stats:")
    final_stats = sim.get_stats()
    print(f"  Population: {final_stats['population']}")
    print(f"  Mean Wealth: {final_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {final_stats['survival_rate']:.2f}")
    print(f"  Utilitarian Welfare: {final_stats['utilitarian_welfare']:.2f}")
    print(f"  Nash Welfare: {final_stats['nash_welfare']:.2f}")
    print(f"  Rawlsian Welfare: {final_stats['rawlsian_welfare']:.2f}")
    print(f"  Welfare Gini: {final_stats['welfare_gini']:.3f}")
    print(f"  Mean Lifespan Utilization: {final_stats['mean_lifespan_utilization']:.3f}")

    return sim.logger.run_dir, final_stats


def compare_goals(goals_to_test, ticks=100, seed=42):
    """Run experiments with multiple goal presets and compare results."""

    results = {}

    print(f"\n{'='*80}")
    print("GOAL COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Testing {len(goals_to_test)} goal presets with {ticks} ticks, seed={seed}")

    for goal in goals_to_test:
        try:
            run_dir, final_stats = run_goal_experiment(goal, ticks, seed)
            results[goal] = {
                'run_dir': run_dir,
                'final_stats': final_stats
            }
        except Exception as e:
            print(f"Error running {goal}: {e}")
            continue

    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print("<30")
    print("-" * 80)

    for goal, data in results.items():
        stats = data['final_stats']
        print("<12"
               "<8.2f"
               "<12.2f"
               "<12.2f"
               "<12.2f")

    print("-" * 80)

    # Generate comparison plots if we have multiple results
    if len(results) >= 2:
        print("\nGenerating comparison plots...")

        # Create comparison directory
        try:
            from redblackbench.sugarscape.welfare_plots import WelfarePlotter
        except ImportError:
            print("Warning: Matplotlib not available. Skipping plot generation.")
            return results

        comparison_dir = Path("results/sugarscape/goal_comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Get CSV paths for the first two goals (or more if desired)
        csv_paths = []
        goal_names = []
        for goal, data in list(results.items())[:4]:  # Compare up to 4 goals
            csv_path = Path(data['run_dir']) / "metrics.csv"
            if csv_path.exists():
                csv_paths.append(str(csv_path))
                goal_names.append(goal)

        if len(csv_paths) >= 2:
            WelfarePlotter.generate_comparison_plots(
                llm_csv_path=csv_paths[0],
                baseline_csv_path=csv_paths[1],
                output_dir=str(comparison_dir)
            )

            print(f"Comparison plots saved to: {comparison_dir}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sugarscape Goal Experiments")

    parser.add_argument("--goals", nargs="+",
                        choices=["survival", "wealth", "egalitarian", "utilitarian"],
                        default=["survival", "wealth", "egalitarian", "utilitarian"],
                        help="Goal presets to test")

    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--single", type=str,
                        choices=["survival", "wealth", "egalitarian", "utilitarian"],
                        help="Run single goal experiment (overrides --goals)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please export it: export OPENROUTER_API_KEY='sk-...'")
        sys.exit(1)

    if args.single:
        # Run single experiment
        run_goal_experiment(args.single, args.ticks, args.seed)
    else:
        # Run comparison
        compare_goals(args.goals, args.ticks, args.seed)


if __name__ == "__main__":
    main()

