import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation

def run_experiment(ticks=500, experiment_name="baseline"):
    print(f"Running Sugarscape Simulation [{experiment_name}]...")
    sim = SugarSimulation(experiment_name=experiment_name)
    sim.run(steps=ticks)
    
    # Return the logger's directory to know where to plot
    return sim.logger.run_dir, sim.logger.plots_dir

def plot_all(run_dir, plots_dir):
    print(f"Generating plots in {plots_dir}...")
    plots_dir = str(plots_dir)
    run_dir = str(run_dir)
    
    # Load metrics CSV
    metrics_file = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print("No metrics.csv found!")
        return
        
    df = pd.read_csv(metrics_file)
    
    # Load Snapshots
    with open(os.path.join(run_dir, "initial_state.json"), 'r') as f:
        initial_state = json.load(f)
    with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
        final_state = json.load(f)
        
    # 1. Wealth Distribution
    plot_wealth_distribution(initial_state, final_state, plots_dir)
    
    # 2. Lorenz Curve
    plot_lorenz_curve(final_state, plots_dir)
    
    # 3. Inequality Time Series
    plot_metrics_series(df, 'gini', 'Gini Coefficient', 'inequality_series.png', plots_dir)
    
    # 4. Spatial Clustering (Moran's I)
    plot_metrics_series(df, 'moran_i', "Moran's I (Spatial Clustering)", 'spatial_clustering.png', plots_dir)
    
    # 5. Mobility / Exploration
    plot_mobility_metrics(df, plots_dir)
    
    # 6. Spatial Layout
    plot_spatial_layout(initial_state, final_state, plots_dir)

def plot_wealth_distribution(initial, final, output_dir):
    w_init = [a['wealth'] for a in initial['agents']]
    w_final = [a['wealth'] for a in final['agents']]
    
    plt.figure(figsize=(10, 6))
    plt.hist(w_init, bins=20, alpha=0.5, label='Initial (t=0)', density=True)
    plt.hist(w_final, bins=20, alpha=0.5, label='Final', density=True)
    plt.xlabel('Wealth')
    plt.ylabel('Density')
    plt.title('Wealth Distribution Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'wealth_distribution.png'))
    plt.close()

def plot_lorenz_curve(state, output_dir):
    wealths = np.sort([a['wealth'] for a in state['agents']])
    if len(wealths) == 0: return
    
    n = len(wealths)
    cumulative_wealth = np.cumsum(wealths)
    total_wealth = cumulative_wealth[-1]
    
    x = np.linspace(0, 1, n)
    y = cumulative_wealth / total_wealth
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, label='Observed')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
    plt.fill_between(x, x, y, alpha=0.1)
    
    plt.xlabel('Cumulative Share of Population')
    plt.ylabel('Cumulative Share of Wealth')
    plt.title('Lorenz Curve (Final State)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'lorenz_curve.png'))
    plt.close()

def plot_metrics_series(df, column, title, filename, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tick'], df[column])
    plt.xlabel('Ticks')
    plt.ylabel(column)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_mobility_metrics(df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['tick'], df['avg_displacement'], label='Avg Displacement')
    plt.plot(df['tick'], df['avg_exploration'], label='Avg Unique Cells Visited')
    plt.xlabel('Ticks')
    plt.ylabel('Distance / Cells')
    plt.title('Mobility & Exploration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'mobility_stats.png'))
    plt.close()

def plot_spatial_layout(initial, final, output_dir):
    # Use final map dimensions
    sugar_map = np.array(final['sugar_capacity'])
    width, height = sugar_map.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def plot_state(ax, agents, title):
        im = ax.imshow(sugar_map.T, origin='lower', cmap='YlOrBr', alpha=0.5, vmin=0, vmax=4)
        x = [a['pos'][0] for a in agents]
        y = [a['pos'][1] for a in agents]
        ax.scatter(x, y, c='red', s=10, alpha=0.7, label='Agents')
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        return im

    plot_state(ax1, initial['agents'], 'Initial State')
    im = plot_state(ax2, final['agents'], 'Final State')
    
    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Sugar Capacity')
    
    plt.suptitle('Agent Spatial Distribution')
    plt.savefig(os.path.join(output_dir, 'spatial_layout.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Sugarscape visualization")
    parser.add_argument("--name", type=str, default="baseline_no_llm", help="Experiment name (folder category)")
    parser.add_argument("--ticks", type=int, default=500, help="Number of ticks")
    args = parser.parse_args()
    
    run_dir, plots_dir = run_experiment(ticks=args.ticks, experiment_name=args.name)
    plot_all(run_dir, plots_dir)
    
    print(f"\nAnalysis complete. Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
