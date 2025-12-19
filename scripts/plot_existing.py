import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_all(run_dir, plots_dir):
    print(f"Generating plots in {plots_dir}...")
    plots_dir = str(plots_dir)
    run_dir = str(run_dir)
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Load metrics CSV
    metrics_file = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print("No metrics.csv found!")
        return
        
    df = pd.read_csv(metrics_file)
    
    # Load Snapshots
    try:
        with open(os.path.join(run_dir, "initial_state.json"), 'r') as f:
            initial_state = json.load(f)
        with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
            final_state = json.load(f)
    except FileNotFoundError:
        print("State snapshots not found. Skipping state-dependent plots.")
        initial_state = None
        final_state = None
        
    # 1. Wealth Distribution
    if initial_state and final_state:
        plot_wealth_distribution(initial_state, final_state, plots_dir)
    
    # 2. Lorenz Curve
    if final_state:
        plot_lorenz_curve(final_state, plots_dir)
    
    # 3. Inequality Time Series
    plot_metrics_series(df, 'gini', 'Gini Coefficient', 'inequality_series.png', plots_dir)
    
    # 4. Spatial Clustering (Moran's I)
    plot_metrics_series(df, 'moran_i', "Moran's I (Spatial Clustering)", 'spatial_clustering.png', plots_dir)
    
    # 5. Mobility / Exploration
    plot_mobility_metrics(df, plots_dir)
    
    # 6. Spatial Layout
    if initial_state and final_state:
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
    if column not in df.columns:
        print(f"Warning: Column {column} not found in metrics.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(df['tick'], df[column])
    plt.xlabel('Ticks')
    plt.ylabel(column)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_mobility_metrics(df, output_dir):
    if 'avg_displacement' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['tick'], df['avg_displacement'], label='Avg Displacement')
    if 'avg_exploration' in df.columns:
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
        ax.scatter(x, y, c='red', s=50, alpha=0.7, label='Agents')
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

if __name__ == "__main__":
    base_path = "/Users/yifeichen/RedBlackBench"
    
    # 1. LLM Experiment
    llm_dir = os.path.join(base_path, "results/sugarscape/baseline/experiment_20251215_053749")
    llm_plots = os.path.join(llm_dir, "plots")
    print(f"Processing LLM Experiment: {llm_dir}")
    plot_all(llm_dir, llm_plots)
    
    # 2. Control Experiment
    control_dir = os.path.join(base_path, "results/sugarscape/baseline/experiment_20251215_053744")
    control_plots = os.path.join(control_dir, "plots")
    print(f"Processing Control Experiment: {control_dir}")
    plot_all(control_dir, control_plots)
